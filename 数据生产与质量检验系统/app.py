#!/usr/bin/env python
"""Frame-CausalEvidence v3 local annotation and QC system.

The system is intentionally dependency-free. It stores newly produced data in
the system-local ``dataset/`` directory and only accepts the v3 FQ1/FQ2 schema.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
RULES_PATH = APP_DIR / "rules" / "qc_rules.json"
SYSTEM_DATASET_DIR = APP_DIR / "dataset"
SYSTEM_IMAGE_DIR = SYSTEM_DATASET_DIR / "images"
ANNOTATIONS_PATH = SYSTEM_DATASET_DIR / "annotations.jsonl"
OUTPUT_DIR = APP_DIR / "outputs"
BACKUP_DIR = OUTPUT_DIR / "backups"
STATE_DIR = APP_DIR / "state"
REVIEW_LOG_PATH = STATE_DIR / "review_decisions.jsonl"
REVIEW_WORKSPACE_PATH = STATE_DIR / "review_workspace.jsonl"
REVIEW_WORKSPACE_META_PATH = STATE_DIR / "review_workspace_meta.json"

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
APP_VERSION = "2026-07-16-v3-system-local-dataset"

REQUIRED_FIELDS = [
    "sample_id", "relation_group_id", "images", "question_type",
    "reference_frame", "reference_anchor", "target_instance_id", "relatum_instance_id", "object_instances", "bridge_instance_ids", "question",
    "options", "answer", "view_condition", "removed_images", "mss_images", "quality_status",
]

UNIFIED_FIELDS = list(REQUIRED_FIELDS)

DEFAULT_RULES = {
    "question_type": ["FQ1", "FQ2"],
    "reference_frame": ["camera", "object_intrinsic", "world", "image_plane"],
    "view_condition": ["all_views", "one_valid_mss", "remove_required_view", "hard_distractor_added"],
    "quality_status": ["draft", "reviewed", "verified", "rejected"],
    "review_decision": ["approve", "needs_revision", "reject", "verify"],
    "review_issue_type": [
        "answer_error",
        "ambiguous_reference_frame",
        "insufficient_visual_evidence",
        "object_instance_error",
        "bridge_instance_error",
        "mss_error",
        "view_intervention_error",
        "text_leakage",
        "image_path_error",
        "low_confidence_anchor",
        "other",
    ],
    "low_confidence_sample_ids": [],
}


def configure_data_root(data_root: Path) -> None:
    """Point the workbench at an explicit dataset directory."""
    global SYSTEM_DATASET_DIR, SYSTEM_IMAGE_DIR, ANNOTATIONS_PATH
    SYSTEM_DATASET_DIR = data_root.resolve()
    SYSTEM_IMAGE_DIR = SYSTEM_DATASET_DIR / "images"
    ANNOTATIONS_PATH = SYSTEM_DATASET_DIR / "annotations.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_review_workspace_meta() -> dict:
    return {
        "workspace_id": "workspace_empty",
        "filename": "",
        "imported_at": "",
        "sample_count": 0,
    }


def ensure_dirs() -> None:
    SYSTEM_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not ANNOTATIONS_PATH.exists():
        ANNOTATIONS_PATH.write_text("", encoding="utf-8")
    if not REVIEW_LOG_PATH.exists():
        REVIEW_LOG_PATH.write_text("", encoding="utf-8")
    if not REVIEW_WORKSPACE_PATH.exists():
        REVIEW_WORKSPACE_PATH.write_text("", encoding="utf-8")
    if not REVIEW_WORKSPACE_META_PATH.exists():
        REVIEW_WORKSPACE_META_PATH.write_text(
            json.dumps(default_review_workspace_meta(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_rules() -> dict:
    if RULES_PATH.exists():
        with RULES_PATH.open("r", encoding="utf-8") as f:
            configured = json.load(f)
        merged = copy.deepcopy(DEFAULT_RULES)
        merged.update(configured)
        return merged
    return copy.deepcopy(DEFAULT_RULES)


def json_response(handler: SimpleHTTPRequestHandler, payload, status=HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: SimpleHTTPRequestHandler, message: str, status=HTTPStatus.BAD_REQUEST) -> None:
    json_response(handler, {"ok": False, "error": message}, status)


def read_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc}") from exc
    return records


def read_json(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return copy.deepcopy(fallback)
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}") or copy.deepcopy(fallback)
    except json.JSONDecodeError:
        return copy.deepcopy(fallback)


def parse_jsonl_content(content: str) -> list[dict]:
    records: list[dict] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict], backup: bool = True, reason: str = "save") -> None:
    ensure_dirs()
    if backup and path.exists():
        backup_annotations(reason)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_review_workspace(records: list[dict], reason: str = "save") -> None:
    ensure_dirs()
    if REVIEW_WORKSPACE_PATH.exists():
        backup_file(REVIEW_WORKSPACE_PATH, f"review_workspace_{reason}")
    write_jsonl(REVIEW_WORKSPACE_PATH, records, backup=False)


def write_review_workspace_meta(meta: dict) -> None:
    ensure_dirs()
    REVIEW_WORKSPACE_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_file(path: Path, reason: str) -> str | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_-]", "_", reason)[:60] or "backup"
    target = BACKUP_DIR / f"{safe_reason}_{stamp}{path.suffix}"
    shutil.copy2(path, target)
    return str(target)


def backup_annotations(reason: str) -> str | None:
    return backup_file(ANNOTATIONS_PATH, f"annotations_{reason}")


def backup_and_clear_images(reason: str) -> str | None:
    if not SYSTEM_IMAGE_DIR.exists() or not any(SYSTEM_IMAGE_DIR.iterdir()):
        SYSTEM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = re.sub(r"[^A-Za-z0-9_-]", "_", reason)[:40] or "backup"
    target = BACKUP_DIR / f"images_{safe_reason}_{stamp}"
    shutil.copytree(SYSTEM_IMAGE_DIR, target)
    shutil.rmtree(SYSTEM_IMAGE_DIR)
    SYSTEM_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    return str(target)


def normalize_path_under(base: Path, rel_path: str) -> Path:
    if not rel_path:
        raise ValueError("路径为空")
    rel_path = unquote(rel_path).replace("\\", "/")
    if rel_path.startswith("/") or re.match(r"^[A-Za-z]:", rel_path):
        raise ValueError("不允许绝对路径")
    target = (base / rel_path).resolve()
    base_resolved = base.resolve()
    if base_resolved not in target.parents and target != base_resolved:
        raise ValueError("路径越界")
    return target


def resolve_image_path(project_root: Path, rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    if rel_path.startswith("images/"):
        return normalize_path_under(SYSTEM_DATASET_DIR, rel_path)
    if rel_path.startswith("image/"):
        return normalize_path_under(project_root / "dataset", rel_path)
    # Unified Schema also permits paths relative to a batch root, e.g. 001/image1.jpg.
    for base in [project_root / "dataset_ori", project_root]:
        try:
            target = normalize_path_under(base, rel_path)
        except ValueError:
            continue
        if target.exists():
            return target
    return normalize_path_under(project_root / "dataset_ori", rel_path)


def next_sample_id(records: list[dict]) -> str:
    max_id = 0
    pattern = re.compile(r"^fce_(?:pilot|dev|test)_(\d{5})$")
    for record in records:
        match = pattern.match(str(record.get("sample_id", "")))
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"fce_pilot_{max_id + 1:05d}"


def next_relation_group_id(records: list[dict]) -> str:
    max_id = 0
    pattern = re.compile(r"^group_(\d{5})$")
    for record in records:
        match = pattern.match(str(record.get("relation_group_id", "")))
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"group_{max_id + 1:05d}"


def default_sample(sample_id: str) -> dict:
    digits = re.sub(r"\D+", "", sample_id)[-5:] or "00001"
    return {
        "sample_id": sample_id,
        "relation_group_id": f"group_{digits}",
        "quality_status": "draft",
        "images": [],
        "question_type": "FQ1",
        "reference_frame": "camera",
        "reference_anchor": "",
        "target_instance_id": "obj_001",
        "relatum_instance_id": "obj_002",
        "object_instances": {"obj_001": "", "obj_002": ""},
        "bridge_instance_ids": [],
        "question": "",
        "options": {"A": "", "B": "", "C": "", "D": ""},
        "answer": "A",
        "view_condition": "all_views",
        "removed_images": [],
        "mss_images": [],
    }


def merge_sample(original: dict, patch: dict) -> dict:
    merged = copy.deepcopy(original)
    # v3 keeps one current reference-frame answer per row. Saving any legacy
    # record through the current UI removes the obsolete denormalized cache.
    merged.pop("gold_answers", None)
    # 质量状态只能由人工质检结论升级；普通字段修改接口不得自行晋级。
    changed = False
    for key in UNIFIED_FIELDS:
        if key == "quality_status":
            continue
        if key in patch:
            if patch[key] != original.get(key):
                changed = True
            merged[key] = patch[key]
    merged["sample_id"] = original["sample_id"]
    # 已复核、已验收或已拒绝的样本发生内容修改后，旧结论立即失效。
    if changed and original.get("quality_status") != "draft":
        merged["quality_status"] = "draft"
    return merged


def sample_content_changed(original: dict, updated: dict) -> bool:
    return any(
        key != "quality_status" and original.get(key) != updated.get(key)
        for key in UNIFIED_FIELDS
    )


def issue(sample_id: str, severity: str, code: str, message: str) -> dict:
    return {"sample_id": sample_id, "severity": severity, "code": code, "message": message}


def _normalize_semantic_clause(value: str) -> str:
    return re.sub(r"[\s，,；;。.!！?？:：()（）\[\]【】]", "", value).lower()


def _answer_bearing_query_text(question: str, anchor: str) -> str:
    """Remove reference-frame definition clauses before answer-leak checks."""
    question_clauses = [item.strip() for item in re.split(r"[，,；;。]", question) if item.strip()]
    anchor_clauses = [item.strip() for item in re.split(r"[，,；;。]", anchor) if item.strip()]
    normalized_anchor = [_normalize_semantic_clause(item) for item in anchor_clauses]
    definition_patterns = (
        r"(?:以|将).{1,80}(?:为|定义为)(?:正)?前方$",
        r"面向.{0,30}(?:右手侧|右侧).{0,10}(?:为|是)右方$",
        r"(?:拍摄者|相机|参照物|固定场景|场景轴线).{0,80}(?:正向|前向|朝向).{0,30}(?:定义|为)",
    )
    residual: list[str] = []
    for clause in question_clauses:
        normalized = _normalize_semantic_clause(clause)
        overlaps_anchor = any(
            anchor_part
            and min(len(normalized), len(anchor_part)) >= 4
            and (normalized in anchor_part or anchor_part in normalized)
            for anchor_part in normalized_anchor
        )
        is_definition = any(re.search(pattern, clause) for pattern in definition_patterns)
        if not overlaps_anchor and not is_definition:
            residual.append(clause)
    return " ".join(residual)


def file_sha256(project_root: Path, image_path: str) -> str | None:
    try:
        path = resolve_image_path(project_root, image_path)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hash_set(project_root: Path, paths: list[str]) -> set[str]:
    return {value for value in (file_sha256(project_root, str(path)) for path in paths) if value}


def validate_record(record: dict, project_root: Path, rules: dict) -> list[dict]:
    sample_id = str(record.get("sample_id", "<missing>"))
    issues: list[dict] = []
    keys = set(record.keys())
    required = set(REQUIRED_FIELDS)
    allowed_fields = set(UNIFIED_FIELDS)

    for missing in sorted(required - keys):
        issues.append(issue(sample_id, "error", "missing_field", f"缺少字段：{missing}"))
    for extra in sorted(keys - allowed_fields):
        issues.append(issue(sample_id, "error", "extra_field", f"不应包含字段：{extra}"))

    if not re.fullmatch(r"fce_(pilot|dev|test)_[0-9]{5}", sample_id):
        issues.append(issue(sample_id, "error", "invalid_sample_id", "sample_id 必须符合 fce_pilot_00001 / fce_dev_00001 / fce_test_00001 格式"))
    if not re.fullmatch(r"group_[0-9]{5}", str(record.get("relation_group_id", ""))):
        issues.append(issue(sample_id, "error", "invalid_relation_group_id", "relation_group_id 必须符合 group_00001 格式"))

    if "???" in json.dumps(record, ensure_ascii=False):
        issues.append(issue(sample_id, "error", "encoding_placeholder", "发现 ???，疑似中文编码损坏"))
    question = str(record.get("question", "")).strip()
    if not question:
        issues.append(issue(sample_id, "error", "empty_question", "题干为空"))
    elif not 10 <= len(question) <= 300:
        issues.append(issue(sample_id, "error", "invalid_question_length", "题干长度必须为 10--300 个字符"))
    for field in ["question_type", "reference_frame", "view_condition", "quality_status"]:
        allowed = rules.get(field, DEFAULT_RULES.get(field, []))
        if record.get(field) not in allowed:
            issues.append(issue(sample_id, "error", "invalid_enum", f"{field} 取值不合法：{record.get(field)}"))

    images = record.get("images", [])
    if not isinstance(images, list) or not images:
        issues.append(issue(sample_id, "error", "empty_images", "尚未上传或关联图片"))
    else:
        if len(images) > 8:
            issues.append(issue(sample_id, "error", "too_many_images", "每题最多允许 8 张当前输入图片"))
        image_strings = [str(item) for item in images if isinstance(item, str)]
        if len(image_strings) != len(set(image_strings)):
            issues.append(issue(sample_id, "error", "duplicate_image_path", "图片路径重复"))
        for img_path in images:
            if not isinstance(img_path, str) or not img_path:
                issues.append(issue(sample_id, "error", "invalid_image", "图片路径必须是非空字符串"))
                continue
            if not img_path.startswith("images/"):
                issues.append(issue(sample_id, "error", "non_v3_image_path", f"v3 图片路径必须以 images/ 开头：{img_path}"))
            suffix = Path(img_path).suffix
            if suffix != suffix.lower() or suffix not in ALLOWED_IMAGE_EXTS:
                issues.append(issue(sample_id, "error", "invalid_image_extension", f"图片扩展名不支持：{img_path}"))
            try:
                if not resolve_image_path(project_root, img_path).exists():
                    issues.append(issue(sample_id, "error", "missing_image_path", f"图片不存在：{img_path}"))
            except ValueError as exc:
                issues.append(issue(sample_id, "error", "invalid_image_path", f"{img_path}: {exc}"))

    for ref_field in ["mss_images", "removed_images"]:
        refs = record.get(ref_field, [])
        if not isinstance(refs, list):
            issues.append(issue(sample_id, "error", "invalid_ref_array", f"{ref_field} 不是数组"))
            continue
        if len(refs) != len(set(str(value) for value in refs)):
            issues.append(issue(sample_id, "error", "duplicate_image_ref", f"{ref_field} 中存在重复路径"))
        for ref in refs:
            if not isinstance(ref, str):
                issues.append(issue(sample_id, "error", "invalid_image_ref", f"{ref_field} 含非字符串路径"))
                continue
            if not ref.startswith("images/"):
                issues.append(issue(sample_id, "error", "non_v3_image_ref", f"v3 的 {ref_field} 路径必须以 images/ 开头：{ref}"))
            try:
                if not resolve_image_path(project_root, ref).exists():
                    issues.append(issue(sample_id, "error", "missing_image_ref", f"{ref_field} 图片不存在：{ref}"))
            except ValueError as exc:
                issues.append(issue(sample_id, "error", "invalid_image_ref", f"{ref}: {exc}"))
    mss_images = record.get("mss_images", [])
    if not isinstance(mss_images, list):
        mss_images = []
    qt = record.get("question_type")
    if qt in {"FQ1", "FQ2"} and len(mss_images) < 2:
        issues.append(issue(sample_id, "error", "missing_mss_images", f"{qt} 的多图 MSS 必须至少包含 2 张图片"))
    if qt in {"FQ1", "FQ2"}:
        allowed_mss = set(images or [])
        if qt == "FQ2" and record.get("view_condition") == "remove_required_view":
            allowed_mss |= set(record.get("removed_images", []) or [])
        for ref in mss_images:
            if ref not in allowed_mss:
                issues.append(issue(sample_id, "error", "mss_invalid_reference", f"mss_images 不是当前图片或删除前 MSS 的合法引用：{ref}"))

    removed_images = record.get("removed_images", [])
    if isinstance(removed_images, list):
        overlap = set(images or []) & set(removed_images)
        if overlap:
            issues.append(issue(sample_id, "error", "removed_image_still_visible", f"已删除图片仍在当前 images 中：{', '.join(sorted(overlap))}"))
    options = record.get("options", {})
    if not isinstance(options, dict):
        issues.append(issue(sample_id, "error", "invalid_options", "选项不是对象"))
    else:
        option_texts = [str(options.get(opt, "")).strip() for opt in ["A", "B", "C", "D"]]
        for opt in ["A", "B", "C", "D"]:
            if opt not in options:
                issues.append(issue(sample_id, "error", "missing_option", f"缺少选项 {opt}"))
            elif not str(options.get(opt, "")).strip():
                issues.append(issue(sample_id, "error", "empty_option", f"选项 {opt} 为空"))
        if all(option_texts) and len(set(option_texts)) != 4:
            issues.append(issue(sample_id, "error", "duplicate_option_text", "A--D 四个选项文本必须互不相同"))

    answer = record.get("answer")
    if answer not in ["A", "B", "C", "D"]:
        issues.append(issue(sample_id, "error", "invalid_correct_option", "正确选项必须是 A/B/C/D"))
    else:
        correct_text = str(options.get(answer, "") if isinstance(options, dict) else "")
        question = str(record.get("question", ""))
        anchor = str(record.get("reference_anchor", "")).strip()
        query_text = _answer_bearing_query_text(question, anchor)
        option_values = (
            {str(value).strip() for value in options.values() if str(value).strip()}
            if isinstance(options, dict)
            else set()
        )
        neutral_enumeration = len(option_values) == 4 and all(
            value in query_text for value in option_values
        )
        explicit_letter = bool(
            re.search(
                r"(?:答案|正确选项|应选)\s*(?:是|为|：|:)?\s*[ABCD](?![A-Za-z])",
                query_text,
                re.I,
            )
        )
        if explicit_letter or (
            correct_text
            and len(correct_text) >= 2
            and correct_text in query_text
            and not neutral_enumeration
        ):
            issues.append(issue(sample_id, "warning", "possible_answer_leak", "题干可能包含正确选项文本"))

    qt = record.get("question_type")
    if record.get("target_instance_id") == record.get("relatum_instance_id"):
        issues.append(issue(sample_id, "error", "same_object_role", "target_instance_id 与 relatum_instance_id 不得相同"))
    object_instances = record.get("object_instances")
    if not isinstance(object_instances, dict):
        issues.append(issue(sample_id, "error", "invalid_object_instances", "object_instances 必须是对象 ID 到物体说明的映射"))
    else:
        for object_id, label in object_instances.items():
            if not re.match(r"^obj_[0-9]{3}$", str(object_id)):
                issues.append(issue(sample_id, "error", "invalid_object_id", f"对象 ID 格式不合法：{object_id}"))
            if not str(label).strip():
                issues.append(issue(sample_id, "warning", "empty_object_label", f"{object_id} 尚未填写对应物体说明"))
        for role_field in ["target_instance_id", "relatum_instance_id"]:
            role_id = record.get(role_field)
            if role_id and role_id not in object_instances:
                issues.append(issue(sample_id, "error", "object_id_not_defined", f"{role_field}={role_id} 未在 object_instances 中定义"))
    bridge_instance_ids = record.get("bridge_instance_ids")
    if not isinstance(bridge_instance_ids, list):
        issues.append(issue(sample_id, "error", "invalid_bridge_instance_ids", "bridge_instance_ids 必须是按目标到参照顺序排列的对象 ID 数组"))
        bridge_instance_ids = []
    else:
        if len(bridge_instance_ids) != len(set(bridge_instance_ids)):
            issues.append(issue(sample_id, "error", "duplicate_bridge_instance", "bridge_instance_ids 不得重复同一个对象 ID"))
        for bridge_id in bridge_instance_ids:
            if not isinstance(bridge_id, str) or not re.match(r"^obj_[0-9]{3}$", bridge_id):
                issues.append(issue(sample_id, "error", "invalid_bridge_instance_id", f"连接对象 ID 格式不合法：{bridge_id}"))
                continue
            if bridge_id in {record.get("target_instance_id"), record.get("relatum_instance_id")}:
                issues.append(issue(sample_id, "error", "bridge_role_overlap", f"连接对象 {bridge_id} 不能同时作为目标对象或参照对象"))
            if isinstance(object_instances, dict) and bridge_id not in object_instances:
                issues.append(issue(sample_id, "error", "bridge_id_not_defined", f"连接对象 {bridge_id} 未在 object_instances 中定义"))
    if not str(record.get("reference_anchor", "")).strip():
        issues.append(issue(sample_id, "error", "missing_reference_anchor", "必须填写参考系锚点或判定依据"))
    if sample_id in set(rules.get("low_confidence_sample_ids", [])):
        issues.append(issue(sample_id, "warning", "low_confidence_anchor", "该迁移草稿的参考系锚点已知存在语义风险，只能用于系统演练，重构并人工复核前不得晋级"))
    if qt == "FQ1":
        if record.get("reference_frame") not in {"camera", "object_intrinsic", "world"}:
            issues.append(issue(sample_id, "error", "fq1_non_core_frame", "FQ1 只能使用 camera、object_intrinsic 或 world；image_plane 仅作对照题"))
        if record.get("view_condition") != "all_views":
            issues.append(issue(sample_id, "error", "fq1_not_all_views", "FQ1 参考系转换题必须使用 all_views；证据干预请另建 FQ2 单参考系题"))
        if len(images or []) < 2:
            issues.append(issue(sample_id, "error", "fq1_not_multiview", "FQ1 必须至少包含 2 张图片"))
        if list(mss_images) != list(images or []):
            issues.append(issue(sample_id, "error", "fq1_mss_not_all_images", "FQ1 的全部当前输入图片都必须属于 MSS，且顺序与 images 一致"))
    if qt == "FQ2" and record.get("view_condition") not in {"one_valid_mss", "remove_required_view", "hard_distractor_added"}:
        issues.append(issue(sample_id, "error", "fq2_all_views_only", "FQ2 必须使用 MSS、删图或干扰图视图条件"))
    if qt == "FQ2" and record.get("view_condition") in {"one_valid_mss", "hard_distractor_added"}:
        answer_text = str(options.get(answer, "") if isinstance(options, dict) else "").strip()
        if answer_text == "信息不足":
            issues.append(issue(sample_id, "error", "fq2_answerable_condition_is_insufficient", "one_valid_mss 和 hard_distractor_added 的正确答案必须是确定空间方向，不能为“信息不足”"))
    if record.get("view_condition") == "remove_required_view" and not record.get("removed_images"):
        issues.append(issue(sample_id, "error", "missing_removed_images", "删除必要图条件必须填写 removed_images"))
    if record.get("view_condition") != "remove_required_view" and record.get("removed_images"):
        issues.append(issue(sample_id, "error", "unexpected_removed_images", "只有 remove_required_view 允许填写 removed_images；其他条件必须为 []"))
    if qt == "FQ2" and record.get("view_condition") == "one_valid_mss" and list(mss_images) != list(images or []):
        issues.append(issue(sample_id, "error", "fq2_mss_input_mismatch", "one_valid_mss 的 images 必须恰好等于 mss_images"))
    if qt == "FQ2" and record.get("view_condition") == "remove_required_view":
        answer_text = str(options.get(answer, "") if isinstance(options, dict) else "").strip()
        if answer_text != "信息不足":
            issues.append(issue(sample_id, "error", "fq2_removed_answer_not_insufficient", "remove_required_view 的正确选项文本必须为“信息不足”"))
        if not (set(mss_images) & set(record.get("removed_images", []) or [])):
            issues.append(issue(sample_id, "error", "fq2_removed_not_in_mss", "至少一张 removed_images 必须属于删除前 MSS"))
    if qt == "FQ2" and record.get("view_condition") == "hard_distractor_added":
        if not set(mss_images).issubset(set(images or [])) or set(mss_images) == set(images or []):
            issues.append(issue(sample_id, "error", "fq2_missing_actual_distractor", "hard_distractor_added 必须在 MSS 之外至少新增一张当前输入图片"))
    return issues


def validate_dataset(records: list[dict], project_root: Path, rules: dict) -> dict:
    ids = [str(r.get("sample_id", "")) for r in records]
    duplicate_ids = {sid for sid in ids if ids.count(sid) > 1}
    all_issues: list[dict] = []
    by_sample: dict[str, list[dict]] = {}
    for record in records:
        sid = str(record.get("sample_id", "<missing>"))
        sample_issues = validate_record(record, project_root, rules)
        if sid in duplicate_ids:
            sample_issues.append(issue(sid, "error", "duplicate_sample_id", "样本编号重复"))
        by_sample[sid] = sample_issues
        all_issues.extend(sample_issues)

    required_frames = {"camera", "object_intrinsic", "world"}
    fq1_groups: dict[str, list[dict]] = {}
    fq2_groups: dict[str, list[dict]] = {}
    all_groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get("relation_group_id"):
            all_groups.setdefault(str(record["relation_group_id"]), []).append(record)
        if record.get("question_type") == "FQ1" and record.get("relation_group_id"):
            fq1_groups.setdefault(str(record["relation_group_id"]), []).append(record)
        if record.get("question_type") == "FQ2" and record.get("relation_group_id"):
            fq2_groups.setdefault(str(record["relation_group_id"]), []).append(record)
    global_issues: list[dict] = []
    for gid, group_records in sorted(all_groups.items()):
        families = {str(record.get("question_type")) for record in group_records}
        if len(families) > 1:
            global_issues.append({"severity": "error", "code": "mixed_question_type_group", "message": f"题组 {gid} 混合了多个题型：{', '.join(sorted(families))}；一个题组只能属于一个实验族"})
    for gid, group_records in sorted(fq1_groups.items()):
        frames = [str(record.get("reference_frame")) for record in group_records]
        frame_set = set(frames)
        missing = required_frames - frame_set
        if missing:
            global_issues.append(
                {"severity": "warning", "code": "fq1_group_incomplete", "message": f"FQ1 组 {gid} 尚缺核心参考系：{', '.join(sorted(missing))}"}
            )
        if len(frames) != len(frame_set):
            global_issues.append({"severity": "error", "code": "fq1_duplicate_frame", "message": f"FQ1 组 {gid} 同一核心参考系只能有 1 条题目"})
        if frame_set == required_frames and len(group_records) == 3 and len({str(record.get("answer")) for record in group_records}) < 2:
            global_issues.append({"severity": "error", "code": "fq1_no_answer_switch", "message": f"FQ1 组 {gid} 的三个参考系答案完全相同，不能有效测量参考系切换；请重新布置或拒收"})
        if any(record.get("quality_status") == "verified" for record in group_records) and (frame_set != required_frames or len(group_records) != 3):
            global_issues.append({"severity": "error", "code": "fq1_verified_group_incomplete", "message": f"已验证 FQ1 组 {gid} 必须恰好包含 camera、object_intrinsic、world 各 1 条"})
        baseline = group_records[0]
        for record in group_records[1:]:
            for field in ["target_instance_id", "relatum_instance_id", "object_instances", "bridge_instance_ids", "options"]:
                if record.get(field) != baseline.get(field):
                    global_issues.append({"severity": "error", "code": "fq1_group_relation_mismatch", "message": f"FQ1 组 {gid} 的 {field} 不一致；三参考系题只能改变参考系、锚点、题干和答案"})
                    break
            if list(record.get("images", []) or []) != list(baseline.get("images", []) or []):
                global_issues.append({"severity": "error", "code": "fq1_group_image_mismatch", "message": f"FQ1 组 {gid} 的 images 或顺序不一致；参考系转换必须复用同一组图片"})
    for gid, group_records in sorted(fq2_groups.items()):
        required_conditions = {"one_valid_mss", "remove_required_view", "hard_distractor_added"}
        conditions = [str(record.get("view_condition")) for record in group_records]
        condition_set = set(conditions)
        missing_conditions = required_conditions - condition_set
        if missing_conditions:
            global_issues.append({"severity": "warning", "code": "fq2_group_incomplete", "message": f"FQ2 组 {gid} 尚缺视图条件：{', '.join(sorted(missing_conditions))}"})
        if len(conditions) != len(condition_set):
            global_issues.append({"severity": "error", "code": "fq2_duplicate_condition", "message": f"FQ2 组 {gid} 每种视图条件只能有 1 条题目"})
        if any(record.get("quality_status") == "verified" for record in group_records) and (condition_set != required_conditions or len(group_records) != 3):
            global_issues.append({"severity": "error", "code": "fq2_verified_group_incomplete", "message": f"已验证 FQ2 组 {gid} 必须恰好包含 3 种预定义视图条件"})
        baseline = group_records[0]
        for record in group_records[1:]:
            for field in ["target_instance_id", "relatum_instance_id", "object_instances", "reference_frame", "reference_anchor", "bridge_instance_ids", "question", "options"]:
                if record.get(field) != baseline.get(field):
                    global_issues.append({"severity": "error", "code": "fq2_group_relation_mismatch", "message": f"FQ2 组 {gid} 的 {field} 不一致；三个版本只能改变图片集合、删除记录、MSS 和答案"})
                    break
        by_condition = {str(record.get("view_condition")): record for record in group_records}
        mss_record = by_condition.get("one_valid_mss")
        removed_record = by_condition.get("remove_required_view")
        distractor_record = by_condition.get("hard_distractor_added")
        if mss_record and removed_record:
            baseline_mss_hashes = image_hash_set(project_root, list(mss_record.get("mss_images", []) or []))
            removed_mss_hashes = image_hash_set(project_root, list(removed_record.get("mss_images", []) or []))
            removed_hashes = image_hash_set(project_root, list(removed_record.get("removed_images", []) or []))
            current_hashes = image_hash_set(project_root, list(removed_record.get("images", []) or []))
            if baseline_mss_hashes != removed_mss_hashes:
                global_issues.append({"severity": "error", "code": "fq2_mss_asset_mismatch", "message": f"FQ2 组 {gid} 的 MSS 与删必要图版本未引用同一组物理图片"})
            if not removed_hashes or not removed_hashes.issubset(baseline_mss_hashes):
                global_issues.append({"severity": "error", "code": "fq2_removed_asset_not_baseline_mss", "message": f"FQ2 组 {gid} 的 removed_images 必须是基准 MSS 的非空物理子集"})
            if current_hashes != baseline_mss_hashes - removed_hashes:
                global_issues.append({"severity": "error", "code": "fq2_remove_not_pure_deletion", "message": f"FQ2 组 {gid} 的删必要图版本必须严格等于基准 MSS 减去 removed_images，不得同时加入或额外遗漏其他图"})
        if mss_record and distractor_record:
            if image_hash_set(project_root, list(mss_record.get("mss_images", []) or [])) != image_hash_set(project_root, list(distractor_record.get("mss_images", []) or [])):
                global_issues.append({"severity": "error", "code": "fq2_mss_asset_mismatch", "message": f"FQ2 组 {gid} 的 MSS 与加干扰图版本未引用同一组物理图片"})
            if mss_record.get("answer") != distractor_record.get("answer"):
                global_issues.append({"severity": "error", "code": "fq2_distractor_answer_changed", "message": f"FQ2 组 {gid} 加入干扰图后正确答案必须保持不变"})
    all_issues.extend(global_issues)
    return {
        "ok": not any(i.get("severity") == "error" for i in all_issues),
        "issues": all_issues,
        "by_sample": by_sample,
        "global_issues": global_issues,
    }


def coverage(records: list[dict]) -> dict:
    def count(field: str) -> dict:
        result: dict[str, int] = {}
        for record in records:
            value = record
            for part in field.split("."):
                value = value.get(part, {}) if isinstance(value, dict) else {}
            value = str(value)
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    return {
        "total": len(records),
        "image_count": sum(len(r.get("images", []) or []) for r in records),
        "question_type": count("question_type"),
        "reference_frame": count("reference_frame"),
        "view_condition": count("view_condition"),
        "answer": count("answer"),
        "quality_status": count("quality_status"),
        "relation_groups": len({str(r.get("relation_group_id")) for r in records if r.get("relation_group_id")}),
    }


def review_entries() -> list[dict]:
    return read_jsonl(REVIEW_LOG_PATH)


def append_review_entry(entry: dict) -> None:
    with REVIEW_LOG_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def review_progress(records: list[dict], source: str, workspace_id: str | None = None) -> dict:
    sample_ids = {str(record.get("sample_id", "")) for record in records if record.get("sample_id")}
    latest_decisions: dict[str, str] = {}
    for entry in review_entries():
        entry_source = entry.get("source") or "production"
        if entry_source != source:
            continue
        if source == "workspace" and entry.get("workspace_id") != workspace_id:
            continue
        sample_id = str(entry.get("sample_id", ""))
        if sample_id in sample_ids:
            decision = str(entry.get("decision", ""))
            if decision == "invalidate":
                latest_decisions.pop(sample_id, None)
            else:
                latest_decisions[sample_id] = decision

    decision_counts = {"approve": 0, "needs_revision": 0, "reject": 0, "verify": 0}
    for decision in latest_decisions.values():
        if decision in decision_counts:
            decision_counts[decision] += 1
    reviewed = len(latest_decisions)
    total = len(sample_ids)
    return {
        "source": source,
        "workspace_id": workspace_id,
        "total": total,
        "reviewed": reviewed,
        "unreviewed": max(total - reviewed, 0),
        "percent": round((reviewed / total) * 100, 1) if total else 0,
        "approve": decision_counts["approve"],
        "needs_revision": decision_counts["needs_revision"],
        "reject": decision_counts["reject"],
        "verify": decision_counts["verify"],
        "verified": sum(1 for record in records if record.get("quality_status") == "verified"),
    }


def next_image_name(sample: dict, ext: str) -> str:
    max_id = 0
    # 逻辑删除的图片仍保留原文件路径；新上传不能复用其编号，否则会覆盖
    # 已删除文件并破坏 removed_images 的追溯关系。
    image_paths = list(sample.get("images", []) or []) + list(sample.get("removed_images", []) or [])
    for image_path in image_paths:
        match = re.search(r"image(\d+)", Path(str(image_path)).stem)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return f"image{max_id + 1}{ext}"


class WorkbenchHandler(SimpleHTTPRequestHandler):
    server_version = "FrameEvidenceWorkbench/0.2"

    @property
    def project_root(self) -> Path:
        return self.server.project_root  # type: ignore[attr-defined]

    @property
    def rules(self) -> dict:
        return self.server.rules  # type: ignore[attr-defined]

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel = parsed.path.lstrip("/") or "index.html"
        return str((STATIC_DIR / rel).resolve())

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/config":
                self.handle_config()
            elif parsed.path == "/api/dataset":
                self.handle_dataset()
            elif parsed.path == "/api/samples":
                self.handle_samples()
            elif parsed.path == "/api/review-workspace":
                self.handle_review_workspace()
            elif parsed.path == "/api/reviews":
                self.handle_reviews()
            elif parsed.path == "/api/image":
                self.handle_image(parsed)
            elif parsed.path.startswith("/api/"):
                error_response(self, "未知接口", HTTPStatus.NOT_FOUND)
            else:
                super().do_GET()
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/dataset/reset":
                self.handle_dataset_reset()
            elif parsed.path == "/api/dataset/import-jsonl":
                self.handle_import_jsonl()
            elif parsed.path == "/api/dataset/import-smoke":
                self.handle_import_smoke()
            elif parsed.path == "/api/review-workspace/import-jsonl":
                self.handle_review_workspace_import_jsonl()
            elif parsed.path == "/api/review-workspace/export":
                self.handle_review_workspace_export()
            elif parsed.path == "/api/sample":
                self.handle_create_sample()
            elif re.match(r"^/api/sample/[^/]+/clone$", parsed.path):
                self.handle_clone_sample(unquote(parsed.path.split("/")[-2]))
            elif parsed.path.startswith("/api/sample/") and parsed.path.endswith("/images"):
                sample_id = unquote(parsed.path.split("/")[-2])
                self.handle_upload_images(sample_id)
            elif re.match(r"^/api/sample/[^/]+/(logical-remove-image|restore-image)$", parsed.path):
                parts = parsed.path.split("/")
                self.handle_logical_image_visibility(unquote(parts[3]), parts[4] == "logical-remove-image")
            elif parsed.path == "/api/validate":
                self.handle_validate()
            elif parsed.path == "/api/review":
                self.handle_review()
            elif parsed.path == "/api/export":
                self.handle_export()
            else:
                error_response(self, "未知接口", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_PATCH(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/review-workspace/sample/"):
                sample_id = unquote(parsed.path.rsplit("/", 1)[-1])
                self.handle_patch_review_workspace_sample(sample_id)
            elif parsed.path.startswith("/api/sample/"):
                sample_id = unquote(parsed.path.rsplit("/", 1)[-1])
                self.handle_patch_sample(sample_id)
            else:
                error_response(self, "未知接口", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            match = re.match(r"^/api/sample/([^/]+)/image/([^/]+)$", parsed.path)
            if match:
                self.handle_delete_image(unquote(match.group(1)), unquote(match.group(2)))
            else:
                error_response(self, "未知接口", HTTPStatus.NOT_FOUND)
        except Exception as exc:
            error_response(self, str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def current_records(self) -> list[dict]:
        return read_jsonl(ANNOTATIONS_PATH)

    def review_workspace_records(self) -> list[dict]:
        return read_jsonl(REVIEW_WORKSPACE_PATH)

    def review_workspace_meta(self) -> dict:
        meta = read_json(REVIEW_WORKSPACE_META_PATH, default_review_workspace_meta())
        default = default_review_workspace_meta()
        default.update(meta)
        return default

    def handle_config(self) -> None:
        json_response(
            self,
            {
                "ok": True,
                "app_version": APP_VERSION,
                "project_root": str(self.project_root),
                "dataset_dir": str(SYSTEM_DATASET_DIR),
                "annotations_path": str(ANNOTATIONS_PATH),
                "review_log_path": str(REVIEW_LOG_PATH),
                "rules": self.rules,
            },
        )

    def handle_dataset(self) -> None:
        records = self.current_records()
        validation = validate_dataset(records, self.project_root, self.rules)
        json_response(
            self,
            {
                "ok": True,
                "dataset_dir": str(SYSTEM_DATASET_DIR),
                "annotations_path": str(ANNOTATIONS_PATH),
                "image_dir": str(SYSTEM_IMAGE_DIR),
                "coverage": coverage(records),
                "validation": validation,
                "review_progress": review_progress(records, "production"),
            },
        )

    def handle_samples(self) -> None:
        records = self.current_records()
        validation = validate_dataset(records, self.project_root, self.rules)
        json_response(
            self,
            {
                "ok": True,
                "samples": records,
                "validation": validation,
                "coverage": coverage(records),
                "review_progress": review_progress(records, "production"),
            },
        )

    def handle_review_workspace(self) -> None:
        records = self.review_workspace_records()
        meta = self.review_workspace_meta()
        validation = validate_dataset(records, self.project_root, self.rules)
        json_response(
            self,
            {
                "ok": True,
                "workspace_path": str(REVIEW_WORKSPACE_PATH),
                "meta": meta,
                "samples": records,
                "validation": validation,
                "coverage": coverage(records),
                "review_progress": review_progress(records, "workspace", meta.get("workspace_id")),
            },
        )

    def handle_reviews(self) -> None:
        json_response(self, {"ok": True, "reviews": read_jsonl(REVIEW_LOG_PATH)})

    def handle_image(self, parsed) -> None:
        rel_path = parse_qs(parsed.query).get("path", [""])[0]
        try:
            target = resolve_image_path(self.project_root, rel_path)
        except ValueError as exc:
            error_response(self, str(exc), HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            error_response(self, f"图片不存在：{rel_path}", HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_dataset_reset(self) -> None:
        annotation_backup = backup_annotations("reset")
        image_backup = backup_and_clear_images("reset")
        write_jsonl(ANNOTATIONS_PATH, [], backup=False)
        json_response(self, {"ok": True, "coverage": coverage([]), "backup_path": annotation_backup, "image_backup_path": image_backup})

    def handle_import_jsonl(self) -> None:
        payload = read_body(self)
        content = payload.get("content", "")
        if not isinstance(content, str) or not content.strip():
            error_response(self, "导入内容为空")
            return
        records = parse_jsonl_content(content)
        unsupported = sorted({str(record.get("question_type", "")) for record in records} - {"FQ1", "FQ2"})
        if unsupported:
            error_response(self, f"v3 只接受 FQ1/FQ2；发现旧题型：{', '.join(unsupported)}。请按 MIGRATION_V2_TO_V3.md 迁移。", HTTPStatus.CONFLICT)
            return
        validation = validate_dataset(records, self.project_root, self.rules)
        backup_path = backup_annotations("import_jsonl")
        write_jsonl(ANNOTATIONS_PATH, records, backup=False)
        json_response(self, {"ok": True, "count": len(records), "validation": validation, "coverage": coverage(records), "backup_path": backup_path})

    def handle_import_smoke(self) -> None:
        error_response(self, "旧版 smoke 数据使用 QT1--QT8 格式，与当前统一 Schema 不兼容；请迁移后通过 JSONL 导入。", HTTPStatus.CONFLICT)

    def handle_review_workspace_import_jsonl(self) -> None:
        payload = read_body(self)
        content = payload.get("content", "")
        filename = str(payload.get("filename") or "review_import.jsonl")
        if not isinstance(content, str) or not content.strip():
            error_response(self, "导入内容为空")
            return
        records = parse_jsonl_content(content)
        unsupported = sorted({str(record.get("question_type", "")) for record in records} - {"FQ1", "FQ2"})
        if unsupported:
            error_response(self, f"v3 质检工作区只接受 FQ1/FQ2；发现旧题型：{', '.join(unsupported)}。", HTTPStatus.CONFLICT)
            return
        validation = validate_dataset(records, self.project_root, self.rules)
        workspace_id = f"workspace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "workspace_id": workspace_id,
            "filename": Path(filename).name,
            "imported_at": utc_now(),
            "sample_count": len(records),
        }
        write_review_workspace(records, reason="import_jsonl")
        write_review_workspace_meta(meta)
        json_response(
            self,
            {
                "ok": True,
                "count": len(records),
                "meta": meta,
                "validation": validation,
                "coverage": coverage(records),
                "review_progress": review_progress(records, "workspace", workspace_id),
            },
        )

    def handle_create_sample(self) -> None:
        records = self.current_records()
        sample = default_sample(next_sample_id(records))
        records.append(sample)
        write_jsonl(ANNOTATIONS_PATH, records, reason="create_sample")
        json_response(self, {"ok": True, "sample": sample, "issues": validate_record(sample, self.project_root, self.rules)}, HTTPStatus.CREATED)

    def handle_clone_sample(self, sample_id: str) -> None:
        records = self.current_records()
        source = next((r for r in records if r.get("sample_id") == sample_id), None)
        if source is None:
            error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)
            return
        payload = read_body(self)
        question_type = str(payload.get("question_type") or "FQ1")
        if question_type not in {"FQ1", "FQ2"}:
            error_response(self, "派生题型只能是 FQ1 或 FQ2")
            return
        if source.get("question_type") != question_type:
            error_response(self, "派生题必须与来源样本属于同一题型；新对象关系请使用“新建题组”", HTTPStatus.CONFLICT)
            return
        clone = default_sample(next_sample_id(records))
        for key in ["relation_group_id", "images", "object_instances", "bridge_instance_ids", "target_instance_id", "relatum_instance_id"]:
            if key in source:
                clone[key] = json.loads(json.dumps(source[key]))
        if question_type == "FQ1":
            clone["question_type"] = "FQ1"
            existing_frames = {str(record.get("reference_frame")) for record in records if record.get("question_type") == "FQ1" and record.get("relation_group_id") == clone["relation_group_id"]}
            remaining_frames = [frame for frame in ["camera", "object_intrinsic", "world"] if frame not in existing_frames]
            if not remaining_frames:
                error_response(self, "该 FQ1 题组已包含三个核心参考系，不能继续派生", HTTPStatus.CONFLICT)
                return
            clone["reference_frame"] = remaining_frames[0]
            clone["reference_anchor"] = ""
            clone["view_condition"] = "all_views"
            clone["options"] = json.loads(json.dumps(source.get("options", clone["options"])))
            clone["mss_images"] = json.loads(json.dumps(clone.get("images", []) or []))
        else:
            # An FQ2 variant keeps the same relation but may expose a
            # different subset of the candidate images. Restore any logically
            # removed source images as available candidates for the new variant.
            clone["question_type"] = "FQ2"
            clone["images"] = list(dict.fromkeys((source.get("images", []) or []) + (source.get("removed_images", []) or [])))
            clone["removed_images"] = []
            clone["reference_frame"] = source.get("reference_frame", "camera")
            clone["reference_anchor"] = source.get("reference_anchor", "")
            clone["question"] = source.get("question", "")
            clone["options"] = json.loads(json.dumps(source.get("options", clone["options"])))
            clone["answer"] = source.get("answer", "A")
            clone["mss_images"] = json.loads(json.dumps(source.get("mss_images", []) or []))
            existing_conditions = {
                str(record.get("view_condition"))
                for record in records
                if record.get("question_type") == "FQ2" and record.get("relation_group_id") == clone["relation_group_id"]
            }
            remaining_conditions = [
                condition
                for condition in ["one_valid_mss", "remove_required_view", "hard_distractor_added"]
                if condition not in existing_conditions
            ]
            if not remaining_conditions:
                error_response(self, "该 FQ2 题组已包含三个证据条件，不能继续派生", HTTPStatus.CONFLICT)
                return
            clone["view_condition"] = remaining_conditions[0]
        records.append(clone)
        write_jsonl(ANNOTATIONS_PATH, records, reason="clone_sample")
        json_response(self, {"ok": True, "sample": clone, "issues": validate_record(clone, self.project_root, self.rules)}, HTTPStatus.CREATED)

    def handle_patch_sample(self, sample_id: str) -> None:
        payload = read_body(self)
        records = self.current_records()
        for idx, record in enumerate(records):
            if record.get("sample_id") == sample_id:
                updated = merge_sample(record, payload)
                invalidated = sample_content_changed(record, updated) and record.get("quality_status") != "draft"
                records[idx] = updated
                write_jsonl(ANNOTATIONS_PATH, records, reason="patch_sample")
                if invalidated:
                    append_review_entry(
                        {
                            "timestamp": utc_now(),
                            "sample_id": sample_id,
                            "source": "production",
                            "workspace_id": "",
                            "reviewer": "system",
                            "decision": "invalidate",
                            "issue_type": "content_changed",
                            "remark": "样本内容发生修改，旧复核结论自动失效",
                            "final_checks_confirmed": False,
                        }
                    )
                json_response(self, {"ok": True, "sample": updated, "issues": validate_record(updated, self.project_root, self.rules)})
                return
        error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)

    def handle_patch_review_workspace_sample(self, sample_id: str) -> None:
        payload = read_body(self)
        records = self.review_workspace_records()
        meta = self.review_workspace_meta()
        for idx, record in enumerate(records):
            if record.get("sample_id") == sample_id:
                updated = merge_sample(record, payload)
                invalidated = sample_content_changed(record, updated) and record.get("quality_status") != "draft"
                records[idx] = updated
                write_review_workspace(records, reason="patch_sample")
                if invalidated:
                    append_review_entry(
                        {
                            "timestamp": utc_now(),
                            "sample_id": sample_id,
                            "source": "workspace",
                            "workspace_id": meta.get("workspace_id", ""),
                            "reviewer": "system",
                            "decision": "invalidate",
                            "issue_type": "content_changed",
                            "remark": "样本内容发生修改，旧复核结论自动失效",
                            "final_checks_confirmed": False,
                        }
                    )
                json_response(
                    self,
                    {
                        "ok": True,
                        "sample": updated,
                        "issues": validate_record(updated, self.project_root, self.rules),
                        "review_progress": review_progress(records, "workspace", meta.get("workspace_id")),
                    },
                )
                return
        error_response(self, f"未找到质检工作区样本：{sample_id}", HTTPStatus.NOT_FOUND)

    def handle_upload_images(self, sample_id: str) -> None:
        payload = read_body(self)
        files = payload.get("files", [])
        if not isinstance(files, list) or not files:
            error_response(self, "没有上传图片")
            return
        records = self.current_records()
        for record in records:
            if record.get("sample_id") != sample_id:
                continue
            sample_dir = SYSTEM_IMAGE_DIR / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            added = []
            for file_obj in files:
                name = str(file_obj.get("name", "upload.jpg"))
                ext = Path(name).suffix.lower() or ".jpg"
                if ext not in ALLOWED_IMAGE_EXTS:
                    raise ValueError(f"不支持的图片格式：{ext}")
                raw = str(file_obj.get("data", ""))
                if "," in raw and raw.lower().startswith("data:"):
                    raw = raw.split(",", 1)[1]
                data = base64.b64decode(raw)
                filename = next_image_name(record, ext)
                target = sample_dir / filename
                target.write_bytes(data)
                image_path = f"images/{sample_id}/{target.name}"
                record.setdefault("images", []).append(image_path)
                added.append(image_path)
            write_jsonl(ANNOTATIONS_PATH, records, reason="upload_images")
            json_response(self, {"ok": True, "images": added, "sample": record})
            return
        error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)

    def handle_delete_image(self, sample_id: str, image_name: str) -> None:
        records = self.current_records()
        for record in records:
            if record.get("sample_id") != sample_id:
                continue
            candidates = list(record.get("images", []) or []) + list(record.get("removed_images", []) or [])
            removed_path = next((path for path in candidates if Path(str(path)).name == image_name), None)
            if removed_path is None:
                error_response(self, "未找到要永久删除的图片")
                return
            record["images"] = [path for path in record.get("images", []) if path != removed_path]
            for key in ["mss_images", "removed_images"]:
                record[key] = [ref for ref in record.get(key, []) if ref != removed_path]
            if removed_path and str(removed_path).startswith("images/"):
                try:
                    resolved = resolve_image_path(self.project_root, str(removed_path))
                    if resolved.exists():
                        resolved.unlink()
                except ValueError:
                    pass
            write_jsonl(ANNOTATIONS_PATH, records, reason="permanent_delete_image")
            json_response(self, {"ok": True, "sample": record})
            return
        error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)

    def handle_logical_image_visibility(self, sample_id: str, remove_from_input: bool) -> None:
        payload = read_body(self)
        image_path = str(payload.get("path") or "")
        if not image_path:
            error_response(self, "需要图片路径")
            return
        records = self.current_records()
        for record in records:
            if record.get("sample_id") != sample_id:
                continue
            images = list(record.get("images", []) or [])
            removed = list(record.get("removed_images", []) or [])
            if remove_from_input:
                if image_path not in images:
                    error_response(self, "该图片不在当前模型输入中")
                    return
                images = [path for path in images if path != image_path]
                if image_path not in removed:
                    removed.append(image_path)
            else:
                if image_path not in removed:
                    error_response(self, "该图片不在当前题目的删除记录中")
                    return
                removed = [path for path in removed if path != image_path]
                if image_path not in images:
                    images.append(image_path)
            record["images"] = images
            record["removed_images"] = removed
            write_jsonl(ANNOTATIONS_PATH, records, reason="logical_remove_image" if remove_from_input else "restore_image")
            json_response(self, {"ok": True, "sample": record})
            return
        error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)

    def handle_validate(self) -> None:
        records = self.current_records()
        payload = read_body(self)
        if payload.get("source") == "workspace":
            records = self.review_workspace_records()
            meta = self.review_workspace_meta()
            validation = validate_dataset(records, self.project_root, self.rules)
            json_response(
                self,
                {
                    "ok": True,
                    "validation": validation,
                    "coverage": coverage(records),
                    "review_progress": review_progress(records, "workspace", meta.get("workspace_id")),
                },
            )
            return
        sample_id = payload.get("sample_id")
        if sample_id:
            for record in records:
                if record.get("sample_id") == sample_id:
                    json_response(self, {"ok": True, "issues": validate_record(record, self.project_root, self.rules)})
                    return
            error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)
            return
        validation = validate_dataset(records, self.project_root, self.rules)
        json_response(self, {"ok": True, "validation": validation, "coverage": coverage(records), "review_progress": review_progress(records, "production")})

    def handle_review(self) -> None:
        payload = read_body(self)
        sample_id = payload.get("sample_id")
        decision = payload.get("decision")
        if not sample_id or decision not in self.rules.get("review_decision", DEFAULT_RULES["review_decision"]):
            error_response(self, "需要样本编号和有效质检结论")
            return
        reviewer = str(payload.get("reviewer") or "").strip()
        if not reviewer or reviewer == "本地复核员":
            error_response(self, "请填写构题员编号或姓名")
            return
        entry = {
            "timestamp": utc_now(),
            "sample_id": sample_id,
            "source": payload.get("source") or "production",
            "workspace_id": payload.get("workspace_id") or "",
            "reviewer": reviewer,
            "decision": decision,
            "issue_type": payload.get("issue_type") or "",
            "remark": payload.get("remark") or "",
            "final_checks_confirmed": bool(payload.get("final_checks_confirmed")),
        }
        source = entry["source"]
        if source not in {"production", "workspace"}:
            error_response(self, "质检来源必须是 production 或 workspace")
            return
        if source == "workspace":
            entry["workspace_id"] = entry["workspace_id"] or self.review_workspace_meta().get("workspace_id", "")
        records = self.review_workspace_records() if source == "workspace" else self.current_records()
        target_record = next((record for record in records if record.get("sample_id") == sample_id), None)
        if target_record is None:
            error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)
            return
        record_issues = validate_record(target_record, self.project_root, self.rules)
        if decision in {"approve", "verify"} and any(item.get("severity") == "error" for item in record_issues):
            error_response(self, "当前样本仍有阻断性自动质检问题，不能通过或最终验收")
            return
        if decision == "verify":
            if source != "production":
                error_response(self, "外部质检工作区不能写入 verified；请回写生产数据后验收")
                return
            if target_record.get("quality_status") != "reviewed":
                error_response(self, "最终验收前样本必须已处于 reviewed 状态")
                return
            if not entry["final_checks_confirmed"]:
                error_response(self, "最终验收前必须确认构题员自检、结构校验、配额校验和批次检查均已完成")
                return
            candidate_records = copy.deepcopy(records)
            for candidate in candidate_records:
                if candidate.get("sample_id") == sample_id:
                    candidate["quality_status"] = "verified"
                    break
            candidate_validation = validate_dataset(candidate_records, self.project_root, self.rules)
            if not candidate_validation.get("ok"):
                error_response(self, "所在题组尚不满足最终组级约束，不能写入 verified")
                return
        new_status = "verified" if decision == "verify" else ("reviewed" if decision == "approve" else ("draft" if decision == "needs_revision" else "rejected"))
        found = False
        for record in records:
            if record.get("sample_id") == sample_id:
                record["quality_status"] = new_status
                found = True
                break
        if not found:
            error_response(self, f"未找到样本：{sample_id}", HTTPStatus.NOT_FOUND)
            return
        if source == "workspace":
            meta = self.review_workspace_meta()
            entry["workspace_id"] = entry["workspace_id"] or meta.get("workspace_id", "")
            write_review_workspace(records, reason="review")
        else:
            write_jsonl(ANNOTATIONS_PATH, records, reason="review")
        append_review_entry(entry)
        if source == "workspace":
            progress = review_progress(records, "workspace", entry.get("workspace_id"))
        else:
            progress = review_progress(records, "production")
        json_response(self, {"ok": True, "review": entry, "quality_status": new_status, "review_progress": progress})

    def handle_export(self) -> None:
        payload = read_body(self)
        verified_only = bool(payload.get("verified_only"))
        filename = payload.get("filename")
        if filename:
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
            if not filename.endswith(".jsonl"):
                filename += ".jsonl"
        else:
            filename = f"annotations_export_{int(time.time())}.jsonl"
        target = OUTPUT_DIR / filename
        records = self.current_records()
        if verified_only:
            records = [record for record in records if record.get("quality_status") == "verified"]
            if not records:
                error_response(self, "当前没有可发布的 verified 样本")
                return
            validation = validate_dataset(records, self.project_root, self.rules)
            if not validation.get("ok"):
                error_response(self, "verified 子集仍有阻断性问题，禁止发布导出")
                return
            write_jsonl(target, records, backup=False, reason="verified_export")
        else:
            shutil.copy2(ANNOTATIONS_PATH, target)
        json_response(self, {"ok": True, "export_path": str(target), "verified_only": verified_only, "count": len(records)})

    def handle_review_workspace_export(self) -> None:
        payload = read_body(self)
        filename = payload.get("filename")
        if filename:
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
            if not filename.endswith(".jsonl"):
                filename += ".jsonl"
        else:
            filename = f"review_workspace_export_{int(time.time())}.jsonl"
        target = OUTPUT_DIR / filename
        shutil.copy2(REVIEW_WORKSPACE_PATH, target)
        json_response(self, {"ok": True, "export_path": str(target)})


def run_server(project_root: Path, host: str, port: int) -> None:
    ensure_dirs()
    rules = load_rules()
    httpd = ThreadingHTTPServer((host, port), WorkbenchHandler)
    httpd.project_root = project_root.resolve()  # type: ignore[attr-defined]
    httpd.rules = rules  # type: ignore[attr-defined]
    print(f"Frame-CausalEvidence workbench v3.0.3: http://{host}:{port}")
    print(f"project_root={project_root.resolve()}")
    print(f"dataset={SYSTEM_DATASET_DIR}")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-CausalEvidence v3 annotation and QC system")
    parser.add_argument("--root", default=str(APP_DIR.parent), help="v3 project root")
    parser.add_argument(
        "--data-root",
        default=str(APP_DIR / "dataset"),
        help="dataset directory (default: system-local dataset/)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    configure_data_root(Path(args.data_root))
    run_server(Path(args.root), args.host, args.port)


if __name__ == "__main__":
    main()
