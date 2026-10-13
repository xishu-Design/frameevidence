from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps
from jsonschema import Draft202012Validator

from .baselines import shortcut_baselines
from .common import read_jsonl, referenced_paths, sha256_file


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v3_dry_run_annotation_app", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or path.suffix.lstrip(".").upper()
        oriented = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(oriented.get_flattened_data() if hasattr(oriented, "get_flattened_data") else oriented.getdata())
        bits = []
        for row in range(8):
            offset = row * 9
            bits.extend(pixels[offset + column + 1] > pixels[offset + column] for column in range(8))
        dhash = int("".join("1" if value else "0" for value in bits), 2)
        return {
            "format": image_format,
            "width": width,
            "height": height,
            "pixels": width * height,
            "aspect_ratio": width / height if height else 0.0,
            "mode": image.mode,
            "bytes": path.stat().st_size,
            "dhash": f"{dhash:016x}",
        }


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _overlap(first: Counter[int], second: Counter[int]) -> float:
    total_first = sum(first.values())
    total_second = sum(second.values())
    if not total_first or not total_second:
        return 0.0
    return sum(min(first[value] / total_first, second[value] / total_second) for value in set(first) | set(second))


def _manifest_sources(dataset_dir: Path) -> dict[str, int]:
    path = dataset_dir / "merge_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    return dict(Counter(str(row.get("source_dataset", "unknown")) for row in manifest.get("group_mappings", [])))


def audit_dataset(
    v3_root: Path,
    dataset_dir: Path,
    auxiliary_metadata: dict[str, dict[str, Any]] | None = None,
    permutations: int = 1_000,
    seed: int = 20260722,
) -> dict[str, Any]:
    annotation_path = dataset_dir / "annotations.jsonl"
    # The active workspace was renumbered from 03_* to 02_* during the
    # Chinese-directory cleanup.  Keep the historical path as a fallback so
    # archived deliveries remain auditable.
    schema_candidates = (
        v3_root / "02_数据模式与标注试点" / "data_schema.json",
        v3_root / "03_数据模式与标注试点" / "data_schema.json",
    )
    schema_path = next((path for path in schema_candidates if path.is_file()), schema_candidates[0])
    records = read_jsonl(annotation_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    expected_fields = set(schema.get("properties", {}))

    ids: set[str] = set()
    path_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sample_id = str(record.get("sample_id", "<missing>"))
        if sample_id in ids:
            errors.append({"code": "duplicate_sample_id", "sample_id": sample_id, "message": "样本 ID 重复"})
        ids.add(sample_id)
        if set(record) != expected_fields:
            errors.append({
                "code": "field_set_mismatch",
                "sample_id": sample_id,
                "message": f"字段集合不等于权威 17 字段；缺少 {sorted(expected_fields - set(record))}；多出 {sorted(set(record) - expected_fields)}",
            })
        for issue in sorted(validator.iter_errors(record), key=lambda value: list(value.path)):
            location = ".".join(str(part) for part in issue.path) or "<root>"
            errors.append({"code": "json_schema", "sample_id": sample_id, "message": f"{location}: {issue.message}"})
        for field in ("images", "removed_images", "mss_images"):
            for value in record.get(field, []) or []:
                relative = str(value).replace("\\", "/")
                path_groups[relative].add(str(record.get("relation_group_id", "")))
                candidate = dataset_dir / relative
                if not relative.startswith("images/") or ".." in Path(relative).parts or Path(relative).is_absolute():
                    errors.append({"code": "unsafe_image_path", "sample_id": sample_id, "message": f"{field}: {value}"})
                elif not candidate.is_file():
                    errors.append({"code": "missing_image", "sample_id": sample_id, "message": f"{field}: {value}"})

    app_path = v3_root / "数据生产与质量检验系统" / "app.py"
    app = _load_module(app_path)
    app.configure_data_root(dataset_dir)
    business = app.validate_dataset(records, v3_root, app.load_rules())
    for issue in business.get("issues", []):
        normalized = {
            "code": str(issue.get("code", "business_rule")),
            "sample_id": issue.get("sample_id"),
            "message": str(issue.get("message", "")),
        }
        (errors if issue.get("severity") == "error" else warnings).append(normalized)

    unique_references = referenced_paths(records)
    image_metadata: dict[str, dict[str, Any]] = {}
    sha_by_path: dict[str, str] = {}
    for relative in unique_references:
        path = dataset_dir / relative
        if not path.is_file():
            continue
        try:
            image_metadata[relative] = _image_metadata(path)
            sha_by_path[relative] = sha256_file(path)
        except Exception as exc:  # noqa: BLE001 - audit must report corrupt assets
            errors.append({"code": "unreadable_image", "sample_id": None, "message": f"{relative}: {exc}"})

    physical_assets = sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in (dataset_dir / "images").rglob("*")
        if path.is_file()
    )
    orphan_assets = sorted(set(physical_assets) - set(unique_references))
    if orphan_assets:
        warnings.append({"code": "orphan_assets", "sample_id": None, "message": f"存在 {len(orphan_assets)} 张未引用图片"})

    paths_by_sha: dict[str, list[str]] = defaultdict(list)
    for relative, digest in sha_by_path.items():
        paths_by_sha[digest].append(relative)
    exact_cross_group: list[dict[str, Any]] = []
    for digest, paths in paths_by_sha.items():
        groups = sorted({group for path in paths for group in path_groups.get(path, set()) if group})
        if len(groups) > 1:
            exact_cross_group.append({"sha256": digest, "paths": sorted(paths), "relation_groups": groups})

    near_cross_group: list[dict[str, Any]] = []
    metadata_items = sorted(image_metadata.items())
    for index, (first_path, first) in enumerate(metadata_items):
        first_groups = path_groups.get(first_path, set())
        for second_path, second in metadata_items[index + 1 :]:
            second_groups = path_groups.get(second_path, set())
            if first_groups & second_groups or not first_groups or not second_groups:
                continue
            distance = _hamming(str(first["dhash"]), str(second["dhash"]))
            if distance <= 4:
                near_cross_group.append({
                    "first": first_path,
                    "second": second_path,
                    "hamming_distance": distance,
                    "first_groups": sorted(first_groups),
                    "second_groups": sorted(second_groups),
                })

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("relation_group_id", ""))].append(record)
    group_type = {group_id: rows[0].get("question_type") for group_id, rows in groups.items() if rows}
    status_counts = Counter(str(record.get("quality_status")) for record in records)
    answer_counts = Counter(str(record.get("answer")) for record in records)
    frame_counts = Counter(str(record.get("reference_frame")) for record in records)
    condition_counts = Counter(str(record.get("view_condition")) for record in records)
    image_count_by_condition: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        if record.get("question_type") == "FQ2":
            image_count_by_condition[str(record["view_condition"])][len(record.get("images", []))] += 1
    condition_names = sorted(image_count_by_condition)
    overlap = {}
    for index, first in enumerate(condition_names):
        for second in condition_names[index + 1 :]:
            overlap[f"{first}__{second}"] = _overlap(image_count_by_condition[first], image_count_by_condition[second])

    # 本轮采用 dry_run_descriptive：relation_group_id 仅作为行为观测单位，
    # 不引入 split_lock_id，也不进行推断统计。
    shortcuts = shortcut_baselines(records, image_metadata, permutations, seed)

    warning_samples = sorted({str(issue["sample_id"]) for issue in warnings if issue.get("sample_id")})
    warning_groups = sorted({
        str(record["relation_group_id"])
        for record in records
        if str(record.get("sample_id")) in warning_samples
    })
    review_reasons: list[str] = []
    if status_counts.get("draft", 0):
        review_reasons.append(f"{status_counts['draft']} 条记录仍为 draft")
    if warnings:
        review_reasons.append(f"存在 {len(warnings)} 条业务/资产警告")
    known_limitations: list[str] = []
    if shortcuts["overall_risk"] == "HIGH_RISK":
        known_limitations.append("FQ2 图片数量/元数据捷径审计为 HIGH_RISK；本轮保留为已知限制，不阻断探索性模型调用")
    if exact_cross_group or near_cross_group:
        known_limitations.append("存在跨题组重复或感知近重复候选；正式扩充数据时应按物理场景和采集批次隔离")
    status = "BLOCKED" if errors else "REVIEW_REQUIRED" if review_reasons else "PASS"

    formats = Counter(str(value["format"]) for value in image_metadata.values())
    widths = [int(value["width"]) for value in image_metadata.values()]
    heights = [int(value["height"]) for value in image_metadata.values()]
    return {
        "status": status,
        "evaluation_profile": "dry_run_descriptive",
        "result_label": "exploratory_dry_run",
        "dataset": str(dataset_dir.resolve()),
        "samples": len(records),
        "relation_groups": len(groups),
        "groups_by_type": dict(Counter(str(value) for value in group_type.values())),
        "samples_by_type": dict(Counter(str(record.get("question_type")) for record in records)),
        "group_sizes": dict(Counter(str(len(rows)) for rows in groups.values())),
        "referenced_assets": len(unique_references),
        "physical_assets": len(physical_assets),
        "orphan_assets": orphan_assets,
        "quality_status": dict(sorted(status_counts.items())),
        "answer_distribution": dict(sorted(answer_counts.items())),
        "reference_frame_distribution": dict(sorted(frame_counts.items())),
        "view_condition_distribution": dict(sorted(condition_counts.items())),
        "source_group_distribution": _manifest_sources(dataset_dir),
        "image_metadata_summary": {
            "formats": dict(sorted(formats.items())),
            "distinct_sizes": len({(value["width"], value["height"]) for value in image_metadata.values()}),
            "width_range": [min(widths), max(widths)] if widths else None,
            "height_range": [min(heights), max(heights)] if heights else None,
        },
        "image_metadata": image_metadata,
        "image_count_by_fq2_condition": {
            condition: {str(count): amount for count, amount in sorted(values.items())}
            for condition, values in sorted(image_count_by_condition.items())
        },
        "image_count_overlap_coefficients": overlap,
        "duplicates": {
            "exact_cross_group_count": len(exact_cross_group),
            "exact_cross_group": exact_cross_group[:100],
            "near_cross_group_count": len(near_cross_group),
            "near_cross_group": near_cross_group[:100],
            "near_duplicate_rule": "64-bit dHash Hamming distance <= 4；仅作跨题组近重复候选，不自动判定为重复",
        },
        "shortcut_baselines": shortcuts,
        "errors": errors,
        "warnings": warnings,
        "warning_sample_ids": warning_samples,
        "warning_relation_group_ids": warning_groups,
        "review_reasons": review_reasons,
        "known_limitations": known_limitations,
        "statistical_availability": {
            "descriptive_group_metrics": True,
            "confidence_intervals": False,
            "formal_paired_inference": False,
            "power_analysis": False,
            "minimum_detectable_difference": False,
            "reason": "dry_run_descriptive 配置只报告描述性指标",
        },
    }


def render_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Dry Run 数据预检报告",
        "",
        f"- 总体状态：**{audit['status']}**",
        f"- 样本/题组：{audit['samples']} 条 / {audit['relation_groups']} 组",
        f"- FQ1/FQ2 题组：{audit['groups_by_type'].get('FQ1', 0)} / {audit['groups_by_type'].get('FQ2', 0)}",
        f"- 引用/物理图片：{audit['referenced_assets']} / {audit['physical_assets']}",
        f"- 质量状态：`{json.dumps(audit['quality_status'], ensure_ascii=False)}`",
        "",
        "## FQ2 图片数量与捷径",
        "",
        f"- 条件分布：`{json.dumps(audit['image_count_by_fq2_condition'], ensure_ascii=False)}`",
        f"- 两两重叠系数：`{json.dumps(audit['image_count_overlap_coefficients'], ensure_ascii=False)}`",
        f"- 捷径风险：**{audit['shortcut_baselines']['overall_risk']}**",
        "",
        "| 目标 | Baseline | Accuracy | Macro-F1 | 多数类提升 | 置换 p | 风险 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for target, result in audit["shortcut_baselines"]["tasks"].items():
        for baseline in ("count_only", "meta_only"):
            value = result[baseline]
            lines.append(
                f"| {target} | {baseline} | {value['accuracy']:.3f} | {value['macro_f1']:.3f} | "
                f"{value['accuracy_lift']:+.3f} | {value['permutation_p_value']:.4f} | {value['risk']} |"
            )
    count_condition = audit["shortcut_baselines"]["tasks"]["view_condition"]["count_only"]
    lines.extend([
        "",
        f"- count-only 条件预测的全样本经验映射准确率：{count_condition['descriptive_resubstitution_accuracy']:.3f}；"
        f"唯一对应条件且支持数≥2的图片数：`{json.dumps(count_condition['exclusive_count_values_support_ge_2'], ensure_ascii=False)}`。",
        "- 上述经验映射只用于结构性泄漏诊断，不是交叉验证泛化性能；当前 HIGH_RISK 由该结构性规则触发。",
        "",
        "## 评测边界",
        "",
        "- 配置：`dry_run_descriptive`；只报告 relation-group 题组层描述性指标。",
        "- 本轮不计算置信区间、显著性检验、功效分析或最小可检测差异。",
        "- 捷径风险作为 `KNOWN_LIMITATION` 保留，不阻断探索性模型调用。",
        "",
        "## 错误与警告",
        "",
    ])
    if not audit["errors"]:
        lines.append("- 结构性错误：0")
    for issue in audit["errors"]:
        lines.append(f"- [ERROR] `{issue['code']}` {issue.get('sample_id') or ''}：{issue['message']}")
    if not audit["warnings"]:
        lines.append("- 警告：0")
    for issue in audit["warnings"]:
        lines.append(f"- [WARNING] `{issue['code']}` {issue.get('sample_id') or ''}：{issue['message']}")
    lines.extend(["", "## 当前判断", ""])
    if audit["review_reasons"]:
        lines.extend(f"- {reason}" for reason in audit["review_reasons"])
    else:
        lines.append("- 未发现阻断或待复核项。")
    for limitation in audit.get("known_limitations", []):
        lines.append(f"- [KNOWN_LIMITATION] {limitation}")
    lines.append("")
    return "\n".join(lines)
