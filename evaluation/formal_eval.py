#!/usr/bin/env python3
"""FrameEvidence compact-320 formal evaluation entry point.

The CLI deliberately separates no-cost data/table work from paid model calls.
`run` is the only subcommand that contacts model endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V3_ROOT = HERE.parent


def _locate_dry_run_root() -> Path:
    """Locate ``dry_run_eval`` without depending on a localized directory name."""

    candidates = [HERE]
    candidates.extend(
        sorted(
            (path for path in V3_ROOT.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    )
    for candidate in candidates:
        if (candidate / "dry_run_eval" / "runner.py").is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate dry_run_eval below {V3_ROOT}")


DRY_RUN_ROOT = _locate_dry_run_root()
if str(DRY_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(DRY_RUN_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dry_run_eval.audit import audit_dataset  # noqa: E402
from dry_run_eval.common import (  # noqa: E402
    DEFAULT_SEED,
    dataset_fingerprint,
    safe_path_component,
    write_json,
    write_jsonl,
)
from dry_run_eval.prepare import (  # noqa: E402
    IMAGE_ORDERING_POLICY,
    SINGLE_IMAGE_RULE,
    prepare_inputs,
)
from dry_run_eval.runner import check_model_config, run_models  # noqa: E402
from formal_eval_core import (  # noqa: E402
    ANSWER_LABELS,
    build_asset_cluster_map,
    load_cluster_manifest,
    read_jsonl,
    score_run,
    shortcut_audit,
    validate_release_contract,
    write_main_tables,
    write_shortcut_outputs,
)
from formal_eval_dataset import (  # noqa: E402
    DEFAULT_GROUP_TARGETS,
    DEFAULT_TASK_TARGETS,
    build_dataset_statistics,
    generate_stratified_group_split,
    load_cluster_split_overrides,
    load_group_records,
    load_split_manifest,
    validate_split_assignment,
    write_dataset_stats_tables,
    write_split_outputs,
)


def _locate_frame_file(filename: str) -> Path:
    """Resolve a unique project file without spelling its localized parent."""

    matches = sorted(
        (path for path in V3_ROOT.rglob(filename) if path.is_file()),
        key=lambda path: (len(path.relative_to(V3_ROOT).parts), path.as_posix()),
    )
    if matches:
        return matches[0]
    return V3_ROOT / filename


DEFAULT_DATASET = V3_ROOT.parent / "daily_20260922T075500Z" / "dataset"
DEFAULT_POLICY = _locate_frame_file("formal_release_scale_policy.json")
DEFAULT_OUTPUT_ROOT = V3_ROOT / "results" / "formal_release_v1"
DEFAULT_CONFIG = HERE / "configs" / "models.formal.json"
DEFAULT_QUOTA = _locate_frame_file("formal_release_quota_compact_320.csv")
DEFAULT_SPLIT_OVERRIDES = _locate_frame_file("formal_release_split_overrides.csv")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(args: argparse.Namespace) -> tuple[Path, Path, str]:
    dataset = args.dataset.resolve()
    annotation_path = dataset / "annotations.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"缺少正式标注文件：{annotation_path}")
    fingerprint = dataset_fingerprint(dataset)
    evaluation_split = getattr(args, "eval_split", None)
    suffix = f"_{evaluation_split}" if evaluation_split else ""
    run_dir = args.output_root.resolve() / f"formal_{fingerprint[:12]}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return dataset, run_dir, fingerprint


def _base_run_dir(args: argparse.Namespace, fingerprint: str) -> Path:
    return args.output_root.resolve() / f"formal_{fingerprint[:12]}"


def _resolve_split_manifest(args: argparse.Namespace, fingerprint: str) -> Path:
    requested = getattr(args, "split_manifest", None)
    if requested is not None:
        return requested.resolve()
    return _base_run_dir(args, fingerprint) / "manifests" / "split_manifest.csv"


def _group_ids_sha256(group_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(group_ids)).encode("utf-8")).hexdigest()


def _semantic_diagnostics(dataset: Path) -> dict[str, Any]:
    rows = read_jsonl(dataset / "annotations.jsonl")
    typo_samples = [
        str(row.get("sample_id"))
        for row in rows
        if "哪一测" in str(row.get("question", ""))
    ]
    empty_bridge = [
        str(row.get("sample_id"))
        for row in rows
        if not row.get("bridge_instance_ids")
    ]
    return {
        "likely_question_typo_哪一测": {
            "count": len(typo_samples),
            "examples": typo_samples[:10],
            "requires_review": bool(typo_samples),
        },
        "empty_bridge_instance_ids": {
            "count": len(empty_bridge),
            "examples": empty_bridge[:10],
            "requires_review": False,
            "interpretation": (
                "Informational only: empty is valid for direct relations; the all-empty distribution "
                "should still be described if the benchmark claims multi-hop bridge coverage."
            ),
        },
    }


def _write_cluster_map(
    run_dir: Path,
    cluster_by_group: dict[str, str],
    source: str,
    stats: dict[str, Any],
) -> None:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "cluster_map.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relation_group_id", "scene_cluster_id"])
        writer.writeheader()
        for group_id, cluster_id in sorted(cluster_by_group.items()):
            writer.writerow({"relation_group_id": group_id, "scene_cluster_id": cluster_id})
    write_json(
        analysis_dir / "cluster_map_stats.json",
        {"source": source, **stats, "cluster_map": str(csv_path.resolve())},
    )


def _clusters(
    dataset: Path,
    run_dir: Path,
    manifest: Path | None,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    expected_groups = {
        str(row["relation_group_id"])
        for row in read_jsonl(dataset / "annotations.jsonl")
    }
    if manifest is not None:
        resolved = manifest.resolve()
        mapping = load_cluster_manifest(resolved)
        missing = sorted(expected_groups - set(mapping))
        extra = sorted(set(mapping) - expected_groups)
        if missing or extra:
            raise ValueError(
                "scene cluster manifest 与正式题组集合不一致："
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        mapping = {group_id: mapping[group_id] for group_id in sorted(expected_groups)}
        stats = {
            "method": "restricted_scene_cluster_manifest",
            "groups": len(mapping),
            "clusters": len(set(mapping.values())),
            "manifest": str(resolved),
            "manifest_sha256": _file_sha256(resolved),
        }
        source = f"external_scene_cluster_id:{resolved.name}"
    else:
        mapping, stats = build_asset_cluster_map(dataset)
        source = "asset_sha256_connected_components_fallback"
    _write_cluster_map(run_dir, mapping, source, stats)
    return mapping, source, stats


def _render_preflight(report: dict[str, Any]) -> str:
    audit = report["dataset_audit"]
    contract = report["release_contract"]
    clusters = report["clustering"]
    lines = [
        "# FrameEvidence formal-release preflight",
        "",
        f"- Status: **{report['status']}**",
        f"- Dataset fingerprint: `{report['dataset_fingerprint']}`",
        f"- Samples/groups: {contract['records']} / {contract['groups']}",
        f"- FQ1/FQ2 groups: {contract['groups_by_type'].get('FQ1', 0)} / "
        f"{contract['groups_by_type'].get('FQ2', 0)}",
        f"- Dataset warnings/errors: {len(audit.get('warnings', []))} / "
        f"{len(audit.get('errors', []))}",
        f"- Exact asset clusters: {clusters.get('clusters', 'N/A')} "
        f"(fallback={clusters.get('fallback_used', False)})",
        "",
        "## Review reasons",
        "",
    ]
    reasons = report.get("review_reasons", [])
    lines.extend([f"- {reason}" for reason in reasons] or ["- None."])
    lines.extend([
        "",
        "## Blocking errors",
        "",
    ])
    errors = report.get("blocking_errors", [])
    lines.extend([f"- {error}" for error in errors] or ["- None."])
    lines.extend([
        "",
        "## No-cost shortcut audit",
        "",
        "The detailed, frozen-release empirical mappings are in `analysis/shortcut_audit.*`.",
        "They are controls, not model rows in the main table.",
        "",
    ])
    return "\n".join(lines)


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    dataset, run_dir, fingerprint = _context(args)
    dataset_audit = audit_dataset(
        V3_ROOT,
        dataset,
        None,
        args.permutations,
        args.seed,
    )
    dataset_audit["evaluation_profile"] = "formal_release"
    dataset_audit["result_label"] = "formal_release_preflight"
    contract = validate_release_contract(dataset, args.policy.resolve())
    shortcut = shortcut_audit(dataset, args.policy.resolve())
    cluster_map, cluster_source, cluster_stats = _clusters(
        dataset,
        run_dir,
        args.cluster_manifest,
    )
    write_shortcut_outputs(shortcut, run_dir / "analysis")
    semantic = _semantic_diagnostics(dataset)

    blocking_errors = [str(value) for value in contract.get("errors", [])]
    blocking_errors.extend(
        f"{issue.get('code')}: {issue.get('message')}"
        for issue in dataset_audit.get("errors", [])
    )
    review_reasons = [str(value) for value in contract.get("warnings", [])]
    review_reasons.extend(str(value) for value in dataset_audit.get("review_reasons", []))
    exact_reuse = int(dataset_audit.get("duplicates", {}).get("exact_cross_group_count", 0))
    near_reuse = int(dataset_audit.get("duplicates", {}).get("near_cross_group_count", 0))
    if exact_reuse:
        review_reasons.append(
            f"检测到 {exact_reuse} 个跨题组精确重复图片 SHA；违反独立资产设计要求，正式解释前须审计。"
        )
    if near_reuse:
        review_reasons.append(
            f"检测到 {near_reuse} 对跨题组感知近重复候选；需人工确认物理场景复用。"
        )
    if args.cluster_manifest is None:
        review_reasons.append(
            "未提供受限 scene_cluster_id 台账；置信区间暂用精确图片 SHA 连通分量聚类，不能替代场景台账。"
        )
    for name, value in semantic.items():
        if value["requires_review"]:
            review_reasons.append(f"语义诊断 {name}: {value['count']} 条。")
    # Preserve order but avoid repeating the same reason from nested audits.
    review_reasons = list(dict.fromkeys(review_reasons))
    status = "BLOCKED" if blocking_errors else "REVIEW_REQUIRED" if review_reasons else "PASS"
    report = {
        "status": status,
        "evaluation_profile": "formal_release",
        "result_label": "formal_release_preflight",
        "dataset": str(dataset),
        "dataset_fingerprint": fingerprint,
        "policy": str(args.policy.resolve()),
        "policy_sha256": _file_sha256(args.policy.resolve()),
        "dataset_audit": dataset_audit,
        "release_contract": contract,
        "shortcut_audit": shortcut,
        "clustering": {
            "source": cluster_source,
            "fallback_used": args.cluster_manifest is None,
            **cluster_stats,
            "mapped_groups": len(cluster_map),
        },
        "semantic_diagnostics": semantic,
        "blocking_errors": blocking_errors,
        "review_reasons": review_reasons,
    }
    output_dir = run_dir / "data_audit"
    write_json(output_dir / "formal_preflight.json", report)
    (output_dir / "formal_preflight.md").write_text(
        _render_preflight(report),
        encoding="utf-8",
    )
    return report, run_dir


def _load_or_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    _, run_dir, _ = _context(args)
    path = run_dir / "data_audit" / "formal_preflight.json"
    if path.is_file() and not getattr(args, "refresh_preflight", False):
        cached = json.loads(path.read_text(encoding="utf-8-sig"))
        requested_manifest = (
            str(args.cluster_manifest.resolve())
            if getattr(args, "cluster_manifest", None) is not None
            else None
        )
        cached_manifest = cached.get("clustering", {}).get("manifest")
        same_policy = (
            cached.get("policy") == str(args.policy.resolve())
            and cached.get("policy_sha256") == _file_sha256(args.policy.resolve())
        )
        if same_policy and requested_manifest == cached_manifest:
            return cached, run_dir
    return _preflight(args)


def _split_selection(
    args: argparse.Namespace,
    fingerprint: str,
) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    evaluation_split = str(getattr(args, "eval_split", "test")).lower()
    if evaluation_split not in DEFAULT_GROUP_TARGETS:
        raise ValueError(
            f"--eval-split 必须为 {sorted(DEFAULT_GROUP_TARGETS)} 之一，"
            f"实际为 {evaluation_split}"
        )
    split_manifest = _resolve_split_manifest(args, fingerprint)
    assignment, cluster_by_group, cluster_source = load_split_manifest(split_manifest)
    records = read_jsonl(dataset / "annotations.jsonl")
    dataset_group_counts = Counter(str(row["relation_group_id"]) for row in records)
    dataset_groups = set(dataset_group_counts)
    manifest_groups = set(assignment)
    missing = sorted(dataset_groups - manifest_groups)
    extra = sorted(manifest_groups - dataset_groups)
    if missing or extra:
        raise ValueError(
            "split manifest 与正式数据题组不一致："
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    invalid_splits = sorted(set(assignment.values()) - set(DEFAULT_GROUP_TARGETS))
    if invalid_splits:
        raise ValueError(f"split manifest 含非法 split：{invalid_splits}")
    split_group_counts = Counter(assignment.values())
    for split, expected in DEFAULT_GROUP_TARGETS.items():
        if split_group_counts[split] != expected:
            raise ValueError(
                f"split manifest 的 {split} 题组数应为 {expected}，"
                f"实际为 {split_group_counts[split]}"
            )
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for group_id, split in assignment.items():
        cluster_splits[cluster_by_group[group_id]].add(split)
    leaked_clusters = sorted(
        cluster_id for cluster_id, splits in cluster_splits.items() if len(splits) > 1
    )
    if leaked_clusters:
        raise ValueError(
            "split manifest 存在 scene cluster 跨 split 泄漏，"
            f"例如：{leaked_clusters[:3]}"
        )
    selected_group_ids = {
        group_id for group_id, split in assignment.items() if split == evaluation_split
    }
    expected_groups = DEFAULT_GROUP_TARGETS[evaluation_split]
    if len(selected_group_ids) != expected_groups:
        raise ValueError(
            f"{evaluation_split} 必须恰好包含 {expected_groups} 个题组，"
            f"实际为 {len(selected_group_ids)}"
        )
    malformed_groups = sorted(
        group_id
        for group_id in selected_group_ids
        if dataset_group_counts[group_id] != 3
    )
    if malformed_groups:
        raise ValueError(
            f"{evaluation_split} 中每个题组必须恰好有 3 个 source samples，"
            f"异常题组：{malformed_groups[:3]}"
        )
    selected_source_samples = sum(dataset_group_counts[group_id] for group_id in selected_group_ids)
    expected_source_samples = expected_groups * 3
    if selected_source_samples != expected_source_samples:
        raise ValueError(
            f"{evaluation_split} source samples 应为 {expected_source_samples}，"
            f"实际为 {selected_source_samples}"
        )
    split_summary = split_manifest.with_name("split_summary.json")
    if not split_summary.is_file():
        raise FileNotFoundError(f"缺少 split summary：{split_summary}")
    summary = json.loads(split_summary.read_text(encoding="utf-8-sig"))
    manifest_sha256 = _file_sha256(split_manifest)
    if summary.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            "split_summary.json 中的 manifest_sha256 与 split_manifest.csv 不一致"
        )
    return {
        "evaluation_split": evaluation_split,
        "split_manifest": split_manifest,
        "split_summary": split_summary,
        "split_manifest_sha256": manifest_sha256,
        "split_summary_sha256": _file_sha256(split_summary),
        "cluster_source": cluster_source,
        "selected_group_ids": selected_group_ids,
        "selected_group_ids_sha256": _group_ids_sha256(selected_group_ids),
        "expected_groups": expected_groups,
        "expected_source_samples": expected_source_samples,
        "expected_mapping_rows": expected_source_samples * 3,
    }


def _prepared_cache_errors(
    run_dir: Path,
    manifest: dict[str, Any],
    selection: dict[str, Any],
    *,
    fingerprint: str,
    seed: int,
) -> list[str]:
    errors: list[str] = []
    expected_groups = selection["selected_group_ids"]
    expected_rows = selection["expected_mapping_rows"]
    expected_samples = selection["expected_source_samples"]
    selected = manifest.get("selection", {})
    checks = {
        "dataset_fingerprint": (manifest.get("dataset_fingerprint"), fingerprint),
        "seed": (manifest.get("seed"), seed),
        "evaluation_profile": (manifest.get("evaluation_profile"), "formal_release"),
        "evaluation_split": (
            selected.get("evaluation_split"),
            selection["evaluation_split"],
        ),
        "split_manifest_sha256": (
            selected.get("split_manifest_sha256"),
            selection["split_manifest_sha256"],
        ),
        "allowed_group_ids_sha256": (
            selected.get("allowed_group_ids_sha256"),
            selection["selected_group_ids_sha256"],
        ),
        "selected_group_count": (
            selected.get("selected_group_count"),
            selection["expected_groups"],
        ),
        "source_samples": (manifest.get("source_samples"), expected_samples),
        "public_items": (manifest.get("public_items"), expected_rows),
        "ordering": (manifest.get("ordering"), IMAGE_ORDERING_POLICY),
        "single_image_rule": (manifest.get("single_image_rule"), SINGLE_IMAGE_RULE),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}: cached={actual!r}, expected={expected!r}")

    mapping_path = run_dir / "private" / "evaluation_mapping.jsonl"
    public_path = run_dir / "prepared" / "prepared_inputs.jsonl"
    try:
        mapping = read_jsonl(mapping_path)
        public = read_jsonl(public_path)
    except Exception as exc:  # noqa: BLE001 - converted into a cache mismatch report
        errors.append(f"无法读取缓存 JSONL: {exc}")
        return errors
    if len(mapping) != expected_rows:
        errors.append(f"mapping rows: cached={len(mapping)}, expected={expected_rows}")
    if len(public) != expected_rows:
        errors.append(f"public rows: cached={len(public)}, expected={expected_rows}")
    mapping_groups = {str(row.get("relation_group_id")) for row in mapping}
    if mapping_groups != expected_groups:
        errors.append(
            "mapping groups 与所选 split 不一致："
            f"missing={sorted(expected_groups - mapping_groups)[:3]} "
            f"extra={sorted(mapping_groups - expected_groups)[:3]}"
        )
    sample_ids = {str(row.get("sample_id")) for row in mapping}
    if len(sample_ids) != expected_samples:
        errors.append(
            f"mapping unique source samples: cached={len(sample_ids)}, expected={expected_samples}"
        )
    group_row_counts = Counter(str(row.get("relation_group_id")) for row in mapping)
    malformed_group_rows = sorted(
        group_id for group_id in expected_groups if group_row_counts[group_id] != 9
    )
    if malformed_group_rows:
        errors.append(
            "mapping 中每个题组必须恰好有 9 行（3 samples × 3 modes），"
            f"异常题组={malformed_group_rows[:3]}"
        )
    modes = Counter(str(row.get("evaluation_mode")) for row in mapping)
    expected_per_mode = expected_samples
    expected_modes = {mode: expected_per_mode for mode in ("all-images", "text-only", "single-image")}
    if dict(modes) != expected_modes:
        errors.append(f"mapping modes: cached={dict(modes)}, expected={expected_modes}")
    modes_by_sample: dict[str, list[str]] = defaultdict(list)
    for row in mapping:
        modes_by_sample[str(row.get("sample_id"))].append(str(row.get("evaluation_mode")))
    required_modes = {"all-images", "text-only", "single-image"}
    malformed_samples = sorted(
        sample_id
        for sample_id, sample_modes in modes_by_sample.items()
        if len(sample_modes) != 3 or set(sample_modes) != required_modes
    )
    if malformed_samples:
        errors.append(
            "mapping 中每个 source sample 必须恰好覆盖 3 个 evaluation modes，"
            f"异常 sample={malformed_samples[:3]}"
        )
    mapping_eval_ids = [str(row.get("eval_id", "")) for row in mapping]
    public_eval_ids = [str(row.get("eval_id", "")) for row in public]
    if (
        not all(mapping_eval_ids)
        or len(set(mapping_eval_ids)) != len(mapping_eval_ids)
        or mapping_eval_ids != public_eval_ids
    ):
        errors.append("public/private eval_id 顺序、唯一性或完整性校验失败")
    return errors


def _copy_split_provenance(run_dir: Path, selection: dict[str, Any]) -> None:
    target_dir = run_dir / "manifests"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (selection["split_manifest"], "split_manifest.csv"),
        (selection["split_summary"], "split_summary.json"),
    ):
        target = target_dir / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)


def _ensure_prepared(args: argparse.Namespace, report: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    fingerprint = str(report.get("dataset_fingerprint") or dataset_fingerprint(args.dataset.resolve()))
    selection = _split_selection(args, fingerprint)
    manifest_path = run_dir / "run_manifest.json"
    mapping_path = run_dir / "private" / "evaluation_mapping.jsonl"
    public_path = run_dir / "prepared" / "prepared_inputs.jsonl"
    cache_paths = (manifest_path, mapping_path, public_path)
    existing_count = sum(path.is_file() for path in cache_paths)
    force_prepare = bool(getattr(args, "force_prepare", False))
    if existing_count and not force_prepare:
        if existing_count != len(cache_paths):
            raise RuntimeError(
                "发现不完整的 prepared 缓存；为防止 split 污染，"
                "已阻断自动覆盖。确认后使用 --force-prepare 重建。"
            )
        cached = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        errors = _prepared_cache_errors(
            run_dir,
            cached,
            selection,
            fingerprint=fingerprint,
            seed=args.seed,
        )
        if errors:
            raise RuntimeError(
                "prepared 缓存与当前 split 不一致；为防止 LoRA/train 数据"
                "泄漏已阻断。如确需重建，使用 --force-prepare。差异："
                + "; ".join(errors[:8])
            )
        _copy_split_provenance(run_dir, selection)
        return cached
    if force_prepare:
        if (run_dir / "prepared").exists():
            shutil.rmtree(run_dir / "prepared")
        if mapping_path.exists():
            mapping_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    _copy_split_provenance(run_dir, selection)
    manifest = prepare_inputs(
        args.dataset.resolve(),
        run_dir,
        report["dataset_audit"],
        args.seed,
        evaluation_profile="formal_release",
        result_label="formal_release",
        allowed_group_ids=selection["selected_group_ids"],
        evaluation_split=selection["evaluation_split"],
        split_manifest=selection["split_manifest"],
    )
    errors = _prepared_cache_errors(
        run_dir,
        manifest,
        selection,
        fingerprint=fingerprint,
        seed=args.seed,
    )
    if errors:
        raise RuntimeError("新生成的 prepared 输入未通过 split 完整性校验：" + "; ".join(errors[:8]))
    return manifest


def _formal_config(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    config = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if config.get("evaluation_profile") != "formal_release":
        raise ValueError("正式评测配置的 evaluation_profile 必须为 formal_release")
    return config


def command_preflight(args: argparse.Namespace) -> int:
    report, run_dir = _preflight(args)
    print(json.dumps({
        "status": report["status"],
        "run_dir": str(run_dir),
        "blocking_errors": report["blocking_errors"],
        "review_reasons": report["review_reasons"],
    }, ensure_ascii=False, indent=2))
    return 2 if report["status"] == "BLOCKED" else 1 if report["status"] == "REVIEW_REQUIRED" else 0


def command_prepare(args: argparse.Namespace) -> int:
    report, run_dir = _load_or_preflight(args)
    if report["status"] == "BLOCKED":
        print("正式数据预检为 BLOCKED，未生成模型输入。", file=sys.stderr)
        return 2
    manifest = _ensure_prepared(args, report, run_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def command_check_config(args: argparse.Namespace) -> int:
    _formal_config(args.config)
    result = check_model_config(
        args.config.resolve(),
        set(args.models) if args.models else None,
        require_env=not args.skip_env_check,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


def _review_gate(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if report["status"] == "BLOCKED":
        raise RuntimeError("正式数据预检为 BLOCKED；禁止调用模型 API")
    if report["status"] == "REVIEW_REQUIRED" and not args.allow_review_required:
        raise RuntimeError(
            "正式数据预检为 REVIEW_REQUIRED；审阅 data_audit/formal_preflight.md 后，"
            "如确认继续，请显式添加 --allow-review-required"
        )


def command_run(args: argparse.Namespace) -> int:
    _formal_config(args.config)
    report, run_dir = _load_or_preflight(args)
    _review_gate(args, report)
    prepared_manifest = _ensure_prepared(args, report, run_dir)
    snapshot = run_dir / "config" / "models.formal.snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config.resolve(), snapshot)
    write_json(snapshot.parent / "formal_provenance.json", {
        "evaluation_profile": "formal_release",
        "dataset_fingerprint": report["dataset_fingerprint"],
        "policy_sha256": report["policy_sha256"],
        "model_config_source": str(args.config.resolve()),
        "model_config_sha256": _file_sha256(args.config.resolve()),
        "model_config_snapshot": str(snapshot.resolve()),
        "formal_eval_py_sha256": _file_sha256(Path(__file__).resolve()),
        "formal_eval_core_sha256": _file_sha256(HERE / "formal_eval_core.py"),
        "seed": args.seed,
        "requested_modes": sorted(set(args.modes)),
        "evaluation_split": prepared_manifest.get("selection", {}).get("evaluation_split"),
        "split_manifest": prepared_manifest.get("selection", {}).get("split_manifest"),
        "split_manifest_sha256": prepared_manifest.get("selection", {}).get("split_manifest_sha256"),
        "prepared_source_samples": prepared_manifest.get("source_samples"),
        "prepared_items": prepared_manifest.get("public_items"),
    })
    run_audit = dict(report["dataset_audit"])
    run_audit["known_limitations"] = report["review_reasons"]
    run_audit["shortcut_baselines"] = {
        "overall_risk": (
            "REVIEW_REQUIRED"
            if any(
                not task.get("gate", {}).get("passed", True)
                for task in report["shortcut_audit"]["baselines"]["count-only"]["tasks"].values()
            )
            else "LOW_OBSERVED_RISK"
        )
    }
    summary = run_models(
        run_dir,
        args.config.resolve(),
        run_audit,
        set(args.models) if args.models else None,
        set(args.modes),
        args.limit,
        args.retry_failed,
        args.limit_groups,
        evaluation_profile="formal_release",
        result_label="formal_release",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def command_import_predictions(args: argparse.Namespace) -> int:
    report, run_dir = _load_or_preflight(args)
    if report["status"] == "BLOCKED":
        raise RuntimeError("正式数据预检为 BLOCKED；拒绝导入结果")
    _ensure_prepared(args, report, run_dir)
    source = args.source.resolve()
    rows = read_jsonl(source)
    mapping = {
        str(row["eval_id"]): row
        for row in read_jsonl(run_dir / "private" / "evaluation_mapping.jsonl")
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        eval_id = str(row.get("eval_id", ""))
        if not eval_id or eval_id not in mapping:
            errors.append(f"未知或缺失 eval_id: {eval_id or '<missing>'}")
            continue
        if eval_id in seen:
            errors.append(f"重复 eval_id: {eval_id}")
            continue
        seen.add(eval_id)
        expected_mode = str(mapping[eval_id]["evaluation_mode"])
        supplied_mode = str(row.get("evaluation_mode") or expected_mode)
        if supplied_mode != expected_mode:
            errors.append(f"{eval_id} evaluation_mode 应为 {expected_mode}，实际为 {supplied_mode}")
            continue
        answer = row.get("parsed_answer") or row.get("answer")
        normalized.append({
            **row,
            "eval_id": eval_id,
            "model_id": args.model_id,
            "evaluation_mode": expected_mode,
            "parsed_answer": answer if answer in ANSWER_LABELS else None,
            "status": row.get("status") or ("OK" if answer in ANSWER_LABELS else "PARSE_ERROR"),
            "evaluation_profile": "formal_release",
            "result_label": "formal_release",
        })
    if errors:
        raise ValueError(f"导入文件有 {len(errors)} 个错误，例如：{errors[:5]}")
    target = run_dir / "predictions" / f"{safe_path_component(args.model_id)}.jsonl"
    if target.exists() and not args.replace:
        raise FileExistsError(f"目标已存在：{target}；如确需覆盖，请添加 --replace")
    write_jsonl(target, normalized)
    source_digest = _file_sha256(source)
    write_json(target.with_suffix(".import.json"), {
        "evaluation_profile": "formal_release",
        "evaluation_split": str(getattr(args, "eval_split", "test")),
        "model_id": args.model_id,
        "rows": len(normalized),
        "source": str(source),
        "source_sha256": source_digest,
        "normalized_prediction_file": str(target.resolve()),
        "normalized_prediction_sha256": _file_sha256(target),
    })
    print(json.dumps({
        "status": "PASS",
        "model_id": args.model_id,
        "rows": len(normalized),
        "prediction_file": str(target.resolve()),
        "source_sha256": source_digest,
    }, ensure_ascii=False, indent=2))
    return 0


def _score(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    report, run_dir = _load_or_preflight(args)
    if report["status"] == "BLOCKED":
        raise RuntimeError("正式数据预检为 BLOCKED；拒绝计分")
    _ensure_prepared(args, report, run_dir)
    cluster_map, source, cluster_stats = _clusters(
        args.dataset.resolve(),
        run_dir,
        args.cluster_manifest,
    )
    mapping_groups = {
        str(row["relation_group_id"])
        for row in read_jsonl(run_dir / "private" / "evaluation_mapping.jsonl")
    }
    missing_groups = sorted(mapping_groups - set(cluster_map))
    if missing_groups:
        raise ValueError(f"评分映射缺少 scene cluster：{missing_groups[:3]}")
    cluster_map = {
        group_id: cluster_map[group_id]
        for group_id in sorted(mapping_groups)
    }
    filtered_stats = {
        **cluster_stats,
        "method": f"{cluster_stats.get('method', 'unknown')}_filtered_to_evaluation_mapping",
        "groups": len(cluster_map),
        "clusters": len(set(cluster_map.values())),
        "evaluation_split": str(getattr(args, "eval_split", "test")),
    }
    _write_cluster_map(run_dir, cluster_map, source, filtered_stats)
    metrics = score_run(
        run_dir,
        cluster_by_group=cluster_map,
        cluster_source=source,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    return metrics, run_dir


def command_score(args: argparse.Namespace) -> int:
    metrics, run_dir = _score(args)
    print(json.dumps({
        "status": metrics["status"],
        "models": len(metrics["models"]),
        "run_dir": str(run_dir),
        "cluster_source": metrics["confidence_interval"]["cluster_source"],
    }, ensure_ascii=False, indent=2))
    return 0 if metrics["status"] == "PASS" else 1


def command_shortcuts(args: argparse.Namespace) -> int:
    _, run_dir, _ = _context(args)
    result = shortcut_audit(args.dataset.resolve(), args.policy.resolve())
    paths = write_shortcut_outputs(result, run_dir / "analysis")
    failed = [
        task_name
        for task_name, task in result["baselines"]["count-only"]["tasks"].items()
        if not task["gate"]["passed"]
    ]
    print(json.dumps({"status": "REVIEW_REQUIRED" if failed else "PASS", "failed_gates": failed, **paths}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def command_table(args: argparse.Namespace) -> int:
    _formal_config(args.config)
    metrics, run_dir = _score(args)
    report, _ = _load_or_preflight(args)
    shortcut = shortcut_audit(args.dataset.resolve(), args.policy.resolve())
    outputs = write_main_tables(
        metrics,
        run_dir / "tables",
        args.config.resolve(),
        release_status=report["status"],
        shortcut_status=shortcut["status"],
    )
    shortcut_outputs = write_shortcut_outputs(shortcut, run_dir / "analysis")
    print(json.dumps({
        "status": metrics["status"],
        "models_scored": len(metrics["models"]),
        "tables": outputs,
        "shortcut_audit": shortcut_outputs,
    }, ensure_ascii=False, indent=2))
    return 0 if metrics["status"] == "PASS" else 1


def _metadata_db(args: argparse.Namespace, dataset: Path) -> Path | None:
    if args.metadata_db is not None:
        return args.metadata_db.resolve()
    candidate = dataset.parent / "frameevidence.sqlite3"
    return candidate.resolve() if candidate.is_file() else None


def command_split(args: argparse.Namespace) -> int:
    dataset, run_dir, _ = _context(args)
    groups = load_group_records(
        dataset,
        quota_path=args.quota.resolve() if args.quota else None,
        metadata_db=_metadata_db(args, dataset),
    )
    if args.cluster_manifest is not None:
        cluster_path = args.cluster_manifest.resolve()
        cluster_by_group = load_cluster_manifest(cluster_path)
        cluster_source = f"external_scene_cluster_id:{cluster_path.name}"
        cluster_warning = None
    else:
        if not args.allow_asset_cluster_fallback:
            raise ValueError(
                "缺少真实 scene_cluster_id 台账。若只生成明确标记为 provisional 的草案，"
                "请显式加入 --allow-asset-cluster-fallback"
            )
        cluster_by_group, cluster_stats = build_asset_cluster_map(dataset)
        cluster_source = "exact_image_sha256_connected_components_fallback"
        cluster_warning = cluster_stats.get("warning")
    split_overrides_path = (
        args.split_overrides.resolve() if args.split_overrides is not None else None
    )
    split_overrides = load_cluster_split_overrides(split_overrides_path)
    assignment, diagnostics = generate_stratified_group_split(
        groups,
        cluster_by_group,
        seed=args.seed,
        attempts=args.search_attempts,
        group_targets=DEFAULT_GROUP_TARGETS,
        task_targets=DEFAULT_TASK_TARGETS,
        cluster_overrides=split_overrides,
    )
    outputs = write_split_outputs(
        groups,
        assignment,
        cluster_by_group,
        diagnostics,
        run_dir / "manifests",
        cluster_source=cluster_source,
        cluster_warning=cluster_warning,
        dataset_path=dataset,
        policy_path=args.policy.resolve(),
        split_overrides_path=split_overrides_path,
    )
    status = (
        "PROVISIONAL_NOT_FOR_PAPER"
        if cluster_source == "exact_image_sha256_connected_components_fallback"
        else "FROZEN_CANDIDATE"
    )
    print(json.dumps({
        "status": status,
        "groups": diagnostics["group_counts"],
        "task_groups": diagnostics["task_group_counts"],
        "clusters": diagnostics["cluster_counts"],
        "balance_score": diagnostics["balance_score"],
        "cluster_overrides": diagnostics["cluster_overrides"],
        "outputs": outputs,
    }, ensure_ascii=False, indent=2))
    return 1 if status == "PROVISIONAL_NOT_FOR_PAPER" else 0


def command_table1(args: argparse.Namespace) -> int:
    dataset, run_dir, _ = _context(args)
    manifest_path = (
        args.split_manifest.resolve()
        if args.split_manifest is not None
        else run_dir / "manifests" / "split_manifest.csv"
    )
    groups = load_group_records(
        dataset,
        quota_path=args.quota.resolve() if args.quota else None,
        metadata_db=_metadata_db(args, dataset),
    )
    assignment, cluster_by_group, cluster_source = load_split_manifest(manifest_path)
    split_status = None
    split_summary_path = manifest_path.with_name("split_summary.json")
    if split_summary_path.is_file():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        if str(split_summary.get("manifest_sha256", "")) != _file_sha256(manifest_path):
            raise ValueError("split_summary manifest SHA-256 does not match split_manifest.csv")
        split_status = str(split_summary.get("status", "UNKNOWN"))
    validate_split_assignment(
        groups,
        assignment,
        cluster_by_group,
        group_targets=DEFAULT_GROUP_TARGETS,
        task_targets=DEFAULT_TASK_TARGETS,
    )
    statistics = build_dataset_statistics(
        groups,
        assignment,
        cluster_by_group,
        cluster_source=cluster_source,
        split_status=split_status,
    )
    outputs = write_dataset_stats_tables(statistics, run_dir / "tables")
    print(json.dumps({
        "status": statistics["table_status"],
        "split_manifest": str(manifest_path),
        "group_split_counts": statistics["group_split_counts"],
        "tables": outputs,
    }, ensure_ascii=False, indent=2))
    return 1 if statistics["table_status"] == "PROVISIONAL_NOT_FOR_PAPER" else 0


def command_all(args: argparse.Namespace) -> int:
    report, run_dir = _preflight(args)
    _review_gate(args, report)
    _ensure_prepared(args, report, run_dir)
    run_status = command_run(args)
    command_table(args)
    return run_status


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="兼容旧数据审计的诊断置换次数；正式捷径门槛另由 deterministic audit 计算",
    )
    parser.add_argument(
        "--cluster-manifest",
        type=Path,
        default=None,
        help="CSV/JSONL，字段 relation_group_id, scene_cluster_id；缺省时用图片 SHA 连通分量保守回退",
    )
    parser.add_argument("--refresh-preflight", action="store_true")


def _scoring(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)


def _evaluation_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "train/dev/test 题组划分 CSV；缺省时从无 split 后缀的 "
            "formal_<fingerprint>/manifests/split_manifest.csv 读取"
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "dev", "test"],
        default="test",
        help="当前正式评测仅准备和计分的 split（缺省：test）",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="显式废弃当前 prepared 缓存并按 split manifest 重建",
    )


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["all-images", "text-only", "single-image"],
        default=["all-images"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--allow-review-required",
        action="store_true",
        help="确认已审阅预检警告后，允许付费/外部模型调用",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FrameEvidence compact-320 正式评测与主表生成器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, handler in (
        ("preflight", command_preflight),
        ("prepare", command_prepare),
        ("shortcuts", command_shortcuts),
    ):
        child = subparsers.add_parser(name)
        _common(child)
        if name == "prepare":
            _evaluation_split_args(child)
        child.set_defaults(handler=handler)

    for name, handler in (("score", command_score), ("table", command_table)):
        child = subparsers.add_parser(name)
        _common(child)
        _evaluation_split_args(child)
        _scoring(child)
        if name == "table":
            child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        child.set_defaults(handler=handler)

    for name, handler in (("run", command_run), ("all", command_all)):
        child = subparsers.add_parser(name)
        _common(child)
        _evaluation_split_args(child)
        _model_args(child)
        if name == "all":
            _scoring(child)
        child.set_defaults(handler=handler)

    split = subparsers.add_parser("split")
    _common(split)
    split.add_argument("--quota", type=Path, default=DEFAULT_QUOTA)
    split.add_argument(
        "--metadata-db",
        type=Path,
        default=None,
        help="生产 SQLite；缺省时自动查找 dataset 上一级的 frameevidence.sqlite3",
    )
    split.add_argument("--search-attempts", type=int, default=400)
    split.add_argument(
        "--split-overrides",
        type=Path,
        default=DEFAULT_SPLIT_OVERRIDES,
        help="整簇 split override CSV；字段 scene_cluster_id, split, reason",
    )
    split.add_argument(
        "--allow-asset-cluster-fallback",
        action="store_true",
        help="显式允许用精确图片 SHA 连通分量生成仅供审阅的 provisional split",
    )
    split.set_defaults(handler=command_split)

    table1 = subparsers.add_parser("table1")
    _common(table1)
    table1.add_argument("--quota", type=Path, default=DEFAULT_QUOTA)
    table1.add_argument("--metadata-db", type=Path, default=None)
    table1.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="缺省时读取当前数据指纹目录下 manifests/split_manifest.csv",
    )
    table1.set_defaults(handler=command_table1)

    check = subparsers.add_parser("check-config")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--models", nargs="+", default=None)
    check.add_argument("--skip-env-check", action="store_true")
    check.set_defaults(handler=command_check_config)

    importer = subparsers.add_parser("import-predictions")
    _common(importer)
    _evaluation_split_args(importer)
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument("--model-id", required=True)
    importer.add_argument("--replace", action="store_true")
    importer.set_defaults(handler=command_import_predictions)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001 - CLI returns concise, machine-readable failures
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
