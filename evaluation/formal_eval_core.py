from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


ANSWER_LABELS = ("A", "B", "C", "D")
FQ1_FRAMES = ("camera", "object_intrinsic", "world")
FQ2_CONDITIONS = (
    "one_valid_mss",
    "remove_required_view",
    "hard_distractor_added",
)
MODEL_MODES = ("all-images", "text-only", "single-image")
MAIN_COLUMNS = (
    "item_accuracy",
    "valid_response_rate",
    "camera",
    "object_intrinsic",
    "world",
    "FQ1_GEM",
    "one_valid_mss",
    "remove_required_view",
    "hard_distractor_added",
    "NTS",
    "DIS",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} 不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("无法对空数组计算分位数")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_interval(
    outcomes: dict[str, int],
    cluster_by_group: dict[str, str],
    repetitions: int = 10_000,
    seed: int = 20260927,
) -> list[float] | None:
    if not outcomes:
        return None
    missing = sorted(set(outcomes) - set(cluster_by_group))
    if missing:
        raise ValueError(f"聚类清单缺少 {len(missing)} 个题组，例如 {missing[:3]}")
    values_by_cluster: dict[str, list[int]] = defaultdict(list)
    for group_id, value in outcomes.items():
        values_by_cluster[str(cluster_by_group[group_id])].append(int(value))
    clusters = sorted(values_by_cluster)
    if not clusters:
        return None
    if repetitions <= 0:
        raise ValueError("cluster bootstrap repetitions 必须大于 0")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(clusters) for _ in clusters]
        successes = sum(sum(values_by_cluster[cluster]) for cluster in sampled)
        total = sum(len(values_by_cluster[cluster]) for cluster in sampled)
        estimates.append(successes / total)
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _metric_record(
    outcomes: dict[str, int],
    cluster_by_group: dict[str, str] | None,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    numerator = sum(outcomes.values())
    denominator = len(outcomes)
    if cluster_by_group is not None:
        if not cluster_by_group:
            raise ValueError("正式聚类映射不能为空")
        interval = cluster_bootstrap_interval(
            outcomes,
            cluster_by_group,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        method = "cluster_bootstrap_percentile"
        cluster_count = len({cluster_by_group[group] for group in outcomes})
    else:
        interval = wilson_interval(numerator, denominator)
        method = "wilson"
        cluster_count = None
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": _rate(numerator, denominator),
        "ci95": interval,
        "ci_method": method,
        "clusters": cluster_count,
    }


def classification_metrics(
    y_true: list[str],
    y_pred: list[str | None],
    labels: Iterable[str],
) -> dict[str, Any]:
    ordered_labels = list(labels)
    unknown_gold = sorted(set(y_true) - set(ordered_labels))
    if unknown_gold:
        raise ValueError(f"gold label 不在允许集合中：{unknown_gold}")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")
    prediction_labels = ordered_labels + ["INVALID"]
    confusion = {
        actual: {predicted: 0 for predicted in prediction_labels}
        for actual in ordered_labels
    }
    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual not in confusion:
            continue
        normalized = predicted if predicted in ordered_labels else "INVALID"
        confusion[actual][normalized] += 1
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    per_class: dict[str, Any] = {}
    for label in ordered_labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in ordered_labels if actual != label)
        fn = sum(
            count
            for predicted, count in confusion[label].items()
            if predicted != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred, strict=True))
    return {
        "n": len(y_true),
        "correct": correct,
        "accuracy": _rate(correct, len(y_true)),
        "macro_precision": sum(precision_values) / len(precision_values) if precision_values else None,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _parse_answer(prediction: dict[str, Any]) -> str | None:
    answer = prediction.get("parsed_answer") or prediction.get("answer")
    if answer in ANSWER_LABELS:
        return str(answer)
    raw = str(prediction.get("raw_response", "")).strip().upper()
    return raw if raw in ANSWER_LABELS else None


def score_mode(
    mapping: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    mode: str,
    cluster_by_group: dict[str, str] | None = None,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260927,
) -> dict[str, Any]:
    expected = [row for row in mapping if row.get("evaluation_mode") == mode]
    scored: list[dict[str, Any]] = []
    for row in expected:
        prediction = predictions.get(str(row["eval_id"]), {})
        answer = _parse_answer(prediction)
        scored.append({
            **row,
            "predicted_answer": answer,
            "correct": answer == row.get("gold_answer"),
            "prediction_status": prediction.get("status", "MISSING"),
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[str(row["relation_group_id"])].append(row)
    fq1 = {
        group_id: rows
        for group_id, rows in grouped.items()
        if rows and rows[0].get("question_type") == "FQ1"
    }
    fq2 = {
        group_id: rows
        for group_id, rows in grouped.items()
        if rows and rows[0].get("question_type") == "FQ2"
    }

    fq1_outcomes = {
        group_id: int(len(rows) == 3 and all(bool(row["correct"]) for row in rows))
        for group_id, rows in fq1.items()
    }
    nts_outcomes: dict[str, int] = {}
    dis_outcomes: dict[str, int] = {}
    for group_id, rows in fq2.items():
        by_condition = {str(row.get("view_condition")): row for row in rows}
        mss = by_condition.get("one_valid_mss")
        deleted = by_condition.get("remove_required_view")
        added = by_condition.get("hard_distractor_added")
        nts_outcomes[group_id] = int(bool(mss and deleted and mss["correct"] and deleted["correct"]))
        dis_outcomes[group_id] = int(bool(mss and added and mss["correct"] and added["correct"]))

    by_frame: dict[str, Any] = {}
    for frame in FQ1_FRAMES:
        rows = [
            row for row in scored
            if row.get("question_type") == "FQ1" and row.get("reference_frame") == frame
        ]
        correct = sum(bool(row["correct"]) for row in rows)
        by_frame[frame] = {"n": len(rows), "numerator": correct, "accuracy": _rate(correct, len(rows))}
    by_condition: dict[str, Any] = {}
    for condition in FQ2_CONDITIONS:
        rows = [
            row for row in scored
            if row.get("question_type") == "FQ2" and row.get("view_condition") == condition
        ]
        correct = sum(bool(row["correct"]) for row in rows)
        by_condition[condition] = {"n": len(rows), "numerator": correct, "accuracy": _rate(correct, len(rows))}

    correct = sum(bool(row["correct"]) for row in scored)
    valid = sum(row["predicted_answer"] in ANSWER_LABELS for row in scored)
    classification = classification_metrics(
        [str(row["gold_answer"]) for row in scored],
        [row["predicted_answer"] for row in scored],
        ANSWER_LABELS,
    )
    main_metrics = {
        "FQ1_GEM": _metric_record(
            fq1_outcomes,
            cluster_by_group,
            bootstrap_repetitions,
            seed + 101,
        ),
        "NTS": _metric_record(
            nts_outcomes,
            cluster_by_group,
            bootstrap_repetitions,
            seed + 211,
        ),
        "DIS": _metric_record(
            dis_outcomes,
            cluster_by_group,
            bootstrap_repetitions,
            seed + 307,
        ),
    }
    received = sum(str(row["eval_id"]) in predictions for row in expected)
    return {
        "evaluation_mode": mode,
        "status": "PASS" if received == len(expected) else "INCOMPLETE",
        "coverage": {
            "expected": len(expected),
            "received": received,
            "valid_answers": valid,
            "invalid_or_missing": len(expected) - valid,
        },
        "item_accuracy": {
            "numerator": correct,
            "denominator": len(scored),
            "value": _rate(correct, len(scored)),
        },
        "valid_response_rate": {
            "numerator": valid,
            "denominator": len(scored),
            "value": _rate(valid, len(scored)),
        },
        "classification": classification,
        "main_metrics": main_metrics,
        "fq1_by_reference_frame": by_frame,
        "fq2_by_view_condition": by_condition,
        "group_outcomes": {
            "FQ1_GEM": fq1_outcomes,
            "NTS": nts_outcomes,
            "DIS": dis_outcomes,
        },
        "analysis_unit": "relation_group_id",
        "missing_eval_ids": [
            str(row["eval_id"])
            for row in expected
            if str(row["eval_id"]) not in predictions
        ][:100],
    }


def score_run(
    run_dir: Path,
    cluster_by_group: dict[str, str] | None = None,
    cluster_source: str = "none",
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260927,
    require_clusters: bool = True,
) -> dict[str, Any]:
    mapping_path = run_dir / "private" / "evaluation_mapping.jsonl"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"缺少私有评分映射：{mapping_path}")
    mapping = read_jsonl(mapping_path)
    mapping_ids = [str(row.get("eval_id", "")) for row in mapping]
    mapping_id_counts = Counter(mapping_ids)
    duplicate_mapping_ids = sorted(value for value, count in mapping_id_counts.items() if value and count > 1)
    if not all(mapping_ids) or duplicate_mapping_ids:
        raise ValueError(f"评分映射含缺失/重复 eval_id：{duplicate_mapping_ids[:5]}")
    expected_groups = {str(row["relation_group_id"]) for row in mapping}
    if require_clusters and cluster_by_group is None:
        raise ValueError("正式评分必须提供 scene_cluster_id 或显式的资产聚类回退映射")
    if cluster_by_group is not None:
        missing_clusters = sorted(expected_groups - set(cluster_by_group))
        extra_clusters = sorted(set(cluster_by_group) - expected_groups)
        if missing_clusters or extra_clusters:
            raise ValueError(
                "聚类映射与正式题组集合不一致："
                f"missing={missing_clusters[:3]} extra={extra_clusters[:3]}"
            )
        if len(set(cluster_by_group.values())) < 2:
            raise ValueError("cluster bootstrap 至少需要 2 个不同聚类")
        if not cluster_source or cluster_source == "none":
            raise ValueError("提供聚类映射时必须记录非空 cluster_source")
    prediction_dir = run_dir / "predictions"
    prediction_files = sorted(prediction_dir.glob("*.jsonl")) if prediction_dir.is_dir() else []
    models: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    mapping_by_id = {str(row["eval_id"]): row for row in mapping}
    for path in prediction_files:
        rows = read_jsonl(path)
        if not rows:
            continue
        ids = {str(row.get("model_id") or path.stem) for row in rows}
        if len(ids) != 1:
            raise ValueError(f"{path} 混有多个 model_id：{sorted(ids)}")
        model_id = next(iter(ids))
        if model_id in seen_model_ids:
            raise ValueError(f"model_id {model_id} 出现在多个预测文件中")
        seen_model_ids.add(model_id)
        prediction_ids = [str(row.get("eval_id", "")) for row in rows]
        prediction_id_counts = Counter(prediction_ids)
        duplicate_ids = sorted(value for value, count in prediction_id_counts.items() if value and count > 1)
        unknown_ids = sorted(set(prediction_ids) - set(mapping_by_id))
        if not all(prediction_ids) or duplicate_ids or unknown_ids:
            raise ValueError(
                f"{path} 的 eval_id 非法：duplicates={duplicate_ids[:3]} unknown={unknown_ids[:3]}"
            )
        for row in rows:
            eval_id = str(row["eval_id"])
            expected_mode = str(mapping_by_id[eval_id]["evaluation_mode"])
            if str(row.get("evaluation_mode")) != expected_mode:
                raise ValueError(
                    f"{path}: {eval_id} evaluation_mode 应为 {expected_mode}，"
                    f"实际为 {row.get('evaluation_mode')}"
                )
        predictions = {str(row["eval_id"]): row for row in rows}
        available_modes = {
            str(row.get("evaluation_mode"))
            for row in rows
            if row.get("evaluation_mode") in MODEL_MODES
        }
        results = {
            mode: score_mode(
                mapping,
                predictions,
                mode,
                cluster_by_group=cluster_by_group,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            for mode in MODEL_MODES
            if mode in available_modes
        }
        models.append({
            "model_id": model_id,
            "prediction_file": str(path.resolve()),
            "prediction_sha256": sha256_file(path),
            "results": results,
        })
    all_images = [model.get("results", {}).get("all-images") for model in models]
    status = (
        "NOT_RUN"
        if not models
        else "PASS"
        if all(result and result["status"] == "PASS" for result in all_images)
        else "INCOMPLETE"
    )
    result = {
        "status": status,
        "evaluation_profile": "formal_release",
        "result_label": "formal_release",
        "models": models,
        "analysis_unit": "relation_group_id",
        "confidence_interval": {
            "level": 0.95,
            "cluster_source": cluster_source,
            "cluster_count": len(set(cluster_by_group.values())) if cluster_by_group else None,
            "bootstrap_repetitions": bootstrap_repetitions if cluster_by_group else None,
            "fallback": (
                None
                if cluster_by_group is not None
                else "Wilson interval (explicitly allowed without a cluster mapping)"
            ),
        },
        "mapping_sha256": sha256_file(mapping_path),
    }
    write_json(run_dir / "analysis" / "formal_metrics.json", result)
    write_metrics_long_csv(result, run_dir / "analysis" / "formal_metrics_long.csv")
    return result


def write_metrics_long_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "model_id",
        "evaluation_mode",
        "metric",
        "numerator",
        "denominator",
        "value",
        "ci95_low",
        "ci95_high",
        "ci_method",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model in metrics.get("models", []):
            for mode, result in model.get("results", {}).items():
                base = {
                    "model_id": model["model_id"],
                    "evaluation_mode": mode,
                }
                for name in ("item_accuracy", "valid_response_rate"):
                    value = result[name]
                    writer.writerow({**base, "metric": name, **value})
                classification = result["classification"]
                for name in ("macro_precision", "macro_recall", "macro_f1"):
                    writer.writerow({
                        **base,
                        "metric": name,
                        "denominator": classification["n"],
                        "value": classification[name],
                    })
                for frame, value in result["fq1_by_reference_frame"].items():
                    writer.writerow({
                        **base,
                        "metric": f"fq1_{frame}_accuracy",
                        "numerator": value["numerator"],
                        "denominator": value["n"],
                        "value": value["accuracy"],
                    })
                for condition, value in result["fq2_by_view_condition"].items():
                    writer.writerow({
                        **base,
                        "metric": f"fq2_{condition}_accuracy",
                        "numerator": value["numerator"],
                        "denominator": value["n"],
                        "value": value["accuracy"],
                    })
                for name, value in result["main_metrics"].items():
                    interval = value.get("ci95") or [None, None]
                    writer.writerow({
                        **base,
                        "metric": name,
                        "numerator": value["numerator"],
                        "denominator": value["denominator"],
                        "value": value["value"],
                        "ci95_low": interval[0],
                        "ci95_high": interval[1],
                        "ci_method": value.get("ci_method"),
                    })


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if left < right:
            self.parent[right] = left
        else:
            self.parent[left] = right


def build_asset_cluster_map(dataset_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    dataset_root = dataset_dir.resolve()
    records = read_jsonl(dataset_dir / "annotations.jsonl")
    groups = sorted({str(row["relation_group_id"]) for row in records})
    union = _UnionFind(groups)
    groups_by_digest: dict[str, set[str]] = defaultdict(set)
    path_digest: dict[str, str] = {}
    for row in records:
        group_id = str(row["relation_group_id"])
        for field in ("images", "mss_images", "removed_images"):
            for relative in row.get(field, []) or []:
                normalized = str(relative).replace("\\", "/")
                if normalized not in path_digest:
                    candidate = (dataset_root / normalized).resolve()
                    try:
                        candidate.relative_to(dataset_root)
                    except ValueError as exc:
                        raise ValueError(f"图片路径越出数据目录：{normalized}") from exc
                    if not candidate.is_file():
                        raise FileNotFoundError(f"缺少图片：{candidate}")
                    path_digest[normalized] = sha256_file(candidate)
                groups_by_digest[path_digest[normalized]].add(group_id)
    for digest_groups in groups_by_digest.values():
        ordered = sorted(digest_groups)
        for group_id in ordered[1:]:
            union.union(ordered[0], group_id)
    members: dict[str, list[str]] = defaultdict(list)
    for group_id in groups:
        members[union.find(group_id)].append(group_id)
    ordered_components = sorted(members.values(), key=lambda values: values[0])
    cluster_by_group = {
        group_id: f"asset_component_{index:04d}"
        for index, component in enumerate(ordered_components, 1)
        for group_id in component
    }
    shared_digests = {
        digest: sorted(value)
        for digest, value in groups_by_digest.items()
        if len(value) > 1
    }
    stats = {
        "method": "exact_image_sha256_connected_components",
        "groups": len(groups),
        "physical_paths": len(path_digest),
        "unique_sha256": len(groups_by_digest),
        "shared_sha256": len(shared_digests),
        "clusters": len(ordered_components),
        "singleton_clusters": sum(len(component) == 1 for component in ordered_components),
        "largest_cluster_groups": max((len(component) for component in ordered_components), default=0),
        "warning": (
            "Exact-asset dependence lower-bound / partial fallback only; it cannot detect the same "
            "scene after viewpoint, crop, resize, or re-encoding changes and does not replace the "
            "restricted scene_cluster_id ledger."
        ),
    }
    return cluster_by_group, stats


def load_cluster_manifest(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    mapping: dict[str, str] = {}
    for row in rows:
        group_id = str(row.get("relation_group_id", "")).strip()
        cluster_id = str(row.get("scene_cluster_id", "")).strip()
        if not group_id or not cluster_id:
            raise ValueError("聚类清单每行必须包含 relation_group_id 和 scene_cluster_id")
        if group_id in mapping and mapping[group_id] != cluster_id:
            raise ValueError(f"题组 {group_id} 对应多个 scene_cluster_id")
        mapping[group_id] = cluster_id
    return mapping


def _best_empirical_prediction(
    features: list[str],
    targets: list[str],
) -> tuple[list[str], dict[str, str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for feature, target in zip(features, targets, strict=True):
        counts[feature][target] += 1
    rules = {
        feature: sorted(counter, key=lambda label: (-counter[label], label))[0]
        for feature, counter in sorted(counts.items())
    }
    return [rules[feature] for feature in features], rules


def _insufficient_position(record: dict[str, Any]) -> str:
    matches = [
        str(label)
        for label, text in dict(record.get("options", {})).items()
        if str(text).strip() == "信息不足"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{record.get('sample_id')} 必须恰有一个‘信息不足’选项，实际为 {matches}"
        )
    return matches[0]


def shortcut_audit(dataset_dir: Path, policy_path: Path) -> dict[str, Any]:
    contract = validate_release_contract(dataset_dir, policy_path)
    if contract["status"] == "BLOCKED":
        raise ValueError(
            "正式发布契约未通过，不能计算捷径门槛："
            + "; ".join(contract["errors"][:5])
        )
    records = [
        row for row in read_jsonl(dataset_dir / "annotations.jsonl")
        if row.get("question_type") == "FQ2"
    ]
    records.sort(key=lambda row: (str(row["relation_group_id"]), str(row["view_condition"])))
    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    tier = policy["tiers"][policy["active_tier"]]
    layout = tier["quota"]["answer_layout"]
    thresholds = {
        "view_condition": int(layout["fq2_count_only_condition_max_correct"]),
        "gold_answer": int(layout["fq2_count_only_answer_max_correct"]),
    }
    baselines: dict[str, Any] = {}
    for baseline_name, feature_builder in (
        ("count-only", lambda row: str(len(row.get("images", [])))),
        (
            "count+option-schema",
            lambda row: f"{len(row.get('images', []))}|{_insufficient_position(row)}",
        ),
    ):
        features = [feature_builder(row) for row in records]
        tasks: dict[str, Any] = {}
        for task_name, target_builder, labels in (
            ("view_condition", lambda row: str(row["view_condition"]), FQ2_CONDITIONS),
            ("gold_answer", lambda row: str(row["answer"]), ANSWER_LABELS),
        ):
            targets = [target_builder(row) for row in records]
            predicted, rules = _best_empirical_prediction(features, targets)
            metrics = classification_metrics(targets, predicted, labels)
            deletion = metrics["per_class"].get("remove_required_view", {})
            task = {
                **metrics,
                "decision_rules": rules,
                "deletion_recall": deletion.get("recall") if task_name == "view_condition" else None,
            }
            if baseline_name == "count-only":
                maximum = thresholds[task_name]
                task["gate"] = {
                    "expected_n": 480,
                    "maximum_correct": maximum,
                    "observed_correct": metrics["correct"],
                    "passed": metrics["n"] == 480 and metrics["correct"] <= maximum,
                }
            elif task_name == "gold_answer":
                task["diagnostic_risk"] = (
                    "HIGH_OBSERVED_LEAKAGE"
                    if (metrics["accuracy"] or 0.0) >= 0.35
                    else "REVIEW_REQUIRED" if (metrics["accuracy"] or 0.0) > 0.25 else "LOW_OBSERVED_RISK"
                )
            tasks[task_name] = task
        baselines[baseline_name] = {"feature": baseline_name, "tasks": tasks}
    return {
        "evaluation_profile": "formal_release",
        "scope": "FQ2 only",
        "samples": len(records),
        "groups": len({str(row["relation_group_id"]) for row in records}),
        "policy": str(policy_path.resolve()),
        "policy_version": policy.get("version"),
        "active_tier": policy.get("active_tier"),
        "baselines": baselines,
        "status": (
            "PASS"
            if all(
                task["gate"]["passed"]
                for task in baselines["count-only"]["tasks"].values()
            )
            else "BLOCKED"
        ),
        "interpretation": (
            "Best empirical mappings are diagnostic upper bounds on the frozen release. "
            "They are not VLM scores and do not receive pixels, object names, or questions."
        ),
    }


def write_shortcut_outputs(audit: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "shortcut_audit.json"
    csv_path = output_dir / "shortcut_audit.csv"
    md_path = output_dir / "shortcut_audit.md"
    write_json(json_path, audit)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = (
            "baseline",
            "task",
            "correct",
            "n",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "deletion_recall",
            "gate_maximum_correct",
            "gate_passed",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for baseline, value in audit["baselines"].items():
            for task_name, metrics in value["tasks"].items():
                gate = metrics.get("gate") or {}
                writer.writerow({
                    "baseline": baseline,
                    "task": task_name,
                    "correct": metrics["correct"],
                    "n": metrics["n"],
                    "accuracy": metrics["accuracy"],
                    "macro_precision": metrics["macro_precision"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "deletion_recall": metrics.get("deletion_recall"),
                    "gate_maximum_correct": gate.get("maximum_correct"),
                    "gate_passed": gate.get("passed"),
                })
    lines = [
        "# Formal-release shortcut audit",
        "",
        f"FQ2 scope: {audit['groups']} groups / {audit['samples']} items.",
        "",
        "| Baseline | Target | Correct | Accuracy | Macro-F1 | Delete recall | Frozen gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for baseline, value in audit["baselines"].items():
        for task_name, metrics in value["tasks"].items():
            gate = metrics.get("gate")
            gate_text = "N/A" if not gate else (
                f"{gate['observed_correct']}/{gate['maximum_correct']} "
                f"({'PASS' if gate['passed'] else 'FAIL'})"
            )
            deletion = metrics.get("deletion_recall")
            lines.append(
                f"| {baseline} | {task_name} | {metrics['correct']}/{metrics['n']} | "
                f"{_pct(metrics['accuracy'])} | {_pct(metrics['macro_f1'])} | "
                f"{_pct(deletion)} | {gate_text} |"
            )
    lines.extend(["", audit["interpretation"], ""])
    option_schema_gold = audit["baselines"]["count+option-schema"]["tasks"]["gold_answer"]
    if option_schema_gold.get("diagnostic_risk") == "HIGH_OBSERVED_LEAKAGE":
        lines.extend([
            "**Review required:** count+option-schema predicts the gold option far above the 25% "
            "four-way reference. This diagnostic has no frozen pass/fail threshold, but it must be "
            "reported as an observed option-layout shortcut.",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
    }


def validate_release_contract(dataset_dir: Path, policy_path: Path) -> dict[str, Any]:
    records = read_jsonl(dataset_dir / "annotations.jsonl")
    policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
    active_tier = str(policy["active_tier"])
    tier = policy["tiers"][active_tier]
    errors: list[str] = []
    warnings: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get("relation_group_id", ""))].append(row)

    # The formal contract is intentionally redundant with the production
    # validator: this standalone scorer must never turn a malformed mapping
    # into a publishable table merely because another preflight was skipped.
    frame_root = policy_path.parent.parent
    schema_candidates = sorted(
        (path for path in frame_root.rglob("data_schema.json") if path.is_file()),
        key=lambda path: (len(path.relative_to(frame_root).parts), path.as_posix()),
    )
    schema_path = schema_candidates[0] if schema_candidates else frame_root / "data_schema.json"
    if not schema_path.is_file():
        errors.append(f"缺少权威 17 字段 Schema：{schema_path}")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
        validator = Draft202012Validator(schema)
        expected_fields = set(schema.get("properties", {}))
        for row in records:
            sample_id = str(row.get("sample_id", "<missing>"))
            if set(row) != expected_fields:
                errors.append(f"{sample_id} 字段集合不等于权威 17 字段")
            for issue in validator.iter_errors(row):
                location = ".".join(str(part) for part in issue.path) or "<root>"
                errors.append(f"{sample_id} Schema {location}: {issue.message}")

    expected_samples = int(tier["questions_total"])
    expected_groups = int(tier["groups_total"])
    expected_per_task = int(tier["groups_per_task"])
    if len(records) != expected_samples:
        errors.append(f"题目数应为 {expected_samples}，实际为 {len(records)}")
    if len(groups) != expected_groups:
        errors.append(f"题组数应为 {expected_groups}，实际为 {len(groups)}")
    bad_sizes = {group: len(rows) for group, rows in groups.items() if len(rows) != 3}
    if bad_sizes:
        errors.append(f"存在 {len(bad_sizes)} 个非三题组，例如 {list(bad_sizes.items())[:3]}")
    statuses = Counter(str(row.get("quality_status")) for row in records)
    if statuses != {"verified": len(records)}:
        errors.append(f"正式集必须全为 verified，实际分布为 {dict(statuses)}")
    group_types: Counter[str] = Counter()
    for group_id, rows in groups.items():
        types = {str(row.get("question_type")) for row in rows}
        if len(types) != 1:
            errors.append(f"{group_id} 混有多个 question_type：{sorted(types)}")
            continue
        question_type = next(iter(types))
        group_types[question_type] += 1
        if question_type == "FQ1":
            frames = Counter(str(row.get("reference_frame")) for row in rows)
            if frames != Counter(FQ1_FRAMES):
                errors.append(f"{group_id} FQ1 参考系不完整：{dict(frames)}")
            if len({str(row.get("answer")) for row in rows}) < 2:
                errors.append(f"{group_id} FQ1 三参考系没有产生答案切换")
            baseline = rows[0]
            for row in rows:
                if row.get("view_condition") != "all_views":
                    errors.append(f"{group_id} FQ1 view_condition 必须为 all_views")
                if list(row.get("mss_images", []) or []) != list(row.get("images", []) or []):
                    errors.append(f"{group_id} FQ1 必须满足 mss_images == images")
                if row.get("removed_images"):
                    errors.append(f"{group_id} FQ1 removed_images 必须为空")
                for field in (
                    "images",
                    "object_instances",
                    "target_instance_id",
                    "relatum_instance_id",
                    "bridge_instance_ids",
                    "options",
                ):
                    if row.get(field) != baseline.get(field):
                        errors.append(f"{group_id} FQ1 组内字段 {field} 不一致")
        elif question_type == "FQ2":
            conditions = Counter(str(row.get("view_condition")) for row in rows)
            if conditions != Counter(FQ2_CONDITIONS):
                errors.append(f"{group_id} FQ2 条件不完整：{dict(conditions)}")
            by_condition = {str(row.get("view_condition")): row for row in rows}
            baseline = by_condition.get("one_valid_mss")
            deleted = by_condition.get("remove_required_view")
            added = by_condition.get("hard_distractor_added")
            if baseline and deleted and added:
                for row in (deleted, added):
                    for field in (
                        "question",
                        "options",
                        "reference_frame",
                        "reference_anchor",
                        "object_instances",
                        "target_instance_id",
                        "relatum_instance_id",
                        "bridge_instance_ids",
                    ):
                        if row.get(field) != baseline.get(field):
                            errors.append(f"{group_id} FQ2 组内字段 {field} 不一致")
                mss = list(baseline.get("mss_images", []) or [])
                removed = list(deleted.get("removed_images", []) or [])
                if list(baseline.get("images", []) or []) != mss:
                    errors.append(f"{group_id} FQ2 MSS 版必须满足 images == mss_images")
                if list(deleted.get("mss_images", []) or []) != mss:
                    errors.append(f"{group_id} FQ2 删除版 mss_images 与基准不一致")
                if list(added.get("mss_images", []) or []) != mss:
                    errors.append(f"{group_id} FQ2 干扰版 mss_images 与基准不一致")
                expected_removed = int(tier["quota"]["fq2"]["delete_count"])
                if len(removed) != expected_removed:
                    errors.append(f"{group_id} FQ2 必须纯删除 {expected_removed} 张必要图")
                if set(deleted.get("images", []) or []) != set(mss) - set(removed):
                    errors.append(f"{group_id} FQ2 删除版不是 MSS 的纯删除")
                added_only = set(added.get("images", []) or []) - set(mss)
                expected_added = int(tier["quota"]["fq2"]["distractor_count"])
                if not set(mss).issubset(set(added.get("images", []) or [])) or len(added_only) != expected_added:
                    errors.append(f"{group_id} FQ2 必须在完整 MSS 上纯加入 {expected_added} 张干扰图")
                if baseline.get("answer") != added.get("answer"):
                    errors.append(f"{group_id} FQ2 MSS 与干扰版金答案必须相同")
                try:
                    insufficient = _insufficient_position(baseline)
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    if deleted.get("answer") != insufficient:
                        errors.append(f"{group_id} FQ2 删除版必须以“信息不足”为金答案")
                    if baseline.get("answer") == insufficient or added.get("answer") == insufficient:
                        errors.append(f"{group_id} FQ2 MSS/干扰版不得以“信息不足”为金答案")
        else:
            errors.append(f"{group_id} 含未知题型 {question_type}")
    for question_type in ("FQ1", "FQ2"):
        if group_types[question_type] != expected_per_task:
            errors.append(
                f"{question_type} 应为 {expected_per_task} 组，实际为 {group_types[question_type]}"
            )
    answer_counts = Counter(str(row.get("answer")) for row in records)
    if set(answer_counts) - set(ANSWER_LABELS):
        errors.append(f"存在非法答案标签：{sorted(set(answer_counts) - set(ANSWER_LABELS))}")
    expected_answer = len(records) // 4 if len(records) % 4 == 0 else None
    if expected_answer is not None and any(answer_counts[label] != expected_answer for label in ANSWER_LABELS):
        errors.append(f"全局答案位置不符合冻结平衡：{dict(answer_counts)}")
    sample_ids = [str(row.get("sample_id", "")) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample_id 存在重复")

    expected_group_ids = {
        f"group_{number:05d}"
        for number in range(
            int(policy["first_group_number"]),
            int(policy["first_group_number"]) + expected_groups,
        )
    }
    if set(groups) != expected_group_ids:
        errors.append(
            f"relation_group_id 集合不等于冻结连续范围；missing={sorted(expected_group_ids - set(groups))[:3]} "
            f"extra={sorted(set(groups) - expected_group_ids)[:3]}"
        )
    expected_sample_ids = {
        f"{policy['sample_id_prefix']}_{number:05d}"
        for number in range(int(tier["first_sample_number"]), int(tier["last_sample_number"]) + 1)
    }
    if set(sample_ids) != expected_sample_ids:
        errors.append(
            f"sample_id 集合不等于冻结连续范围；missing={sorted(expected_sample_ids - set(sample_ids))[:3]} "
            f"extra={sorted(set(sample_ids) - expected_sample_ids)[:3]}"
        )

    layout = tier["quota"]["answer_layout"]
    for frame in FQ1_FRAMES:
        observed = Counter(
            str(row["answer"])
            for row in records
            if row.get("question_type") == "FQ1" and row.get("reference_frame") == frame
        )
        expected = Counter({key: int(value) for key, value in layout["fq1_per_frame_position"][frame].items()})
        if observed != expected:
            errors.append(f"FQ1 {frame} 答案位置不符合冻结配额：{dict(observed)}")
    fq2_rows = [row for row in records if row.get("question_type") == "FQ2"]
    observed_fq2_answers = Counter(str(row["answer"]) for row in fq2_rows)
    expected_fq2_answers = Counter({key: int(value) for key, value in layout["fq2_global_answer_position"].items()})
    if observed_fq2_answers != expected_fq2_answers:
        errors.append(f"FQ2 全局答案位置不符合冻结配额：{dict(observed_fq2_answers)}")

    fq2_groups = [rows for rows in groups.values() if rows and rows[0].get("question_type") == "FQ2"]
    observed_frames: Counter[str] = Counter()
    observed_mss_counts: Counter[str] = Counter()
    observed_spatial: dict[str, Counter[str]] = defaultdict(Counter)
    observed_insufficient: dict[str, Counter[str]] = defaultdict(Counter)
    observed_pairs: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for rows in fq2_groups:
        by_condition = {str(row["view_condition"]): row for row in rows}
        baseline = by_condition.get("one_valid_mss")
        deleted = by_condition.get("remove_required_view")
        if baseline is None or deleted is None:
            continue
        count = str(len(baseline.get("mss_images", []) or []))
        observed_frames[str(baseline["reference_frame"])] += 1
        observed_mss_counts[count] += 1
        spatial = str(baseline["answer"])
        insufficient = str(deleted["answer"])
        observed_spatial[count][spatial] += 1
        observed_insufficient[count][insufficient] += 1
        observed_pairs[count][spatial][insufficient] += 1
    fq2_quota = tier["quota"]["fq2"]
    if observed_frames != Counter({key: int(value) for key, value in fq2_quota["reference_frame_totals"].items()}):
        errors.append(f"FQ2 reference_frame 配额不符：{dict(observed_frames)}")
    if observed_mss_counts != Counter({key: int(value) for key, value in fq2_quota["mss_count_totals"].items()}):
        errors.append(f"FQ2 MSS 图片数配额不符：{dict(observed_mss_counts)}")
    for count in sorted(layout["fq2_spatial_position_by_mss"]):
        if observed_spatial[count] != Counter({key: int(value) for key, value in layout["fq2_spatial_position_by_mss"][count].items()}):
            errors.append(f"FQ2 MSS={count} 空间答案位置配额不符")
        if observed_insufficient[count] != Counter({key: int(value) for key, value in layout["fq2_insufficient_position_by_mss"][count].items()}):
            errors.append(f"FQ2 MSS={count} 信息不足位置配额不符")
        for spatial, expected_values in layout["fq2_ordered_pair_by_mss"][count].items():
            if observed_pairs[count][spatial] != Counter({key: int(value) for key, value in expected_values.items()}):
                errors.append(f"FQ2 MSS={count}, spatial={spatial} 有序位置对配额不符")

    fq1_representatives = []
    for rows in groups.values():
        if not rows or rows[0].get("question_type") != "FQ1":
            continue
        representative = next((row for row in rows if row.get("reference_frame") == "camera"), None)
        if representative is not None:
            fq1_representatives.append(representative)
    observed_fq1_counts = Counter(str(len(row.get("images", []) or [])) for row in fq1_representatives)
    expected_fq1_counts = Counter({key: int(value) for key, value in tier["quota"]["fq1"]["image_count_totals"].items()})
    if observed_fq1_counts != expected_fq1_counts:
        errors.append(f"FQ1 图片数配额不符：{dict(observed_fq1_counts)}")

    quota_path = policy_path.parent / str(tier["quota_filename"])
    quota_rows: list[dict[str, str]] = []
    if not quota_path.is_file():
        errors.append(f"缺少冻结 quota CSV：{quota_path}")
    else:
        if sha256_file(quota_path) != str(tier["quota_sha256"]):
            errors.append("冻结 quota CSV 的 SHA-256 与 policy 不一致")
        with quota_path.open("r", encoding="utf-8-sig", newline="") as stream:
            quota_rows = list(csv.DictReader(stream))
        quota_groups = {str(row.get("planned_relation_group_id")) for row in quota_rows}
        if len(quota_rows) != expected_groups or quota_groups != expected_group_ids:
            errors.append("quota CSV 的题组集合/行数与冻结 policy 不一致")
        for row in quota_rows:
            group_id = str(row.get("planned_relation_group_id"))
            actual = groups.get(group_id, [])
            if actual and str(row.get("question_type")) != str(actual[0].get("question_type")):
                errors.append(f"{group_id} 的题型与 quota CSV 不一致")

    dataset_root = dataset_dir.resolve()
    referenced: set[str] = set()
    for row in records:
        for field in ("images", "mss_images", "removed_images"):
            for relative in row.get(field, []) or []:
                normalized = str(relative).replace("\\", "/")
                referenced.add(normalized)
                candidate = (dataset_root / normalized).resolve()
                try:
                    candidate.relative_to(dataset_root)
                except ValueError:
                    errors.append(f"图片路径越出数据目录：{normalized}")
                    continue
                if not candidate.is_file():
                    errors.append(f"缺少图片：{normalized}")
    physical = {
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "images").rglob("*")
        if path.is_file()
    }
    if physical != referenced:
        errors.append(
            f"图片资产闭包不一致：missing={len(referenced - physical)}, orphan={len(physical - referenced)}"
        )
    return {
        "status": "BLOCKED" if errors else "REVIEW_REQUIRED" if warnings else "PASS",
        "active_tier": active_tier,
        "policy_version": policy.get("version"),
        "records": len(records),
        "groups": len(groups),
        "groups_by_type": dict(group_types),
        "quality_status": dict(statuses),
        "answer_distribution": dict(answer_counts),
        "quota_csv": str(quota_path.resolve()) if quota_path.is_file() else str(quota_path),
        "referenced_assets": len(referenced),
        "physical_assets": len(physical),
        "errors": errors,
        "warnings": warnings,
    }


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metadata(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    metadata: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for model in config.get("models", []):
        if not isinstance(model, dict) or not model.get("model_id"):
            continue
        model_id = str(model["model_id"])
        if model_id in metadata:
            raise ValueError(f"模型配置含重复 model_id：{model_id}")
        if "category" not in model or "setting" not in model:
            raise ValueError(f"模型 {model_id} 必须显式配置 category 与 setting")
        metadata[model_id] = model
        order.append(model_id)
    return metadata, order


def _main_values(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {column: None for column in MAIN_COLUMNS}
    frames = result["fq1_by_reference_frame"]
    conditions = result["fq2_by_view_condition"]
    main = result["main_metrics"]
    return {
        "item_accuracy": result["item_accuracy"]["value"],
        "valid_response_rate": result["valid_response_rate"]["value"],
        "camera": frames["camera"]["accuracy"],
        "object_intrinsic": frames["object_intrinsic"]["accuracy"],
        "world": frames["world"]["accuracy"],
        "FQ1_GEM": main["FQ1_GEM"]["value"],
        "one_valid_mss": conditions["one_valid_mss"]["accuracy"],
        "remove_required_view": conditions["remove_required_view"]["accuracy"],
        "hard_distractor_added": conditions["hard_distractor_added"]["accuracy"],
        "NTS": main["NTS"]["value"],
        "DIS": main["DIS"]["value"],
        "FQ1_GEM_ci95": main["FQ1_GEM"].get("ci95"),
        "NTS_ci95": main["NTS"].get("ci95"),
        "DIS_ci95": main["DIS"].get("ci95"),
    }


def _category(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "open": "open-source",
        "opensource": "open-source",
        "open-source": "open-source",
        "closed": "closed-source",
        "closedsource": "closed-source",
        "closed-source": "closed-source",
        "other": "other",
        "human": "other",
    }
    if normalized not in aliases:
        raise ValueError(f"未知模型类别：{value!r}；应为 open-source、closed-source 或 other")
    return aliases[normalized]


def build_main_rows(metrics: dict[str, Any], config_path: Path | None = None) -> list[dict[str, Any]]:
    config = _load_config(config_path)
    metadata, configured_order = _metadata(config)
    by_id = {str(model["model_id"]): model for model in metrics.get("models", [])}
    missing_metadata = sorted(set(by_id) - set(metadata))
    if missing_metadata:
        raise ValueError(f"下列预测模型未写入表格配置：{missing_metadata}")
    ordered_ids = configured_order + sorted(set(by_id) - set(configured_order))
    rows: list[dict[str, Any]] = []
    for model_id in ordered_ids:
        meta = metadata.get(model_id, {})
        result = by_id.get(model_id, {}).get("results", {}).get("all-images")
        values = _main_values(result)
        rows.append({
            "model_id": model_id,
            "display_name": str(meta.get("display_name") or model_id),
            "setting": str(meta["setting"]),
            "category": _category(meta.get("category")),
            "coverage_status": result.get("status") if result else "NOT_RUN",
            "coverage": result.get("coverage") if result else None,
            "theoretical": False,
            **values,
        })
    if bool(config.get("include_random_references", True)):
        rows.extend([
            {
                "model_id": "random-iid-uniform",
                "display_name": "Random (iid uniform)",
                "setting": "theory",
                "category": "other",
                "coverage_status": "REFERENCE",
                "coverage": None,
                "theoretical": True,
                "item_accuracy": 1 / 4,
                "valid_response_rate": 1.0,
                "camera": 1 / 4,
                "object_intrinsic": 1 / 4,
                "world": 1 / 4,
                "FQ1_GEM": 1 / 64,
                "one_valid_mss": 1 / 4,
                "remove_required_view": 1 / 4,
                "hard_distractor_added": 1 / 4,
                "NTS": 1 / 16,
                "DIS": 1 / 16,
                "FQ1_GEM_ci95": None,
                "NTS_ci95": None,
                "DIS_ci95": None,
            },
            {
                "model_id": "random-group-consistent",
                "display_name": "Random (group-consistent)",
                "setting": "theory",
                "category": "other",
                "coverage_status": "REFERENCE",
                "coverage": None,
                "theoretical": True,
                "item_accuracy": 1 / 4,
                "valid_response_rate": 1.0,
                "camera": 1 / 4,
                "object_intrinsic": 1 / 4,
                "world": 1 / 4,
                "FQ1_GEM": 0.0,
                "one_valid_mss": 1 / 4,
                "remove_required_view": 1 / 4,
                "hard_distractor_added": 1 / 4,
                "NTS": 0.0,
                "DIS": 1 / 4,
                "FQ1_GEM_ci95": None,
                "NTS_ci95": None,
                "DIS_ci95": None,
            },
        ])
    return rows


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}"


def _pct_ci(value: float | None, interval: list[float] | None) -> str:
    if value is None:
        return "--"
    if not interval:
        return _pct(value)
    return f"{_pct(value)} [{_pct(interval[0])}, {_pct(interval[1])}]"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_metric(row: dict[str, Any], name: str) -> str:
    value = row.get(name)
    interval = row.get(f"{name}_ci95")
    if value is None:
        return "--"
    if interval:
        return rf"\shortstack{{{_pct(value)}\\{{\scriptsize [{_pct(interval[0])}, {_pct(interval[1])}]}}}}"
    return _pct(value)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_main_markdown(
    rows: list[dict[str, Any]],
    ci_description: str,
    table_status: str = "DRAFT",
) -> str:
    title_suffix = "" if table_status == "PUBLICATION_READY" else " [DRAFT — NOT FOR PAPER]"
    lines = [
        f"# FrameEvidence main results (all-images){title_suffix}",
        "",
        f"Table status: **{table_status}**.",
        "All values are percentages. Invalid or missing responses count as incorrect; Valid is reported separately.",
        f"FQ1-GEM, NTS, and DIS show 95% intervals using {ci_description}.",
        "Denominators for a complete run are 960 items overall, 160 items per FQ1 frame/"
        "FQ2 condition, and 160 groups for each of FQ1-GEM, NTS, and DIS.",
        "",
        "| Model | Setting | Item Acc. | Valid | Camera | Object | Fixed | FQ1-GEM | MSS | Delete | Add | NTS | DIS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "open-source": "Open-source MLLMs",
        "closed-source": "Closed-source MLLMs",
        "other": "Other methods / references",
    }
    for category in ("open-source", "closed-source", "other"):
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        lines.append(f"| *{labels[category]}* |  |  |  |  |  |  |  |  |  |  |  |  |")
        for row in selected:
            display = str(row["display_name"])
            if row["coverage_status"] not in {"PASS", "REFERENCE"}:
                display += f" ({row['coverage_status']})"
            lines.append(
                f"| {_markdown_cell(display)} | {_markdown_cell(row['setting'])} | {_pct(row['item_accuracy'])} | "
                f"{_pct(row['valid_response_rate'])} | {_pct(row['camera'])} | "
                f"{_pct(row['object_intrinsic'])} | {_pct(row['world'])} | "
                f"{_pct_ci(row['FQ1_GEM'], row.get('FQ1_GEM_ci95'))} | "
                f"{_pct(row['one_valid_mss'])} | {_pct(row['remove_required_view'])} | "
                f"{_pct(row['hard_distractor_added'])} | "
                f"{_pct_ci(row['NTS'], row.get('NTS_ci95'))} | "
                f"{_pct_ci(row['DIS'], row.get('DIS_ci95'))} |"
            )
    lines.extend([
        "",
        "Camera/Object/Fixed and MSS/Delete/Add are per-condition item accuracies. "
        "The theoretical random rows are references, not measured model runs.",
        "",
    ])
    return "\n".join(lines)


def render_main_latex(
    rows: list[dict[str, Any]],
    ci_description: str,
    table_status: str = "DRAFT",
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.1pt}",
        r"\begin{tabular}{llrrrrrrrrrrr}",
        r"\toprule",
        r"Model & Setting & \multicolumn{2}{c}{Overall} & \multicolumn{4}{c}{FQ1} & \multicolumn{5}{c}{FQ2} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-13}",
        r" & & Item Acc. & Valid & Camera & Object & Fixed & FQ1-GEM & MSS & Delete & Add & NTS & DIS \\",
        r"\midrule",
    ]
    labels = {
        "open-source": "Open-source MLLMs",
        "closed-source": "Closed-source MLLMs",
        "other": "Other methods / references",
    }
    first_block = True
    for category in ("open-source", "closed-source", "other"):
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        if not first_block:
            lines.append(r"\addlinespace")
        first_block = False
        lines.append(rf"\multicolumn{{13}}{{l}}{{\textit{{{labels[category]}}}}} \\")
        for row in selected:
            display = row["display_name"]
            if row["coverage_status"] not in {"PASS", "REFERENCE"}:
                display += f" ({row['coverage_status']})"
            values = [
                _pct(row["item_accuracy"]),
                _pct(row["valid_response_rate"]),
                _pct(row["camera"]),
                _pct(row["object_intrinsic"]),
                _pct(row["world"]),
                _latex_metric(row, "FQ1_GEM"),
                _pct(row["one_valid_mss"]),
                _pct(row["remove_required_view"]),
                _pct(row["hard_distractor_added"]),
                _latex_metric(row, "NTS"),
                _latex_metric(row, "DIS"),
            ]
            lines.append(
                f"{_latex_escape(display)} & {_latex_escape(row['setting'])} & "
                + " & ".join(values)
                + r" \\"
            )
    status_sentence = (
        "All-images results on the complete verified compact-320 release. "
        if table_status == "PUBLICATION_READY"
        else "Draft table scaffold; publication gates are not yet satisfied. "
    )
    caption = (
        status_sentence
        +
        "All values are percentages; invalid or missing responses count as incorrect. "
        f"Bracketed ranges are 95\\% intervals for the three group-level metrics using {_latex_escape(ci_description)}. "
        "Complete-run denominators are 960 items overall and 160 groups per task. "
        "Camera, Object, and Fixed denote camera, object-intrinsic, and fixed-scene reference frames. "
        "MSS is one valid subset-minimal set; Delete requires the information-insufficient answer "
        "after removing one required view; Add appends one hard non-MSS distractor."
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\label{tab:main-results}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def _write_main_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "category",
        "model_id",
        "display_name",
        "setting",
        "coverage_status",
        *MAIN_COLUMNS,
        "FQ1_GEM_ci95_low",
        "FQ1_GEM_ci95_high",
        "NTS_ci95_low",
        "NTS_ci95_high",
        "DIS_ci95_low",
        "DIS_ci95_high",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {field: row.get(field) for field in fields}
            for metric in ("FQ1_GEM", "NTS", "DIS"):
                interval = row.get(f"{metric}_ci95") or [None, None]
                flattened[f"{metric}_ci95_low"] = interval[0]
                flattened[f"{metric}_ci95_high"] = interval[1]
            writer.writerow(flattened)


def _input_control_rows(metrics: dict[str, Any], config_path: Path | None) -> list[dict[str, Any]]:
    config = _load_config(config_path)
    metadata, configured_order = _metadata(config)
    by_id = {str(model["model_id"]): model for model in metrics.get("models", [])}
    missing_metadata = sorted(set(by_id) - set(metadata))
    if missing_metadata:
        raise ValueError(f"下列预测模型未写入表格配置：{missing_metadata}")
    ordered_ids = configured_order + sorted(set(by_id) - set(configured_order))
    rows: list[dict[str, Any]] = []
    for model_id in ordered_ids:
        model = by_id.get(model_id, {})
        results = model.get("results", {})
        meta = metadata.get(model_id, {})
        all_images = results.get("all-images", {}).get("item_accuracy", {}).get("value")
        configured_modes = tuple(
            mode for mode in MODEL_MODES
            if mode in set(meta.get("evaluation_modes", MODEL_MODES))
        )
        for mode in configured_modes:
            result = results.get(mode)
            item = result.get("item_accuracy", {}).get("value") if result else None
            main = result.get("main_metrics", {}) if result else {}
            rows.append({
                "model_id": model_id,
                "display_name": str(meta.get("display_name") or model_id),
                "category": _category(meta.get("category")),
                "evaluation_mode": mode,
                "coverage_status": result.get("status") if result else "NOT_RUN",
                "item_accuracy": item,
                "valid_response_rate": result.get("valid_response_rate", {}).get("value") if result else None,
                "FQ1_GEM": main.get("FQ1_GEM", {}).get("value"),
                "NTS": main.get("NTS", {}).get("value"),
                "DIS": main.get("DIS", {}).get("value"),
                "delta_item_vs_all_images": None if item is None or all_images is None else item - all_images,
            })
    return rows


def write_input_controls(metrics: dict[str, Any], config_path: Path | None, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _input_control_rows(metrics, config_path)
    csv_path = output_dir / "input_controls.csv"
    md_path = output_dir / "input_controls.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["model_id"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Model input controls",
        "",
        "| Model | Mode | Item Acc. | Valid | FQ1-GEM | NTS | DIS | Δ Item vs All | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {_markdown_cell(row['display_name'])} | {row['evaluation_mode']} | "
            f"{_pct(row['item_accuracy'])} | {_pct(row['valid_response_rate'])} | "
            f"{_pct(row['FQ1_GEM'])} | {_pct(row['NTS'])} | {_pct(row['DIS'])} | "
            f"{_pct(row['delta_item_vs_all_images'])} | {row['coverage_status']} |"
        )
    lines.extend([
        "",
        "Count-only and count+option-schema are data-level empirical shortcut audits and are reported separately.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path.resolve()), "markdown": str(md_path.resolve())}


def write_main_tables(
    metrics: dict[str, Any],
    output_dir: Path,
    config_path: Path | None = None,
    release_status: str | None = None,
    shortcut_status: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    metadata, _ = _metadata(config)
    rows = build_main_rows(metrics, config_path)
    confidence = metrics.get("confidence_interval", {})
    source = str(confidence.get("cluster_source") or "none")
    completed_models = sum(
        all(
            model.get("results", {}).get(mode, {}).get("status") == "PASS"
            for mode in MODEL_MODES
        )
        and _category(metadata.get(str(model.get("model_id")), {}).get("category"))
        in {"open-source", "closed-source"}
        for model in metrics.get("models", [])
    )
    minimum_models = int(config.get("publication_gate", {}).get("minimum_complete_models", 8))
    require_human = bool(config.get("publication_gate", {}).get("require_human_consensus", True))
    gate_reasons: list[str] = []
    if metrics.get("status") != "PASS":
        gate_reasons.append(f"formal_metrics status={metrics.get('status', 'MISSING')}")
    if completed_models < minimum_models:
        gate_reasons.append(
            f"models complete in all-images/text-only/single-image={completed_models}, "
            f"required>={minimum_models}"
        )
    if require_human:
        human_complete = any(
            _category(metadata.get(str(model.get("model_id")), {}).get("category")) == "other"
            and bool(metadata.get(str(model.get("model_id")), {}).get("is_human_consensus"))
            and bool(metadata.get(str(model.get("model_id")), {}).get("human_protocol_verified"))
            and model.get("results", {}).get("all-images", {}).get("status") == "PASS"
            for model in metrics.get("models", [])
        )
        if not human_complete:
            gate_reasons.append("complete human-consensus all-images row is missing")
    if release_status != "PASS":
        gate_reasons.append(f"release preflight status={release_status or 'MISSING'}")
    if shortcut_status != "PASS":
        gate_reasons.append(f"count-only shortcut gate status={shortcut_status or 'MISSING'}")
    if not source or source == "none":
        gate_reasons.append("cluster source is missing")
    table_status = "PUBLICATION_READY" if not gate_reasons else "DRAFT_NOT_FOR_PAPER"
    table_rows = (
        [row for row in rows if row["coverage_status"] in {"PASS", "REFERENCE"}]
        if table_status == "PUBLICATION_READY"
        else rows
    )
    ci_description = f"cluster bootstrap ({source})"
    json_path = output_dir / "main_table.json"
    csv_path = output_dir / "main_table.csv"
    md_path = output_dir / "main_table.md"
    tex_path = output_dir / "main_table.tex"
    write_json(json_path, {
        "evaluation_profile": "formal_release",
        "table_status": table_status,
        "publication_gate_reasons": gate_reasons,
        "complete_three_mode_models": completed_models,
        "minimum_complete_models": minimum_models,
        "confidence_interval": confidence,
        "rows": table_rows,
    })
    _write_main_csv(table_rows, csv_path)
    md_path.write_text(render_main_markdown(table_rows, ci_description, table_status), encoding="utf-8")
    tex_path.write_text(render_main_latex(table_rows, ci_description, table_status), encoding="utf-8")
    controls = write_input_controls(metrics, config_path, output_dir)
    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
        "latex": str(tex_path.resolve()),
        "input_controls_csv": controls["csv"],
        "input_controls_markdown": controls["markdown"],
        "table_status": table_status,
    }
