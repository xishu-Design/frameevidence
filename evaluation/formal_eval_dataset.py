from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from formal_eval_core import (
    ANSWER_LABELS,
    FQ1_FRAMES,
    FQ2_CONDITIONS,
    read_jsonl,
    sha256_file,
    write_json,
)


SPLITS = ("train", "dev", "test")
DEFAULT_GROUP_TARGETS = {"train": 192, "dev": 32, "test": 96}
DEFAULT_TASK_TARGETS = {
    "train": {"FQ1": 96, "FQ2": 96},
    "dev": {"FQ1": 16, "FQ2": 16},
    "test": {"FQ1": 48, "FQ2": 48},
}

SCENE_LABELS = {
    "indoor_tabletop": "Indoor tabletop",
    "indoor_large_space": "Indoor large space",
    "outdoor_public_space": "Outdoor public space",
}
RELATION_LABELS = {
    "axial": "Axial",
    "quadrant": "Quadrant",
}
FRAME_LABELS = {
    "camera": "FQ1: Camera frame",
    "object_intrinsic": "FQ1: Object-intrinsic frame",
    "world": "FQ1: Fixed-scene frame",
}
CONDITION_LABELS = {
    "one_valid_mss": "FQ2: Complete MSS",
    "remove_required_view": "FQ2: Required-view removed",
    "hard_distractor_added": "FQ2: Hard distractor added",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_cluster_split_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    """Load explicit whole-cluster split assignments.

    Overrides are intentionally cluster-scoped: a paper/demo example must never
    be moved across splits without every asset-connected group moving with it.
    """

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"缺少 split override 文件：{path}")
    overrides: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        cluster_id = str(row.get("scene_cluster_id", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        reason = str(row.get("reason", "")).strip()
        if not cluster_id or not split:
            raise ValueError(
                f"{path} 每行必须含 scene_cluster_id 和 split"
            )
        if split not in SPLITS:
            raise ValueError(f"{path} 中 {cluster_id} 的 split 非法：{split}")
        if cluster_id in overrides:
            raise ValueError(f"{path} 中 cluster 重复：{cluster_id}")
        overrides[cluster_id] = {
            "split": split,
            "reason": reason,
        }
    return overrides


def _quota_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"缺少配额文件：{path}")
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        group_id = str(row.get("planned_relation_group_id", "")).strip()
        if not group_id:
            raise ValueError(f"{path} 存在缺少 planned_relation_group_id 的行")
        if group_id in result:
            raise ValueError(f"{path} 中题组重复：{group_id}")
        result[group_id] = row
    return result


def _sqlite_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"缺少生产数据库：{path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT release_slot, relation_group_id, question_type, "
            "relation_stratum, scene_category, status FROM tasks"
        ).fetchall()
    finally:
        connection.close()
    return {str(row["relation_group_id"]): dict(row) for row in rows}


def load_group_records(
    dataset_dir: Path,
    *,
    quota_path: Path | None = None,
    metadata_db: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the 320 atomic triplet groups and the metadata needed for splitting."""

    annotation_path = dataset_dir / "annotations.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"缺少标注文件：{annotation_path}")
    records = read_jsonl(annotation_path)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        group_id = str(row.get("relation_group_id", "")).strip()
        if not group_id:
            raise ValueError("annotations.jsonl 存在缺少 relation_group_id 的记录")
        by_group[group_id].append(row)

    quotas = _quota_metadata(quota_path)
    database = _sqlite_metadata(metadata_db)
    groups: dict[str, dict[str, Any]] = {}
    for group_id, rows in sorted(by_group.items()):
        if len(rows) != 3:
            raise ValueError(f"题组 {group_id} 应有 3 道题，实际为 {len(rows)}")
        task_values = {str(row.get("question_type", "")) for row in rows}
        if len(task_values) != 1 or next(iter(task_values)) not in {"FQ1", "FQ2"}:
            raise ValueError(f"题组 {group_id} 的 question_type 不一致：{sorted(task_values)}")
        question_type = next(iter(task_values))
        quota = quotas.get(group_id, {})
        db_row = database.get(group_id, {})
        scene = str(db_row.get("scene_category") or quota.get("scene_category") or "").strip()
        relation = str(db_row.get("relation_stratum") or "").strip()
        release_slot_raw = db_row.get("release_slot") or quota.get("release_slot")
        release_slot = int(release_slot_raw) if str(release_slot_raw or "").strip() else None
        image_paths = {
            str(image).replace("\\", "/")
            for row in rows
            for image in (row.get("images") or [])
        }
        mss_sizes = [len(row.get("mss_images") or []) for row in rows]
        reference_frames = sorted({str(row.get("reference_frame", "")) for row in rows})
        answers = Counter(str(row.get("answer", "")) for row in rows)
        if set(answers) - set(ANSWER_LABELS):
            raise ValueError(f"题组 {group_id} 存在非法答案标签：{sorted(answers)}")
        groups[group_id] = {
            "relation_group_id": group_id,
            "release_slot": release_slot,
            "question_type": question_type,
            "scene_category": scene or None,
            "relation_stratum": relation or None,
            "image_count": len(image_paths),
            "mss_count": max(mss_sizes, default=0),
            "reference_frames": reference_frames,
            "answer_counts": dict(answers),
            "records": rows,
        }

    if quotas:
        missing = sorted(set(groups) - set(quotas))
        if missing:
            raise ValueError(f"配额文件未覆盖全部题组，缺少：{missing[:5]}")
    if database:
        missing = sorted(set(groups) - set(database))
        if missing:
            raise ValueError(f"生产数据库未覆盖全部题组，缺少：{missing[:5]}")
    return groups


def _group_features(group: dict[str, Any]) -> Counter[str]:
    features: Counter[str] = Counter()
    task = str(group["question_type"])
    features[f"task:{task}"] += 1
    if group.get("scene_category"):
        features[f"scene:{task}:{group['scene_category']}"] += 1
    if group.get("relation_stratum"):
        features[f"relation:{task}:{group['relation_stratum']}"] += 1
    if task == "FQ1":
        features[f"fq1_image_count:{group['image_count']}"] += 1
    else:
        frames = group.get("reference_frames") or []
        if len(frames) == 1:
            features[f"fq2_reference_frame:{frames[0]}"] += 1
        features[f"fq2_mss_count:{group['mss_count']}"] += 1
    for answer, count in dict(group.get("answer_counts", {})).items():
        features[f"answer:{answer}"] += int(count)
    return features


def _cluster_records(
    groups: dict[str, dict[str, Any]],
    cluster_by_group: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if set(groups) != set(cluster_by_group):
        missing = sorted(set(groups) - set(cluster_by_group))
        extra = sorted(set(cluster_by_group) - set(groups))
        raise ValueError(
            f"聚类映射必须精确覆盖题组；missing={missing[:5]}, extra={extra[:5]}"
        )
    members: dict[str, list[str]] = defaultdict(list)
    for group_id, cluster_id in cluster_by_group.items():
        if not str(cluster_id).strip():
            raise ValueError(f"题组 {group_id} 的 cluster id 为空")
        members[str(cluster_id)].append(group_id)
    result: dict[str, dict[str, Any]] = {}
    for cluster_id, group_ids in sorted(members.items()):
        task_counts = Counter(groups[group_id]["question_type"] for group_id in group_ids)
        features: Counter[str] = Counter()
        for group_id in group_ids:
            features.update(_group_features(groups[group_id]))
        result[cluster_id] = {
            "cluster_id": cluster_id,
            "group_ids": sorted(group_ids),
            "task_counts": dict(task_counts),
            "features": features,
        }
    return result


def _random_exact_subset(
    clusters: dict[str, dict[str, Any]],
    target: dict[str, int],
    rng: random.Random,
) -> set[str] | None:
    ordered = list(clusters)
    rng.shuffle(ordered)
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    target_pair = (int(target["FQ1"]), int(target["FQ2"]))
    for cluster_id in ordered:
        task_counts = clusters[cluster_id]["task_counts"]
        vector = (int(task_counts.get("FQ1", 0)), int(task_counts.get("FQ2", 0)))
        updated = dict(states)
        for state, chosen in list(states.items()):
            candidate = (state[0] + vector[0], state[1] + vector[1])
            if candidate[0] > target_pair[0] or candidate[1] > target_pair[1]:
                continue
            if candidate not in updated or rng.random() < 0.08:
                updated[candidate] = chosen + (cluster_id,)
        states = updated
    selected = states.get(target_pair)
    return set(selected) if selected is not None else None


def _feature_totals(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
) -> dict[str, Counter[str]]:
    result = {split: Counter() for split in SPLITS}
    for group_id, split in assignment.items():
        result[split].update(_group_features(groups[group_id]))
    return result


def _balance_score(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    group_targets: dict[str, int],
) -> float:
    observed = _feature_totals(groups, assignment)
    totals: Counter[str] = Counter()
    for split in SPLITS:
        totals.update(observed[split])
    total_groups = sum(group_targets.values())
    score = 0.0
    for feature, total in totals.items():
        if feature.startswith("task:"):
            continue
        if feature.startswith("answer:"):
            weight = 4.0
        elif feature.startswith(("scene:", "relation:")):
            weight = 4.0
        else:
            weight = 1.5
        for split in SPLITS:
            expected = total * group_targets[split] / total_groups
            difference = observed[split][feature] - expected
            score += weight * difference * difference / max(1.0, expected)
    return score


def _improve_by_equal_vector_swaps(
    groups: dict[str, dict[str, Any]],
    cluster_by_group: dict[str, str],
    assignment: dict[str, str],
    group_targets: dict[str, int],
    *,
    max_passes: int = 12,
) -> tuple[dict[str, str], float, int]:
    """Improve soft balance without changing hard task/group counts."""

    clusters = _cluster_records(groups, cluster_by_group)
    split_by_cluster = {
        cluster_id: assignment[value["group_ids"][0]]
        for cluster_id, value in clusters.items()
    }

    def expand() -> dict[str, str]:
        return {
            group_id: split_by_cluster[cluster_by_group[group_id]]
            for group_id in groups
        }

    current = expand()
    current_score = _balance_score(groups, current, group_targets)
    swaps = 0
    ordered = sorted(clusters)
    for _ in range(max_passes):
        best_pair: tuple[str, str] | None = None
        best_score = current_score
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                if split_by_cluster[left] == split_by_cluster[right]:
                    continue
                if clusters[left]["task_counts"] != clusters[right]["task_counts"]:
                    continue
                split_by_cluster[left], split_by_cluster[right] = (
                    split_by_cluster[right],
                    split_by_cluster[left],
                )
                candidate = expand()
                score = _balance_score(groups, candidate, group_targets)
                split_by_cluster[left], split_by_cluster[right] = (
                    split_by_cluster[right],
                    split_by_cluster[left],
                )
                if score < best_score - 1e-12:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        split_by_cluster[left], split_by_cluster[right] = (
            split_by_cluster[right],
            split_by_cluster[left],
        )
        current = expand()
        current_score = best_score
        swaps += 1
    return current, current_score, swaps


def apply_cluster_split_overrides(
    groups: dict[str, dict[str, Any]],
    cluster_by_group: dict[str, str],
    assignment: dict[str, str],
    overrides: dict[str, dict[str, str]] | None,
    *,
    group_targets: dict[str, int] | None = None,
    task_targets: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Apply declared whole-cluster moves and re-run all hard split checks."""

    overrides = dict(overrides or {})
    if not overrides:
        return dict(assignment), {"applied": False, "count": 0, "changes": []}

    if set(groups) != set(cluster_by_group):
        missing = sorted(set(groups) - set(cluster_by_group))
        extra = sorted(set(cluster_by_group) - set(groups))
        raise ValueError(
            f"聚类映射必须精确覆盖题组；missing={missing[:5]}, extra={extra[:5]}"
        )
    members: dict[str, list[str]] = defaultdict(list)
    for group_id, cluster_id in cluster_by_group.items():
        cluster_id = str(cluster_id).strip()
        if not cluster_id:
            raise ValueError(f"题组 {group_id} 的 cluster id 为空")
        members[cluster_id].append(group_id)
    unknown = sorted(set(overrides) - set(members))
    if unknown:
        raise ValueError(f"split override 引用未知 cluster：{unknown[:5]}")

    updated = dict(assignment)
    changes: list[dict[str, Any]] = []
    for cluster_id in sorted(overrides):
        group_ids = sorted(members[cluster_id])
        prior_splits = {assignment[group_id] for group_id in group_ids}
        if len(prior_splits) != 1:
            raise ValueError(
                f"override 前 cluster 已跨 split：{cluster_id} -> {sorted(prior_splits)}"
            )
        before = next(iter(prior_splits))
        after = overrides[cluster_id]["split"]
        for group_id in group_ids:
            updated[group_id] = after
        changes.append({
            "scene_cluster_id": cluster_id,
            "before": before,
            "after": after,
            "group_ids": list(group_ids),
            "reason": overrides[cluster_id].get("reason", ""),
        })

    validation = validate_split_assignment(
        groups,
        updated,
        cluster_by_group,
        group_targets=group_targets,
        task_targets=task_targets,
    )
    return updated, {
        "applied": True,
        "count": len(changes),
        "changes": changes,
        "validation": validation,
    }


def generate_stratified_group_split(
    groups: dict[str, dict[str, Any]],
    cluster_by_group: dict[str, str],
    *,
    seed: int,
    attempts: int = 400,
    group_targets: dict[str, int] | None = None,
    task_targets: dict[str, dict[str, int]] | None = None,
    cluster_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Create a deterministic cluster-disjoint 60/10/30 split.

    Task totals are hard constraints. Scene, relation, image/MSS counts, reference
    frames, and answer positions are optimized as soft balance objectives.
    """

    group_targets = dict(group_targets or DEFAULT_GROUP_TARGETS)
    task_targets = {
        split: dict(values)
        for split, values in (task_targets or DEFAULT_TASK_TARGETS).items()
    }
    if set(group_targets) != set(SPLITS) or set(task_targets) != set(SPLITS):
        raise ValueError(f"split 必须恰为 {SPLITS}")
    if sum(group_targets.values()) != len(groups):
        raise ValueError(
            f"split 组数之和 {sum(group_targets.values())} 与数据集 {len(groups)} 不一致"
        )
    for split in SPLITS:
        if sum(task_targets[split].values()) != group_targets[split]:
            raise ValueError(f"{split} 的题型目标与总组数不一致")
    actual_tasks = Counter(group["question_type"] for group in groups.values())
    target_tasks = Counter()
    for split in SPLITS:
        target_tasks.update(task_targets[split])
    if actual_tasks != target_tasks:
        raise ValueError(f"题型目标 {dict(target_tasks)} 与实际 {dict(actual_tasks)} 不一致")
    if attempts < 1:
        raise ValueError("attempts 必须大于 0")

    clusters = _cluster_records(groups, cluster_by_group)
    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score: float | None = None
    feasible_attempts = 0
    cluster_ids = set(clusters)
    for _ in range(attempts):
        test_clusters = _random_exact_subset(clusters, task_targets["test"], rng)
        if test_clusters is None:
            continue
        remaining = {
            cluster_id: value
            for cluster_id, value in clusters.items()
            if cluster_id not in test_clusters
        }
        dev_clusters = _random_exact_subset(remaining, task_targets["dev"], rng)
        if dev_clusters is None:
            continue
        train_clusters = cluster_ids - test_clusters - dev_clusters
        split_by_cluster = {
            **{cluster_id: "train" for cluster_id in train_clusters},
            **{cluster_id: "dev" for cluster_id in dev_clusters},
            **{cluster_id: "test" for cluster_id in test_clusters},
        }
        assignment = {
            group_id: split_by_cluster[cluster_by_group[group_id]]
            for group_id in groups
        }
        counts = Counter(assignment.values())
        if any(counts[split] != group_targets[split] for split in SPLITS):
            continue
        feasible_attempts += 1
        score = _balance_score(groups, assignment, group_targets)
        signature = tuple((group_id, assignment[group_id]) for group_id in sorted(assignment))
        best_signature = (
            tuple((group_id, best_assignment[group_id]) for group_id in sorted(best_assignment))
            if best_assignment is not None
            else None
        )
        if (
            best_score is None
            or score < best_score - 1e-12
            or (abs(score - best_score) <= 1e-12 and signature < best_signature)
        ):
            best_assignment = assignment
            best_score = score

    if best_assignment is None:
        raise ValueError(
            "在给定聚类约束下找不到满足题型与组数目标的 split；"
            "不得通过拆分 cluster 绕过，请增加 attempts 或调整协议目标"
        )
    best_assignment, best_score, local_swaps = _improve_by_equal_vector_swaps(
        groups,
        cluster_by_group,
        best_assignment,
        group_targets,
    )
    best_assignment, override_diagnostics = apply_cluster_split_overrides(
        groups,
        cluster_by_group,
        best_assignment,
        cluster_overrides,
        group_targets=group_targets,
        task_targets=task_targets,
    )
    best_score = _balance_score(groups, best_assignment, group_targets)
    validation = validate_split_assignment(
        groups,
        best_assignment,
        cluster_by_group,
        group_targets=group_targets,
        task_targets=task_targets,
    )
    diagnostics = {
        **validation,
        "seed": seed,
        "search_attempts": attempts,
        "feasible_attempts": feasible_attempts,
        "balance_score": best_score,
        "local_balance_swaps": local_swaps,
        "cluster_overrides": override_diagnostics,
        "feature_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in _feature_totals(groups, best_assignment).items()
        },
    }
    return best_assignment, diagnostics


def validate_split_assignment(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    cluster_by_group: dict[str, str],
    *,
    group_targets: dict[str, int] | None = None,
    task_targets: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    missing = sorted(set(groups) - set(assignment))
    extra = sorted(set(assignment) - set(groups))
    if missing or extra:
        raise ValueError(f"split 未精确覆盖数据；missing={missing[:5]}, extra={extra[:5]}")
    invalid = sorted({value for value in assignment.values() if value not in SPLITS})
    if invalid:
        raise ValueError(f"非法 split 标签：{invalid}")
    if set(groups) != set(cluster_by_group):
        raise ValueError("cluster_by_group 未精确覆盖全部题组")
    splits_by_cluster: dict[str, set[str]] = defaultdict(set)
    for group_id, cluster_id in cluster_by_group.items():
        splits_by_cluster[cluster_id].add(assignment[group_id])
    leaked = {
        cluster_id: sorted(splits)
        for cluster_id, splits in splits_by_cluster.items()
        if len(splits) != 1
    }
    if leaked:
        first = next(iter(sorted(leaked)))
        raise ValueError(f"cluster 跨 split 泄漏：{first} -> {leaked[first]}")

    group_counts = Counter(assignment.values())
    task_counts = {split: Counter() for split in SPLITS}
    item_counts = Counter()
    cluster_counts = Counter()
    for group_id, split in assignment.items():
        task_counts[split][groups[group_id]["question_type"]] += 1
        item_counts[split] += len(groups[group_id]["records"])
    for cluster_id, splits in splits_by_cluster.items():
        cluster_counts[next(iter(splits))] += 1
    if group_targets:
        for split, expected in group_targets.items():
            if group_counts[split] != expected:
                raise ValueError(f"{split} 组数应为 {expected}，实际为 {group_counts[split]}")
    if task_targets:
        for split, expected in task_targets.items():
            for task, count in expected.items():
                if task_counts[split][task] != count:
                    raise ValueError(
                        f"{split}/{task} 应为 {count} 组，实际为 {task_counts[split][task]}"
                    )
    return {
        "status": "PASS",
        "group_counts": {split: group_counts[split] for split in SPLITS},
        "item_counts": {split: item_counts[split] for split in SPLITS},
        "task_group_counts": {
            split: {task: task_counts[split][task] for task in ("FQ1", "FQ2")}
            for split in SPLITS
        },
        "cluster_counts": {split: cluster_counts[split] for split in SPLITS},
        "clusters_total": len(splits_by_cluster),
        "cluster_leakage_count": 0,
    }


def write_split_outputs(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    cluster_by_group: dict[str, str],
    diagnostics: dict[str, Any],
    output_dir: Path,
    *,
    cluster_source: str,
    cluster_warning: str | None,
    dataset_path: Path,
    policy_path: Path,
    split_overrides_path: Path | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "relation_group_id",
            "scene_cluster_id",
            "split",
            "release_slot",
            "question_type",
            "scene_category",
            "relation_stratum",
            "cluster_source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group_id in sorted(groups):
            group = groups[group_id]
            writer.writerow({
                "relation_group_id": group_id,
                "scene_cluster_id": cluster_by_group[group_id],
                "split": assignment[group_id],
                "release_slot": group.get("release_slot") or "",
                "question_type": group["question_type"],
                "scene_category": group.get("scene_category") or "",
                "relation_stratum": group.get("relation_stratum") or "",
                "cluster_source": cluster_source,
            })
    provisional = cluster_source == "exact_image_sha256_connected_components_fallback"
    summary = {
        "status": "PROVISIONAL_NOT_FOR_PAPER" if provisional else "FROZEN_CANDIDATE",
        "protocol": "proposed_60_10_30_split_v1",
        "dataset": str(dataset_path.resolve()),
        "annotations_sha256": sha256_file(dataset_path / "annotations.jsonl"),
        "policy": str(policy_path.resolve()),
        "policy_sha256": sha256_file(policy_path),
        "split_overrides": (
            str(split_overrides_path.resolve()) if split_overrides_path else None
        ),
        "split_overrides_sha256": (
            sha256_file(split_overrides_path) if split_overrides_path else None
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "cluster_source": cluster_source,
        "cluster_warning": cluster_warning,
        "warning": (
            "This split is provisional because exact-image SHA components cannot detect "
            "same-scene reuse after viewpoint, crop, resize, or re-encoding. Replace it with "
            "the restricted scene_cluster_id ledger before publication."
            if provisional
            else None
        ),
        "diagnostics": diagnostics,
    }
    summary_path = output_dir / "split_summary.json"
    write_json(summary_path, summary)
    return {"manifest": str(manifest_path), "summary": str(summary_path)}


def load_split_manifest(path: Path) -> tuple[dict[str, str], dict[str, str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 split manifest：{path}")
    rows = _read_csv(path)
    assignment: dict[str, str] = {}
    clusters: dict[str, str] = {}
    sources: set[str] = set()
    for row in rows:
        group_id = str(row.get("relation_group_id", "")).strip()
        cluster_id = str(row.get("scene_cluster_id", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        if not group_id or not cluster_id or not split:
            raise ValueError("split manifest 每行必须含 relation_group_id, scene_cluster_id, split")
        if group_id in assignment:
            raise ValueError(f"split manifest 中题组重复：{group_id}")
        assignment[group_id] = split
        clusters[group_id] = cluster_id
        if str(row.get("cluster_source", "")).strip():
            sources.add(str(row["cluster_source"]).strip())
    if len(sources) > 1:
        raise ValueError(f"split manifest 混用了多个 cluster_source：{sorted(sources)}")
    source = next(iter(sources), "external_scene_cluster_id")
    return assignment, clusters, source


def _count_rows(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    predicate: Any,
) -> dict[str, int]:
    counts = Counter()
    for group_id, group in groups.items():
        split = assignment[group_id]
        for row in group["records"]:
            if predicate(group, row):
                counts[split] += 1
    return {split: counts[split] for split in SPLITS}


def _table_row(label: str, counts: dict[str, int], denominator: int) -> dict[str, Any]:
    total = sum(counts[split] for split in SPLITS)
    return {
        "row_type": "data",
        "label": label,
        **{split: counts[split] for split in SPLITS},
        "total": total,
        "ratio": 100.0 * total / denominator if denominator else 0.0,
    }


def _section(label: str) -> dict[str, Any]:
    return {"row_type": "section", "label": label}


def build_dataset_statistics(
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    cluster_by_group: dict[str, str],
    *,
    cluster_source: str,
    split_status: str | None = None,
) -> dict[str, Any]:
    validation = validate_split_assignment(groups, assignment, cluster_by_group)
    total_items = sum(len(group["records"]) for group in groups.values())
    all_counts = _count_rows(groups, assignment, lambda _group, _row: True)
    rows: list[dict[str, Any]] = [_table_row("FrameEvidence", all_counts, total_items)]

    rows.append(_section("Question Types"))
    for task in ("FQ1", "FQ2"):
        rows.append(_table_row(
            task,
            _count_rows(groups, assignment, lambda group, _row, task=task: group["question_type"] == task),
            total_items,
        ))

    if all(group.get("scene_category") for group in groups.values()):
        rows.append(_section("Scene Categories"))
        for scene in SCENE_LABELS:
            rows.append(_table_row(
                SCENE_LABELS[scene],
                _count_rows(
                    groups,
                    assignment,
                    lambda group, _row, scene=scene: group.get("scene_category") == scene,
                ),
                total_items,
            ))

    if all(group.get("relation_stratum") for group in groups.values()):
        rows.append(_section("Relation Types"))
        for relation in RELATION_LABELS:
            rows.append(_table_row(
                RELATION_LABELS[relation],
                _count_rows(
                    groups,
                    assignment,
                    lambda group, _row, relation=relation: group.get("relation_stratum") == relation,
                ),
                total_items,
            ))

    rows.append(_section("Controlled Conditions"))
    for frame in FQ1_FRAMES:
        rows.append(_table_row(
            FRAME_LABELS[frame],
            _count_rows(
                groups,
                assignment,
                lambda group, row, frame=frame: (
                    group["question_type"] == "FQ1" and row.get("reference_frame") == frame
                ),
            ),
            total_items,
        ))
    for condition in FQ2_CONDITIONS:
        rows.append(_table_row(
            CONDITION_LABELS[condition],
            _count_rows(
                groups,
                assignment,
                lambda group, row, condition=condition: (
                    group["question_type"] == "FQ2" and row.get("view_condition") == condition
                ),
            ),
            total_items,
        ))

    rows.append(_section("Answer Distribution"))
    for answer in ANSWER_LABELS:
        rows.append(_table_row(
            answer,
            _count_rows(
                groups,
                assignment,
                lambda _group, row, answer=answer: row.get("answer") == answer,
            ),
            total_items,
        ))

    provisional = (
        split_status == "PROVISIONAL_NOT_FOR_PAPER"
        if split_status is not None
        else cluster_source == "exact_image_sha256_connected_components_fallback"
    )
    table_status = (
        split_status
        if split_status in {"PASS", "FROZEN_CANDIDATE"}
        else "PROVISIONAL_NOT_FOR_PAPER" if provisional else "FROZEN_CANDIDATE"
    )
    return {
        "table_status": table_status,
        "title": "Statistics of FrameEvidence",
        "counting_unit": "question_instance",
        "independent_analysis_unit": "relation_group_id",
        "questions_total": total_items,
        "groups_total": len(groups),
        "group_split_counts": validation["group_counts"],
        "task_group_counts": validation["task_group_counts"],
        "cluster_source": cluster_source,
        "cluster_counts": validation["cluster_counts"],
        "rows": rows,
    }


def render_dataset_stats_markdown(statistics: dict[str, Any]) -> str:
    lines = [
        f"# Table 1 — {statistics['title']}",
        "",
        f"**Status:** `{statistics['table_status']}`",
        "",
        "| Category | Train | Dev | Test | Total | Ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in statistics["rows"]:
        if row["row_type"] == "section":
            lines.append(f"| ***{row['label']}*** |  |  |  |  |  |")
        else:
            lines.append(
                f"| {row['label']} | {row['train']:,} | {row['dev']:,} | "
                f"{row['test']:,} | {row['total']:,} | {row['ratio']:.2f} |"
            )
    group_counts = statistics["group_split_counts"]
    lines.extend([
        "",
        (
            "Counts are question instances. The questions form "
            f"{statistics['groups_total']} three-question relation groups, assigned atomically as "
            f"{group_counts['train']}/{group_counts['dev']}/{group_counts['test']} "
            "train/dev/test groups."
        ),
        f"Cluster source: `{statistics['cluster_source']}`.",
    ])
    if statistics["table_status"] == "PROVISIONAL_NOT_FOR_PAPER":
        lines.append(
            "This draft uses exact-image SHA-256 connected components only and must be "
            "regenerated with the restricted scene-cluster ledger before publication."
        )
    return "\n".join(lines) + "\n"


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
    }
    return "".join(replacements.get(character, character) for character in value)


def render_dataset_stats_latex(statistics: dict[str, Any]) -> str:
    group_counts = statistics["group_split_counts"]
    provisional = statistics["table_status"] == "PROVISIONAL_NOT_FOR_PAPER"
    caption = (
        "Statistics of FrameEvidence. Counts are question instances; the 960 questions form "
        f"320 three-question relation groups assigned atomically as {group_counts['train']}/"
        f"{group_counts['dev']}/{group_counts['test']} train/dev/test groups. "
        "Ratio is the percentage of all question instances."
    )
    if provisional:
        caption += (
            " This draft uses exact-image SHA-256 connected components and must be regenerated "
            "with the restricted scene-cluster ledger before publication."
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Train} & \textbf{Dev} & \textbf{Test} & \textbf{Total} & \textbf{Ratio} \\",
        r"\midrule",
    ]
    for row in statistics["rows"]:
        label = _latex_escape(str(row["label"]))
        if row["row_type"] == "section":
            lines.append(rf"\rowcolor{{purple!8}} \multicolumn{{6}}{{c}}{{\textbf{{\textit{{{label}}}}}}} \\")
        else:
            lines.append(
                f"{label} & {row['train']:,} & {row['dev']:,} & {row['test']:,} & "
                f"{row['total']:,} & {row['ratio']:.2f} \\\\"
            )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_latex_escape(caption)}}}",
        r"\label{tab:dataset-statistics}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def write_dataset_stats_tables(statistics: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "table1_dataset_statistics.json"
    csv_path = output_dir / "table1_dataset_statistics.csv"
    markdown_path = output_dir / "table1_dataset_statistics.md"
    latex_path = output_dir / "table1_dataset_statistics.tex"
    write_json(json_path, statistics)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("row_type", "label", "train", "dev", "test", "total", "ratio"),
        )
        writer.writeheader()
        for row in statistics["rows"]:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    markdown_path.write_text(render_dataset_stats_markdown(statistics), encoding="utf-8")
    latex_path.write_text(render_dataset_stats_latex(statistics), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "latex": str(latex_path),
    }
