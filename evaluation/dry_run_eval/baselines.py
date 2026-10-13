from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or sorted(set(y_true) | set(y_pred))
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred, strict=True):
        confusion.setdefault(actual, {label: 0 for label in labels})
        if predicted not in confusion[actual]:
            for row in confusion.values():
                row.setdefault(predicted, 0)
            if predicted not in labels:
                labels.append(predicted)
        confusion[actual][predicted] += 1
    accuracy = sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true) if y_true else 0.0
    f1_values: list[float] = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(actual, {}).get(label, 0) for actual in labels if actual != label)
        fn = sum(confusion.get(label, {}).get(predicted, 0) for predicted in labels if predicted != label)
        denominator = 2 * tp + fp + fn
        f1_values.append((2 * tp / denominator) if denominator else 0.0)
    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "labels": labels,
        "confusion_matrix": confusion,
    }


def majority_predictions(y_train: list[str], n: int) -> list[str]:
    counts = Counter(y_train)
    winner = sorted(counts, key=lambda label: (-counts[label], label))[0]
    return [winner] * n


def _folds(clusters: list[str]) -> list[tuple[list[int], list[int]]]:
    unique = sorted(set(clusters))
    return [
        (
            [index for index, cluster in enumerate(clusters) if cluster != held_out],
            [index for index, cluster in enumerate(clusters) if cluster == held_out],
        )
        for held_out in unique
    ]


def frequency_cv(feature: list[int], target: list[str], clusters: list[str]) -> list[str]:
    predictions = [""] * len(target)
    for train, test in _folds(clusters):
        global_counts = Counter(target[index] for index in train)
        fallback = sorted(global_counts, key=lambda label: (-global_counts[label], label))[0]
        by_value: dict[int, Counter[str]] = defaultdict(Counter)
        for index in train:
            by_value[feature[index]][target[index]] += 1
        for index in test:
            counts = by_value.get(feature[index])
            predictions[index] = (
                sorted(counts, key=lambda label: (-counts[label], label))[0] if counts else fallback
            )
    return predictions


def frequency_fit_predict(feature: list[int], target: list[str]) -> list[str]:
    global_counts = Counter(target)
    fallback = sorted(global_counts, key=lambda label: (-global_counts[label], label))[0]
    by_value: dict[int, Counter[str]] = defaultdict(Counter)
    for value, label in zip(feature, target, strict=True):
        by_value[value][label] += 1
    return [
        sorted(by_value[value], key=lambda label: (-by_value[value][label], label))[0]
        if by_value.get(value) else fallback
        for value in feature
    ]


class PreparedGaussianNB:
    """Precompute fold-specific feature scaling; permutation tests only refit labels."""

    def __init__(self, features: list[list[float]], clusters: list[str]):
        self.features = features
        self.fold_data: list[tuple[list[int], list[int], dict[int, list[float]]]] = []
        for train, test in _folds(clusters):
            x_train = [features[index] for index in train]
            width = len(x_train[0])
            center = [statistics.fmean(row[column] for row in x_train) for column in range(width)]
            scale = []
            for column in range(width):
                values = [row[column] for row in x_train]
                deviation = statistics.pstdev(values)
                scale.append(deviation if math.isfinite(deviation) and deviation >= 1e-9 else 1.0)
            standardized = {
                index: [(features[index][column] - center[column]) / scale[column] for column in range(width)]
                for index in train + test
            }
            self.fold_data.append((train, test, standardized))

    def predict(self, target: list[str]) -> list[str]:
        predictions = [""] * len(target)
        for train, test, standardized in self.fold_data:
            train_target = [target[index] for index in train]
            labels = sorted(set(train_target))
            width = len(standardized[train[0]])
            class_statistics: dict[str, tuple[list[float], list[float], float]] = {}
            for label in labels:
                indices = [index for index in train if target[index] == label]
                count = len(indices)
                sums = [0.0] * width
                squares = [0.0] * width
                for index in indices:
                    vector = standardized[index]
                    for column, value in enumerate(vector):
                        sums[column] += value
                        squares[column] += value * value
                mean = [value / count for value in sums]
                variance = [max(squares[column] / count - mean[column] ** 2, 0.0) + 1e-6 for column in range(width)]
                class_statistics[label] = (mean, variance, math.log(count / len(train)))
            for index in test:
                vector = standardized[index]
                scores = {}
                for label, (mean, variance, prior) in class_statistics.items():
                    scores[label] = prior - 0.5 * sum(
                        math.log(2 * math.pi * variance[column])
                        + ((vector[column] - mean[column]) ** 2) / variance[column]
                        for column in range(width)
                    )
                predictions[index] = sorted(scores, key=lambda label: (-scores[label], label))[0]
        return predictions


def gaussian_nb_cv(features: list[list[float]], target: list[str], clusters: list[str]) -> list[str]:
    return PreparedGaussianNB(features, clusters).predict(target)


def _legacy_gaussian_nb_cv_reference(features: list[list[float]], target: list[str], clusters: list[str]) -> list[str]:
    """Readable reference retained for regression comparison in development."""
    predictions = [""] * len(target)
    for train, test in _folds(clusters):
        x_train = [features[index] for index in train]
        train_target = [target[index] for index in train]
        labels = sorted(set(train_target))
        width = len(x_train[0])
        center = [statistics.fmean(row[column] for row in x_train) for column in range(width)]
        scale = []
        for column in range(width):
            values = [row[column] for row in x_train]
            deviation = statistics.pstdev(values)
            scale.append(deviation if math.isfinite(deviation) and deviation >= 1e-9 else 1.0)
        standardized_train = [
            [(row[column] - center[column]) / scale[column] for column in range(width)]
            for row in x_train
        ]
        standardized_test = [
            [(features[index][column] - center[column]) / scale[column] for column in range(width)]
            for index in test
        ]
        class_statistics: dict[str, tuple[list[float], list[float], float]] = {}
        for label in labels:
            rows = [standardized_train[i] for i, value in enumerate(train_target) if value == label]
            mean = [statistics.fmean(row[column] for row in rows) for column in range(width)]
            variance = [statistics.pvariance([row[column] for row in rows]) + 1e-6 for column in range(width)]
            prior = math.log(len(rows) / len(x_train))
            class_statistics[label] = (mean, variance, prior)
        for local_index, vector in zip(test, standardized_test, strict=True):
            scores = {}
            for label, (mean, variance, prior) in class_statistics.items():
                scores[label] = prior - 0.5 * sum(
                    math.log(2 * math.pi * variance[column])
                    + ((vector[column] - mean[column]) ** 2) / variance[column]
                    for column in range(width)
                )
            predictions[local_index] = sorted(scores, key=lambda label: (-scores[label], label))[0]
    return predictions


def _permute_group_bundles(target: list[str], clusters: list[str], rng: random.Random) -> list[str]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)
    ordered = sorted(members)
    donors = ordered.copy()
    rng.shuffle(donors)
    permuted = target.copy()
    for receiver, donor in zip(ordered, donors, strict=True):
        receiver_indices = members[receiver]
        donor_indices = members[donor]
        if len(receiver_indices) != len(donor_indices):
            raise ValueError("组级置换要求每个 cluster 具有相同记录数")
        for receiver_index, donor_index in zip(receiver_indices, donor_indices, strict=True):
            permuted[receiver_index] = target[donor_index]
    return permuted


def _permute_within_groups(target: list[str], clusters: list[str], rng: random.Random) -> list[str]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        members[cluster].append(index)
    permuted = target.copy()
    for indices in members.values():
        values = [target[index] for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values, strict=True):
            permuted[index] = value
    return permuted


def permutation_pvalue(
    predictor: Callable[[list[str]], list[str]],
    target: list[str],
    clusters: list[str],
    observed_accuracy: float,
    permutations: int,
    seed: int,
    within_groups: bool = False,
) -> float:
    if not target or permutations <= 0:
        return float("nan")
    rng = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = _permute_within_groups(target, clusters, rng) if within_groups else _permute_group_bundles(target, clusters, rng)
        predicted = predictor(permuted)
        accuracy = sum(a == b for a, b in zip(permuted, predicted, strict=True)) / len(permuted)
        if accuracy >= observed_accuracy - 1e-12:
            exceed += 1
    return (exceed + 1) / (permutations + 1)


def meta_feature_vector(record: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> list[float]:
    rows = [metadata[path] for path in record.get("images", []) if path in metadata]
    if not rows:
        return [0.0] * 16
    numeric_names = ("width", "height", "aspect_ratio", "pixels", "bytes")
    values: list[float] = []
    for name in numeric_names:
        array = [float(row[name]) for row in rows]
        values.extend([statistics.fmean(array), min(array), max(array)])
    jpeg_fraction = sum(str(row.get("format", "")).upper() == "JPEG" for row in rows) / len(rows)
    values.append(jpeg_fraction)
    return values


def shortcut_baselines(
    records: list[dict[str, Any]],
    image_metadata: dict[str, dict[str, Any]],
    permutations: int = 1_000,
    seed: int = 20260722,
) -> dict[str, Any]:
    fq2 = sorted(
        [record for record in records if record.get("question_type") == "FQ2"],
        key=lambda row: (str(row["relation_group_id"]), str(row["view_condition"])),
    )
    groups = [str(record["relation_group_id"]) for record in fq2]
    clusters = groups
    unit = "relation_group_id_descriptive"
    counts = [len(record.get("images", [])) for record in fq2]
    meta = [meta_feature_vector(record, image_metadata) for record in fq2]
    prepared_meta = PreparedGaussianNB(meta, clusters)
    results: dict[str, Any] = {"resampling_unit": unit, "permutations": permutations, "tasks": {}}
    for target_name, target in (
        ("view_condition", [str(record["view_condition"]) for record in fq2]),
        ("answer", [str(record["answer"]) for record in fq2]),
    ):
        labels = sorted(set(target))
        majority = majority_predictions(target, len(target))
        count_pred = frequency_cv(counts, target, clusters)
        count_resub = frequency_fit_predict(counts, target)
        meta_pred = prepared_meta.predict(target)
        majority_metrics = classification_metrics(target, majority, labels.copy())
        count_metrics = classification_metrics(target, count_pred, labels.copy())
        meta_metrics = classification_metrics(target, meta_pred, labels.copy())
        count_p = permutation_pvalue(
            lambda permuted: frequency_cv(counts, permuted, clusters),
            target,
            clusters,
            count_metrics["accuracy"],
            permutations,
            seed + (11 if target_name == "answer" else 3),
            target_name == "view_condition",
        )
        meta_p = permutation_pvalue(
            prepared_meta.predict,
            target,
            clusters,
            meta_metrics["accuracy"],
            permutations,
            seed + (29 if target_name == "answer" else 17),
            target_name == "view_condition",
        )
        for metrics, p_value in ((count_metrics, count_p), (meta_metrics, meta_p)):
            metrics["majority_accuracy"] = majority_metrics["accuracy"]
            metrics["accuracy_lift"] = metrics["accuracy"] - majority_metrics["accuracy"]
            metrics["permutation_p_value"] = p_value
            metrics["risk"] = (
                "HIGH_RISK"
                if metrics["accuracy_lift"] >= 0.10 and p_value < 0.05
                else "REVIEW_REQUIRED" if metrics["accuracy_lift"] > 0 else "LOW_OBSERVED_RISK"
            )
        count_resub_metrics = classification_metrics(target, count_resub, labels.copy())
        value_targets: dict[int, Counter[str]] = defaultdict(Counter)
        for value, label in zip(counts, target, strict=True):
            value_targets[value][label] += 1
        exclusive = {
            str(value): {"target": next(iter(counter)), "support": sum(counter.values())}
            for value, counter in sorted(value_targets.items())
            if len(counter) == 1 and sum(counter.values()) >= 2
        }
        count_metrics["descriptive_resubstitution_accuracy"] = count_resub_metrics["accuracy"]
        count_metrics["descriptive_resubstitution_lift"] = count_resub_metrics["accuracy"] - majority_metrics["accuracy"]
        count_metrics["exclusive_count_values_support_ge_2"] = exclusive
        if (
            target_name == "view_condition"
            and count_metrics["descriptive_resubstitution_lift"] >= 0.10
            and exclusive
        ):
            count_metrics["risk"] = "HIGH_RISK"
            count_metrics["risk_basis"] = "结构性诊断：部分图片数量唯一对应条件且全样本经验映射较多数类提升至少 10 个百分点；CV 受 12 组小样本和近平票影响。"
        results["tasks"][target_name] = {
            "majority": majority_metrics,
            "count_only": count_metrics,
            "meta_only": meta_metrics,
        }
    risks = [
        metrics[baseline]["risk"]
        for metrics in results["tasks"].values()
        for baseline in ("count_only", "meta_only")
    ]
    results["overall_risk"] = "HIGH_RISK" if "HIGH_RISK" in risks else "REVIEW_REQUIRED" if "REVIEW_REQUIRED" in risks else "LOW_OBSERVED_RISK"
    return results
