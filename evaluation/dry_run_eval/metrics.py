from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_json
from .runner import parse_answer

ANSWER_LABELS = ["A", "B", "C", "D", "INVALID"]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _matrix(pairs: list[tuple[str | None, str | None]]) -> dict[str, dict[str, int]]:
    matrix = {first: {second: 0 for second in ANSWER_LABELS} for first in ANSWER_LABELS}
    for first, second in pairs:
        matrix[first if first in ANSWER_LABELS else "INVALID"][second if second in ANSWER_LABELS else "INVALID"] += 1
    return matrix


def _mcnemar(first: list[bool], second: list[bool]) -> dict[str, Any]:
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(not a and b for a, b in zip(first, second, strict=True))
    discordant = first_only + second_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(first_only, second_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p_descriptive": p_value,
        "note": "relation_group_id 题组层描述性结果；Dry Run 不作推断统计。",
    }


def _transition_for_groups(
    grouped: dict[str, list[dict[str, Any]]],
    first_field: str,
    first_value: str,
    second_field: str,
    second_value: str,
) -> dict[str, Any]:
    pairs: list[tuple[str | None, str | None]] = []
    first_correct: list[bool] = []
    second_correct: list[bool] = []
    for rows in grouped.values():
        first = next((row for row in rows if row[first_field] == first_value), None)
        second = next((row for row in rows if row[second_field] == second_value), None)
        if first is None or second is None:
            continue
        pairs.append((first["predicted_answer"], second["predicted_answer"]))
        first_correct.append(first["correct"])
        second_correct.append(second["correct"])
    return {
        "n_groups": len(pairs),
        "answer_matrix": _matrix(pairs),
        "paired_correctness": _mcnemar(first_correct, second_correct),
    }


def score_mode(
    mapping: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    mode: str,
    excluded_groups: set[str] | None = None,
) -> dict[str, Any]:
    excluded_groups = excluded_groups or set()
    expected = [
        row for row in mapping
        if row["evaluation_mode"] == mode and row["relation_group_id"] not in excluded_groups
    ]
    scored: list[dict[str, Any]] = []
    for row in expected:
        prediction = predictions.get(row["eval_id"], {})
        answer = prediction.get("parsed_answer") or prediction.get("answer")
        if answer not in {"A", "B", "C", "D"} and prediction.get("raw_response"):
            answer, _ = parse_answer(str(prediction["raw_response"]))
        scored.append({
            **row,
            "predicted_answer": answer if answer in {"A", "B", "C", "D"} else None,
            "correct": answer == row["gold_answer"],
            "prediction_status": prediction.get("status", "MISSING"),
        })
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[row["relation_group_id"]].append(row)
    fq1 = {group: rows for group, rows in groups.items() if rows and rows[0]["question_type"] == "FQ1"}
    fq2 = {group: rows for group, rows in groups.items() if rows and rows[0]["question_type"] == "FQ2"}
    fq1_success = {group: int(len(rows) == 3 and all(row["correct"] for row in rows)) for group, rows in fq1.items()}
    nts_success: dict[str, int] = {}
    dis_success: dict[str, int] = {}
    mss_correct_conditioned: list[tuple[bool, bool, bool]] = []
    distractor_consistency: list[int] = []
    distractor_categories: Counter[str] = Counter()
    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"NTS": [], "DIS": []})
    for group, rows in fq2.items():
        by_condition = {row["view_condition"]: row for row in rows}
        mss = by_condition.get("one_valid_mss")
        deleted = by_condition.get("remove_required_view")
        distractor = by_condition.get("hard_distractor_added")
        nts = bool(mss and deleted and mss["correct"] and deleted["correct"])
        dis = bool(mss and distractor and mss["correct"] and distractor["correct"])
        nts_success[group] = int(nts)
        dis_success[group] = int(dis)
        if mss:
            mss_correct_conditioned.append((
                mss["correct"],
                bool(deleted and deleted["correct"]),
                bool(distractor and distractor["correct"]),
            ))
            key = str(mss["image_count"])
            strata[key]["NTS"].append(int(nts))
            strata[key]["DIS"].append(int(dis))
        if mss and distractor:
            first = mss["predicted_answer"]
            second = distractor["predicted_answer"]
            if first is None or second is None:
                distractor_categories["invalid_or_missing"] += 1
            else:
                distractor_consistency.append(int(first == second))
                if mss["correct"] and distractor["correct"]:
                    distractor_categories["correct_correct"] += 1
                elif mss["correct"]:
                    distractor_categories["correct_wrong"] += 1
                elif distractor["correct"]:
                    distractor_categories["wrong_correct"] += 1
                elif first == second:
                    distractor_categories["wrong_wrong_same"] += 1
                else:
                    distractor_categories["wrong_wrong_changed"] += 1

    accuracy = _rate(sum(row["correct"] for row in scored), len(scored))
    by_frame = {}
    for frame in ("camera", "object_intrinsic", "world"):
        rows = [row for row in scored if row["question_type"] == "FQ1" and row["reference_frame"] == frame]
        by_frame[frame] = {"n": len(rows), "accuracy": _rate(sum(row["correct"] for row in rows), len(rows))}
    by_condition = {}
    for condition in ("one_valid_mss", "remove_required_view", "hard_distractor_added"):
        rows = [row for row in scored if row["question_type"] == "FQ2" and row["view_condition"] == condition]
        by_condition[condition] = {"n": len(rows), "accuracy": _rate(sum(row["correct"] for row in rows), len(rows))}
    conditioned = [value for value in mss_correct_conditioned if value[0]]
    conditional_delete = _rate(sum(value[1] for value in conditioned), len(conditioned))
    conditional_distractor = _rate(sum(value[2] for value in conditioned), len(conditioned))

    fq1_value = _rate(sum(fq1_success.values()), len(fq1_success))
    nts_value = _rate(sum(nts_success.values()), len(nts_success))
    dis_value = _rate(sum(dis_success.values()), len(dis_success))
    frame_product = math.prod(by_frame[frame]["accuracy"] or 0.0 for frame in ("camera", "object_intrinsic", "world"))
    nts_independent = (by_condition["one_valid_mss"]["accuracy"] or 0.0) * (by_condition["remove_required_view"]["accuracy"] or 0.0)
    dis_independent = (by_condition["one_valid_mss"]["accuracy"] or 0.0) * (by_condition["hard_distractor_added"]["accuracy"] or 0.0)
    fq1_grouped = fq1
    fq2_grouped = fq2
    transitions = {
        "fq1_camera_to_object_intrinsic": _transition_for_groups(fq1_grouped, "reference_frame", "camera", "reference_frame", "object_intrinsic"),
        "fq1_camera_to_fixed_scene": _transition_for_groups(fq1_grouped, "reference_frame", "camera", "reference_frame", "world"),
        "fq1_object_intrinsic_to_fixed_scene": _transition_for_groups(fq1_grouped, "reference_frame", "object_intrinsic", "reference_frame", "world"),
        "fq2_mss_to_delete": _transition_for_groups(fq2_grouped, "view_condition", "one_valid_mss", "view_condition", "remove_required_view"),
        "fq2_mss_to_distractor": _transition_for_groups(fq2_grouped, "view_condition", "one_valid_mss", "view_condition", "hard_distractor_added"),
    }
    invalid = [row for row in scored if row["predicted_answer"] is None]
    return {
        "evaluation_mode": mode,
        "coverage": {
            "expected": len(expected),
            "received": sum(row["eval_id"] in predictions for row in expected),
            "valid_answers": sum(row["predicted_answer"] is not None for row in scored),
            "invalid_or_missing": len(invalid),
        },
        "item_accuracy": accuracy,
        "main_metrics": {
            "FQ1_GEM": {"numerator": sum(fq1_success.values()), "denominator": len(fq1_success), "value": fq1_value},
            "NTS": {"numerator": sum(nts_success.values()), "denominator": len(nts_success), "value": nts_value},
            "DIS": {"numerator": sum(dis_success.values()), "denominator": len(dis_success), "value": dis_value},
        },
        "fq1_by_reference_frame": by_frame,
        "fq2_by_view_condition": by_condition,
        "conditional_metrics": {
            "mss_correct_denominator": len(conditioned),
            "P_delete_insufficient_given_mss_correct": conditional_delete,
            "P_distractor_correct_given_mss_correct": conditional_distractor,
        },
        "prediction_consistency": {
            "MSS_to_distractor": {
                "numerator": sum(distractor_consistency),
                "denominator": len(distractor_consistency),
                "value": _rate(sum(distractor_consistency), len(distractor_consistency)),
                "outcome_categories": {
                    key: distractor_categories.get(key, 0)
                    for key in (
                        "correct_correct",
                        "wrong_wrong_same",
                        "correct_wrong",
                        "wrong_correct",
                        "wrong_wrong_changed",
                        "invalid_or_missing",
                    )
                },
            },
        },
        "random_expectations": {
            "item_independent_uniform": {
                "FQ1_GEM": 1 / 64,
                "NTS": 1 / 16,
                "DIS": 1 / 16,
                "MSS_to_distractor_prediction_consistency": 1 / 4,
            },
            "group_consistent_uniform": {
                "FQ1_GEM": 0.0,
                "NTS": 0.0,
                "DIS": 1 / 4,
                "MSS_to_distractor_prediction_consistency": 1.0,
            },
        },
        "product_of_marginals_reference": {
            "FQ1_GEM": frame_product,
            "NTS": nts_independent,
            "DIS": dis_independent,
        },
        "transitions": transitions,
        "by_mss_image_count": {
            key: {
                metric: {"n": len(values), "value": _rate(sum(values), len(values))}
                for metric, values in metric_values.items()
            }
            for key, metric_values in sorted(strata.items(), key=lambda item: int(item[0]))
        },
        "analysis_type": "relation_group_id_descriptive",
        "invalid_examples": [
            {"eval_id": row["eval_id"], "sample_id": row["sample_id"], "status": row["prediction_status"]}
            for row in invalid[:20]
        ],
    }


def _write_long_csv(path: Path, models: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model_id", "evaluation_mode", "metric", "numerator", "denominator", "value"])
        writer.writeheader()
        for model in models:
            for mode, result in model.get("results", {}).items():
                for metric, value in result["main_metrics"].items():
                    writer.writerow({
                        "model_id": model["model_id"],
                        "evaluation_mode": mode,
                        "metric": metric,
                        "numerator": value["numerator"],
                        "denominator": value["denominator"],
                        "value": value["value"],
                    })


def score_run(
    run_dir: Path,
) -> dict[str, Any]:
    mapping = read_jsonl(run_dir / "private" / "evaluation_mapping.jsonl")
    prediction_dir = run_dir / "predictions"
    model_files = sorted(path for path in prediction_dir.glob("*.jsonl") if path.name != "human_responses.jsonl") if prediction_dir.is_dir() else []
    models: list[dict[str, Any]] = []
    for path in model_files:
        rows = read_jsonl(path)
        if not rows:
            continue
        model_id = str(rows[0].get("model_id", path.stem))
        selection_path = path.with_name(f"{path.stem}.selection.json")
        selection: dict[str, Any] | None = None
        model_mapping = mapping
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            # A full run also writes a selection manifest so the exact requested
            # modes remain auditable. Only a genuinely limited smoke/group run
            # should narrow the scoring denominator; otherwise a later full
            # baseline run would accidentally hide previously completed modes.
            is_limited = selection.get("limit") is not None or selection.get("limit_groups") is not None
            selected_eval_ids = {str(value) for value in selection.get("selected_eval_ids", [])}
            if is_limited and selected_eval_ids:
                model_mapping = [row for row in mapping if str(row["eval_id"]) in selected_eval_ids]
        predictions = {str(row["eval_id"]): row for row in rows}
        available_modes = {
            str(row.get("evaluation_mode"))
            for row in rows
            if row.get("evaluation_mode") in {"all-images", "text-only", "single-image"}
        }
        results = {
            mode: score_mode(model_mapping, predictions, mode)
            for mode in ("all-images", "text-only", "single-image")
            if mode in available_modes
        }
        models.append({
            "model_id": model_id,
            "prediction_file": str(path.resolve()),
            "selection": selection,
            "results": results,
        })
    result = {
        "status": "PASS" if models else "NOT_RUN",
        "evaluation_profile": "dry_run_descriptive",
        "result_label": "exploratory_dry_run",
        "models": models,
        "analysis_unit": "relation_group_id",
        "analysis_type": "descriptive_only",
        "inferential_statistics": "not_computed",
    }
    write_json(run_dir / "metrics.json", result)
    _write_long_csv(run_dir / "metrics_long.csv", models)
    return result
