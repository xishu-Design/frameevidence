"""Offline scoring of sanitized FrameEvidence results; Python standard library only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

LABELS = "ABCD"
FAMILIES = ("internvl3-8b", "qwen25-vl-7b")
SEEDS = (20260722, 20260723, 20260724)
SETTINGS = ("base-multiseed-control", *(f"lora-seed{s}" for s in SEEDS))
MODEL_IDS = tuple(f"{family}-{setting}" for family in FAMILIES for setting in SETTINGS)
INCOMPLETE_MODEL = "qwen25-vl-7b-base-multiseed-control"
ORDINARY_METRICS = ("macro_precision", "macro_recall", "macro_f1", "accuracy", "FQ1_GEM", "NTS", "DIS")
CIRCULAR_METRICS = ("vanilla", "mean_rotation", "circular", "consistency", "drop")
# This order is the manuscript main table, never the multiseed run roster.
MAIN_CONFIGS = (
    ("qwen25-vl-7b-base", "Qwen2.5-VL-7B", "Base"),
    ("internvl3-8b-base", "InternVL3-8B", "Base"),
    ("qwen3-vl-8b-base", "Qwen3-VL-8B", "Base"),
    ("llava-onevision-7b-base", "LLaVA-OneVision-7B", "Base"),
    ("minicpm-v-4_5-base", "MiniCPM-V 4.5", "Base"),
    ("mistral-small-3_2-24b-base", "Mistral Small 3.2 24B", "Base"),
    ("qwen25-vl-32b-base", "Qwen2.5-VL-32B", "Base"),
    ("internvl35-14b-base", "InternVL3.5-14B", "Base"),
    ("phi4-multimodal-base", "Phi-4-Multimodal", "Base"),
    ("kimi-vl-a3b-base", "Kimi-VL-A3B", "Base"),
    ("qwen25-vl-7b-lora", "Qwen2.5-VL-7B", "LoRA"),
    ("internvl3-8b-lora", "InternVL3-8B", "LoRA"),
    *((f"gpt-5.6-{shot}shot", "GPT-5.6", f"{shot}-shot") for shot in range(4)),
    *((f"claude-fable-5.1-chat-{shot}shot", "Claude Fable 5.1", f"{shot}-shot") for shot in range(4)),
)
MAIN_MODEL_IDS = tuple(row[0] for row in MAIN_CONFIGS)
MAPPING_FIELDS = ("eval_id", "relation_group_id", "evaluation_mode", "question_type", "reference_frame", "view_condition", "gold_answer")
PREDICTION_FIELDS = ("eval_id", "model_id", "evaluation_mode", "status", "parsed_answer", "parse_status")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed(rows):
    result = {}
    for row in rows:
        eid = row["eval_id"]
        require(eid not in result, f"Duplicate evaluation identifier: {eid}")
        result[eid] = row
    return result


def valid_answer(prediction):
    answer = prediction.get("parsed_answer")
    return answer if prediction.get("status") == "OK" and answer in tuple(LABELS) else None


def coverage(predictions):
    statuses = Counter(row.get("status", "MISSING_STATUS") for row in predictions.values())
    invalid_reasons = Counter()
    for row in predictions.values():
        if valid_answer(row) is None:
            reason = row.get("status", "MISSING_STATUS")
            invalid_reasons["INVALID_ANSWER" if reason == "OK" else reason] += 1
    invalid = sum(invalid_reasons.values())
    return {"records": len(predictions), "valid": len(predictions) - invalid,
            "invalid": invalid, "status_counts": dict(sorted(statuses.items())),
            "invalid_reasons": dict(sorted(invalid_reasons.items()))}


def score_ordinary(mapping, predictions):
    require(set(mapping) == set(predictions), "Ordinary prediction/mapping ID sets differ")
    require(bool(mapping), "Empty ordinary mapping")
    confusion = {gold: {pred: 0 for pred in (*LABELS, "INVALID")} for gold in LABELS}
    groups = defaultdict(list)
    for eid, item in mapping.items():
        require(item["gold_answer"] in tuple(LABELS), f"Invalid gold: {eid}")
        require(item["evaluation_mode"] == predictions[eid]["evaluation_mode"] == "all-images", f"Mode mismatch: {eid}")
        answer = valid_answer(predictions[eid])
        confusion[item["gold_answer"]][answer or "INVALID"] += 1
        groups[item["relation_group_id"]].append((item, answer == item["gold_answer"]))
    class_scores = []
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[gold][label] for gold in LABELS)
        support = sum(confusion[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_scores.append((precision, recall, f1))
    joint = {metric: [0, 0] for metric in ("FQ1_GEM", "NTS", "DIS")}
    for gid, variants in groups.items():
        require(len(variants) == 3, f"Incomplete triplet: {gid}")
        families = {item["question_type"] for item, _ in variants}
        if families == {"FQ1"}:
            require({item["reference_frame"] for item, _ in variants} == {"camera", "object_intrinsic", "world"}, f"Bad reference frames: {gid}")
            joint["FQ1_GEM"][0] += int(all(ok for _, ok in variants))
            joint["FQ1_GEM"][1] += 1
        else:
            require(families == {"FQ2"}, f"Mixed or unknown question family: {gid}")
            conditions = {item["view_condition"]: ok for item, ok in variants}
            require(set(conditions) == {"one_valid_mss", "remove_required_view", "hard_distractor_added"}, f"Bad interventions: {gid}")
            for metric, condition in (("NTS", "remove_required_view"), ("DIS", "hard_distractor_added")):
                joint[metric][0] += int(conditions["one_valid_mss"] and conditions[condition])
                joint[metric][1] += 1
    correct = sum(confusion[label][label] for label in LABELS)
    return {"n": len(mapping), "correct": correct, "accuracy": correct / len(mapping),
            **{name: sum(row[i] for row in class_scores) / 4 for i, name in enumerate(ORDINARY_METRICS[:3])},
            "joint": joint, "coverage": coverage(predictions), "confusion": confusion}


def circular_bundles(mapping, ordinary):
    bundles = defaultdict(dict)
    for eid, row in mapping.items():
        sid, rotation = row["source_eval_id"], row["circular_rotation"]
        require(sid in ordinary, f"Unknown Circular source: {sid}")
        require(type(rotation) is int and rotation in range(4), f"Invalid rotation: {eid}")
        require(rotation not in bundles[sid], f"Duplicate source rotation: {sid}/{rotation}")
        expected = {new: LABELS[(i + rotation) % 4] for i, new in enumerate(LABELS)}
        require(row["option_map_new_to_original"] == expected, f"Invalid option permutation: {eid}")
        require(row["gold_answer"] in tuple(LABELS), f"Invalid Circular gold: {eid}")
        require(expected[row["gold_answer"]] == ordinary[sid]["gold_answer"], f"Incorrect gold remapping: {eid}")
        require(row["evaluation_mode"] == "all-images", f"Unexpected Circular mode: {eid}")
        bundles[sid][rotation] = eid
    require(set(bundles) == set(ordinary), "Circular sources differ from ordinary items")
    require(all(set(bundle) == set(range(4)) for bundle in bundles.values()), "Incomplete four-rotation bundle")
    return bundles


def score_circular(mapping, predictions, ordinary):
    require(set(mapping) == set(predictions), "Circular prediction/mapping ID sets differ")
    bundles = circular_bundles(mapping, ordinary)
    require(bool(bundles), "Empty Circular mapping")
    require(all(row["evaluation_mode"] == "all-images" for row in predictions.values()), "Circular prediction mode mismatch")
    correct = {eid: valid_answer(predictions[eid]) == row["gold_answer"] for eid, row in mapping.items()}
    rotation_correct = [sum(correct[bundle[r]] for bundle in bundles.values()) for r in range(4)]
    strict = sum(all(correct[eid] for eid in bundle.values()) for bundle in bundles.values())
    consistent = 0
    for bundle in bundles.values():
        answers = [mapping[eid]["option_map_new_to_original"].get(valid_answer(predictions[eid])) for eid in bundle.values()]
        consistent += int(None not in answers and len(set(answers)) == 1)
    n = len(bundles)
    return {"source_items": n, "rotation_items": len(mapping), "rotation_correct": rotation_correct,
            "circular_correct": strict, "consistent_items": consistent,
            "vanilla": rotation_correct[0] / n * 100,
            "mean_rotation": sum(rotation_correct) / len(mapping) * 100,
            "circular": strict / n * 100, "consistency": consistent / n * 100,
            "drop": (rotation_correct[0] - strict) / n * 100, "coverage": coverage(predictions)}


def ordinary_percent(row, metric):
    if metric in row:
        return row[metric] * 100
    numerator, denominator = row["joint"][metric]
    require(denominator > 0, f"Empty denominator: {metric}")
    return numerator / denominator * 100


def seed_summary(models, circular=False):
    result = {}
    for family in FAMILIES:
        selected = [row for row in models if row["model_id"] in {f"{family}-lora-seed{s}" for s in SEEDS}]
        require(len(selected) == 3 and len({row["model_id"] for row in selected}) == 3, f"Expected three distinct LoRA seeds: {family}")
        result[family] = {}
        for metric in CIRCULAR_METRICS if circular else ORDINARY_METRICS:
            values = [row[metric] if circular else ordinary_percent(row, metric) for row in selected]
            result[family][metric] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    return result


def assert_matches(actual, expected, location="reference"):
    """Compare recorded reference fields, allowing only floating-point roundoff."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Type mismatch: {location}")
        for key, value in expected.items():
            require(key in actual, f"Missing reference field: {location}/{key}")
            assert_matches(actual[key], value, f"{location}/{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"List mismatch: {location}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            assert_matches(left, right, f"{location}/{i}")
    elif isinstance(expected, float):
        require(isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"Numeric mismatch: {location}: {actual} != {expected}")
    else:
        require(actual == expected, f"Value mismatch: {location}: {actual} != {expected}")


def verify_integrity(bundle):
    manifest = read_json(bundle / "bundle_manifest.json")
    for relative, expected in manifest["sha256"].items():
        target = (bundle / relative).resolve()
        require(target.is_relative_to(bundle.resolve()), f"Unsafe manifest path: {relative}")
        require(target.is_file() and sha256(target) == expected, f"Bundle digest mismatch: {relative}")
    return len(manifest["sha256"])


def reproduce_primary_main(bundle, metadata):
    """Score the primary 20-row cohort independently of multiseed controls."""
    configs = [row for row in metadata["configs"] if row["experiment"] == "primary_main"]
    require(len(configs) == 20 and {row["model_id"] for row in configs} == set(MAIN_MODEL_IDS), "Primary main configuration roster mismatch")
    configs = {row["model_id"]: row for row in configs}
    mapping_path = bundle / "inputs/primary_main_items.jsonl"
    require(sha256(mapping_path) == metadata["cohorts"]["primary_main"]["packaged_mapping_sha256"], "Primary main mapping binding mismatch")
    mapping = indexed(read_jsonl(mapping_path))
    require(len(mapping) == 288, "Expected 288 primary main questions")
    models = []
    for mid, display_model, setting in MAIN_CONFIGS:
        config = configs[mid]
        require(config["run_cohort"] == "primary_main", f"Primary main cohort mismatch: {mid}")
        path = bundle / f"inputs/primary_main/{mid}.jsonl"
        require(sha256(path) == config["packaged_prediction_sha256"], f"Primary main prediction binding mismatch: {mid}")
        pred = indexed(read_jsonl(path))
        require(all(row["model_id"] == mid for row in pred.values()), f"Primary main model mismatch: {mid}")
        row = score_ordinary(mapping, pred)
        require(all(pair[1] == 48 for pair in row["joint"].values()), "Expected 48 primary main groups for each joint metric")
        expected_invalid = {"CALL_ERROR": 10} if mid == "llava-onevision-7b-base" else {}
        require(row["coverage"]["invalid_reasons"] == expected_invalid, f"Primary main invalid coverage changed: {mid}")
        row.update(model_id=mid, display_model=display_model, setting=setting, run_cohort="primary_main")
        row["table_main_display_percent"] = {metric: f"{ordinary_percent(row, metric):.2f}" for metric in ORDINARY_METRICS}
        models.append(row)
    return {"models": models}


def reproduce(bundle, verify=True):
    hashes = verify_integrity(bundle) if verify else None
    metadata = read_json(bundle / "metadata.json")
    ordinary = indexed(read_jsonl(bundle / "inputs/ordinary_items.jsonl"))
    circular = indexed(read_jsonl(bundle / "inputs/circular_items.jsonl"))
    require(len(ordinary) == 288, "Expected 288 ordinary questions")
    require(len(circular) == 1152, "Expected 1,152 Circular rotations")
    ordinary_models, circular_models, incomplete = [], [], []
    for mid in MODEL_IDS:
        pred = indexed(read_jsonl(bundle / f"inputs/ordinary/{mid}.jsonl"))
        require(all(row["model_id"] == mid for row in pred.values()), f"Model mismatch: {mid}")
        row = score_ordinary(ordinary, pred)
        require(all(pair[1] == 48 for pair in row["joint"].values()), "Expected 48 groups for each joint metric")
        require(row["coverage"]["invalid"] == 0, f"Ordinary run unexpectedly invalid: {mid}")
        row.update(model_id=mid, family=next(f for f in FAMILIES if mid.startswith(f + "-")))
        row["setting"] = mid[len(row["family"]) + 1:]
        ordinary_models.append(row)
        pred_c = indexed(read_jsonl(bundle / f"inputs/circular/{mid}.jsonl"))
        require(all(row["model_id"] == mid for row in pred_c.values()), f"Circular model mismatch: {mid}")
        circ = score_circular(circular, pred_c, ordinary)
        circ["model_id"] = mid
        bundles = circular_bundles(circular, ordinary)
        circ["ordinary_vs_circular_r0"] = {"same_predictions": sum(valid_answer(pred[sid]) == valid_answer(pred_c[bundle[0]]) and pred[sid]["status"] == pred_c[bundle[0]]["status"] for sid, bundle in bundles.items()),
                                          "total": 288, "ordinary_correct": row["correct"]}
        if mid == INCOMPLETE_MODEL:
            require(circ["coverage"]["valid"] == 706 and circ["coverage"]["invalid_reasons"] == {"CALL_ERROR": 446}, "Incomplete Qwen Base evidence changed")
            circ["evidence_status"] = "incomplete_located_copy_diagnostic_only"
            incomplete.append(circ)
        else:
            require(circ["coverage"]["invalid"] == 0, f"Circular run unexpectedly invalid: {mid}")
            circ["evidence_status"] = "complete_local_predictions_recomputed"
            circular_models.append(circ)
    result = {"ordinary": {"models": ordinary_models, "families": seed_summary(ordinary_models)},
              "circular_complete": {"models": circular_models, "families": seed_summary(circular_models, circular=True)},
              "circular_incomplete_diagnostic": {"models": incomplete},
              "author_confirmed_summary": read_json(bundle / "author_confirmed_summary.json")}
    if metadata["schema_version"] == 2:
        result["primary_main"] = reproduce_primary_main(bundle, metadata)
    else:
        require(metadata["schema_version"] == 1, "Unsupported bundle schema")
    assert_matches(result, read_json(bundle / "expected_metrics.json"))
    result["verification"] = {"status": "PASS", "reference_match": True, "float_absolute_tolerance": 1e-12,
                              "bundle_hashes_checked": hashes, "ordinary_models": 8, "complete_circular_models": 7,
                              "author_summary_independently_recomputed": False}
    if "primary_main" in result:
        result["verification"].update(primary_main_models=20, primary_main_predictions=5760,
                                      primary_main_invalid_predictions=10, main_table_display_value_checks=140,
                                      separate_run_cohorts=True)
    return result


def write_results(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "recomputed_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tables = (("ordinary", ORDINARY_METRICS), ("circular_complete", CIRCULAR_METRICS), ("circular_incomplete_diagnostic", CIRCULAR_METRICS))
    if "primary_main" in result:
        tables = (("primary_main", ORDINARY_METRICS), *tables)
    for name, metrics in tables:
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model_id", *(metric + ("_pp" if metric == "drop" else "_percent") for metric in metrics), "valid", "invalid"])
            for row in result[name]["models"]:
                values = [ordinary_percent(row, metric) if name in ("ordinary", "primary_main") else row[metric] for metric in metrics]
                writer.writerow([row["model_id"], *(f"{value:.12f}" for value in values), row["coverage"]["valid"], row["coverage"]["invalid"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, help="Optional output directory; source bundle remains read-only")
    args = parser.parse_args()
    try:
        result = reproduce(args.bundle.resolve())
        if args.out:
            require(not args.out.resolve().is_relative_to(args.bundle.resolve()), "Output must be outside the source bundle")
            write_results(result, args.out.resolve())
        print(json.dumps(result["verification"], indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
