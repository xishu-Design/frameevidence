"""Recompute FrameEvidence main, multiseed and GPT controls without network access."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_scorer", HERE / "frozen_bundle/scorer.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

def pair_counts(gold, left, right):
    scorer.require(len(gold) == len(left) == len(right) and len(gold) > 0, "Pair lengths differ or empty")
    valid = lambda x: x in tuple(scorer.LABELS)
    return {
        "n": len(gold),
        "at_least_one_correct": sum(a == g or b == g for g, a, b in zip(gold, left, right)),
        "both_correct": sum(a == g and b == g for g, a, b in zip(gold, left, right)),
        "consistency": sum(valid(a) and valid(b) and a == b for a, b in zip(left, right)),
        "consistently_wrong": sum(valid(a) and valid(b) and a == b and a != g for g, a, b in zip(gold, left, right)),
    }

def reproduce_gpt():
    root = HERE / "gpt_controls"
    receipt = scorer.read_json(root / "export_receipt.json")
    for name, item in receipt["files"].items():
        target = (root / name).resolve()
        scorer.require(target.is_relative_to(root.resolve()), "Unsafe GPT input path")
        scorer.require(scorer.sha256(target) == item["export_sha256"], "GPT input changed: " + name)
    ordinary = scorer.indexed(scorer.read_jsonl(root / "ordinary_items.jsonl"))
    mapping = scorer.indexed(scorer.read_jsonl(root / "circular_items.jsonl"))
    circular = scorer.indexed(scorer.read_jsonl(root / "circular_predictions.jsonl"))
    repeat = scorer.indexed(scorer.read_jsonl(root / "repeat_predictions.jsonl"))
    scorer.require(len(ordinary) == len(repeat) == 288 and len(mapping) == len(circular) == 1152, "GPT coverage count mismatch")
    for values in (circular, repeat):
        scorer.require({row["model_id"] for row in values.values()} == {"gpt-5.6-0shot"}, "Unexpected GPT model identifier")
        scorer.require(scorer.coverage(values)["invalid"] == 0, "GPT invalid records")
    circ_result = scorer.score_circular(mapping, circular, ordinary)
    repeat_result = scorer.score_ordinary(ordinary, repeat)
    bundles = scorer.circular_bundles(mapping, ordinary)
    source_ids = sorted(bundles)
    gold = [ordinary[sid]["gold_answer"] for sid in source_ids]
    rotations = []
    for r in range(4):
        rotations.append([mapping[eid]["option_map_new_to_original"][scorer.valid_answer(circular[eid])]
                          for sid in source_ids for eid in [bundles[sid][r]]])
    repeated = [scorer.valid_answer(repeat[sid]) for sid in source_ids]
    pairs = {"repeat": pair_counts(gold, rotations[0], repeated)}
    for r in (1, 2, 3):
        pairs["R" + str(r)] = pair_counts(gold, rotations[0], rotations[r])
    expected = scorer.read_json(root / "expected_metrics.json")
    scorer.assert_matches(repeat_result, expected["new_ordinary_metrics"], "GPT repeat")
    scorer.assert_matches(circ_result, expected["historical_circular_context_only"], "GPT Circular")
    for label, row in pairs.items():
        prefix = "same_order_pair" if label == "repeat" else "rotation_pair_0_" + label[-1]
        for metric in ("both_correct", "consistency"):
            reference = expected["matched_two_execution_sets"][prefix + "_" + metric]
            scorer.require(row[metric] == reference["successes"] and row["n"] == reference["n"], "GPT pair reference mismatch")
        scorer.require(row["consistently_wrong"] == row["consistency"] - row["both_correct"], "Wrong-consistency identity failed")
        row["display_percent"] = {key: f"{100 * value / row['n']:.2f}" for key, value in row.items() if key != "n"}
    return {"status": "PASS", "circular": circ_result, "repeat": repeat_result, "two_execution_pairs": pairs,
            "source_files_unchanged_during_export": receipt["source_files_unchanged"]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=HERE / "recomputed")
    a = p.parse_args()
    out = a.out.resolve()
    scorer.require(out != HERE and not out.is_relative_to(HERE / "frozen_bundle") and not out.is_relative_to(HERE / "gpt_controls"), "Output cannot replace packaged inputs")
    result = scorer.reproduce(HERE / "frozen_bundle")
    gpt = reproduce_gpt()
    scorer.write_results(result, out)
    (out / "gpt_controls.json").write_text(json.dumps(gpt, indent=2) + "\n", encoding="utf-8")
    with (out / "gpt_table6.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "repeat", "R1", "R2", "R3"])
        for metric in ("at_least_one_correct", "both_correct", "consistency", "consistently_wrong"):
            writer.writerow([metric, *(gpt["two_execution_pairs"][arm]["display_percent"][metric] for arm in ("repeat", "R1", "R2", "R3"))])
    status = {"status": "PASS", "primary_main_models": 20, "ordinary_multiseed_models": 8, "open_circular_complete_models": 7,
              "gpt_circular_valid": 1152, "gpt_repeat_valid": 288, "gpt_table6_display_cells": 16,
              "qwen_base_circular_summary_independently_recomputed": False, "api_calls": 0}
    (out / "verification.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
