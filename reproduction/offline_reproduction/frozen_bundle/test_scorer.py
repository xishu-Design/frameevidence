"""Hand-calculated metric contracts, adverse inputs, and packaged-data regression."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if (HERE / "scorer.py").exists():
    import scorer
    BUNDLE = HERE
else:
    import offline_reproduction_scorer as scorer
    BUNDLE = HERE.parent / "materials/offline_reproduction_20261009"
    if (HERE.parent / "materials/offline_reproduction_20261009_v2/bundle_manifest.json").is_file():
        BUNDLE = HERE.parent / "materials/offline_reproduction_20261009_v2"


def ordinary_fixture():
    mapping, pred = {}, {}
    conditions = ("one_valid_mss", "remove_required_view", "hard_distractor_added")
    frames = ("camera", "object_intrinsic", "world")
    for i, (gold, answer) in enumerate(zip("ABCADA", "ABCADB")):
        eid = str(i)
        mapping[eid] = {"eval_id": eid, "relation_group_id": "f1" if i < 3 else "f2",
                        "evaluation_mode": "all-images", "question_type": "FQ1" if i < 3 else "FQ2",
                        "reference_frame": frames[i % 3], "view_condition": conditions[i % 3], "gold_answer": gold}
        pred[eid] = {"eval_id": eid, "evaluation_mode": "all-images", "status": "OK", "parsed_answer": answer}
    return mapping, pred


def circular_fixture(gold="B", semantic_answers="BBBB"):
    ordinary = {"s": {"gold_answer": gold}}
    mapping, pred = {}, {}
    for r in range(4):
        labels = {new: "ABCD"[(i + r) % 4] for i, new in enumerate("ABCD")}
        reverse = {old: new for new, old in labels.items()}
        eid = str(r)
        mapping[eid] = {"eval_id": eid, "source_eval_id": "s", "circular_rotation": r,
                        "evaluation_mode": "all-images", "option_map_new_to_original": labels,
                        "gold_answer": reverse[gold]}
        pred[eid] = {"eval_id": eid, "evaluation_mode": "all-images", "status": "OK", "parsed_answer": reverse[semantic_answers[r]]}
    return ordinary, mapping, pred


class MetricTests(unittest.TestCase):
    def test_hand_computed_macro_and_joint(self):
        mapping, pred = ordinary_fixture()
        row = scorer.score_ordinary(mapping, pred)
        self.assertEqual(row["correct"], 5)
        self.assertAlmostEqual(row["accuracy"], 5 / 6)
        self.assertAlmostEqual(row["macro_precision"], 0.875)
        self.assertAlmostEqual(row["macro_recall"], (2 / 3 + 3) / 4)
        self.assertAlmostEqual(row["macro_f1"], (0.8 + 2 / 3 + 2) / 4)
        self.assertEqual(row["joint"], {"FQ1_GEM": [1, 1], "NTS": [1, 1], "DIS": [0, 1]})

    def test_invalids_remain_in_denominator_and_status_overrides_letter(self):
        mapping, pred = ordinary_fixture()
        pred["0"].update(status="CALL_ERROR", parsed_answer="A")
        pred["1"]["parsed_answer"] = "not-a-label"
        row = scorer.score_ordinary(mapping, pred)
        self.assertEqual((row["correct"], row["n"]), (3, 6))
        self.assertEqual(row["coverage"]["invalid_reasons"], {"CALL_ERROR": 1, "INVALID_ANSWER": 1})
        self.assertEqual(row["confusion"]["A"]["INVALID"], 1)
        self.assertEqual(row["joint"]["FQ1_GEM"], [0, 1])

    def test_no_predicted_class_uses_zero_precision(self):
        mapping, pred = ordinary_fixture()
        for value in pred.values():
            value["parsed_answer"] = "A"
        row = scorer.score_ordinary(mapping, pred)
        self.assertAlmostEqual(row["macro_precision"], (3 / 6) / 4)
        self.assertAlmostEqual(row["macro_recall"], 1 / 4)

    def test_missing_and_duplicate_ids_fail(self):
        mapping, pred = ordinary_fixture()
        del pred["0"]
        with self.assertRaisesRegex(ValueError, "ID sets"):
            scorer.score_ordinary(mapping, pred)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            scorer.indexed([{"eval_id": "x"}, {"eval_id": "x"}])

    def test_bad_triplet_and_mode_fail(self):
        mapping, pred = ordinary_fixture()
        mapping["0"]["reference_frame"] = "world"
        with self.assertRaisesRegex(ValueError, "reference frames"):
            scorer.score_ordinary(mapping, pred)
        mapping, pred = ordinary_fixture()
        pred["0"]["evaluation_mode"] = "text-only"
        with self.assertRaisesRegex(ValueError, "Mode mismatch"):
            scorer.score_ordinary(mapping, pred)

    def test_circular_semantic_mapping_and_strict_all_four(self):
        ordinary, mapping, pred = circular_fixture()
        row = scorer.score_circular(mapping, pred, ordinary)
        self.assertEqual(row["rotation_correct"], [1, 1, 1, 1])
        self.assertEqual((row["circular"], row["consistency"], row["drop"]), (100, 100, 0))
        ordinary, mapping, pred = circular_fixture(semantic_answers="BBBA")
        row = scorer.score_circular(mapping, pred, ordinary)
        self.assertEqual((row["vanilla"], row["mean_rotation"], row["circular"], row["consistency"], row["drop"]), (100, 75, 0, 0, 100))

    def test_consistent_wrong_is_not_correct(self):
        ordinary, mapping, pred = circular_fixture(semantic_answers="AAAA")
        row = scorer.score_circular(mapping, pred, ordinary)
        self.assertEqual((row["circular"], row["consistency"]), (0, 100))

    def test_circular_invalid_is_not_semantically_consistent(self):
        ordinary, mapping, pred = circular_fixture()
        pred["0"]["status"] = "CALL_ERROR"
        row = scorer.score_circular(mapping, pred, ordinary)
        self.assertEqual((row["mean_rotation"], row["circular"], row["consistency"]), (75, 0, 0))

    def test_corrupt_rotation_and_gold_fail(self):
        ordinary, mapping, pred = circular_fixture()
        mapping["1"]["option_map_new_to_original"]["A"] = "A"
        with self.assertRaisesRegex(ValueError, "permutation"):
            scorer.score_circular(mapping, pred, ordinary)
        ordinary, mapping, pred = circular_fixture()
        mapping["1"]["gold_answer"] = "D"
        with self.assertRaisesRegex(ValueError, "gold remapping"):
            scorer.score_circular(mapping, pred, ordinary)

    def test_sample_sd_uses_n_minus_one_and_requires_three_seeds(self):
        rows = [{"model_id": f"{family}-lora-seed{seed}", **{metric: value for metric in scorer.CIRCULAR_METRICS}}
                for family in scorer.FAMILIES for seed, value in zip(scorer.SEEDS, (10, 20, 30))]
        result = scorer.seed_summary(rows, circular=True)
        self.assertEqual(result[scorer.FAMILIES[0]]["circular"], {"mean": 20, "sample_sd": 10.0})
        with self.assertRaisesRegex(ValueError, "three distinct"):
            scorer.seed_summary(rows[:-1], circular=True)

    def test_reference_changes_fail(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            scorer.assert_matches({"correct": 17}, {"correct": 18})


class BundleTests(unittest.TestCase):
    def test_complete_bundle_matches_saved_references(self):
        result = scorer.reproduce(BUNDLE)
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertEqual(len(result["ordinary"]["models"]), 8)
        self.assertEqual(len(result["circular_complete"]["models"]), 7)
        self.assertEqual(result["circular_incomplete_diagnostic"]["models"][0]["coverage"]["invalid_reasons"], {"CALL_ERROR": 446})
        self.assertIs(result["author_confirmed_summary"]["completed_run_independently_recomputed"], False)
        self.assertTrue(any(row["ordinary_vs_circular_r0"]["same_predictions"] < 288 for row in result["circular_complete"]["models"]))

    def test_copy_runs_without_repository_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory(prefix="frameevidence_offline_") as temporary:
            root = Path(temporary)
            isolated = root / "bundle"
            shutil.copytree(BUNDLE, isolated, ignore=shutil.ignore_patterns("__pycache__"))
            command = [sys.executable, "-I", "-B", str(isolated / "scorer.py"), "--out", str(root / "out")]
            process = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = json.loads((root / "out/recomputed_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verification"]["complete_circular_models"], 7)
            target = isolated / "inputs/ordinary_items.jsonl"
            target.write_bytes(target.read_bytes() + b"\n")
            process = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 1)
            self.assertIn("Bundle digest mismatch", process.stdout)

    def test_export_has_only_whitelisted_scoring_fields(self):
        for relative, fields in (("inputs/ordinary_items.jsonl", {"eval_id", "relation_group_id", "evaluation_mode", "question_type", "reference_frame", "view_condition", "gold_answer"}),
                                 ("inputs/circular_items.jsonl", {"eval_id", "source_eval_id", "evaluation_mode", "circular_rotation", "gold_answer", "option_map_new_to_original"})):
            for row in scorer.read_jsonl(BUNDLE / relative):
                self.assertEqual(set(row), fields)
        prediction_fields = {"eval_id", "model_id", "evaluation_mode", "status", "parsed_answer", "parse_status"}
        for path in (BUNDLE / "inputs").glob("*/*.jsonl"):
            for row in scorer.read_jsonl(path):
                self.assertEqual(set(row), prediction_fields)
        for path in BUNDLE.rglob("*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/root/", text)
            self.assertNotIn("D:\\\\", text)
            self.assertNotIn("api_key", text)

    def test_primary_main_twenty_rows_and_table_rounding(self):
        if scorer.read_json(BUNDLE / "metadata.json")["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        result = scorer.reproduce(BUNDLE)
        models = result["primary_main"]["models"]
        self.assertEqual([row["model_id"] for row in models], list(scorer.MAIN_MODEL_IDS))
        self.assertEqual(sum(row["n"] for row in models), 5760)
        for row in models:
            self.assertEqual(row["n"], 288)
            self.assertTrue(all(pair[1] == 48 for pair in row["joint"].values()))
            self.assertEqual(row["table_main_display_percent"], {metric: f"{scorer.ordinary_percent(row, metric):.2f}" for metric in scorer.ORDINARY_METRICS})
        self.assertEqual(result["verification"]["main_table_display_value_checks"], 140)

    def test_primary_main_llava_errors_keep_full_denominator(self):
        if scorer.read_json(BUNDLE / "metadata.json")["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        models = scorer.reproduce(BUNDLE)["primary_main"]["models"]
        llava = next(row for row in models if row["model_id"] == "llava-onevision-7b-base")
        self.assertEqual((llava["correct"], llava["n"]), (84, 288))
        self.assertAlmostEqual(llava["accuracy"], 84 / 288)
        self.assertEqual(llava["coverage"]["valid"], 278)
        self.assertEqual(llava["coverage"]["invalid_reasons"], {"CALL_ERROR": 10})
        self.assertEqual(sum(row["INVALID"] for row in llava["confusion"].values()), 10)
        self.assertEqual(sum(row["coverage"]["invalid"] for row in models), 10)

    def test_primary_main_is_not_pooled_into_multiseed_or_circular(self):
        if scorer.read_json(BUNDLE / "metadata.json")["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        result = scorer.reproduce(BUNDLE)
        self.assertTrue(set(scorer.MAIN_MODEL_IDS).isdisjoint(scorer.MODEL_IDS))
        self.assertEqual(len(result["ordinary"]["models"]), 8)
        self.assertEqual(result["ordinary"]["families"], scorer.seed_summary(result["ordinary"]["models"]))
        self.assertEqual(len(result["circular_complete"]["models"]), 7)
        self.assertEqual(len(result["circular_incomplete_diagnostic"]["models"]), 1)
        self.assertIs(result["author_confirmed_summary"]["completed_run_independently_recomputed"], False)
        qwen_main = next(row for row in result["primary_main"]["models"] if row["model_id"] == "qwen25-vl-7b-base")
        qwen_control = next(row for row in result["ordinary"]["models"] if row["model_id"] == "qwen25-vl-7b-base-multiseed-control")
        self.assertNotEqual(qwen_main["correct"], qwen_control["correct"])

    def test_primary_main_cohort_and_prediction_bindings_reject_substitution(self):
        metadata = scorer.read_json(BUNDLE / "metadata.json")
        if metadata["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        changed = copy.deepcopy(metadata)
        config = next(row for row in changed["configs"] if row["experiment"] == "primary_main")
        config["run_cohort"] = "ordinary_multiseed"
        with self.assertRaisesRegex(ValueError, "cohort mismatch"):
            scorer.reproduce_primary_main(BUNDLE, changed)
        changed = copy.deepcopy(metadata)
        config = next(row for row in changed["configs"] if row["experiment"] == "primary_main")
        config["packaged_prediction_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prediction binding mismatch"):
            scorer.reproduce_primary_main(BUNDLE, changed)
        changed = copy.deepcopy(metadata)
        changed["cohorts"]["primary_main"]["packaged_mapping_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "mapping binding mismatch"):
            scorer.reproduce_primary_main(BUNDLE, changed)

    def test_primary_main_whitelist_and_source_unchanged_receipt(self):
        if scorer.read_json(BUNDLE / "metadata.json")["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        for row in scorer.read_jsonl(BUNDLE / "inputs/primary_main_items.jsonl"):
            self.assertEqual(set(row), set(scorer.MAPPING_FIELDS))
        receipt = scorer.read_json(BUNDLE / "source_verification.json")
        self.assertIs(receipt["all_source_files_unchanged"], True)
        self.assertEqual(receipt["source_files_checked_before_and_after"], len(receipt["source_sha256"]))
        self.assertTrue(any(key.startswith("primary_main_run/") for key in receipt["source_sha256"]))
        self.assertTrue(any(key.startswith("downloaded_run/") for key in receipt["source_sha256"]))

    def test_primary_main_csv_and_portable_copy(self):
        if scorer.read_json(BUNDLE / "metadata.json")["schema_version"] != 2:
            self.skipTest("Primary main cohort is a schema-v2 extension")
        with tempfile.TemporaryDirectory(prefix="frameevidence_primary_") as temporary:
            root = Path(temporary)
            isolated = root / "bundle"
            shutil.copytree(BUNDLE, isolated, ignore=shutil.ignore_patterns("__pycache__"))
            process = subprocess.run([sys.executable, "-I", "-B", str(isolated / "scorer.py"), "--out", str(root / "out")], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            result = scorer.read_json(root / "out/recomputed_metrics.json")
            self.assertEqual(result["verification"]["primary_main_models"], 20)
            self.assertEqual(len((root / "out/primary_main.csv").read_text(encoding="utf-8").splitlines()), 21)
            self.assertEqual({path.name for path in (root / "out").glob("*.csv")}, {"primary_main.csv", "ordinary.csv", "circular_complete.csv", "circular_incomplete_diagnostic.csv"})


if __name__ == "__main__":
    unittest.main()
