"""Independent pair-count tests and retained GPT regression."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import run_reproduction as r

class SupplementalTests(unittest.TestCase):
    def test_hand_counted_pairs(self):
        actual = r.pair_counts(list("ABCD"), list("ABAA"), list("BBAA"))
        self.assertEqual(actual, {"n": 4, "at_least_one_correct": 2, "both_correct": 1, "consistency": 3, "consistently_wrong": 2})

    def test_invalid_answers_never_count_consistent(self):
        actual = r.pair_counts(["A", "B"], [None, "B"], [None, None])
        self.assertEqual(actual, {"n": 2, "at_least_one_correct": 1, "both_correct": 0, "consistency": 0, "consistently_wrong": 0})

    def test_pair_lengths_and_empty_rejected(self):
        for args in (([], [], []), (["A"], [], ["A"])):
            with self.assertRaises(ValueError):
                r.pair_counts(*args)

    def test_gpt_complete_retained_inputs(self):
        result = r.reproduce_gpt()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["circular"]["rotation_correct"], [131, 120, 122, 120])
        self.assertEqual(result["circular"]["circular_correct"], 54)
        self.assertEqual(result["circular"]["consistent_items"], 78)
        self.assertEqual(result["repeat"]["correct"], 130)
        self.assertEqual([result["two_execution_pairs"][arm]["at_least_one_correct"] for arm in ("repeat", "R1", "R2", "R3")], [179, 161, 165, 170])
        self.assertEqual([result["two_execution_pairs"][arm]["consistently_wrong"] for arm in ("repeat", "R1", "R2", "R3")], [72, 81, 70, 67])

    def test_gpt_export_field_whitelists(self):
        fields = {
            "ordinary_items.jsonl": set(r.scorer.MAPPING_FIELDS),
            "circular_items.jsonl": {"eval_id", "source_eval_id", "evaluation_mode", "circular_rotation", "gold_answer", "option_map_new_to_original"},
            "circular_predictions.jsonl": set(r.scorer.PREDICTION_FIELDS),
            "repeat_predictions.jsonl": set(r.scorer.PREDICTION_FIELDS),
        }
        for name, allowed in fields.items():
            for row in r.scorer.read_jsonl(r.HERE / "gpt_controls" / name):
                self.assertEqual(set(row), allowed)
        for path in (r.HERE / "gpt_controls").glob("*.json*"):
            text = path.read_text(encoding="utf-8")
            for forbidden in ("/root/", "D:\\\\", "api_key", "https://", "raw_response"):
                self.assertNotIn(forbidden, text)

    def test_whole_package_portable_in_isolated_python(self):
        with tempfile.TemporaryDirectory(prefix="frameevidence_gpt_portable_") as directory:
            destination = Path(directory) / "portable"
            shutil.copytree(r.HERE, destination, ignore=shutil.ignore_patterns("__pycache__", "tmp", "recomputed"))
            output = Path(directory) / "result"
            process = subprocess.run([sys.executable, "-I", "-B", str(destination / "run_reproduction.py"), "--out", str(output)],
                                     cwd=directory, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            status = json.loads((output / "verification.json").read_text(encoding="utf-8"))
            self.assertEqual(status["gpt_table6_display_cells"], 16)
            self.assertEqual(status["api_calls"], 0)

if __name__ == "__main__":
    unittest.main()
