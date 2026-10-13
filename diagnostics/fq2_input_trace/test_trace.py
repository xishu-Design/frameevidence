"""Structural tests. Synthetic test cases are not benchmark evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import unittest

from build_trace import reference_trace, stratum


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.base = ["a", "b", "c"]
        self.registry = {"a": "S01", "b": "S02", "c": "S03", "d": "S04"}

    def test_removed_anchor(self):
        refs = reference_trace("以图一为参考", self.base, ["b", "c"], self.registry)
        self.assertEqual(stratum(refs), "anchor_removed")
        self.assertEqual(refs[0]["original_anchor_id"], "S01")
        self.assertEqual(refs[0]["current_positional_target_id"], "S02")

    def test_retained_shifted_anchor(self):
        refs = reference_trace("以图二为参考", self.base, ["b", "c"], self.registry)
        self.assertEqual(stratum(refs), "anchor_retained_shifted")
        self.assertEqual(refs[0]["current_positional_target_id"], "S03")

    def test_out_of_range(self):
        refs = reference_trace("以图三为参考", self.base, ["a", "c"], self.registry)
        self.assertEqual(stratum(refs), "anchor_retained_shifted")
        self.assertEqual(refs[0]["positional_status"], "out_of_range")

    def test_stable_and_no_ordinal(self):
        self.assertEqual(stratum(reference_trace("以图二为参考", self.base, ["a", "b"], self.registry)), "ordinal_stable")
        self.assertEqual(stratum(reference_trace("杯子在哪", self.base, ["a", "b"], self.registry)), "no_ordinal")


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).resolve().parent
        cls.rows = [json.loads(line) for line in (folder / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.verification = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
        cls.examples = json.loads((folder / "examples_zh_en.json").read_text(encoding="utf-8"))

    def test_scope_and_counts(self):
        self.assertEqual(len(self.rows), 238)
        self.assertEqual(len({r["case_id"] for r in self.rows}), 238)
        self.assertEqual(len({r["group_id"] for r in self.rows}), 48)
        self.assertEqual(Counter(r["condition"] for r in self.rows), {
            "one_valid_mss": 48, "remove_required_view": 48,
            "hard_distractor_added": 48, "leave_one_mss_member_out": 94})

    def test_no_new_gold_or_semantic_validation(self):
        for row in self.rows:
            self.assertIsNone(row["new_adjudicated_answer"])
            self.assertFalse(row["semantic_validation_performed"])
            self.assertFalse(row["stable_ids_present_in_original_prompt"])
            if row["kind"] == "additional_deletion_candidate":
                self.assertIsNone(row["source_eval_id"])
                self.assertIsNone(row["released_answer_reference"])
            else:
                self.assertIn(row["released_answer_reference"], "ABCD")

    def test_every_mss_member_has_one_unique_deletion(self):
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["group_id"]].append(row)
        for gid, rows in groups.items():
            mss = next(r for r in rows if r["condition"] == "one_valid_mss")
            ids = [im["stable_original_id"] for im in mss["actual_image_sequence"]]
            deletions = [r for r in rows if r["condition"] in {"remove_required_view", "leave_one_mss_member_out"}]
            self.assertEqual({r["removed_original_ids"][0] for r in deletions}, set(ids), gid)
            self.assertEqual(len(deletions), len(ids), gid)
            for row in deletions:
                self.assertEqual(len(row["removed_original_ids"]), 1)
                self.assertEqual([i["stable_original_id"] for i in row["actual_image_sequence"]],
                                 [sid for sid in ids if sid not in row["removed_original_ids"]])

    def test_examples_and_original_prompt(self):
        self.assertEqual({e["group_id"] for e in self.examples}, {"group_00392", "group_00368", "group_00311"})
        for example in self.examples:
            self.assertEqual(len(example["published_triplet"]), 3)
            for row in example["published_triplet"]:
                self.assertIn(example["question_zh"], row["original_model_prompt_zh"])
                self.assertEqual(row["options_zh"], example["options_zh"])

    def test_read_only_no_calls(self):
        self.assertEqual(self.verification["original_prompts_checked"], 144)
        self.assertEqual(self.verification["previous_diagnostic_cases_checked"], 238)
        self.assertEqual(self.verification["new_model_calls"], 0)
        self.assertEqual(self.verification["new_human_records"], 0)
        self.assertFalse(self.verification["source_files_modified"])


if __name__ == "__main__":
    unittest.main()
