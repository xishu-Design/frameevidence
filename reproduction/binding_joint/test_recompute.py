"""Pure standard-library unit tests, with no network or model calls."""
import copy
import unittest

import recompute as r


def fixture():
    metadata, predictions = [], []
    answers = {("g1", r.ARMS[0]): "AAB", ("g1", r.ARMS[1]): "AAA",
               ("g2", r.ARMS[0]): "AAA", ("g2", r.ARMS[1]): "BAA"}
    for gid in ("g1", "g2"):
        for arm in r.ARMS:
            for c, answer in zip(r.CONDITIONS, answers[gid, arm]):
                eid = gid + arm + c
                metadata.append({"eval_id": eid, "group_id": gid, "arm": arm, "condition": c,
                                 "source_eval_id": gid+c, "stratum": "no_ordinal", "released_label_reference": "A"})
                predictions.append({"eval_id": eid, "status": "OK", "parsed_answer": answer,
                                    "evaluation_mode": "all-images", "model_id": "test", "api_model": "test"})
    return metadata, predictions


def normalized(metadata=None, predictions=None):
    a, b = fixture()
    return r.normalize(a if metadata is None else metadata, b if predictions is None else predictions,
                       "test", (1,), expected_groups=2)


class JointTests(unittest.TestCase):
    def test_joint_truth_table(self):
        self.assertEqual(r.group_metrics(dict(zip(r.CONDITIONS, (True, False, True)))),
                         {"MSS": True, "Deletion": False, "Addition": True, "NTS": False, "DIS": True, "All3": False})

    def test_hand_counted_joint_and_paired_structure(self):
        summary, groups, paired, details = r.calculate(normalized())
        original = next(x for x in summary if x["arm"] == r.ARMS[0])
        stable = next(x for x in summary if x["arm"] == r.ARMS[1])
        self.assertEqual((original["NTS"], original["DIS"], original["All3"]), (2, 1, 1))
        self.assertEqual((stable["NTS"], stable["DIS"], stable["All3"]), (1, 1, 1))
        dis = next(x for x in paired if x["metric"] == "DIS")
        self.assertEqual((dis["both"], dis["gain"], dis["loss"], dis["neither"]), (0, 1, 1, 0))
        for metric in paired:
            self.assertEqual(sum(metric[k] for k in ("both", "gain", "loss", "neither")), 2)
            self.assertEqual(metric["difference_pp"], 50*(metric["gain"]-metric["loss"]))

    def test_missing_prediction_rejected(self):
        a, b = fixture()
        with self.assertRaisesRegex(ValueError, "Prediction IDs"):
            normalized(a, b[:-1])

    def test_duplicate_prediction_rejected(self):
        a, b = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate eval_id"):
            normalized(a, b+[copy.deepcopy(b[0])])

    def test_duplicate_metadata_rejected(self):
        a, b = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate eval_id"):
            normalized(a+[copy.deepcopy(a[0])], b)

    def test_mismatched_paired_label_rejected(self):
        a, b = fixture()
        a[3]["released_label_reference"] = "B"
        with self.assertRaisesRegex(ValueError, "source or label"):
            normalized(a, b)

    def test_source_reused_across_conditions_rejected(self):
        a, b = fixture()
        a[1]["source_eval_id"] = a[0]["source_eval_id"]
        with self.assertRaisesRegex(ValueError, "Source reused"):
            normalized(a, b)

    def test_invalid_status_does_not_gain_correctness(self):
        a, b = fixture()
        b[0]["status"] = "CALL_ERROR"
        b[0]["parsed_answer"] = "A"
        summary, *_ = r.calculate(normalized(a, b))
        original = next(x for x in summary if x["arm"] == r.ARMS[0])
        self.assertEqual((original["groups"], original["items"], original["valid_responses"], original["NTS"]), (2, 6, 5, 1))

    def test_new_semantic_gold_rejected(self):
        a, b = fixture()
        a[0]["new_semantic_gold"] = "A"
        with self.assertRaisesRegex(ValueError, "semantic gold"):
            normalized(a, b)

    def test_condition_pairing_rejected(self):
        a, b = fixture()
        a[1]["condition"] = a[0]["condition"]
        with self.assertRaises(ValueError):
            normalized(a, b)


if __name__ == "__main__":
    unittest.main()
