"""Verify packaged image bytes and all relative delivered-image references."""
import hashlib
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parent


def rows(name):
    return [json.loads(line) for line in (BASE / name).read_text(encoding="utf-8").splitlines() if line]


class BundledImagesTests(unittest.TestCase):
    def test_all_image_bytes_match(self):
        inventory = json.loads((BASE / "image_files.json").read_text(encoding="utf-8"))
        self.assertEqual(len(inventory), 125)
        for rel, metadata in inventory.items():
            path = BASE / rel
            self.assertTrue(path.resolve().is_relative_to(BASE))
            data = path.read_bytes()
            self.assertEqual(len(data), metadata["bytes"], rel)
            self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"], rel)

    def test_every_case_image_is_present(self):
        for case in rows("cases.jsonl"):
            for image in case["actual_image_sequence"]:
                self.assertTrue((BASE / image["prepared_ref"]).is_file(), case["case_id"])

    def test_registries_have_no_absolute_paths(self):
        for row in rows("image_registry.jsonl"):
            self.assertNotIn("prepared_absolute_path", row)
            self.assertFalse(Path(row["prepared_ref"]).is_absolute())
            self.assertTrue((BASE / row["prepared_ref"]).is_file())
        verification = json.loads((BASE / "verification.json").read_text(encoding="utf-8"))
        for value in verification["source_paths"].values():
            self.assertNotIn(":", value)
            self.assertFalse(value.startswith(("/", "\\")))


if __name__ == "__main__":
    unittest.main()
