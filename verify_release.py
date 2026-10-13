"""Read-only verification of the FrameEvidence data and release manifest."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

sys.dont_write_bytecode = True

EXPECTED_FINGERPRINT = "2c289604aa45e7373ae9dcaa6737ffe74ed31f5b9aa0005e85ad3c22a6dde5bc"
EXPECTED_ANNOTATIONS_SHA256 = "9fe3013fe5abe61257490f44111607f23db6a9171bfd29c917e18ddae28c4496"
EXPECTED_SPLIT_SHA256 = "bc9afad049e3c6cbdc9b5328ea3c2bdb9be70f0ada784033c820ce80ee6a53d9"
EXPECTED_SPLITS = {"train": 192, "dev": 32, "test": 96}
FIELDS = {
    "answer", "bridge_instance_ids", "images", "mss_images", "object_instances",
    "options", "quality_status", "question", "question_type", "reference_anchor",
    "reference_frame", "relation_group_id", "relatum_instance_id", "removed_images",
    "sample_id", "target_instance_id", "view_condition",
}
MANIFEST_NAME = "release_manifest.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Empty or non-string relative path")
    require("\\" not in relative and ":" not in relative, "Non-portable relative path: " + relative)
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Unsafe relative path: " + relative)
    require(not PurePosixPath(relative).is_absolute(), "Absolute path: " + relative)
    path = root.joinpath(*parts)
    require(path.resolve().is_relative_to(root.resolve()), "Path escapes its directory: " + relative)
    require(path.is_file() and not path.is_symlink(), "Missing or linked file: " + relative)
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def manifest_entries(payload: dict) -> dict[str, tuple[str, int | None]]:
    require(isinstance(payload, dict), "Release manifest must be a JSON object")
    raw = payload.get("files", payload.get("sha256", payload))
    if isinstance(raw, list):
        pairs = []
        for entry in raw:
            require(isinstance(entry, dict) and "path" in entry, "Manifest entry needs a path")
            pairs.append((entry["path"], entry))
    else:
        require(isinstance(raw, dict), "Manifest files must be an object or a list")
        pairs = list(raw.items())
    result = {}
    for relative, entry in pairs:
        require(isinstance(relative, str), "Manifest paths must be strings")
        require(relative not in result, "Duplicate manifest path: " + relative)
        require(relative != MANIFEST_NAME, "The release manifest must exclude itself")
        if isinstance(entry, str):
            digest, size = entry, None
        else:
            require(isinstance(entry, dict), "Invalid manifest entry: " + relative)
            digest = entry.get("sha256")
            size = entry.get("bytes", entry.get("size"))
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None,
                "Invalid SHA-256: " + relative)
        require(size is None or (type(size) is int and size >= 0), "Invalid byte count: " + relative)
        result[relative] = (digest.lower(), size)
    require(bool(result), "Release manifest is empty")
    return result


def public_files(root: Path) -> set[str]:
    result = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = [name for name in names
                    if name not in {".git", "__pycache__", ".pytest_cache"}
                    and not (parent == root and name == "outputs")]
        for name in names:
            require(not (parent / name).is_symlink(), "Linked directory in release: " + name)
        for name in filenames:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_NAME:
                continue
            require(not path.is_symlink(), "Linked file in release: " + relative)
            result.add(relative)
    return result


def verify_manifest(root: Path) -> tuple[int, int]:
    entries = manifest_entries(read_json(root / MANIFEST_NAME))
    actual = public_files(root)
    expected = set(entries)
    require(actual == expected,
            "Release manifest coverage mismatch; missing=" + repr(sorted(expected - actual)[:8])
            + "; unlisted=" + repr(sorted(actual - expected)[:8]))
    total_bytes = 0
    for relative, (expected_digest, expected_size) in sorted(entries.items()):
        path = safe_path(root, relative)
        size = path.stat().st_size
        require(expected_size is None or size == expected_size, "Byte count mismatch: " + relative)
        require(sha256(path) == expected_digest, "Release digest mismatch: " + relative)
        total_bytes += size
    return len(entries), total_bytes


def verify_dataset(root: Path, check_schema: bool) -> dict:
    dataset = root / "dataset"
    annotations = safe_path(dataset, "annotations.jsonl")
    require(sha256(annotations) == EXPECTED_ANNOTATIONS_SHA256, "Frozen annotations SHA-256 mismatch")
    records = [json.loads(line) for line in annotations.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    require(len(records) == 960, "Expected exactly 960 annotation records")
    groups = defaultdict(list)
    sample_ids, references = set(), set()
    for index, row in enumerate(records, 1):
        require(isinstance(row, dict) and set(row) == FIELDS, "Annotation field mismatch at row " + str(index))
        sample_id, group_id = row["sample_id"], row["relation_group_id"]
        require(isinstance(sample_id, str) and bool(sample_id) and sample_id not in sample_ids,
                "Missing or duplicate sample ID at row " + str(index))
        require(isinstance(group_id, str) and bool(group_id), "Missing group ID at row " + str(index))
        sample_ids.add(sample_id)
        groups[group_id].append(row)
        require(row["answer"] in ("A", "B", "C", "D"), "Invalid answer label at row " + str(index))
        require(isinstance(row["options"], dict) and set(row["options"]) == set("ABCD"),
                "Invalid option inventory at row " + str(index))
        for field in ("images", "removed_images", "mss_images"):
            paths = row[field]
            require(isinstance(paths, list), "Image references must be lists at row " + str(index))
            for relative in paths:
                require(isinstance(relative, str) and relative.startswith("images/"),
                        "Invalid image reference at row " + str(index))
                safe_path(dataset, relative)
                references.add(relative)
    require(len(groups) == 320 and all(len(rows) == 3 for rows in groups.values()),
            "Expected 320 complete three-question groups")
    require(Counter(row["question_type"] for row in records) == {"FQ1": 480, "FQ2": 480},
            "Expected 480 FQ1 and 480 FQ2 questions")
    for group_id, rows in groups.items():
        types = {row["question_type"] for row in rows}
        require(len(types) == 1, "Mixed question types in group " + group_id)
        if types == {"FQ1"}:
            require({row["reference_frame"] for row in rows} == {"camera", "object_intrinsic", "world"},
                    "Incomplete FQ1 frame triplet: " + group_id)
        else:
            require({row["view_condition"] for row in rows} ==
                    {"one_valid_mss", "remove_required_view", "hard_distractor_added"},
                    "Incomplete FQ2 condition triplet: " + group_id)
    image_files = {path.relative_to(dataset).as_posix() for path in (dataset / "images").rglob("*")
                   if path.is_file()}
    require(len(image_files) == 1120 and len(references) == 1120, "Expected 1,120 image files and references")
    require(image_files == references, "Image inventory does not equal annotation references")
    require(all(Path(relative).suffix.lower() == ".jpg" for relative in image_files),
            "The frozen image inventory must contain only JPEG files")
    fingerprint = hashlib.sha256(annotations.read_bytes())
    for relative in sorted(references):
        fingerprint.update(relative.encode("utf-8"))
        fingerprint.update(bytes.fromhex(sha256(safe_path(dataset, relative))))
    require(fingerprint.hexdigest() == EXPECTED_FINGERPRINT, "Frozen dataset fingerprint mismatch")

    split_path = safe_path(root, "splits/split_manifest.csv")
    require(sha256(split_path) == EXPECTED_SPLIT_SHA256, "Frozen split manifest SHA-256 mismatch")
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    assignment = {row["relation_group_id"]: row["split"] for row in split_rows}
    require(len(split_rows) == len(assignment) == 320 and set(assignment) == set(groups),
            "Split manifest must cover every group exactly once")
    require(Counter(assignment.values()) == EXPECTED_SPLITS, "Frozen split group counts differ")
    split_items = {name: sum(len(groups[group]) for group, split in assignment.items() if split == name)
                   for name in EXPECTED_SPLITS}
    require(split_items == {"train": 576, "dev": 96, "test": 288}, "Frozen split item counts differ")
    schema_status = "not_requested"
    if check_schema:
        try:
            from jsonschema import Draft202012Validator
        except ImportError as exc:
            raise ValueError("--schema requires jsonschema; install requirements.txt") from exc
        schema = read_json(safe_path(root, "02_数据模式与标注试点/data_schema.json"))
        stack = [schema]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if "$ref" in value:
                    require(isinstance(value["$ref"], str) and value["$ref"].startswith("#"),
                            "Schema validation permits local references only")
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for index, record in enumerate(records, 1):
            require(validator.is_valid(record), "JSON Schema validation failed at row " + str(index))
        schema_status = "PASS"
    return {"annotations": len(records), "groups": len(groups), "images": len(image_files),
            "dataset_fingerprint": fingerprint.hexdigest(), "split_groups": dict(EXPECTED_SPLITS),
            "split_items": split_items, "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "schema_validation": schema_status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Extracted release root; defaults to this script's directory")
    parser.add_argument("--schema", action="store_true", help="Also validate the local JSON Schema")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        count, size = verify_manifest(root)
        result = verify_dataset(root, args.schema)
        print(json.dumps({"status": "PASS", "manifest_files": count, "manifest_bytes": size,
                          **result, "files_written": 0, "network_requests": 0}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "files_written": 0,
                          "network_requests": 0}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
