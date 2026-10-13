#!/usr/bin/env python3
"""Deterministic four-rotation Circular evaluation for FrameEvidence.

The protocol creates four answer-order variants per frozen test item.  Images,
question text, model decoding settings, and the train/test split remain fixed.
Only the A--D option positions change.  Results live below a separate circular
run directory, so existing Vanilla predictions are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PROTOCOL_ID = "frameevidence-circular-cyclic4-v1"
LABELS = ("A", "B", "C", "D")
SUPPORTED_MODES = ("all-images", "text-only", "single-image")
PROMPT_TEMPLATE = (
    "请根据当前提供的信息回答下面的四选一问题。\n"
    "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
    "只输出一个答案字母（A、B、C 或 D），不要解释。"
)


def _install_dry_run_import_path() -> None:
    direct = HERE / "dry_run_eval" / "runner.py"
    if direct.is_file():
        sys.path.insert(0, str(HERE))
        return
    for child in HERE.parent.iterdir():
        if child.is_dir() and (child / "dry_run_eval" / "runner.py").is_file():
            sys.path.insert(0, str(child))
            return
    raise RuntimeError("could not locate dry_run_eval/runner.py")


_install_dry_run_import_path()
from dry_run_eval.runner import check_model_config, run_models  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    )


def rotate_options(
    options: dict[str, Any], gold_answer: str, rotation: int
) -> tuple[dict[str, str], str, dict[str, str]]:
    """Left-cycle option contents and return remapped gold plus new->old map."""
    if set(options) != set(LABELS):
        raise ValueError("Circular requires exactly four A/B/C/D options")
    if gold_answer not in LABELS:
        raise ValueError(f"invalid gold answer: {gold_answer}")
    if rotation not in range(4):
        raise ValueError(f"rotation must be 0..3, got {rotation}")
    new_to_original = {
        new_label: LABELS[(new_index + rotation) % 4]
        for new_index, new_label in enumerate(LABELS)
    }
    rotated = {
        new_label: str(options[old_label])
        for new_label, old_label in new_to_original.items()
    }
    rotated_gold = next(
        new_label for new_label, old_label in new_to_original.items() if old_label == gold_answer
    )
    return rotated, rotated_gold, new_to_original


def _default_dataset() -> Path:
    return HERE.parent.parent / "daily_20260922T075500Z" / "dataset"


def _default_base_run() -> Path:
    return HERE.parent / "results" / "formal_release_v1" / "formal_2c289604aa45_test"


def _validate_base_run(base_run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    required = (
        base_run / "run_manifest.json",
        base_run / "prepared" / "prepared_inputs.jsonl",
        base_run / "private" / "evaluation_mapping.jsonl",
        base_run / "manifests" / "split_manifest.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _read_json(required[0])
    if manifest.get("evaluation_profile") != "formal_release":
        raise RuntimeError("base run is not a formal_release run")
    selection = manifest.get("selection", {})
    if selection.get("evaluation_split") != "test":
        raise RuntimeError("Circular is restricted to the frozen test split")
    split_sha = selection.get("split_manifest_sha256") or manifest.get("split_manifest_sha256")
    if split_sha != _sha256_file(required[3]):
        raise RuntimeError("base run split manifest SHA-256 mismatch")
    return manifest, _read_jsonl(required[1]), _read_jsonl(required[2])


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_circular(
    base_run: Path,
    dataset: Path,
    output_dir: Path | None = None,
    *,
    modes: set[str] | None = None,
    expected_samples_per_mode: int = 288,
) -> dict[str, Any]:
    base_run = base_run.resolve()
    dataset = dataset.resolve()
    modes = modes or {"all-images"}
    if not modes or not modes.issubset(SUPPORTED_MODES):
        raise ValueError(f"unsupported modes: {sorted(modes)}")
    if expected_samples_per_mode <= 0:
        raise ValueError("expected_samples_per_mode must be positive")
    output_dir = (output_dir or (base_run / "circular" / "cyclic4_v1")).resolve()
    base_manifest, base_public, base_private = _validate_base_run(base_run)
    base_public_by_id = {str(row.get("eval_id")): row for row in base_public}
    if len(base_public_by_id) != len(base_public):
        raise RuntimeError("base prepared inputs contain duplicate eval_id values")

    annotation_path = dataset / "annotations.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    records = _read_jsonl(annotation_path)
    records_by_sample = {str(row.get("sample_id")): row for row in records}
    if len(records_by_sample) != len(records):
        raise RuntimeError("annotations contain duplicate sample_id values")

    source_rows = [row for row in base_private if str(row.get("evaluation_mode")) in modes]
    per_mode = defaultdict(int)
    for row in source_rows:
        per_mode[str(row.get("evaluation_mode"))] += 1
    expected_counts = {mode: expected_samples_per_mode for mode in sorted(modes)}
    if dict(sorted(per_mode.items())) != expected_counts:
        raise RuntimeError(
            f"frozen source count mismatch: {dict(sorted(per_mode.items()))}; expected {expected_counts}"
        )
    group_ids = {str(row.get("relation_group_id")) for row in source_rows}
    if expected_samples_per_mode == 288 and len(group_ids) != 96:
        raise RuntimeError(f"frozen test group count must be 96, got {len(group_ids)}")

    protocol_binding = {
        "protocol_id": PROTOCOL_ID,
        "base_run": str(base_run),
        "base_run_manifest_sha256": _sha256_file(base_run / "run_manifest.json"),
        "base_prepared_sha256": _sha256_file(base_run / "prepared" / "prepared_inputs.jsonl"),
        "base_mapping_sha256": _sha256_file(base_run / "private" / "evaluation_mapping.jsonl"),
        "annotations_sha256": _sha256_file(annotation_path),
        "modes": sorted(modes),
        "rotations": [0, 1, 2, 3],
        "rotation_definition": "new_label[j] receives original_label[(j+rotation) mod 4]",
    }
    protocol_sha = _canonical_sha256(protocol_binding)
    if output_dir.exists():
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"non-empty or incomplete Circular output exists: {output_dir}")
        existing = _read_json(manifest_path)
        if existing.get("circular_protocol_sha256") != protocol_sha:
            raise RuntimeError("existing Circular run has a different protocol binding; use a new output directory")
        return existing

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".cyclic4_v1.", dir=output_dir.parent))
    try:
        public_rows: list[dict[str, Any]] = []
        private_rows: list[dict[str, Any]] = []
        copied_refs: set[str] = set()
        for source in source_rows:
            source_eval_id = str(source["eval_id"])
            public_source = base_public_by_id.get(source_eval_id)
            if public_source is None:
                raise RuntimeError(f"missing public base item: {source_eval_id}")
            sample_id = str(source["sample_id"])
            record = records_by_sample.get(sample_id)
            if record is None:
                raise RuntimeError(f"missing source annotation: {sample_id}")
            if dict(record.get("options", {})) != dict(source.get("options", {})):
                raise RuntimeError(f"annotation/base mapping option mismatch: {sample_id}")
            canonical_prompt = PROMPT_TEMPLATE.format(
                question=str(record["question"]), **dict(record["options"])
            )
            if canonical_prompt != str(public_source.get("prompt", "")):
                raise RuntimeError(f"rotation-0 prompt does not match frozen Vanilla input: {sample_id}")
            for image_ref in public_source.get("image_refs", []):
                image_ref = str(image_ref)
                if image_ref in copied_refs:
                    continue
                source_image = base_run / "prepared" / image_ref
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                _copy_or_link(source_image, stage / "prepared" / image_ref)
                copied_refs.add(image_ref)
            for rotation in range(4):
                options, gold, new_to_original = rotate_options(
                    dict(record["options"]), str(record["answer"]), rotation
                )
                eval_id = "circ_" + hashlib.sha256(
                    f"{PROTOCOL_ID}\0{source_eval_id}\0{rotation}".encode("utf-8")
                ).hexdigest()[:20]
                public_rows.append({
                    "eval_id": eval_id,
                    "evaluation_mode": str(source["evaluation_mode"]),
                    "prompt": PROMPT_TEMPLATE.format(question=str(record["question"]), **options),
                    "image_refs": list(public_source.get("image_refs", [])),
                })
                private_rows.append({
                    **source,
                    "eval_id": eval_id,
                    "source_eval_id": source_eval_id,
                    "circular_protocol_id": PROTOCOL_ID,
                    "circular_rotation": rotation,
                    "gold_answer": gold,
                    "gold_semantic_answer": str(record["answer"]),
                    "options": options,
                    "option_map_new_to_original": new_to_original,
                })

        order = sorted(range(len(public_rows)), key=lambda index: public_rows[index]["eval_id"])
        public_rows = [public_rows[index] for index in order]
        private_by_id = {row["eval_id"]: row for row in private_rows}
        private_rows = [private_by_id[row["eval_id"]] for row in public_rows]
        _write_jsonl(stage / "prepared" / "prepared_inputs.jsonl", public_rows)
        _write_jsonl(stage / "private" / "evaluation_mapping.jsonl", private_rows)
        (stage / "manifests").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_run / "manifests" / "split_manifest.csv", stage / "manifests" / "split_manifest.csv")
        base_summary = base_run / "manifests" / "split_summary.json"
        if base_summary.is_file():
            shutil.copy2(base_summary, stage / "manifests" / "split_summary.json")

        selection = dict(base_manifest.get("selection", {}))
        manifest = {
            "schema_version": "frameevidence-circular-run-v1",
            "evaluation_profile": "formal_release",
            "result_label": "formal_release_circular_v1",
            "dataset_fingerprint": base_manifest.get("dataset_fingerprint"),
            "dataset": str(dataset),
            "seed": base_manifest.get("seed"),
            "prompt_template_sha256": _sha256_bytes(
                (PROMPT_TEMPLATE + "\n" + PROTOCOL_ID).encode("utf-8")
            ),
            "evaluation_split": "test",
            "split_manifest_sha256": selection.get("split_manifest_sha256"),
            "allowed_group_ids_sha256": selection.get("allowed_group_ids_sha256"),
            "source_samples": len({str(row["sample_id"]) for row in source_rows}),
            "public_items": len(public_rows),
            "evaluation_modes": sorted(modes),
            "ordering": "deterministic eval_id order; image order inherited byte-for-byte from Vanilla",
            "single_image_rule": base_manifest.get("single_image_rule", "inherited from Vanilla"),
            "selection": {
                **selection,
                "evaluation_split": "test",
                "selected_group_count": len(group_ids),
            },
            "circular_protocol": protocol_binding,
            "circular_protocol_sha256": protocol_sha,
            "base_run": str(base_run),
        }
        _write_json(stage / "run_manifest.json", manifest)
        os.replace(stage, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "value": successes / total if total else None,
        "numerator": successes,
        "denominator": total,
        "ci95_wilson": _wilson(successes, total),
    }


def _validate_circular_mapping(mapping: list[dict[str, Any]]) -> None:
    eval_ids: set[str] = set()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relation_rotation_maps: dict[tuple[str, int], dict[str, str]] = {}
    for row in mapping:
        eval_id = str(row.get("eval_id", ""))
        source_eval_id = str(row.get("source_eval_id", ""))
        if not eval_id or eval_id in eval_ids or not source_eval_id:
            raise RuntimeError("Circular mapping has blank/duplicate eval_id or blank source_eval_id")
        eval_ids.add(eval_id)
        by_source[source_eval_id].append(row)
        rotation = row.get("circular_rotation")
        if rotation not in range(4):
            raise RuntimeError(f"invalid Circular rotation for {eval_id}: {rotation}")
        mapping_new_to_old = row.get("option_map_new_to_original")
        if not isinstance(mapping_new_to_old, dict) or set(mapping_new_to_old) != set(LABELS):
            raise RuntimeError(f"invalid option map for {eval_id}")
        if set(mapping_new_to_old.values()) != set(LABELS):
            raise RuntimeError(f"option map is not bijective for {eval_id}")
        relation_rotation = (str(row.get("relation_group_id")), int(rotation))
        previous_map = relation_rotation_maps.setdefault(relation_rotation, mapping_new_to_old)
        if previous_map != mapping_new_to_old:
            raise RuntimeError(f"relation group changes permutation within rotation: {relation_rotation}")
        semantic_gold = str(row.get("gold_semantic_answer", ""))
        displayed_gold = str(row.get("gold_answer", ""))
        if mapping_new_to_old.get(displayed_gold) != semantic_gold:
            raise RuntimeError(f"remapped gold is inconsistent for {eval_id}")
    for source_eval_id, rows in by_source.items():
        rotations = {row.get("circular_rotation") for row in rows}
        if len(rows) != 4 or rotations != set(range(4)):
            raise RuntimeError(f"{source_eval_id} does not contain exactly rotations 0..3")
        stable_fields = ("sample_id", "relation_group_id", "evaluation_mode", "question_type")
        for field in stable_fields:
            if len({str(row.get(field)) for row in rows}) != 1:
                raise RuntimeError(f"{source_eval_id} changes {field} across rotations")


def _score_subset(
    mapping: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scored_rows: list[dict[str, Any]] = []
    rotation_correct = {rotation: 0 for rotation in range(4)}
    valid = 0
    total_correct = 0
    for row in mapping:
        prediction = predictions.get(str(row["eval_id"]), {})
        answer = prediction.get("parsed_answer") or prediction.get("answer")
        answer = str(answer) if answer in LABELS else None
        correct = answer == row["gold_answer"]
        semantic = row["option_map_new_to_original"].get(answer) if answer else None
        scored = {**row, "predicted_answer": answer, "predicted_semantic_answer": semantic, "correct": correct}
        scored_rows.append(scored)
        by_source[str(row["source_eval_id"])].append(scored)
        by_relation[str(row["relation_group_id"])].append(scored)
        valid += int(answer in LABELS)
        total_correct += int(correct)
        rotation_correct[int(row["circular_rotation"])] += int(correct)
    strict = 0
    consistent = 0
    vanilla_correct = 0
    complete_groups = 0
    for source_rows in by_source.values():
        rotations = {int(row["circular_rotation"]): row for row in source_rows}
        if set(rotations) != set(range(4)):
            continue
        complete_groups += 1
        vanilla_correct += int(rotations[0]["correct"])
        strict += int(all(rotations[index]["correct"] for index in range(4)))
        semantic_answers = [rotations[index]["predicted_semantic_answer"] for index in range(4)]
        consistent += int(all(value in LABELS for value in semantic_answers) and len(set(semantic_answers)) == 1)
    permutation_correct = sum(rotation_correct[index] for index in (1, 2, 3))
    all_rotations_valid = sum(
        all(row["predicted_answer"] in LABELS for row in source_rows)
        for source_rows in by_source.values()
        if len(source_rows) == 4
    )
    fq1_total = fq1_correct = fq2_total = nts_correct = dis_correct = 0
    for relation_rows in by_relation.values():
        question_types = {str(row.get("question_type")) for row in relation_rows}
        if question_types == {"FQ1"}:
            fq1_total += 1
            fq1_correct += int(len(relation_rows) == 12 and all(row["correct"] for row in relation_rows))
        elif question_types == {"FQ2"}:
            fq2_total += 1
            by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in relation_rows:
                by_condition[str(row.get("view_condition"))].append(row)
            required = {
                "one_valid_mss": by_condition.get("one_valid_mss", []),
                "remove_required_view": by_condition.get("remove_required_view", []),
                "hard_distractor_added": by_condition.get("hard_distractor_added", []),
            }
            nts_rows = required["one_valid_mss"] + required["remove_required_view"]
            dis_rows = required["one_valid_mss"] + required["hard_distractor_added"]
            nts_correct += int(len(nts_rows) == 8 and all(row["correct"] for row in nts_rows))
            dis_correct += int(len(dis_rows) == 8 and all(row["correct"] for row in dis_rows))

    result = {
        "coverage": {
            "valid": valid,
            "expected": len(mapping),
            "invalid_or_missing": len(mapping) - valid,
        },
        "vanilla_accuracy": _metric(vanilla_correct, complete_groups),
        "mean_rotation_accuracy": _metric(total_correct, len(mapping)),
        "permuted_only_accuracy": _metric(permutation_correct, complete_groups * 3),
        "strict_circular_accuracy": _metric(strict, complete_groups),
        "semantic_consistency": _metric(consistent, complete_groups),
        "all_rotations_valid_rate": _metric(all_rotations_valid, complete_groups),
        "rotation_accuracy": {
            str(index): _metric(rotation_correct[index], complete_groups)
            for index in range(4)
        },
        "circular_drop": (
            (vanilla_correct - strict) / complete_groups if complete_groups else None
        ),
        "strict_group_metrics": {
            "FQ1_GEM_all_rotations": _metric(fq1_correct, fq1_total),
            "NTS_all_rotations": _metric(nts_correct, fq2_total),
            "DIS_all_rotations": _metric(dis_correct, fq2_total),
        },
    }
    question_types = sorted({str(row.get("question_type")) for row in scored_rows})
    if len(question_types) > 1:
        result["by_question_type"] = {
            question_type: _score_subset(
                [row for row in mapping if str(row.get("question_type")) == question_type],
                predictions,
            )
            for question_type in question_types
        }
    return result


def score_circular(
    run_dir: Path,
    *,
    model_ids: set[str] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("circular_protocol", {}).get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("run directory is not a compatible Circular run")
    mapping = _read_jsonl(run_dir / "private" / "evaluation_mapping.jsonl")
    _validate_circular_mapping(mapping)
    mapping_by_mode = {
        mode: [row for row in mapping if row.get("evaluation_mode") == mode]
        for mode in SUPPORTED_MODES
        if any(row.get("evaluation_mode") == mode for row in mapping)
    }
    display_names: dict[str, str] = {}
    if config_path is not None:
        config = _read_json(config_path.resolve())
        display_names = {
            str(row.get("model_id")): str(row.get("display_name") or row.get("model_id"))
            for row in config.get("models", [])
            if isinstance(row, dict)
        }
    models: list[dict[str, Any]] = []
    for prediction_path in sorted((run_dir / "predictions").glob("*.jsonl")):
        rows = _read_jsonl(prediction_path)
        row_model_ids = {str(row.get("model_id")) for row in rows if row.get("model_id")}
        if len(row_model_ids) != 1:
            raise RuntimeError(f"prediction file has zero or multiple model_id values: {prediction_path}")
        model_id = next(iter(row_model_ids))
        if model_ids and model_id not in model_ids:
            continue
        by_eval: dict[str, dict[str, Any]] = {}
        for row in rows:
            eval_id = str(row.get("eval_id", ""))
            if not eval_id or eval_id in by_eval:
                raise RuntimeError(f"duplicate or blank eval_id in {prediction_path}")
            by_eval[eval_id] = row
        mode_results: dict[str, Any] = {}
        complete = True
        for mode, mode_mapping in mapping_by_mode.items():
            result = _score_subset(mode_mapping, by_eval)
            complete &= result["coverage"]["invalid_or_missing"] == 0
            mode_results[mode] = result
        models.append({
            "model_id": model_id,
            "display_name": display_names.get(model_id, model_id),
            "prediction_file": str(prediction_path),
            "prediction_sha256": _sha256_file(prediction_path),
            "status": "PASS" if complete else "INCOMPLETE",
            "modes": mode_results,
        })
    if not models:
        raise RuntimeError("no matching Circular prediction files found")
    status = "PASS" if all(model["status"] == "PASS" for model in models) else "INCOMPLETE"
    result = {
        "schema_version": "frameevidence-circular-metrics-v1",
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "circular_protocol_sha256": manifest["circular_protocol_sha256"],
        "mapping_sha256": _sha256_file(run_dir / "private" / "evaluation_mapping.jsonl"),
        "models": models,
    }
    _write_json(run_dir / "analysis" / "circular_metrics.json", result)
    _write_circular_tables(result, run_dir / "tables")
    return result


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def _latex(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def _write_circular_tables(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "circular_results.csv"
    rows: list[dict[str, Any]] = []
    for model in metrics["models"]:
        for mode, result in model["modes"].items():
            rows.append({
                "model_id": model["model_id"],
                "display_name": model["display_name"],
                "evaluation_mode": mode,
                "valid": result["coverage"]["valid"],
                "expected": result["coverage"]["expected"],
                "vanilla_accuracy": result["vanilla_accuracy"]["value"],
                "mean_rotation_accuracy": result["mean_rotation_accuracy"]["value"],
                "permuted_only_accuracy": result["permuted_only_accuracy"]["value"],
                "strict_circular_accuracy": result["strict_circular_accuracy"]["value"],
                "semantic_consistency": result["semantic_consistency"]["value"],
                "circular_drop": result["circular_drop"],
            })
    fields = list(rows[0])
    lines = [",".join(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            text = "" if value is None else str(value)
            values.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(values))
    _atomic_text(csv_path, "\n".join(lines) + "\n")

    header = "| Model | Mode | Vanilla | Mean rot. | Circular (all 4) | Consistency | Drop |\n|---|---|---:|---:|---:|---:|---:|"
    markdown_rows = [header]
    latex_rows = []
    for row in rows:
        markdown_rows.append(
            f"| {row['display_name']} | {row['evaluation_mode']} | "
            f"{_pct(row['vanilla_accuracy'])} | {_pct(row['mean_rotation_accuracy'])} | "
            f"{_pct(row['strict_circular_accuracy'])} | {_pct(row['semantic_consistency'])} | "
            f"{_pct(row['circular_drop'])} |"
        )
        latex_rows.append(
            f"{_latex(str(row['display_name']))} & {_latex(str(row['evaluation_mode']))} & "
            f"{_pct(row['vanilla_accuracy'])} & {_pct(row['mean_rotation_accuracy'])} & "
            f"{_pct(row['strict_circular_accuracy'])} & {_pct(row['semantic_consistency'])} & "
            f"{_pct(row['circular_drop'])} \\\\"
        )
    _atomic_text(output_dir / "circular_results.md", "\n".join(markdown_rows) + "\n")
    latex = """\\begin{table*}[t]
  \\centering
  \\caption{Circular answer-order robustness on FrameEvidence. Circular requires all four cyclic option rotations to be correct; Drop is Vanilla minus Circular in percentage points.}
  \\label{tab:frameevidence-circular}
  \\small
  \\begin{tabular}{llrrrrr}
    \\hline
    Model & Mode & Vanilla & Mean rot. & Circular & Consistency & Drop \\\\
    \\hline
""" + "\n".join(f"    {row}" for row in latex_rows) + """
    \\hline
  \\end{tabular}
\\end{table*}
"""
    _atomic_text(output_dir / "circular_results.tex", latex)


def command_prepare(args: argparse.Namespace) -> int:
    result = prepare_circular(
        args.base_run,
        args.dataset,
        args.output_dir,
        modes=set(args.modes),
        expected_samples_per_mode=args.expected_samples_per_mode,
    )
    print(json.dumps({
        "status": "PASS",
        "output_dir": str((args.output_dir or (args.base_run / "circular" / "cyclic4_v1")).resolve()),
        "public_items": result["public_items"],
        "protocol_sha256": result["circular_protocol_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


def command_check(args: argparse.Namespace) -> int:
    result = check_model_config(
        args.config.resolve(),
        set(args.models) if args.models else None,
        require_env=not args.skip_env_check,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


def command_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("circular_protocol", {}).get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("prepare the Circular run before invoking models")
    if args.limit is not None:
        raise ValueError("Circular rejects --limit because it can split a four-rotation bundle; use --limit-groups")
    config_snapshot = run_dir / "config" / "models.circular.snapshot.json"
    config_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config.resolve(), config_snapshot)
    result = run_models(
        run_dir,
        args.config.resolve(),
        {"shortcut_baselines": {"overall_risk": "AUDITED_SEPARATELY"}, "known_limitations": []},
        set(args.models) if args.models else None,
        set(args.modes),
        args.limit,
        args.retry_failed,
        args.limit_groups,
        evaluation_profile="formal_release",
        result_label="formal_release_circular_v1",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


def command_score(args: argparse.Namespace) -> int:
    result = score_circular(
        args.run_dir,
        model_ids=set(args.models) if args.models else None,
        config_path=args.config,
    )
    print(json.dumps({
        "status": result["status"],
        "models": len(result["models"]),
        "analysis": str((args.run_dir / "analysis" / "circular_metrics.json").resolve()),
        "table": str((args.run_dir / "tables" / "circular_results.tex").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base-run", type=Path, default=_default_base_run())
    prepare.add_argument("--dataset", type=Path, default=_default_dataset())
    prepare.add_argument("--output-dir", type=Path, default=None)
    prepare.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=["all-images"])
    prepare.add_argument("--expected-samples-per-mode", type=int, default=288)
    prepare.set_defaults(handler=command_prepare)

    check = commands.add_parser("check-config")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--models", nargs="+", default=None)
    check.add_argument("--skip-env-check", action="store_true")
    check.set_defaults(handler=command_check)

    run = commands.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--models", nargs="+", default=None)
    run.add_argument("--modes", nargs="+", choices=SUPPORTED_MODES, default=["all-images"])
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--limit-groups", type=int, default=None)
    run.add_argument("--retry-failed", action="store_true")
    run.set_defaults(handler=command_run)

    score = commands.add_parser("score")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--config", type=Path, default=None)
    score.add_argument("--models", nargs="+", default=None)
    score.set_defaults(handler=command_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
