#!/usr/bin/env python3
"""Aggregate exactly three FrameEvidence LoRA seeds as mean and sample SD."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


VANILLA_METRICS = ("item_accuracy", "FQ1_GEM", "NTS", "DIS")
CIRCULAR_METRICS = (
    "vanilla_accuracy",
    "mean_rotation_accuracy",
    "strict_circular_accuracy",
    "semantic_consistency",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _value(v: Any, label: str) -> float:
    if not isinstance(v, dict) or not isinstance(v.get("value"), (int, float)):
        raise RuntimeError(f"missing numeric metric: {label}")
    return float(v["value"])


def _summary(values: list[float]) -> dict[str, Any]:
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("three finite seed values are required")
    return {"values": values, "mean": mean(values), "sample_sd": stdev(values), "n_seeds": 3}


def aggregate(
    formal_metrics: Path,
    config_path: Path,
    circular_metrics: Path | None = None,
) -> dict[str, Any]:
    formal = _read(formal_metrics)
    config = _read(config_path)
    formal_by_id = {str(row["model_id"]): row for row in formal.get("models", [])}
    circular_by_id: dict[str, dict[str, Any]] = {}
    if circular_metrics is not None:
        circular = _read(circular_metrics)
        circular_by_id = {str(row["model_id"]): row for row in circular.get("models", [])}

    configured: dict[str, list[dict[str, Any]]] = {}
    for row in config.get("models", []):
        if not isinstance(row, dict) or row.get("variant") != "lora":
            continue
        if not isinstance(row.get("training_seed"), int):
            raise RuntimeError(f"LoRA row lacks integer training_seed: {row.get('model_id')}")
        configured.setdefault(str(row["family"]), []).append(row)
    if not configured:
        raise RuntimeError("no LoRA seed rows in config")

    families: list[dict[str, Any]] = []
    for family, rows in sorted(configured.items()):
        rows.sort(key=lambda row: int(row["training_seed"]))
        seeds = [int(row["training_seed"]) for row in rows]
        if len(rows) != 3 or len(set(seeds)) != 3:
            raise RuntimeError(f"{family} must have exactly three unique seed rows")
        family_result: dict[str, Any] = {
            "family": family,
            "seeds": seeds,
            "model_ids": [str(row["model_id"]) for row in rows],
            "modes": {},
        }
        for mode in ("all-images", "text-only", "single-image"):
            mode_metrics: dict[str, Any] = {}
            for metric in VANILLA_METRICS:
                values: list[float] = []
                for row in rows:
                    model_id = str(row["model_id"])
                    model_result = formal_by_id.get(model_id)
                    if model_result is None:
                        raise RuntimeError(f"formal metrics missing {model_id}")
                    mode_result = model_result.get("results", {}).get(mode)
                    if not isinstance(mode_result, dict) or mode_result.get("status") != "PASS":
                        raise RuntimeError(f"formal result is not PASS: {model_id}/{mode}")
                    metric_record = (
                        mode_result.get("item_accuracy")
                        if metric == "item_accuracy"
                        else mode_result.get("main_metrics", {}).get(metric)
                    )
                    values.append(_value(metric_record, f"{model_id}/{mode}/{metric}"))
                mode_metrics[metric] = _summary(values)
            family_result["modes"][mode] = mode_metrics
        if circular_metrics is not None:
            circular_summary: dict[str, Any] = {}
            for metric in CIRCULAR_METRICS:
                values = []
                for row in rows:
                    model_id = str(row["model_id"])
                    model_result = circular_by_id.get(model_id)
                    if model_result is None or model_result.get("status") != "PASS":
                        raise RuntimeError(f"Circular result is missing/not PASS: {model_id}")
                    values.append(
                        _value(
                            model_result.get("modes", {}).get("all-images", {}).get(metric),
                            f"{model_id}/Circular/{metric}",
                        )
                    )
                circular_summary[metric] = _summary(values)
            family_result["circular_all_images"] = circular_summary
        families.append(family_result)
    return {
        "schema_version": "frameevidence-lora-three-seed-aggregate-v1",
        "aggregation": "arithmetic mean and sample standard deviation across exactly three frozen seeds",
        "families": families,
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def write_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "lora_three_seed_aggregate.json"
    csv_path = output_dir / "lora_three_seed_aggregate.csv"
    tex_path = output_dir / "lora_three_seed_aggregate.tex"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for family in result["families"]:
        for mode, metrics in family["modes"].items():
            for metric, summary in metrics.items():
                rows.append({"family": family["family"], "protocol": "Vanilla", "mode": mode, "metric": metric, **summary})
        for metric, summary in family.get("circular_all_images", {}).items():
            rows.append({"family": family["family"], "protocol": "Circular", "mode": "all-images", "metric": metric, **summary})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["family", "protocol", "mode", "metric", "mean", "sample_sd", "n_seeds", "values"])
        writer.writeheader()
        writer.writerows(rows)
    latex_rows = []
    for family in result["families"]:
        metrics = family["modes"]["all-images"]
        values = []
        for key in VANILLA_METRICS:
            summary = metrics[key]
            values.append(f"{_pct(summary['mean'])} $\\pm$ {_pct(summary['sample_sd'])}")
        circular = family.get("circular_all_images", {}).get("strict_circular_accuracy")
        circular_value = "--" if circular is None else f"{_pct(circular['mean'])} $\\pm$ {_pct(circular['sample_sd'])}"
        family_name = family["family"].replace("_", r"\_")
        latex_rows.append(
            f"    {family_name} & " + " & ".join(values) + f" & {circular_value} \\\\"
        )
    tex = """\\begin{table*}[t]
  \\centering
  \\caption{Three-seed FrameEvidence LoRA results (mean $\\pm$ sample standard deviation, percentage points).}
  \\label{tab:lora-three-seed}
  \\small
  \\begin{tabular}{lrrrrr}
    \\hline
    Family & Item Acc. & FQ1-GEM & NTS & DIS & Circular \\\\
    \\hline
""" + "\n".join(latex_rows) + """
    \\hline
  \\end{tabular}
\\end{table*}
"""
    tex_path.write_text(tex, encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "latex": str(tex_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--circular-metrics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.formal_metrics, args.config, args.circular_metrics)
    outputs = write_outputs(result, args.output_dir)
    print(json.dumps({"status": "PASS", "families": len(result["families"]), "outputs": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
