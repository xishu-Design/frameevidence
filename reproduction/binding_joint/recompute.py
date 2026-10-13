"""Recompute paired FQ2 joint label agreement from retained requests only.

Python standard library. No network, inference, image access or source writes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ARMS = ("original_wording", "stable_id_candidate")
CONDITIONS = ("one_valid_mss", "remove_required_view", "hard_distractor_added")
METRICS = ("MSS", "Deletion", "Addition", "NTS", "DIS", "All3")
EXPECTED_STRATA = {"anchor_retained_shifted": 6, "ordinal_stable": 2, "no_ordinal": 4}
HERE = Path(__file__).resolve().parent


def require(test, message):
    if not test:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def unique(rows, key="eval_id"):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, f"Duplicate {key}: {value}")
        result[value] = row
    return result


def group_metrics(matches):
    require(set(matches) == set(CONDITIONS), "Each group requires exactly three conditions")
    mss, deletion, addition = (bool(matches[name]) for name in CONDITIONS)
    return dict(zip(METRICS, (mss, deletion, addition, mss and deletion,
                              mss and addition, mss and deletion and addition)))


def normalize(metadata, predictions, cohort, expected_blocks, expected_groups=12, check_model=True):
    """Join inputs to outcomes, keeping invalid outcomes in allocated denominators."""
    meta, pred = unique(metadata), unique(predictions)
    require(set(meta) == set(pred), "Prediction IDs differ from allocated metadata IDs")
    require(len(meta) == expected_groups * 3 * 2 * len(expected_blocks), "Incorrect allocated count")
    cells, pairs, strata, sources = {}, {}, {}, {}
    rows = []
    for eid, item in sorted(meta.items()):
        block = item.get("block", 1)
        arm = item.get("diagnostic_arm", item.get("arm"))
        gid, condition, source = item["group_id"], item["condition"], item["source_eval_id"]
        label, stratum = item["released_label_reference"], item["stratum"]
        require(block in expected_blocks and arm in ARMS, "Unexpected block or arm")
        require(condition in CONDITIONS and label in tuple("ABCD"), "Invalid condition or label")
        require(item.get("new_semantic_gold") is None, "New semantic gold may not be introduced")
        require(gid not in strata or strata[gid] == stratum, "Within-group stratum mismatch")
        strata[gid] = stratum
        source_key = (gid, condition)
        require(source_key not in sources or sources[source_key] == (source, label), "Across-arm/block source or label mismatch")
        sources[source_key] = source, label
        require(source not in pairs or pairs[source] == source_key, "Source reused across conditions or groups")
        pairs[source] = source_key
        key = (block, arm, gid, condition)
        require(key not in cells, "Duplicate block/arm/group/condition")
        cells[key] = eid
        outcome = pred[eid]
        require(outcome.get("evaluation_mode") == "all-images", "Unexpected evaluation mode")
        for name, expected in (("block", block), ("diagnostic_arm", arm), ("source_eval_id", source)):
            require(name not in outcome or outcome[name] == expected, f"Prediction {name} mismatch")
        require(outcome.get("status") in ("OK", "PARSE_ERROR", "CALL_ERROR"), "Unknown response status")
        valid = outcome["status"] == "OK" and outcome.get("parsed_answer") in tuple("ABCD")
        rows.append({"cohort": cohort, "block": block, "arm": arm, "group_id": gid,
                     "condition": condition, "stratum": stratum, "source_eval_id": source,
                     "eval_id": eid, "released_label": label, "status": outcome["status"],
                     "parsed_answer": outcome.get("parsed_answer"), "valid": valid,
                     "label_match": valid and outcome.get("parsed_answer") == label})
    require(len(strata) == expected_groups, "Incorrect group count")
    for block in expected_blocks:
        for arm in ARMS:
            for gid in strata:
                require(all((block, arm, gid, c) in cells for c in CONDITIONS), "Incomplete paired triplet")
    if check_model:
        model_ids = {(x.get("model_id"), x.get("api_model")) for x in predictions}
        require(len(model_ids) == 1 and all(next(iter(model_ids))), "Mixed or missing model identity")
    return rows


def calculate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"], row["block"], row["arm"], row["group_id"]].append(row)
    group_rows = []
    for (cohort, block, arm, gid), values in sorted(groups.items()):
        require(len(values) == 3, "Wrong triplet size")
        matches = {v["condition"]: v["label_match"] for v in values}
        require(len(matches) == 3, "Repeated condition")
        group_rows.append({"cohort": cohort, "block": block, "arm": arm, "group_id": gid,
                           "stratum": values[0]["stratum"], "valid_responses": sum(v["valid"] for v in values),
                           **{key: int(value) for key, value in group_metrics(matches).items()}})
    summary = []
    for cohort, block, arm in sorted({(x["cohort"], x["block"], x["arm"]) for x in rows}):
        selected = [x for x in group_rows if (x["cohort"], x["block"], x["arm"]) == (cohort, block, arm)]
        n = len(selected)
        record = {"cohort": cohort, "block": block, "arm": arm, "groups": n, "items": 3*n,
                  "valid_responses": sum(x["valid_responses"] for x in selected)}
        record["label_matches"] = sum(sum(x[k] for k in METRICS[:3]) for x in selected)
        record["label_match_percent"] = 100 * record["label_matches"] / (3*n)
        for metric in METRICS:
            record[metric] = sum(x[metric] for x in selected)
            record[metric + "_percent"] = 100 * record[metric] / n
        summary.append(record)
    paired_groups, paired_summary = [], []
    for cohort, block in sorted({(x["cohort"], x["block"]) for x in group_rows}):
        by_arm = {arm: {x["group_id"]: x for x in group_rows if (x["cohort"], x["block"], x["arm"]) == (cohort, block, arm)} for arm in ARMS}
        require(set(by_arm[ARMS[0]]) == set(by_arm[ARMS[1]]), "Unpaired arm groups")
        for metric in METRICS:
            selected = []
            for gid in sorted(by_arm[ARMS[0]]):
                a, b = (by_arm[arm][gid][metric] for arm in ARMS)
                state = "both" if a and b else "gain" if b else "loss" if a else "neither"
                selected.append({"cohort": cohort, "block": block, "metric": metric, "group_id": gid,
                                 "original": a, "stable_id": b, "outcome": state, "difference": b-a})
            paired_groups.extend(selected)
            counts = Counter(x["outcome"] for x in selected)
            paired_summary.append({"cohort": cohort, "block": block, "metric": metric, "groups": len(selected),
                                   **{state: counts[state] for state in ("both", "gain", "loss", "neither")},
                                   "difference_pp": 100 * (counts["gain"]-counts["loss"]) / len(selected)})
    return summary, group_rows, paired_summary, paired_groups


def pooled_counterbalanced(summary, paired_summary):
    """Equal-weight repeated-block aggregates, not 24 independent groups."""
    result = {"unique_groups": 12, "repeated_blocks": 2, "group_block_observations": 24, "arms": {}, "paired": {}}
    for arm in ARMS:
        selected = [r for r in summary if r["cohort"] == "counterbalanced144" and r["arm"] == arm]
        require({r["block"] for r in selected} == {1, 2} and len(selected) == 2, "Need two complete blocks")
        totals = {k: sum(r[k] for r in selected) for k in ("items", "valid_responses", "label_matches", *METRICS)}
        totals["label_match_percent"] = 100 * totals["label_matches"] / totals["items"]
        totals.update({k + "_percent": 100 * totals[k] / 24 for k in METRICS})
        result["arms"][arm] = totals
    for metric in METRICS:
        selected = [r for r in paired_summary if r["cohort"] == "counterbalanced144" and r["metric"] == metric]
        totals = {k: sum(r[k] for r in selected) for k in ("both", "gain", "loss", "neither")}
        totals["difference_pp"] = 100 * (totals["gain"]-totals["loss"]) / 24
        result["paired"][metric] = totals
    return result


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, help="Optional original paper root to rebuild and verify the source snapshot")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {"pilot_metadata": "materials/binding_validation_20261010/prepared_72.metadata.jsonl",
             "pilot_predictions": "materials/model_supplements_20261010/gpt_binding_stable72/predictions.jsonl",
             "counterbalanced_schedule": "materials/diagnostic_controls_20261011/binding_counterbalanced144/schedule.jsonl",
             "counterbalanced_predictions": "materials/diagnostic_controls_20261011/binding_counterbalanced144/predictions.jsonl"}
    if args.paper_root:
        paper = args.paper_root.resolve()
        before = {k: hashlib.sha256((paper / p).read_bytes()).hexdigest() for k, p in paths.items()}
        pilot = normalize(read_rows(paper / paths["pilot_metadata"]), read_rows(paper / paths["pilot_predictions"]), "pilot72", (1,))
        counterbalanced = normalize(read_rows(paper / paths["counterbalanced_schedule"]), read_rows(paper / paths["counterbalanced_predictions"]), "counterbalanced144", (1, 2))
        fields = ("cohort", "block", "arm", "group_id", "condition", "stratum", "source_eval_id", "eval_id", "released_label", "status", "parsed_answer")
        snapshot = [{k: row[k] for k in fields} for row in pilot + counterbalanced]
        snapshot_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in snapshot)
        (out / "inputs.jsonl").write_text(snapshot_text, encoding="utf-8", newline="\n")
        sources = {"paths_relative_to_paper_root": paths, "source_digests": before,
                   "snapshot_digest": hashlib.sha256(snapshot_text.encode()).hexdigest()}
        (out / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
        after = {k: hashlib.sha256((paper / p).read_bytes()).hexdigest() for k, p in paths.items()}
        require(before == after, "Source changed during scoring")
    else:
        sources = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
        require(hashlib.sha256((HERE / "inputs.jsonl").read_bytes()).hexdigest() == sources["snapshot_digest"], "Input snapshot changed")
        snapshot = read_rows(HERE / "inputs.jsonl")
        before = sources["source_digests"]
        paths = sources["paths_relative_to_paper_root"]
        def from_snapshot(cohort, blocks):
            chosen = [row for row in snapshot if row["cohort"] == cohort]
            metadata = [dict(row, released_label_reference=row["released_label"]) for row in chosen]
            predictions = [dict(row, evaluation_mode="all-images") for row in chosen]
            return normalize(metadata, predictions, cohort, blocks, check_model=False)
        require({r["cohort"] for r in snapshot} == {"pilot72", "counterbalanced144"}, "Snapshot cohort mismatch")
        pilot, counterbalanced = from_snapshot("pilot72", (1,)), from_snapshot("counterbalanced144", (1, 2))
    for rows in (pilot, counterbalanced):
        strata = {r["group_id"]: r["stratum"] for r in rows}
        require(dict(Counter(strata.values())) == EXPECTED_STRATA, "Subset no longer has the frozen 6/2/4 composition")
    pilot_sources = {(r["group_id"], r["condition"]): (r["source_eval_id"], r["released_label"]) for r in pilot}
    new_sources = {(r["group_id"], r["condition"]): (r["source_eval_id"], r["released_label"]) for r in counterbalanced}
    require(pilot_sources == new_sources, "Pilot and repeated blocks no longer use the same source cases/labels")
    summary, groups, paired, paired_groups = calculate(pilot + counterbalanced)
    pooled = pooled_counterbalanced(summary, paired)
    result = {"status": "PASS", "estimand": "Agreement with unchanged released labels, not newly adjudicated semantic gold",
              "primary_evidence": "counterbalanced144", "pilot_separate": True,
              "scope": {"unique_groups": 12, "total_fq2_test_groups": 48, "removed_anchor_groups_excluded": 36,
                        "strata": EXPECTED_STRATA, "new_model_calls": 0, "new_human_records": 0,
                        "new_semantic_gold": False, "full_fq2_repaired": False},
              "source_paths_relative_to_paper_root": paths, "source_digests": before,
              "summary": summary, "paired_summary": paired, "pooled_counterbalanced": pooled}
    result["sources_unchanged"] = True
    result["source_verification"] = "original files rechecked" if args.paper_root else "verified sanitized snapshot"
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, data in (("metrics_by_block_arm.csv", summary), ("group_outcomes.csv", groups),
                       ("paired_gains_losses.csv", paired), ("paired_group_outcomes.csv", paired_groups),
                       ("scoring_inputs.csv", pilot + counterbalanced)):
        write_csv(out / name, data)
    lines = ["# Fixed-ID joint diagnostic", "", "## Two counterbalanced blocks (primary)", "",
             "Each arm in each block contains the same 12 complete FQ2 groups (36 items). All counts below use released labels.", "",
             "| Block | Arm | Item match | MSS | Deletion | Addition | NTS | DIS | All three |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["cohort"] == "counterbalanced144":
            lines.append(f"| {r['block']} | {r['arm']} | {r['label_matches']}/36 | " + " | ".join(f"{r[k]}/12" for k in METRICS) + " |")
    lines += ["", "### Paired gains and losses", "", "Stable-ID minus original. Each row pairs the same 12 groups.", "",
              "| Block | Metric | Both | Gains | Losses | Neither | Difference (pp) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        if r["cohort"] == "counterbalanced144":
            lines.append(f"| {r['block']} | {r['metric']} | {r['both']} | {r['gain']} | {r['loss']} | {r['neither']} | {r['difference_pp']:+.2f} |")
    lines += ["", "### Equal-weight two-block summary", "",
              "The two blocks are repeated observations of the same 12 groups. Counts below use 24 group-block observations.", "",
              "| Prompt | Item match /72 | NTS /24 | DIS /24 | All three /24 |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        v = pooled["arms"][arm]
        lines.append(f"| {arm} | {v['label_matches']} ({v['label_match_percent']:.2f}%) | {v['NTS']} ({v['NTS_percent']:.2f}%) | {v['DIS']} ({v['DIS_percent']:.2f}%) | {v['All3']} ({v['All3_percent']:.2f}%) |")
    lines += ["", "The small item-level increase (37 to 38 released-label matches) does not extend to joint success. NTS decreases in both blocks, DIS is unchanged then lower, and all-three success decreases then increases. In the two-block aggregate, NTS has two paired gains and six losses, DIS has two gains and three losses, and all-three success has two gains and two losses. Thus this fixed-ID rewrite does not show a consistent joint-agreement benefit on the retained subset. The comparison does not isolate wording effects from execution variability."]
    lines += ["", "## Initial 72-request pilot (not pooled)", "", "| Arm | Item match | MSS | Deletion | Addition | NTS | DIS | All three |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["cohort"] == "pilot72":
            lines.append(f"| {r['arm']} | {r['label_matches']}/36 | " + " | ".join(f"{r[k]}/12" for k in METRICS) + " |")
    lines += ["", "## Scope", "", "The 12 groups comprise six retained-but-shifted anchors, two stable ordinals and four questions without ordinal references. The 36 removed-anchor groups are excluded. Stable-ID wording adds mappings and instructions as well as changing the reference expression. No labels were re-adjudicated. These scores therefore measure released-label joint agreement on a targeted subset, not a repaired full-FQ2 benchmark.", "", "The pooled file reports 24 group-block observations from 12 unique groups, not 24 independent groups. The initial pilot is separate. No API calls or human responses were generated.", "", "## Recompute", "", "Run `python recompute.py` and `python -m unittest -v test_recompute`. Default execution reads the sanitized `inputs.jsonl` and needs no repository, model, image or third-party library. Optional `--paper-root PATH` rebuilds the snapshot from original read-only files and checks their unchanged contents. `sources.json` records the original relative paths. The snapshot has only request identity, group, condition, block, arm, released label, status and parsed answer fields."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tex = [r"\begin{table}[t]", r"\centering", r"\caption{Fixed-ID joint released-label agreement on 12 FQ2 groups. Counts are per block, not independent replications.}",
           r"\label{tab:binding-joint-subset}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrr}",
           r"\hline", r"Block & Prompt & Match/36 & NTS/12 & DIS/12 & All3/12 \\", r"\hline"]
    for r in summary:
        if r["cohort"] == "counterbalanced144":
            arm = "Original" if r["arm"] == ARMS[0] else "Fixed IDs"
            tex.append(f"{r['block']} & {arm} & {r['label_matches']} & {r['NTS']} & {r['DIS']} & {r['All3']} " + r"\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (out / "table_binding_joint.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "sources_unchanged": True, "summary": summary, "pooled_counterbalanced": pooled}, indent=2))


if __name__ == "__main__":
    main()
