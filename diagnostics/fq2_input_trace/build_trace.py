"""Rebuild FQ2 input traces from frozen files, without model calls or adjudication.

The 94 extra deletions are candidates with no released answer. S labels are
audit identifiers only and are not inserted into the original model prompt.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re


CONDITIONS = ("one_valid_mss", "remove_required_view", "hard_distractor_added")
ORDINAL = re.compile(r"图(?:片)?([一二三四五1-5])|第\s*([一二三四五1-5])\s*张图(?:片)?")
NUMBERS = {**dict(zip("一二三四五", range(1, 6))), **{str(i): i for i in range(1, 6)}}
EXAMPLES = {
    "group_00392": {
        "question_zh": "以图一镜头朝向为正前方，面对前方时右手侧为右方，烤箱位于椅子的哪一侧？",
        "question_en": "Take the camera's viewing direction in Image 1 as forward, and the right-hand side when facing forward as right. On which side of the chair is the oven?",
        "options_en": {"A": "Front", "B": "Right", "C": "Behind", "D": "Insufficient information"},
        "observation_en": "The released deletion removes S01. The first image delivered to the model is then S02, while the unchanged question still refers to Image 1.",
    },
    "group_00368": {
        "question_zh": "以图二镜头朝向为正前方，面对前方时右手侧为右方，盆栽位于桌子的哪一侧？",
        "question_en": "Take the camera's viewing direction in Image 2 as forward, and the right-hand side when facing forward as right. On which side of the table is the potted plant?",
        "options_en": {"A": "Back-right", "B": "Front-right", "C": "Insufficient information", "D": "Back-left"},
        "observation_en": "The released deletion removes S01 and retains the original anchor S02 in the first position. The second image delivered to the model is now S03, while the question still refers to Image 2.",
    },
    "group_00311": {
        "question_zh": "以图二镜头朝向为正前方，面对前方时右手侧为右方,游戏手柄位于矿泉水瓶的那一侧？",
        "question_en": "Take the camera's viewing direction in Image 2 as forward, and the right-hand side when facing forward as right. On which side of the bottled water is the game controller?",
        "options_en": {"A": "Back-left", "B": "Insufficient information", "C": "Back-right", "D": "Front-left"},
        "observation_en": "The released deletion removes S03. The original anchor S02 remains in the second position, so the explicit ordinal continues to identify the same image.",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def index(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_digest(source):
    from PIL import Image, ImageOps
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=True, progressive=False, exif=b"")
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def prompt(question, options):
    return ("请根据当前提供的信息回答下面的四选一问题。\n" + question + "\n" +
            "\n".join(f"{letter}. {options[letter]}" for letter in "ABCD") +
            "\n只输出一个答案字母（A、B、C 或 D），不要解释。")


def reference_trace(question, base_images, current_images, registry):
    result = []
    for match in ORDINAL.finditer(question):
        number = NUMBERS[match.group(1) or match.group(2)]
        original = base_images[number - 1] if number <= len(base_images) else None
        current = current_images[number - 1] if number <= len(current_images) else None
        result.append({"literal_zh": match.group(0), "ordinal": number,
                       "original_anchor_id": registry.get(original),
                       "current_positional_target_id": registry.get(current),
                       "original_anchor_retained": original in current_images,
                       "positional_status": "out_of_range" if current is None else
                       "same_image" if original == current else "different_image"})
    return result


def stratum(refs):
    if not refs:
        return "no_ordinal"
    if not refs[0]["original_anchor_retained"]:
        return "anchor_removed"
    return "ordinal_stable" if refs[0]["positional_status"] == "same_image" else "anchor_retained_shifted"


def build(root, output):
    manuscript = root / "framebench/FrameEvidence/05_论文工程/manuscript_rebuild"
    dataset = root / "framebench/daily_20260922T075500Z/dataset"
    run = root / "frameevidence_merge_20261006_200305/results/formal_2c289604aa45_test"
    sources = {
        "annotations": dataset / "annotations.jsonl",
        "mapping": run / "private/evaluation_mapping.jsonl",
        "prepared": run / "prepared/prepared_inputs.jsonl",
        "run_manifest": run / "run_manifest.json",
        "split": root / "framebench/FrameEvidence/results/formal_release_v1/formal_2c289604aa45/manifests/split_manifest.csv",
        "previous_cases": manuscript / "materials/binding_validation_20261010/case_reference.jsonl",
        "retained_runner": root / "formal_eval_runtime/dry_run_eval/runner.py",
    }
    before = {key: digest(path) for key, path in sources.items()}
    tree = ast.parse(sources["retained_runner"].read_text(encoding="utf-8-sig"))
    system_values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT" for target in node.targets)]
    require(len(system_values) == 1, "Missing/ambiguous retained system prompt")
    system_prompt = system_values[0]
    annotations = index(read_jsonl(sources["annotations"]), "sample_id")
    prepared = index(read_jsonl(sources["prepared"]), "eval_id")
    manifest = json.loads(sources["run_manifest"].read_text(encoding="utf-8-sig"))
    require(manifest["evaluation_split"] == "test", "Not a test run")
    with sources["split"].open(encoding="utf-8-sig", newline="") as handle:
        expected_groups = {r["relation_group_id"] for r in csv.DictReader(handle)
                           if r["split"] == "test" and r["question_type"] == "FQ2"}
    mappings = [r for r in read_jsonl(sources["mapping"])
                if r["question_type"] == "FQ2" and r["evaluation_mode"] == "all-images"]
    require(len(mappings) == 144 and len(expected_groups) == 48, "Unexpected frozen test scope")
    groups = defaultdict(dict)
    assets = {}
    for mapping in mappings:
        gid, condition = mapping["relation_group_id"], mapping["view_condition"]
        require(gid in expected_groups and condition not in groups[gid], "Wrong group/duplicate condition")
        ann, item = annotations[mapping["sample_id"]], prepared[mapping["eval_id"]]
        require(ann["relation_group_id"] == gid, "Annotation group mismatch")
        require(ann["images"] == mapping["source_image_paths"], "Source order mismatch")
        require(item["image_refs"] == mapping["public_image_refs"], "Prepared order mismatch")
        require(ann["options"] == mapping["options"] and ann["answer"] == mapping["gold_answer"], "Original options/labels mismatch")
        require(item["prompt"] == prompt(ann["question"], ann["options"]), "Original prompt mismatch")
        for source, ref in zip(ann["images"], item["image_refs"], strict=True):
            require(source not in assets or assets[source] == ref, "Inconsistent asset mapping")
            assets[source] = ref
        groups[gid][condition] = (mapping, ann, item)
    require(set(groups) == expected_groups, "Test group membership mismatch")
    for source, ref in assets.items():
        require(normalized_digest(dataset / source) == digest(run / "prepared" / ref),
                f"Normalized image mismatch: {source}")
    cases, group_trace, registry_rows = [], [], []
    for gid, variants in sorted(groups.items()):
        require(set(variants) == set(CONDITIONS), f"Incomplete group: {gid}")
        _, base, _ = variants["one_valid_mss"]
        _, deletion, _ = variants["remove_required_view"]
        _, addition, _ = variants["hard_distractor_added"]
        base_order = base["images"]
        require(base_order == base["mss_images"] and len(set(base_order)) == len(base_order), "MSS structure mismatch")
        removed = [path for path in base_order if path not in deletion["images"]]
        added = [path for path in addition["images"] if path not in base_order]
        require(len(removed) == 1 and deletion["images"] == [p for p in base_order if p != removed[0]], "Published deletion mismatch")
        require(len(added) == 1 and [p for p in addition["images"] if p not in added] == base_order, "Addition order mismatch")
        registry = {path: f"S{i:02d}" for i, path in enumerate(dict.fromkeys([*base_order, *addition["images"]]), 1)}
        refs = reference_trace(deletion["question"], base_order, deletion["images"], registry)
        category = stratum(refs)
        group_trace.append({"group_id": gid, "published_deletion_stratum": category,
                            "mss_size": len(base_order), "mss_ids": [registry[p] for p in base_order],
                            "published_deletion_ids": [registry[p] for p in deletion["images"]],
                            "addition_ids": [registry[p] for p in addition["images"]],
                            "removed_id": registry[removed[0]], "added_id": registry[added[0]],
                            "published_deletion_references": refs})
        for source, stable_id in registry.items():
            registry_rows.append({"group_id": gid, "stable_id": stable_id,
                                  "source_path": source, "prepared_ref": assets[source],
                                  "prepared_file": assets[source]})
        plans = []
        for condition in CONDITIONS:
            mapping, ann, item = variants[condition]
            plans.append({"case_id": mapping["eval_id"], "source_eval_id": mapping["eval_id"],
                          "source_sample_id": mapping["sample_id"], "condition": condition,
                          "kind": "published_case", "question_zh": ann["question"], "options_zh": ann["options"],
                          "original_model_prompt_zh": item["prompt"], "paths": ann["images"],
                          "released_answer_reference": ann["answer"]})
        for source in base_order:
            if source != removed[0]:
                plans.append({"case_id": gid + "|loo|" + source, "source_eval_id": None,
                              "source_sample_id": None, "condition": "leave_one_mss_member_out",
                              "kind": "additional_deletion_candidate", "question_zh": base["question"],
                              "options_zh": base["options"], "original_model_prompt_zh": prompt(base["question"], base["options"]),
                              "paths": [p for p in base_order if p != source], "released_answer_reference": None})
        for plan in plans:
            paths = plan.pop("paths")
            sequence = [{"current_position": position, "stable_original_id": registry[path],
                         "original_mss_position": base_order.index(path) + 1 if path in base_order else None,
                         "source_path": path, "prepared_ref": assets[path]}
                        for position, path in enumerate(paths, 1)]
            cases.append({"group_id": gid, "published_deletion_stratum": category, **plan,
                          "retained_system_prompt_zh": system_prompt,
                          "actual_image_sequence": sequence,
                          "removed_original_ids": [registry[p] for p in base_order if p not in paths],
                          "added_original_ids": [registry[p] for p in paths if p not in base_order],
                          "ordinal_trace": reference_trace(plan["question_zh"], base_order, paths, registry),
                          "stable_ids_present_in_original_prompt": False,
                          "new_adjudicated_answer": None, "semantic_validation_performed": False})
    previous = index(read_jsonl(sources["previous_cases"]), "case_key")
    require(len(cases) == 238 and set(previous) == {c["case_id"] for c in cases}, "Coverage changed from existing diagnostic")
    for case in cases:
        old = previous[case["case_id"]]
        require(case["original_model_prompt_zh"] == old["original_prompt"], "Previous prompt changed")
        require([r["source_path"] for r in case["actual_image_sequence"]] == old["source_images"], "Previous image order changed")
        require(case["released_answer_reference"] == old["released_label_reference"], "Previous label changed")
    examples = []
    for gid, translation in EXAMPLES.items():
        selected = [c for c in cases if c["group_id"] == gid and c["kind"] == "published_case"]
        require(len(selected) == 3 and all(c["question_zh"] == translation["question_zh"] for c in selected), "Translation source text mismatch")
        examples.append({"group_id": gid, "translation_status": "English translation for reading, not a model input",
                         **translation, "options_zh": selected[0]["options_zh"], "published_triplet": selected,
                         "system_prompt_zh": system_prompt,
                         "system_prompt_en": "You are participating in a blinded four-choice evaluation. Output only A, B, C, or D, without explanation.",
                         "user_prompt_en": ("Answer the following four-choice question using the information currently provided.\n" +
                                            translation["question_en"] + "\n" +
                                            "\n".join(f"{letter}. {translation['options_en'][letter]}" for letter in "ABCD") +
                                            "\nOutput only one answer letter (A, B, C, or D), without explanation."),
                         "image_registry": [r for r in registry_rows if r["group_id"] == gid]})
    counts = Counter(c["condition"] for c in cases)
    require(dict(counts) == {"one_valid_mss": 48, "remove_required_view": 48, "hard_distractor_added": 48, "leave_one_mss_member_out": 94}, "Unexpected coverage")
    require(all(c["released_answer_reference"] is None for c in cases if c["kind"] != "published_case"), "Extra candidate acquired a label")
    require(before == {key: digest(path) for key, path in sources.items()}, "Frozen input changed during audit")
    result = {"status": "INPUT_TRACE_VERIFIED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "group_count": 48, "case_count": len(cases), "published_cases": 144,
              "extra_deletion_candidates_without_labels": 94, "single_member_deletions": 142,
              "condition_counts": dict(counts), "mss_size_groups": dict(Counter(g["mss_size"] for g in group_trace)),
              "published_deletion_strata": dict(Counter(g["published_deletion_stratum"] for g in group_trace)),
              "source_asset_paths_verified": len(assets), "normalized_unique_image_files": len(set(assets.values())),
              "original_prompts_checked": len(mappings), "previous_diagnostic_cases_checked": len(previous),
              "new_model_calls": 0, "new_human_records": 0, "new_adjudicated_labels": 0,
              "source_files_modified": False, "semantic_validation_performed": False,
              "source_paths": {key: path.relative_to(root).as_posix() for key, path in sources.items()}, "source_sha256": before}
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("cases.jsonl", cases), ("image_registry.jsonl", registry_rows)):
        (output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for filename, data in (("verification.json", result), ("group_traces.json", group_trace), ("examples_zh_en.json", examples)):
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "condition_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["case_id", "group_id", "condition", "kind", "source_eval_id", "released_answer_reference", "image_order", "removed_ids", "added_ids"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            writer.writerow({**{key: c[key] for key in fields[:6]},
                             "image_order": " ".join(r["stable_original_id"] for r in c["actual_image_sequence"]),
                             "removed_ids": " ".join(c["removed_original_ids"]), "added_ids": " ".join(c["added_original_ids"])})
    (output / "examples_zh_en.md").write_text(render_examples(examples), encoding="utf-8")
    return result


def render_examples(examples):
    lines = ["# FQ2 原文、译文与模型实际图序", "",
             "以下中文来自冻结输入，逐字保留。英文仅供阅读，没有替换实验输入。S01、S02等仅是跨条件溯源编号，原模型请求没有这些编号。表内答案为发布标签，不是本次新增裁决。", ""]
    names = {"one_valid_mss": "MSS", "remove_required_view": "Published deletion", "hard_distractor_added": "Hard addition"}
    for example in examples:
        lines += ["## " + example["group_id"], "", "原题", "", example["question_zh"], "",
                  "英文对照", "", example["question_en"], "",
                  "| Option | 原文 | Translation |", "| --- | --- | --- |"]
        for key in "ABCD":
            lines.append(f"| {key} | {example['options_zh'][key]} | {example['options_en'][key]} |")
        lines += ["", "| Condition | Delivered order | Released answer |", "| --- | --- | --- |"]
        for c in example["published_triplet"]:
            order = " → ".join(r["stable_original_id"] for r in c["actual_image_sequence"])
            lines.append(f"| {names[c['condition']]} | {order} | {c['released_answer_reference']} |")
        lines += ["", example["observation_en"], "", "图片定位", ""]
        for image in example["image_registry"]:
            lines.append(f"- {image['stable_id']}: `{image['source_path']}` → `{image['prepared_ref']}`")
        lines += ["", "原始系统指令", "", "```text", example["system_prompt_zh"], "```", "",
                  "完整原始用户请求文本（三个发布条件相同）", "", "```text",
                  example["published_triplet"][0]["original_model_prompt_zh"], "```", "",
                  "系统指令英文对照", "", example["system_prompt_en"], "",
                  "用户请求英文对照", "", "```text", example["user_prompt_en"], "```", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = build(args.root.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key not in {"source_paths", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
