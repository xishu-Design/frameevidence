from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .common import DEFAULT_SEED, TOOL_VERSION, dataset_fingerprint, read_jsonl, sha256_file, stable_hash, write_json, write_jsonl

PROMPT_TEMPLATE = """请根据当前提供的信息回答下面的四选一问题。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n只输出一个答案字母（A、B、C 或 D），不要解释。"""
IMAGE_ORDERING_POLICY = "题目按 seed+eval_id 哈希；图片严格保留冻结 annotations.jsonl 中的原始顺序"
SINGLE_IMAGE_RULE = "选取冻结标注顺序中的第一张图片；不对图片重排或按哈希重新编号"


def _normalize_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        image.save(target, format="JPEG", quality=95, optimize=True, progressive=False, exif=b"")


def _prompt(record: dict[str, Any]) -> str:
    options = record["options"]
    return PROMPT_TEMPLATE.format(question=record["question"], **options)


def prepare_inputs(
    dataset_dir: Path,
    run_dir: Path,
    audit: dict[str, Any],
    seed: int = DEFAULT_SEED,
    evaluation_profile: str = "dry_run_descriptive",
    result_label: str = "exploratory_dry_run",
    allowed_group_ids: set[str] | None = None,
    evaluation_split: str | None = None,
    split_manifest: Path | None = None,
) -> dict[str, Any]:
    all_records = read_jsonl(dataset_dir / "annotations.jsonl")
    fingerprint = dataset_fingerprint(dataset_dir, all_records)
    normalized_allowed_groups = (
        {str(group_id) for group_id in allowed_group_ids}
        if allowed_group_ids is not None
        else None
    )
    records = (
        [
            record
            for record in all_records
            if str(record.get("relation_group_id")) in normalized_allowed_groups
        ]
        if normalized_allowed_groups is not None
        else all_records
    )
    selected_group_ids = sorted({str(record["relation_group_id"]) for record in records})
    allowed_group_ids_sha256 = (
        hashlib.sha256("\n".join(sorted(normalized_allowed_groups)).encode("utf-8")).hexdigest()
        if normalized_allowed_groups is not None
        else None
    )
    split_manifest_path = split_manifest.resolve() if split_manifest is not None else None
    split_manifest_sha256 = (
        sha256_file(split_manifest_path)
        if split_manifest_path is not None
        else None
    )
    prepared_images = run_dir / "prepared" / "images"
    digest_by_relative: dict[str, str] = {}
    public_ref_by_relative: dict[str, str] = {}
    current_paths = sorted({path for record in records for path in record.get("images", [])})
    for relative in current_paths:
        source = dataset_dir / relative
        digest = sha256_file(source)
        digest_by_relative[relative] = digest
        opaque = stable_hash(seed, digest)[:20]
        public_ref = f"images/img_{opaque}.jpg"
        public_ref_by_relative[relative] = public_ref
        _normalize_image(source, run_dir / "prepared" / public_ref)

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    warning_samples = set(audit.get("warning_sample_ids", []))
    for record in records:
        # Image position is semantic data in FrameEvidence: many questions refer
        # to 图一/图二/第 N 张图.  Never hash-sort this list.  The frozen
        # annotation order is the presentation order used by both training and
        # evaluation, while only the order of *questions* is randomized below.
        ordered_images = list(record.get("images", []))
        modes = {
            "all-images": ordered_images,
            "text-only": [],
            "single-image": ordered_images[:1],
        }
        for mode, image_paths in modes.items():
            eval_id = f"eval_{stable_hash(seed, record['sample_id'], mode)[:20]}"
            image_refs = [public_ref_by_relative[path] for path in image_paths]
            public_rows.append({
                "eval_id": eval_id,
                "evaluation_mode": mode,
                "prompt": _prompt(record),
                "image_refs": image_refs,
            })
            private_rows.append({
                "eval_id": eval_id,
                "sample_id": record["sample_id"],
                "relation_group_id": record["relation_group_id"],
                "evaluation_mode": mode,
                "question_type": record["question_type"],
                "reference_frame": record["reference_frame"],
                "view_condition": record["view_condition"],
                "gold_answer": record["answer"],
                "options": record["options"],
                "image_count": len(record.get("images", [])),
                "selected_image_count": len(image_paths),
                "source_image_paths": image_paths,
                "public_image_refs": image_refs,
                "warning_flag": record["sample_id"] in warning_samples,
            })

    order = sorted(range(len(public_rows)), key=lambda index: stable_hash(seed, public_rows[index]["eval_id"]))
    public_rows = [public_rows[index] for index in order]
    private_by_id = {row["eval_id"]: row for row in private_rows}
    private_rows = [private_by_id[row["eval_id"]] for row in public_rows]
    public_path = run_dir / "prepared" / "prepared_inputs.jsonl"
    private_path = run_dir / "private" / "evaluation_mapping.jsonl"
    write_jsonl(public_path, public_rows)
    write_jsonl(private_path, private_rows)
    manifest = {
        "tool_version": TOOL_VERSION,
        "evaluation_profile": evaluation_profile,
        "result_label": result_label,
        "evaluation_split": evaluation_split,
        "dataset_fingerprint": fingerprint,
        "dataset": str(dataset_dir.resolve()),
        "seed": seed,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "normalization": {
            "exif_transpose": True,
            "color_mode": "RGB",
            "maximum_long_edge": 2048,
            "output_format": "JPEG",
            "jpeg_quality": 95,
            "metadata_stripped": True,
            "upscaling": False,
        },
        "ordering": IMAGE_ORDERING_POLICY,
        "single_image_rule": SINGLE_IMAGE_RULE,
        "public_items": len(public_rows),
        "source_samples": len(records),
        "evaluation_modes": ["all-images", "text-only", "single-image"],
        "source_image_paths": len(public_ref_by_relative),
        "normalized_unique_images": len(set(public_ref_by_relative.values())),
        "allowed_group_ids_count": (
            len(normalized_allowed_groups)
            if normalized_allowed_groups is not None
            else None
        ),
        "allowed_group_ids_sha256": allowed_group_ids_sha256,
        "split_manifest": str(split_manifest_path) if split_manifest_path is not None else None,
        "split_manifest_sha256": split_manifest_sha256,
        "selection": {
            "evaluation_split": evaluation_split,
            "group_filter_applied": normalized_allowed_groups is not None,
            "selected_group_count": len(selected_group_ids),
            "selected_group_ids": selected_group_ids,
            "allowed_group_ids_sha256": allowed_group_ids_sha256,
            "split_manifest": str(split_manifest_path) if split_manifest_path is not None else None,
            "split_manifest_sha256": split_manifest_sha256,
        },
        "paths": {
            "public_inputs": str(public_path.resolve()),
            "private_mapping": str(private_path.resolve()),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest
