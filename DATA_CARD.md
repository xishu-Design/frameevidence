# FrameEvidence data card

FrameEvidence evaluates spatial reference-frame switching and evidence
interventions in multiple-choice visual question answering. The released
benchmark contains 960 Chinese questions arranged in 320 complete groups of
three, with 1,120 referenced JPEG files. Each question has four options, A–D,
and one released answer label.

## Task structure

| Task | Groups | Questions | Within-group comparison |
| --- | ---: | ---: | --- |
| FQ1 | 160 | 480 | Camera, object-intrinsic and world reference frames |
| FQ2 | 160 | 480 | One annotated MSS, a required-view deletion and a hard-distractor addition |

MSS denotes an annotated minimal sufficient set. These annotations define the
released conditions; their presence does not independently establish semantic
minimal sufficiency for every example. The paper discusses reference-binding
ambiguity and the limits of the deletion intervention.

## Files and annotation fields

`dataset/annotations.jsonl` contains one JSON object per question.
`dataset/images/` contains the 1,120 referenced JPEG files. Image paths in
`images`, `mss_images` and `removed_images` are relative to `dataset/`.
Preserve their order: some questions refer to images by their current position.

The 17 annotation fields are `sample_id`, `relation_group_id`, `question_type`,
`question`, `options`, `answer`, `reference_frame`, `reference_anchor`,
`target_instance_id`, `relatum_instance_id`, `object_instances`,
`bridge_instance_ids`, `view_condition`, `images`, `mss_images`,
`removed_images` and `quality_status`. Identifiers describe benchmark samples,
groups and annotated objects. The schema is provided in
`02_数据模式与标注试点/data_schema.json`.

## Frozen split

| Split | Groups | Questions | FQ1 groups | FQ2 groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 192 | 576 | 96 | 96 |
| Dev | 32 | 96 | 16 | 16 |
| Test | 96 | 288 | 48 | 48 |

`splits/split_manifest.csv` assigns each group exactly once. All three
questions in a group remain together. The manifest uses connected components
of exact image SHA-256 matches as its cluster basis. This prevents those exact
image components from crossing splits; it does not establish complete physical
scene independence across viewpoints, crops, resizing or re-encoding.
Reuse this manifest for comparisons with the paper; do not regenerate it.

The dataset fingerprint binds the annotation bytes and the sorted image paths
and image digests:

```text
2c289604aa45e7373ae9dcaa6737ffe74ed31f5b9aa0005e85ad3c22a6dde5bc
```

The frozen split manifest SHA-256 is:

```text
bc9afad049e3c6cbdc9b5328ea3c2bdb9be70f0ada784033c820ce80ee6a53d9
```

## Evaluation and diagnostic coverage

The main held-out evaluation uses 288 questions. Input preparation produces
all-images, text-only and single-image variants for each question: 864
prepared records in total. Group scores require jointly correct answers within
the specified triplet or pair; individual responses must not be counted as
independent groups.

`diagnostics/fq2_input_trace/` covers the 48 test FQ2 groups: 144 published
conditions, 94 additional deletion candidates and 125 prepared images. The
additional candidates have no released or newly adjudicated answer labels.
They do not expand the 960-question benchmark. Structural trace checks verify
input identity, image order and coverage; they do not establish semantic label
correctness or provide new human assessments.

`reproduction/` contains retained predictions for the documented scoring
cohorts and the separate reference-binding diagnostics. It supports arithmetic
reproduction over stored parsed answers. The incomplete Qwen Base Circular
diagnostic and the separately attributed summary retain their distinct status.
No missing prediction or participant record is synthesized.

## Intended use and limitations

Use the benchmark to study reference-frame sensitivity, joint response
correctness and responses to the specified evidence interventions. Read the
paper's qualifications before interpreting a deletion score as evidence of
causal reasoning. Rewriting an ordinal reference cannot recover information
from a removed view. Textual or option-layout regularities and image reuse can
also affect observed performance.

The public package contains annotation records, image files, evaluation code
and the documented diagnostics. It excludes annotation-platform databases,
account records, provider credentials, inference logs and model weights.
See the paper for collection and analysis context. This data card does not
grant a new license for underlying image assets; their applicable rights and
terms remain in force.

## Integrity check

From the repository root, run `python -B verify_release.py`. This read-only
check validates the release manifest, all referenced image paths, the frozen
data fingerprint and the split. After installing `requirements.txt`, add
`--schema` to validate each record against the local JSON Schema. No model or
network call is made by either command.
