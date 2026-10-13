# FrameEvidence — anonymous software supplement

This archive provides offline scoring code, retained prediction records and
regression tests for the submitted paper. It requires Python 3.9 or later and
only the standard library. No installation, model weights, GPU, credentials or
network connection is required.

## Run

Extract the archive, enter `FrameEvidence_software`, and run:

```console
python -B verify_software.py
```

On Windows, `py -3` may replace `python`. The verifier writes its logs and
recalculated results to the sibling `software_verification` directory. It does
not overwrite input records. An alternate external directory can be supplied
with `--output`.

## Contents

| Directory | Retained material |
| --- | --- |
| `offline_reproduction` | 20 primary model configurations, 8 ordinary multiseed configurations, open-model Circular results, and GPT Circular and same-order controls |
| `binding_joint` | 216 records from the reference-binding diagnostics, comprising the 72-request pilot and two 72-request execution blocks |

The verifier checks 140 primary-table metric cells and computes the paired GPT
controls. Three-seed summaries use sample standard deviation. Diagnostic
binding results preserve the pilot and repeated blocks as separate cohorts.
The associated 36 unit tests cover scoring, invalid responses, paired item
identity and input completeness.

`offline_reproduction/README.md` and `frozen_bundle/README.md` describe the
metric definitions, output files and retained-result coverage. The
author-confirmed Qwen Base Circular summary is separate from its incomplete
retained diagnostic execution. The scorer preserves this distinction and does
not replace failed calls or infer missing predictions. The analytic random
baseline and participant-level human summary are not per-item model cohorts.

## Data attachment

The separate `FrameEvidence_data.zip` contains delivered FQ2 prompts, image
sequences, prepared images and trace tests. It is not required to run the
prediction scorer. Neither archive performs new model calls or human annotation.
