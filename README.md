# FrameEvidence

This repository accompanies the FrameEvidence paper and contains the full
960-question benchmark, its frozen train/dev/test split, evaluation code,
retained-result reproduction code and FQ2 input diagnostics.

The benchmark comprises 320 complete three-question groups and 1,120 JPEG
files. The paper's main evaluation uses the 96 held-out groups, or 288
questions. Read [DATA_CARD.md](DATA_CARD.md) for field definitions, split
construction and interpretation limits, and [the paper](paper/paper.pdf) for
the experimental results.

## Repository contents

| Path | Contents |
| --- | --- |
| `dataset/` | Full `annotations.jsonl` and all referenced images |
| `splits/` | Frozen split manifest and accompanying split records |
| `evaluation/` | Formal evaluation, input preparation, scoring and Circular tools |
| `protocol/` | Release policy, group quotas and recorded split overrides |
| `02_数据模式与标注试点/` | Annotation JSON Schema used by the evaluator |
| `数据生产与质量检验系统/` | Validation code and QC rules required by preflight |
| `reproduction/` | Retained predictions, offline scorers and regression tests |
| `diagnostics/` | Delivered FQ2 prompts, ordered image traces and structural tests |
| `paper/paper.pdf` | Accompanying paper |

The split retains each three-question group and separates components defined
by exact image SHA-256 matches. It does not guarantee independence of all
physical scenes across transformed images or viewpoints. Keep the supplied
split unchanged when comparing results with the paper.

## Verify the download

Use Python 3.10 or later. From the repository root:

```console
python -B verify_release.py
```

This command uses only the Python standard library. It checks file hashes,
the complete 960-question/320-group/1,120-image inventory, image-path safety,
the data fingerprint and the frozen split. It reads files without changing
them and does not contact any service. `release_manifest.json` covers public
files except itself; `.git`, Python caches and the generated `outputs/`
directory are outside this inventory.

To enable annotation-schema validation and use the formal evaluator:

```console
python -m pip install -r requirements.txt
python -B verify_release.py --schema
```

Installation may download dependencies. The verification and preparation
commands below operate locally after those dependencies are installed.

## Reproduce retained results offline

The retained-result scorers require only the Python standard library:

```console
python -B reproduction/verify_software.py --output outputs/software_verification
python -B diagnostics/verify_data.py --output outputs/data_verification
```

The software verifier recomputes 140 metric cells for the 20 primary model
configurations, the retained multiseed and GPT controls, and the separate
reference-binding diagnostics. It runs the included scoring tests and writes
derived results and logs under the requested output directory. The data
verifier checks the FQ2 trace contracts and prepared-image bytes.

These commands score stored parsed predictions; they do not rerun model
inference or training. Read `reproduction/offline_reproduction/README.md` for
cohort coverage. The incomplete Qwen Base Circular diagnostic is distinct
from the separately attributed summary. The package does not infer missing
answers or recreate participant-level human records.

## Prepare the held-out inputs

Run from the repository root with the explicit release paths:

```console
python -B evaluation/formal_eval.py prepare --dataset dataset --output-root outputs/evaluation --policy protocol/formal_release_scale_policy.json --split-manifest splits/split_manifest.csv --eval-split test
```

Preparation audits the local data and uses the frozen test assignment. It
creates 864 records: 288 questions in each of the all-images, text-only and
single-image modes. It preserves the supplied image order and writes generated
inputs, answer mappings and audit reports below
`outputs/evaluation/formal_2c289604aa45_test/`. No model is called.

Preflight can report `REVIEW_REQUIRED` for the documented data limitations.
A standalone `preflight` command returns exit code 1 for that status and 2 for
`BLOCKED`; these states must not be relabeled as a successful semantic audit.
Preparation proceeds only when no blocking error is present.

The evaluator accepts reader-supplied model configurations. Configure a
service and its environment variables explicitly before using the `run`
subcommand. Model calls are separate from downloading, verifying, preparing
or scoring the data. Credentials and model weights are not included.

For predictions produced in the prepared run, recompute formal scores with:

```console
python -B evaluation/formal_eval.py score --dataset dataset --output-root outputs/evaluation --policy protocol/formal_release_scale_policy.json --split-manifest splits/split_manifest.csv --eval-split test
python -B evaluation/circular_eval.py --help
```

`score` requires existing prediction files; use the retained-result commands
above to reproduce the packaged results without performing new inference.
Do not regenerate the split or replace released labels when reproducing the
paper's evaluation.

## Scope and rights

FQ2 diagnostics include 94 additional, unlabelled deletion candidates. They
are separate from the full benchmark and are not new model or human answers.
Input coverage and path checks do not certify spatial labels, semantic
minimal sufficiency or causal reasoning.

The repository excludes service credentials, account databases, inference
logs and model weights. No new image-redistribution license is granted by
this README; applicable rights and terms for the underlying assets remain in
force.
