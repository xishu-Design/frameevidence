# FrameEvidence offline result reproduction

This compact package recalculates retained prediction metrics using Python
3.9+ and its standard library. No model, GPU, network, API key, image, prompt,
or original repository is required. Extract the whole folder and run from it.

```console
python -B run_reproduction.py
python -B -m unittest discover -s frozen_bundle -p test_scorer.py -v
python -B -m unittest -v test_reproduction
```

On Windows, `py -3` may replace `python`. Results appear in `recomputed/`.
An alternate destination can be supplied with `--out <directory>`.
Input files remain unchanged. Tests require a writable temporary directory.

## Included results

| Cohort | Coverage | Output |
| --- | --- | --- |
| Main table | 20 configurations, 288 questions each | `primary_main.csv` |
| Ordinary multiseed controls | 8 configurations, 288 questions each | `ordinary.csv` |
| Complete open-weight Circular | 7 configurations, 1,152 predictions each | `circular_complete.csv` |
| GPT Circular | 1,152 valid predictions | `gpt_controls.json` |
| GPT same-order repeat | 288 valid predictions | `gpt_controls.json` |
| GPT two-execution comparisons | 4 metrics across repeat and 3 rotation pairs | `gpt_table6.csv` |

The scorer checks all 140 original main-table metric cells. GPT controls are
computed separately from retained predictions and checked against the saved
historical results. The package preserves ordinary and Circular executions as
separate cohorts. Three-seed summaries use sample standard deviation.
The 20 main-table configurations are model-result rows. The analytic random
baseline and the separately supplied human summary are not prediction cohorts
in this package.

Validated on 12 October 2026 with all **26 tests passing** (20 frozen-bundle
tests and 6 supplemental tests), including copied-package execution in
isolated Python.

## Evidence boundary

The author-confirmed Qwen Base Circular summary is preserved separately in
`frozen_bundle/author_confirmed_summary.json`. The retained diagnostic copy has
706 valid rows and 446 call failures. Its separate CSV retains the full
denominator and is not labeled as a complete reproduced execution. Neither
this diagnostic nor the author summary contributes to LoRA three-seed means.

The original v2 scorer and sanitized inputs are reused unchanged in
`frozen_bundle/`. Its README contains the detailed metric contracts. The new
GPT inputs contain only opaque evaluation IDs, option-label mappings, labels,
stored parsed answers, and statuses. They omit source paths, raw responses,
prompts, credentials, endpoints, participant data, and images. This package
checks scoring arithmetic, not the semantic correctness of dataset labels or
MSS judgments. No asset redistribution license is inferred or granted here.

## Main outputs

`recomputed_metrics.json` contains full-precision main, ordinary and Circular
metrics with integer counts. `gpt_controls.json` contains GPT source counts
and paired counts. `verification.json` gives a concise completion status.
CSV values are percentages, and Circular drop is in percentage points.

The repeat is a later execution rather than an additional simultaneous draw.
Paired comparisons describe observed runs and do not isolate service drift
from order sensitivity. No new inference is performed by these commands.
