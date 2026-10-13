# FrameEvidence — anonymous data supplement

This archive supplies the FQ2 input traces used for the paper's evidence and
reference-binding analysis. It covers all 48 held-out FQ2 groups, with 144
published conditions and 125 unique prepared images. It is an analysis data
attachment, not a replacement for the full 960-question benchmark.

## Run

Extract the archive, enter `FrameEvidence_data`, and run:

```console
python -B verify_data.py
```

The verifier requires Python 3.9 or later and only the standard library. It
writes its report to the sibling `data_verification` directory without modifying
the inputs. It checks image bytes, relative paths, condition counts, original
prompts and image-order traces. All 12 structural tests must pass.

## Contents

`fq2_input_trace/cases.jsonl` records the delivered Chinese prompts, answer
options, image order and released answer references. The accompanying CSV and
JSON files index groups and image sequences. `images/` contains the prepared
images. Three translated examples and two bilingual appendix plates provide
readable illustrations.

The traces also include 94 additional single-image deletion conditions used
to enumerate the remaining MSS members. These have no adjudicated answer
labels and are explicitly distinguished from the 144 published conditions.
They are not additional scored benchmark items or human responses.

The nested README documents the fields and optional image-plate rendering
commands. Image files are included for reviewing the submitted research. This
attachment does not grant a separate asset redistribution license.
