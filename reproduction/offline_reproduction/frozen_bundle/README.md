# FrameEvidence local offline scoring reproduction, v2

This self-contained Python 3.9+ standard-library bundle recomputes retained
predictions without models, training, network access, third-party packages,
images, prompts, or the original repository. Copy the entire folder to any
location and run from that directory:

```console
python -B scorer.py
python -B -m unittest -v test_scorer
python -B scorer.py --out ../recomputed_offline_results_v2
```

On Windows, `py -3` can replace `python`. The output directory must be outside
this frozen bundle. The first command checks bundled SHA-256 digests, recomputes
metrics, and compares the pre-existing references. It returns `PASS` only after
every check passes; errors exit with status 1. Optional output comprises
`recomputed_metrics.json` and four separate CSV files. Tests include copying
the bundle outside the repository and running Python in isolated mode.

## Separate experiment cohorts

- `primary_main`: the 20 configurations in `materials/table_main.tex`, in table
  order, each with 288 all-images predictions (5,760 records). This includes
  12 open-weight configurations and eight gateway configurations. All seven
  displayed metrics are checked against 140 saved table cells as well as exact
  counts, confusion matrices, and saved `formal_metrics.json` values.
- `ordinary`: the eight later ordinary multiseed configurations, each with 288
  all-images predictions (2,304 records). These are a separate result cohort;
  primary Base/LoRA predictions are not substituted for multiseed controls.
- `circular_complete`: seven configurations with four rotations of 288 source
  questions (8,064 valid records). Six are LoRA runs with labeled training
  seeds 20260722, 20260723, and 20260724, three per family.
- `circular_incomplete_diagnostic`: the located Qwen Base Circular copy, with
  1,152 rows: 706 valid and 446 `CALL_ERROR`. The denominator retains errors.
  These diagnostic calculations are not the completed-run result.
- `author_confirmed_summary`: the separately attributed Qwen Base completed
  summary (14.93% rotation-zero accuracy, 18.75% mean-rotation accuracy, 2.08%
  strict Circular accuracy, 5.90% consistency, 12.85 percentage-point drop).
  It has not been independently recomputed from complete predictions. Rounded
  percentages are not converted to item counts. These five rounded values also
  equal the incomplete-copy diagnostics when errors count as incorrect; their
  agreement therefore cannot independently establish a completed execution.

The main table uses its own sanitized mapping and prediction namespace, even
where dataset identities coincide with the later runs. Neither main-table
predictions nor the author summary enter the three-seed means or sample SDs.
Ordinary predictions and Circular rotation zero are separate executions; their
agreement is reported without substituting one for the other.

## Scoring contract

An answer is valid only when stored `status == "OK"` and `parsed_answer` is A,
B, C, or D. Stored parsing results are scored; raw text and parser execution
are outside this bundle. Non-OK rows count as incorrect even if they contain a
letter. Missing/duplicate IDs or malformed triplets cause failure rather than
a reduced denominator. LLaVA in the primary cohort retains all ten `CALL_ERROR`
rows: 84 correct / 288 questions, with 278 valid answers. The other 19 primary
configurations, all eight ordinary configurations, and seven complete Circular
configurations have no invalid rows. Unexpected coverage changes fail checks.

For each ordinary/primary configuration, accuracy is correct / 288. Per-label
precision is TP / predicted count, recall is TP / gold support, and F1 is their
harmonic mean, with zero for a zero denominator. Macro-P/R/F1 average A--D.
Invalid answers contribute to gold support but to no valid predicted class.

FQ1-GEM requires all camera, object-intrinsic, and world answers in a triplet
correct. NTS requires both `one_valid_mss` and `remove_required_view` correct.
DIS requires both `one_valid_mss` and `hard_distractor_added` correct. Each
joint denominator is 48 groups. Table percentages are formatted to two decimal
places after scoring; full-precision values and integer counts remain in JSON.

For Circular rotation r, new label j receives original label `(j+r) mod 4`.
The scorer checks permutations and rotated gold against source gold. Vanilla
is r=0 correct / 288; mean-rotation accuracy uses all 1,152 rows; strict Circular
accuracy requires all four rotations correct. Consistency requires all four
valid answers to map to one original label, which may itself be incorrect.
Drop is Vanilla minus strict Circular accuracy in percentage points. Means
and sample SD use only each family's three LoRA runs, with SD denominator 2.

## Provenance and limits

`metadata.json` preserves source aliases, dataset/split identity, receipt and
prediction hashes, available model labels, gateway API model identifiers, shot
counts, selection-ordering seeds, and labeled multiseed training seeds. The
primary imported LoRA training seed and exact API identity are not inferred.
Normalized import receipts bind the imported prediction bytes; original server
bytes named by those receipts were not independently checked in this export.
Gateway checkpoint, selection, and embedded selection-provenance digests are
checked; metadata records how many raw rows actually embed provenance, since
some legacy imported and gateway rows lack that field. Receipt agreement does not establish
historical model identity or successful execution beyond the retained records.

`expected_metrics.json` contains pre-existing references, not newly generated
ground truth. Exact integer counts, saved confusion matrices, and all main-table
display cells must agree; full-precision floats permit absolute error 1e-12.
`source_verification.json` records before/after hashes for every source read by
the exporter. `bundle_manifest.json` covers the exported files. These local
checks are not external signatures or independent attestations.

Included inputs are only opaque IDs, labels, group/rotation structure, parsed
predictions, statuses, and parser statuses. Images, paths, prompts/options,
raw responses, errors, endpoints, credentials, and human data are excluded.
The bundle establishes arithmetic over retained labels and answers. It does
not establish semantic gold/MSS validity, reasoning processes, successful model
training reproduction, independent Qwen Base Circular completion, significance,
confidence intervals, human results, or asset redistribution permission.

This v2 folder is separate from `offline_reproduction_20261009`; the original
bundle and review packets are unchanged. Repository export command:

```console
py -3 -B scripts/build_offline_reproduction.py --run <downloaded-multiseed-run> --main-run <primary-main-run> --out materials/offline_reproduction_20261009_v2
```

The builder requires a README in a new directory under manuscript materials
and refuses an existing bundle manifest. The standalone scorer needs none of
the source directories and never runs inference or training.
