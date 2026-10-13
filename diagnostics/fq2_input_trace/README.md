# Original FQ2 inputs

This export records all 48 test groups. It includes 238 conditions comprising the
144 published conditions and 94 additional single-image deletion candidates.
The latter have no answer labels. Every `new_adjudicated_answer` remains null.

- `cases.jsonl` contains the complete delivered Chinese system and user prompts,
  option order, image order and original released answer references.
- `condition_index.csv` indexes conditions and their image sequences.
- `group_traces.json` records the 48 original triplets and ordinal-reference traces.
- `image_registry.jsonl` maps group-local S identifiers to original asset paths
  and bundled prepared images. Resolve `prepared_ref` relative to this directory.
- `images/` contains all 125 unique prepared images, copied without modification.
- `image_files.json` records byte sizes and content digests for integrity checks.
- `examples_zh_en.json` contains three complete Chinese examples and English
  translations, including system instructions and image registries.
- `examples_zh_en.md` provides a readable subset of the same examples.
- `example_plates/` contains the two bilingual plates used in the appendix.
- `draw_input_trace_examples.py` regenerates the plates from bundled relative
  image references. Rendering additionally requires ReportLab, Pillow and supplied
  Chinese and Latin TrueType fonts.
- `verification.json` preserves the original structural verification results.

S identifiers are display-only trace labels. They were not added to the original
model prompts. An input-order check establishes what the model received, not
whether a spatial answer or MSS judgment is semantically correct.

## Verify this extracted package

```console
python -B -m unittest -v test_trace test_bundled_images
```

These tests use only the Python standard library. They check the original trace
contracts, every bundled image reference and byte identity, all 238 case IDs,
and the absence of newly assigned answers.

## Rebuild from the full source repository

`build_trace.py` is the source exporter. Its rebuild additionally requires Pillow
and the original annotation, prepared-input, mapping, split and diagnostic files.
Those original repository files are not duplicated here. Their relative locations
are recorded in `verification.json`. Supply the root of that repository explicitly
and write to a fresh derived directory

```console
python build_trace.py --root <repository-root> --output <new-export-directory>
```

The source exporter does not make network requests or change its source files.
It recreates trace metadata. The bundled `images/` directory is supplied separately
for offline inspection and image-byte verification.

To redraw the example plates with your own installed fonts

```console
python draw_input_trace_examples.py --examples examples_zh_en.json --output redrawn_plates --chinese-font <Chinese-font.ttf> --latin-font <Latin-font.ttf> --latin-bold-font <Latin-bold-font.ttf>
```

Fonts are not redistributed. The pre-rendered PDFs already embed the glyphs needed
for these examples, so Overleaf does not need a Chinese font or XeLaTeX.
