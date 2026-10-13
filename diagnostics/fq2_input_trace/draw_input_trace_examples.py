"""Render source-grounded bilingual input plates without altering source images.

The complete Chinese prompt is retained in the source JSON. This display places
options on one line and shares the unchanged answer-format instructions in TeX.
S01 etc. are display-only trace identifiers, never asserted to be model inputs.
"""
from pathlib import Path
import argparse
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

WIDTH = 468
FONT = 9.5
LEADING = 11.5


def paragraph(text, font="Times-Roman", size=FONT):
    return Paragraph(text, ParagraphStyle("trace", fontName=font, fontSize=size,
                                          leading=LEADING, spaceAfter=0))


def draw_case(example, output, title, image_root):
    left = 0
    registry = example["image_registry"]
    n = len(registry)
    gap = 5
    cell_width = (WIDTH - gap * (n - 1)) / n
    assets = []
    for image in registry:
        path = image_root / image["prepared_ref"]
        with Image.open(path) as im:
            w, h = im.size
        assets.append((path, min(cell_width / w, 92 / h) * w,
                       min(cell_width / w, 92 / h) * h, image["stable_id"]))
    zh = paragraph(escape(example["question_zh"]), "SourceChinese", 10)
    en = paragraph(escape(example["question_en"]))
    options = paragraph(" &nbsp;&nbsp; ".join(
        f"{key}. <font name='SourceChinese'>{escape(example['options_zh'][key])}</font>"
        f" / {escape(example['options_en'][key])}" for key in "ABCD"), size=9)
    note = paragraph(escape(example["observation_en"]), size=9)
    blocks = [(zh, 5), (en, 5), (options, 8)]
    text_height = sum(p.wrap(WIDTH, 1000)[1] + after for p, after in blocks)
    note_height = note.wrap(WIDTH, 1000)[1]
    height = 18 + 108 + text_height + 59 + note_height + 2
    c = canvas.Canvas(str(output), pagesize=(WIDTH, height), pageCompression=1)
    c.setTitle(f"FrameEvidence input trace {example['group_id']}")
    c.setAuthor("")
    y = height - 11
    c.setFont("Times-Bold", 10.5)
    c.drawString(left, y, title)
    y -= 7
    c.setLineWidth(.6)
    c.line(0, y, WIDTH, y)
    y -= 13
    for j, (path, image_w, image_h, stable_id) in enumerate(assets):
        x = j * (cell_width + gap)
        c.setFont("Times-Bold", 9)
        c.drawString(x, y, stable_id)
        c.drawImage(ImageReader(str(path)), x + (cell_width - image_w) / 2,
                    y - 4 - image_h, image_w, image_h)
    y -= 103
    for p, after in blocks:
        _, ph = p.wrap(WIDTH, 1000)
        p.drawOn(c, 0, y - ph)
        y -= ph + after
    c.setFont("Times-Bold", 9)
    for x, label in ((0, "Input"), (70, "Actual image sequence"),
                     (258, "Ordinal now refers to"), (393, "Released label")):
        c.drawString(x, y, label)
    y -= 4
    c.setLineWidth(.4)
    c.line(0, y, WIDTH, y)
    short = {"one_valid_mss": "MSS", "remove_required_view": "Deletion",
             "hard_distractor_added": "Addition"}
    for row in example["published_triplet"]:
        y -= 12
        c.setFont("Times-Roman", 9.5)
        c.drawString(0, y, short[row["condition"]])
        c.drawString(70, y, ", ".join(i["stable_original_id"] for i in row["actual_image_sequence"]))
        trace = row["ordinal_trace"][0]
        target = trace["current_positional_target_id"] or "Out of range"
        c.drawString(258, y, target)
        c.drawString(410, y, row["released_answer_reference"])
    y -= 5
    c.line(0, y, WIDTH, y)
    y -= 5
    note.drawOn(c, 0, y - note_height)
    assert y - note_height >= 0, (height, y, note_height)
    c.save()
    return {"file": output.name, "group": example["group_id"],
            "width_pt": WIDTH, "height_pt": height, "image_count": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chinese-font", type=Path, required=True)
    p.add_argument("--latin-font", type=Path, required=True)
    p.add_argument("--latin-bold-font", type=Path, required=True)
    args = p.parse_args()
    pdfmetrics.registerFont(TTFont("SourceChinese", str(args.chinese_font)))
    pdfmetrics.registerFont(TTFont("Times-Roman", str(args.latin_font)))
    pdfmetrics.registerFont(TTFont("Times-Bold", str(args.latin_bold_font)))
    examples = {r["group_id"]: r for r in json.loads(args.examples.read_text(encoding="utf-8"))}
    args.output.mkdir(parents=True, exist_ok=True)
    descriptions = [("group_00392", "(a) Original anchor removed - group 00392"),
                    ("group_00368", "(b) Original anchor retained at a new position - group 00368")]
    records = [draw_case(examples[gid], args.output / f"{gid}.pdf", title, args.examples.resolve().parent)
               for gid, title in descriptions]
    (args.output / "render_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
