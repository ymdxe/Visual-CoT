#!/usr/bin/env python3
"""Render a benchmark sample as a 3-panel demo (original + bbox overlay + crop).

Reads one entry from a VisCoT-style benchmark JSON (image field is a 2-element
list: [original_path, "original_path###[x1,y1,x2,y2]"]), then composes a side-by-side
figure showing the original image, the bbox-annotated image, and the cropped region.

Usage:
    python tools/thesis/render_sample_triptych.py \\
        --benchmark data/benchmarks/small_benchmark_docvqa100.json \\
        --question-id 791 \\
        --image-folder playground/data \\
        --output images/benchmarks/example_docvqa_gnnp0227_6.png
"""
import argparse
import json
import os
import re
from PIL import Image, ImageDraw, ImageFont


def parse_pixel_bbox(image_entry):
    """Extract [x1, y1, x2, y2] pixel bbox from 'path###[x1,y1,x2,y2]'."""
    m = re.search(r"###\s*\[([^\]]+)\]", image_entry)
    if not m:
        raise ValueError(f"No bbox found in: {image_entry}")
    return [int(float(x.strip())) for x in m.group(1).split(",")]


def find_entry(benchmark_path, question_id):
    with open(benchmark_path) as f:
        data = json.load(f)
    for item in data:
        if item.get("question_id") == question_id:
            return item
    raise ValueError(f"question_id={question_id} not found in {benchmark_path}")


def extract_qa(conversations):
    question = ""
    gpt_bbox = ""
    answer = ""
    for i, turn in enumerate(conversations):
        val = turn.get("value", "")
        if turn["from"] == "human" and i == 0:
            question = val.replace("<image>", "").strip()
        elif turn["from"] == "gpt" and i == 1:
            gpt_bbox = val.strip()
        elif turn["from"] == "gpt" and i == 3:
            answer = val.strip()
    return question, gpt_bbox, answer


def render_triptych(original, bbox_pixel, gpt_bbox_text, question, answer, output_path):
    img = Image.open(original).convert("RGB")
    W, H = img.size

    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = bbox_pixel
    line_w = max(4, int(min(W, H) * 0.006))
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_w)

    crop = img.crop((x1, y1, x2, y2))

    target_h = 600
    def resize_keep_h(im):
        ratio = target_h / im.height
        return im.resize((int(im.width * ratio), target_h), Image.LANCZOS)

    panels = [resize_keep_h(img), resize_keep_h(annotated), resize_keep_h(crop)]
    captions = [
        "(a) Original",
        f"(b) With predicted bbox\n{gpt_bbox_text}",
        f"(c) Cropped region\n-> answer: {answer}",
    ]

    pad = 20
    cap_h = 80
    gap = 30
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1) + pad * 2
    total_h = target_h + cap_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    cdraw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()

    x = pad
    for panel, cap in zip(panels, captions):
        canvas.paste(panel, (x, pad))
        cdraw.multiline_text(
            (x, pad + target_h + 10), cap, fill=(0, 0, 0), font=font, spacing=4
        )
        x += panel.width + gap

    qfont = font
    title = f"Q: {question}"
    if len(title) > 120:
        title = title[:117] + "..."
    canvas2 = Image.new("RGB", (total_w, total_h + 60), (255, 255, 255))
    canvas2.paste(canvas, (0, 60))
    ImageDraw.Draw(canvas2).text(
        (pad, 20), title, fill=(0, 0, 0), font=qfont
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas2.save(output_path)
    print(f"Saved: {output_path}  ({canvas2.size[0]}x{canvas2.size[1]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--question-id", type=int, required=True)
    p.add_argument("--image-folder", default="playground/data")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    entry = find_entry(args.benchmark, args.question_id)
    original_rel = entry["image"][0]
    bbox_pixel = parse_pixel_bbox(entry["image"][1])
    question, gpt_bbox, answer = extract_qa(entry["conversations"])

    original_abs = os.path.join(args.image_folder, original_rel)
    render_triptych(original_abs, bbox_pixel, gpt_bbox, question, answer, args.output)


if __name__ == "__main__":
    main()
