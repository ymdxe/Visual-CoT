import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.rc import apply, area as box_area, box_from, ok_box
from tools.rc_sum import hit


def load_det(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_img(row):
    img = row.get("image")
    if isinstance(img, list):
        return str(img[0]).split("###")[0]
    return str(img).split("###")[0]


def fit(img, size):
    canvas = Image.new("RGB", size, "white")
    got = ImageOps.contain(img.convert("RGB"), size)
    x = (size[0] - got.width) // 2
    y = (size[1] - got.height) // 2
    canvas.paste(got, (x, y))
    return canvas


def draw_box(img, box):
    out = img.convert("RGB").copy()
    if ok_box(box):
        w, h = out.size
        x1, y1, x2, y2 = box
        d = ImageDraw.Draw(out)
        d.rectangle((x1 * w, y1 * h, x2 * w, y2 * h), outline="red", width=max(3, w // 160))
    return out


def score(row):
    return hit(row.get("text"), row.get("gt_answer") or row.get("answer"), clean=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--det", required=True)
    parser.add_argument("--img-root", required=True)
    parser.add_argument("--res", nargs="*", default=[])
    parser.add_argument("--out", default="results/rc/fig")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    data = json.load(open(args.bench, encoding="utf-8"))
    det = load_det(args.det)
    res = {}
    for path in args.res:
        mode = Path(path).stem.split("_")[0]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    res[(mode, str(row.get("question_id")))] = row

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["qid", "mode", "box_ok", "area", "gold", "pred", "hit_n", "file"]
    idx_rows = []
    labels = ["orig", "mark", "blur", "zoom", "mix"]
    cell = (280, 230)
    pad = 12
    label_h = 28
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, row in enumerate(data[: args.n]):
        qid = str(row.get("question_id", i))
        drow = det[i] if i < len(det) else {}
        box = box_from(drow.get("text") or drow.get("bbox"))
        img_path = os.path.join(args.img_root, get_img(row))
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        views = [draw_box(img, box)]
        for mode in labels[1:]:
            got, _ = apply(img, box, mode, q=row.get("conversations", [{}])[0].get("value", ""))
            views.append(got)

        sheet = Image.new("RGB", (len(labels) * cell[0] + (len(labels) + 1) * pad, cell[1] + label_h + 2 * pad), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (label, view) in enumerate(zip(labels, views)):
            x = pad + j * (cell[0] + pad)
            draw.text((x, pad), label, fill="black", font=font)
            sheet.paste(fit(view, cell), (x, pad + label_h))
        file = out / f"{i:06d}_{qid}.png"
        sheet.save(file)

        a = box_area(box)
        for mode in labels:
            r = res.get((mode, qid))
            idx_rows.append(
                {
                    "qid": qid,
                    "mode": mode,
                    "box_ok": ok_box(box),
                    "area": "" if a is None else a,
                    "gold": "" if r is None else (r.get("gt_answer") or r.get("answer")),
                    "pred": "" if r is None else r.get("text"),
                    "hit_n": "" if r is None else score(r),
                    "file": str(file),
                }
            )

    with open(out / "fig_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(idx_rows)
    print(out)


if __name__ == "__main__":
    main()
