import argparse
import json
import os
import textwrap
from PIL import Image, ImageDraw


def resolve(root, image):
    path = os.path.join(root, image)
    if os.path.exists(path): return path
    if image and image.startswith("cot/cub/"):
        alt = os.path.join("downloads/cub/CUB_200_2011/images", image[len("cot/cub/"):])
        if os.path.exists(alt): return alt
    return path


def draw_box(img, box):
    out = img.copy().convert("RGB")
    if isinstance(box, list) and len(box) >= 4:
        w, h = out.size; x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1:
            x1, x2 = x1*w, x2*w; y1, y2 = y1*h, y2*h
        ImageDraw.Draw(out).rectangle([x1, y1, x2, y2], outline="red", width=max(3, w//180))
    return out


def crop(img, box):
    if not isinstance(box, list) or len(box) < 4: return img.copy()
    w, h = img.size; x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1:
        x1, x2 = x1*w, x2*w; y1, y2 = y1*h, y2*h
    return img.crop((int(max(0,x1)), int(max(0,y1)), int(min(w,x2)), int(min(h,y2))))


def make_case(row, image_root, out_path):
    img = Image.open(resolve(image_root, row.get("image"))).convert("RGB")
    boxed = draw_box(img, row.get("bbox_pred")); local = crop(img, row.get("bbox_pred")).resize((260, 260))
    boxed.thumbnail((520, 360)); canvas = Image.new("RGB", (820, 520), "white")
    canvas.paste(boxed, (20, 20)); canvas.paste(local, (540, 20)); draw = ImageDraw.Draw(canvas)
    text = f"Q: {row.get('prompt','')}\nGT: {row.get('gt_answer')}\nPred: {row.get('pred_answer', row.get('text'))}\nMode: {row.get('mode')} / {row.get('crop_mode')} / {row.get('evidence_mode')}\nTag: {row.get('error_tag')}"
    y = 390
    for line in textwrap.wrap(text, width=95):
        draw.text((20, y), line, fill=(0,0,0)); y += 18
    canvas.save(out_path)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--answers-jsonl", required=True); p.add_argument("--image-root", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--error-tag", default=None); p.add_argument("--top-k", type=int, default=5); args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); made = 0
    with open(args.answers_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if args.error_tag and row.get("error_tag") != args.error_tag: continue
            try:
                make_case(row, args.image_root, os.path.join(args.output_dir, f"case_{made:03d}_{row.get('question_id')}.jpg")); made += 1
            except Exception as exc:
                print(f"skip {row.get('question_id')}: {exc}")
            if made >= args.top_k: break
    print(f"wrote {made} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
