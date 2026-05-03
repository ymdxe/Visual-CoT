import argparse
import json
import os
import re
import uuid


def box_from(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(v) for v in value[:4]]
        except (TypeError, ValueError):
            return None
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
    if len(nums) < 4:
        return None
    try:
        return [float(v) for v in nums[:4]]
    except ValueError:
        return None


def clip_box(box, width=None, height=None):
    box = box_from(box)
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
        if width and height:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        else:
            m = max(abs(x1), abs(y1), abs(x2), abs(y2), 1.0)
            x1, y1, x2, y2 = x1 / m, y1 / m, x2 / m, y2 / m
    x1, y1, x2, y2 = [min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2)]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def infer_image(row):
    if "img_path" in row:
        return row["img_path"]
    image = row.get("image")
    if isinstance(image, list):
        return image[0]
    return image


def infer_prompt(row):
    if "expression" in row:
        return row["expression"]
    conv = row.get("conversations") or []
    if conv:
        return conv[0].get("value", "").replace("<image>\n", "").replace("<image>", "")
    return row.get("prompt") or row.get("question") or ""


def infer_box(row):
    width, height = row.get("width"), row.get("height")
    if "bbox" in row:
        box = clip_box(row.get("bbox"), width, height)
        if box is not None:
            return box
    conv = row.get("conversations") or []
    if len(conv) > 1:
        box = clip_box(conv[1].get("value"), width, height)
        if box is not None:
            return box
    image = row.get("image") or []
    if isinstance(image, list) and len(image) > 1 and "###" in image[1]:
        return clip_box(image[1].split("###", 1)[1], width, height)
    return None


def main():
    parser = argparse.ArgumentParser(description="Build oracle detection JSONL from benchmark annotations.")
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    rows = load_records(args.question_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as out:
        for local_idx, row in enumerate(rows):
            box = infer_box(row)
            text = None if box is None else "[%.3f, %.3f, %.3f, %.3f]" % tuple(box)
            out.write(json.dumps({
                "question_id": row.get("question_id", local_idx),
                "dataset": args.dataset_name or row.get("dataset"),
                "image": infer_image(row),
                "prompt": infer_prompt(row),
                "bbox_gt": box,
                "bbox_pred_raw": text,
                "bbox_pred": box,
                "bbox_parse_ok": box is not None,
                "height": row.get("height"),
                "width": row.get("width"),
                "latency_ms": None,
                "peak_gpu_memory_mb": None,
                "answer_id": uuid.uuid4().hex,
                "model_id": "oracle",
                "text": text,
                "bbox": row.get("bbox"),
                "metadata": {"source": args.question_file},
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
