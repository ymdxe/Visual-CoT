import argparse
import csv
import json
import os
import re
from collections import Counter


def norm(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_ok(row):
    pred = norm(row.get("pred_answer", row.get("text")))
    gold = norm(row.get("gt_answer", row.get("answer")))
    if not pred or not gold:
        return None
    return gold in pred or pred in gold


def to_box(value):
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
        m = max(abs(x1), abs(y1), abs(x2), abs(y2), 1.0)
        x1, y1, x2, y2 = x1 / m, y1 / m, x2 / m, y2 / m
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iou(a, b):
    a, b = to_box(a), to_box(b)
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2)-max(ax1, bx1)) * max(0, min(ay2, by2)-max(ay1, by1))
    den = max(0, ax2-ax1)*max(0, ay2-ay1) + max(0, bx2-bx1)*max(0, by2-by1) - inter
    return inter / den if den else None


def tag(row):
    ans = answer_ok(row); biou = iou(row.get("bbox_pred"), row.get("bbox_gt"))
    if ans is None or biou is None:
        return "unknown"
    bbox = biou >= 0.5
    if bbox and ans: return "bbox_correct_answer_correct"
    if bbox and not ans: return "bbox_correct_answer_wrong"
    if not bbox and ans: return "bbox_wrong_answer_correct"
    return "bbox_wrong_answer_wrong"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--answers-jsonl", required=True); p.add_argument("--output-jsonl", required=True); p.add_argument("--output-csv", required=True); args = p.parse_args()
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True); os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    counts = Counter()
    with open(args.answers_jsonl, "r", encoding="utf-8") as src, open(args.output_jsonl, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip(): continue
            row = json.loads(line); row["error_tag"] = tag(row); counts[row["error_tag"]] += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["error_tag", "count"]); w.writerows(sorted(counts.items()))


if __name__ == "__main__":
    main()
