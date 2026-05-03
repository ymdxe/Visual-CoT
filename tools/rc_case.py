import argparse
import csv
import json
import os

from tools.rc_sum import hit


def load(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("question_id"))] = row
    return rows


def score(row):
    return hit(row.get("text"), row.get("gt_answer") or row.get("answer"), clean=True) or 0.0


def meta(row, key, default=None):
    return (row.get("metadata") or {}).get(key, default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--crop")
    parser.add_argument("--out", default="results/rc/cases.csv")
    args = parser.parse_args()

    base = load(args.base)
    mix = load(args.mix)
    crop = load(args.crop) if args.crop else {}
    rows = []
    for qid, mrow in mix.items():
        brow = base.get(qid)
        if brow is None:
            continue
        b = score(brow)
        m = score(mrow)
        area = meta(mrow, "area")
        box_ok = meta(mrow, "box_ok")
        case = None
        if m > b:
            case = "mix_win"
        elif b > m:
            case = "mix_lose"
        elif box_ok is False:
            case = "box_bad"
        elif area is not None and area < 0.03:
            case = "box_small"
        if args.crop and qid in crop and score(crop[qid]) < b:
            case = "crop_bad"
        if case:
            rows.append(
                {
                    "case": case,
                    "qid": qid,
                    "area": area,
                    "base": b,
                    "mix": m,
                    "gold": mrow.get("gt_answer") or mrow.get("answer"),
                    "pred": mrow.get("text"),
                    "prompt": mrow.get("prompt"),
                }
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fields = ["case", "qid", "area", "base", "mix", "gold", "pred", "prompt"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
