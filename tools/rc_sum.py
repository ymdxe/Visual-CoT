import argparse
import csv
import json
import os
import re
import string
from collections import defaultdict
from statistics import mean


ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def norm(x):
    x = "" if x is None else str(x).lower()
    x = x.translate(str.maketrans("", "", string.punctuation))
    x = ARTICLES.sub(" ", x)
    return " ".join(x.split())


def hit(pred, gold, clean=False):
    if pred is None or gold is None:
        return None
    p = norm(pred) if clean else str(pred).strip().lower()
    g = norm(gold) if clean else str(gold).strip().lower()
    if not p or not g:
        return 0.0
    return 1.0 if p in g or g in p else 0.0


def fnum(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def avg(xs):
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if x is not None]
    return mean(vals) if vals else None


def mode_from_path(path):
    return os.path.basename(os.path.dirname(path))


def read_rows(paths):
    groups = defaultdict(list)
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row.get("metadata") or {}
                mode = meta.get("mode") or meta.get("compress_mode") or mode_from_path(path)
                policy = meta.get("policy") or meta.get("visual_input_policy") or "na"
                tag = os.path.basename(os.path.dirname(path))
                pad = meta.get("pad")
                blur = meta.get("blur")
                scale = meta.get("scale")
                groups[(tag, mode, policy, pad, blur, scale)].append(row)
    return groups


def calc(paths):
    rows = []
    for (tag, mode, policy, pad, blur, scale), data in sorted(read_rows(paths).items()):
        pre, gen, e2e, mem, imgs, box_ok, area = [], [], [], [], [], [], []
        raw, clean = [], []
        for row in data:
            meta = row.get("metadata") or {}
            pre.append(meta.get("pre", meta.get("preprocess_latency_sec")))
            gen.append(meta.get("gen", meta.get("generation_latency_sec", meta.get("latency_sec"))))
            e2e.append(meta.get("e2e", meta.get("end_to_end_latency_sec")))
            mem.append(meta.get("mem", meta.get("max_gpu_memory_mb")))
            imgs.append(meta.get("imgs", meta.get("num_visual_images")))
            ok = meta.get("box_ok", meta.get("bbox_valid"))
            if ok is not None:
                box_ok.append(1.0 if ok else 0.0)
            area.append(meta.get("area", meta.get("bbox_area")))
            gold = row.get("gt_answer") or row.get("answer")
            pred = row.get("text")
            raw.append(hit(pred, gold, clean=False))
            clean.append(hit(pred, gold, clean=True))
        rows.append(
            {
                "mode": mode,
                "policy": policy,
                "tag": tag,
                "pad": pad,
                "blur": blur,
                "scale": scale,
                "n": len(data),
                "pre": avg(pre),
                "gen": avg(gen),
                "e2e": avg(e2e),
                "mem": avg(mem),
                "imgs": avg(imgs),
                "box_ok": avg(box_ok),
                "area": avg(area),
                "hit": avg(raw),
                "hit_n": avg(clean),
            }
        )
    return rows


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    if x is None:
        return ""
    return str(x)


def write(rows, out_csv, out_md):
    fields = ["tag", "mode", "policy", "pad", "blur", "scale", "n", "pre", "gen", "e2e", "mem", "imgs", "box_ok", "area", "hit", "hit_n"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---" if i < 2 else "---:" for i in range(len(fields))]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row[k]) for k in fields) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--csv", default="results/rc/sum.csv")
    parser.add_argument("--md", default="results/rc/sum.md")
    args = parser.parse_args()
    rows = calc(args.inputs)
    write(rows, args.csv, args.md)
    print(args.csv)
    print(args.md)


if __name__ == "__main__":
    main()
