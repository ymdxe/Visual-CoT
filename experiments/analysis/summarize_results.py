import argparse
import csv
import json
import os
import re
from collections import defaultdict


def norm(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def hit(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return None, None
    return int(p == g), int(g in p or p in g)


def anls(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return None
    m, n = len(p), len(g)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (p[i - 1] != g[j - 1]))
            prev = cur
    score = 1.0 - dp[n] / max(m, n)
    return score if score >= 0.5 else 0.0


def box(value):
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
    return [max(0, min(1, x1)), max(0, min(1, y1)), max(0, min(1, x2)), max(0, min(1, y2))]


def iou(a, b):
    a, b = box(a), box(b)
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = area_a + area_b - inter
    return inter / den if den else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paths(root):
    if os.path.isfile(root):
        return [root]
    out = []
    for base, _, names in os.walk(root):
        for name in names:
            if name.endswith(".jsonl"):
                out.append(os.path.join(base, name))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="Summarize Visual-CoT experiment JSONL outputs.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in paths(args.input_root):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("dataset") or "unknown", row.get("mode") or row.get("crop_mode") or "unknown", row.get("bbox_source") or "unknown", row.get("crop_mode") or "unknown", row.get("evidence_mode") or "unknown")
                groups[key].append(row)
    rows = []
    for (dataset, mode, source, crop_mode, evidence_mode), vals in sorted(groups.items()):
        ems, subs, anls_vals, ious, bbox_acc = [], [], [], [], []
        for row in vals:
            em, sub = hit(row.get("pred_answer", row.get("text")), row.get("gt_answer", row.get("answer")))
            ems.append(em); subs.append(sub); anls_vals.append(anls(row.get("pred_answer", row.get("text")), row.get("gt_answer", row.get("answer"))))
            biou = iou(row.get("bbox_pred"), row.get("bbox_gt"))
            ious.append(biou); bbox_acc.append(None if biou is None else int(biou >= 0.5))
        rows.append({
            "dataset": dataset, "mode": mode, "bbox_source": source, "crop_mode": crop_mode, "evidence_mode": evidence_mode, "n": len(vals),
            "em": mean(ems), "substring": mean(subs), "textvqa_acc": mean(subs), "anls": mean(anls_vals), "bbox_iou": mean(ious), "bbox_acc": mean(bbox_acc),
            "bbox_parse_ok": mean([r.get("bbox_parse_ok") for r in vals]), "latency_ms": mean([r.get("latency_ms") for r in vals]),
            "peak_gpu_memory_mb": mean([r.get("peak_gpu_memory_mb") for r in vals]), "num_visual_inputs": mean([r.get("num_visual_inputs") for r in vals]),
            "num_visual_tokens_est": mean([r.get("num_visual_tokens_est") for r in vals]), "crop_ratio": mean([r.get("crop_ratio") for r in vals]),
        })
    fields = ["dataset","mode","bbox_source","crop_mode","evidence_mode","n","em","substring","textvqa_acc","anls","bbox_iou","bbox_acc","bbox_parse_ok","latency_ms","peak_gpu_memory_mb","num_visual_inputs","num_visual_tokens_est","crop_ratio"]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("| Dataset | Mode | BBox | Crop | Evidence | N | EM | Substr | BBox Acc | Latency | Mem | Inputs | Crop Ratio |\n")
        f.write("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            f.write(f"| {r['dataset']} | {r['mode']} | {r['bbox_source']} | {r['crop_mode']} | {r['evidence_mode']} | {r['n']} | {fmt(r['em'])} | {fmt(r['substring'])} | {fmt(r['bbox_acc'])} | {fmt(r['latency_ms'])} | {fmt(r['peak_gpu_memory_mb'])} | {fmt(r['num_visual_inputs'])} | {fmt(r['crop_ratio'])} |\n")


if __name__ == "__main__":
    main()
