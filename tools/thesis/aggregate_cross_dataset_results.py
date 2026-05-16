import argparse
import csv
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path


PUNCT_TABLE = str.maketrans("", "", string.punctuation + "，。！？；：、“”‘’（）【】《》—…")


def norm(text):
    text = "" if text is None else str(text)
    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def as_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def contains_match(pred, gold):
    pred = norm(pred)
    gold = norm(gold)
    if not pred or not gold:
        return None
    return 1.0 if gold in pred or pred in gold else 0.0


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def anls(pred, gold):
    pred = norm(pred)
    gold = norm(gold)
    if not pred or not gold:
        return None
    dist = levenshtein(pred, gold)
    score = 1.0 - dist / max(len(pred), len(gold))
    return score if score >= 0.5 else 0.0


def iter_records(raw_dir):
    for path in sorted(Path(raw_dir).glob("answer_*.jsonl")):
        mode = path.stem.replace("answer_", "")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_mode"] = row.get("mode") or mode
                row["_source_file"] = str(path)
                yield row


def metric_row(dataset, mode, rows):
    exact_vals = []
    contains_vals = []
    anls_vals = []
    for row in rows:
        pred = row.get("answer_extracted") or row.get("pred_answer") or row.get("text")
        gold = row.get("gt_answer")
        exact_vals.append(row.get("exact_match") if row.get("exact_match") is not None else (1.0 if norm(pred) and norm(pred) == norm(gold) else 0.0 if norm(gold) else None))
        contains_vals.append(row.get("contains_match") if row.get("contains_match") is not None else contains_match(pred, gold))
        anls_vals.append(row.get("anls") if row.get("anls") is not None else anls(pred, gold))
    return {
        "dataset": dataset,
        "mode": mode,
        "N": len(rows),
        "exact_match_mean": mean(exact_vals),
        "contains_match_mean": mean(contains_vals),
        "anls_mean": mean(anls_vals),
        "bbox_parse_success_rate": mean([r.get("bbox_parse_ok") for r in rows]),
        "bbox_acc_at_05": mean([r.get("bbox_correct_at_05") for r in rows]),
        "mean_bbox_iou": mean([r.get("bbox_iou") for r in rows]),
        "mean_crop_ratio": mean([r.get("crop_ratio") for r in rows]),
        "mean_latency_ms_total": mean([r.get("latency_ms_total") or r.get("latency_ms") for r in rows]),
        "mean_latency_ms_generate": mean([r.get("latency_ms_generate") for r in rows]),
        "mean_peak_gpu_memory_mb": mean([r.get("peak_gpu_memory_mb") for r in rows]),
        "mean_num_visual_inputs": mean([r.get("num_visual_inputs") for r in rows]),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "dataset",
        "mode",
        "N",
        "exact_match_mean",
        "contains_match_mean",
        "anls_mean",
        "bbox_parse_success_rate",
        "bbox_acc_at_05",
        "mean_bbox_iou",
        "mean_crop_ratio",
        "mean_latency_ms_total",
        "mean_latency_ms_generate",
        "mean_peak_gpu_memory_mb",
        "mean_num_visual_inputs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_md(path, rows):
    fields = [
        "dataset",
        "mode",
        "N",
        "contains_match_mean",
        "anls_mean",
        "mean_latency_ms_total",
        "mean_peak_gpu_memory_mb",
        "mean_num_visual_inputs",
        "mean_crop_ratio",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Region Evidence Summary\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---" for _ in fields]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(k)) for k in fields) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate cross-dataset thesis results by dataset and mode.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()

    records = list(iter_records(args.raw_dir))
    if not records:
        raise FileNotFoundError(f"No answer JSONL files found in {args.raw_dir}")
    groups = defaultdict(list)
    overall = defaultdict(list)
    for row in records:
        dataset = row.get("dataset") or "unknown"
        mode = row.get("_mode") or "unknown"
        groups[(dataset, mode)].append(row)
        overall[mode].append(row)

    by_dataset = [metric_row(dataset, mode, rows) for (dataset, mode), rows in sorted(groups.items())]
    by_overall = [metric_row("ALL", mode, rows) for mode, rows in sorted(overall.items())]
    metrics_dir = Path(args.metrics_dir)
    write_csv(metrics_dir / "summary_by_dataset.csv", by_dataset)
    write_csv(metrics_dir / "summary_cross_overall.csv", by_overall)
    write_md(metrics_dir / "summary_by_dataset.md", by_dataset)
    write_md(metrics_dir / "summary_cross_overall.md", by_overall)
    print(f"Wrote cross-dataset metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
