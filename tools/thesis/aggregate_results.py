import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


def norm(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
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


def ratio_delta(value, base):
    value = as_float(value)
    base = as_float(base)
    if value is None or base in (None, 0):
        return None
    return (base - value) / base


def iter_jsonl(raw_dir):
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        if not path.name.startswith("answer_"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_source_file"] = str(path)
                    yield row


def get_mode(row):
    return row.get("mode") or Path(row.get("_source_file", "unknown")).stem.replace("answer_", "")


def metric_row(mode, rows):
    exact_vals = []
    contains_vals = []
    bbox_parse = []
    bbox_acc = []
    extraction_ok = []
    answer_lengths = []
    for row in rows:
        pred = row.get("answer_extracted") or row.get("pred_answer") or row.get("text")
        gold = row.get("gt_answer")
        exact_vals.append(row.get("exact_match") if row.get("exact_match") is not None else int(norm(pred) == norm(gold)) if norm(pred) and norm(gold) else None)
        contains_vals.append(row.get("contains_match"))
        bbox_parse.append(row.get("bbox_parse_ok"))
        bbox_acc.append(row.get("bbox_correct_at_05"))
        extraction_ok.append(1 if row.get("answer_extracted") not in (None, "") else 0)
        answer_lengths.append(row.get("answer_length_tokens") or len(str(pred or "").split()))
    return {
        "mode": mode,
        "N": len(rows),
        "exact_match_mean": mean(exact_vals),
        "contains_match_mean": mean(contains_vals),
        "bbox_parse_success_rate": mean(bbox_parse),
        "bbox_acc_at_05": mean(bbox_acc),
        "mean_bbox_iou": mean([r.get("bbox_iou") for r in rows]),
        "mean_crop_ratio": mean([r.get("crop_ratio") for r in rows]),
        "mean_latency_ms_total": mean([r.get("latency_ms_total") or r.get("latency_ms") for r in rows]),
        "mean_latency_ms_generate": mean([r.get("latency_ms_generate") for r in rows]),
        "mean_peak_gpu_memory_mb": mean([r.get("peak_gpu_memory_mb") for r in rows]),
        "mean_num_visual_inputs": mean([r.get("num_visual_inputs") for r in rows]),
        "mean_answer_length": mean(answer_lengths),
        "mean_reasoning_tokens": mean([r.get("raw_reasoning_tokens") or r.get("reasoning_length_tokens") for r in rows]),
        "mean_compressed_reasoning_tokens": mean([r.get("compressed_reasoning_tokens") for r in rows]),
        "mean_reasoning_compression_ratio": mean([r.get("reasoning_compression_ratio") for r in rows]),
        "extraction_success_rate": mean(extraction_ok),
    }


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_md(path, rows):
    fields = [
        "mode",
        "N",
        "exact_match_mean",
        "contains_match_mean",
        "mean_latency_ms_total",
        "mean_latency_ms_generate",
        "mean_peak_gpu_memory_mb",
        "mean_num_visual_inputs",
        "mean_reasoning_tokens",
        "mean_compressed_reasoning_tokens",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Thesis Experiment Summary\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---" for _ in fields]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(k)) for k in fields) + " |\n")


def build_cross_table(records):
    counts = defaultdict(int)
    for row in records:
        bbox_ok = row.get("bbox_correct_at_05")
        answer_ok = row.get("contains_match")
        if bbox_ok is None:
            key = "bbox_missing_or_parse_failed"
        elif bbox_ok and answer_ok:
            key = "bbox_correct_answer_correct"
        elif bbox_ok and not answer_ok:
            key = "bbox_correct_answer_wrong"
        elif not bbox_ok and answer_ok:
            key = "bbox_wrong_answer_correct"
        else:
            key = "bbox_wrong_answer_wrong"
        counts[(get_mode(row), key)] += 1
    keys = [
        "bbox_correct_answer_correct",
        "bbox_correct_answer_wrong",
        "bbox_wrong_answer_correct",
        "bbox_wrong_answer_wrong",
        "bbox_missing_or_parse_failed",
    ]
    modes = sorted({m for m, _ in counts})
    return [{"mode": mode, **{k: counts[(mode, k)] for k in keys}} for mode in modes], ["mode"] + keys


def main():
    parser = argparse.ArgumentParser(description="Aggregate thesis Visual-CoT JSONL outputs.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()

    records = list(iter_jsonl(args.raw_dir))
    if not records:
        raise FileNotFoundError(f"No JSONL files found in {args.raw_dir}")
    groups = defaultdict(list)
    for row in records:
        groups[get_mode(row)].append(row)
    summary = [metric_row(mode, rows) for mode, rows in sorted(groups.items())]
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary[0].keys())
    write_csv(metrics_dir / "summary.csv", summary, summary_fields)
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_md(metrics_dir / "summary.md", summary)

    base = next((r for r in summary if r["mode"] == "pred_bbox"), None)
    compression_rows = []
    reasoning_rows = []
    for row in summary:
        compression_rows.append({
            "mode": row["mode"],
            "compression_type": row["mode"],
            "visual_input_reduction": ratio_delta(row["mean_num_visual_inputs"], base["mean_num_visual_inputs"]) if base else None,
            "latency_reduction_vs_pred_bbox": ratio_delta(row["mean_latency_ms_total"], base["mean_latency_ms_total"]) if base else None,
            "memory_reduction_vs_pred_bbox": ratio_delta(row["mean_peak_gpu_memory_mb"], base["mean_peak_gpu_memory_mb"]) if base else None,
            "accuracy_delta_vs_pred_bbox": (as_float(row["contains_match_mean"]) - as_float(base["contains_match_mean"])) if base and as_float(row["contains_match_mean"]) is not None and as_float(base["contains_match_mean"]) is not None else None,
        })
        reasoning_rows.append({
            "mode": row["mode"],
            "reasoning_length_reduction": ratio_delta(row["mean_compressed_reasoning_tokens"], row["mean_reasoning_tokens"]),
            "answer_accuracy_delta": (as_float(row["contains_match_mean"]) - as_float(base["contains_match_mean"])) if base and as_float(row["contains_match_mean"]) is not None and as_float(base["contains_match_mean"]) is not None else None,
            "generation_latency_delta": (as_float(row["mean_latency_ms_generate"]) - as_float(base["mean_latency_ms_generate"])) if base and as_float(row["mean_latency_ms_generate"]) is not None and as_float(base["mean_latency_ms_generate"]) is not None else None,
            "extraction_success_rate": row["extraction_success_rate"],
        })
    write_csv(metrics_dir / "compression_efficiency.csv", compression_rows, list(compression_rows[0].keys()))
    write_csv(metrics_dir / "reasoning_optimization.csv", reasoning_rows, list(reasoning_rows[0].keys()))
    cross_rows, cross_fields = build_cross_table(records)
    write_csv(metrics_dir / "bbox_answer_cross_table.csv", cross_rows, cross_fields)
    error_rows = [r for r in records if not r.get("contains_match")]
    error_fields = ["question_id", "mode", "image", "gt_answer", "answer_extracted", "bbox_iou", "error_type", "notes"]
    write_csv(metrics_dir / "error_cases.csv", error_rows, error_fields)
    print(f"Wrote metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
