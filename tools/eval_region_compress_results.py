import argparse
import csv
import glob
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation + "，。！？；：、“”‘’（）【】《》—…")


def normalize_answer(text):
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_match(pred, gold, normalized=False):
    if pred is None or gold is None:
        return 0.0
    pred = str(pred).strip()
    gold = str(gold).strip()
    if not pred or not gold:
        return 0.0
    if normalized:
        pred = normalize_answer(pred)
        gold = normalize_answer(gold)
    else:
        pred = pred.lower()
        gold = gold.lower()
    if not pred or not gold:
        return 0.0
    return 1.0 if gold in pred or pred in gold else 0.0


def _mode_from_path(path):
    parts = Path(path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _get_ground_truth(row):
    for key in ("gt_answer", "ground_truth", "answer", "label"):
        if row.get(key) not in (None, ""):
            return row.get(key)
    conversations = row.get("raw_conversations") or row.get("conversations")
    if isinstance(conversations, list):
        for turn in reversed(conversations):
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                value = turn.get("value")
                if value not in (None, ""):
                    return value
    return None


def _safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_files(files):
    grouped = defaultdict(list)
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row.get("metadata") or {}
                mode = meta.get("compress_mode") or row.get("compress_mode") or _mode_from_path(path)
                policy = meta.get("visual_input_policy") or row.get("visual_input_policy") or "unknown"
                grouped[(mode, policy)].append((row, path))

    rows = []
    for mode, policy in sorted(grouped):
        records = grouped[(mode, policy)]
        latencies = []
        preprocess_latencies = []
        generation_latencies = []
        end_to_end_latencies = []
        memories = []
        num_visual_images = []
        bbox_valid = []
        bbox_areas = []
        raw_scores = []
        norm_scores = []

        for row, _path in records:
            meta = row.get("metadata") or {}
            latencies.append(meta.get("latency_sec"))
            preprocess_latencies.append(meta.get("preprocess_latency_sec"))
            generation_latencies.append(meta.get("generation_latency_sec"))
            end_to_end_latencies.append(meta.get("end_to_end_latency_sec"))
            memories.append(meta.get("max_gpu_memory_mb"))
            num_visual_images.append(meta.get("num_visual_images"))
            if "bbox_valid" in meta:
                bbox_valid.append(1.0 if meta.get("bbox_valid") else 0.0)
            if meta.get("bbox_area") is not None:
                bbox_areas.append(meta.get("bbox_area"))

            pred = row.get("text")
            gold = _get_ground_truth(row)
            raw_scores.append(contains_match(pred, gold, normalized=False))
            norm_scores.append(contains_match(pred, gold, normalized=True))

        rows.append(
            {
                "compress_mode": mode,
                "visual_input_policy": policy,
                "num_samples": len(records),
                "avg_latency_sec": _safe_mean(latencies),
                "avg_preprocess_latency_sec": _safe_mean(preprocess_latencies),
                "avg_generation_latency_sec": _safe_mean(generation_latencies),
                "avg_end_to_end_latency_sec": _safe_mean(end_to_end_latencies),
                "avg_max_gpu_memory_mb": _safe_mean(memories),
                "avg_num_visual_images": _safe_mean(num_visual_images),
                "bbox_valid_rate": _safe_mean(bbox_valid),
                "avg_bbox_area": _safe_mean(bbox_areas),
                "contains_match": _safe_mean(raw_scores),
                "normalized_contains_match": _safe_mean(norm_scores),
            }
        )
    return rows


def _fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(rows, csv_path, md_path):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "compress_mode",
        "visual_input_policy",
        "num_samples",
        "avg_latency_sec",
        "avg_preprocess_latency_sec",
        "avg_generation_latency_sec",
        "avg_end_to_end_latency_sec",
        "avg_max_gpu_memory_mb",
        "avg_num_visual_images",
        "bbox_valid_rate",
        "avg_bbox_area",
        "contains_match",
        "normalized_contains_match",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| compress_mode | visual_input_policy | num_samples | avg_latency_sec | avg_preprocess_latency_sec | avg_generation_latency_sec | avg_end_to_end_latency_sec | avg_max_gpu_memory_mb | avg_num_visual_images | bbox_valid_rate | avg_bbox_area | contains_match | normalized_contains_match |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(_fmt(row[name]) for name in fieldnames)
                + " |\n"
            )
    print(f"[INFO] wrote {csv_path}")
    print(f"[INFO] wrote {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="result jsonl files")
    parser.add_argument("--inputs", nargs="*", default=None, help="result jsonl files or glob patterns")
    parser.add_argument("--output-csv", default="results/region_compress/summary.csv")
    parser.add_argument("--output-md", default="results/region_compress/summary.md")
    args = parser.parse_args()

    patterns = []
    if args.inputs:
        patterns.extend(args.inputs)
    patterns.extend(args.files)
    if not patterns:
        patterns = ["results/region_compress/*/*.jsonl"]

    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])
    files = sorted({f for f in files if os.path.isfile(f)})
    if not files:
        raise FileNotFoundError("No result jsonl files found.")

    rows = summarize_files(files)
    write_outputs(rows, args.output_csv, args.output_md)


if __name__ == "__main__":
    main()
