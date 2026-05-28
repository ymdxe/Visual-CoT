import argparse
import json
import random
from pathlib import Path

from PIL import Image


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_names(sample):
    images = sample.get("image") or []
    if isinstance(images, str):
        images = [images]
    return [str(item).split("###")[0] for item in images if item]


def first_image_name(sample):
    names = image_names(sample)
    return names[0] if names else None


def image_exists(sample, image_folder):
    name = first_image_name(sample)
    return bool(name) and (image_folder / name).exists()


def question_text(sample):
    conversations = sample.get("conversations") or []
    if not conversations:
        return ""
    text = conversations[0].get("value", "")
    text = text.replace("<image>\n", "").replace("<image>", "").strip()
    markers = [
        "Please provide the bounding box coordinate of the region",
        "Please provide the bounding box",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[0]
            break
    return text.strip()


def gt_answer(sample):
    for key in ("answer", "gt_answer", "label"):
        if sample.get(key) not in (None, ""):
            return sample.get(key)
    conversations = sample.get("conversations") or []
    for turn in reversed(conversations):
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            value = turn.get("value")
            if value and not str(value).strip().startswith("["):
                return value
    return None


def pixel_bbox_from_image_field(sample):
    images = sample.get("image") or []
    if not isinstance(images, list) or len(images) < 2:
        return None
    text = str(images[1])
    if "###" not in text:
        return None
    bbox_text = text.split("###", 1)[1].strip().strip("[]")
    try:
        values = [float(v.strip()) for v in bbox_text.split(",")[:4]]
    except ValueError:
        return None
    return values if len(values) == 4 else None


def image_size(sample, image_folder):
    name = first_image_name(sample)
    if not name:
        return None, None
    try:
        with Image.open(image_folder / name) as img:
            return img.size
    except Exception:
        return sample.get("width"), sample.get("height")


def detection_row(sample, image_folder):
    name = first_image_name(sample)
    bbox = pixel_bbox_from_image_field(sample)
    width, height = image_size(sample, image_folder)
    if not name or bbox is None or not width or not height:
        return None
    norm = [
        bbox[0] / float(width),
        bbox[1] / float(height),
        bbox[2] / float(width),
        bbox[3] / float(height),
    ]
    return {
        "img_path": name,
        "expression": question_text(sample),
        "text": "[%.6f, %.6f, %.6f, %.6f]" % tuple(norm),
        "bbox": bbox,
        "bbox_normalized": norm,
        "height": height,
        "width": width,
        "question_id": sample.get("question_id"),
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "gt_answer": gt_answer(sample),
    }


def load_candidates(benchmark_dir, image_folder, datasets, seed):
    rng = random.Random(seed)
    pools = {}
    stats = {}
    for dataset in datasets:
        path = benchmark_dir / f"{dataset}.json"
        if not path.exists():
            pools[dataset] = []
            stats[dataset] = {
                "missing_file": True,
                "total": 0,
                "available": 0,
                "missing_images": 0,
                "selected": 0,
            }
            continue
        samples = load_json(path)
        valid = []
        missing_images = 0
        for sample in samples:
            sample.setdefault("dataset", dataset)
            if image_exists(sample, image_folder):
                valid.append(sample)
            else:
                missing_images += 1
        rng.shuffle(valid)
        pools[dataset] = valid
        stats[dataset] = {
            "missing_file": False,
            "total": len(samples),
            "available": len(valid),
            "missing_images": missing_images,
            "selected": 0,
        }
    return pools, stats


def balanced_select(pools, stats, priority, target_samples):
    selected = []
    offsets = {name: 0 for name in priority}
    while len(selected) < target_samples:
        progressed = False
        for dataset in priority:
            if len(selected) >= target_samples:
                break
            pool = pools.get(dataset) or []
            pos = offsets.get(dataset, 0)
            if pos >= len(pool):
                continue
            sample = pool[pos]
            offsets[dataset] = pos + 1
            selected.append(sample)
            stats[dataset]["selected"] += 1
            progressed = True
        if not progressed:
            break
    return selected


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats_md(path, stats, selected_count, target_samples, priority):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cross-Dataset Region Benchmark Build\n\n")
        f.write(f"- target_samples: {target_samples}\n")
        f.write(f"- selected_samples: {selected_count}\n")
        f.write(f"- dataset_priority: {', '.join(priority)}\n\n")
        f.write("| dataset | selected | available | total | missing_images | missing_file |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for dataset in priority:
            row = stats.get(dataset, {})
            f.write(
                f"| {dataset} | {row.get('selected', 0)} | {row.get('available', 0)} | "
                f"{row.get('total', 0)} | {row.get('missing_images', 0)} | "
                f"{row.get('missing_file', False)} |\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build a balanced cross-dataset benchmark for region evidence experiments."
    )
    parser.add_argument("--benchmark-dir", default="./viscot_benchmark/benchmark")
    parser.add_argument("--image-folder", default="./playground/data")
    parser.add_argument("--datasets", default="textvqa,docvqa,sroie")
    parser.add_argument("--fallback-datasets", default="visual7w,gqa,infographicsvqa")
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--output", default="./data/benchmarks/small_benchmark_cross_region100.json")
    parser.add_argument("--det-output", default="./data/benchmarks/small_benchmark_cross_region100_det.jsonl")
    parser.add_argument("--stats-json", default="./data/benchmarks/small_benchmark_cross_region100_stats.json")
    parser.add_argument("--stats-md", default="./data/benchmarks/small_benchmark_cross_region100_stats.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    primary = [d.strip() for d in args.datasets.split(",") if d.strip()]
    fallback = [d.strip() for d in args.fallback_datasets.split(",") if d.strip()]
    priority = []
    for dataset in primary + fallback:
        if dataset not in priority:
            priority.append(dataset)

    benchmark_dir = Path(args.benchmark_dir)
    image_folder = Path(args.image_folder)
    pools, stats = load_candidates(benchmark_dir, image_folder, priority, args.seed)
    selected = balanced_select(pools, stats, priority, args.target_samples)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    det_rows = []
    skipped_det = 0
    for sample in selected:
        row = detection_row(sample, image_folder)
        if row is None:
            skipped_det += 1
        else:
            det_rows.append(row)
    write_jsonl(Path(args.det_output), det_rows)

    stats_payload = {
        "target_samples": args.target_samples,
        "selected_samples": len(selected),
        "detection_samples": len(det_rows),
        "skipped_detection_rows": skipped_det,
        "dataset_priority": priority,
        "stats": stats,
    }
    Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.stats_json).open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)
    write_stats_md(Path(args.stats_md), stats, len(selected), args.target_samples, priority)

    print(f"[INFO] wrote {len(selected)} samples to {output}")
    print(f"[INFO] wrote {len(det_rows)} detection rows to {args.det_output}")
    if len(selected) < args.target_samples:
        print(f"[WARN] selected fewer samples than requested: {len(selected)} < {args.target_samples}")
    if skipped_det:
        print(f"[WARN] skipped detection rows without usable bbox/size: {skipped_det}")


if __name__ == "__main__":
    main()
