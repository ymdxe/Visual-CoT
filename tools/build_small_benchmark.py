import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _image_exists(sample, image_folder: Path) -> bool:
    images = sample.get("image") or []
    if isinstance(images, str):
        images = [images]
    if not images:
        return False
    image_name = str(images[0]).split("###")[0]
    return (image_folder / image_name).exists()


def _question_text(sample):
    conversations = sample.get("conversations") or []
    if not conversations:
        return ""
    text = conversations[0].get("value", "")
    text = text.replace("<image>\n", "").replace("<image>", "").strip()
    if "Please provide the bounding box coordinate of the region" in text:
        text = text.split("Please provide the bounding box coordinate of the region")[0]
    elif "Please provide the bounding box" in text:
        text = text.split("Please provide the bounding box")[0]
    return text.strip()


def _pixel_bbox_from_image_field(sample):
    images = sample.get("image") or []
    if not isinstance(images, list) or len(images) < 2 or "###" not in str(images[1]):
        return None
    bbox_text = str(images[1]).split("###", 1)[1].strip().strip("[]")
    try:
        return [float(v.strip()) for v in bbox_text.split(",")[:4]]
    except ValueError:
        return None


def _det_row(sample, image_folder: Path):
    images = sample.get("image") or []
    if isinstance(images, str):
        images = [images]
    if not images:
        return None
    image_name = str(images[0]).split("###")[0]
    image_path = image_folder / image_name
    bbox = _pixel_bbox_from_image_field(sample)
    if bbox is None:
        return None
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width = sample.get("width")
        height = sample.get("height")
    if width is None or height is None:
        return None
    norm_bbox = [
        bbox[0] / float(width),
        bbox[1] / float(height),
        bbox[2] / float(width),
        bbox[3] / float(height),
    ]
    bbox_text = "[%.6f, %.6f, %.6f, %.6f]" % tuple(norm_bbox)
    return {
        "img_path": image_name,
        "expression": _question_text(sample),
        "text": bbox_text,
        "bbox": bbox,
        "bbox_normalized": norm_bbox,
        "height": height,
        "width": width,
        "question_id": sample.get("question_id"),
        "split": sample.get("split"),
    }


def build_small_benchmark(
    benchmark_dir: Path,
    image_folder: Path,
    datasets,
    num_per_dataset: int,
    output: Path,
    det_output: Path,
    seed: int,
):
    random.seed(seed)
    all_samples = []
    det_rows = []
    stats = {}

    for dataset in datasets:
        dataset = dataset.strip()
        if not dataset:
            continue
        path = benchmark_dir / f"{dataset}.json"
        if not path.exists():
            stats[dataset] = {"selected": 0, "missing_file": True, "skipped_images": 0}
            print(f"[WARN] missing benchmark file: {path}")
            continue

        samples = _load_json(path)
        random.shuffle(samples)
        selected = []
        skipped_images = 0
        for sample in samples:
            if not _image_exists(sample, image_folder):
                skipped_images += 1
                continue
            selected.append(sample)
            if len(selected) >= num_per_dataset:
                break
        all_samples.extend(selected)
        if det_output is not None:
            for sample in selected:
                row = _det_row(sample, image_folder)
                if row is not None:
                    det_rows.append(row)
        stats[dataset] = {
            "selected": len(selected),
            "missing_file": False,
            "skipped_images": skipped_images,
        }
        print(
            f"[INFO] {dataset}: selected={len(selected)}, skipped_missing_images={skipped_images}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"[INFO] wrote {len(all_samples)} samples to {output}")
    if det_output is not None:
        det_output.parent.mkdir(parents=True, exist_ok=True)
        with det_output.open("w", encoding="utf-8") as f:
            for row in det_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[INFO] wrote {len(det_rows)} detection samples to {det_output}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="./viscot_benchmark/benchmark")
    parser.add_argument("--image-folder", default="./playground/data")
    parser.add_argument("--datasets", default="gqa,textvqa,docvqa")
    parser.add_argument("--num-per-dataset", type=int, default=10)
    parser.add_argument("--output", default="./data/benchmarks/small_benchmark.json")
    parser.add_argument("--det-output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    build_small_benchmark(
        Path(args.benchmark_dir),
        Path(args.image_folder),
        datasets,
        args.num_per_dataset,
        Path(args.output),
        Path(args.det_output) if args.det_output else None,
        args.seed,
    )


if __name__ == "__main__":
    main()
