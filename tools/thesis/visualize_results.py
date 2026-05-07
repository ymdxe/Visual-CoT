import argparse
import csv
from pathlib import Path


def as_float(value):
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_plot(fig, out_base):
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=180)
    fig.savefig(str(out_base) + ".pdf")


def bar(ax, rows, y_key, title, ylabel):
    labels = [r["mode"] for r in rows]
    vals = [as_float(r.get(y_key)) or 0 for r in rows]
    ax.bar(labels, vals, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)


def main():
    parser = argparse.ArgumentParser(description="Visualize thesis experiment summaries.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(metrics_dir / "summary.csv")
    comp_rows = load_rows(metrics_dir / "compression_efficiency.csv") if (metrics_dir / "compression_efficiency.csv").exists() else rows
    reason_rows = load_rows(metrics_dir / "reasoning_optimization.csv") if (metrics_dir / "reasoning_optimization.csv").exists() else rows

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [r["mode"] for r in rows]
    exact = [as_float(r.get("exact_match_mean")) or 0 for r in rows]
    contains = [as_float(r.get("contains_match_mean")) or 0 for r in rows]
    x = range(len(rows))
    ax.bar([i - 0.18 for i in x], exact, width=0.36, label="Exact")
    ax.bar([i + 0.18 for i in x], contains, width=0.36, label="Contains")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by mode"); ax.legend()
    save_plot(fig, figures_dir / "accuracy_bar"); plt.close(fig)

    for name, key, title, ylabel in [
        ("latency_bar", "mean_latency_ms_total", "Latency by mode", "ms"),
        ("memory_bar", "mean_peak_gpu_memory_mb", "Peak GPU memory by mode", "MB"),
        ("reasoning_length_bar", "mean_reasoning_tokens", "Reasoning length by mode", "tokens"),
        ("reasoning_compression_ratio", "mean_reasoning_compression_ratio", "Reasoning compression ratio", "compressed/raw"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4))
        bar(ax, rows, key, title, ylabel)
        save_plot(fig, figures_dir / name); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        xval = as_float(r.get("mean_latency_ms_total"))
        yval = as_float(r.get("contains_match_mean"))
        if xval is None or yval is None:
            continue
        ax.scatter(xval, yval, s=60)
        ax.annotate(r["mode"], (xval, yval), fontsize=8)
    ax.set_xlabel("Mean latency (ms)"); ax.set_ylabel("Contains match"); ax.set_title("Accuracy-latency tradeoff")
    save_plot(fig, figures_dir / "accuracy_latency_scatter"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["mode"] for r in comp_rows]
    xvals = [as_float(r.get("latency_reduction_vs_pred_bbox")) or 0 for r in comp_rows]
    yvals = [as_float(r.get("accuracy_delta_vs_pred_bbox")) or 0 for r in comp_rows]
    ax.axhline(0, color="#999999", linewidth=0.8); ax.axvline(0, color="#999999", linewidth=0.8)
    ax.scatter(xvals, yvals, color="#F58518")
    for label, xv, yv in zip(labels, xvals, yvals):
        ax.annotate(label, (xv, yv), fontsize=8)
    ax.set_xlabel("Latency reduction vs pred_bbox"); ax.set_ylabel("Accuracy delta vs pred_bbox")
    ax.set_title("Compression accuracy-efficiency tradeoff")
    save_plot(fig, figures_dir / "compression_accuracy_tradeoff"); plt.close(fig)

    raw_dir = metrics_dir.parent / "raw"
    bbox_ious = []
    crop_ratios = []
    for path in raw_dir.glob("*.jsonl"):
        import json
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if as_float(row.get("bbox_iou")) is not None:
                    bbox_ious.append(as_float(row.get("bbox_iou")))
                if as_float(row.get("crop_ratio")) is not None:
                    crop_ratios.append(as_float(row.get("crop_ratio")))
    for name, vals, title, xlabel in [
        ("bbox_iou_hist", bbox_ious, "BBox IoU distribution", "IoU"),
        ("crop_ratio_hist", crop_ratios, "Crop ratio distribution", "crop area / image area"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals or [0], bins=12, color="#54A24B", edgecolor="white")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        save_plot(fig, figures_dir / name); plt.close(fig)

    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
