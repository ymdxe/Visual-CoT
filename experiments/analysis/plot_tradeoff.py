import argparse
import csv
import os


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return float("nan")


def plot(rows, xkey, ykey, out):
    import matplotlib.pyplot as plt
    xs, ys, labels = [], [], []
    for row in rows:
        xs.append(fnum(row, xkey)); ys.append(fnum(row, ykey)); labels.append(row.get("mode") or row.get("crop_mode") or "mode")
    plt.figure(figsize=(7, 5)); plt.scatter(xs, ys)
    for x, y, label in zip(xs, ys, labels):
        if x == x and y == y:
            plt.annotate(label, (x, y), fontsize=8)
    plt.xlabel(xkey); plt.ylabel(ykey); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summary-csv", required=True); p.add_argument("--output-dir", required=True); args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); rows = load_rows(args.summary_csv)
    plot(rows, "latency_ms", "substring", os.path.join(args.output_dir, "accuracy_latency.png"))
    plot(rows, "peak_gpu_memory_mb", "substring", os.path.join(args.output_dir, "accuracy_memory.png"))
    plot(rows, "crop_ratio", "substring", os.path.join(args.output_dir, "accuracy_crop_ratio.png"))


if __name__ == "__main__":
    main()
