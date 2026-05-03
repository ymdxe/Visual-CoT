import argparse
import csv
import os


def rows(path):
    with open(path, "r", encoding="utf-8") as f: return list(csv.DictReader(f))


def write_md(path, title, data, fields):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n"); f.write("| " + " | ".join(fields) + " |\n"); f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in data: f.write("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |\n")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summary-csv", required=True); p.add_argument("--output-dir", required=True); args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); data = rows(args.summary_csv)
    write_md(os.path.join(args.output_dir, "main_results.md"), "Main Results", data, ["dataset","mode","substring","bbox_acc","latency_ms","peak_gpu_memory_mb"])
    write_md(os.path.join(args.output_dir, "ablation_results.md"), "Ablation Results", data, ["mode","substring","bbox_acc","latency_ms","peak_gpu_memory_mb","bbox_source"])
    write_md(os.path.join(args.output_dir, "efficiency_results.md"), "Efficiency Results", data, ["mode","crop_ratio","num_visual_inputs","num_visual_tokens_est","latency_ms","peak_gpu_memory_mb","substring"])


if __name__ == "__main__": main()
