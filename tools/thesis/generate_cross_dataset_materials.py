import argparse
import csv
import json
from pathlib import Path


def load_csv(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value):
    if value in (None, "", "NA"):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return str(value)


def table(rows, fields):
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join(["---" for _ in fields]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper materials for cross-dataset region experiments.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--stats-json", default=None)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    overall = load_csv(metrics_dir / "summary_cross_overall.csv")
    by_dataset = load_csv(metrics_dir / "summary_by_dataset.csv")
    stats = {}
    if args.stats_json and Path(args.stats_json).exists():
        with open(args.stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

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
    (paper_dir / "cross_dataset_tables.md").write_text(
        "# Cross-Dataset Tables\n\n"
        "## Overall\n\n"
        + table(overall, fields)
        + "\n\n## By Dataset\n\n"
        + table(by_dataset, fields),
        encoding="utf-8",
    )

    selected = stats.get("selected_samples", "NA")
    det = stats.get("detection_samples", "NA")
    priority = ", ".join(stats.get("dataset_priority", [])) or "textvqa, docvqa, sroie"
    (paper_dir / "cross_dataset_experiment_section.md").write_text(
        "# 跨数据集区域证据补充实验草稿\n\n"
        f"本轮补充实验从 `{priority}` 等数据集中构建跨数据集小样本集合，"
        f"目标样本数为 100，实际回答样本数为 {selected}，可用于 detection 的样本数为 {det}。"
        "实验沿用 Visual-CoT 的 detection-answer 两阶段流程，并比较整图输入、预测框区域、仅局部区域、"
        "低清整图+高清局部、结构化提示、无图像输入、中心框和随机框等设置。\n\n"
        "该实验的目的不是证明模型结构被重新训练或完成模型内部压缩，而是检验输入侧区域证据增强方法"
        "在 CUB 以外的文字、文档或区域指代类样本上是否仍然具有作用，并通过随机框、中心框负对照"
        "分析语言先验和中心偏置对结果解释的影响。\n\n"
        "正文中建议将本实验作为 CUB100 之后的进一步验证：若 `pred_bbox`、`lowres_full_highrescrop` 或"
        "`structured_evidence` 在 TextVQA/DocVQA/SROIE 上相对 `full` 或 `woimg` 保持优势，说明区域证据"
        "对局部细节任务具有一定泛化价值；若随机框或中心框接近预测框，则应强调当前模型仍存在语言先验"
        "或数据偏置，不能把结果解释为定位能力的单独提升。\n",
        encoding="utf-8",
    )

    (paper_dir / "cross_dataset_figures.md").write_text(
        "# Cross-Dataset Figures\n\n"
        "- `figures/accuracy_bar.png`\n"
        "- `figures/latency_bar.png`\n"
        "- `figures/memory_bar.png`\n"
        "- `figures/accuracy_latency_scatter.png`\n"
        "- `metrics/summary_by_dataset.csv`\n"
        "- `metrics/summary_cross_overall.csv`\n",
        encoding="utf-8",
    )
    print(f"Wrote cross-dataset paper materials to {paper_dir}")


if __name__ == "__main__":
    main()
