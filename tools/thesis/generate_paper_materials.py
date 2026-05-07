import argparse
import csv
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


def write_table(path, title, rows, fields):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---" for _ in fields]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Generate thesis paper materials from real metrics.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--paper-dir", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    summary = load_csv(metrics_dir / "summary.csv")
    compression = load_csv(metrics_dir / "compression_efficiency.csv")
    reasoning = load_csv(metrics_dir / "reasoning_optimization.csv")
    cross = load_csv(metrics_dir / "bbox_answer_cross_table.csv")

    tables_path = paper_dir / "paper_tables.md"
    tables_path.write_text("# Paper Tables\n", encoding="utf-8")
    write_table(
        tables_path,
        "不同输入压缩策略准确率与效率对比",
        summary,
        ["mode", "N", "contains_match_mean", "mean_latency_ms_total", "mean_peak_gpu_memory_mb", "mean_num_visual_inputs", "mean_crop_ratio"],
    )
    write_table(
        tables_path,
        "图像压缩效率对比",
        compression,
        ["mode", "visual_input_reduction", "latency_reduction_vs_pred_bbox", "memory_reduction_vs_pred_bbox", "accuracy_delta_vs_pred_bbox"],
    )
    write_table(
        tables_path,
        "推理优化策略对比",
        reasoning,
        ["mode", "reasoning_length_reduction", "answer_accuracy_delta", "generation_latency_delta", "extraction_success_rate"],
    )
    write_table(tables_path, "bbox 正确性与 answer 正确性交叉表", cross, cross[0].keys() if cross else ["mode"])

    figure_names = [
        "accuracy_bar.png",
        "latency_bar.png",
        "memory_bar.png",
        "accuracy_latency_scatter.png",
        "compression_accuracy_tradeoff.png",
        "reasoning_length_bar.png",
        "reasoning_compression_ratio.png",
        "bbox_iou_hist.png",
        "crop_ratio_hist.png",
    ]
    with open(paper_dir / "paper_figures.md", "w", encoding="utf-8") as f:
        f.write("# Paper Figures\n\n")
        for name in figure_names:
            f.write(f"- `{Path(args.figures_dir) / name}`\n")

    modes = {row.get("mode") for row in summary}
    with open(paper_dir / "method_section_addition.md", "w", encoding="utf-8") as f:
        f.write(
            "# 方法补充\n\n"
            "本文在 Visual-CoT 的 detection-answer 两阶段流程上进行输入侧与推理侧的轻量优化。"
            "输入侧实验包括全图输入、预测框双输入、仅裁剪区域输入、低分辨率全图加高分辨率裁剪区域输入，以及中心框、随机框、无图像等负对照。"
            "这些方法属于区域级视觉输入压缩，不等同于底层视觉 token 剪枝或模型参数剪枝。\n\n"
            "推理侧实验包括结构化证据提示、规则版冗余文本剪除、先预测后解释的两阶段输出策略，以及启发式关键步骤选择接口。"
            "规则压缩发生在生成之后，因此主要减少后续展示或二次处理文本长度，不能直接降低第一次生成延迟。\n"
        )

    with open(paper_dir / "experiment_section_draft.md", "w", encoding="utf-8") as f:
        f.write("# 实验结果草稿\n\n")
        f.write(f"本轮实验实际产生的模式包括：{', '.join(sorted(m for m in modes if m))}。\n\n")
        f.write("所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。\n\n")
        for row in summary:
            f.write(
                f"- `{row.get('mode')}`: N={row.get('N')}, contains_match={fmt(row.get('contains_match_mean'))}, "
                f"latency={fmt(row.get('mean_latency_ms_total'))} ms, memory={fmt(row.get('mean_peak_gpu_memory_mb'))} MB。\n"
            )

    with open(paper_dir / "limitation_and_future_work.md", "w", encoding="utf-8") as f:
        f.write(
            "# 不足与展望\n\n"
            "- 当前样本规模仍有限，正式结论需要结合更大样本和多次运行稳定性验证。\n"
            "- CUB 图像可能存在主体居中偏置，因此 center_bbox 和 random_bbox 负对照需要作为解释边界。\n"
            "- 输入侧区域压缩不等同于底层视觉 token 剪枝，也不等同于模型参数剪枝。\n"
            "- ViT/CLIP 视觉编码器的结构化剪枝可能需要重新微调，否则容易造成特征分布破坏。\n"
            "- PCA 特征压缩如果没有训练 adapter，通常只能作为接口和可行性检查，不能直接声明为高性能压缩方法。\n"
            "- 本轮只实现启发式关键步骤选择接口，完整强化学习控制器尚未作为主实验完成。\n"
        )
    print(f"Wrote paper materials to {paper_dir}")


if __name__ == "__main__":
    main()
