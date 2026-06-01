"""毕设贡献流程图：本文相对上游 Visual-CoT 的工作

按 nature-figure 工作流（schematic-led composite）输出可发表级图示。
- 上行：上游三阶段 baseline pipeline（灰）
- 中行：方法学贡献 §4.2 / §4.4.1 / §5.2 / §4.6
- 下行：工程层改造（builder / det_loader / cot_loader）

输出：images/contributions_flowchart.{png,svg}
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

# nature-figure rcParams（editable text / 7pt）
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
})

# NMI pastel 调色板（低饱和，directional cue 留给 green/red）
COLOR = {
    "baseline":  "#CFD3DB",      # 灰：上游 pipeline
    "baseline_edge": "#6B7280",
    "method_a":  "#A8C4DD",      # 蓝：§4.4.1 评分门控
    "method_b":  "#C8DBC0",      # 沙绿：§5.2 输入压缩
    "method_c":  "#E8C9A0",      # 暖沙：§4.2 结构化证据
    "method_d":  "#D9C2DB",      # 雾紫：§4.6 DSAC
    "eng":       "#EDE3D0",      # 米色：工程改造
    "eng_edge":  "#A08E6F",
    "text":      "#1F2937",
    "muted":     "#4B5563",
    "arrow":     "#374151",
    "callout":   "#9CA3AF",
}


def fancy_box(ax, x, y, w, h, label, fc, ec, fontsize=8, fontweight="normal",
              pad=0.012, va="center", ha="center"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.012",
        linewidth=0.8,
        edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label,
            ha=ha, va=va, color=COLOR["text"],
            fontsize=fontsize, fontweight=fontweight,
            zorder=3, linespacing=1.35)
    return box


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.0, style="-|>", mutation=10, ls="-"):
    color = color or COLOR["arrow"]
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style,
                        mutation_scale=mutation,
                        linewidth=lw, color=color,
                        linestyle=ls, zorder=2.5,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def dashed_drop(ax, x1, y1, x2, y2):
    """虚线引线（baseline → 贡献 callout）"""
    arrow(ax, x1, y1, x2, y2, color=COLOR["callout"], lw=0.6, style="-", ls=(0, (3, 2)))


def main(out_dir: Path):
    fig = plt.figure(figsize=(7.09, 5.12), dpi=300)  # 180mm × 130mm
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------- Title ----------
    ax.text(0.02, 0.965, "本文相对上游 Visual-CoT 的工作版图",
            fontsize=11, fontweight="bold", color=COLOR["text"], ha="left", va="top")
    ax.text(0.02, 0.935,
            "灰色为上游三阶段 baseline，彩色为本文方法学贡献（§4.2 / §4.4.1 / §5.2 / §4.6），底栏为工程层改造",
            fontsize=7.5, color=COLOR["muted"], ha="left", va="top")

    # ---------- Baseline pipeline (top row) ----------
    # 4 boxes equally spaced
    base_y, base_h = 0.79, 0.085
    base_w = 0.165
    base_gap = (1 - 0.04 - 4 * base_w) / 3
    base_x0 = 0.02

    stages = [
        ("输入\n图像 I + 问题 q", "Input"),
        ("区域定位\nDθ : I, q → b_hat", "Stage 1"),
        ("局部裁剪\nC(I, b_hat, α)", "Stage 2"),
        ("答案生成\nAφ : I, q, b_hat → y", "Stage 3"),
    ]
    base_centers = []
    for i, (label, _) in enumerate(stages):
        x = base_x0 + i * (base_w + base_gap)
        fancy_box(ax, x, base_y, base_w, base_h, label,
                  fc=COLOR["baseline"], ec=COLOR["baseline_edge"],
                  fontsize=8, fontweight="bold")
        base_centers.append(x + base_w / 2)

    # arrows between baseline boxes
    for i in range(3):
        x_from = base_x0 + i * (base_w + base_gap) + base_w
        x_to = base_x0 + (i + 1) * (base_w + base_gap)
        arrow(ax,
              x_from + 0.003, base_y + base_h/2,
              x_to - 0.003, base_y + base_h/2,
              lw=1.2, mutation=12)

    # ---------- Methodological contributions (middle row) ----------
    # 4 callout panels, aligned roughly under stages 1..3
    mid_y_top = 0.66
    mid_h = 0.20
    callout_w = 0.21

    callouts = [
        # (center_x_target, panel_x, color, header, body, baseline_idx_for_dropline)
        (base_centers[1], base_centers[1] - callout_w/2, COLOR["method_a"],
         "§4.4.1  边界框评分门控（本文提出）",
         "score(b_hat) = (λ_1·s_parse + λ_2·s_size + λ_3·s_focus) / Σ λ_i\n"
         "阈值化回退：score < θ  →  FallbackBox\n"
         "CUB100 / 跨域 100 上 +0.02 ~ +0.05 CM",
         1),
        (base_centers[2], base_centers[2] - callout_w/2, COLOR["method_b"],
         "§5.2  区域级视觉输入压缩",
         "Φ_full / Φ_pred / Φ_crop / Φ_mix 四算子\n"
         "压缩率 η(Φ) = 1 − τ(Φ)/τ(Φ_pred)\n"
         "Φ_crop：η = 0.5，延迟最低 (212 ms)",
         2),
        (base_centers[3], base_centers[3] - callout_w/2, COLOR["method_c"],
         "§4.2  结构化区域证据提示",
         "π_struct(q, b) = q ‖ bbox(b) ‖ T_schema\n"
         "[Region] ‖ [VE] ‖ [Reasoning] ‖ [Answer]\n"
         "结构一致性得分 SCS，跨域最优 CM = 0.43",
         3),
    ]
    for cx, px, fc, header, body, _ in callouts:
        # box
        fancy_box(ax, px, mid_y_top - mid_h, callout_w, mid_h, "",
                  fc=fc, ec=COLOR["baseline_edge"])
        # header
        ax.text(px + callout_w/2, mid_y_top - 0.020, header,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=COLOR["text"], zorder=4)
        # body
        ax.text(px + callout_w/2, mid_y_top - 0.055, body,
                ha="center", va="top", fontsize=7,
                color=COLOR["text"], linespacing=1.55, zorder=4)
        # dashed drop line from baseline
        dashed_drop(ax, cx, base_y, cx, mid_y_top - 0.001)

    # ---------- DSAC bridging callout (spans stage2..stage3) ----------
    dsac_left = base_centers[2] - 0.02
    dsac_right = base_centers[3] + 0.02
    dsac_x = dsac_left
    dsac_w = dsac_right - dsac_left
    dsac_y_top = mid_y_top - mid_h - 0.030
    dsac_h = 0.118
    fancy_box(ax, dsac_x, dsac_y_top - dsac_h, dsac_w, dsac_h, "",
              fc=COLOR["method_d"], ec=COLOR["baseline_edge"])
    ax.text(dsac_x + dsac_w/2, dsac_y_top - 0.018,
            "§4.6  双端自适应控制器 DSAC（本文提出）",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=COLOR["text"], zorder=4)
    ax.text(dsac_x + dsac_w/2, dsac_y_top - 0.046,
            "composite = α · score(b_hat) + (1 − α) · SCS(y)    "
            "三档决策：accept / re-extract / fallback\n"
            "联通输入端边界框可信度与输出端结构一致性，回退率与 IoU≥0.5 幸存率联合可控",
            ha="center", va="top", fontsize=7,
            color=COLOR["text"], linespacing=1.55, zorder=4)
    # link DSAC to both stage2 and stage3
    arrow(ax, base_centers[2], mid_y_top - mid_h - 0.001,
          dsac_x + dsac_w * 0.30, dsac_y_top, color=COLOR["callout"],
          lw=0.6, style="-", ls=(0, (3, 2)), mutation=4)
    arrow(ax, base_centers[3], mid_y_top - mid_h - 0.001,
          dsac_x + dsac_w * 0.70, dsac_y_top, color=COLOR["callout"],
          lw=0.6, style="-", ls=(0, (3, 2)), mutation=4)

    # ---------- Engineering improvements (bottom banner) ----------
    eng_y_top = 0.24
    eng_h = 0.20
    eng_x0 = 0.02
    eng_total_w = 1 - 0.04
    eng_w = (eng_total_w - 2 * 0.014) / 3

    eng_items = [
        ("llava/model/builder.py",
         "+ 4-bit (NF4 + double-quant)\n"
         "  / 8-bit / bf16 加载\n"
         "+ 关键字参数化接口\n"
         "  向后兼容"),
        ("llava/eval/model_cot_det_loader.py",
         "+ question_id 修复\n"
         "+ 量化 CLI、--dataset-name\n"
         "  / --max-samples\n"
         "+ JSONL：bbox_pred / bbox_iou\n"
         "  / latency_ms / peak_gpu_mem\n"
         "+ --save-vis 可视化检测框"),
        ("llava/eval/model_cot_loader.py\n(200 → 1043 行)",
         "+ 16 modes × 30+ CLI flags\n"
         "  × 25+ JSONL 字段\n"
         "+ score_pred_bbox\n"
         "+ parse_structured_output\n"
         "+ DSAC / 视觉剪枝 / PCA\n"
         "+ 推理压缩 feasibility 接口"),
    ]
    for i, (header, body) in enumerate(eng_items):
        x = eng_x0 + i * (eng_w + 0.014)
        fancy_box(ax, x, eng_y_top - eng_h, eng_w, eng_h, "",
                  fc=COLOR["eng"], ec=COLOR["eng_edge"])
        ax.text(x + eng_w/2, eng_y_top - 0.012, header,
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color=COLOR["text"], zorder=4, linespacing=1.3)
        ax.text(x + 0.010, eng_y_top - 0.055, body,
                ha="left", va="top", fontsize=6.4,
                color=COLOR["text"], linespacing=1.6, zorder=4)

    # section labels (left margin)  — 移除，避免与内容框重叠

    # ---------- Footer (source data / archetype tag) ----------
    ax.text(0.98, 0.012,
            "Archetype: schematic-led composite  ·  Source: CLAUDE.md §2-§3, STATUS.md round3",
            fontsize=6.5, color=COLOR["muted"], ha="right", va="bottom",
            style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "contributions_flowchart.png"
    svg_path = out_dir / "contributions_flowchart.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_path, svg_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[3] / "images"
    png, svg = main(out_dir)
    print(f"wrote {png}")
    print(f"wrote {svg}")
