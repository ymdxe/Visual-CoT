"""Re-render the 5 figures used in chapters 3/4/5 of the thesis with Chinese labels.

Targets the exact images referenced in `5论文初稿第三版-...docx`:
  image9.png  ← 图 3.7  (IoU 分箱)
  image12.png ← 图 4.6  (边界框评分门控消融)
  image17.png ← 图 5.4  (低清分辨率灵敏性扫描)
  image18.png ← 图 5.5  (裁剪 padding 系数灵敏性扫描)
  image20.png ← 图 5.6  (多种子稳定性)

Uses Noto Sans CJK JP for Chinese glyphs (it shares the CJK Unified Ideographs glyph
table with the SC variant, so simplified Chinese renders correctly).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Font setup: force-register Noto CJK TTC ----------

for fp in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})

PALETTE = {
    "full": "#7C8FA3",
    "pred_bbox": "#4F6F8E",
    "structured_evidence": "#4A90B8",
    "lowres_full_highrescrop": "#79B2C9",
    "crop_only": "#A6CFD6",
    "default": "#E07B39",
    "strict": "#C25E1F",
    "lax": "#F2A26B",
    "no_focus": "#D1D1D1",
    "no_size": "#B5B5B5",
    "oracle_fallback": "#6F9A4F",
    "baseline_no_scoring": "#7C8FA3",
}

EXP_ROOT = Path("/root/code/Visual-CoT/experiments/thesis_experiments/runs/20260522_round2_extensions")
OUT_DIR = Path("/root/code/Visual-CoT/Obsidian/assets/round2/paper_figures_zh")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CN_LABEL = {
    "pred_bbox": "预测区域",
    "structured_evidence": "结构化提示",
    "lowres_full_highrescrop": "低清整图+局部",
    "crop_only": "仅局部",
    "random_bbox": "随机区域",
    "center_bbox": "中心区域",
    "oracle_bbox": "理想区域",
    "full": "整图",
    "woimg": "无图像",
    "baseline_no_scoring": "基线\n（无评分）",
    "default_t05_l1_1_0.5": "默认\nθ=0.5",
    "strict_t07_l1_1_0.5": "严格\nθ=0.7",
    "lax_t03_l1_1_0.5": "宽松\nθ=0.3",
    "no_focus_l1_1_0": "无 focus\nλ₃=0",
    "no_size_l1_0_0.5": "无 size\nλ₂=0",
    "oracle_fallback": "Oracle\n回退",
}


def short_color(key):
    if key.startswith("default"):  return PALETTE["default"]
    if key.startswith("strict"):   return PALETTE["strict"]
    if key.startswith("lax"):      return PALETTE["lax"]
    if key.startswith("no_focus"): return PALETTE["no_focus"]
    if key.startswith("no_size"):  return PALETTE["no_size"]
    if key.startswith("oracle"):   return PALETTE["oracle_fallback"]
    return PALETTE.get(key, "#444444")


def save_png(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.{{png,svg,pdf}}")


# ---------- 图 3.7: IoU 分箱 ----------

def fig_iou_bins():
    df = pd.read_csv(EXP_ROOT / "analysis/cross_region100/A1_iou_bins.csv")
    df = df.dropna(subset=["contains_match"])
    modes = ["pred_bbox", "structured_evidence", "lowres_full_highrescrop", "crop_only"]
    bins = sorted({(r["iou_low"], r["iou_high"]) for _, r in df.iterrows()})
    bin_labels = [f"[{lo:.1f}, {hi:.2f})" for lo, hi in bins]
    fig, ax = plt.subplots(figsize=(120/25.4, 75/25.4))
    width = 0.18
    x = np.arange(len(bins))
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode].sort_values("iou_low")
        vals = sub["contains_match"].tolist()
        ax.bar(x + (i - 1.5) * width, vals, width=width,
               color=PALETTE[mode], label=CN_LABEL[mode],
               edgecolor="white", linewidth=0.5)
    n_text = "样本数 = " + ", ".join(str(int(v)) for v in df[df["mode"] == "pred_bbox"].sort_values("iou_low")["n"])
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_xlabel("预测边界框 IoU 区间")
    ax.set_ylabel("包含性匹配")
    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=1.2)
    ax.annotate(n_text, xy=(0.5, -0.30), xycoords="axes fraction",
                ha="center", va="top", fontsize=7, color="#555")
    ax.set_title("按预测边界框 IoU 分箱的答案准确率")
    fig.tight_layout()
    save_png(fig, "fig_3_7_iou_bins_zh")


# ---------- 图 4.6: 评分门控消融 ----------

def fig_bbox_scoring():
    df = pd.read_csv(EXP_ROOT / "metrics/C2_bbox_scoring.csv")
    order = [
        "baseline_no_scoring",
        "lax_t03_l1_1_0.5",
        "no_focus_l1_1_0",
        "no_size_l1_0_0.5",
        "default_t05_l1_1_0.5",
        "strict_t07_l1_1_0.5",
        "oracle_fallback",
    ]
    df = df.set_index("label").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 80/25.4),
                             gridspec_kw={"width_ratios": [1, 0.9]})

    # Panel a
    ax = axes[0]
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    cm = df["contains_match"].tolist()
    colors = [short_color(k) for k in labels]
    bars = ax.bar(x, cm, color=colors, edgecolor="white", linewidth=0.6)
    for bar, val, fr in zip(bars, cm, df["fallback_rate"].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#222")
        if fr and fr > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.05,
                    f"回退{fr:.0%}", ha="center", va="top", fontsize=7, color="white",
                    weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CN_LABEL[k] for k in labels], fontsize=7.5)
    ax.set_ylabel("包含性匹配")
    ax.set_ylim(0.35, 0.55)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(df.loc[df["label"] == "baseline_no_scoring", "contains_match"].iloc[0],
               color=PALETTE["full"], linestyle=":", linewidth=0.9, alpha=0.8)
    ax.text(-0.06, 1.05, "a", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")
    ax.set_title("不同评分配置下的答案准确率")

    # Panel b
    ax2 = axes[1]
    iou_kept = df["iou_at_05_only_kept"].tolist()
    valid = [(k, v, c) for k, v, c in zip(labels, iou_kept, [short_color(k) for k in labels])
             if v == v and v is not None]
    if valid:
        ks, vs, cs = zip(*valid)
        xs = np.arange(len(ks))
        bars2 = ax2.bar(xs, vs, color=cs, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars2, vs):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color="#222")
        ax2.set_xticks(xs)
        ax2.set_xticklabels([CN_LABEL[k] for k in ks], fontsize=7.5)
        ax2.set_ylabel("保留样本中 IoU≥0.5 的比例")
        ax2.set_ylim(0, max(vs) * 1.25)
        ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax2.set_axisbelow(True)
    ax2.text(-0.08, 1.05, "b", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", va="bottom")
    ax2.set_title("门控筛选后的边界框质量")
    fig.tight_layout()
    save_png(fig, "fig_4_6_bbox_scoring_zh")


# ---------- 图 5.4: lowres 灵敏性 ----------

def fig_lowres_sweep():
    df = pd.read_csv(EXP_ROOT / "metrics/B1_lowres_sweep.csv").sort_values("lowres_size")
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(170/25.4, 75/25.4))
    x = df["lowres_size"].tolist()
    ax_acc.plot(x, df["contains_match_mean"], "o-",
                color=PALETTE["structured_evidence"], linewidth=1.6, markersize=6, label="包含性匹配")
    ax_acc.plot(x, df["exact_match_mean"], "s--",
                color=PALETTE["default"], linewidth=1.3, markersize=5, label="严格匹配")
    ax_acc.set_xlabel("低清分辨率 $s_{low}$（px）")
    ax_acc.set_ylabel("准确率")
    ax_acc.set_ylim(0.2, 0.45)
    ax_acc.set_xticks(x)
    ax_acc.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax_acc.set_axisbelow(True)
    ax_acc.legend(loc="upper left", fontsize=7.5)
    ax_acc.text(-0.12, 1.05, "a", transform=ax_acc.transAxes,
                fontsize=11, fontweight="bold", va="bottom")
    ax_acc.set_title("准确率随低清分辨率的变化")

    ax_lat.plot(x, df["latency_ms_mean"], "o-",
                color=PALETTE["pred_bbox"], linewidth=1.6, markersize=6, label="总延迟")
    ax_lat.plot(x, df["generate_ms_mean"], "s--",
                color=PALETTE["lowres_full_highrescrop"], linewidth=1.3, markersize=5, label="生成延迟")
    ax_lat.plot(x, df["preprocess_ms_mean"], "d:", color="#3D3D3D",
                linewidth=1.0, markersize=4, label="预处理延迟")
    ax_lat.set_xlabel("低清分辨率 $s_{low}$（px）")
    ax_lat.set_ylabel("延迟（ms）")
    ax_lat.set_xticks(x)
    ax_lat.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax_lat.set_axisbelow(True)
    ax_lat.legend(loc="upper left", fontsize=7.5)
    ax_lat.text(-0.12, 1.05, "b", transform=ax_lat.transAxes,
                fontsize=11, fontweight="bold", va="bottom")
    ax_lat.set_title("延迟随低清分辨率的变化")
    fig.tight_layout()
    save_png(fig, "fig_5_4_lowres_sweep_zh")


# ---------- 图 5.5: padding 灵敏性 ----------

def fig_padding_sweep():
    df = pd.read_csv(EXP_ROOT / "metrics/B2_padding_sweep.csv").sort_values("crop_pad")
    fig, ax = plt.subplots(figsize=(120/25.4, 75/25.4))
    x = df["crop_pad"].tolist()
    ax.plot(x, df["contains_match_mean"], "o-",
            color=PALETTE["structured_evidence"], linewidth=1.7, markersize=7, label="包含性匹配")
    ax.plot(x, df["exact_match_mean"], "s--",
            color=PALETTE["default"], linewidth=1.3, markersize=6, label="严格匹配")
    ax.set_xlabel("裁剪 padding 系数 α")
    ax.set_ylabel("准确率")
    ax.set_xticks(x)
    ax.set_ylim(0.22, 0.46)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("裁剪 padding 系数 α 的灵敏性")
    fig.tight_layout()
    save_png(fig, "fig_5_5_padding_sweep_zh")


# ---------- 图 5.6: 多 seed 稳定性 ----------

def fig_seed_stability():
    df = pd.read_csv(EXP_ROOT / "metrics/C1_seed_stability.csv").sort_values("cm_mean")
    fig, ax = plt.subplots(figsize=(130/25.4, 80/25.4))
    modes = df["mode"].tolist()
    y = np.arange(len(modes))
    cm = df["cm_mean"].tolist()
    cm_err = df["cm_std"].tolist()
    colors = [PALETTE.get(m, "#444444") for m in modes]
    ax.barh(y, cm, xerr=cm_err, color=colors, edgecolor="white",
            linewidth=0.5, capsize=4, error_kw=dict(elinewidth=0.9, ecolor="#222"))
    for yi, val, err in zip(y, cm, cm_err):
        ax.text(val + err + 0.005, yi, f"{val:.3f} ± {err:.3f}",
                va="center", fontsize=7.5, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels([CN_LABEL.get(m, m).replace("\n", " ") for m in modes], fontsize=8)
    ax.set_xlabel("包含性匹配（3 种子均值 ± 标准差）")
    ax.set_xlim(0, max(np.array(cm) + np.array(cm_err)) * 1.30)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("多种子稳定性（T = 0.2, top_p = 0.95）")
    fig.tight_layout()
    save_png(fig, "fig_5_6_seed_stability_zh")


def main():
    print(f"输出目录: {OUT_DIR}")
    fig_iou_bins()
    fig_bbox_scoring()
    fig_lowres_sweep()
    fig_padding_sweep()
    fig_seed_stability()
    print("完成")


if __name__ == "__main__":
    main()
