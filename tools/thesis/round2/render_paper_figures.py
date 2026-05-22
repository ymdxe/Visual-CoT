"""Regenerate all Round-2 figures with nature-figure publication styling.

Outputs SVG (editable text), PDF (editable text), PNG (preview), TIFF (600 dpi for print)
to Obsidian/assets/round2/paper_figures/.

Style follows the nature-figure quick-start:
- Arial sans-serif, base 7pt (paper-ready)
- No top/right spines, 0.8pt axis line
- Editable text in SVG / PDF
- Restrained palette: neutral baseline + signal blue + accent orange + warning red

Color families (NMI-pastel inspired):
- BASELINE  #4F6F8E  steel blue       — full / pred_bbox
- EVIDENCE  #4A90B8  medium blue      — structured_evidence, lowres+highrescrop
- ALGORITHM #E07B39  warm orange      — bbox-scoring (this paper's new method)
- NEG_CTRL  #B0B0B0  neutral grey     — random_bbox / center_bbox
- FAILURE   #C0504D  brick red        — woimg
- ORACLE    #6F9A4F  olive green      — oracle_bbox / oracle fallback
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------- Global styling ----------

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.dpi": 120,
})

PALETTE = {
    "full": "#7C8FA3",
    "pred_bbox": "#4F6F8E",
    "structured_evidence": "#4A90B8",
    "lowres_full_highrescrop": "#79B2C9",
    "crop_only": "#A6CFD6",
    "random_bbox": "#B0B0B0",
    "center_bbox": "#909090",
    "woimg": "#C0504D",
    "oracle_bbox": "#6F9A4F",
    "baseline_no_scoring": "#7C8FA3",
    "default": "#E07B39",
    "strict": "#C25E1F",
    "lax": "#F2A26B",
    "no_focus": "#D1D1D1",
    "no_size": "#B5B5B5",
    "oracle_fallback": "#6F9A4F",
}

EXPERIMENT_ROOT = Path("/root/code/Visual-CoT/experiments/thesis_experiments/runs/20260522_round2_extensions")
OUT_DIR = Path("/root/code/Visual-CoT/Obsidian/assets/round2/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_all(fig, name: str, dpi_tiff: int = 600) -> None:
    """Save the same figure as SVG / PDF / PNG / TIFF."""
    base = OUT_DIR / name
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{base}.tiff", bbox_inches="tight", dpi=dpi_tiff,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"  saved {name}.{{svg,pdf,png,tiff}}")


def pretty_label(key: str) -> str:
    mapping = {
        "pred_bbox": "Pred BBox",
        "structured_evidence": "Structured\nEvidence",
        "lowres_full_highrescrop": "Lowres+Crop",
        "crop_only": "Crop Only",
        "random_bbox": "Random BBox",
        "center_bbox": "Center BBox",
        "oracle_bbox": "Oracle BBox",
        "full": "Full Image",
        "woimg": "No Image",
        "baseline_no_scoring": "Baseline\n(no scoring)",
        "default_t05_l1_1_0.5": "Default\nθ=0.5",
        "strict_t07_l1_1_0.5": "Strict\nθ=0.7",
        "lax_t03_l1_1_0.5": "Lax\nθ=0.3",
        "no_focus_l1_1_0": "No focus\nλ₃=0",
        "no_size_l1_0_0.5": "No size\nλ₂=0",
        "oracle_fallback": "Oracle\nfallback",
    }
    return mapping.get(key, key)


def short_color(key: str) -> str:
    # Pull leading token for scoring-family entries
    if key.startswith("default"):
        return PALETTE["default"]
    if key.startswith("strict"):
        return PALETTE["strict"]
    if key.startswith("lax"):
        return PALETTE["lax"]
    if key.startswith("no_focus"):
        return PALETTE["no_focus"]
    if key.startswith("no_size"):
        return PALETTE["no_size"]
    if key.startswith("oracle_fallback"):
        return PALETTE["oracle_fallback"]
    return PALETTE.get(key, "#444444")


# ---------- Figure 1 — A1: IoU bins ----------

def figure_iou_bins():
    df = pd.read_csv(EXPERIMENT_ROOT / "analysis/cross_region100/A1_iou_bins.csv")
    df = df.dropna(subset=["contains_match"])
    modes = ["pred_bbox", "structured_evidence", "lowres_full_highrescrop", "crop_only"]
    bins = sorted(df[["iou_low", "iou_high"]].drop_duplicates().values.tolist())
    bin_labels = [f"[{lo:.1f}, {hi:.2f})" for lo, hi in bins]
    fig, ax = plt.subplots(figsize=(120/25.4, 70/25.4))
    width = 0.18
    x = np.arange(len(bins))
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode].sort_values("iou_low")
        vals = sub["contains_match"].tolist()
        n_per_bin = sub["n"].tolist()
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=PALETTE[mode],
               label=pretty_label(mode).replace("\n", " "), edgecolor="white", linewidth=0.5)
    # n annotation under x-axis
    n_text = "n = " + ", ".join(str(int(v)) for v in df[df["mode"] == "pred_bbox"].sort_values("iou_low")["n"])
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=7.5)
    ax.set_xlabel("Predicted bbox IoU vs ground truth", fontsize=8)
    ax.set_ylabel("Contains Match", fontsize=8)
    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.8, 0.1))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.8, ncol=2, handlelength=1.2)
    ax.annotate(n_text, xy=(0.5, -0.30), xycoords="axes fraction",
                ha="center", va="top", fontsize=6.5, color="#555")
    ax.set_title("Answer accuracy stratified by predicted-bbox IoU", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_a1_iou_bins")


# ---------- Figure 2 — A2: Latency breakdown ----------

def figure_latency_breakdown():
    df = pd.read_csv(EXPERIMENT_ROOT / "analysis/cross_region100/A2_latency_breakdown.csv")
    df = df.sort_values("total_mean_ms").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(150/25.4, 75/25.4))
    modes = df["mode"].tolist()
    x = np.arange(len(modes))
    pre = df["preprocess_mean_ms"].fillna(0).tolist()
    gen = df["generate_mean_ms"].fillna(0).tolist()
    colors = [short_color(m) for m in modes]
    ax.bar(x, gen, color=colors, edgecolor="white", linewidth=0.5, label="Generate")
    ax.bar(x, pre, bottom=gen, color="#3D3D3D", edgecolor="white", linewidth=0.5, label="Preprocess")
    for xi, p, g in zip(x, pre, gen):
        ax.text(xi, g + p + 15, f"{g + p:.0f}", ha="center", va="bottom", fontsize=6.8, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(m) for m in modes], rotation=0, fontsize=7.2)
    ax.set_ylabel("Latency (ms)", fontsize=8)
    ax.set_ylim(0, max(np.array(pre) + np.array(gen)) * 1.18)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
    ax.set_title("Per-stage latency decomposition (cross-domain, n=100)", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_a2_latency_breakdown")


# ---------- Figure 3 — A4: Pareto frontier ----------

def figure_pareto():
    df = pd.read_csv(EXPERIMENT_ROOT / "analysis/cross_region100/A4_pareto.csv")
    df = df.copy()
    # manual annotation offsets to avoid overlaps (mode -> (dx_pt, dy_pt))
    OFFS_LAT = {
        "structured_evidence": (8, 6),
        "pred_bbox":           (-12, 10),
        "lowres_full_highrescrop": (8, -2),
        "full":                (-12, -14),
        "random_bbox":         (8, -4),
        "center_bbox":         (-50, 10),
        "crop_only":           (8, 6),
        "woimg":               (-50, -4),
    }
    OFFS_TOK = {
        "full":                (-32, -14),
        "crop_only":           (8, -6),
        "pred_bbox":           (8, 10),
        "structured_evidence": (8, -2),
        "lowres_full_highrescrop": (-50, 10),
        "random_bbox":         (-90, -4),
        "center_bbox":         (-60, -16),
        "woimg":               (8, 6),
    }
    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 80/25.4))

    for ax, xkey, xlabel, panel, off_map in zip(
            axes,
            ["latency_ms", "tokens"],
            ["Latency (ms)", "Visual token budget τ"],
            ["a", "b"],
            [OFFS_LAT, OFFS_TOK]):
        xs = df[xkey].tolist()
        ys = df["contains_match"].tolist()
        labels = df["mode"].tolist()
        colors = [short_color(m) for m in labels]
        ax.scatter(xs, ys, s=80, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
        # Pareto frontier
        order = sorted(zip(xs, ys, labels), key=lambda t: t[0])
        fx, fy = [], []
        best = -np.inf
        for xv, yv, _ in order:
            if yv > best:
                best = yv
                fx.append(xv); fy.append(yv)
        ax.plot(fx, fy, "--", color=PALETTE["default"], linewidth=1.2, label="Pareto frontier", zorder=2)
        for xv, yv, lab in zip(xs, ys, labels):
            dx, dy = off_map.get(lab, (8, 6))
            ax.annotate(pretty_label(lab).replace("\n", " "), (xv, yv),
                        textcoords="offset points", xytext=(dx, dy), fontsize=6.8, color="#222",
                        ha="left" if dx >= 0 else "left")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Contains Match" if panel == "a" else "", fontsize=8)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0.02, 0.52)
        ax.text(-0.10, 1.05, panel, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
        ax.legend(loc="lower right", fontsize=7, handlelength=1.6)
    axes[0].set_title("Accuracy vs. latency", fontsize=8.5, pad=4)
    axes[1].set_title("Accuracy vs. visual tokens", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_a4_pareto")


# ---------- Figure 4 — B1: lowres sweep ----------

def figure_lowres_sweep():
    df = pd.read_csv(EXPERIMENT_ROOT / "metrics/B1_lowres_sweep.csv").sort_values("lowres_size")
    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(170/25.4, 70/25.4))
    x = df["lowres_size"].tolist()
    ax_acc.plot(x, df["contains_match_mean"], "o-", color=PALETTE["structured_evidence"],
                linewidth=1.4, markersize=5, label="Contains Match")
    ax_acc.plot(x, df["exact_match_mean"], "s--", color=PALETTE["default"],
                linewidth=1.2, markersize=4, label="Exact Match")
    ax_acc.set_xlabel("Low-resolution branch size $s_{low}$ (px)", fontsize=8)
    ax_acc.set_ylabel("Accuracy", fontsize=8)
    ax_acc.set_ylim(0.2, 0.45)
    ax_acc.set_xticks(x)
    ax_acc.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax_acc.set_axisbelow(True)
    ax_acc.legend(loc="upper left", fontsize=7, handlelength=1.6)
    ax_acc.text(-0.12, 1.05, "a", transform=ax_acc.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
    ax_acc.set_title("Accuracy vs. low-res size", fontsize=8.5, pad=4)

    ax_lat.plot(x, df["latency_ms_mean"], "o-", color=PALETTE["pred_bbox"],
                linewidth=1.4, markersize=5, label="Total")
    ax_lat.plot(x, df["generate_ms_mean"], "s--", color=PALETTE["lowres_full_highrescrop"],
                linewidth=1.2, markersize=4, label="Generate")
    ax_lat.plot(x, df["preprocess_ms_mean"], "d:", color="#3D3D3D",
                linewidth=1.0, markersize=4, label="Preprocess")
    ax_lat.set_xlabel("Low-resolution branch size $s_{low}$ (px)", fontsize=8)
    ax_lat.set_ylabel("Latency (ms)", fontsize=8)
    ax_lat.set_xticks(x)
    ax_lat.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax_lat.set_axisbelow(True)
    ax_lat.legend(loc="upper left", fontsize=7, handlelength=1.6)
    ax_lat.text(-0.12, 1.05, "b", transform=ax_lat.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
    ax_lat.set_title("Latency vs. low-res size", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_b1_lowres_sweep")


# ---------- Figure 5 — B2: padding sweep ----------

def figure_padding_sweep():
    df = pd.read_csv(EXPERIMENT_ROOT / "metrics/B2_padding_sweep.csv").sort_values("crop_pad")
    fig, ax = plt.subplots(figsize=(120/25.4, 70/25.4))
    x = df["crop_pad"].tolist()
    ax.plot(x, df["contains_match_mean"], "o-", color=PALETTE["structured_evidence"],
            linewidth=1.5, markersize=6, label="Contains Match")
    ax.plot(x, df["exact_match_mean"], "s--", color=PALETTE["default"],
            linewidth=1.2, markersize=5, label="Exact Match")
    ax.set_xlabel("Crop padding multiplier α", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_xticks(x)
    ax.set_ylim(0.22, 0.46)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.6)
    ax.set_title("Sensitivity to crop padding α", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_b2_padding_sweep")


# ---------- Figure 6 — C1: seed stability ----------

def figure_seed_stability():
    df = pd.read_csv(EXPERIMENT_ROOT / "metrics/C1_seed_stability.csv").sort_values("cm_mean")
    fig, ax = plt.subplots(figsize=(120/25.4, 80/25.4))
    modes = df["mode"].tolist()
    y = np.arange(len(modes))
    cm = df["cm_mean"].tolist()
    cm_err = df["cm_std"].tolist()
    colors = [PALETTE.get(m, "#444444") for m in modes]
    ax.barh(y, cm, xerr=cm_err, color=colors, edgecolor="white",
            linewidth=0.5, capsize=4, error_kw=dict(elinewidth=0.9, ecolor="#222"))
    for yi, val, err in zip(y, cm, cm_err):
        ax.text(val + err + 0.005, yi, f"{val:.3f} ± {err:.3f}",
                va="center", fontsize=7, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_label(m).replace("\n", " ") for m in modes], fontsize=7.5)
    ax.set_xlabel("Contains Match (mean ± std over 3 seeds)", fontsize=8)
    ax.set_xlim(0, max(np.array(cm) + np.array(cm_err)) * 1.30)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Multi-seed stability (T = 0.2, top_p = 0.95)", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_c1_seed_stability")


# ---------- Figure 7 — C2: bbox scoring ablation ----------

def figure_bbox_scoring():
    df = pd.read_csv(EXPERIMENT_ROOT / "metrics/C2_bbox_scoring.csv")
    # Stable order for narrative
    order = [
        "baseline_no_scoring",
        "lax_t03_l1_1_0.5",
        "no_focus_l1_1_0",
        "no_size_l1_0_0.5",
        "default_t05_l1_1_0.5",
        "strict_t07_l1_1_0.5",
        "oracle_fallback",
    ]
    df = df.set_index("label").reindex([k for k in order if k in df["label"].values
                                        or k in df.index]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 80/25.4),
                             gridspec_kw={"width_ratios": [1, 0.9]})

    # Panel a: contains_match bars
    ax = axes[0]
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    cm = df["contains_match"].tolist()
    colors = [short_color(k) for k in labels]
    bars = ax.bar(x, cm, color=colors, edgecolor="white", linewidth=0.6)
    for bar, val, fr in zip(bars, cm, df["fallback_rate"].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="#222")
        if fr and fr > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.05,
                    f"FR={fr:.0%}", ha="center", va="top", fontsize=6.5, color="white",
                    weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(k) for k in labels], fontsize=7.2)
    ax.set_ylabel("Contains Match", fontsize=8)
    ax.set_ylim(0.35, 0.55)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(df.loc[df["label"] == "baseline_no_scoring", "contains_match"].iloc[0],
               color=PALETTE["full"], linestyle=":", linewidth=0.9, alpha=0.8)
    ax.text(-0.06, 1.05, "a", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom")
    ax.set_title("Answer accuracy across scoring configurations", fontsize=8.5, pad=4)

    # Panel b: IoU@0.5 only on kept samples — "survivor pool quality"
    ax2 = axes[1]
    iou_kept = df["iou_at_05_only_kept"].tolist()
    valid = [(k, v, c) for k, v, c in zip(labels, iou_kept, colors) if v == v and v is not None]
    if valid:
        ks, vs, cs = zip(*valid)
        xs = np.arange(len(ks))
        bars2 = ax2.bar(xs, vs, color=cs, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars2, vs):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#222")
        ax2.set_xticks(xs)
        ax2.set_xticklabels([pretty_label(k) for k in ks], fontsize=7.2)
        ax2.set_ylabel("IoU ≥ 0.5 rate (only kept samples)", fontsize=8)
        ax2.set_ylim(0, max(vs) * 1.25)
        ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax2.set_axisbelow(True)
    ax2.text(-0.08, 1.05, "b", transform=ax2.transAxes,
             fontsize=10, fontweight="bold", va="bottom")
    ax2.set_title("Survivor pool cleanliness", fontsize=8.5, pad=4)
    fig.tight_layout()
    save_all(fig, "fig_c2_bbox_scoring")


def main():
    print(f"writing to {OUT_DIR}")
    figure_iou_bins()
    figure_latency_breakdown()
    figure_pareto()
    figure_lowres_sweep()
    figure_padding_sweep()
    figure_seed_stability()
    figure_bbox_scoring()
    print("done.")


if __name__ == "__main__":
    main()
