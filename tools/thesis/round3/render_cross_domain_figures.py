#!/usr/bin/env python3
"""§4.7 跨域 DSAC 验证两张论文图（中文版）。

- fig_4_12_cross_domain_cm_zh: 5 数据集 × 4 配置 分组柱状图（CM）
- fig_4_13_dsac_gain_by_difficulty_zh: 按数据集展示 DSAC tlow06+oracle 相对 SE 基线的 CM 增益

输出：experiments/thesis_experiments/runs/20260523_cross_domain_dsac/figures/{png,pdf,svg}
"""
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

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
    "se":       "#7C8FA3",
    "default":  "#4F6F8E",
    "tlow05":   "#E07B39",
    "tlow06":   "#6F9A4F",
}

RUN = Path("experiments/thesis_experiments/runs/20260523_cross_domain_dsac")
OUT_DIR = RUN / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    ("docvqa",          "DocVQA"),
    ("infographicsvqa", "InfographicsVQA"),
    ("visual7w",        "Visual7W"),
    ("gqa_spatial",     "GQA-spatial"),
    ("cub100",          "CUB100"),
]


def load_jsonl(p: Path):
    if not p.exists():
        return None
    return [json.loads(l) for l in p.open() if l.strip()]


def cm_of(rows):
    if rows is None:
        return None
    xs = [r["contains_match"] for r in rows if r.get("contains_match") is not None]
    return sum(xs) / len(xs) if xs else None


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.{{png,svg,pdf}}")


def fig_4_12():
    """5 数据集 × 3 配置 分组柱状图。
    跳过 DSAC default（与 SE baseline 完全等价），改画 SE / tlow05+center / tlow06+oracle，
    避免标签重叠。
    """
    summary_path = RUN / "metrics" / "cross_domain_summary.json"
    summary = json.load(summary_path.open(encoding="utf-8"))

    config_tags = [
        ("se_baseline",        "SE 基线 (= DSAC default)", PALETTE["se"]),
        ("dsac_tlow05_center", "DSAC tlow05+center",        PALETTE["tlow05"]),
        ("dsac_tlow06_oracle", "DSAC tlow06+oracle (上限)", PALETTE["tlow06"]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(DOMAINS))
    width = 0.25
    for i, (tag, label, color) in enumerate(config_tags):
        ys = []
        for domain, _ in DOMAINS:
            s = summary.get(domain, {}).get(tag)
            ys.append(np.nan if s is None or s.get("cm") is None else s["cm"])
        offset = (i - (len(config_tags) - 1) / 2) * width
        bars = ax.bar(x + offset, ys, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for b, y in zip(bars, ys):
            if not np.isnan(y):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.012, f"{y:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([d[1] for d in DOMAINS], rotation=0)
    ax.set_ylabel("ContainsMatch")
    ax.set_title("图 4.12  跨领域 DSAC 验证 — ContainsMatch（n=100/域）")
    ax.set_ylim(0, 0.95)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    save(fig, "fig_4_12_cross_domain_cm_zh")


def _paired_gap(a_rows, se_rows):
    """返回 (n_paired, b10, b01, gap_cm)。"""
    if a_rows is None or se_rows is None:
        return None
    se_by = {r["question_id"]: r.get("contains_match") for r in se_rows}
    b10 = b01 = n = a_correct = se_correct = 0
    for r in a_rows:
        qid = r["question_id"]
        d_cm = r.get("contains_match")
        s_cm = se_by.get(qid)
        if d_cm is None or s_cm is None:
            continue
        n += 1
        a_correct += int(d_cm == 1)
        se_correct += int(s_cm == 1)
        if d_cm == 1 and s_cm == 0:
            b10 += 1
        elif d_cm == 0 and s_cm == 1:
            b01 += 1
    if n == 0:
        return None
    return n, b10, b01, (a_correct - se_correct) / n


def fig_4_13():
    """DSAC tlow06+oracle 相对 SE 基线的 ΔCM（按数据集，含 χ² 标注）。

    每个数据集一根柱子；显示 +Δ；并标 χ²/p-value（来自 McNemar）。
    """
    points = []
    for domain, desc in DOMAINS:
        if domain == "cub100":
            cub_base = Path("experiments/thesis_experiments/runs/20260522_dsac_v7/raw_cub")
            se   = load_jsonl(cub_base / "dsac_se_baseline.jsonl")
            oracle = load_jsonl(cub_base / "dsac_tlow06_oracle.jsonl")
        else:
            se     = load_jsonl(RUN / "raw" / f"{domain}_se_baseline.jsonl")
            oracle = load_jsonl(RUN / "raw" / f"{domain}_dsac_tlow06_oracle.jsonl")
        got = _paired_gap(oracle, se)
        if got is None:
            continue
        n, b10, b01, gap = got
        # McNemar 连续性校正
        disc = b10 + b01
        if disc == 0:
            chi2 = 0.0
        else:
            chi2 = (abs(b10 - b01) - 1) ** 2 / disc
        # 单自由度 χ² 的 p 值（erf 近似）
        from math import erf, sqrt
        if chi2 <= 0:
            p = 1.0
        else:
            z = sqrt(chi2)
            p = (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))) * 2.0
        points.append((desc, gap, chi2, p, n, b10, b01))

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    labels = [p[0] for p in points]
    gaps = [p[1] for p in points]
    ps = [p[3] for p in points]
    colors = [PALETTE["tlow06"] if p < 0.05 else PALETTE["default"] for p in ps]
    x = np.arange(len(points))
    bars = ax.bar(x, gaps, 0.55, color=colors, edgecolor="white", linewidth=0.5)
    for b, (desc, gap, chi2, p, n, b10, b01) in zip(bars, points):
        sig = "*" if p < 0.05 else ""
        ax.text(b.get_x() + b.get_width() / 2, max(gap, 0) + 0.003,
                f"+{gap:.3f}{sig}\nb10={b10}/b01={b01}\np={p:.3f}",
                ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="#777", linewidth=0.8, linestyle="--", alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("ΔContainsMatch (Oracle − SE)")
    ax.set_title("图 4.13  DSAC tlow06+oracle 相对 SE 基线的 CM 增益（per dataset）")
    ymax = max(gaps) + 0.05
    ax.set_ylim(-0.01, ymax)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    # 子标题脚注
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PALETTE["tlow06"], label="McNemar p < 0.05"),
        Patch(facecolor=PALETTE["default"], label="McNemar p ≥ 0.05"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, "fig_4_13_dsac_gain_by_difficulty_zh")


if __name__ == "__main__":
    fig_4_12()
    fig_4_13()
