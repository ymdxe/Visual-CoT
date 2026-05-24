#!/usr/bin/env python3
"""DSAC 实验配图：composite 直方图 + 决策档位 CM 柱状图 + α 扫描对照。

输出：experiments/thesis_experiments/runs/20260522_dsac_v7/figures/{fig_4_7_dsac_flow.txt, fig_4_8_composite_hist.png, fig_4_9_decision_cm.png, fig_4_10_alpha_sweep.png}
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 中文字体（与 round2 保持一致）
for ttc in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]:
    if os.path.exists(ttc):
        try:
            fm.fontManager.addfont(ttc)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RUN = Path("experiments/thesis_experiments/runs/20260522_dsac_v7")
FIG_DIR = RUN / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load(tag):
    fp = RUN / "raw" / f"dsac_{tag}.jsonl"
    return [json.loads(l) for l in open(fp, "r", encoding="utf-8") if l.strip()]


# ===== 图 4.8 composite 分布直方图 =====
def fig_composite_hist():
    rows = load("default")
    composites = [r["metadata"]["composite_gate"]["composite"] for r in rows]
    s_in = [r["metadata"]["composite_gate"]["s_in"] for r in rows]
    s_out = [r["metadata"]["composite_gate"]["s_out"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2), dpi=200)
    ax.hist(composites, bins=15, color="#4C72B0", edgecolor="black", alpha=0.8, label="composite")
    ax.hist(s_in, bins=15, color="#DD8452", edgecolor="black", alpha=0.45, label="score(b) 输入端")
    ax.axvline(0.3, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.6, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(0.7, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.30, ax.get_ylim()[1] * 0.92, "θ_low=0.3", fontsize=8, ha="center")
    ax.text(0.50, ax.get_ylim()[1] * 0.84, "θ_low=0.5", fontsize=8, ha="center", color="red")
    ax.text(0.60, ax.get_ylim()[1] * 0.76, "θ_low=0.6", fontsize=8, ha="center", color="red")
    ax.text(0.70, ax.get_ylim()[1] * 0.92, "θ_high=0.7", fontsize=8, ha="center")
    ax.set_xlabel("得分")
    ax.set_ylabel("样本数")
    ax.set_title("复合得分 composite 与输入端 score 的分布（DSAC 默认配置, 跨数据集 100）")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_8_composite_hist.png", bbox_inches="tight")
    plt.close()


# ===== 图 4.9 决策档位 CM 柱状图（strict 配置 & tlow06_oracle 配置） =====
def fig_decision_cm():
    cfgs = ["alpha07", "strict", "tlow06_oracle"]
    titles = ["α=0.7, θ=(0.3,0.7)", "α=0.5, θ=(0.4,0.6) strict", "α=0.5, θ_low=0.6 + oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200, sharey=True)
    for ax, tag, title in zip(axes, cfgs, titles):
        rows = load(tag)
        buckets = {"accept": [], "re_extract": [], "fallback": []}
        for r in rows:
            d = r["metadata"]["composite_gate"]["decision"]
            cm = r.get("contains_match")
            if cm is not None:
                buckets[d].append(cm)
        names = ["accept", "re_extract", "fallback"]
        ns = [len(buckets[k]) for k in names]
        cms = [np.mean(buckets[k]) if buckets[k] else 0 for k in names]
        bars = ax.bar(names, cms,
                      color=["#55A868", "#4C72B0", "#C44E52"],
                      edgecolor="black", alpha=0.85)
        for bar, n, cm in zip(bars, ns, cms):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(0.02, h + 0.02),
                    f"n={n}\nCM={cm:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("包含性匹配 CM")
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("DSAC 三档决策样本数与包含性匹配（跨数据集 100）", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG_DIR / "fig_4_9_decision_cm.png", bbox_inches="tight")
    plt.close()


# ===== 图 4.10 α 扫描 + θ_low 扫描 (CM vs config) =====
def fig_alpha_sweep():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)

    # 左：α 扫描
    alpha_tags = ["output_only", "alpha03", "default", "alpha07", "input_only"]
    alpha_xs = [0.0, 0.3, 0.5, 0.7, 1.0]
    alpha_cms = []
    for tag in alpha_tags:
        rows = load(tag)
        alpha_cms.append(np.mean([r["contains_match"] for r in rows if r.get("contains_match") is not None]))
    ax1.plot(alpha_xs, alpha_cms, marker="o", color="#4C72B0", linewidth=2, label="DSAC CM")
    ax1.axhline(0.429, color="#DD8452", linestyle="--", linewidth=1.2, label="基线 structured_evidence (0.429)")
    ax1.axhline(0.414, color="#55A868", linestyle=":", linewidth=1.2, label="基线 pred_bbox (0.414)")
    ax1.set_xlabel("α（输入端权重）")
    ax1.set_ylabel("包含性匹配 CM")
    ax1.set_title("α 扫描（θ_low=0.3, θ_high=0.7, center fallback）")
    ax1.set_ylim(0.40, 0.48)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(linestyle="--", alpha=0.3)
    for x, y in zip(alpha_xs, alpha_cms):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    # 右：θ_low 扫描（center vs oracle fallback）
    tags_c = ["default", "strict", "tlow05", "tlow055", "tlow06"]
    tags_o = ["oracle_fb", None, None, None, "tlow06_oracle"]
    xs = [0.3, 0.4, 0.5, 0.55, 0.6]
    cms_c, cms_o = [], []
    for tag in tags_c:
        rows = load(tag)
        cms_c.append(np.mean([r["contains_match"] for r in rows if r.get("contains_match") is not None]))
    for tag in tags_o:
        if tag is None:
            cms_o.append(None)
            continue
        rows = load(tag)
        cms_o.append(np.mean([r["contains_match"] for r in rows if r.get("contains_match") is not None]))
    ax2.plot(xs, cms_c, marker="o", color="#4C72B0", linewidth=2, label="center 回退框")
    ax2.plot([xs[0], xs[-1]], [cms_o[0], cms_o[-1]], marker="s",
             color="#C44E52", linewidth=2, label="oracle 回退框（上限）", linestyle="-")
    ax2.axhline(0.429, color="#DD8452", linestyle="--", linewidth=1.2, label="基线 structured_evidence (0.429)")
    ax2.set_xlabel("θ_low（fallback 触发阈值）")
    ax2.set_ylabel("包含性匹配 CM")
    ax2.set_title("θ_low 扫描（α=0.5, θ_high=0.7）")
    ax2.set_ylim(0.28, 0.52)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(linestyle="--", alpha=0.3)
    for x, y in zip(xs, cms_c):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color="#4C72B0")
    for x, y in zip([xs[0], xs[-1]], [cms_o[0], cms_o[-1]]):
        if y is not None:
            ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=9, color="#C44E52")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_4_10_alpha_theta_sweep.png", bbox_inches="tight")
    plt.close()


# ===== 图 4.7 流程图（ASCII，便于直接放进 docx 或 graphviz 转图） =====
def fig_flow_ascii():
    txt = """
DSAC 双端自适应控制器流程（§4.6 算法 6）

  +-------------+    +------------------+    +---------------+
  |  问题 q +   | -> |  §4.5 score(b̂)  | -> |  §5.2 Φ 视觉   |
  |  预测框 b̂  |    |  输入端评分 s_in |    |  输入构造      |
  +-------------+    +------------------+    +-------+-------+
                                                     |
                                                     v
                                              +------+-------+
                                              |  §4.2 π_struct|
                                              |  结构化提示   |
                                              +------+-------+
                                                     |
                                                     v
                                              +------+-------+
                                              |  A_φ 第一次   |
                                              |  生成 y_1     |
                                              +------+-------+
                                                     |
                                                     v
  +-----------------+    +-----------------+    +----+---+
  | s_out=SCS(y_1)  | -> | composite =     | -> | 三档    |
  | 输出端结构一致性 |    | α·s_in+(1-α)s_out |    | 决策   |
  +-----------------+    +-----------------+    +----+---+
                                                     |
                  +----------------------------------+----------------------------------+
                  | composite ≥ θ_high                 | θ_low ≤ composite < θ_high      | composite < θ_low
                  v                                    v                                  v
          +------+------+                      +------+------+                     +------+--------+
          | accept       |                    | re_extract   |                     | fallback       |
          | Section_A(y) |                    | Adaptive     |                     | 换回退框 b'    |
          |              |                    | extraction   |                     | 二次生成 y_2   |
          +------+------+                      +------+------+                     +------+--------+
                  \\__________________________________/                                  /
                                       \\_________________________________________/
                                                          v
                                                     最终答案 ŷ*
"""
    (FIG_DIR / "fig_4_7_dsac_flow.txt").write_text(txt, encoding="utf-8")


# ===== 图 4.11 SCS 四段产出分布（证明跨数据集 100 上 SCS 恒为 0.5） =====
def fig_scs_distribution():
    rows = load("default")
    flags_count = {"region": 0, "visual_evidence": 0, "reasoning": 0, "answer": 0}
    scs_values = []
    for r in rows:
        cg = r["metadata"]["composite_gate"]
        scs_values.append(cg["s_out"])
        for k, v in cg["scs_flags"].items():
            flags_count[k] += int(v)
    n = len(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

    # 左：四段产出率柱图
    keys = ["region", "visual_evidence", "reasoning", "answer"]
    labels_zh = ["[Region]", "[Visual Evidence]", "[Reasoning]", "[Answer]"]
    rates = [flags_count[k] / n for k in keys]
    colors = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]
    bars = ax1.bar(labels_zh, rates, color=colors, edgecolor="black", alpha=0.85)
    for bar, r in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 max(0.02, bar.get_height() + 0.02),
                 f"{r*100:.0f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("产出率")
    ax1.set_title(f"结构化模板四段产出率（n={n}，跨数据集）")
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # 右：SCS 值分布饼图
    from collections import Counter
    cnt = Counter(round(v * 4) / 4 for v in scs_values)
    pie_labels = [f"SCS={k}\n(n={v})" for k, v in sorted(cnt.items())]
    pie_vals = [v for _, v in sorted(cnt.items())]
    ax2.pie(pie_vals, labels=pie_labels, autopct="%1.0f%%",
            colors=["#C44E52", "#DD8452", "#4C72B0", "#55A868", "#8172B2"][:len(pie_vals)],
            startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.6})
    ax2.set_title("SCS 分布（跨数据集场景下恒为 0.5）")

    fig.suptitle("§4.6 输出端信号在跨数据集 100 上的退化证据", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG_DIR / "fig_4_11_scs_distribution.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_flow_ascii()
    fig_composite_hist()
    fig_decision_cm()
    fig_alpha_sweep()
    fig_scs_distribution()
    print("Figures written:", sorted(p.name for p in FIG_DIR.iterdir()))
