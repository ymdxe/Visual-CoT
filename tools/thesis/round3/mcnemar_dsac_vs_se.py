#!/usr/bin/env python3
"""McNemar 显著性检验：DSAC default vs structured_evidence baseline。

输出：experiments/thesis_experiments/runs/20260522_dsac_v7/metrics/mcnemar_dsac_vs_se.md
"""
import json
from pathlib import Path
from math import erf, sqrt

DSAC_PATH = Path("experiments/thesis_experiments/runs/20260522_dsac_v7/raw/dsac_default.jsonl")
SE_PATH   = Path("experiments/thesis_experiments/runs/20260516_cross_region100_public3/raw/answer_structured_evidence.jsonl")
OUT_PATH  = Path("experiments/thesis_experiments/runs/20260522_dsac_v7/metrics/mcnemar_dsac_vs_se.md")


def load_by_qid(path):
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            qid = d.get("question_id")
            cm = d.get("contains_match")
            em = d.get("exact_match")
            if qid is None or cm is None:
                continue
            table[qid] = {"cm": int(cm), "em": int(em) if em is not None else None}
    return table


def chi2_sf_1df(x):
    """Survival function of chi-square with 1 df = 2 * (1 - Phi(sqrt(x))) for x>=0."""
    if x <= 0:
        return 1.0
    z = sqrt(x)
    # P(Z>z) for standard normal:
    return erf(-z / sqrt(2.0)) * -0.5 + 0.5 if False else (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))) * 2.0


def mcnemar(a, b, name_a, name_b, metric):
    """Paired McNemar test. a, b are dicts qid -> {cm, em}."""
    common = sorted(set(a.keys()) & set(b.keys()))
    n = len(common)
    b01 = 0  # a wrong, b right
    b10 = 0  # a right, b wrong
    b00 = 0
    b11 = 0
    for q in common:
        va = a[q][metric]
        vb = b[q][metric]
        if va == 1 and vb == 1:
            b11 += 1
        elif va == 0 and vb == 0:
            b00 += 1
        elif va == 1 and vb == 0:
            b10 += 1
        else:
            b01 += 1
    disc = b01 + b10
    if disc == 0:
        return {"n": n, "b00": b00, "b01": b01, "b10": b10, "b11": b11,
                "stat": 0.0, "p": 1.0, "note": "no discordant pairs"}
    stat = (abs(b01 - b10) - 1) ** 2 / disc  # continuity-corrected
    p = chi2_sf_1df(stat)
    return {"n": n, "b00": b00, "b01": b01, "b10": b10, "b11": b11,
            "disc": disc, "stat": stat, "p": p,
            f"acc_{name_a}": (b10 + b11) / n,
            f"acc_{name_b}": (b01 + b11) / n}


def main():
    dsac = load_by_qid(DSAC_PATH)
    se = load_by_qid(SE_PATH)
    print(f"Loaded DSAC: {len(dsac)} rows, SE: {len(se)} rows")
    common = sorted(set(dsac.keys()) & set(se.keys()))
    print(f"Common qids: {len(common)}")

    cm_res = mcnemar(dsac, se, "dsac", "se", "cm")
    em_res = mcnemar(dsac, se, "dsac", "se", "em")
    print("CM:", cm_res)
    print("EM:", em_res)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# McNemar 配对显著性检验：DSAC default vs structured_evidence baseline\n\n")
        f.write("**配对依据**：question_id 同集；跨数据集 100 样本（TextVQA + GQA + Visual7W）。\n\n")
        f.write(f"**样本数**：DSAC default n={len(dsac)}，SE baseline n={len(se)}，**配对成功 n={len(common)}**\n\n")
        f.write("公式：χ² = (|b₀₁ − b₁₀| − 1)² / (b₀₁ + b₁₀)（含连续性校正），自由度 1。\n\n")
        f.write("---\n\n")

        f.write("## 一、ContainsMatch 配对表\n\n")
        f.write("|  | SE 错 (cm=0) | SE 对 (cm=1) | 合计 |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| DSAC 错 | b00={cm_res['b00']} | b01={cm_res['b01']} | {cm_res['b00']+cm_res['b01']} |\n")
        f.write(f"| DSAC 对 | b10={cm_res['b10']} | b11={cm_res['b11']} | {cm_res['b10']+cm_res['b11']} |\n")
        f.write(f"| 合计 | {cm_res['b00']+cm_res['b10']} | {cm_res['b01']+cm_res['b11']} | {cm_res['n']} |\n\n")
        f.write(f"- 不一致对 disc = b₀₁ + b₁₀ = {cm_res.get('disc', 0)}\n")
        f.write(f"- χ² (含连续性校正) = **{cm_res['stat']:.4f}**\n")
        f.write(f"- p-value (双尾，自由度 1) = **{cm_res['p']:.4f}**\n")
        f.write(f"- DSAC CM = {cm_res['acc_dsac']:.3f}，SE CM = {cm_res['acc_se']:.3f}，gap = {cm_res['acc_dsac']-cm_res['acc_se']:+.3f}\n\n")

        f.write("## 二、ExactMatch 配对表\n\n")
        f.write("|  | SE 错 (em=0) | SE 对 (em=1) | 合计 |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| DSAC 错 | b00={em_res['b00']} | b01={em_res['b01']} | {em_res['b00']+em_res['b01']} |\n")
        f.write(f"| DSAC 对 | b10={em_res['b10']} | b11={em_res['b11']} | {em_res['b10']+em_res['b11']} |\n")
        f.write(f"| 合计 | {em_res['b00']+em_res['b10']} | {em_res['b01']+em_res['b11']} | {em_res['n']} |\n\n")
        f.write(f"- 不一致对 disc = {em_res.get('disc', 0)}\n")
        f.write(f"- χ² = **{em_res['stat']:.4f}**\n")
        f.write(f"- p-value = **{em_res['p']:.4f}**\n")
        f.write(f"- DSAC EM = {em_res['acc_dsac']:.3f}，SE EM = {em_res['acc_se']:.3f}，gap = {em_res['acc_dsac']-em_res['acc_se']:+.3f}\n\n")

        f.write("---\n\n")
        f.write("## 三、结论\n\n")
        cm_sig = "显著" if cm_res['p'] < 0.05 else "**不显著**"
        em_sig = "显著" if em_res['p'] < 0.05 else "**不显著**"
        f.write(f"- CM 维度：p = {cm_res['p']:.4f} → α=0.05 下{cm_sig}。\n")
        f.write(f"- EM 维度：p = {em_res['p']:.4f} → α=0.05 下{em_sig}。\n\n")

        if cm_res['p'] >= 0.05:
            f.write("**诚实警示（必须写入论文 §4.6）**：DSAC default 相对 structured_evidence baseline 的 +0.005 CM 提升在 McNemar 配对检验下 p>0.05，**不具有统计显著性**。需以以下两条之一支撑章节论证：\n\n")
            f.write("1. composite 信号的判别力（accept vs re_extract gap +0.114 CM，见 dsac_decision_breakdown.md），这是 *样本可信度信号* 的证据，与“DSAC 默认配置能涨点”是两回事。\n")
            f.write("2. oracle fallback 上限设定 (tlow06_oracle)：CM=0.459 vs SE baseline=0.429，提升 +0.030。这条 *依赖未来工程的回退框*，写为“上限”而非“主结果”。\n\n")

    print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    main()
