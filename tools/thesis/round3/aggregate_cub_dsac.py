#!/usr/bin/env python3
"""CUB100 DSAC 验证聚合：4 配置的 CM/EM/Latency/决策档位分布。

输出：experiments/thesis_experiments/runs/20260522_dsac_v7/metrics/cub_dsac_validation.md
"""
import json
from pathlib import Path
from collections import Counter

RUN = Path("experiments/thesis_experiments/runs/20260522_dsac_v7")
RAW = RUN / "raw_cub"
OUT = RUN / "metrics" / "cub_dsac_validation.md"

CONFIGS = [
    ("se_baseline", "SE baseline（无 DSAC, 无 bbox scoring）"),
    ("default",     "DSAC default (α=0.5, θ=(0.3,0.7), center fb)"),
    ("tlow05",      "DSAC tlow05 (α=0.5, θ_low=0.5, center fb)"),
    ("tlow06_oracle","DSAC tlow06+oracle (α=0.5, θ_low=0.6, oracle fb，上限)"),
]


def load(tag):
    fp = RAW / f"dsac_{tag}.jsonl"
    return [json.loads(l) for l in open(fp, "r", encoding="utf-8") if l.strip()]


def summarize(rows):
    cms = [r["contains_match"] for r in rows if r.get("contains_match") is not None]
    ems = [r["exact_match"] for r in rows if r.get("exact_match") is not None]
    lats = [r.get("latency_ms_total") for r in rows if r.get("latency_ms_total") is not None]
    n = len(rows)
    cm = sum(cms) / len(cms) if cms else 0
    em = sum(ems) / len(ems) if ems else 0
    lat = sum(lats) / len(lats) if lats else 0
    # decision distribution
    dec_counts = Counter()
    scs_vals = []
    sin_vals = []
    second_pass = 0
    for r in rows:
        md = r.get("metadata", {})
        cg = md.get("composite_gate")
        if cg is None:
            continue
        dec_counts[cg["decision"]] += 1
        scs_vals.append(cg["s_out"])
        sin_vals.append(cg["s_in"])
        if cg.get("second_pass"):
            second_pass += 1
    return {
        "n": n, "n_cm": len(cms), "cm": cm, "em": em, "lat": lat,
        "dec": dec_counts, "second_pass": second_pass,
        "scs_dist": Counter(round(v * 4) / 4 for v in scs_vals),
        "sin_mean": sum(sin_vals) / len(sin_vals) if sin_vals else None,
    }


def main():
    results = {tag: summarize(load(tag)) for tag, _ in CONFIGS}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# CUB100 单域 DSAC 验证（2026-05-22）\n")
    lines.append("**目的**：检查 DSAC 在单域 CUB（细粒度鸟类识别）上的行为是否与跨数据集 100 不同——\n")
    lines.append("特别关注 SCS 是否仍恒为 0.5、fallback 档是否能在默认 θ_low=0.3 下触发。\n\n")
    lines.append(f"**数据**：small_benchmark_cub100.json (n=100)，VisCoT-7b-224，bf16，T=0.2，top_p=0.95，seed=42。\n\n")

    lines.append("## 一、主表\n\n")
    lines.append("| 配置 | n | CM | EM | Latency | accept | re_extract | fallback | second_pass | SCS mean | s_in mean |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, desc in CONFIGS:
        r = results[tag]
        n = r["n"]
        a = r["dec"].get("accept", 0)
        re_ = r["dec"].get("re_extract", 0)
        fb = r["dec"].get("fallback", 0)
        sp = r["second_pass"]
        scs_disp = "—"
        if r["scs_dist"]:
            total = sum(r["scs_dist"].values())
            scs_disp = sum(k * v for k, v in r["scs_dist"].items()) / total
            scs_disp = f"{scs_disp:.3f}"
        sin_disp = f"{r['sin_mean']:.3f}" if r["sin_mean"] is not None else "—"
        lines.append(f"| {desc} | {n} | {r['cm']:.3f} | {r['em']:.3f} | {r['lat']:.0f} ms | {a} | {re_} | {fb} | {sp} | {scs_disp} | {sin_disp} |\n")

    lines.append("\n## 二、SCS 分布对照（跨数据集 vs CUB）\n\n")
    lines.append("跨数据集 100 上 SCS 几乎恒为 0.5（仅产出 [Reasoning]+[Answer]）。CUB100 上的 SCS 分布如下：\n\n")
    cub_default = results["default"]
    lines.append(f"| SCS 值 | 样本数 (CUB DSAC default) |\n|---|---:|\n")
    for k in sorted(cub_default["scs_dist"].keys()):
        lines.append(f"| {k:.2f} | {cub_default['scs_dist'][k]} |\n")

    lines.append("\n## 三、关键观察\n\n")
    se = results["se_baseline"]
    dflt = results["default"]
    t05 = results["tlow05"]
    tor = results["tlow06_oracle"]

    obs = []
    obs.append(f"1. **SE baseline (CUB100)**：CM={se['cm']:.3f}, EM={se['em']:.3f}, latency={se['lat']:.0f} ms。与历史 CUB100 run 的 structured_evidence (CM=0.73, EM=0.73) 对照可看出方差/温度差异。")
    obs.append(f"2. **DSAC default (CUB100)**：CM={dflt['cm']:.3f}, accept={dflt['dec'].get('accept', 0)}, re_extract={dflt['dec'].get('re_extract', 0)}, fallback={dflt['dec'].get('fallback', 0)}。"
               f"{'fallback 在默认 θ_low=0.3 下触发' if dflt['dec'].get('fallback', 0) > 0 else 'fallback 仍未触发，与跨数据集行为一致'}。")
    obs.append(f"3. **DSAC tlow05 (CUB100, center fb)**：CM={t05['cm']:.3f}, fallback={t05['dec'].get('fallback', 0)}, second_pass={t05['second_pass']}。"
               f"{'center 回退框在 CUB 上同样拖累' if t05['cm'] < dflt['cm'] - 0.02 else 'CUB 上 center 回退框不像跨数据集那样拖累'}（CUB 的图片以鸟类为主体，中心通常含目标）。")
    obs.append(f"4. **DSAC tlow06_oracle (CUB100)**：CM={tor['cm']:.3f}, 相对 SE baseline 提升 {tor['cm']-se['cm']:+.3f}。"
               f"{'oracle 上限在 CUB 仍能涨点' if tor['cm'] > se['cm'] + 0.01 else 'oracle 上限在 CUB 没有显著提升（SCS 区分度依然不足）'}。")

    for o in obs:
        lines.append(f"{o}\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Written: {OUT}")
    # also print to stdout
    print("".join(lines))


if __name__ == "__main__":
    main()
