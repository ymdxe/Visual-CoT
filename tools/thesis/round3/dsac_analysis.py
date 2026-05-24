#!/usr/bin/env python3
"""DSAC (§4.6) 结果聚合脚本：读 raw/dsac_*.jsonl，输出三张表 + 一个汇总 md。

输入：experiments/thesis_experiments/runs/20260522_dsac_v7/raw/dsac_{tag}.jsonl × 8
输出：metrics/dsac_{main_table, decision_breakdown, ablation}.{csv, md}
      paper_materials/section_4_6_results.md
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

CONFIGS = [
    ("default",        "α=0.5, θ=(0.3,0.7), fb=center"),
    ("alpha03",        "α=0.3, θ=(0.3,0.7), fb=center"),
    ("alpha07",        "α=0.7, θ=(0.3,0.7), fb=center"),
    ("strict",         "α=0.5, θ=(0.4,0.6), fb=center"),
    ("lax",            "α=0.5, θ=(0.2,0.8), fb=center"),
    ("oracle_fb",      "α=0.5, θ=(0.3,0.7), fb=oracle"),
    ("input_only",     "α=1.0 (仅输入端)"),
    ("output_only",    "α=0.0 (仅输出端)"),
    ("tlow05",         "α=0.5, θ_low=0.50, fb=center (fallback 触发)"),
    ("tlow055",        "α=0.5, θ_low=0.55, fb=center (fallback 增加)"),
    ("tlow06",         "α=0.5, θ_low=0.60, fb=center (fallback 多)"),
    ("tlow06_oracle",  "α=0.5, θ_low=0.60, fb=oracle (上限)"),
]


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean(xs, default=None):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


def safe_int(xs, default=0):
    return sum(1 for x in xs if x)


def aggregate(rows):
    n = len(rows)
    cm = [r.get("contains_match") for r in rows if r.get("contains_match") is not None]
    em = [r.get("exact_match") for r in rows if r.get("exact_match") is not None]
    lat = [r.get("latency_ms_total") for r in rows]
    gates = [r["metadata"].get("composite_gate") for r in rows if r.get("metadata")]
    gates = [g for g in gates if g]
    decisions = [g["decision"] for g in gates]
    second_pass = sum(1 for g in gates if g.get("second_pass"))
    composite = [g["composite"] for g in gates]
    s_in = [g["s_in"] for g in gates]
    s_out = [g["s_out"] for g in gates]
    return {
        "n": n,
        "n_gate": len(gates),
        "cm_mean": mean(cm),
        "em_mean": mean(em),
        "latency_mean_ms": mean(lat),
        "second_pass_rate": second_pass / len(gates) if gates else None,
        "accept_rate":     decisions.count("accept") / len(gates) if gates else None,
        "re_extract_rate": decisions.count("re_extract") / len(gates) if gates else None,
        "fallback_rate":   decisions.count("fallback") / len(gates) if gates else None,
        "composite_mean":  mean(composite),
        "s_in_mean":       mean(s_in),
        "s_out_mean":      mean(s_out),
        "n_accept":     decisions.count("accept"),
        "n_re_extract": decisions.count("re_extract"),
        "n_fallback":   decisions.count("fallback"),
    }


def decision_breakdown(rows, tag):
    """按 decision 分组的 CM 与延迟。"""
    buckets = defaultdict(list)
    for r in rows:
        g = (r.get("metadata") or {}).get("composite_gate")
        if not g:
            continue
        buckets[g["decision"]].append(r)
    out = []
    for d in ("accept", "re_extract", "fallback"):
        bucket = buckets.get(d, [])
        cm = [r.get("contains_match") for r in bucket if r.get("contains_match") is not None]
        lat = [r.get("latency_ms_total") for r in bucket if r.get("latency_ms_total") is not None]
        out.append({
            "config": tag,
            "decision": d,
            "n": len(bucket),
            "cm_mean": mean(cm),
            "latency_mean_ms": mean(lat),
        })
    return out


def write_md_table(path, headers, rows, fmt=None):
    fmt = fmt or {}
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            cells = []
            for h in headers:
                v = row.get(h)
                if v is None:
                    cells.append("—")
                elif isinstance(v, float):
                    cells.append(fmt.get(h, "{:.3f}").format(v))
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")


def write_csv(path, headers, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def baseline_cm_from_existing(source_run):
    """读取 20260516_cross_region100_public3 中各 baseline 的 CM 作为参照。"""
    base = Path(source_run) / "raw"
    out = {}
    for stem in ("pred_bbox", "structured_evidence", "woimg", "full", "crop_only",
                 "lowres_full_highrescrop", "random_bbox", "center_bbox"):
        fp = base / f"answer_{stem}.jsonl"
        if fp.exists():
            rows = load(fp)
            cms = [r.get("contains_match") for r in rows if r.get("contains_match") is not None]
            ems = [r.get("exact_match") for r in rows if r.get("exact_match") is not None]
            lats = [r.get("latency_ms_total") for r in rows]
            out[stem] = {
                "n": len(rows),
                "cm_mean": mean(cms),
                "em_mean": mean(ems),
                "latency_mean_ms": mean(lats),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="experiments/thesis_experiments/runs/20260522_dsac_v7")
    parser.add_argument("--source-run", default="experiments/thesis_experiments/runs/20260516_cross_region100_public3")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "raw"
    metrics_dir = run_dir / "metrics"
    paper_dir = run_dir / "paper_materials"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    # ===== 主表：8 个 DSAC 配置 × (CM, EM, latency, 二次生成率, 三档分布) =====
    main_rows = []
    bd_rows = []
    raw_cache = {}
    for tag, desc in CONFIGS:
        fp = raw_dir / f"dsac_{tag}.jsonl"
        if not fp.exists():
            print(f"[warn] {fp} missing, skip.")
            continue
        rows = load(fp)
        raw_cache[tag] = rows
        agg = aggregate(rows)
        agg["config"] = tag
        agg["desc"] = desc
        main_rows.append(agg)
        bd_rows.extend(decision_breakdown(rows, tag))

    # 加入跨数据集 100 中已经跑过的 baseline 做对照
    baselines = baseline_cm_from_existing(args.source_run)
    baseline_rows = []
    for stem, m in baselines.items():
        baseline_rows.append({
            "config": stem,
            "desc": "(baseline) " + stem,
            "n": m["n"],
            "cm_mean": m["cm_mean"],
            "em_mean": m["em_mean"],
            "latency_mean_ms": m["latency_mean_ms"],
        })

    # === 写主表 ===
    headers_main = ["config", "desc", "n", "cm_mean", "em_mean", "latency_mean_ms",
                    "second_pass_rate", "accept_rate", "re_extract_rate", "fallback_rate",
                    "composite_mean", "s_in_mean", "s_out_mean"]
    fmt_main = {
        "cm_mean": "{:.3f}", "em_mean": "{:.3f}",
        "latency_mean_ms": "{:.1f}",
        "second_pass_rate": "{:.2%}",
        "accept_rate": "{:.2%}", "re_extract_rate": "{:.2%}", "fallback_rate": "{:.2%}",
        "composite_mean": "{:.3f}", "s_in_mean": "{:.3f}", "s_out_mean": "{:.3f}",
    }
    write_csv(metrics_dir / "dsac_main_table.csv", headers_main, main_rows)
    write_md_table(metrics_dir / "dsac_main_table.md", headers_main, main_rows, fmt=fmt_main)

    # === 写档位剖析 ===
    headers_bd = ["config", "decision", "n", "cm_mean", "latency_mean_ms"]
    fmt_bd = {"cm_mean": "{:.3f}", "latency_mean_ms": "{:.1f}"}
    write_csv(metrics_dir / "dsac_decision_breakdown.csv", headers_bd, bd_rows)
    write_md_table(metrics_dir / "dsac_decision_breakdown.md", headers_bd, bd_rows, fmt=fmt_bd)

    # === 写消融表（α 扫描 + 阈值扫描） ===
    abl_tags = ["output_only", "alpha03", "default", "alpha07", "input_only",
                "strict", "lax", "oracle_fb"]
    abl_rows = [r for tag in abl_tags for r in main_rows if r["config"] == tag]
    headers_abl = ["config", "desc", "cm_mean", "em_mean", "latency_mean_ms",
                   "second_pass_rate", "accept_rate", "re_extract_rate", "fallback_rate"]
    write_csv(metrics_dir / "dsac_ablation.csv", headers_abl, abl_rows)
    write_md_table(metrics_dir / "dsac_ablation.md", headers_abl, abl_rows, fmt=fmt_main)

    # === 写基线对照表 ===
    headers_bl = ["config", "desc", "n", "cm_mean", "em_mean", "latency_mean_ms"]
    write_csv(metrics_dir / "baselines.csv", headers_bl, baseline_rows)
    write_md_table(metrics_dir / "baselines.md", headers_bl, baseline_rows, fmt={
        "cm_mean": "{:.3f}", "em_mean": "{:.3f}", "latency_mean_ms": "{:.1f}",
    })

    # === 写章节素材 ===
    def get(tag, key):
        for r in main_rows:
            if r["config"] == tag:
                return r.get(key)
        return None

    cm_default = get("default", "cm_mean")
    cm_strict = get("strict", "cm_mean")
    cm_oracle = get("oracle_fb", "cm_mean")
    cm_input = get("input_only", "cm_mean")
    cm_output = get("output_only", "cm_mean")
    cm_pred_baseline = (baselines.get("pred_bbox") or {}).get("cm_mean")
    cm_se_baseline = (baselines.get("structured_evidence") or {}).get("cm_mean")

    with open(paper_dir / "section_4_6_results.md", "w", encoding="utf-8") as f:
        f.write("# §4.6 双端自适应控制器 实验结果填充\n\n")
        f.write("> 数据集：跨数据集 100 样本（TextVQA + GQA + Visual7W），seed=42，bf16，VisCoT-7b-224。\n\n")
        f.write("## 主表（8 配置 × 关键指标）\n\n")
        with open(metrics_dir / "dsac_main_table.md", encoding="utf-8") as g:
            f.write(g.read())
        f.write("\n## 决策档位剖析\n\n")
        with open(metrics_dir / "dsac_decision_breakdown.md", encoding="utf-8") as g:
            f.write(g.read())
        f.write("\n## 与跨数据集 100 中已有 baseline 对照\n\n")
        with open(metrics_dir / "baselines.md", encoding="utf-8") as g:
            f.write(g.read())
        f.write("\n## 关键数字（用于章节正文）\n\n")
        f.write(f"- DSAC 默认配置 (α=0.5, θ=(0.3,0.7), center fallback) CM = **{cm_default}**\n")
        f.write(f"- DSAC strict (α=0.5, θ=(0.4,0.6)) CM = **{cm_strict}**\n")
        f.write(f"- DSAC oracle fallback (上限) CM = **{cm_oracle}**\n")
        f.write(f"- 消融 α=1.0（仅输入端，等价 §4.4.1）CM = **{cm_input}**\n")
        f.write(f"- 消融 α=0.0（仅输出端 SCS）CM = **{cm_output}**\n")
        if cm_pred_baseline is not None:
            f.write(f"- 基线 pred_bbox (跨数据集 100 已发布) CM = **{cm_pred_baseline}**\n")
        if cm_se_baseline is not None:
            f.write(f"- 基线 structured_evidence (跨数据集 100 已发布) CM = **{cm_se_baseline}**\n")

    # 打印简要总结到 stdout
    print("\n=== DSAC 主表 ===")
    for r in main_rows:
        print(f"  {r['config']:14s} n={r['n']:3d}  CM={r['cm_mean']:.3f}  EM={r['em_mean']:.3f}  "
              f"lat={r['latency_mean_ms']:.0f} ms  decisions: acc={r['accept_rate']:.1%} "
              f"re={r['re_extract_rate']:.1%} fb={r['fallback_rate']:.1%}  "
              f"second_pass={r['second_pass_rate']:.1%}")
    print("\n=== 跨数据集 100 基线 ===")
    for r in baseline_rows:
        print(f"  {r['config']:24s} n={r['n']:3d}  CM={r['cm_mean']:.3f}  EM={r['em_mean']:.3f}  "
              f"lat={r['latency_mean_ms']:.0f} ms")


if __name__ == "__main__":
    main()
