#!/usr/bin/env python3
"""跨领域 DSAC 验证聚合（4 域 × 5 配置）。

输出：
- runs/20260523_cross_domain_dsac/metrics/cross_domain_table.md（5 数据集 × 5 配置 主表，含 CUB 行）
- runs/20260523_cross_domain_dsac/metrics/cross_domain_summary.json
- 同时为每域写 metrics/<domain>_summary.md（含决策档位分布）
"""
import json
from pathlib import Path
from collections import Counter

RUN = Path("experiments/thesis_experiments/runs/20260523_cross_domain_dsac")
RAW = RUN / "raw"
METRICS = RUN / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("se_baseline",         "SE 基线"),
    ("pred_bbox",           "pred_bbox 基线"),
    ("dsac_default",        "DSAC default (θ_low=0.3, center fb)"),
    ("dsac_tlow05_center",  "DSAC tlow05+center"),
    ("dsac_tlow06_oracle",  "DSAC tlow06+oracle (上限)"),
]

DOMAINS = [
    ("docvqa",          "DocVQA (文档文本)"),
    ("infographicsvqa", "InfographicsVQA (图表信息图)"),
    ("visual7w",        "Visual7W (通用 VQA)"),
    ("gqa_spatial",     "GQA-spatial (关系推理)"),
]

# CUB 来自 20260522 单域复验（已存在）
CUB_RUN = Path("experiments/thesis_experiments/runs/20260522_dsac_v7/raw_cub")
CUB_TAG_MAP = {
    "se_baseline":         "se_baseline",
    "pred_bbox":           None,                  # CUB 验证未做 pred_bbox 单独行
    "dsac_default":        "default",
    "dsac_tlow05_center":  "tlow05",
    "dsac_tlow06_oracle":  "tlow06_oracle",
}


def load_jsonl(path: Path):
    if not path.exists():
        return None
    return [json.loads(l) for l in path.open() if l.strip()]


def summarize(rows):
    if rows is None:
        return None
    cms = [r["contains_match"] for r in rows if r.get("contains_match") is not None]
    ems = [r["exact_match"] for r in rows if r.get("exact_match") is not None]
    lats = [r.get("latency_ms_total") for r in rows if r.get("latency_ms_total") is not None]
    dec = Counter()
    scs_vals, sin_vals = [], []
    second_pass = 0
    for r in rows:
        md = r.get("metadata", {})
        cg = md.get("composite_gate")
        if cg:
            dec[cg["decision"]] += 1
            scs_vals.append(cg.get("s_out"))
            sin_vals.append(cg.get("s_in"))
            if cg.get("second_pass"):
                second_pass += 1
    return {
        "n": len(rows),
        "n_paired": len(cms),
        "cm": sum(cms) / len(cms) if cms else None,
        "em": sum(ems) / len(ems) if ems else None,
        "latency_ms": sum(lats) / len(lats) if lats else None,
        "decision": dict(dec),
        "second_pass": second_pass,
        "scs_mean": (sum(v for v in scs_vals if v is not None) /
                     max(1, len([v for v in scs_vals if v is not None]))) if scs_vals else None,
        "sin_mean": (sum(v for v in sin_vals if v is not None) /
                     max(1, len([v for v in sin_vals if v is not None]))) if sin_vals else None,
    }


def fmt_cell(s):
    if s is None or s.get("cm") is None:
        return "—"
    return f"{s['cm']:.3f}/{s['em']:.3f}"


def build_per_domain_md(domain, desc, results):
    lines = [f"# {desc} DSAC 验证（n=100, 跨域 §4.7）\n\n"]
    lines.append("| 配置 | n | CM | EM | Latency | accept | re_extract | fallback | second_pass | SCS mean | s_in mean |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, label in CONFIGS:
        s = results.get(tag)
        if s is None:
            lines.append(f"| {label} | — | — | — | — | — | — | — | — | — | — |\n")
            continue
        d = s.get("decision", {})
        cm = f"{s['cm']:.3f}" if s.get("cm") is not None else "—"
        em = f"{s['em']:.3f}" if s.get("em") is not None else "—"
        lat = f"{s['latency_ms']:.0f}" if s.get("latency_ms") is not None else "—"
        scs = f"{s['scs_mean']:.3f}" if s.get("scs_mean") is not None else "—"
        sin = f"{s['sin_mean']:.3f}" if s.get("sin_mean") is not None else "—"
        lines.append(
            f"| {label} | {s['n']} | {cm} | {em} | {lat} ms | "
            f"{d.get('accept', 0)} | {d.get('re_extract', 0)} | {d.get('fallback', 0)} | "
            f"{s['second_pass']} | {scs} | {sin} |\n"
        )
    return "".join(lines)


def main():
    summary = {}  # domain -> tag -> stats

    # 4 个跨域
    for domain, _ in DOMAINS:
        per = {}
        for tag, _ in CONFIGS:
            per[tag] = summarize(load_jsonl(RAW / f"{domain}_{tag}.jsonl"))
        summary[domain] = per

    # CUB 行（复用 20260522_dsac_v7/raw_cub）
    cub = {}
    for tag, cub_tag in CUB_TAG_MAP.items():
        if cub_tag is None:
            cub[tag] = None
        else:
            cub[tag] = summarize(load_jsonl(CUB_RUN / f"dsac_{cub_tag}.jsonl"))
    summary["cub100"] = cub

    # 写 summary.json
    (METRICS / "cross_domain_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 每域 md
    for domain, desc in DOMAINS:
        md = build_per_domain_md(domain, desc, summary[domain])
        (METRICS / f"{domain}_summary.md").write_text(md, encoding="utf-8")
    # CUB 也补一份（来源是 20260522 run，但放本 metrics/ 便于 §4.7 引用）
    (METRICS / "cub100_summary.md").write_text(
        build_per_domain_md("cub100", "CUB100（细粒度鸟类）", summary["cub100"]),
        encoding="utf-8",
    )

    # 主表
    out = []
    out.append("# 跨领域 DSAC 验证主表（CM / EM，§4.7）\n\n")
    out.append("**数据**：每域 n=100，VisCoT-7b-224，bf16，T=0.2，top_p=0.95，seed=42。\n\n")
    out.append("| 数据集 \\\\ 配置 | " + " | ".join(label for _, label in CONFIGS) + " |\n")
    out.append("|---" + "|---:" * len(CONFIGS) + "|\n")
    all_domains = [(d, desc) for d, desc in DOMAINS] + [("cub100", "CUB100（细粒度鸟类，对照）")]
    for domain, desc in all_domains:
        cells = []
        for tag, _ in CONFIGS:
            cells.append(fmt_cell(summary.get(domain, {}).get(tag)))
        out.append(f"| {desc} | " + " | ".join(cells) + " |\n")

    # CM 单值表（更易阅读）
    out.append("\n## CM 单值（用于图 4.12）\n\n")
    out.append("| 数据集 \\\\ 配置 | " + " | ".join(label for _, label in CONFIGS) + " |\n")
    out.append("|---" + "|---:" * len(CONFIGS) + "|\n")
    for domain, desc in all_domains:
        cells = []
        for tag, _ in CONFIGS:
            s = summary.get(domain, {}).get(tag)
            if s is None or s.get("cm") is None:
                cells.append("—")
            else:
                cells.append(f"{s['cm']:.3f}")
        out.append(f"| {desc} | " + " | ".join(cells) + " |\n")

    # 决策档位分布
    out.append("\n## 决策档位分布（accept / re_extract / fallback）\n\n")
    out.append("| 数据集 \\\\ 配置 | " + " | ".join(label for _, label in CONFIGS if "dsac" in _.lower() or "DSAC" in label) + " |\n")
    out.append("|---" + "|---:" * 3 + "|\n")
    dsac_tags = [(t, l) for t, l in CONFIGS if t.startswith("dsac")]
    for domain, desc in all_domains:
        cells = []
        for tag, _ in dsac_tags:
            s = summary.get(domain, {}).get(tag)
            if s is None:
                cells.append("—")
            else:
                d = s.get("decision", {})
                cells.append(
                    f"a{d.get('accept', 0)}/r{d.get('re_extract', 0)}/f{d.get('fallback', 0)}"
                )
        out.append(f"| {desc} | " + " | ".join(cells) + " |\n")

    (METRICS / "cross_domain_table.md").write_text("".join(out), encoding="utf-8")
    print(f"Written: {METRICS / 'cross_domain_table.md'}")
    print(f"Written: {METRICS / 'cross_domain_summary.json'}")
    for domain, _ in DOMAINS:
        print(f"  per-domain: {METRICS / (domain + '_summary.md')}")


if __name__ == "__main__":
    main()
