"""Aggregate all Phase 1/2/3 outputs into a single round-2 markdown report.

Usage:
    python tools/thesis/round2/aggregate_round2.py \\
        --base experiments/thesis_experiments/runs/20260522_round2_extensions \\
        --baseline-run experiments/thesis_experiments/runs/20260516_cross_region100_public3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_csv_if(path: Path) -> str:
    if not path.exists():
        return "_(missing)_"
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.empty:
            return "_(empty)_"
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception as exc:
        return f"_(failed to read: {exc})_"


def summarise_dir(dir_path: Path) -> List[Dict]:
    rows = []
    for p in sorted(dir_path.glob("answer_*.jsonl")):
        data = [json.loads(l) for l in p.open() if l.strip()]
        cm = sum(r.get("contains_match") or 0 for r in data) / len(data)
        em = sum(r.get("exact_match") or 0 for r in data) / len(data)
        lat = [r.get("latency_ms_total") for r in data if r.get("latency_ms_total") is not None]
        rows.append({"file": p.name, "n": len(data),
                     "cm": cm, "em": em,
                     "lat_ms": (sum(lat) / len(lat)) if lat else None})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    base = Path(args.base)
    baseline = Path(args.baseline_run)
    out_path = Path(args.out) if args.out else (base / "ROUND2_REPORT.md")

    metrics = base / "metrics"
    analysis_xr = base / "analysis" / "cross_region100"
    analysis_cub = base / "analysis" / "cub100"

    parts = []
    parts.append("# Round-2 Extension Report\n\n")
    parts.append(f"Base run: `{base}`\n\n")
    parts.append(f"Baseline reference: `{baseline}`\n\n")
    parts.append("All experiments below were produced on a single RTX 3090 24 GB with VisCoT-7b-224 in bf16. ")
    parts.append("Sample count is 100 cross-dataset (TextVQA + GQA + Visual7W) unless noted.\n\n")

    parts.append("---\n\n## Phase 1: Zero-GPU analyses on existing data\n\n")
    parts.append("### A1 — Accuracy stratified by predicted-bbox IoU (cross-domain)\n\n")
    parts.append(read_csv_if(analysis_xr / "A1_iou_bins.csv"))
    parts.append("\n\nFigure: `analysis/cross_region100/A1_iou_bins.png`.\n\n")

    parts.append("### A2 — Latency decomposition (cross-domain)\n\n")
    parts.append(read_csv_if(analysis_xr / "A2_latency_breakdown.csv"))
    parts.append("\n\nFigure: `analysis/cross_region100/A2_latency_breakdown.png`. ")
    parts.append("Generate stage dominates; preprocess stays under 12 ms across all modes.\n\n")

    parts.append("### A3 — Question-type stratification (cross-domain)\n\n")
    a3_md = (analysis_xr / "A3_question_types.md")
    parts.append(a3_md.read_text(encoding="utf-8") if a3_md.exists() else "_(missing)_")
    parts.append("\n\n")

    parts.append("### A4 — Pareto frontier on (latency, accuracy) and (tokens, accuracy)\n\n")
    parts.append(read_csv_if(analysis_xr / "A4_pareto.csv"))
    parts.append("\n\nFigure: `analysis/cross_region100/A4_pareto.png`.\n\n")

    parts.append("### A5 — Structure Conformance Score for `structured_evidence`\n\n")
    a5_md = (analysis_xr / "A5_scs.md")
    parts.append(a5_md.read_text(encoding="utf-8") if a5_md.exists() else "_(missing)_")
    parts.append("\n\n")

    parts.append("### A6 — McNemar paired significance test\n\n")
    a6_md = (analysis_xr / "A6_mcnemar.md")
    parts.append(a6_md.read_text(encoding="utf-8") if a6_md.exists() else "_(missing)_")
    parts.append("\n\n")

    parts.append("---\n\n## Phase 2: Sensitivity sweeps\n\n")
    parts.append("### B1 — `lowres_size` sweep on `lowres_full_highres_crop`\n\n")
    parts.append(read_csv_if(metrics / "B1_lowres_sweep.csv"))
    parts.append("\n\nFigure: `metrics/B1_lowres_sweep.png`.\n\n")

    parts.append("### B2 — `crop_pad` α sweep on `pred_bbox`\n\n")
    parts.append(read_csv_if(metrics / "B2_padding_sweep.csv"))
    parts.append("\n\nFigure: `metrics/B2_padding_sweep.png`.\n\n")

    parts.append("---\n\n## Phase 3: Stability + new algorithm\n\n")
    parts.append("### C1 — Multi-seed stability (T=0.2, top_p=0.95)\n\n")
    c1_md = (metrics / "C1_seed_stability.md")
    parts.append(c1_md.read_text(encoding="utf-8") if c1_md.exists() else "_(pending)_")
    parts.append("\n\n")

    parts.append("### C2 — BBox-scoring ablation (Algorithm 5)\n\n")
    c2_md = (metrics / "C2_bbox_scoring.md")
    parts.append(c2_md.read_text(encoding="utf-8") if c2_md.exists() else "_(pending)_")
    parts.append("\n\n")

    parts.append("---\n\n## Raw run summaries\n\n")
    for sub in ("lowres_sweep", "padding_sweep", "seed_stability", "bbox_scoring"):
        d = base / "raw" / sub
        if not d.exists():
            continue
        parts.append(f"### `{sub}`\n\n")
        rows = summarise_dir(d)
        if not rows:
            parts.append("_(no files yet)_\n\n")
            continue
        parts.append("| file | n | contains_match | exact_match | latency_ms |\n")
        parts.append("|---|---:|---:|---:|---:|\n")
        for r in rows:
            lat = "" if r["lat_ms"] is None else f"{r['lat_ms']:.1f}"
            parts.append(f"| {r['file']} | {r['n']} | {r['cm']:.4f} | {r['em']:.4f} | {lat} |\n")
        parts.append("\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
