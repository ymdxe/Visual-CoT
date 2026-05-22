"""Phase 1 zero-GPU analyses on existing JSONL runs.

Produces:
  A1 iou_bins             — bbox IoU vs answer accuracy bins
  A2 latency_breakdown    — preprocess vs generate latency per mode
  A3 question_types       — accuracy stratified by heuristic question type
  A4 pareto               — accuracy vs latency / token Pareto frontier
  A5 scs                  — structured evidence conformance score
  A6 mcnemar              — paired significance test between mode pairs

Usage:
    python tools/thesis/round2/phase1_analysis.py --analysis all \\
        --run-dir experiments/thesis_experiments/runs/20260516_cross_region100_public3 \\
        --out-dir experiments/thesis_experiments/runs/20260522_round2_extensions/analysis
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


# ---------- IO ----------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def mode_from_filename(p: Path) -> str:
    return p.stem.replace("answer_", "")


def discover_answer_files(run_dir: Path) -> Dict[str, Path]:
    raw = run_dir / "raw"
    out = {}
    for p in sorted(raw.glob("answer_*.jsonl")):
        out[mode_from_filename(p)] = p
    return out


# ---------- A1: IoU vs accuracy bins ----------

IOU_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def a1_iou_bins(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    records = []
    for mode in ("pred_bbox", "structured_evidence", "lowres_full_highrescrop", "crop_only"):
        if mode not in rows_by_mode:
            continue
        for low, high in IOU_BINS:
            bucket = [
                r for r in rows_by_mode[mode]
                if r.get("bbox_iou") is not None and low <= r["bbox_iou"] < high
            ]
            if not bucket:
                records.append({"mode": mode, "iou_low": low, "iou_high": high, "n": 0,
                                "contains_match": None, "exact_match": None})
                continue
            cm = [r.get("contains_match") for r in bucket if r.get("contains_match") is not None]
            em = [r.get("exact_match") for r in bucket if r.get("exact_match") is not None]
            records.append({
                "mode": mode,
                "iou_low": low,
                "iou_high": high,
                "n": len(bucket),
                "contains_match": (sum(cm) / len(cm)) if cm else None,
                "exact_match": (sum(em) / len(em)) if em else None,
            })
    df = pd.DataFrame(records)
    csv_path = out_dir / "A1_iou_bins.csv"
    df.to_csv(csv_path, index=False)

    # Plot: grouped bars per IoU bin
    fig, ax = plt.subplots(figsize=(8, 4.5))
    modes = [m for m in df["mode"].unique()]
    width = 0.8 / max(1, len(modes))
    bin_labels = [f"[{lo:.1f},{hi:.1f})" for lo, hi in IOU_BINS]
    x = np.arange(len(IOU_BINS))
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode].sort_values(["iou_low"]).reset_index(drop=True)
        vals = [(v if v is not None else 0.0) for v in sub["contains_match"]]
        ax.bar(x + i * width, vals, width=width, label=mode)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Contains Match")
    ax.set_xlabel("Predicted bbox IoU vs ground truth")
    ax.set_title("A1: Answer accuracy stratified by bbox IoU")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "A1_iou_bins.png", dpi=180)
    plt.close(fig)
    return df


# ---------- A2: Latency breakdown ----------

def a2_latency_breakdown(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for mode, items in rows_by_mode.items():
        pre = [r.get("latency_ms_preprocess") for r in items if r.get("latency_ms_preprocess") is not None]
        gen = [r.get("latency_ms_generate") for r in items if r.get("latency_ms_generate") is not None]
        tot = [r.get("latency_ms_total") for r in items if r.get("latency_ms_total") is not None]
        rows.append({
            "mode": mode,
            "n": len(items),
            "preprocess_mean_ms": (sum(pre) / len(pre)) if pre else None,
            "generate_mean_ms": (sum(gen) / len(gen)) if gen else None,
            "total_mean_ms": (sum(tot) / len(tot)) if tot else None,
        })
    df = pd.DataFrame(rows).sort_values("total_mean_ms")
    df.to_csv(out_dir / "A2_latency_breakdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    modes = df["mode"].tolist()
    pre = df["preprocess_mean_ms"].fillna(0).tolist()
    gen = df["generate_mean_ms"].fillna(0).tolist()
    x = np.arange(len(modes))
    ax.bar(x, pre, label="preprocess", color="#5fa9f8")
    ax.bar(x, gen, bottom=pre, label="generate", color="#ff9966")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("A2: Per-stage latency decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "A2_latency_breakdown.png", dpi=180)
    plt.close(fig)
    return df


# ---------- A3: Question type stratification ----------

OCR_RX = re.compile(
    r"\b(written|writes|reads?|says?|word|words|text|letter|letters|number|"
    r"sign|caption|title|label|date|time|price|amount|brand|name|website|"
    r"license|address|phone|page|chapter)\b",
    re.I,
)
COUNT_RX = re.compile(r"\b(how many|count|number of)\b", re.I)
COLOR_RX = re.compile(r"\b(color|colour|shape|made of|material)\b", re.I)
SPATIAL_RX = re.compile(
    r"^(where|which side)\b|"
    r"\bnear the|\bbehind the|\bin front of|\bsitting on|\bon top of|\bunder the|"
    r"\bnext to|\babove the|\bbelow the\b",
    re.I,
)
IDENTITY_RX = re.compile(
    r"^(who|whose)\b|"
    r"\b(what kind|what type|what animal|what species|what breed)\b",
    re.I,
)
YESNO_RX = re.compile(r"^(is|are|does|do|can|has|have|was|were)\b", re.I)


def classify_question(prompt: str) -> str:
    if not prompt:
        return "other"
    text = prompt.lower()
    text = re.sub(r"predicted region bbox.*", "", text, flags=re.I | re.S)
    text = re.sub(r"please answer.*", "", text, flags=re.I | re.S)
    text = text.replace("<image>", " ").strip()
    if COUNT_RX.search(text):
        return "counting"
    if COLOR_RX.search(text):
        return "color_shape"
    if OCR_RX.search(text):
        return "ocr_text"
    if SPATIAL_RX.search(text):
        return "spatial"
    if IDENTITY_RX.search(text):
        return "identity"
    if YESNO_RX.match(text):
        return "yes_no"
    return "other"


def a3_question_types(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    cells: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    type_counts: Counter = Counter()
    sample_type = {}
    for mode, items in rows_by_mode.items():
        for r in items:
            qid = r.get("question_id")
            qtype = sample_type.get(qid)
            if qtype is None:
                qtype = classify_question(r.get("prompt") or r.get("prompt_text") or "")
                sample_type[qid] = qtype
                type_counts[qtype] += 1
            cm = r.get("contains_match")
            if cm is None:
                continue
            cells[(mode, qtype)].append(cm)
    modes = sorted(rows_by_mode.keys())
    qtypes = sorted(type_counts.keys())
    table = []
    for mode in modes:
        row = {"mode": mode}
        for qtype in qtypes:
            vals = cells[(mode, qtype)]
            row[f"{qtype}_n"] = len(vals)
            row[f"{qtype}_acc"] = (sum(vals) / len(vals)) if vals else None
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(out_dir / "A3_question_types.csv", index=False)
    with (out_dir / "A3_question_types.md").open("w", encoding="utf-8") as f:
        f.write("# A3: Question-type stratified contains_match\n\n")
        f.write("Question type counts (per sample, dataset-level): " + ", ".join(f"{k}={v}" for k, v in type_counts.most_common()) + "\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    return df


# ---------- A4: Pareto frontier ----------

def a4_pareto(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for mode, items in rows_by_mode.items():
        cm = [r.get("contains_match") for r in items if r.get("contains_match") is not None]
        em = [r.get("exact_match") for r in items if r.get("exact_match") is not None]
        lat = [r.get("latency_ms_total") for r in items if r.get("latency_ms_total") is not None]
        tok = [r.get("num_visual_tokens_est") for r in items if r.get("num_visual_tokens_est") is not None]
        mem = [r.get("peak_gpu_memory_mb") for r in items if r.get("peak_gpu_memory_mb") is not None]
        if not cm or not lat:
            continue
        rows.append({
            "mode": mode,
            "contains_match": sum(cm) / len(cm),
            "exact_match": (sum(em) / len(em)) if em else None,
            "latency_ms": sum(lat) / len(lat),
            "tokens": (sum(tok) / len(tok)) if tok else 0,
            "memory_mb": (sum(mem) / len(mem)) if mem else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "A4_pareto.csv", index=False)

    def _scatter(ax, x, y, labels, xlabel, ylabel):
        ax.scatter(x, y, s=70, c="#2266aa")
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
        # Pareto: maximise y, minimise x
        pairs = sorted(zip(x, y, labels), key=lambda p: p[0])
        frontier_x, frontier_y = [], []
        best = -math.inf
        for xi, yi, _ in pairs:
            if yi > best:
                best = yi
                frontier_x.append(xi)
                frontier_y.append(yi)
        ax.plot(frontier_x, frontier_y, "--", color="#d24d4d", label="Pareto frontier")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = df["mode"].tolist()
    _scatter(axes[0], df["latency_ms"].tolist(), df["contains_match"].tolist(),
             labels, "Latency (ms)", "Contains Match")
    axes[0].set_title("A4a: accuracy vs latency")
    _scatter(axes[1], df["tokens"].tolist(), df["contains_match"].tolist(),
             labels, "Visual tokens (est.)", "Contains Match")
    axes[1].set_title("A4b: accuracy vs visual tokens")
    fig.tight_layout()
    fig.savefig(out_dir / "A4_pareto.png", dpi=180)
    plt.close(fig)
    return df


# ---------- A5: Structure conformance score ----------

def a5_scs(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    target_modes = [m for m in rows_by_mode if m in ("structured_evidence",)]
    rows = []
    for mode in target_modes:
        items = rows_by_mode[mode]
        per_dataset: Dict[str, List[float]] = defaultdict(list)
        per_dataset_full: Dict[str, List[int]] = defaultdict(list)
        for r in items:
            ds = r.get("dataset") or "ALL"
            parts = [r.get("region_text"), r.get("visual_evidence_text"),
                     r.get("reasoning_text"), r.get("answer_extracted")]
            present = sum(1 for p in parts if p)
            score = present / 4.0
            per_dataset[ds].append(score)
            per_dataset_full[ds].append(1 if present == 4 else 0)
        all_scores = [s for v in per_dataset.values() for s in v]
        all_full = [s for v in per_dataset_full.values() for s in v]
        rows.append({
            "mode": mode,
            "dataset": "ALL",
            "n": len(all_scores),
            "scs_mean": sum(all_scores) / len(all_scores) if all_scores else None,
            "scs_full4_rate": sum(all_full) / len(all_full) if all_full else None,
        })
        for ds, scores in sorted(per_dataset.items()):
            full = per_dataset_full[ds]
            rows.append({
                "mode": mode,
                "dataset": ds,
                "n": len(scores),
                "scs_mean": sum(scores) / len(scores),
                "scs_full4_rate": sum(full) / len(full),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "A5_scs.csv", index=False)
    with (out_dir / "A5_scs.md").open("w", encoding="utf-8") as f:
        f.write("# A5: Structure Conformance Score\n\n")
        f.write("SCS_mean = average fraction of the four sections present; full4_rate = proportion with all four sections.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    return df


# ---------- A6: McNemar paired test ----------

def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """Return (statistic, p-value). Continuity-corrected when b+c>=25."""
    n = b + c
    if n == 0:
        return float("nan"), 1.0
    if n < 25:
        # exact binomial two-sided
        from math import comb
        k = min(b, c)
        p = 0.0
        for i in range(0, k + 1):
            p += comb(n, i) * (0.5 ** n)
        return float("nan"), min(1.0, 2 * p)
    stat = (abs(b - c) - 1) ** 2 / n
    return stat, 1.0 - chi2.cdf(stat, df=1)


PAIRS = [
    ("full", "pred_bbox"),
    ("pred_bbox", "structured_evidence"),
    ("pred_bbox", "lowres_full_highrescrop"),
    ("pred_bbox", "crop_only"),
    ("full", "structured_evidence"),
    ("random_bbox", "pred_bbox"),
    ("center_bbox", "pred_bbox"),
    ("woimg", "full"),
]


def a6_mcnemar(rows_by_mode: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> pd.DataFrame:
    by_mode_qid: Dict[str, Dict[Any, int]] = {}
    for mode, items in rows_by_mode.items():
        by_mode_qid[mode] = {r["question_id"]: int(r.get("contains_match") or 0)
                             for r in items if "question_id" in r}
    out = []
    for a, b in PAIRS:
        if a not in by_mode_qid or b not in by_mode_qid:
            continue
        qids = set(by_mode_qid[a]) & set(by_mode_qid[b])
        a_only = b_only = both = neither = 0
        for q in qids:
            ya = by_mode_qid[a][q]
            yb = by_mode_qid[b][q]
            if ya == 0 and yb == 1:
                b_only += 1  # b wins
            elif ya == 1 and yb == 0:
                a_only += 1  # a wins
            elif ya == 1 and yb == 1:
                both += 1
            else:
                neither += 1
        stat, p = mcnemar(a_only, b_only)
        out.append({
            "mode_a": a, "mode_b": b, "n_paired": len(qids),
            "a_wins": a_only, "b_wins": b_only, "both_right": both, "both_wrong": neither,
            "mcnemar_stat": stat, "p_value": p,
            "delta_b_minus_a": (b_only - a_only) / len(qids) if qids else None,
        })
    df = pd.DataFrame(out)
    df.to_csv(out_dir / "A6_mcnemar.csv", index=False)
    with (out_dir / "A6_mcnemar.md").open("w", encoding="utf-8") as f:
        f.write("# A6: McNemar paired significance test on contains_match\n\n")
        f.write("`b_wins` = samples where mode_b is correct but mode_a is wrong (i.e. b's gain over a).\n")
        f.write("`p_value` < 0.05 indicates the asymmetry is unlikely under H0 (mode_a == mode_b).\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    return df


# ---------- Driver ----------

def run_all(run_dir: Path, out_dir: Path, label: str) -> None:
    out_dir = out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_answer_files(run_dir)
    rows_by_mode = {m: load_jsonl(p) for m, p in files.items()}
    rows_by_mode = {m: v for m, v in rows_by_mode.items() if v}
    print(f"[{label}] modes loaded: {list(rows_by_mode)}  | sizes: " +
          ", ".join(f"{m}={len(v)}" for m, v in rows_by_mode.items()))
    a1 = a1_iou_bins(rows_by_mode, out_dir)
    a2 = a2_latency_breakdown(rows_by_mode, out_dir)
    a3 = a3_question_types(rows_by_mode, out_dir)
    a4 = a4_pareto(rows_by_mode, out_dir)
    a5 = a5_scs(rows_by_mode, out_dir)
    a6 = a6_mcnemar(rows_by_mode, out_dir)
    print(f"  A1 bins rows={len(a1)}; A2 modes={len(a2)}; A3 q-types cols={[c for c in a3.columns if c.endswith('_acc')]}")
    print(f"  A4 modes={len(a4)}; A5 rows={len(a5)}; A6 pairs={len(a6)}")
    print(f"  wrote outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True,
                        help="Path to a thesis run directory containing raw/answer_*.jsonl. "
                             "Repeat for multiple sources.")
    parser.add_argument("--label", action="append", default=None,
                        help="Short label per --run-dir, used as subfolder name in --out-dir.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    labels = args.label or [p.name for p in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit("--label count must match --run-dir count")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rd, lbl in zip(run_dirs, labels):
        run_all(rd, out_dir, lbl)


if __name__ == "__main__":
    main()
