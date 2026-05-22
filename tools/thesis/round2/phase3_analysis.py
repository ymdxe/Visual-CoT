"""Phase 3 analysis:
  C1 — multi-seed stability mean ± std for core modes.
  C2 — bbox-scoring algorithm ablations vs no-scoring baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def metrics_of(rows: List[Dict]) -> Dict[str, float]:
    cm = [r.get("contains_match") or 0 for r in rows]
    em = [r.get("exact_match") or 0 for r in rows]
    lat = [r.get("latency_ms_total") for r in rows if r.get("latency_ms_total") is not None]
    return {
        "n": len(rows),
        "contains_match": sum(cm) / len(cm) if cm else 0.0,
        "exact_match": sum(em) / len(em) if em else 0.0,
        "latency_ms": sum(lat) / len(lat) if lat else None,
    }


# ---------------- C1 ----------------

SEED_RE = re.compile(r"answer_(?P<mode>[a-z_]+)_seed(?P<seed>\d+)\.jsonl")


def c1_stability(seed_dir: Path, out_dir: Path) -> pd.DataFrame:
    records: Dict[str, Dict[str, List[float]]] = {}
    for path in sorted(seed_dir.glob("answer_*.jsonl")):
        m = SEED_RE.match(path.name)
        if not m:
            continue
        mode = m.group("mode")
        seed = int(m.group("seed"))
        met = metrics_of(load_jsonl(path))
        records.setdefault(mode, {"seed": [], "cm": [], "em": [], "lat": []})
        records[mode]["seed"].append(seed)
        records[mode]["cm"].append(met["contains_match"])
        records[mode]["em"].append(met["exact_match"])
        if met["latency_ms"] is not None:
            records[mode]["lat"].append(met["latency_ms"])

    rows = []
    for mode, dat in records.items():
        rows.append({
            "mode": mode,
            "n_seeds": len(dat["seed"]),
            "cm_mean": statistics.mean(dat["cm"]) if dat["cm"] else None,
            "cm_std": statistics.stdev(dat["cm"]) if len(dat["cm"]) > 1 else 0.0,
            "em_mean": statistics.mean(dat["em"]) if dat["em"] else None,
            "em_std": statistics.stdev(dat["em"]) if len(dat["em"]) > 1 else 0.0,
            "lat_mean": statistics.mean(dat["lat"]) if dat["lat"] else None,
            "lat_std": statistics.stdev(dat["lat"]) if len(dat["lat"]) > 1 else 0.0,
            "seeds_used": ",".join(str(s) for s in sorted(dat["seed"])),
        })
    df = pd.DataFrame(rows).sort_values("mode")
    df.to_csv(out_dir / "C1_seed_stability.csv", index=False)
    with (out_dir / "C1_seed_stability.md").open("w", encoding="utf-8") as f:
        f.write("# C1: Multi-seed stability (T=0.2, top_p=0.95)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    if not df.empty:
        modes = df["mode"].tolist()
        means = df["cm_mean"].tolist()
        stds = df["cm_std"].tolist()
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(modes))
        ax.bar(x, means, yerr=stds, capsize=4, color="#3366aa")
        ax.set_xticks(list(x))
        ax.set_xticklabels(modes, rotation=20, ha="right")
        ax.set_ylabel("Contains Match (mean ± std over seeds)")
        ax.set_title("C1: Multi-seed stability")
        ax.set_ylim(0, max(means + [0.5]) * 1.2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "C1_seed_stability.png", dpi=180)
        plt.close(fig)
    return df


# ---------------- C2 ----------------

def c2_bbox_scoring(scoring_dir: Path, baseline_jsonl: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    # baseline (no scoring) for direct comparison
    if baseline_jsonl.exists():
        base = load_jsonl(baseline_jsonl)
        met = metrics_of(base)
        rows.append({
            "label": "baseline_no_scoring",
            "threshold": None, "lambdas": None, "fallback": None,
            "n": met["n"],
            "contains_match": met["contains_match"],
            "exact_match": met["exact_match"],
            "latency_ms": met["latency_ms"],
            "fallback_rate": 0.0,
            "iou_at_05_after_gate": None,
            "iou_at_05_only_kept": None,
        })

    for path in sorted(scoring_dir.glob("answer_*.jsonl")):
        data = load_jsonl(path)
        met = metrics_of(data)
        # decode score config from JSONL
        first = data[0]
        meta_score = (first.get("metadata") or {}).get("bbox_score") or {}
        thr = meta_score.get("threshold")
        lambdas = meta_score.get("lambdas")
        fallback_kind = meta_score.get("fallback_kind")
        triggered = [(((r.get("metadata") or {}).get("bbox_score") or {}).get("triggered")) for r in data]
        fr = sum(1 for t in triggered if t) / len(triggered) if triggered else 0.0
        # IoU on kept (non-fallback) samples
        kept_iou = [r.get("bbox_iou") for r, t in zip(data, triggered)
                    if (not t) and r.get("bbox_iou") is not None]
        all_iou_after = [r.get("bbox_iou") for r in data if r.get("bbox_iou") is not None]
        iou_kept = sum(1 for v in kept_iou if v >= 0.5) / len(kept_iou) if kept_iou else None
        iou_after = sum(1 for v in all_iou_after if v >= 0.5) / len(all_iou_after) if all_iou_after else None
        rows.append({
            "label": path.stem.replace("answer_", ""),
            "threshold": thr,
            "lambdas": ",".join(f"{v:g}" for v in lambdas) if lambdas else None,
            "fallback": fallback_kind,
            "n": met["n"],
            "contains_match": met["contains_match"],
            "exact_match": met["exact_match"],
            "latency_ms": met["latency_ms"],
            "fallback_rate": fr,
            "iou_at_05_after_gate": iou_after,
            "iou_at_05_only_kept": iou_kept,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "C2_bbox_scoring.csv", index=False)
    with (out_dir / "C2_bbox_scoring.md").open("w", encoding="utf-8") as f:
        f.write("# C2: bbox-scoring algorithm ablation (vs pred_bbox baseline)\n\n")
        f.write("`fallback_rate` = fraction of samples whose predicted bbox failed the gate and was replaced.\n")
        f.write("`iou_at_05_only_kept` = IoU>=0.5 accuracy *only* on samples kept after the gate (i.e. how clean is the survivor pool).\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    if len(df) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        labels = df["label"].tolist()
        x = range(len(labels))
        ax.bar(x, df["contains_match"].tolist(), color="#5fa9f8")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("Contains Match")
        ax.set_title("C2: bbox-scoring contains_match across configurations")
        ax.set_ylim(0, max(df["contains_match"].tolist()) * 1.25 if df["contains_match"].any() else 0.6)
        ax.grid(axis="y", alpha=0.3)
        # annotate fallback rate
        for xi, fr, cm in zip(x, df["fallback_rate"].tolist(), df["contains_match"].tolist()):
            if fr is None:
                continue
            ax.text(xi, cm + 0.01, f"fr={fr:.0%}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "C2_bbox_scoring.png", dpi=180)
        plt.close(fig)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dir", required=True)
    p.add_argument("--scoring-dir", required=True)
    p.add_argument("--baseline-jsonl", required=True,
                   help="Path to the no-scoring pred_bbox JSONL from the cross_region100_public3 run.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c1 = c1_stability(Path(args.seed_dir), out_dir)
    print(f"C1 produced {len(c1)} rows")
    c2 = c2_bbox_scoring(Path(args.scoring_dir), Path(args.baseline_jsonl), out_dir)
    print(f"C2 produced {len(c2)} rows")


if __name__ == "__main__":
    main()
