"""Phase 2 sweep analysis: lowres_size and crop_pad ablations.

Produces:
  B1_lowres_sweep.{csv,png}     — accuracy / latency vs lowres_size
  B2_padding_sweep.{csv,png}    — accuracy / latency vs crop_pad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(rows: List[Dict]) -> Dict[str, float]:
    cm = [r.get("contains_match") for r in rows if r.get("contains_match") is not None]
    em = [r.get("exact_match") for r in rows if r.get("exact_match") is not None]
    lat = [r.get("latency_ms_total") for r in rows if r.get("latency_ms_total") is not None]
    pre = [r.get("latency_ms_preprocess") for r in rows if r.get("latency_ms_preprocess") is not None]
    gen = [r.get("latency_ms_generate") for r in rows if r.get("latency_ms_generate") is not None]
    mem = [r.get("peak_gpu_memory_mb") for r in rows if r.get("peak_gpu_memory_mb") is not None]
    return {
        "n": len(rows),
        "contains_match_mean": sum(cm) / len(cm) if cm else None,
        "exact_match_mean": sum(em) / len(em) if em else None,
        "latency_ms_mean": sum(lat) / len(lat) if lat else None,
        "preprocess_ms_mean": sum(pre) / len(pre) if pre else None,
        "generate_ms_mean": sum(gen) / len(gen) if gen else None,
        "peak_gpu_memory_mb_mean": sum(mem) / len(mem) if mem else None,
    }


def analyse_lowres(sweep_dir: Path, out_dir: Path):
    rows = []
    for path in sorted(sweep_dir.glob("answer_lowres*.jsonl")):
        data = load_jsonl(path)
        size = data[0].get("lowres_size") or int("".join(c for c in path.stem if c.isdigit()))
        s = summarise(data)
        s["lowres_size"] = size
        s["file"] = path.name
        rows.append(s)
    df = pd.DataFrame(rows).sort_values("lowres_size").reset_index(drop=True)
    df.to_csv(out_dir / "B1_lowres_sweep.csv", index=False)

    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df["lowres_size"].tolist()
    ax_acc.plot(x, df["contains_match_mean"], "o-", label="contains_match")
    ax_acc.plot(x, df["exact_match_mean"], "s--", label="exact_match")
    ax_acc.set_xlabel("lowres_size (px, low-res branch resolution)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("B1a: Accuracy vs lowres_size")
    ax_acc.set_ylim(0.2, 0.55)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=9)

    ax_lat.plot(x, df["latency_ms_mean"], "o-", color="#cc6633", label="total")
    ax_lat.plot(x, df["generate_ms_mean"], "s--", color="#3399cc", label="generate")
    ax_lat.plot(x, df["preprocess_ms_mean"], "d--", color="#669933", label="preprocess")
    ax_lat.set_xlabel("lowres_size (px)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("B1b: Latency vs lowres_size")
    ax_lat.grid(alpha=0.3)
    ax_lat.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "B1_lowres_sweep.png", dpi=180)
    plt.close(fig)
    print(f"B1 wrote {out_dir / 'B1_lowres_sweep.csv'}")
    return df


def analyse_padding(sweep_dir: Path, out_dir: Path):
    rows = []
    for path in sorted(sweep_dir.glob("answer_pad*.jsonl")):
        data = load_jsonl(path)
        pad = data[0].get("crop_pad")
        if pad is None:
            tag = path.stem.replace("answer_pad", "")
            pad = float(tag) / 10 if tag.isdigit() and len(tag) == 2 else float(tag)
        s = summarise(data)
        s["crop_pad"] = pad
        s["file"] = path.name
        rows.append(s)
    df = pd.DataFrame(rows).sort_values("crop_pad").reset_index(drop=True)
    df.to_csv(out_dir / "B2_padding_sweep.csv", index=False)

    fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df["crop_pad"].tolist()
    ax_acc.plot(x, df["contains_match_mean"], "o-", label="contains_match")
    ax_acc.plot(x, df["exact_match_mean"], "s--", label="exact_match")
    ax_acc.set_xlabel("crop_pad α (multiplier around bbox)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("B2a: Accuracy vs crop padding")
    ax_acc.set_ylim(0.2, 0.5)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=9)

    ax_lat.plot(x, df["latency_ms_mean"], "o-", color="#cc6633", label="total")
    ax_lat.plot(x, df["generate_ms_mean"], "s--", color="#3399cc", label="generate")
    ax_lat.set_xlabel("crop_pad α")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title("B2b: Latency vs crop padding")
    ax_lat.grid(alpha=0.3)
    ax_lat.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "B2_padding_sweep.png", dpi=180)
    plt.close(fig)
    print(f"B2 wrote {out_dir / 'B2_padding_sweep.csv'}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lowres-dir", required=True)
    p.add_argument("--padding-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analyse_lowres(Path(args.lowres_dir), out_dir)
    analyse_padding(Path(args.padding_dir), out_dir)


if __name__ == "__main__":
    main()
