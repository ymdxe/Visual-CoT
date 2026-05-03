#!/usr/bin/env bash
set -euo pipefail

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
ANS=""; OUT="./experiments/figures/cases"; TAG=""; TOP=5; IMG_ROOT="./playground/data"
usage(){ echo "Usage: bash experiments/scripts/run_visualizations.sh --answers-jsonl FILE [--error-tag TAG] [--top-k 5] [--output-dir DIR]"; }
while [[ $# -gt 0 ]]; do case "$1" in --help|-h) usage; exit 0;; --answers-jsonl) ANS="$2"; shift 2;; --error-tag) TAG="$2"; shift 2;; --top-k) TOP="$2"; shift 2;; --output-dir) OUT="$2"; shift 2;; --image-root) IMG_ROOT="$2"; shift 2;; *) echo "unknown arg: $1"; usage; exit 2;; esac; done
[[ -n "$ANS" ]] || { usage; exit 2; }
CMD=(python experiments/analysis/visualize_cases.py --answers-jsonl "$ANS" --image-root "$IMG_ROOT" --output-dir "$OUT" --top-k "$TOP")
[[ -z "$TAG" ]] || CMD+=(--error-tag "$TAG")
"${CMD[@]}"
