#!/usr/bin/env bash
set -euo pipefail

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
RUN=0; MAX_SAMPLES=100
while [[ $# -gt 0 ]]; do case "$1" in --run) RUN=1; shift;; --max-samples) MAX_SAMPLES="$2"; shift 2;; --help|-h) echo "Usage: bash experiments/scripts/run_formal_eval.sh [--run] [--max-samples 100]"; exit 0;; *) echo "unknown arg: $1"; exit 2;; esac; done
cat <<CMD
Recommended formal command:
  bash experiments/scripts/run_diagnostics.sh --datasets cub --max-samples $MAX_SAMPLES --output-root ./experiments/results/formal
CMD
if [[ "$RUN" == "1" ]]; then bash experiments/scripts/run_diagnostics.sh --datasets cub --max-samples "$MAX_SAMPLES" --output-root ./experiments/results/formal; else echo "dry run only. Add --run to execute."; fi
