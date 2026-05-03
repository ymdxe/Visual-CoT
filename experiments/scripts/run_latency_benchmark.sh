#!/usr/bin/env bash
set -euo pipefail

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
MODEL_PATH="./checkpoints/VisCoT-7b-224"; DATASET="cub"; MAX_SAMPLES=20; OUTPUT_ROOT="./experiments/results/latency"
usage(){ echo "Usage: bash experiments/scripts/run_latency_benchmark.sh [--model-path PATH] [--dataset cub] [--max-samples 20] [--output-root DIR]"; }
while [[ $# -gt 0 ]]; do case "$1" in --help|-h) usage; exit 0;; --model-path) MODEL_PATH="$2"; shift 2;; --dataset) DATASET="$2"; shift 2;; --max-samples) MAX_SAMPLES="$2"; shift 2;; --output-root) OUTPUT_ROOT="$2"; shift 2;; *) echo "unknown arg: $1"; usage; exit 2;; esac; done
mkdir -p "$OUTPUT_ROOT"
Q="./viscot_benchmark/benchmark/${DATASET}.json"; DET="./experiments/results/pilot/det/${DATASET}.jsonl"
if [[ ! -f "$DET" ]]; then python tools/make_oracle_detection_jsonl.py --question-file "$Q" --output-file "$OUTPUT_ROOT/${DATASET}_det.jsonl" --dataset-name "$DATASET" --max-samples "$MAX_SAMPLES"; DET="$OUTPUT_ROOT/${DATASET}_det.jsonl"; fi
for mode in full pred_bbox crop_only lowres_full_highrescrop; do python -m llava.eval.model_cot_loader --model-path "$MODEL_PATH" --question-file "$Q" --image-folder ./playground/data --answers-file "$OUTPUT_ROOT/${DATASET}_${mode}.jsonl" --detection-file "$DET" --dataset-name "$DATASET" --mode "$mode" --max-samples "$MAX_SAMPLES" --temperature 0 --conv-mode vicuna_v1 --record-latency --record-memory; done
python experiments/analysis/summarize_results.py --input-root "$OUTPUT_ROOT" --output-csv "$OUTPUT_ROOT/latency_summary.csv" --output-md "$OUTPUT_ROOT/latency_summary.md"
