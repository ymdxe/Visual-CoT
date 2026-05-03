#!/usr/bin/env bash
set -euo pipefail

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
MODEL_PATH="./checkpoints/VisCoT-7b-224"; DATASETS="cub"; MAX_SAMPLES=50; OUTPUT_ROOT="./experiments/results/diagnostics"; LOAD_ARG=""; IMAGE_ROOT="./playground/data"; QUESTION_ROOT="./viscot_benchmark/benchmark"; MODES="full pred_bbox random_bbox center_bbox woimg crop_only lowres_full_highrescrop structured_evidence"
usage(){ echo "Usage: bash experiments/scripts/run_diagnostics.sh [--model-path PATH] [--datasets cub] [--max-samples 50] [--output-root DIR] [--load-4bit|--load-8bit]"; }
while [[ $# -gt 0 ]]; do case "$1" in --help|-h) usage; exit 0;; --model-path) MODEL_PATH="$2"; shift 2;; --datasets) DATASETS="$2"; shift 2;; --max-samples) MAX_SAMPLES="$2"; shift 2;; --output-root) OUTPUT_ROOT="$2"; shift 2;; --load-4bit) LOAD_ARG="--load-4bit"; shift;; --load-8bit) LOAD_ARG="--load-8bit"; shift;; *) echo "unknown arg: $1"; usage; exit 2;; esac; done
mkdir -p "$OUTPUT_ROOT/answers" "$OUTPUT_ROOT/det" ./experiments/tables
IFS=',' read -ra DS <<< "$DATASETS"
for ds in "${DS[@]}"; do
  q_file="$QUESTION_ROOT/${ds}.json"; [[ -f "$q_file" ]] || { echo "skip $ds: missing $q_file"; continue; }
  det_file="./experiments/results/pilot/det/${ds}.jsonl"
  if [[ ! -f "$det_file" ]]; then python tools/make_oracle_detection_jsonl.py --question-file "$q_file" --output-file "$OUTPUT_ROOT/det/${ds}.jsonl" --dataset-name "$ds" --max-samples "$MAX_SAMPLES"; det_file="$OUTPUT_ROOT/det/${ds}.jsonl"; fi
  for mode in $MODES; do
    python -m llava.eval.model_cot_loader --model-path "$MODEL_PATH" --question-file "$q_file" --image-folder "$IMAGE_ROOT" --answers-file "$OUTPUT_ROOT/answers/${ds}_${mode}.jsonl" --detection-file "$det_file" --dataset-name "$ds" --mode "$mode" --max-samples "$MAX_SAMPLES" --temperature 0 --conv-mode vicuna_v1 --record-latency --record-memory $LOAD_ARG
  done
done
python experiments/analysis/summarize_results.py --input-root "$OUTPUT_ROOT/answers" --output-csv ./experiments/tables/diagnostics_summary.csv --output-md ./experiments/tables/diagnostics_summary.md
