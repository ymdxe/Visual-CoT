#!/usr/bin/env bash
set -euo pipefail

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
MODEL_PATH="./checkpoints/VisCoT-7b-224"
DATASETS="cub"
MAX_SAMPLES=20
OUTPUT_ROOT="./experiments/results/pilot"
LOAD_ARG=""
IMAGE_ROOT="./playground/data"
QUESTION_ROOT="./viscot_benchmark/benchmark"
DET_ROOT="./viscot_benchmark/benchmark_det"
usage(){ cat <<USAGE
Usage: bash experiments/scripts/run_pilot.sh [options]
  --model-path PATH
  --datasets LIST        default: cub
  --max-samples N        default: 20
  --load-4bit | --load-8bit
  --output-root DIR
USAGE
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --max-samples) MAX_SAMPLES="$2"; shift 2;;
    --load-4bit) LOAD_ARG="--load-4bit"; shift;;
    --load-8bit) LOAD_ARG="--load-8bit"; shift;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    *) echo "unknown arg: $1"; usage; exit 2;;
  esac
done
[[ -d "$MODEL_PATH" ]] || { echo "missing model: $MODEL_PATH"; exit 0; }
mkdir -p "$OUTPUT_ROOT/det" "$OUTPUT_ROOT/answers"
IFS=',' read -ra DS <<< "$DATASETS"
for ds in "${DS[@]}"; do
  q_file="$QUESTION_ROOT/${ds}.json"
  det_in="$DET_ROOT/${ds}.jsonl"
  [[ -f "$q_file" ]] || { echo "skip $ds: missing $q_file"; continue; }
  if [[ -f "$det_in" ]]; then
    python -m llava.eval.model_cot_det_loader --model-path "$MODEL_PATH" --question-file "$det_in" --image-folder "$IMAGE_ROOT" --answers-file "$OUTPUT_ROOT/det/${ds}.jsonl" --dataset-name "$ds" --max-samples "$MAX_SAMPLES" --temperature 0 --conv-mode vicuna_v1 --record-latency --record-memory $LOAD_ARG
    det_file="$OUTPUT_ROOT/det/${ds}.jsonl"
  else
    python tools/make_oracle_detection_jsonl.py --question-file "$q_file" --output-file "$OUTPUT_ROOT/det/${ds}.jsonl" --dataset-name "$ds" --max-samples "$MAX_SAMPLES"
    det_file="$OUTPUT_ROOT/det/${ds}.jsonl"
  fi
  for mode in full pred_bbox random_bbox center_bbox woimg; do
    python -m llava.eval.model_cot_loader --model-path "$MODEL_PATH" --question-file "$q_file" --image-folder "$IMAGE_ROOT" --answers-file "$OUTPUT_ROOT/answers/${ds}_${mode}.jsonl" --detection-file "$det_file" --dataset-name "$ds" --mode "$mode" --max-samples "$MAX_SAMPLES" --temperature 0 --conv-mode vicuna_v1 --record-latency --record-memory $LOAD_ARG
  done
done
python experiments/analysis/summarize_results.py --input-root "$OUTPUT_ROOT/answers" --output-csv "$OUTPUT_ROOT/summary.csv" --output-md "$OUTPUT_ROOT/summary.md"
