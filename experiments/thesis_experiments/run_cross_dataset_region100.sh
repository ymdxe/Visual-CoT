#!/usr/bin/env bash
set -euo pipefail

TARGET_SAMPLES=100
RUN_NAME=""
MODEL_PATH="./checkpoints/VisCoT-7b-224"
IMAGE_FOLDER="./playground/data"
BENCHMARK_DIR="./viscot_benchmark/benchmark"
DATASETS="textvqa,docvqa,sroie"
FALLBACK_DATASETS="visual7w,gqa,infographicsvqa"
CONV_MODE="vicuna_v1"
LOAD_4BIT=0
RUN_RC=1
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-samples|--max-samples) TARGET_SAMPLES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --fallback-datasets) FALLBACK_DATASETS="$2"; shift 2 ;;
    --conv-mode) CONV_MODE="$2"; shift 2 ;;
    --load-4bit) LOAD_4BIT=1; shift ;;
    --no-rc) RUN_RC=0; shift ;;
    --smoke) SMOKE=1; TARGET_SAMPLES=5; shift ;;
    --help|-h)
      echo "Usage: bash experiments/thesis_experiments/run_cross_dataset_region100.sh [--smoke] [--target-samples 100]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
PIP_BIN="${PIP_BIN:-$(command -v pip || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python/python3 not found; please activate the Visual-CoT environment first." >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  if [[ "$SMOKE" == "1" ]]; then
    RUN_NAME="$(date +%Y%m%d_%H%M%S)_cross_region_smoke"
  else
    RUN_NAME="$(date +%Y%m%d_%H%M%S)_cross_region100"
  fi
fi

RUN_DIR="experiments/thesis_experiments/runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"/{configs,commands,logs,raw,metrics,figures/case_visualizations,paper_materials,env}

QUESTION_FILE="$RUN_DIR/configs/small_benchmark_cross_region100.json"
DET_QUESTION_FILE="$RUN_DIR/configs/small_benchmark_cross_region100_det.jsonl"
BUILD_STATS_JSON="$RUN_DIR/configs/small_benchmark_cross_region100_stats.json"
BUILD_STATS_MD="$RUN_DIR/configs/small_benchmark_cross_region100_stats.md"

cat > "$RUN_DIR/configs/cross_region100.yaml" <<YAML
target_samples: ${TARGET_SAMPLES}
model_path: ${MODEL_PATH}
benchmark_dir: ${BENCHMARK_DIR}
image_folder: ${IMAGE_FOLDER}
datasets: ${DATASETS}
fallback_datasets: ${FALLBACK_DATASETS}
conv_mode: ${CONV_MODE}
load_4bit: ${LOAD_4BIT}
run_region_compress: ${RUN_RC}
YAML

{
  echo "branch=$(git branch --show-current)"
  git rev-parse HEAD
  git status --short
} > "$RUN_DIR/env/git_info.txt"
(nvidia-smi || true) > "$RUN_DIR/env/nvidia_smi.txt"
("$PYTHON_BIN" -VV; uname -a) > "$RUN_DIR/env/system_info.txt" 2>&1
if [[ -n "$PIP_BIN" ]]; then
  "$PIP_BIN" freeze > "$RUN_DIR/env/pip_freeze.txt" 2>&1 || true
else
  (conda env export || true) > "$RUN_DIR/env/pip_freeze.txt" 2>&1
fi

"$PYTHON_BIN" tools/build_cross_region_benchmark.py \
  --benchmark-dir "$BENCHMARK_DIR" \
  --image-folder "$IMAGE_FOLDER" \
  --datasets "$DATASETS" \
  --fallback-datasets "$FALLBACK_DATASETS" \
  --target-samples "$TARGET_SAMPLES" \
  --output "$QUESTION_FILE" \
  --det-output "$DET_QUESTION_FILE" \
  --stats-json "$BUILD_STATS_JSON" \
  --stats-md "$BUILD_STATS_MD" \
  > "$RUN_DIR/logs/build_benchmark.log" 2>&1

SELECTED_SAMPLES="$("$PYTHON_BIN" - "$BUILD_STATS_JSON" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    print(int(stats.get("selected_samples") or 0))
except Exception:
    print(0)
PY
)"

if [[ "$SELECTED_SAMPLES" -le 0 ]]; then
  {
    echo "# Cross-Dataset Run Stopped"
    echo
    echo "No usable cross-dataset samples were selected from the requested benchmark files."
    echo "This usually means the JSON annotations exist but the referenced images are missing under IMAGE_FOLDER=${IMAGE_FOLDER}."
    echo "Detection and answer stages were skipped to avoid producing empty or misleading results."
    echo
    echo "See: ${BUILD_STATS_MD}"
  } > "$RUN_DIR/paper_materials/cross_dataset_missing_data.md"
  "$PYTHON_BIN" tools/thesis/generate_cross_dataset_materials.py \
    --metrics-dir "$RUN_DIR/metrics" \
    --paper-dir "$RUN_DIR/paper_materials" \
    --stats-json "$BUILD_STATS_JSON" \
    > "$RUN_DIR/logs/cross_paper_materials.log" 2>&1 || true
  echo "$RUN_DIR"
  exit 0
fi

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --temperature 0
  --conv-mode "$CONV_MODE"
  --max-samples "$TARGET_SAMPLES"
  --record-latency
  --record-memory
)
if [[ "$LOAD_4BIT" == "1" ]]; then
  COMMON_ARGS+=(--load-4bit)
fi

run_detection() {
  "$PYTHON_BIN" -m llava.eval.model_cot_det_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$DET_QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$RUN_DIR/raw/detection.jsonl" \
    --temperature 0 \
    --conv-mode "$CONV_MODE" \
    --max-samples "$TARGET_SAMPLES" \
    --save-jsonl \
    $([[ "$LOAD_4BIT" == "1" ]] && echo "--load-4bit") \
    > "$RUN_DIR/logs/detection.log" 2>&1
}

run_answer() {
  local mode="$1"; shift
  "$PYTHON_BIN" -m llava.eval.model_cot_loader \
    "${COMMON_ARGS[@]}" \
    --detection-file "$RUN_DIR/raw/detection.jsonl" \
    --mode "$mode" \
    --answers-file "$RUN_DIR/raw/answer_${mode}.jsonl" \
    "$@" \
    > "$RUN_DIR/logs/answer_${mode}.log" 2>&1
}

{
  printf 'RUN_DIR=%s\n' "$RUN_DIR"
  printf 'TARGET_SAMPLES=%s\n' "$TARGET_SAMPLES"
  printf 'DATASETS=%s\n' "$DATASETS"
  printf 'FALLBACK_DATASETS=%s\n' "$FALLBACK_DATASETS"
  declare -f run_detection
  declare -f run_answer
} > "$RUN_DIR/commands/run_commands.sh"

run_detection

MODES=(full pred_bbox crop_only lowres_full_highrescrop structured_evidence woimg center_bbox random_bbox)
for mode in "${MODES[@]}"; do
  run_answer "$mode"
done

"$PYTHON_BIN" tools/thesis/aggregate_results.py --raw-dir "$RUN_DIR/raw" --metrics-dir "$RUN_DIR/metrics" > "$RUN_DIR/logs/aggregate.log" 2>&1
"$PYTHON_BIN" tools/thesis/aggregate_cross_dataset_results.py --raw-dir "$RUN_DIR/raw" --metrics-dir "$RUN_DIR/metrics" > "$RUN_DIR/logs/aggregate_cross_dataset.log" 2>&1
"$PYTHON_BIN" tools/thesis/visualize_results.py --metrics-dir "$RUN_DIR/metrics" --figures-dir "$RUN_DIR/figures" > "$RUN_DIR/logs/visualize.log" 2>&1
"$PYTHON_BIN" tools/thesis/generate_paper_materials.py --metrics-dir "$RUN_DIR/metrics" --figures-dir "$RUN_DIR/figures" --paper-dir "$RUN_DIR/paper_materials" > "$RUN_DIR/logs/paper_materials.log" 2>&1
"$PYTHON_BIN" tools/thesis/generate_cross_dataset_materials.py --metrics-dir "$RUN_DIR/metrics" --paper-dir "$RUN_DIR/paper_materials" --stats-json "$BUILD_STATS_JSON" > "$RUN_DIR/logs/cross_paper_materials.log" 2>&1

if [[ "$RUN_RC" == "1" ]]; then
  RC_DIR="results/rc/cross_region100"
  mkdir -p "$RC_DIR"
  MODEL="$MODEL_PATH" MODES="mark blur mask zoom mix" OUT_DIR="$RC_DIR" IMG_DIR="results/rc/img/cross_region100" \
    bash scripts/rc/run.sh cross_region100 "$QUESTION_FILE" "$RUN_DIR/raw/detection.jsonl" "$IMAGE_FOLDER" \
    > "$RUN_DIR/logs/region_compress_rc.log" 2>&1 || {
      echo "[WARN] region compression supplement failed; see $RUN_DIR/logs/region_compress_rc.log" | tee -a "$RUN_DIR/commands/rerun_notes.md"
    }
fi

echo "$RUN_DIR"
