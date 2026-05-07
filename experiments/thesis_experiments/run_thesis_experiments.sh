#!/usr/bin/env bash
set -euo pipefail

MAX_SAMPLES=3
RUN_NAME=""
STAGE="smoke"
MODEL_PATH="./checkpoints/VisCoT-7b-224"
QUESTION_FILE="./small_benchmark_cub100.json"
DET_QUESTION_FILE="./viscot_benchmark/benchmark_det/cub.jsonl"
IMAGE_FOLDER="./playground/data"
CONV_MODE="vicuna_v1"
LOAD_4BIT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --question-file) QUESTION_FILE="$2"; shift 2 ;;
    --det-question-file) DET_QUESTION_FILE="$2"; shift 2 ;;
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --no-4bit) LOAD_4BIT=0; shift ;;
    --help|-h)
      echo "Usage: bash experiments/thesis_experiments/run_thesis_experiments.sh --stage smoke|cub20|cub100|advanced|reasoning --max-samples N"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(date +%Y%m%d_%H%M%S)_${STAGE}"
fi

RUN_DIR="experiments/thesis_experiments/runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"/{configs,commands,logs,raw,metrics,figures/case_visualizations,paper_materials,env}

cat > "$RUN_DIR/configs/cub100_ablation.yaml" <<YAML
max_samples: ${MAX_SAMPLES}
model_path: ${MODEL_PATH}
question_file: ${QUESTION_FILE}
detection_question_file: ${DET_QUESTION_FILE}
image_folder: ${IMAGE_FOLDER}
conv_mode: ${CONV_MODE}
load_4bit: ${LOAD_4BIT}
YAML
cat > "$RUN_DIR/configs/pruning_config.yaml" <<YAML
vision_prune_amounts: [0.1, 0.2]
vision_prune_target: vision_tower
feature_pca_dims: [512, 256]
YAML
cat > "$RUN_DIR/configs/reasoning_config.yaml" <<YAML
reasoning_compression: rule
max_reasoning_sentences: 2
max_evidence_sentences: 3
step_selector: heuristic
YAML

{
  echo "branch=$(git branch --show-current)"
  git rev-parse HEAD
  git status --short
} > "$RUN_DIR/env/git_info.txt"
(nvidia-smi || true) > "$RUN_DIR/env/nvidia_smi.txt"
(python -VV; uname -a) > "$RUN_DIR/env/system_info.txt" 2>&1
(pip freeze || conda env export || true) > "$RUN_DIR/env/pip_freeze.txt"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --question-file "$QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --temperature 0
  --conv-mode "$CONV_MODE"
  --max-samples "$MAX_SAMPLES"
  --dataset-name cub
  --record-latency
  --record-memory
)
if [[ "$LOAD_4BIT" == "1" ]]; then
  COMMON_ARGS+=(--load-4bit)
fi

run_detection() {
  python -m llava.eval.model_cot_det_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$DET_QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$RUN_DIR/raw/detection.jsonl" \
    --temperature 0 \
    --conv-mode "$CONV_MODE" \
    --max-samples "$MAX_SAMPLES" \
    --save-jsonl \
    $([[ "$LOAD_4BIT" == "1" ]] && echo "--load-4bit") \
    > "$RUN_DIR/logs/detection.log" 2>&1
}

run_answer() {
  local mode="$1"; shift
  python -m llava.eval.model_cot_loader \
    "${COMMON_ARGS[@]}" \
    --detection-file "$RUN_DIR/raw/detection.jsonl" \
    --mode "$mode" \
    --answers-file "$RUN_DIR/raw/answer_${mode}.jsonl" \
    "$@" \
    > "$RUN_DIR/logs/answer_${mode}.log" 2>&1
}

{
  printf 'RUN_DIR=%s\n' "$RUN_DIR"
  printf 'MAX_SAMPLES=%s\n' "$MAX_SAMPLES"
  printf 'STAGE=%s\n' "$STAGE"
  declare -f run_detection
  declare -f run_answer
} > "$RUN_DIR/commands/run_commands.sh"
touch "$RUN_DIR/commands/rerun_notes.md"

run_detection

case "$STAGE" in
  smoke)
    MODES=(full pred_bbox crop_only structured_evidence)
    ;;
  cub20|cub100)
    MODES=(full pred_bbox crop_only lowres_full_highrescrop structured_evidence center_bbox random_bbox woimg oracle_bbox)
    ;;
  reasoning)
    MODES=(structured_evidence answer_reasoning_compressed direct_then_explain)
    ;;
  advanced)
    MODES=(vision_pruned feature_pca)
    ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    exit 2
    ;;
esac

for mode in "${MODES[@]}"; do
  case "$mode" in
    answer_reasoning_compressed)
      run_answer "$mode" --reasoning-compression rule --max-reasoning-sentences 2 --max-evidence-sentences 3
      ;;
    direct_then_explain)
      run_answer "$mode" --answer-policy direct_then_explain --explanation-max-new-tokens 64
      ;;
    vision_pruned)
      python -m llava.eval.model_cot_loader "${COMMON_ARGS[@]}" --detection-file "$RUN_DIR/raw/detection.jsonl" --mode pred_bbox --vision-prune --vision-prune-type linear_ln_structured --vision-prune-amount 0.1 --vision-prune-target vision_tower --vision-prune-dry-run > "$RUN_DIR/logs/answer_vision_pruned.log" 2>&1 || true
      ;;
    feature_pca)
      run_answer "$mode" --feature-compression pca --pca-dim 256 || true
      ;;
    *)
      run_answer "$mode"
      ;;
  esac
done

python tools/thesis/aggregate_results.py --raw-dir "$RUN_DIR/raw" --metrics-dir "$RUN_DIR/metrics" > "$RUN_DIR/logs/aggregate.log" 2>&1
python tools/thesis/visualize_results.py --metrics-dir "$RUN_DIR/metrics" --figures-dir "$RUN_DIR/figures" > "$RUN_DIR/logs/visualize.log" 2>&1
python tools/thesis/generate_paper_materials.py --metrics-dir "$RUN_DIR/metrics" --figures-dir "$RUN_DIR/figures" --paper-dir "$RUN_DIR/paper_materials" > "$RUN_DIR/logs/paper_materials.log" 2>&1

echo "$RUN_DIR"
