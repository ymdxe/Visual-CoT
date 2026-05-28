#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:-VisCoT-7b-224}
QUESTION_FILE=${2:-./data/benchmarks/small_benchmark.json}
DETECTION_FILE=${3:-./results/region_compress/detection.jsonl}
IMAGE_FOLDER=${4:-./playground/data}

# MODEL_NAME is used as the output file stem. For experiment labels such as
# VisCoT-7b-224_predbbox_cub30, reuse the base checkpoint automatically.
MODEL_ID=${MODEL_NAME%%_*}
DEFAULT_MODEL_PATH="./checkpoints/${MODEL_ID}"
if [ -d "./checkpoints/llava-${MODEL_ID}" ] || [ -L "./checkpoints/llava-${MODEL_ID}" ]; then
    DEFAULT_MODEL_PATH="./checkpoints/llava-${MODEL_ID}"
fi
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH}}
PYTHON_BIN=${PYTHON_BIN:-python}
BBOX_EXPAND_RATIO=${BBOX_EXPAND_RATIO:-1.2}
VISUAL_INPUT_POLICY=${VISUAL_INPUT_POLICY:-auto}
SAVE_COMPRESSED_IMAGES=${SAVE_COMPRESSED_IMAGES:-false}
COMPRESSED_IMAGE_DIR=${COMPRESSED_IMAGE_DIR:-./results/region_compress/compressed_images}

if [ ! -e "${MODEL_PATH}" ]; then
    echo "[ERROR] model path not found: ${MODEL_PATH}"
    echo "Set MODEL_PATH explicitly if MODEL_NAME is only an experiment label."
    exit 1
fi

if [ ! -f "${QUESTION_FILE}" ]; then
    echo "[ERROR] question file not found: ${QUESTION_FILE}"
    echo "Build a small benchmark first, for example:"
    echo "python tools/build_small_benchmark.py --benchmark-dir ./viscot_benchmark/benchmark --image-folder ./playground/data --datasets gqa,textvqa,docvqa --num-per-dataset 10 --output ./data/benchmarks/small_benchmark.json"
    exit 1
fi

if [ ! -f "${DETECTION_FILE}" ]; then
    echo "[ERROR] detection file not found: ${DETECTION_FILE}"
    echo "Run detection on the same small benchmark first and pass the resulting jsonl as the third argument."
    exit 1
fi

if [ ! -d "${IMAGE_FOLDER}" ]; then
    echo "[ERROR] image folder not found: ${IMAGE_FOLDER}"
    exit 1
fi

if [ -n "${MODES:-}" ]; then
    read -r -a MODES_ARRAY <<< "${MODES}"
else
    MODES_ARRAY=(none crop_only blur downsample mask)
fi

for MODE in "${MODES_ARRAY[@]}"; do
    OUT_DIR="./results/region_compress/${MODE}"
    OUT_FILE="${OUT_DIR}/${MODEL_NAME}.jsonl"
    mkdir -p "${OUT_DIR}"

    CMD=(
        "${PYTHON_BIN}" llava/eval/model_cot_loader_region_compress.py
        --model-path "${MODEL_PATH}"
        --question-file "${QUESTION_FILE}"
        --image-folder "${IMAGE_FOLDER}"
        --answers-file "${OUT_FILE}"
        --temperature 0
        --conv-mode vicuna_v1
        --with-cot true
        --detection-file "${DETECTION_FILE}"
        --compress-mode "${MODE}"
        --visual-input-policy "${VISUAL_INPUT_POLICY}"
        --bbox-expand-ratio "${BBOX_EXPAND_RATIO}"
        --compressed-image-dir "${COMPRESSED_IMAGE_DIR}/${MODE}"
    )

    if [ "${SAVE_COMPRESSED_IMAGES}" = "true" ]; then
        CMD+=(--save-compressed-images)
    fi

    echo "[INFO] running mode=${MODE}, visual_input_policy=${VISUAL_INPUT_POLICY}, model_path=${MODEL_PATH}, output=${OUT_FILE}"
    "${CMD[@]}"
done

"${PYTHON_BIN}" tools/eval_region_compress_results.py     --inputs ./results/region_compress/*/"${MODEL_NAME}.jsonl"     --output-csv ./results/region_compress/summary.csv     --output-md ./results/region_compress/summary.md
