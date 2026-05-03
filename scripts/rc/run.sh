#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-cub100}
Q=${2:-./small_benchmark_cub100.json}
DET=${3:-./results/region_compress/detection_pred_cub100.jsonl}
IMG_ROOT=${4:-./playground/data}

MODEL=${MODEL:-./checkpoints/llava-VisCoT-7b-224}
PY=${PY:-python}
POLICY=${POLICY:-auto}
PAD=${PAD:-1.2}
BLUR=${BLUR:-8}
SCALE=${SCALE:-0.25}
SAVE_IMG=${SAVE_IMG:-false}
IMG_DIR=${IMG_DIR:-./results/rc/img/${TAG}}
OUT_DIR=${OUT_DIR:-./results/rc/${TAG}}
MODES=${MODES:-base crop blur low mask mark zoom mix}

if [ ! -e "${MODEL}" ]; then
  echo "[ERR] missing model: ${MODEL}"
  exit 1
fi
if [ ! -f "${Q}" ]; then
  echo "[ERR] missing q: ${Q}"
  exit 1
fi
if [ ! -f "${DET}" ]; then
  echo "[ERR] missing det: ${DET}"
  exit 1
fi
if [ ! -d "${IMG_ROOT}" ]; then
  echo "[ERR] missing img-root: ${IMG_ROOT}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
read -r -a MODE_LIST <<< "${MODES}"

for M in "${MODE_LIST[@]}"; do
  P="${POLICY}"
  if [ "${M}" = "base" ] && [ "${POLICY}" = "auto" ]; then
    P="auto"
  fi
  OUT="${OUT_DIR}/${M}_${P}.jsonl"
  CMD=(
    "${PY}" llava/eval/cot_rc.py
    --model "${MODEL}"
    --q "${Q}"
    --det "${DET}"
    --img-root "${IMG_ROOT}"
    --out "${OUT}"
    --temperature 0
    --conv-mode vicuna_v1
    --with-cot true
    --mode "${M}"
    --policy "${P}"
    --pad "${PAD}"
    --blur "${BLUR}"
    --scale "${SCALE}"
    --img-dir "${IMG_DIR}/${M}"
  )
  if [ "${SAVE_IMG}" = "true" ]; then
    CMD+=(--save-img)
  fi
  echo "[RUN] mode=${M} policy=${P} out=${OUT}"
  "${CMD[@]}"
done

"${PY}" tools/rc_sum.py --in "${OUT_DIR}"/*.jsonl --csv "${OUT_DIR}/sum.csv" --md "${OUT_DIR}/sum.md"
