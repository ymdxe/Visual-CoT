#!/usr/bin/env bash
# Phase 2: small GPU sweeps over lowres_size and crop_pad.
# Reuses cross_region100_public3 question + detection inputs.

set -euo pipefail

REPO=/root/code/Visual-CoT
RUN=$REPO/experiments/thesis_experiments/runs/20260516_cross_region100_public3
OUT=$REPO/experiments/thesis_experiments/runs/20260522_round2_extensions
MODEL=$REPO/checkpoints/VisCoT-7b-224

QUESTION=$RUN/configs/small_benchmark_cross_region100.json
DET=$RUN/raw/detection.jsonl
IMAGES=$REPO/playground/data
CONV=vicuna_v1

mkdir -p "$OUT/raw/lowres_sweep" "$OUT/raw/padding_sweep" "$OUT/logs"

if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi

cd "$REPO"

run_one () {
  local label="$1" outfile="$2"; shift 2
  if [[ -s "$outfile" ]]; then
    echo "[skip] $label already has results at $outfile"
    return 0
  fi
  echo "[run] $label -> $outfile"
  python3 -m llava.eval.model_cot_loader \
    --model-path "$MODEL" \
    --question-file "$QUESTION" \
    --detection-file "$DET" \
    --image-folder "$IMAGES" \
    --answers-file "$outfile" \
    --conv-mode "$CONV" \
    --temperature 0 \
    --record-latency --record-memory \
    "$@" 2>&1 | tail -3
}

# B1: lowres_size sweep
for S in 64 96 112 144 196; do
  run_one "lowres_$S" "$OUT/raw/lowres_sweep/answer_lowres${S}.jsonl" \
    --mode lowres_full_highrescrop --lowres-size "$S"
done

# B2: crop_pad sweep at pred_bbox
for P in 1.0 1.2 1.5 2.0; do
  TAG=${P//./}
  run_one "pad_${P}" "$OUT/raw/padding_sweep/answer_pad${TAG}.jsonl" \
    --mode pred_bbox --crop-pad "$P"
done

echo "[done] phase 2 sweeps complete"
