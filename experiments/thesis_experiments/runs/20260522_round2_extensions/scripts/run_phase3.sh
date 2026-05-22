#!/usr/bin/env bash
# Phase 3:
#   C1 — multi-seed stability for 3 core modes (sampling at T=0.2 so seeds actually move)
#   C2 — bbox-scoring algorithm ablations (deterministic at T=0)
# Inputs reuse cross_region100_public3 question + detection files.

set -euo pipefail

REPO=/root/code/Visual-CoT
RUN=$REPO/experiments/thesis_experiments/runs/20260516_cross_region100_public3
OUT=$REPO/experiments/thesis_experiments/runs/20260522_round2_extensions
MODEL=$REPO/checkpoints/VisCoT-7b-224

QUESTION=$RUN/configs/small_benchmark_cross_region100.json
DET=$RUN/raw/detection.jsonl
IMAGES=$REPO/playground/data
CONV=vicuna_v1

mkdir -p "$OUT/raw/seed_stability" "$OUT/raw/bbox_scoring" "$OUT/logs"

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
    --record-latency --record-memory \
    "$@" 2>&1 | tail -2
}

# C1: 3 modes × 3 seeds at temperature 0.2 (so seeds actually change the sample)
for MODE in pred_bbox structured_evidence lowres_full_highrescrop; do
  for SEED in 42 1234 2026; do
    run_one "C1_${MODE}_seed${SEED}" \
      "$OUT/raw/seed_stability/answer_${MODE}_seed${SEED}.jsonl" \
      --mode "$MODE" --seed "$SEED" --temperature 0.2 --top_p 0.95
  done
done

# C2: bbox-scoring ablations (deterministic, T=0)
COMMON=( --mode pred_bbox --temperature 0 --bbox-scoring )

# default
run_one "C2_default" \
  "$OUT/raw/bbox_scoring/answer_default_t05_l1_1_0.5.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.5 --bbox-score-lambdas "1.0,1.0,0.5"

# strict (more fallback)
run_one "C2_strict" \
  "$OUT/raw/bbox_scoring/answer_strict_t07_l1_1_0.5.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.7 --bbox-score-lambdas "1.0,1.0,0.5"

# lax (less fallback)
run_one "C2_lax" \
  "$OUT/raw/bbox_scoring/answer_lax_t03_l1_1_0.5.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.3 --bbox-score-lambdas "1.0,1.0,0.5"

# no focus term
run_one "C2_no_focus" \
  "$OUT/raw/bbox_scoring/answer_no_focus_l1_1_0.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.5 --bbox-score-lambdas "1.0,1.0,0.0"

# no size term
run_one "C2_no_size" \
  "$OUT/raw/bbox_scoring/answer_no_size_l1_0_0.5.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.5 --bbox-score-lambdas "1.0,0.0,0.5"

# oracle fallback (upper-bound proxy on the gated fraction)
run_one "C2_oracle_fallback" \
  "$OUT/raw/bbox_scoring/answer_oracle_fallback.jsonl" \
  "${COMMON[@]}" --bbox-score-threshold 0.5 --bbox-score-lambdas "1.0,1.0,0.5" \
  --bbox-score-fallback oracle

echo "[done] phase 3 complete"
