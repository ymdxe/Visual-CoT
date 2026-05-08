#!/usr/bin/env bash
set -euo pipefail
cd /root/code/Visual-CoT
if [[ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]]; then
  source /usr/local/miniconda3/etc/profile.d/conda.sh
  conda activate viscot >/dev/null 2>&1 || true
fi
ADV=experiments/thesis_experiments/runs/20260508_034500_advanced20
COMMON=(--model-path ./checkpoints/VisCoT-7b-224 --question-file ./small_benchmark_cub100.json --image-folder ./playground/data --temperature 0 --conv-mode vicuna_v1 --max-samples 20 --dataset-name cub --record-latency --record-memory --detection-file "$ADV/raw/detection.jsonl" --mode auto --bbox-source pred --crop-mode full_crop)
{
  echo "[$(date -Is)] advanced supplement start"
  python -m llava.eval.model_cot_loader "${COMMON[@]}" --vision-prune --vision-prune-type linear_ln_structured --vision-prune-amount 0.1 --vision-prune-target vision_tower --answers-file "$ADV/raw/answer_vision_pruned_01.jsonl" > "$ADV/logs/answer_vision_pruned_01.log" 2>&1 || echo "vision_pruned_01 failed" >> "$ADV/logs/answer_vision_pruned_01.log"
  python -m llava.eval.model_cot_loader "${COMMON[@]}" --vision-prune --vision-prune-type linear_ln_structured --vision-prune-amount 0.2 --vision-prune-target vision_tower --answers-file "$ADV/raw/answer_vision_pruned_02.jsonl" > "$ADV/logs/answer_vision_pruned_02.log" 2>&1 || echo "vision_pruned_02 failed" >> "$ADV/logs/answer_vision_pruned_02.log"
  python -m llava.eval.model_cot_loader "${COMMON[@]}" --feature-compression pca --pca-dim 512 --answers-file "$ADV/raw/answer_feature_pca_512.jsonl" > "$ADV/logs/answer_feature_pca_512.log" 2>&1 || echo "feature_pca_512 failed" >> "$ADV/logs/answer_feature_pca_512.log"
  if [[ -f "$ADV/raw/answer_feature_pca.jsonl" && ! -f "$ADV/raw/answer_feature_pca_256.jsonl" ]]; then
    cp "$ADV/raw/answer_feature_pca.jsonl" "$ADV/raw/answer_feature_pca_256.jsonl"
    cp "$ADV/logs/answer_feature_pca.log" "$ADV/logs/answer_feature_pca_256.log" || true
  fi
  python tools/thesis/aggregate_results.py --raw-dir "$ADV/raw" --metrics-dir "$ADV/metrics" > "$ADV/logs/aggregate.log" 2>&1
  python tools/thesis/visualize_results.py --metrics-dir "$ADV/metrics" --figures-dir "$ADV/figures" > "$ADV/logs/visualize.log" 2>&1
  python tools/thesis/generate_paper_materials.py --metrics-dir "$ADV/metrics" --figures-dir "$ADV/figures" --paper-dir "$ADV/paper_materials" > "$ADV/logs/paper_materials.log" 2>&1
  echo "[$(date -Is)] reasoning stage start"
  bash experiments/thesis_experiments/run_thesis_experiments.sh --stage reasoning --max-samples 100 --run-name 20260508_045500_reasoning100
  REAS=experiments/thesis_experiments/runs/20260508_045500_reasoning100
  RCOMMON=(--model-path ./checkpoints/VisCoT-7b-224 --question-file ./small_benchmark_cub100.json --image-folder ./playground/data --temperature 0 --conv-mode vicuna_v1 --max-samples 100 --dataset-name cub --record-latency --record-memory --detection-file "$REAS/raw/detection.jsonl" --mode auto --bbox-source pred --crop-mode full_crop --evidence-mode structured)
  python -m llava.eval.model_cot_loader "${RCOMMON[@]}" --step-selector heuristic --step-selector-topk 2 --answers-file "$REAS/raw/answer_heuristic_step_selector.jsonl" > "$REAS/logs/answer_heuristic_step_selector.log" 2>&1 || echo "heuristic step selector failed" >> "$REAS/logs/answer_heuristic_step_selector.log"
  python tools/thesis/aggregate_results.py --raw-dir "$REAS/raw" --metrics-dir "$REAS/metrics" > "$REAS/logs/aggregate.log" 2>&1
  python tools/thesis/visualize_results.py --metrics-dir "$REAS/metrics" --figures-dir "$REAS/figures" > "$REAS/logs/visualize.log" 2>&1
  python tools/thesis/generate_paper_materials.py --metrics-dir "$REAS/metrics" --figures-dir "$REAS/figures" --paper-dir "$REAS/paper_materials" > "$REAS/logs/paper_materials.log" 2>&1
  echo "[$(date -Is)] continuation done"
} >> experiments/thesis_experiments/runs/20260508_034500_advanced20/logs/continue_advanced_reasoning.log 2>&1

