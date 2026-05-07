# Visual-CoT Thesis Experiment Plan

## Current Repository Structure

- Repository root: `/root/code/Visual-CoT`
- Current branch: `thesis/visual-pruning-reasoning-exp`
- Detection benchmark input: `viscot_benchmark/benchmark_det/cub.jsonl`
- Answer benchmark input: `viscot_benchmark/benchmark/cub.json`
- Small benchmark files already present: `small_benchmark.json`, `small_benchmark_cub30.json`, `small_benchmark_cub100.json`
- CUB image data is available under `downloads/cub/CUB_200_2011/images`
- Checkpoint is available under `checkpoints/VisCoT-7b-224`
- Existing earlier experiment outputs live under `experiments/results/`, `experiments/tables/`, `experiments/figures/`, and `results/region_compress/`. They are useful references but are not treated as this run's new results.

## Detection Entry

- Primary detection entry: `llava/eval/model_cot_det_loader.py`
- Legacy benchmark launcher: `scripts/v1_5/eval/cot_benchmark.sh`
- Detection output is JSONL through `--answers-file`, with parsed fields such as `bbox_gt`, `bbox_pred_raw`, `bbox_pred`, and `bbox_parse_ok`.
- The detector already exposes `--load-4bit`, `--load-8bit`, `--precision`, `--max-samples`, `--save-jsonl`, and optional visualization arguments.

## Answer Entry

- Primary answer entry: `llava/eval/model_cot_loader.py`
- Existing mode presets include `full`, `pred_bbox`, `oracle_bbox`, `random_bbox`, `center_bbox`, `woimg`, `crop_only`, `structured_evidence`, and `lowres_full_highrescrop`.
- The answer loader already accepts `--mode`, `--bbox-source`, `--crop-mode`, `--evidence-mode`, `--detection-file`, `--max-samples`, `--load-4bit`, `--load-8bit`, and `--precision`.
- This task will extend the answer entry with thesis-specific JSONL fields, reasoning compression, direct-answer policy, heuristic step selection, and safe advanced compression dry-run hooks.

## BBox Parsing, Cropping, Image Reading, Prompt Construction

- BBox parsing and normalization are in `parse_box`, `norm_box`, `oracle_box`, `pred_box_for`, and `select_box` in `llava/eval/model_cot_loader.py`.
- Center and random negative controls are implemented by `select_box`.
- Crop construction is handled by `square_crop`, `lowres_full`, and `prepare_images`.
- Image resolution fallback for CUB is handled by `resolve_image`.
- Prompt construction is handled by `make_conv_prompt`; `structured_evidence` already uses a fixed structured prompt.

## Model Loading Entry

- Model loading entry: `llava/model/builder.py::load_pretrained_model`
- 4-bit and 8-bit are supported with mutually exclusive flags.
- 4-bit uses `BitsAndBytesConfig` with NF4.
- `precision` controls `torch.bfloat16` or `torch.float16`.
- Vision tower loading happens in `load_pretrained_model` through `model.get_vision_tower()`, `vision_tower.load_model()`, and `vision_tower.to(...)`.
- The image processor is read from the vision tower.

## Image Preprocessing Entry

- Loader-side image list construction happens in `prepare_images`.
- Tensor preprocessing happens in `build_tensor`, which calls `llava.mm_utils.process_images`.
- Low-resolution full image plus high-resolution crop is represented as two visual inputs.

## Visual Encoder Feature Entry

- Vision encoder wrapper: `llava/model/multimodal_encoder/clip_encoder.py::CLIPVisionTower.forward_func`
- Feature selection: `CLIPVisionTower.feature_select`
- Model integration point: `llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal`
- The key tensor is `image_features`, produced by `self.encode_images(...)` before being inserted into text embeddings.
- PCA insertion is high risk unless the projected dimension is restored or an adapter is trained; this run will provide a dry-run/interface and record feasibility instead of claiming trained feature compression.

## Result Output JSONL

- Detection JSONL: configured by detector `--answers-file`.
- Answer JSONL: configured by answer loader `--answers-file` or `--log-jsonl`.
- This task will write all new outputs under `experiments/thesis_experiments/runs/<timestamp>/raw/`.

## Existing Tools

- Existing region compression helper: `tools/region_compress.py`
- Existing region-compression summarizer: `tools/eval_region_compress_results.py`
- Existing general analysis scripts: `experiments/analysis/summarize_results.py`, `make_tables.py`, `plot_tradeoff.py`, `visualize_cases.py`
- This task will add thesis-specific scripts under `tools/thesis/` because the required schema and outputs are broader than the existing analysis tools.

## Files To Add Or Modify

- Modify `llava/eval/model_cot_loader.py` to add thesis JSONL fields and lightweight inference-flow options.
- Add `tools/thesis/aggregate_results.py`.
- Add `tools/thesis/visualize_results.py`.
- Add `tools/thesis/generate_paper_materials.py`.
- Add `tools/thesis/reasoning_step_selector.py`.
- Add `experiments/thesis_experiments/run_thesis_experiments.sh`.
- Add run configs under `experiments/thesis_experiments/configs/`.
- Add this plan at `experiments/thesis_experiments/PLAN.md`.

## Low-Risk Experiments

- `full`: full image only.
- `pred_bbox`: predicted bbox with full image and crop.
- `crop_only`: predicted crop only.
- `lowres_full_highrescrop`: low-resolution full image plus high-resolution crop.
- `center_bbox`: center-region negative control.
- `random_bbox`: random-region negative control.
- `woimg`: no-image language-prior control.
- `structured_evidence`: fixed evidence-oriented output format.
- `structured_evidence + rule compression`: post-generation text compression; it reduces downstream text length, not first-pass generation latency.

## High-Risk Extension Experiments

- `vision_prune`: ViT/CLIP structured pruning dry-run and small-sample feasibility only unless stable.
- `feature_pca`: feature compression interface and feasibility notes; without a trained adapter, PCA reduction can break downstream dimensions or fail to reduce actual compute.
- `direct_then_explain`: can improve output controllability but may increase total latency because it uses two generations.
- `step_selector heuristic`: a lightweight proxy for key-step selection; it must not be described as completed RL training.

## Command-Line Parameter Design

- Core selection: `--mode`, `--bbox-source`, `--crop-mode`, `--evidence-mode`
- Inference controls: `--max-samples`, `--seed`, `--output-file`, `--detection-file`, `--image-folder`, `--question-file`, `--model-path`, `--temperature`, `--conv-mode`, `--load-4bit`, `--load-8bit`
- Input compression: `--lowres-size`, `--crop-size`
- Reasoning optimization: `--reasoning-compression`, `--max-reasoning-sentences`, `--max-evidence-sentences`, `--deduplicate-sentences`, `--answer-policy`, `--explanation-max-new-tokens`, `--step-selector`, `--step-selector-topk`
- Model-side feasibility: `--vision-prune`, `--vision-prune-type`, `--vision-prune-amount`, `--vision-prune-target`, `--vision-prune-dry-run`
- Feature compression feasibility: `--feature-compression`, `--pca-fit-samples`, `--pca-dim`, `--pca-position`, `--pca-save-path`, `--pca-load-path`

## Result Directory Structure

Each run writes to:

`experiments/thesis_experiments/runs/<timestamp>/`

Required subdirectories:

- `configs/`
- `commands/`
- `logs/`
- `raw/`
- `metrics/`
- `figures/case_visualizations/`
- `paper_materials/`
- `env/`

## Smoke Test Plan

- Generate or reuse a three-sample CUB subset.
- Run `full`, `pred_bbox`, `crop_only`, and `structured_evidence`.
- Verify each JSONL row has the required thesis fields, using `null` for unavailable values.
- Run `tools/thesis/aggregate_results.py`.
- Run `tools/thesis/visualize_results.py`.
- Generate paper-material stubs from real metrics.

## CUB100 Formal Experiment Plan

- Run detection on CUB100 if a current detection JSONL is not already generated inside this run directory.
- Run Level 1 modes: `full`, `pred_bbox`, `crop_only`, `lowres_full_highrescrop`, `structured_evidence`, `center_bbox`, `random_bbox`, `woimg`, and `oracle_bbox` if GT bbox is available.
- Run advanced feasibility modes on CUB20/CUB100 only after smoke tests: `vision_prune` dry-run and small inference, `feature_pca` dry-run/interface check.
- Run reasoning optimization: `structured_evidence`, `structured_evidence + rule compression`, `direct_then_explain`, and `heuristic step selector` if stable.
- Aggregate results, generate figures, and write thesis-ready method/result/limitation materials.
