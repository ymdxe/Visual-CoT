# Visual-CoT Thesis Experiment Results

## Run Inventory

- Smoke failure run: `experiments/thesis_experiments/runs/20260508_010000_smoke`
- Smoke success run: `experiments/thesis_experiments/runs/20260508_011500_smoke_no4bit`
- CUB20 validation run: `experiments/thesis_experiments/runs/20260508_012500_cub20`
- CUB100 main run: `experiments/thesis_experiments/runs/20260508_013500_cub100`
- Advanced feasibility run: `experiments/thesis_experiments/runs/20260508_034500_advanced20`
- Reasoning optimization run: `experiments/thesis_experiments/runs/20260508_045500_reasoning100`

The initial 4-bit smoke run failed during detection with a dtype mismatch:
`expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`.
All successful runs used the non-4-bit path.

## CUB100 Main Results

| mode | N | exact_match | contains_match | latency_ms | peak_gpu_memory_mb | visual_inputs |
|---|---:|---:|---:|---:|---:|---:|
| full | 100 | 0.0000 | 0.7000 | 692.5032 | 13886.5939 | 1.0000 |
| pred_bbox | 100 | 0.5100 | 0.8000 | 354.4706 | 14212.4493 | 2.0000 |
| crop_only | 100 | 0.6600 | 0.6900 | 211.8008 | 13888.2580 | 1.0000 |
| lowres_full_highrescrop | 100 | 0.7200 | 0.7900 | 277.8703 | 14176.6949 | 2.0000 |
| structured_evidence | 100 | 0.7300 | 0.7300 | 264.9504 | 14272.1887 | 2.0000 |
| center_bbox | 100 | 0.4000 | 0.8300 | 414.3402 | 14212.4493 | 2.0000 |
| random_bbox | 100 | 0.6000 | 0.8200 | 310.6738 | 14212.4493 | 2.0000 |
| oracle_bbox | 100 | 0.3600 | 0.8200 | 437.6364 | 14212.4493 | 2.0000 |
| woimg | 100 | 0.0000 | 0.4700 | 514.9730 | 13616.0077 | 0.0000 |

Key observations:

- `pred_bbox` improves contains match over `full` from 0.7000 to 0.8000 while reducing measured latency.
- `crop_only` reduces latency substantially but lowers contains match to 0.6900, so the crop alone loses useful global context.
- `lowres_full_highrescrop` keeps contains match close to `pred_bbox` while reducing latency.
- `woimg` still reaches 0.4700 contains match, so language priors are non-negligible.
- `center_bbox` and `random_bbox` are unexpectedly high. These negative controls indicate CUB center bias and answer-prior effects; the detector result should not be over-claimed.

## Advanced Feasibility Results

| mode | N | exact_match | contains_match | latency_ms | note |
|---|---:|---:|---:|---:|---|
| feature_pca_256 | 20 | 0.5000 | 0.8500 | 385.8309 | Interface/feasibility only; no trained adapter. |
| feature_pca_512 | 20 | 0.5000 | 0.8500 | 383.8382 | Interface/feasibility only; no trained adapter. |
| vision_pruned_10 | 20 | 0.5500 | 0.6000 | 331.4417 | Linear structured pruning ran without fine-tuning. |
| vision_pruned_20 | 20 | 0.5000 | 0.5000 | 232.7996 | Larger pruning degrades accuracy; not a main result. |

The visual encoder is CLIP ViT-style. The pruning experiment is therefore reported as linear structured pruning feasibility, not convolution channel pruning. Because no fine-tuning was performed, these results should be written as a feasibility check rather than a completed model-compression method.

## Reasoning Optimization Results

| mode | N | exact_match | contains_match | latency_ms | compressed_tokens |
|---|---:|---:|---:|---:|---:|
| structured_evidence | 100 | 0.7300 | 0.7300 | 261.0688 | NA |
| answer_reasoning_compressed | 100 | 0.7300 | 0.7300 | 260.3771 | 1.0300 |
| direct_then_explain | 100 | 0.7700 | 0.7700 | 776.9798 | NA |
| heuristic_step_selector | 100 | 0.7300 | 0.7300 | 261.2435 | NA |

Rule compression is post-generation, so it reduces downstream text length rather than first-pass generation latency. `direct_then_explain` improves exact and contains match in this run but has much higher latency because it uses two generations.

## Paths

- CUB100 summary: `experiments/thesis_experiments/runs/20260508_013500_cub100/metrics/summary.csv`
- CUB100 figures: `experiments/thesis_experiments/runs/20260508_013500_cub100/figures/`
- CUB100 paper materials: `experiments/thesis_experiments/runs/20260508_013500_cub100/paper_materials/`
- Advanced summary: `experiments/thesis_experiments/runs/20260508_034500_advanced20/metrics/summary.csv`
- Reasoning summary: `experiments/thesis_experiments/runs/20260508_045500_reasoning100/metrics/summary.csv`

## Limitations

- These CUB100 results are still sample-limited and should not be described as a full benchmark conclusion.
- Input-side crop compression is not model parameter pruning and not bottom-level visual token pruning.
- Visual pruning and PCA feature compression were only feasibility experiments.
- PCA compression does not reduce actual model compute unless paired with a compatible trained projection or adapter.
- The heuristic step selector is not reinforcement learning training; it is only a lightweight proxy and interface.
