# tools/thesis/round3 — DEPRECATED (not used in paper 16)

This directory contains the analysis and figure-rendering scripts that backed
the §4.4.1 **bbox scoring gating (DSAC)** experiment from earlier paper drafts.

**Status (2026-05-25):** the latest paper version (`16论文初稿第三版-...docx`)
removed §4.4.1 entirely. The current Chapter 4.4 is now "推理组织策略比较"
(rule compression / direct-then-explain / heuristic step selection) — see
paper 16 §4.4 for details. Nothing in this directory is cited or rendered by
the live paper anymore.

The scripts are kept so the historical Round-3 runs and McNemar / DSAC
ablations remain reproducible:

| File | What it did |
|---|---|
| `dsac_analysis.py` | Per-sample DSAC score / fallback / triggered breakdown. |
| `aggregate_cub_dsac.py` | CUB-100 DSAC sweep aggregation (θ × fallback grid). |
| `aggregate_cross_domain_dsac.py` | Cross-dataset DSAC sweep aggregation. |
| `mcnemar_dsac_vs_se.py` | McNemar test for DSAC vs. structured evidence. |
| `mcnemar_cross_domain.py` | McNemar test on cross-dataset 100. |
| `render_dsac_figures.py` | DSAC ablation bar charts (Round-3). |
| `render_cross_domain_figures.py` | Cross-domain DSAC figures (Round-3). |
| `render_contributions_flowchart.py` | Three-contributions flowchart (Round-3). |
| `inject_v7.py` | Distributed injection into v7 docx (Round-3 paper assets). |

The corresponding code in `llava/eval/model_cot_loader.py` is also marked
`[DEPRECATED]`:

- `score_pred_bbox()` and its docstring
- CLI flags `--bbox-scoring`, `--bbox-score-threshold`, `--bbox-score-lambdas`,
  `--bbox-score-target-area`, `--bbox-score-area-sigma`, `--bbox-score-fallback`

Leave `--bbox-scoring` **off** for any run that should match paper 16. If
DSAC is ever revived, restore the §4.4.1 narrative in the paper before
re-running these scripts.
