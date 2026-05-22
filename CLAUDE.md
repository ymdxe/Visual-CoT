# CLAUDE.md

本仓库是 **deepcs233/Visual-CoT** 上游的派生，用于"基于模态混合的视觉语言模型推理增强方法的设计与实现"毕业设计。本文件汇总相对原项目的**全部改动**与**新增实验**，给未来 Claude 会话快速理解上下文。

> 配合：
> - [`AGENTS.md`](AGENTS.md) — 工作规则与边界
> - [`STATUS.md`](STATUS.md) — 进度状态快照（每次工作后更新）

---

## 1. 原项目背景（不变量）

- **上游**：[deepcs233/Visual-CoT](https://github.com/deepcs233/Visual-CoT) (NeurIPS 2024 Spotlight)，基于 LLaVA
- **核心 pipeline**：`图像 + 问题 → 区域定位(D_θ) → 局部裁剪 → 答案生成(A_φ)`
- **base model**：VisCoT-7B-336 / VisCoT-7B-224 / VisCoT-13B-{224,336}
- **本仓库使用**：`checkpoints/VisCoT-7b-224`（24GB 显存 + bf16 可跑）

---

## 2. 相对原项目的代码改动

### 2.1 `llava/model/builder.py`

| 原版 | 改动 |
|---|---|
| 只支持 fp16 默认加载 | 新增 `load_4bit / load_8bit / precision` 参数 |
| 无量化配置 | 4-bit 走 NF4 + bnb double-quant |
| 接口 4 个位置参数 | 接口扩展为关键字参数（向后兼容） |

> ⚠️ 注意：4-bit 路径在 detection 阶段曾出现 `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`，**当前不要走 4-bit**，bf16 是验证过的。

### 2.2 `llava/eval/model_cot_det_loader.py`（detection 阶段）

| 原版 | 改动 |
|---|---|
| `idx = 0`（所有 question_id 都是 0） | 改为 `idx = line.get("question_id", local_idx)` |
| 无量化 / 数据集名 / 样本数 CLI | 新增 `--load-4bit / --load-8bit / --precision / --dataset-name / --max-samples` |
| 仅打印 bbox 文本 | JSONL 输出含 `bbox_pred / bbox_pred_raw / bbox_parse_ok / latency_ms / peak_gpu_memory_mb / bbox_gt / height / width / answer_id / model_id / metadata` |
| 无可视化 | 新增 `--save-vis` 在原图上画出 bbox |
| 无延迟统计 | 新增 `--record-latency / --record-memory` |

### 2.3 `llava/eval/model_cot_loader.py`（answer 阶段）— **核心改造**

代码量：原 ~200 行 → 现 **1043 行**。

**新增的全局结构**：

| 概念 | 实现 |
|---|---|
| `MODE_PRESETS` 字典 | 16 个模式预设，每个是 `(bbox_source, crop_mode, evidence_mode, without_image)` 四元组 |
| `PROMPT_VERSION` 字典 | 4 种 prompt 变体的版本标识 |
| `score_pred_bbox()` | **本文新算法：边界框评分门控**（见 §3.3） |
| `select_box()` 改 4-元组返回 | 第四个返回值是 `score_info` |
| `apply_mode()` | mode 字符串 → CLI 参数解析的统一入口 |
| `parse_structured_output()` | 结构化输出的 [Region]/[VE]/[Reasoning]/[Answer] 解析 |
| `apply_reasoning_compression()` | 规则版冗余文本压缩 |
| `apply_vision_pruning()` | CLIP 视觉编码器的结构化剪枝（feasibility） |
| `heuristic_select_steps()` | 启发式关键步骤选择 |

**新增的 CLI 参数（30+ 个）**：

```
--load-4bit / --load-8bit / --precision {bf16,fp16}
--mode {full,pred_bbox,oracle_bbox,random_bbox,center_bbox,woimg,crop_only,
        structured_evidence,region_caption,answer_verifier,
        lowres_full_highrescrop,answer_reasoning_compressed,
        direct_then_explain,vision_pruned,feature_pca,topk_crops,auto}
--bbox-source {pred,oracle,random,center,none}
--crop-mode {full_only,full_crop,crop_only,lowres_full_highres_crop}
--evidence-mode {none,structured,caption,verify}
--detection-file <jsonl>
--max-samples <int> / --seed <int>
--dataset-name <str>
--record-latency / --record-memory
--lowres-size <int=112>
--crop-size <int=336>
--crop-pad <float=1.2>                  ← 本次新增
--reasoning-compression {none,rule}
--max-reasoning-sentences / --max-evidence-sentences
--answer-policy {single,direct_then_explain}
--step-selector {none,heuristic}
--vision-prune / --vision-prune-{type,amount,target,dry-run}
--feature-compression {none,pca} / --pca-{fit-samples,dim,position,save-path,load-path}
--save-crop-dir / --log-jsonl
# 本文新算法 ↓
--bbox-scoring
--bbox-score-threshold <float=0.4>
--bbox-score-lambdas <str="1.0,1.0,0.5">
--bbox-score-target-area <float=0.1>
--bbox-score-area-sigma <float=1.0>
--bbox-score-fallback {center,oracle,none}
```

**JSONL 输出 schema（25+ 字段）**：

```
question_id, dataset, image, mode, reasoning_compression, answer_policy,
step_selector, evidence_mode, crop_mode, bbox_source, prompt_version,
prompt, gt_answer, pred_answer, answer_extracted,
normalized_gt, normalized_pred, exact_match, contains_match,
bbox_gt, bbox_pred_raw, bbox_pred, bbox_parse_ok, bbox_iou, bbox_correct_at_05,
crop_ratio, lowres_size, crop_size, crop_pad,
region_text, visual_evidence_text, reasoning_text,
compressed_reasoning, raw_reasoning_tokens, compressed_reasoning_tokens,
reasoning_compression_ratio, reasoning_length_tokens, answer_length_tokens,
latency_ms_{total,preprocess,generate}, stage1_latency_ms, stage2_latency_ms,
peak_gpu_memory_mb, num_visual_inputs, num_visual_tokens_est,
error_tag, prompt_text, model_path, model_id, answer_id, timestamp,
metadata: {detection_record, saved_crop, hit, bbox_iou, num_crops_requested,
           bbox_score: {score, components, lambdas, focus_word, threshold,
                        triggered, used_fallback, original_box, fallback_kind},
           region_caption, verifier_judgement, vision_prune_report, ...}
```

### 2.4 `llava/eval/model_cot_loader_region_compress.py`（新文件）

682 行的变体，专门做区域级输入压缩比较。原项目无对应文件。

---

## 3. 三大方法贡献（核心论文卖点）

### 3.1 结构化区域证据提示（§4.2）

固定模板 `𝒯_schema = [Region] || [Visual Evidence] || [Reasoning] || [Answer]`，让模型在答前组织局部证据。

- 实现：`make_conv_prompt()` 的 `args.evidence_mode == "structured"` 分支
- 公式：`π_struct(q, b) = q ‖ bbox(b) ‖ 𝒯_schema`
- 量化：结构一致性得分 `SCS(y) = (1/4) Σ 1[section_k ∈ y]`
- 实测：SCS_mean = 0.50（仅 2/4 段稳定产出），但严格匹配 0.260 → 0.317

### 3.2 区域级视觉输入压缩（§5.2）

四种视觉输入算子 Φ 的统一形式化：
```
Φ_full(I,b)  = {I}
Φ_pred(I,b)  = {I, C(I,b,α)}
Φ_crop(I,b)  = {C(I,b,α)}
Φ_mix(I,b)   = {R(I,s_low), C(I,b,α)}
```

- 压缩率 `η(Φ) = 1 - τ(Φ)/τ(Φ_pred)`，`τ = |Φ| · 576`
- `Φ_crop` η = 0.5，`Φ_mix` η = 0 但通过信息瓶颈正则
- 实现：`prepare_images()` + `square_crop()` + `lowres_full()`

### 3.3 边界框评分门控算法（§4.4.1 本文提出）

```
score(b̂) = (λ₁·s_parse + λ₂·s_size + λ₃·s_focus) / Σᵢ λᵢ

s_parse = 1[b̂ 解析成功]
s_size  = exp(-(log ρ(b̂) - log ρ*)² / (2σ²))      高斯面积先验
s_focus = max{1 - ‖c(b̂) - c_word‖₂/0.5 :
              word ∈ q ∩ KeywordSet}              方位关键词聚焦
```

- 阈值化回退：`score < θ → b* = FallbackBox()`（中心框 / Oracle）
- 实现：`score_pred_bbox()` 在 `llava/eval/model_cot_loader.py` 第 305-376 行附近
- 实验：6 种配置消融（默认 / 严格 / 宽松 / 无 focus / 无 size / oracle 回退）

---

## 4. 新增实验目录

```
experiments/thesis_experiments/runs/
├── 20260508_010000_smoke              ← 4-bit 失败 smoke
├── 20260508_011500_smoke_no4bit       ← bf16 smoke 成功
├── 20260508_012500_cub20              ← CUB 20 样本验证
├── 20260508_013500_cub100             ← CUB100 主实验（9 modes × 100）
├── 20260508_034500_advanced20         ← 高级压缩可行性（PCA/剪枝）
├── 20260508_045500_reasoning100       ← 推理优化（4 modes × 100）
├── 20260516_cross_region100_public3   ← 跨数据集 100（TextVQA+GQA+Visual7W，8 modes）
├── 20260516_cross_region_smoke_*      ← 跨数据集 smoke 系列
└── 20260522_round2_extensions         ← Round-2 答辩补全（详见 STATUS.md §3）
```

每个 run 都含 `configs/ / commands/ / env/ / raw/ / metrics/ / figures/ / paper_materials/` 七个标准子目录。

---

## 5. 实验关键数字

### 5.1 CUB100（FINAL_RESULTS.md 口径）

| Mode | EM | CM | Latency | 备注 |
|---|---:|---:|---:|---|
| pred_bbox | 0.51 | 0.80 | 354 ms | Visual-CoT 主流程 |
| structured_evidence | **0.73** | 0.73 | 265 ms | 最佳 EM |
| lowres+highres | 0.72 | 0.79 | 278 ms | 接近 pred_bbox，更快 |
| crop_only | 0.66 | 0.69 | **212 ms** | 最快 |
| **诚实警示** | | | | random_bbox 0.82 / center_bbox 0.83 → CUB 有中心偏置 |

### 5.2 跨数据集 100（TextVQA + GQA + Visual7W）

| Mode | CM | EM | Latency | 备注 |
|---|---:|---:|---:|---|
| woimg | 0.08 | 0.01 | 927 ms | 跨域时语言先验失效 |
| pred_bbox | 0.41 | 0.26 | 303 ms | |
| **structured_evidence** | **0.43** | **0.32** | 414 ms | |
| crop_only | 0.32 | 0.25 | **212 ms** | |

### 5.3 Round-2 新结果

**C1 多 seed 稳定性**（T=0.2, top_p=0.95, 3 种子）：
- pred_bbox: 0.393 ± 0.025
- structured_evidence: **0.443 ± 0.012**
- lowres_full_highrescrop: 0.363 ± 0.006
- 提升 0.05 > 噪声 std 0.025 → 结论显著

**C2 边界框评分门控**：
- baseline 0.41 → default θ=0.5 → 0.42（回退率 54%）
- → strict θ=0.7 → **0.43**（回退率 79%，幸存样本 IoU≥0.5 比例 0.32 → 0.67）
- → oracle fallback → **0.46**（+0.05 vs baseline）

---

## 6. 新增工具脚本

### `tools/`（thesis-level，原项目无）

| 文件 | 用途 |
|---|---|
| `build_cross_region_benchmark.py` | 跨数据集 100 样本构造器（textvqa+docvqa+sroie + 兜底） |
| `make_oracle_detection_jsonl.py` | 从 GT 生成 oracle bbox JSONL |
| `eval_region_compress_results.py` | 区域压缩结果聚合 |
| `region_compress.py` | 区域压缩辅助 |
| `thesis/aggregate_results.py` | CUB100 等聚合 |
| `thesis/generate_paper_materials.py` | 论文章节素材生成 |
| `thesis/visualize_results.py` | 结果可视化 |
| `thesis/reasoning_step_selector.py` | 启发式步骤选择 |
| `thesis/aggregate_cross_dataset_results.py` | 跨数据集聚合 |
| `thesis/generate_cross_dataset_materials.py` | 跨数据集论文素材 |
| `thesis/round2/phase1_analysis.py` | A1-A6 零 GPU 分析 |
| `thesis/round2/phase2_analysis.py` | B1/B2 灵敏性聚合 |
| `thesis/round2/phase3_analysis.py` | C1/C2 聚合 |
| `thesis/round2/aggregate_round2.py` | 汇总到 ROUND2_REPORT.md |
| `thesis/round2/render_paper_figures.py` | 英文版图渲染 |
| `thesis/round2/render_paper_figures_zh.py` | **中文版图渲染** |
| `thesis/round2/inject_into_thesis.py` | 集中式注入（已废弃） |
| `thesis/round2/inject_distributed.py` | **分散式注入（10 个章节边界）** |

### `experiments/scripts/` 与 `experiments/analysis/`

原项目无 `experiments/` 目录。本仓库的 `experiments/` 下：

```
scripts/run_{pilot,diagnostics,formal_eval,latency_benchmark,visualizations}.sh
analysis/{classify_errors,summarize_results,plot_tradeoff,make_tables,visualize_cases}.py
configs/baseline_*.yaml + ours_*.yaml （12 个 mode 配置）
```

---

## 7. 论文与文档资产

### Obsidian 仓库结构

```
Obsidian/
├── 6论文初稿第三版-...docx         ← 当前活跃版（中文图）
├── 6 修改说明-20260522.md          ← 版本变更说明
├── 论文初稿-..._审阅意见_20260521.md ← 导师审阅原文
├── assets/round2/
│   ├── paper_figures/      ← 英文图（备份）
│   └── paper_figures_zh/   ← 中文图（v6 使用）
└── 过程文档/                ← 历史归档（v2/v3集中式/v3分散式/v5）
```

### 论文文件版本演进

| # | 状态 | 主要变化 |
|---|---|---|
| 第二版 | 归档 | 导师审阅版本 |
| v3 集中式 | 废弃 | 章末"增补 A/B/C" |
| v3 分散式 | 归档 | 按 10 个小节边界分散 |
| v5 | 归档 | 用户手动整理后版本 |
| **v6** | **活跃** | 5 张新增图汉化 |

---

## 8. 工程约定（详见 AGENTS.md）

- 论文版本号用前序号（`5论文…` → `6论文…`），正文标题不变
- 修改说明同样前序号（`6 修改说明-...md`）
- 历史版本归 `Obsidian/过程文档/`，根目录只留最新版 + 审阅意见
- 实验目录用 `runs/<时间戳>_<标签>/`，不污染老 run
- JSONL 新字段加在尾部，不重排
- 图脚本统一在 `tools/thesis/round2/`，不在别处重画

---

## 9. 不在本仓库做的事

- ❌ 不重新训练大模型
- ❌ 不跑完整 benchmark
- ❌ 不做底层 visual token / patch 级别的剪枝（feasibility 接口除外）
- ❌ 不依赖外部 API 打分（自评 normalize_match）
- ❌ 不动 `Obsidian/过程文档/`

---

## 10. 进入新会话时的标准动作

1. 读 `STATUS.md` 确认当前位置（当前版本、待办、已知问题）
2. 读 `AGENTS.md` 确认工作规则
3. 必要时读本文件（CLAUDE.md）补全项目改造历史
4. 退出前更新 `STATUS.md`（版本表、未完成事项、已知问题）
