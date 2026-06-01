# 毕设代码改动报告（相对原仓库 deepcs233/Visual-CoT）

> **论文**：基于模态混合的视觉语言模型推理增强方法的设计与实现
> **学生**：张恒（20226451）  **导师**：马安香 讲师
> **学院**：计算机科学与工程学院 / 人工智能
> **当前论文版本**：`16论文初稿第三版-20226451-张恒-马安香.docx`（2026-05-25）

---

## 0. 快速查看 diff（一站式入口）

| 视图 | 链接 |
|---|---|
| **全量 GitHub Compare**（推荐） | https://github.com/ymdxe/Visual-CoT/compare/257c2e8...new |
| 仅本地查看核心改动 | `git diff 257c2e8..new -- llava/` |
| 仅本地查看新增工具 | `git diff --name-only --diff-filter=A 257c2e8..new -- tools/ experiments/` |

- **Base ref** `257c2e8` = `Initial Visual-CoT project snapshot`（干净的上游代码，无任何本毕设修改）
- **Head ref** `new` = 当前 paper 16 对应的代码，已把 paper 16 不再引用的 §4.4.1 评分门控代码标记为 `[DEPRECATED]`

---

## 1. 一句话总览

> 相对上游 `257c2e8`，本仓库共 **修改 3 个核心 Python 文件 + `.gitignore`**、**新增 421 个文件**（其中绝大多数是实验配置、日志与论文素材，真正的"新增代码"集中在 `tools/thesis/` 与 `experiments/scripts/`）。
>
> **未改动任何模型训练代码，未改动模型结构本身**。所有工作都在 Visual-CoT 已有的"区域定位 → 回答生成"两阶段推理流程上做评测扩展与实验工具。

| 维度 | 数字 |
|---|---|
| 修改的上游文件 | 3 个 .py + 1 个 .gitignore |
| `llava/eval/model_cot_loader.py` 行数 | **200 → 1620**（+1432） |
| `llava/eval/model_cot_det_loader.py` 行数 | **143 → 305**（+162） |
| `llava/model/builder.py` 行数 | +20 |
| 新增 `tools/thesis/` Python 脚本 | 18 个 |
| 新增 `experiments/` 七层规范目录 | configs + scripts + analysis + 7 个 runs |

---

## 2. 三个核心代码文件的改动

### 2.1 `llava/model/builder.py` — 模型加载接口扩展（+20 行）

直接 diff：https://github.com/ymdxe/Visual-CoT/compare/257c2e8...new#diff-llava%2Fmodel%2Fbuilder.py

| 项 | 原版 | 我的版本 | 对应论文 |
|---|---|---|---|
| 函数签名 | 只支持 bf16 默认 | 新增 `precision="bf16"` 关键字参数 | §3.2 |
| 量化加载 | 无显式接口 | 增加 `load_4bit` / `load_8bit` 显式参数与冲突检查 | §2.6、§5.4 |
| 4-bit 配置 | 硬编码 `bnb_4bit_compute_dtype=bfloat16` | 跟随 `compute_dtype` 切换 | §5.4 |
| 模型名识别 | 仅 `"llava" in name` | 扩展同时识别 `"viscot"` | 工程兼容 |
| `vision_tower.to(dtype=...)` | 硬编码 bf16 | 跟随 `compute_dtype` | 工程 |

### 2.2 `llava/eval/model_cot_det_loader.py` — 区域定位阶段（+162 行）

直接 diff：https://github.com/ymdxe/Visual-CoT/compare/257c2e8...new#diff-llava%2Feval%2Fmodel_cot_det_loader.py

| 改动 | 原版 | 我的版本 | 对应论文 |
|---|---|---|---|
| **question_id Bug 修复** | 永远 `idx = 0` | `idx = line.get("question_id", local_idx)` | §3.2 |
| CLI 扩展 | 仅基本参数 | `--load-4bit / --load-8bit / --precision / --dataset-name / --max-samples` | §3.5.1 |
| 输出格式 | 只打印 bbox | JSONL 11 字段：`bbox_pred / bbox_pred_raw / bbox_parse_ok / latency_ms / peak_gpu_memory_mb / bbox_gt / height / width / answer_id / model_id / metadata` | §3.2 统一日志 |
| 可视化 | 无 | `--save-vis` 在原图上画 bbox | 辅助分析 |
| 效率统计 | 无 | `--record-latency / --record-memory` | 表 3.2 |
| 边界框解析 | 简单 | 三层兜底：`parse_bbox` / `norm_bbox` / `pixel_to_norm` | §3.1 |
| 工程兼容 | 无 | `patch_transformers_llava_registration` + `patch_optional_flash_attn` | 环境兼容 |

### 2.3 `llava/eval/model_cot_loader.py` — 回答阶段（核心改造，200 → 1620 行）

直接 diff：https://github.com/ymdxe/Visual-CoT/compare/257c2e8...new#diff-llava%2Feval%2Fmodel_cot_loader.py

#### A. 全局数据结构（论文方法的代码锚点）

```python
# 第 94-99 行
PROMPT_VERSION = {
    "none":       "v1_plain",
    "structured": "v2_structured_evidence",  # § 4.2 结构化提示
    "caption":    "v3_region_caption",
    "verify":     "v4_answer_verifier",
}

# 第 101-119 行：17 个模式预设统一入口
MODE_PRESETS = {
    "full":                       ("none",   "full_only",                 "none",       False),
    "pred_bbox":                  ("pred",   "full_crop",                 "none",       False),  # § 3.5 主流程
    "oracle_bbox":                ("oracle", "full_crop",                 "none",       False),  # § 3.5.3 理想参考
    "random_bbox":                ("random", "full_crop",                 "none",       False),  # § 3.5.3 负对照
    "center_bbox":                ("center", "full_crop",                 "none",       False),  # § 3.5.3 中心
    "woimg":                      ("none",   "full_only",                 "none",       True),   # § 3.5.3 无图像
    "crop_only":                  ("pred",   "crop_only",                 "none",       False),  # § 5.2 Φ_crop
    "structured_evidence":        ("pred",   "full_crop",                 "structured", False),  # § 4.2 主方法
    "region_caption":             ("pred",   "full_crop",                 "caption",    False),
    "answer_verifier":            ("pred",   "full_crop",                 "verify",     False),
    "lowres_full_highrescrop":    ("pred",   "lowres_full_highres_crop",  "none",       False),  # § 5.2 Φ_mix
    "answer_reasoning_compressed":("pred",   "full_crop",                 "structured", False),  # § 4.4 规则压缩
    "direct_then_explain":        ("pred",   "full_crop",                 "none",       False),  # § 4.4 先答后释
    "vision_pruned":              ("pred",   "full_crop",                 "none",       False),  # § 5.4 视觉剪枝
    "feature_pca":                ("pred",   "full_crop",                 "none",       False),  # § 5.4 PCA
    "topk_crops":                 ("pred",   "full_crop",                 "none",       False),
    "dsac":                       ("pred",   "full_crop",                 "structured", False),  # [DEPRECATED]
}
```

#### B. 论文方法对应的核心函数

| 论文段落 | 函数 | 行号 | 作用 |
|---|---|---|---|
| §4.2 结构化提示 | `make_conv_prompt(args, ..., evidence_mode="structured")` | 506 | 拼出 `[Region] ‖ [Visual Evidence] ‖ [Reasoning] ‖ [Answer]` 模板 |
| §4.2.2 字段抽取 | `parse_structured_output()` | 670 | 解析四段输出 |
| §4.2.2 结构一致性 | `compute_scs()` | 700 | `SCS = (1/4) Σ 1[section_k ∈ y]` |
| §4.4 规则压缩 | `apply_reasoning_compression()` | 784 | 去重 + 关键句保留 |
| §4.4 启发式步骤选择 | `heuristic_select_steps()` | 819 | 按关键词与句长选 top-k |
| §4.4 先答后释 | `make_direct_answer_prompt` + `make_explanation_prompt` | 573 / 582 | 两阶段生成 |
| §5.2 Φ_crop（仅局部） | `square_crop()` | 467 | 围绕 bbox 中心按 α·s(b) 裁剪 |
| §5.2 Φ_mix（低清整图） | `lowres_full()` | 483 | 缩到 s_low 再上采样回原图 |
| §5.2 视觉输入算子 Φ | `prepare_images()` | 490 | 根据 `crop_mode` 选 4 种输入策略之一 |
| §5.4 视觉剪枝 | `apply_vision_pruning()` | 838 | CLIP ViT 线性层结构化剪枝（feasibility） |
| §3.4 IoU / 包含匹配 | `box_iou` / `contains_match` | 185 / 642 | 评价指标 |
| §3.4 ANLS（归一化前） | `normalize_answer` | 633 | 答案归一化 |
| **[paper 16 移除]** §4.4.1 评分门控 | `score_pred_bbox()` | 334 | 已标记 `[DEPRECATED]` |

#### C. 30+ 个 CLI 参数（按论文章节分组）

```bash
# 基础设施（§3.2）
--load-4bit / --load-8bit / --precision {bf16,fp16}
--dataset-name / --max-samples / --seed

# 模式统一入口（§3.5 / §4 / §5 / §6）
--mode {17 个预设}

# 区域来源（§3.3 五类对照）
--bbox-source {pred,oracle,random,center,none}

# 视觉输入算子 Φ（§5.2）
--crop-mode {full_only,full_crop,crop_only,lowres_full_highres_crop}
--lowres-size 112    # ← s_low，§5.3 图 5.5 扫描
--crop-size 336
--crop-pad 1.2       # ← α，§5.3 图 5.6 扫描

# 提示组织 π（§4.2）
--evidence-mode {none,structured,caption,verify}

# 推理组织（§4.4）
--reasoning-compression {none,rule}
--max-reasoning-sentences / --max-evidence-sentences
--answer-policy {single,direct_then_explain}
--step-selector {none,heuristic}

# 模型侧压缩（§5.4 feasibility）
--vision-prune / --vision-prune-{type,amount,target,dry-run}
--feature-compression {none,pca} / --pca-{fit-samples,dim,position,save-path,load-path}

# 日志（§3.2）
--record-latency / --record-memory
--save-crop-dir / --log-jsonl

# [DEPRECATED — paper 16 已移除] §4.4.1 边界框评分门控
--bbox-scoring / --bbox-score-{threshold,lambdas,target-area,area-sigma,fallback}
```

#### D. JSONL 输出 schema（论文表格的数据源）

新输出字段约 50 个，关键的：

- **样本身份**：`question_id, dataset, image, mode, prompt_version`
- **答案匹配**（§3.4）：`exact_match, contains_match, normalized_gt, normalized_pred`
- **bbox 评估**（§3.4）：`bbox_gt, bbox_pred, bbox_iou, bbox_correct_at_05`
- **效率**（§3.2, §5.3）：`latency_ms_{total,preprocess,generate}, peak_gpu_memory_mb, num_visual_inputs, num_visual_tokens_est`
- **结构化输出**（§4.2）：`region_text, visual_evidence_text, reasoning_text`
- **压缩相关**（§4.4）：`compressed_reasoning, reasoning_compression_ratio`
- **错误分类**（§6.2）：`error_tag`

---

## 3. 新增的实验工具脚本

> 直接列表查看：https://github.com/ymdxe/Visual-CoT/tree/new/tools
> 实验目录：https://github.com/ymdxe/Visual-CoT/tree/new/experiments

### 3.1 `tools/` 顶层

| 文件 | 用途 | 论文 |
|---|---|---|
| `build_cross_region_benchmark.py` | 跨数据集 100 样本采样器（TextVQA + Visual7W + GQA） | §6.3 |
| `make_oracle_detection_jsonl.py` | 从 GT 生成 oracle bbox JSONL | §3.5.3 |
| `build_gqa_spatial.py` | GQA 空间关系子集构造（未在 paper 16 引用） | — |

### 3.2 `tools/thesis/` 第一轮（CUB100 + 高级压缩）

| 文件 | 用途 | 论文 |
|---|---|---|
| `aggregate_results.py` | CUB100 结果聚合到 summary.csv/json/md | §3.5 |
| `generate_paper_materials.py` | 生成 paper_tables.md / paper_figures.md | §3.5, §4, §5 |
| `visualize_results.py` | 9 张配套图（bbox_iou_hist / latency_bar / scatter 等） | §3.5, §5.3 |
| `reasoning_step_selector.py` | 启发式步骤选择算法独立模块 | §4.4 |
| `aggregate_cross_dataset_results.py` | 跨数据集 100 聚合 | §6.3 |
| `generate_cross_dataset_materials.py` | 跨数据集论文素材 | §6.3, §6.4 |

### 3.3 `tools/thesis/round2/` 第二轮（导师审阅后补全）

| 文件 | 用途 | 论文 |
|---|---|---|
| `phase1_analysis.py` | A1–A6 零 GPU 分析（IoU 分箱等） | §3.5.4 图 3.7 |
| `phase2_analysis.py` | B1/B2 灵敏度聚合（s_low 与 α 扫描） | §5.3 图 5.5, 5.6 |
| `phase3_analysis.py` | C1/C2 多种子稳定性聚合 | §5.4 图 5.8 |
| `aggregate_round2.py` | 汇总到 ROUND2_REPORT.md | 内部 |
| `render_paper_figures.py` | 英文图渲染（备份） | — |
| `render_paper_figures_zh.py` | **中文图渲染（paper 16 实际使用）** | 全文图 |
| `inject_distributed.py` | 按 10 个章节边界分散注入 docx | 编辑工具 |

### 3.4 `tools/thesis/round3/` 第三轮 — **paper 16 已废弃**

详见目录内 `README.md`：https://github.com/ymdxe/Visual-CoT/blob/new/tools/thesis/round3/README.md

所有 DSAC（评分门控）相关分析：
- `dsac_analysis.py` / `aggregate_cub_dsac.py` / `aggregate_cross_domain_dsac.py`
- `mcnemar_dsac_vs_se.py` / `mcnemar_cross_domain.py`
- `render_dsac_figures.py` / `render_cross_domain_figures.py`
- `render_contributions_flowchart.py` / `inject_v7.py`

### 3.5 `experiments/` 七层目录（原项目无）

```
experiments/
├── configs/        — 12 个 mode 的 YAML 配置（7 baseline + 5 ours）
├── scripts/        — 5 个 .sh 启动脚本（pilot / diagnostics / formal_eval / latency / vis）
├── analysis/       — 5 个分析 Python（classify_errors / summarize / plot_tradeoff / make_tables / visualize_cases）
└── thesis_experiments/
    ├── PLAN.md / FINAL_RESULTS.md
    ├── run_thesis_experiments.sh         # CUB100 主实验入口
    ├── run_cross_dataset_region100.sh    # 跨数据集 100 入口
    └── runs/<时间戳>_<标签>/             # 每个 run 七个标准子目录
        ├── configs/ commands/ env/       # 复现性
        ├── raw/                          # JSONL 原始日志
        ├── metrics/                      # summary.csv/json/md
        ├── figures/                      # 9 张图
        └── paper_materials/              # 5 份论文素材 md
```

---

## 4. 改动与论文章节的完整对应表

| 论文章节 | 论文页 | 代码实现 | run 目录 |
|---|---|---|---|
| §3.1 区域定位流程复现 | p.15 | `model_cot_det_loader.py` + `parse_bbox/norm_bbox` | `runs/20260508_013500_cub100/` |
| §3.2 实验平台与日志 | p.16 | `builder.py` 量化接口 + `model_cot_loader.py` JSONL | 所有 run/env/ |
| §3.3 局部依据诊断设计 | p.17 | `MODE_PRESETS` 五类对照 | `runs/20260508_013500_cub100/configs/` |
| §3.4 评价指标 | p.18 | `box_iou` / `contains_match` / ANLS | `tools/thesis/aggregate_*.py` |
| §3.5.2 IoU 分布、bbox×ans 交叉表 | p.21-22 | `tools/thesis/round2/phase1_analysis.py` | `paper_figures_zh/` |
| 表 3.2 主结果 | p.23 | `aggregate_results.py` | `runs/20260508_013500_cub100/metrics/summary.csv` |
| 图 3.7 IoU 分箱包含匹配 | p.24 | `tools/thesis/round2/phase1_analysis.py` | `runs/20260522_round2/phase1/` |
| §4.2 结构化提示 | p.27-28 | `parse_structured_output` + `make_conv_prompt(evidence_mode="structured")` | `runs/.../answer_structured_evidence.jsonl` |
| §4.4 推理组织 + 表 4.1 + 图 4.1/4.2 | p.29-30 | `apply_reasoning_compression` / `heuristic_select_steps` / 两阶段生成 | `runs/20260508_045500_reasoning100/` |
| §5.2 输入算子 Φ 形式化 | p.33-34 | `prepare_images` / `square_crop` / `lowres_full` | — |
| 表 5.1 + 图 5.7 模型侧压缩 | p.38-40 | `apply_vision_pruning` + `--feature-compression pca` | `runs/20260508_034500_advanced20/` |
| 图 5.5 / 5.6 灵敏度扫描 | p.36-37 | `tools/thesis/round2/phase2_analysis.py` | `runs/20260522_round2/phase2/` |
| 图 5.8 多种子稳定性 | p.40 | `tools/thesis/round2/phase3_analysis.py` | `runs/20260522_round2/phase3/` |
| 表 6.1 / 6.2 跨数据集 | p.46-48 | `build_cross_region_benchmark.py` + `aggregate_cross_dataset_results.py` | `runs/20260516_cross_region100_public3/` |
| 图 6.3 / 6.4 跨数据集图 | p.48-49 | `tools/thesis/generate_cross_dataset_materials.py` | 同上 |

---

## 5. 答辩 FAQ

**Q1：你具体改了哪些代码？**
> 三个文件：`llava/eval/model_cot_loader.py`（200 → 1620 行，加入 17 个实验模式与统一 JSONL 日志）、`llava/eval/model_cot_det_loader.py`（修了 `question_id` bug 并补全延迟显存日志）、`llava/model/builder.py`（加量化/精度接口）。模型权重和模型结构本身一行没改。

**Q2：原项目和你版本的最大区别？**
> 原项目以刷榜为目的，跑完得分就结束。我把它改造成可诊断工具：每条样本生成约 50 个 JSONL 字段，把模式预设、负对照、输入压缩、提示组织都做成命令行开关，让论文里每张图表都能追溯到具体的 run 目录。

**Q3：哪些是"新方法"，哪些是"工程改造"？**
> 新方法层面：(1) §4.2 结构化提示（`parse_structured_output` + 4 段模板）、(2) §5.2 全局低清 + 局部高清组合 Φ_mix（`lowres_full` + 双输入构造）、(3) §6 跨数据集小样本验证设计（`build_cross_region_benchmark.py`）。其余是工程改造，让这些方法能稳定跑出可比较的实验结果。

**Q4：为什么有些代码标 `[DEPRECATED]`？**
> Paper 16 把 §4.4.1 边界框评分门控算法（DSAC）整段移除，导师评价是"增益不显著且解释成本高"。代码保留并加 `[DEPRECATED]` 标记，是为了让 Round-3 历史 run 仍可复现，但论文不再宣传这是本文方法。

**Q5：复现性怎么保证？**
> 每个 run 目录都自动留 `env/git_info.txt`（commit hash）、`env/pip_freeze.txt`、`env/nvidia_smi.txt`、`commands/run_commands.sh`、`configs/*.yaml`。任何人拿到 run 目录，照着 `run_commands.sh` 一行行执行就能复现表里的数字。

**Q6：负对照（random/center/无图像）为何那么高？**
> 这正是论文 §6.1 重点讨论的发现。CUB 鸟类图主体大多在中央，问题模板固定为 Yes/No → 中心框与语言先验本身就能命中。所以我特意把负对照写进表 3.2，并跨域到 TextVQA / Visual7W / GQA 100 样本（表 6.1）做交叉验证——跨域场景下负对照退化得快（`woimg` ANLS = 0.0101），而结构化提示仍保持 0.3870，证明方法不是只对 CUB 有效。

---

## 6. 一句话技术栈回顾

> 本仓库 = 上游 Visual-CoT 的 LLaVA 推理代码 + 我加的"模式预设字典 + 统一 JSONL 日志 + 17 种实验模式 + 跨数据集基准构造器 + 30+ 个命令行参数 + 七层 run 目录规范 + 18 个 thesis 工具脚本"。
>
> **模型不动、训练不做、所有改动都在评测侧与实验组织侧**。
