# 评估脚本功能说明（EVAL_LOADERS）

> 本文整理 `llava/eval/model_cot_det_loader.py` 与 `llava/eval/model_cot_loader.py` 两个评估脚本的功能与定位，方便答辩与代码走读。

---

## 1. 整体定位：两阶段推理的两个引擎

Visual-CoT 的核心 pipeline 是 **两阶段推理**：

```
第一阶段（det_loader.py）            第二阶段（cot_loader.py，本仓库核心改造）
┌─────────────────────┐              ┌────────────────────────────┐
│ 图像 + 问题          │              │ 图像 + 问题 + 已预测 bbox   │
│        ↓            │              │        ↓                   │
│ 模型预测 bbox        │ ──────────→  │ 按 bbox 裁出局部 → 答题      │
│        ↓            │   JSONL      │        ↓                   │
│ detection.jsonl     │              │ answer.jsonl               │
└─────────────────────┘              └────────────────────────────┘
```

- `model_cot_det_loader.py`：**只跑第一阶段**——给一批图像问题，输出"答案应该在哪一块"的 bbox JSONL
- `model_cot_loader.py`：**跑第二阶段**——读上面那份 JSONL，按指定的"喂图方式 + 提示风格 + 答案组织策略"组合出答案

两份 JSONL 的字段是**协议化对齐**的（都用 `question_id` 做主键），让 round2/round3 所有聚合脚本能在同一口径下分析。

---

## 2. `model_cot_det_loader.py`（一阶段定位器，~301 行）

### 2.1 它在干什么

把"先指框"那一步**独立成可离线复用、统一格式、带延迟/显存日志的批跑脚本**。

每条样本的 prompt 固定为：

```
{expression}. Please provide the bounding box coordinate of the region
that can help you answer the question better.
```

模型自由生成一段含 4 个浮点数的文本，本脚本负责把它解析、归一、记录、（可选）画在原图上。

### 2.2 核心运行骨架

```
eval_model(args)  ← 第 188 行
  ├─ patch_transformers_llava_registration()   兼容补丁
  ├─ patch_optional_flash_attn()               flash-attn 缺包 stub
  ├─ load_pretrained_model(...)                bf16/fp16/8-bit
  ├─ load_questions / get_chunk
  └─ for each line:
       prompt = "{expr}. Please provide the bounding box ..."
       output = model.generate(...)            ← 计延迟 + peak GPU mem
       raw_box = output                        ← 模型原始字符串
       pred_box = parse_bbox(raw_box)          ← 正则抠 4 个浮点数
       pred_box = norm_bbox(pred_box)          ← 归一到 [0,1]
       maybe_save_vis(...)                     ← 可选：画框存 jpg
       写入 detection.jsonl
```

### 2.3 关键函数表

| 行号 | 函数 | 干啥 |
|---|---|---|
| 24-34 | `patch_transformers_llava_registration` | 避免 transformers 注册 LLaVA 时报"已注册" |
| 37-57 | `patch_optional_flash_attn` | 没装 flash_attn 时打 stub，让 3090 也能跑 |
| 85-97 | `parse_bbox` | 从模型自由文本里抠 4 个浮点数（首选 `tools.rc.box_from`，缺则正则兜底） |
| 100-108 | `norm_bbox` | 裁 \[0,1\] 区间 + 检查 x2>x1, y2>y1 |
| 111-117 | `pixel_to_norm` | 像素坐标自动转归一坐标（看最大值是否 >1） |
| 120-128 | `resolve_image` | 路径兜底：`cot/cub/...` → `downloads/cub/CUB_200_2011/images/...` |
| 131-160 | `CustomDataset` | 拼 detection prompt + 加 `<image>` token + 过 conversation 模板 |
| 169-185 | `maybe_save_vis` | `--save-vis`：在原图画红框存 jpg 肉眼检查 |
| 188-274 | `eval_model` | 主循环：加载模型 → 逐样本前向 → 解析 → 写 JSONL |

### 2.4 相对上游的改造（CLAUDE.md §2.2）

| 项 | 原版 | 改后 |
|---|---|---|
| `question_id` | 永远是 `0` | `idx = line.get("question_id", local_idx)` —— 不修就跨阶段对齐崩了 |
| 量化 | 无 | 加 `--load-4bit / --load-8bit / --precision` |
| 数据集名 / 样本数 | 无 | 加 `--dataset-name / --max-samples` |
| 输出字段 | 仅 `text` | 25+ 字段：`bbox_pred_raw / bbox_pred / bbox_parse_ok / latency_ms / peak_gpu_memory_mb / bbox_gt / height / width / answer_id / model_id / metadata` |
| 可视化 | 无 | `--save-vis` 在原图画框 |
| 延迟 / 显存 | 无 | `--record-latency / --record-memory` |

### 2.5 注意事项

- **不要走 `--load-4bit`**：CLAUDE.md §2.1 记了 detection 阶段会 `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`
- 输入 question file schema：`{img_path, expression, question_id, bbox, height, width, dataset, split}`，每行一条 JSON
- 与 `tools/make_oracle_detection_jsonl.py` 互补：后者直接用 GT bbox 伪造 detection 文件，作为 oracle 上限对照

### 2.6 谁会用它

- `experiments/thesis_experiments/runs/<时间戳>_*/raw/det_*.jsonl` 都是它产的
- `model_cot_loader.py` 用 `--detection-file` 消费它的输出

---

## 3. `model_cot_loader.py`（二阶段答题器，1312 行）

### 3.1 它在干什么

**本仓库改造最重的核心文件**——原项目 ~200 行，现 1312 行。承载了三大方法贡献的**全部实现**。一句话：**给一批"图像 + 问题 + 已预测 bbox"，按指定的"喂图方式 + 提示风格 + 答案组织策略"组合跑一遍，输出 25+ 字段的 JSONL，让所有对照实验在同一口径下可比**。

### 3.2 为什么膨胀到 1312 行——四件大事

| 贡献 | 在本文件里的实现 | 关键行号 |
|---|---|---|
| **§4.2 结构化区域证据提示** | `make_conv_prompt()` 的 `evidence_mode == "structured"` 分支，吐 `[Region]/[Visual Evidence]/[Reasoning]/[Answer]` 四段 | 506 |
| | `parse_structured_output()` 解析 | 670 |
| | `compute_scs()` 计算结构一致性 | 700 |
| **§5.2 区域级视觉输入压缩** | `square_crop()` + `lowres_full()` + `prepare_images()` 实现 Φ_full / Φ_pred / Φ_crop / Φ_mix 四种喂图算子 | 467 / 483 / 490 |
| **§4.4.1 边界框评分门控（本文新算法）** | `score_pred_bbox()`：s_parse + s_size + s_focus 三项加权 | 334 |
| | `select_box()` 做阈值化回退（4 元组返回，第四个是 score_info） | 409 |
| **§4.6 DSAC 双端自适应控制器（Round-3 新算法）** | `composite_decision()`：composite = α·score + (1-α)·SCS 三档决策 | 729 |
| | `adaptive_extract()`：中档 re_extract 自适应抽取 | 715 |

### 3.3 核心运行骨架

```
eval_model(args)  ← 第 913 行
  ├─ apply_mode(args)               第 890 行：mode 字符串 → 16 个预设组合
  ├─ load_pretrained_model(...)     支持 bf16/fp16/8-bit
  ├─ load_questions / load_detection
  ├─ apply_vision_pruning(model)    第 838 行：CLIP 结构化剪枝（feasibility）
  └─ for each line:
       box, score_info = select_box(...)              ← 评分门控决定最终 bbox
       images = prepare_images(args, img, box)         ← 决定喂几张、什么分辨率
       prompt = make_conv_prompt(...)                  ← 决定提示风格
       output = generate(...)
       parsed = parse_structured_output(output)        ← 抽 4 段
       scs   = compute_scs(parsed)
       reasoning = apply_reasoning_compression(parsed) ← 规则压缩
       em / cm / iou / latency / mem ...
       写入 JSONL（25+ 字段，metadata 尾部带 bbox_score 与 composite_gate）
```

### 3.4 16 种 mode 预设（`MODE_PRESETS`，第 101 行）

每个 mode 是一个 4 元组 `(bbox_source, crop_mode, evidence_mode, without_image)`，覆盖整篇论文的所有对照组：

| 类别 | mode |
|---|---|
| **诊断对照** | `full / pred_bbox / oracle_bbox / random_bbox / center_bbox / woimg / crop_only` |
| **提示风格** | `structured_evidence / region_caption / answer_verifier / direct_then_explain` |
| **输入压缩** | `lowres_full_highrescrop` |
| **答案组织** | `answer_reasoning_compressed` |
| **可行性** | `vision_pruned / feature_pca / topk_crops` |
| **DSAC** | `dsac` = `("pred", "full_crop", "structured", False)` + `--adaptive-gate` |

一个 `--mode pred_bbox` 就把 bbox 来源、裁图方式、提示模式、是否带图全设好了——这是为什么实验脚本都能写得很短。

### 3.5 CLI 参数 30+ 个（按功能分组）

| 组 | 参数 |
|---|---|
| 模型加载 | `--load-4bit / --load-8bit / --precision` |
| 数据 | `--detection-file / --max-samples / --seed / --dataset-name` |
| 模式控制 | `--mode / --bbox-source / --crop-mode / --evidence-mode` |
| 输入压缩 | `--lowres-size / --crop-size / --crop-pad` |
| 答案组织 | `--reasoning-compression / --answer-policy / --step-selector` |
| 视觉剪枝 | `--vision-prune / --vision-prune-{type,amount,target}` |
| 特征压缩 | `--feature-compression / --pca-{fit-samples,dim,position}` |
| **本文新算法** | `--bbox-scoring / --bbox-score-{threshold,lambdas,target-area,area-sigma,fallback}` |
| **DSAC** | `--adaptive-gate / --adaptive-{alpha,theta-low,theta-high,fallback}` |
| 日志 | `--save-crop-dir / --log-jsonl / --record-latency / --record-memory` |

### 3.6 JSONL 输出 schema 是聚合分析的"协议"

第 25+ 个字段不是凑数，是为了让下游 `tools/thesis/round{2,3}/` 里所有聚合脚本（`phase{1,2,3}_analysis.py / dsac_analysis.py / mcnemar_*.py`）能在**任何 mode 上**做：

| 维度 | 字段 |
|---|---|
| 答题正确性 | `exact_match / contains_match / normalized_{gt,pred}` |
| 定位正确性 | `bbox_iou / bbox_correct_at_05` |
| 效率 | `latency_ms_{total,preprocess,generate} / stage1_latency_ms / stage2_latency_ms / peak_gpu_memory_mb / num_visual_tokens_est` |
| 结构化产出 | `region_text / visual_evidence_text / reasoning_text / compressed_reasoning` |
| 算法元信息 | `metadata.bbox_score`（评分门控）、`metadata.composite_gate`（DSAC）——加在 metadata 尾部不重排，向后兼容 |

### 3.7 三个不容易看出来的设计点

1. **`select_box()` 改 4 元组返回**（第 409 行）：第四个返回值 `score_info` 是给评分门控算法用的，所有其他 mode 也会带空 `score_info`，保持 schema 统一
2. **`metadata` 不重排**：新增字段（DSAC 的 `composite_gate`、剪枝的 `vision_prune_report`）一律往尾部加。这是和 round2/round3 聚合脚本的隐性契约
3. **`apply_mode()` 在 args parse 之后才把 mode 拆开**（第 890 行）：所以 CLI 同时给了 `--mode` 和单项参数（`--bbox-source` 等）时，后者覆盖前者——单项参数优先级高，方便做局部消融

### 3.8 关键函数表

| 行号 | 函数 | 干啥 |
|---|---|---|
| 94-99 | `PROMPT_VERSION` | 4 种 prompt 变体的版本标识 |
| 101-120 | `MODE_PRESETS` | 16 个 mode 预设 |
| 277-298 | `oracle_box / pred_box_for` | bbox 来源解析 |
| 307-322 | `_FOCUS_KEYWORDS / _parse_lambdas` | 方位关键词→中心坐标映射；CLI 解析 `"1.0,1.0,0.5"` |
| 334-403 | **`score_pred_bbox`** | **本文核心算法**：score = (λ₁·s_parse + λ₂·s_size + λ₃·s_focus) / Σλ |
| 405-407 | `_center_box` | 默认中心回退框 |
| 409-465 | `select_box` | 综合 bbox 来源 + 评分门控 + 回退策略，返回最终 box |
| 467-481 | `square_crop` | 按 bbox 居中正方形裁，padding 系数 α |
| 483-488 | `lowres_full` | 整图缩到 112/224 等 |
| 490-504 | `prepare_images` | Φ 算子统一入口（full / full_crop / crop_only / lowres_full_highres_crop） |
| 506-559 | `make_conv_prompt` | 结构化提示构造（evidence_mode 四分支） |
| 633-655 | `normalize_answer / contains_match` | 自评匹配 |
| 670-698 | `parse_structured_output` | 抽 [Region]/[VE]/[Reasoning]/[Answer] 四段 |
| 700-713 | `compute_scs` | 结构一致性得分 SCS = (1/4) Σ 1[section_k ∈ y] |
| 715-727 | `adaptive_extract` | DSAC 中档 re_extract 算子 |
| 729-760 | **`composite_decision`** | **DSAC**：composite = α·score + (1-α)·SCS 三档决策 |
| 784-817 | `apply_reasoning_compression` | 规则版冗余文本压缩 |
| 819-828 | `heuristic_select_steps` | 启发式关键步骤选择 |
| 838-888 | `apply_vision_pruning` | CLIP 视觉编码器结构化剪枝（feasibility） |
| 890-911 | `apply_mode` | mode 字符串 → CLI 参数解析的统一入口 |
| 913- | `eval_model` | 主循环 |

### 3.9 谁会调用它

- `experiments/thesis_experiments/runs/<时间戳>_*/scripts/run_*.sh` 全部都在 `python -m llava.eval.model_cot_loader ...`
- 单次实验产出 `raw/answer_*.jsonl`（每个 mode 一个文件），由 round{2,3} 工具聚合到 metrics/figures
- 它是**整个论文实验链路的引擎**——除了 detection（一阶段），其他所有数字都从这里出

---

## 4. 二者协作的最小可运行例子

```bash
# 第一阶段：跑 detection
python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub100.json \
  --image-folder downloads/ \
  --answers-file runs/demo/raw/det_cub100.jsonl \
  --precision bf16 \
  --record-latency --record-memory

# 第二阶段：跑 answer（结构化提示模式）
python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub100.json \
  --detection-file runs/demo/raw/det_cub100.jsonl \
  --image-folder downloads/ \
  --answers-file runs/demo/raw/answer_se.jsonl \
  --mode structured_evidence \
  --precision bf16 \
  --record-latency --record-memory
```

加上 `--bbox-scoring --bbox-score-threshold 0.5 --bbox-score-fallback oracle` 就启用本文边界框评分门控；加上 `--mode dsac --adaptive-gate --adaptive-theta-low 0.6 --adaptive-fallback oracle` 就启用 DSAC。

---

## 5. 一图总结

```
┌────────────────── llava/eval/ ──────────────────┐
│                                                  │
│  model_cot_det_loader.py    (~301 行)            │
│  └─ 第一阶段定位器：图像+问题 → bbox             │
│     输入：question.jsonl                          │
│     输出：detection.jsonl（25+ 字段）             │
│                                                  │
│  model_cot_loader.py        (1312 行，核心)      │
│  └─ 第二阶段答题器：图像+问题+bbox → 答案         │
│     输入：question.jsonl + detection.jsonl       │
│     输出：answer.jsonl（25+ 字段）                │
│     承载：§4.2 结构化提示                         │
│           §4.4.1 边界框评分门控（本文新算法）     │
│           §4.6 DSAC（Round-3 新算法）             │
│           §5.2 视觉输入压缩（4 种 Φ 算子）       │
│           16 种 mode 预设 + 30+ CLI 参数         │
│                                                  │
└──────────────────────────────────────────────────┘
                       ↓
        runs/<时间戳>_*/raw/{det,answer}_*.jsonl
                       ↓
        tools/thesis/round{2,3}/ 各种聚合脚本
                       ↓
            metrics/ + figures/ + paper_materials/
```
