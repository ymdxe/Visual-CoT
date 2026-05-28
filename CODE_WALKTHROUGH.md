# 程序代码讲解文档（Code Walkthrough）

> 本文档面向毕设程序展示，按"**代码流向 → 改动清单 → 运行步骤 → 结果解读**"四个层次详细讲解。  
> 配合 [`DEFENSE_SCRIPT.md`](DEFENSE_SCRIPT.md)（口语化讲稿）和 [`THESIS_CHANGES.md`](THESIS_CHANGES.md)（相对上游 diff）阅读。

---

## 0. 一图看懂整个程序

```
                    输入
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
  benchmark JSON                  原图
  (问题 + GT bbox)              (playground/data/cot/)
       │                             │
       └──────────────┬──────────────┘
                      ▼
       ╔════════════════════════════╗
       ║  Stage 1: Detection 阶段   ║
       ║  model_cot_det_loader.py   ║
       ╚════════════════════════════╝
                      │
                      │  输出 detection.jsonl
                      │  ┌──────────────────────────────┐
                      │  │ question_id / bbox_pred /     │
                      │  │ bbox_pred_raw / bbox_parse_ok │
                      │  │ latency_ms / peak_gpu_mem    │
                      │  └──────────────────────────────┘
                      ▼
       ╔════════════════════════════╗
       ║  Stage 2: Answer 阶段      ║
       ║  model_cot_loader.py        ║
       ║                             ║
       ║  ─ MODE_PRESETS 选模式      ║
       ║  ─ select_box  选边界框     ║
       ║  ─ prepare_images 构视觉输入║
       ║  ─ make_conv_prompt 构提示词║
       ║  ─ generate 生成答案        ║
       ║  ─ parse_structured_output  ║
       ║      解析结构化输出         ║
       ╚════════════════════════════╝
                      │
                      │  输出 answers.jsonl（25+ 字段/行）
                      ▼
       ╔════════════════════════════╗
       ║  聚合 & 可视化              ║
       ║  tools/thesis/              ║
       ║    aggregate_results.py     ║
       ║    visualize_results.py     ║
       ╚════════════════════════════╝
                      │
                      ▼
            summary.md + figures/*.png
```

**核心设计点**：两阶段解耦 + 统一 JSONL schema + 16 种模式预设。

---

## 1. 项目结构总览（仓库根目录）

```
Visual-CoT/
├── llava/                          ← 原项目代码（LLaVA 基座）
│   ├── model/
│   │   └── builder.py              [改] 新增量化加载路径
│   └── eval/
│       ├── model_cot_det_loader.py [改] detection 阶段，新增 JSONL 输出
│       ├── model_cot_loader.py     [改] answer 阶段（核心改造，200 → 1043 行）
│       └── model_cot_loader_region_compress.py [新] 区域压缩变体（682 行）
│
├── data/benchmarks/                [新] 实验数据 JSON
│   ├── small_benchmark_cub100.json
│   ├── small_benchmark_gqa_spatial100.json
│   ├── small_benchmark_visual7w100.json
│   ├── small_benchmark_docvqa100.json
│   ├── small_benchmark_infographicsvqa100.json
│   └── README.md
│
├── tools/                          [新] 工具脚本
│   ├── build_small_benchmark.py        # 构造 benchmark
│   ├── build_cross_region_benchmark.py # 跨数据集 benchmark
│   ├── make_oracle_detection_jsonl.py  # 从 GT 生成 oracle bbox JSONL
│   └── thesis/                         # 论文级工具
│       ├── aggregate_results.py        # 聚合 raw JSONL
│       ├── visualize_results.py        # 出图
│       ├── render_sample_triptych.py   # 三联展示图
│       ├── round2/                     # Round-2 聚合脚本
│       └── round3/                     # Round-3 DSAC 分析
│
├── experiments/                    [新] 实验目录
│   ├── scripts/                    # 一键运行脚本
│   ├── analysis/                   # 分析脚本
│   ├── configs/                    # mode 配置 yaml
│   └── thesis_experiments/runs/    # 所有 run 的标准化目录
│
├── images/                         [新] GitHub 展示图
│   ├── benchmarks/                 # 三联示例图（原图+bbox+裁剪）
│   ├── contributions_flowchart.png # 贡献流程图
│   └── supp_demo_*.png             # 论文 demo
│
├── checkpoints/                    # 模型权重（gitignored）
├── playground/data/                # 原始图像数据（gitignored）
│
├── README.md                       # 上游 README（保留）
├── CLAUDE.md                       [新] 完整改造记录（内部上下文）
├── AGENTS.md                       [新] 工作规则
├── STATUS.md                       [新] 进度快照
├── DEFENSE_SCRIPT.md               [新] 答辩口语化讲稿
├── THESIS_CHANGES.md               [新] 相对上游的精简 diff
└── CODE_WALKTHROUGH.md             ← 本文档
```

---

## 2. 代码改动清单（相对 deepcs233/Visual-CoT 上游）

### 2.1 改动文件一览

| 文件 | 类型 | 原版行数 | 现行数 | 改动密度 |
|---|---|---:|---:|---|
| `llava/model/builder.py` | 改造 | ~100 | ~140 | 新增量化路径 |
| `llava/eval/model_cot_det_loader.py` | 改造 | ~150 | 300 | 改 JSONL 输出 + CLI |
| `llava/eval/model_cot_loader.py` | **核心改造** | ~200 | **1312** | **6 倍扩展** |
| `llava/eval/model_cot_loader_region_compress.py` | 新增 | — | 682 | 区域压缩变体 |
| `tools/build_*.py` | 新增 | — | ~400 | benchmark 构造 |
| `tools/thesis/**` | 新增 | — | ~3000 | 聚合/可视化 |
| `experiments/scripts/*.sh` | 新增 | — | ~600 | 运行脚本 |

### 2.2 `llava/model/builder.py` 改动

| 原版 | 改造后 |
|---|---|
| 只支持 fp16 默认加载 | 新增 `load_4bit / load_8bit / precision` 三个关键字参数 |
| 无量化配置 | 4-bit 走 NF4 + bnb double-quant（`BitsAndBytesConfig`） |
| 接口 4 个位置参数 | 扩展为关键字参数（向后兼容） |

关键代码位置：[`llava/model/builder.py:35-65`](llava/model/builder.py#L35)

```python
def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    precision="bf16",
    **kwargs,
):
    if load_4bit and load_8bit:
        raise ValueError("load_4bit and load_8bit are mutually exclusive")
    compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    ...
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
```

> ⚠️ 注意：4-bit 路径在 detection 阶段曾报 `dtype mismatch`，**当前实验都走 bf16**。

### 2.3 `llava/eval/model_cot_det_loader.py` 改动（detection 阶段）

| 原版 | 改造后 |
|---|---|
| `idx = 0`（所有 question_id 都是 0，bug） | `idx = line.get("question_id", local_idx)` |
| 无量化/数据集/样本数 CLI | 新增 `--load-4bit / --precision / --dataset-name / --max-samples` |
| 仅打印 bbox 文本 | JSONL 输出含 `bbox_pred / bbox_pred_raw / bbox_parse_ok / latency_ms / peak_gpu_memory_mb / bbox_gt / height / width / metadata` |
| 无可视化 | 新增 `--save-vis` 在原图上画 bbox |
| 无延迟统计 | 新增 `--record-latency / --record-memory` |

关键代码位置：`llava/eval/model_cot_det_loader.py:188-280` (`eval_model`) + `:279-300` (CLI)

### 2.4 `llava/eval/model_cot_loader.py` 改动（answer 阶段）— **核心**

**改造规模**：原 200 行 → 现 **1312 行**（含 1043 行新增逻辑 + 269 行 CLI/解析）。

#### 关键全局结构（行号锚点）

| 概念 | 位置 | 说明 |
|---|---|---|
| `PROMPT_VERSION` | `:94-99` | 4 种 prompt 变体版本号 |
| `MODE_PRESETS` | `:101-120` | **16 种推理模式预设**（核心入口） |
| `parse_box / norm_box` | `:139-176` | bbox 解析与归一化 |
| `score_pred_bbox` | `:334-402` | 边界框评分门控（§4.4.1，已废弃） |
| `select_box` | `:409-465` | 边界框选择（pred/oracle/random/center） |
| `square_crop / lowres_full / prepare_images` | `:467-504` | 视觉输入构造（§5.2） |
| `make_conv_prompt` | `:506-560` | **核心提示词构造**（结构化模板入口） |
| `parse_structured_output` | `:670-698` | 解析 `[Region][VE][Reasoning][Answer]` |
| `apply_reasoning_compression` | `:784-817` | 规则版推理压缩 |
| `apply_vision_pruning` | `:838-888` | CLIP 视觉编码器剪枝（feasibility） |
| `apply_mode` | `:890-911` | **mode 字符串 → 4 元组分发器** |
| `eval_model` | `:913-1213` | 主循环（300 行） |
| `build_parser` | `:1215-1310` | CLI 参数（66 个） |

#### 16 种推理模式（`MODE_PRESETS`）

每个模式是 `(bbox_source, crop_mode, evidence_mode, without_image)` 四元组：

```python
MODE_PRESETS = {
    "full":                  ("none",   "full_only",                  "none",       False),
    "pred_bbox":             ("pred",   "full_crop",                  "none",       False),  # VisCoT 原版
    "oracle_bbox":           ("oracle", "full_crop",                  "none",       False),
    "random_bbox":           ("random", "full_crop",                  "none",       False),
    "center_bbox":           ("center", "full_crop",                  "none",       False),
    "woimg":                 ("none",   "full_only",                  "none",       True),   # 不给图
    "crop_only":             ("pred",   "crop_only",                  "none",       False),  # 视觉压缩
    "structured_evidence":   ("pred",   "full_crop",                  "structured", False),  # §4.2 本文方法
    "region_caption":        ("pred",   "full_crop",                  "caption",    False),
    "answer_verifier":       ("pred",   "full_crop",                  "verify",     False),
    "lowres_full_highrescrop": ("pred", "lowres_full_highres_crop",   "none",       False),  # §5.2 本文方法
    "answer_reasoning_compressed": ("pred", "full_crop",              "structured", False),
    "direct_then_explain":   ("pred",   "full_crop",                  "none",       False),
    "vision_pruned":         ("pred",   "full_crop",                  "none",       False),
    "feature_pca":           ("pred",   "full_crop",                  "none",       False),
    "topk_crops":            ("pred",   "full_crop",                  "none",       False),
    "dsac":                  ("pred",   "full_crop",                  "structured", False),  # round3
}
```

**调用入口**：CLI `--mode <name>` → `apply_mode(args)` 把模式展开成 4 个独立变量 → 后续函数按这 4 个变量分发。

---

## 3. 三大核心方法的代码位置

### 3.1 结构化区域证据提示（§4.2）

**目的**：让模型按固定模板 `[Region] → [Visual Evidence] → [Reasoning] → [Answer]` 组织输出。

**实现位置**：`make_conv_prompt()` 的 `evidence_mode == "structured"` 分支（`llava/eval/model_cot_loader.py:515-524`）

```python
if args.evidence_mode == "structured":
    prompt = (
        question
        + "\nProvide your answer in EXACTLY this structure:\n"
        + "[Region] <where to look>\n"
        + "[Visual Evidence] <one sentence quoting what you see>\n"
        + "[Reasoning] <one or two sentences>\n"
        + "[Answer] <final answer>"
    )
```

**输出解析**：`parse_structured_output()` (`:670-698`) 用正则切片，落到 JSONL 的 `region_text / visual_evidence_text / reasoning_text` 字段。

### 3.2 区域级视觉输入压缩（§5.2）

**目的**：通过控制送入视觉编码器的图像组合，减少视觉 token 数。

**统一形式**（4 种算子 Φ）：

```
Φ_full(I,b) = {I}                              视觉 token = 576
Φ_pred(I,b) = {I, C(I,b,α)}                    视觉 token = 1152
Φ_crop(I,b) = {C(I,b,α)}                       视觉 token = 576
Φ_mix(I,b)  = {R(I,s_low), C(I,b,α)}           视觉 token = 1152
```

**实现位置**：`prepare_images()` (`:490-504`)

```python
def prepare_images(args, full_img, box):
    crop_img = square_crop(full_img, box, pad=pad) if box is not None else full_img.copy()
    if args.crop_mode == "full_only":
        return [full_img.copy()], crop_img
    if args.crop_mode == "crop_only":
        return [crop_img], crop_img
    if args.crop_mode == "lowres_full_highres_crop":
        return [lowres_full(full_img, args.lowres_size), crop_img], crop_img
    # 默认 full_crop
    return [full_img.copy(), crop_img], crop_img
```

辅助函数：`square_crop()` (`:467-481`) 做带 padding 的方形裁剪；`lowres_full()` (`:483-488`) 把全图下采样到 `lowres_size`（默认 112）。

### 3.3 边界框评分门控（§4.4.1，paper 16 已弃用）

> ⚠️ 已被 DSAC（Detection Score Aware Crop）取代，函数保留供历史 run 复现。

**实现位置**：`score_pred_bbox()` (`:334-402`)

```
score(b̂) = (λ₁·s_parse + λ₂·s_size + λ₃·s_focus) / Σλᵢ

s_parse = 1[b̂ 能解析到 [0,1]^4]
s_size  = exp(-(log ρ(b̂) - log ρ*)² / (2σ²))   # 面积先验高斯
s_focus = 1 - ‖c(b̂) - c_word‖₂/0.5             # 方位关键词聚焦
```

阈值化回退在 `select_box()` (`:439-456`)：分数低于 θ 时切换到 `_center_box()` 或 oracle。

---

## 4. 数据流：一条样本走完整个 pipeline

以 DocVQA 的 `question_id=791` 为例（"Did John have had any graduate level course in statistics?"）。

### Step 1: 输入数据

`data/benchmarks/small_benchmark_docvqa100.json` 里的一条：

```json
{
  "dataset": "docvqa",
  "question_id": 791,
  "image": [
    "cot/docvqa/gnnp0227_6.png",
    "cot/docvqa/gnnp0227_6.png###[122, 1631, 1204, 1897]"
  ],
  "conversations": [
    {"from": "human", "value": "<image>\nDid John have had any graduate level course in statistics? Please provide the bounding box ..."},
    {"from": "gpt", "value": "[0.159, 0.721, 0.637, 0.839]"},
    {"from": "human", "value": "<image>"},
    {"from": "gpt", "value": "yes"}
  ]
}
```

第二个 `image` 项的 `###[x1,y1,x2,y2]` 是 **像素坐标的 GT bbox**；conversation 里 gpt 第一轮输出的是 **归一化坐标**。

### Step 2: Detection 阶段

```bash
python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_docvqa100.json \
  --image-folder playground/data \
  --answers-file /tmp/demo_det.jsonl \
  --max-samples 1 \
  --precision bf16 \
  --record-latency --record-memory
```

输出 `/tmp/demo_det.jsonl`：

```json
{
  "question_id": 791,
  "prompt": "Did John have had any graduate level course in statistics? ...",
  "bbox_pred": [0.16, 0.72, 0.64, 0.84],
  "bbox_pred_raw": "[0.159, 0.721, 0.637, 0.839]",
  "bbox_parse_ok": true,
  "bbox_gt": [122, 1631, 1204, 1897],
  "height": 2200, "width": 1700,
  "latency_ms": 285.4,
  "peak_gpu_memory_mb": 14102.3,
  "answer_id": "...",
  "model_id": "VisCoT-7b-224"
}
```

### Step 3: Answer 阶段（举例 `structured_evidence` 模式）

```bash
python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_docvqa100.json \
  --image-folder playground/data \
  --detection-file /tmp/demo_det.jsonl \
  --answers-file /tmp/demo_ans.jsonl \
  --mode structured_evidence \
  --max-samples 1 \
  --precision bf16 \
  --record-latency --record-memory
```

**程序内部分发**：

1. `apply_mode(args)` 把 `--mode structured_evidence` 展开成：
   - `bbox_source = "pred"`
   - `crop_mode = "full_crop"`
   - `evidence_mode = "structured"`
   - `without_image = False`
2. `select_box()` 从 `/tmp/demo_det.jsonl` 读 pred bbox `[0.16, 0.72, 0.64, 0.84]`
3. `prepare_images()` 输出 `[原图, 裁剪图]` 两张图（`full_crop` 模式）
4. `make_conv_prompt()` 构造结构化提示词（见 §3.1）
5. `generate()` 让模型生成
6. `parse_structured_output()` 把结果切成 4 段
7. 落 JSONL

### Step 4: 输出 JSONL

`/tmp/demo_ans.jsonl` 单行包含 25+ 字段（精简版）：

```json
{
  "question_id": 791,
  "dataset": "docvqa",
  "mode": "structured_evidence",
  "bbox_source": "pred",
  "crop_mode": "full_crop",
  "evidence_mode": "structured",
  "prompt_version": "v2_structured_evidence",

  "gt_answer": "yes",
  "pred_answer": "[Answer] yes",
  "answer_extracted": "yes",
  "exact_match": 1.0,
  "contains_match": 1.0,

  "bbox_gt": [0.072, 0.741, 0.708, 0.862],
  "bbox_pred": [0.16, 0.72, 0.64, 0.84],
  "bbox_iou": 0.71,
  "bbox_correct_at_05": true,

  "region_text": "the 'graduate level courses in statistics' row",
  "visual_evidence_text": "the row shows 'Yes' checked",
  "reasoning_text": "since Yes is checked, John did take such a course",

  "latency_ms_total": 312.6,
  "latency_ms_generate": 312.4,
  "peak_gpu_memory_mb": 14272.1,
  "num_visual_inputs": 2,
  "num_visual_tokens_est": 1152,
  "reasoning_length_tokens": 18,

  "model_path": "checkpoints/VisCoT-7b-224",
  "metadata": {
    "detection_record": {...},
    "saved_crop": "/tmp/.../791_crop.png",
    "bbox_score": null
  }
}
```

### Step 5: 聚合 & 出图

```bash
python tools/thesis/aggregate_results.py \
  --raw-dir experiments/thesis_experiments/runs/<run>/raw \
  --output-dir experiments/thesis_experiments/runs/<run>/metrics

python tools/thesis/visualize_results.py \
  --metrics-dir experiments/thesis_experiments/runs/<run>/metrics \
  --figures-dir experiments/thesis_experiments/runs/<run>/figures
```

输出 `metrics/summary.md` 是按 mode 分组的指标表，`figures/*.png` 是柱状图、散点图、IoU 直方图等。

---

## 5. 怎么完整跑一次实验

### 5.1 准备工作（一次性）

```bash
# 1. 装依赖（参照 README）
pip install -e .

# 2. 下载模型权重到 checkpoints/VisCoT-7b-224
# 3. 下载图像数据到 playground/data/cot/
```

### 5.2 单 mode 端到端跑（CUB100 上的 `structured_evidence` 模式）

```bash
RUN=experiments/thesis_experiments/runs/$(date +%Y%m%d_%H%M%S)_demo
mkdir -p $RUN/{raw,metrics,figures}

# Stage 1: detection
python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_cub100.json \
  --image-folder playground/data \
  --answers-file $RUN/raw/detection.jsonl \
  --precision bf16 --record-latency --record-memory

# Stage 2: answer
python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_cub100.json \
  --image-folder playground/data \
  --detection-file $RUN/raw/detection.jsonl \
  --answers-file $RUN/raw/structured_evidence.jsonl \
  --mode structured_evidence \
  --precision bf16 --record-latency --record-memory

# Stage 3: 聚合
python tools/thesis/aggregate_results.py \
  --raw-dir $RUN/raw --output-dir $RUN/metrics
```

### 5.3 多 mode 批量跑（推荐）

直接用现成脚本：

```bash
# CUB100 全模式
bash experiments/thesis_experiments/run_thesis_experiments.sh \
  --stage formal --run-name cub100_demo

# 跨数据集 100（TextVQA + GQA + Visual7W）
bash experiments/thesis_experiments/run_cross_dataset_region100.sh \
  --stage formal --run-name cross_demo
```

### 5.4 快速验证（30 秒，2 个样本）

用于现场演示——只跑 2 个样本走完整 pipeline：

```bash
python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_cub100.json \
  --image-folder playground/data \
  --answers-file /tmp/demo_det.jsonl \
  --max-samples 2 --precision bf16

python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file data/benchmarks/small_benchmark_cub100.json \
  --image-folder playground/data \
  --detection-file /tmp/demo_det.jsonl \
  --answers-file /tmp/demo_ans.jsonl \
  --mode structured_evidence \
  --max-samples 2 --precision bf16

cat /tmp/demo_ans.jsonl | python -m json.tool
```

---

## 6. 实验结果（提前跑完，存档可查）

### 6.1 CUB100 主表（路径：`runs/20260508_013500_cub100/metrics/summary.md`）

| Mode | EM | CM | Latency (ms) | 备注 |
|---|---:|---:|---:|---|
| full | 0.00 | 0.70 | 692 | 仅整图，无 CoT |
| pred_bbox | 0.51 | 0.80 | 354 | **VisCoT 原版** |
| oracle_bbox | 0.36 | 0.82 | 437 | GT bbox |
| random_bbox | 0.60 | 0.82 | 311 | 随机框（CUB 中心偏置警示） |
| center_bbox | 0.40 | 0.83 | 414 | 中心框 |
| woimg | 0.00 | 0.47 | 515 | 不看图（语言先验） |
| **crop_only** | 0.66 | 0.69 | **212** | **延迟最低** |
| **structured_evidence** | **0.73** | 0.73 | 265 | **EM 最高** |
| lowres_full_highrescrop | 0.72 | 0.79 | 278 | 接近 pred_bbox 更快 |

### 6.2 跨数据集 100（路径：`runs/20260516_cross_region100_public3/metrics/summary.md`）

| Mode | CM | EM | Latency (ms) | 备注 |
|---|---:|---:|---:|---|
| woimg | 0.08 | 0.01 | 927 | 跨域语言先验失效 |
| pred_bbox | 0.41 | 0.26 | 303 | 基线 |
| **structured_evidence** | **0.43** | **0.32** | 414 | 跨域同样最优 |
| lowres_full_highrescrop | 0.37 | 0.28 | 437 | |
| **crop_only** | 0.32 | 0.25 | **212** | 延迟最低 |

### 6.3 累计推理规模

- 4 个数据集 × 16 种模式（其中 9 种主力）× 3 个随机种子
- 累计 ~5000 次推理
- 所有原始 JSONL 保存在 `experiments/thesis_experiments/runs/<时间戳>/raw/`

---

## 7. JSONL 输出 schema 完整字段（25+）

```
# 标识
question_id, dataset, image, mode, prompt_version,
bbox_source, crop_mode, evidence_mode, reasoning_compression, answer_policy,

# 答案
prompt, gt_answer, pred_answer, answer_extracted,
normalized_gt, normalized_pred, exact_match, contains_match,

# 边界框
bbox_gt, bbox_pred_raw, bbox_pred, bbox_parse_ok,
bbox_iou, bbox_correct_at_05,

# 视觉输入
crop_ratio, lowres_size, crop_size, crop_pad,
num_visual_inputs, num_visual_tokens_est,

# 结构化输出
region_text, visual_evidence_text, reasoning_text,
compressed_reasoning,
raw_reasoning_tokens, compressed_reasoning_tokens, reasoning_compression_ratio,
reasoning_length_tokens, answer_length_tokens,

# 性能
latency_ms_total, latency_ms_preprocess, latency_ms_generate,
stage1_latency_ms, stage2_latency_ms,
peak_gpu_memory_mb,

# 调试
error_tag, prompt_text, model_path, model_id, answer_id, timestamp,
metadata: {detection_record, saved_crop, bbox_score, region_caption, ...}
```

---

## 8. 关键代码位置速查表

| 想找 | 文件:行号 |
|---|---|
| 16 种模式定义 | `llava/eval/model_cot_loader.py:101` |
| mode 字符串分发 | `llava/eval/model_cot_loader.py:890` (`apply_mode`) |
| 结构化提示词构造 | `llava/eval/model_cot_loader.py:506` (`make_conv_prompt`) |
| 4 种视觉算子 | `llava/eval/model_cot_loader.py:490` (`prepare_images`) |
| bbox 评分门控 | `llava/eval/model_cot_loader.py:334` (`score_pred_bbox`, 已废弃) |
| 结构化输出解析 | `llava/eval/model_cot_loader.py:670` (`parse_structured_output`) |
| 量化加载 | `llava/model/builder.py:35` (`load_pretrained_model`) |
| detection 主循环 | `llava/eval/model_cot_det_loader.py:188` (`eval_model`) |
| answer 主循环 | `llava/eval/model_cot_loader.py:913` (`eval_model`) |
| CLI 参数定义 | `llava/eval/model_cot_loader.py:1215` (`build_parser`) |

---

## 9. 答辩可能被问到的代码层面问题

**Q1: 为什么把 detection 和 answer 拆成两阶段？**

A: 三个理由：
1. **可独立替换 detection 源**：通过 `--bbox-source {pred,oracle,random,center}` 切换，等价于换掉整个 detection 阶段。
2. **可独立调优**：detection 失败的样本能单独定位（`bbox_parse_ok=false`）。
3. **节省计算**：同一份 detection.jsonl 可被多种 answer mode 共用，避免重复跑第一阶段。

**Q2: 16 种模式是不是过度设计？**

A: 每种模式对应一个明确的对比维度：
- `full / pred_bbox / crop_only / lowres_full_highrescrop` → **视觉输入压缩**消融
- `pred_bbox / oracle_bbox / random_bbox / center_bbox` → **bbox 质量来源**消融
- `pred_bbox / structured_evidence / region_caption / answer_verifier` → **输出结构**消融
- `woimg` → **图像必要性**对照
- `vision_pruned / feature_pca / topk_crops` → **可行性验证**接口

是消融实验需求驱动，不是空架子。

**Q3: 怎么保证 JSONL schema 25+ 字段都齐全？**

A: `eval_model()` (`:913-1213`) 主循环里每跑一条样本都填一个 `dict`，缺字段填 `NA`，落盘前用统一序列化函数写出。聚合脚本 `aggregate_results.py` 用 pandas 读，遇缺失字段自动补 NaN。

**Q4: 代码改了 1043 行怎么保证不破坏原项目？**

A: 改造遵循**向后兼容**——`--mode full` 退化到原版行为（整图 + 无 bbox + 无结构化）。所有新参数都有默认值。原版的 `model_cot_loader.py` 200 行函数签名完全保留，只是把位置参数扩展为关键字参数。

**Q5: 现场跑一个新的 mode 能跑通吗？**

A: 可以。把 `--mode structured_evidence` 换成 `crop_only` / `lowres_full_highrescrop` 等任意一个 MODE_PRESETS 里的 key，30 秒内（2 样本）出结果。同一份代码、同一个 CLI、同一份 JSONL schema。

---

## 10. 相关文档

| 文档 | 用途 |
|---|---|
| [`DEFENSE_SCRIPT.md`](DEFENSE_SCRIPT.md) | 8 分钟答辩口语化讲稿 |
| [`THESIS_CHANGES.md`](THESIS_CHANGES.md) | 相对上游 diff 的精简说明 |
| [`CLAUDE.md`](CLAUDE.md) | 完整改造记录（含历史） |
| [`STATUS.md`](STATUS.md) | 进度快照（最新状态） |
| [`README.md`](README.md) | 上游 Visual-CoT 原 README（保留） |
| [`images/benchmarks/README.md`](images/benchmarks/README.md) | 展示图说明 |

---

> **最后**：本项目的核心交付**不是模型**，而是**一个完整、可复现、多模式可切换的视觉语言模型推理实验框架**。1043 行核心代码 + 30+ CLI 参数 + 25+ JSONL 字段 + 5000 次累计推理 = 整套工程化产物。
