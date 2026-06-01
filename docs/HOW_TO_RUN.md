# 代码怎么运行 / 结果在哪里（HOW_TO_RUN）

> 答辩准备速查。本文回答两个问题：(1) 怎么把代码跑起来；(2) 跑完后产物在哪、怎么看。

---

## 0. TL;DR（一分钟版本）

```bash
# 一阶段：定位 bbox
python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub100.json \
  --image-folder downloads/ \
  --answers-file runs/demo/raw/detection.jsonl \
  --temperature 0 --conv-mode vicuna_v1 \
  --max-samples 100 --record-latency --record-memory

# 二阶段：答题（结构化提示）
python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub100.json \
  --detection-file runs/demo/raw/detection.jsonl \
  --image-folder downloads/ \
  --answers-file runs/demo/raw/answer_se.jsonl \
  --mode structured_evidence \
  --precision bf16 --record-latency --record-memory

# 聚合 + 图表
python tools/thesis/aggregate_results.py --run-dir runs/demo
python tools/thesis/visualize_results.py  --run-dir runs/demo
```

跑完看三个地方：
- `runs/demo/raw/*.jsonl`：每条样本明细（25+ 字段）
- `runs/demo/metrics/summary.md`：所有 mode 一张主表
- `runs/demo/figures/*.png`：精度/延迟/显存柱图与 Pareto 散点

---

## 1. 环境前置

| 项 | 要求 |
|---|---|
| GPU | 单卡 ≥ 24 GB（RTX 3090 验证过） |
| 精度 | `--precision bf16`（**不要走 4-bit**，CLAUDE.md §2.1 记的 dtype bug 还没修） |
| 模型 | `checkpoints/VisCoT-7b-224`（已下） |
| Python | 仓库根 conda 环境，`pip freeze > env/pip_freeze.txt` 已固化 |
| 字体 | `fonts-noto-cjk`（中文图渲染需要） |

每个 run 跑前**先记录环境**（answer.sh 第一步会做）：

```bash
mkdir -p runs/demo/env
nvidia-smi > runs/demo/env/nvidia_smi.txt
pip freeze > runs/demo/env/pip_freeze.txt
git rev-parse HEAD > runs/demo/env/git_info.txt
uname -a > runs/demo/env/system_info.txt
```

---

## 2. 两阶段流水线总览

```
┌──────────────── 一阶段：定位 ─────────────────┐
│ python -m llava.eval.model_cot_det_loader   │
│   inputs:  question.jsonl + 图像目录         │
│   outputs: detection.jsonl                   │
└──────────────────────────────────────────────┘
                    │ question_id 对齐
                    ▼
┌──────────────── 二阶段：答题 ─────────────────┐
│ python -m llava.eval.model_cot_loader       │
│   inputs:  question.jsonl + detection.jsonl  │
│   outputs: answer_<mode>.jsonl               │
│   按 --mode 切 16 种对照组之一               │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────── 聚合 + 可视化 ────────────────┐
│ tools/thesis/aggregate_results.py           │
│ tools/thesis/visualize_results.py            │
│   outputs: metrics/*.{csv,md,json}           │
│            figures/*.{png,pdf}               │
│            paper_materials/*.md              │
└──────────────────────────────────────────────┘
```

---

## 3. 标准 run 目录结构（七个固定子目录）

每个新 run 都要建在 `experiments/thesis_experiments/runs/<时间戳>_<标签>/`，七个子目录是不变量：

```
runs/<时间戳>_<标签>/
├── configs/         ← yaml + benchmark 数据快照
├── commands/        ← run_commands.sh（实际跑的命令记录）
├── env/             ← 环境快照：nvidia_smi / pip_freeze / git_info
├── logs/            ← 每个 mode 一个 .log（stdout+stderr）
├── raw/             ← detection.jsonl + answer_<mode>.jsonl（原始产物）
├── metrics/         ← 聚合表格：summary.{csv,md,json} 等
├── figures/         ← 图表：精度柱图 / Pareto 散点 / IoU 直方图 等
└── paper_materials/ ← 论文素材：实验段落 + 表 + 图说明（md）
```

举例（`20260516_cross_region100_public3` 实跑产物）：

```
configs/        cross_region100.yaml + small_benchmark_cross_region100.json
                small_benchmark_cross_region100_det.jsonl + _stats.{json,md}
commands/       run_commands.sh
env/            git_info.txt nvidia_smi.txt pip_freeze.txt system_info.txt
raw/            detection.jsonl
                answer_full.jsonl              answer_pred_bbox.jsonl
                answer_crop_only.jsonl         answer_woimg.jsonl
                answer_lowres_full_highrescrop.jsonl
                answer_random_bbox.jsonl       answer_center_bbox.jsonl
                answer_structured_evidence.jsonl
metrics/        summary.{csv,md,json}                ← 主表（每个 mode 一行）
                summary_by_dataset.{csv,md}          ← 按数据集分层
                summary_cross_overall.{csv,md}       ← 跨域池化
                bbox_answer_cross_table.csv
                compression_efficiency.csv
                reasoning_optimization.csv
                error_cases.csv
figures/        accuracy_bar.{png,pdf}               ← EM/CM 柱图
                latency_bar.{png,pdf}                ← 延迟柱图
                memory_bar.{png,pdf}                 ← 显存柱图
                accuracy_latency_scatter.{png,pdf}   ← Pareto
                bbox_iou_hist.{png,pdf}              ← IoU 分布
                crop_ratio_hist.{png,pdf}            ← 裁剪面积比
                compression_accuracy_tradeoff.{png,pdf}
                case_visualizations/                 ← 单样本可视化
paper_materials/ experiment_section_draft.md
                method_section_addition.md
                limitation_and_future_work.md
                paper_tables.md   paper_figures.md
                cross_dataset_experiment_section.md
                cross_dataset_tables.md   cross_dataset_figures.md
```

---

## 4. 标准跑法：一条命令搞定一个完整 run

### 4.1 用现成驱动脚本（推荐）

每个新 run 都附一个 `commands/run_commands.sh`，里面定义 `run_detection` + `run_answer` 两个函数：

```bash
RUN_DIR=experiments/thesis_experiments/runs/20260528_demo
mkdir -p "$RUN_DIR"/{configs,commands,env,logs,raw,metrics,figures,paper_materials}

PYTHON_BIN=python
MODEL_PATH=checkpoints/VisCoT-7b-224
IMAGE_FOLDER=downloads
CONV_MODE=vicuna_v1
TARGET_SAMPLES=100

DET_QUESTION_FILE=small_benchmark_cross_region100_det.jsonl
ANS_QUESTION_FILE=small_benchmark_cross_region100.json

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --question-file "$ANS_QUESTION_FILE"
  --image-folder "$IMAGE_FOLDER"
  --conv-mode "$CONV_MODE"
  --temperature 0.2 --top_p 0.95
  --max-samples "$TARGET_SAMPLES"
  --precision bf16
  --record-latency --record-memory
)

run_detection() {
  "$PYTHON_BIN" -m llava.eval.model_cot_det_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$DET_QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$RUN_DIR/raw/detection.jsonl" \
    --temperature 0 --conv-mode "$CONV_MODE" \
    --max-samples "$TARGET_SAMPLES" --save-jsonl \
    --record-latency --record-memory \
    > "$RUN_DIR/logs/detection.log" 2>&1
}

run_answer() {
  local mode="$1"; shift
  "$PYTHON_BIN" -m llava.eval.model_cot_loader \
    "${COMMON_ARGS[@]}" \
    --detection-file "$RUN_DIR/raw/detection.jsonl" \
    --mode "$mode" \
    --answers-file "$RUN_DIR/raw/answer_${mode}.jsonl" \
    "$@" \
    > "$RUN_DIR/logs/answer_${mode}.log" 2>&1
}

# 实际调用
run_detection
for m in full pred_bbox crop_only lowres_full_highrescrop \
         structured_evidence random_bbox center_bbox woimg; do
  run_answer "$m"
done
```

### 4.2 看现成例子

最完整的成品 run 在：

- `experiments/thesis_experiments/runs/20260516_cross_region100_public3/commands/run_commands.sh`（跨数据集 8 mode）
- `experiments/thesis_experiments/runs/20260522_round2_extensions/scripts/run_phase2_sweeps.sh`（lowres / padding 灵敏性扫描）
- `experiments/thesis_experiments/runs/20260522_dsac_v7/scripts/run_dsac.sh`（DSAC 8 配置）
- `experiments/thesis_experiments/runs/20260523_cross_domain_dsac/scripts/run_all_domains.sh`（跨 4 域 × 5 配置 = 20 runs）

这些脚本都把"建目录 → 落 env → 跑 detection → 跑 8/12 个 mode → 跑聚合"串成一条命令，可以照搬改时间戳。

---

## 5. 跑完去看哪几个文件

按"急到不急"排序：

### 5.1 一眼看全局（30 秒）

**`runs/<run>/metrics/summary.md`**：所有 mode 一张表，包含 EM / CM / 延迟 / 显存 / 视觉输入数。例如：

| mode | N | EM | CM | 延迟 ms | 显存 MB | 视觉输入数 |
|---|---|---:|---:|---:|---:|---:|
| structured_evidence | 100 | 0.3163 | 0.4286 | 414 | 14267 | 2 |
| pred_bbox | 100 | 0.2626 | 0.4141 | 303 | 14208 | 2 |
| crop_only | 100 | 0.2525 | 0.3232 | 212 | 13884 | 1 |
| full | 100 | 0.0909 | 0.3737 | 411 | 13882 | 1 |
| woimg | 100 | 0.0101 | 0.0808 | 927 | 13627 | 0 |

### 5.2 看图（答辩 PPT 直接用）

**`runs/<run>/figures/`**：

| 文件 | 看什么 |
|---|---|
| `accuracy_bar.png` | 各 mode 的 EM / CM 柱图 |
| `latency_bar.png` | 各 mode 平均延迟 |
| `memory_bar.png` | 各 mode 峰值显存 |
| `accuracy_latency_scatter.png` | **Pareto 前沿**：精度 vs 延迟 |
| `bbox_iou_hist.png` | 预测框 IoU 分布 |
| `compression_accuracy_tradeoff.png` | 输入压缩 vs 答案质量 |
| `case_visualizations/` | 单样本图：原图 + 预测框 + GT 框 + 答案对比 |

中文版图（论文 v6+ 用）在：`Obsidian/assets/round2/paper_figures_zh/` 与 `experiments/thesis_experiments/runs/20260523_cross_domain_dsac/figures/`。

### 5.3 看每条样本明细（错例分析）

**`runs/<run>/raw/answer_<mode>.jsonl`**：每行一个样本，25+ 字段。常用查询：

```bash
# 看一条样本完整记录
head -1 runs/<run>/raw/answer_structured_evidence.jsonl | python -m json.tool

# 找答错的样本（contains_match=false 的前 5 条）
jq -c 'select(.contains_match==false) | {qid:.question_id, pred:.pred_answer, gt:.gt_answer, iou:.bbox_iou}' \
  runs/<run>/raw/answer_pred_bbox.jsonl | head -5

# 找延迟最高的 5 条
jq -s 'sort_by(-.latency_ms_total) | .[0:5] | .[] | {qid:.question_id, ms:.latency_ms_total}' \
  runs/<run>/raw/answer_pred_bbox.jsonl
```

每条 JSONL 关键字段（详见 `EVAL_LOADERS.md` §3.6）：

- 答题：`exact_match / contains_match / pred_answer / gt_answer`
- 定位：`bbox_iou / bbox_correct_at_05 / bbox_pred / bbox_gt`
- 效率：`latency_ms_total / latency_ms_generate / peak_gpu_memory_mb / num_visual_inputs`
- 结构化产出：`region_text / visual_evidence_text / reasoning_text`
- 算法元信息：`metadata.bbox_score`（评分门控）/ `metadata.composite_gate`（DSAC）

### 5.4 看论文素材（写答辩稿）

**`runs/<run>/paper_materials/`**：聚合脚本已经把数据格式化成可粘到论文/PPT 的 markdown：

| 文件 | 用途 |
|---|---|
| `experiment_section_draft.md` | 论文实验章节草稿（Markdown 版） |
| `method_section_addition.md` | 方法章节补充段 |
| `limitation_and_future_work.md` | 局限性 + 未来工作 |
| `paper_tables.md` | 主表 markdown |
| `paper_figures.md` | 图表说明 |

跨域 run 还多两份：`cross_dataset_experiment_section.md / cross_dataset_tables.md / cross_dataset_figures.md`。

### 5.5 看出错时的日志

**`runs/<run>/logs/`**：每个 mode 一个 `.log`，包含 tqdm 进度条 + 异常栈：

```bash
tail -50 runs/<run>/logs/answer_structured_evidence.log
grep -i error runs/<run>/logs/*.log
```

---

## 6. 三种最常用的运行场景

### 6.1 跑一个新 mode 做消融

```bash
# 假设 detection 已经跑过，只补一个新 mode
run_answer center_bbox
# 然后单独重新聚合
python tools/thesis/aggregate_results.py --run-dir "$RUN_DIR"
```

### 6.2 启用本文新算法（边界框评分门控）

```bash
run_answer pred_bbox \
  --bbox-scoring \
  --bbox-score-threshold 0.5 \
  --bbox-score-lambdas "1.0,1.0,0.5" \
  --bbox-score-fallback oracle
# JSONL 的 metadata.bbox_score 会带 {score, components, triggered, used_fallback}
```

### 6.3 启用 DSAC 双端自适应控制器

```bash
run_answer dsac \
  --adaptive-gate \
  --adaptive-alpha 0.5 \
  --adaptive-theta-low 0.6 \
  --adaptive-theta-high 0.7 \
  --adaptive-fallback oracle
# JSONL 的 metadata.composite_gate 会带 {alpha, theta_low/high, decision, scs_flags, second_pass}
```

聚合工具会自动识别这两个新字段并生成对应的消融表。

---

## 7. 跑完后聚合 + 图表

```bash
# 主聚合：summary.{csv,md,json} + 各种 csv
python tools/thesis/aggregate_results.py --run-dir "$RUN_DIR"

# 主可视化：figures/*.{png,pdf}
python tools/thesis/visualize_results.py --run-dir "$RUN_DIR"

# 论文素材
python tools/thesis/generate_paper_materials.py --run-dir "$RUN_DIR"
```

跨数据集额外两步：

```bash
python tools/thesis/aggregate_cross_dataset_results.py   --run-dir "$RUN_DIR"
python tools/thesis/generate_cross_dataset_materials.py  --run-dir "$RUN_DIR"
```

Round-2 / Round-3 专用聚合：

```bash
python tools/thesis/round2/phase1_analysis.py  --run-dir "$RUN_DIR"  # A1-A6 零 GPU 分析
python tools/thesis/round2/phase2_analysis.py  --run-dir "$RUN_DIR"  # B1/B2 灵敏性
python tools/thesis/round2/phase3_analysis.py  --run-dir "$RUN_DIR"  # C1/C2 稳定性 + 评分门控
python tools/thesis/round3/dsac_analysis.py    --run-dir "$RUN_DIR"  # DSAC 12 配置
python tools/thesis/round3/mcnemar_dsac_vs_se.py --run-dir "$RUN_DIR"  # 配对显著性
```

---

## 8. 已知问题 / 踩坑提醒

- **不要走 `--load-4bit`**：detection 阶段会 `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`。一律 `--precision bf16`。
- **`question_id` 必须有**：原版默认 0，CLAUDE.md §2.2 修过。但如果你自造 question 文件没填 `question_id` 字段，detection 与 answer 两阶段对不上。
- **CUB 路径兜底**：`cot/cub/...` 自动映射到 `downloads/cub/CUB_200_2011/images/...`，其它数据集要手动把 `img_path` 写对。
- **JSONL 字段不重排**：新增字段（DSAC、bbox_score、剪枝报告）一律往 `metadata` 尾部加。聚合脚本依赖这个契约。
- **跨数据集 SCS 恒为 0.5**：模型只稳定产出 [Reasoning] + [Answer] 段，[Region] / [VE] 段不出。这是模型限制，不是数据集限制。论文 §4.6 已写为"诚实警示"。
- **DSAC 默认配置在跨数据集上 fallback 永不触发**：composite 最小 0.45 ≥ θ_low=0.3。要让控制器分流必须 `--adaptive-theta-low 0.5` 或更高。

---

## 9. 一句话结论

> **"跑代码"= `det_loader.py` 出 detection.jsonl，再用 `cot_loader.py` 按 `--mode` 出 answer_*.jsonl；"看结果"= 先看 `metrics/summary.md` 一张表，再看 `figures/*.png` 几张图，再看 `paper_materials/*.md` 拿现成段落。**
