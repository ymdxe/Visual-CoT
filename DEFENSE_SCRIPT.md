# 毕业设计验收答辩讲解稿

> **课题**：基于模态混合的视觉语言模型推理增强方法的设计与实现
> **作者**：张恒 ｜ 指导教师：马安香
> **日期**：2026-05-28
> **答辩时长**：7-10 分钟（建议 8 分钟主体 + 2 分钟缓冲）
> **本稿定位**：**讲解程序运行**，不是讲论文方法本身

---

## 0. 验收要求回顾

老师给出的三条要求：

1. **每个人程序都提前训练好，不要在验收时进行训练** → 本课题是**推理阶段增强**，不训练大模型，直接加载 VisCoT-7B-224 预训练权重。
2. **按四个步骤验收讲解**：项目背景 → 开发目标 → 核心技术 → 运行程序出结果。
3. **每人 7-10 分钟**。

应对策略：
- 所有实验结果**提前跑完**，存放在 `experiments/thesis_experiments/runs/` 下。
- 现场只演示 **1-2 个样本** 走完整 pipeline，证明程序确实能跑。
- 重心放在**程序的工程结构**与**运行流程**上，方法理论一句话带过。

---

## 1. 时间分配

| 步骤 | 时长 | 关键交付 |
|---|---:|---|
| ① 项目背景 | 1.5 min | 一句话讲清做什么程序 |
| ② 开发目标 | 1.0 min | 程序要解决的工程问题 |
| ③ 核心技术 | 2.5 min | **重点：代码架构 + 关键改造** |
| ④ 运行程序出结果 | 2.5 min | **重点：现场跑通 + 结果展示** |
| 缓冲 / 提问 | 0.5-2 min | 应对老师追问 |
| **合计** | **8 min** | |

---

## 2. 第一步：项目背景（1.5 min）

### 2.1 讲稿

> "各位老师好，我的毕设课题是《基于模态混合的视觉语言模型推理增强方法的设计与实现》，指导老师是马安香老师。
>
> **简单介绍背景**：当前的视觉语言模型（比如 LLaVA、GPT-4V）在回答需要看图像**局部细节**的问题时容易出错——直接看整图会忽略关键小区域。NeurIPS 2024 提出的 **Visual-CoT 项目**用'先定位、再回答'的两阶段思路缓解了这个问题，但它的开源代码**只支持单一推理流程**，**没法做不同推理策略的对比实验**。
>
> **我做的程序**就是基于 Visual-CoT 上游仓库做工程化扩展：在它的推理代码基础上，构建一个**支持多种推理模式统一切换、可量化对比的实验框架**。所有改造都在**推理阶段**，不重新训练模型。"

### 2.2 PPT 建议（1 页）

- 标题："基于 Visual-CoT 的多模式推理实验框架"
- 一句话定位：**零训练 · 多模式 · 工程化**
- 上游 logo + 我的派生仓库 logo

---

## 3. 第二步：开发目标（1 min）

### 3.1 讲稿

> "针对'原项目代码无法做对比实验'这个工程痛点，我设定了四个开发目标：
>
> 1. **多模式推理框架**——在统一的代码入口里支持 **16 种推理模式**的一键切换（整图 / 整图+裁剪 / 仅裁剪 / 低分整图+高分裁剪 / 结构化证据 / 推理压缩 等）；
> 2. **双阶段 Pipeline 解耦**——detection 阶段先输出 bbox JSONL，answer 阶段独立消费，两阶段可以独立替换或测试；
> 3. **统一的 CLI 和 JSONL 接口**——30+ 命令行参数控制每个开关，每条样本输出 25+ 字段的 JSONL，方便后续聚合分析；
> 4. **实验目录标准化**——每次实验自动落到 `runs/<时间戳>/` 下，包含 `configs / commands / env / raw / metrics / figures / paper_materials` **七个标准子目录**，保证可复现。
>
> 最终交付：**1043 行**的核心改造代码（原项目 200 行）、**3 套数据集**上的完整对比实验、**~5000 次推理**的累计结果。"

---

## 4. 第三步：核心技术（2.5 min）

### 4.1 基础技术栈（30s）

> "底层栈：**PyTorch + Transformers + bitsandbytes**，模型是 **VisCoT-7B-224**（LLaVA 架构 + LLaMA-2-7B 主干 + CLIP 视觉编码器），bf16 精度，单卡 24GB 显存可跑。"

### 4.2 关键文件改造（1.5 min，**重点**）

打开 PPT 上的"代码改造一览表"，对着讲：

| 文件 | 原版 | 改造后 |
|---|---|---|
| `llava/model/builder.py` | 仅 fp16 加载 | 新增 `--load-4bit / --load-8bit / --precision` 三种量化路径 |
| `llava/eval/model_cot_det_loader.py` | 仅打印 bbox 文本 | 输出含 `bbox_pred / latency_ms / peak_gpu_mem` 的 JSONL |
| `llava/eval/model_cot_loader.py` | **200 行 → 1043 行** | **核心扩展**：MODE_PRESETS 字典 + 30+ CLI 参数 + 25+ JSONL 字段 |

**重点讲 `model_cot_loader.py` 的三个工程设计**：

> "**第一**，**`MODE_PRESETS` 字典**统一管理 16 种推理模式。每个模式是一个四元组 `(bbox_source, crop_mode, evidence_mode, without_image)`——比如 `pred_bbox` 模式是 `('pred', 'full_crop', 'none', False)`，`structured_evidence` 模式是 `('pred', 'full_crop', 'structured', False)`。命令行加 `--mode xxx` 就能一键切换，**不需要改代码**。
>
> **第二**，**双阶段 pipeline**。detection 阶段（`model_cot_det_loader.py`）扫一遍数据集，把每张图的预测 bbox 落到 JSONL。answer 阶段（`model_cot_loader.py`）通过 `--detection-file` 读这个 JSONL 复用结果——**两阶段彻底解耦**，可以独立调试，可以用不同 detection 源（pred / oracle / random / center）做对比。
>
> **第三**，**统一 JSONL schema**。每条样本输出 25+ 字段，包括 `mode / crop_mode / evidence_mode / bbox_iou / exact_match / contains_match / latency_ms_total / peak_gpu_memory_mb / num_visual_tokens_est / metadata` 等，**所有消融实验共用一个 schema**，后续 `tools/thesis/` 下的聚合脚本可以直接消费。"

### 4.3 工具链（30s）

> "围绕实验框架，还写了配套工具链：
> - `tools/thesis/aggregate_results.py`——聚合 raw JSONL 到 metrics 表；
> - `tools/thesis/visualize_results.py`——生成柱状图、雷达图等；
> - `tools/thesis/round2/render_paper_figures_zh.py`——渲染论文中文版图；
> - `experiments/scripts/run_*.sh`——一键运行 pilot / 诊断 / 正式评测。"

### 4.4 PPT 建议（2 页）

- **页 1**：代码改造对比表（原版 vs 改造后）
- **页 2**：双阶段 pipeline 流程图（detection JSONL → answer 阶段消费）

---

## 5. 第四步：运行程序出结果（2.5 min）

### 5.1 现场演示（1.5 min）

**演示策略**：只跑 **2 个样本**走完整 pipeline，30 秒内出结果。

**演示步骤一：detection 阶段**

```bash
cd /root/code/Visual-CoT

python -m llava.eval.model_cot_det_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub30.json \
  --image-folder downloads/cub \
  --answers-file /tmp/demo_det.jsonl \
  --max-samples 2 \
  --precision bf16 \
  --record-latency --record-memory
```

**讲稿**：
> "第一步是 detection 阶段。命令行加载 7B 模型，对 2 个样本预测边界框，结果落到 `/tmp/demo_det.jsonl`。可以看到每条 JSONL 包含 `bbox_pred / latency_ms / peak_gpu_memory_mb` 等字段。"

**演示步骤二：answer 阶段（结构化证据模式）**

```bash
python -m llava.eval.model_cot_loader \
  --model-path checkpoints/VisCoT-7b-224 \
  --question-file small_benchmark_cub30.json \
  --image-folder downloads/cub \
  --detection-file /tmp/demo_det.jsonl \
  --answers-file /tmp/demo_ans.jsonl \
  --mode structured_evidence \
  --max-samples 2 \
  --precision bf16 \
  --record-latency --record-memory
```

**讲稿**：
> "第二步是 answer 阶段。`--mode structured_evidence` 启用结构化输出，`--detection-file` 复用刚才 detection 阶段的 JSONL。模型按 `[Region][Visual Evidence][Reasoning][Answer]` 四段输出，结果落到 `/tmp/demo_ans.jsonl`。
>
> **关键看点**：只要改 `--mode` 参数，就能切换到 `pred_bbox` / `crop_only` / `lowres_full_highrescrop` 等任意一种模式，**整套实验框架完全统一**。"

**演示步骤三：打开 JSONL 文件**

```bash
cat /tmp/demo_ans.jsonl | python -m json.tool
```

> "可以看到 25+ 字段都齐全，包括 `pred_answer / bbox_iou / latency_ms_total / peak_gpu_memory_mb / metadata` 等。"

### 5.2 提前生成的对比结果（1 min）

打开 PPT 上提前准备好的图表：

#### ① CUB100 主表

| Mode | EM | CM | Latency |
|---|---:|---:|---:|
| pred_bbox（Visual-CoT 原版流程） | 0.51 | 0.80 | 354 ms |
| **structured_evidence** | **0.73** | 0.73 | 265 ms |
| lowres+highres | 0.72 | 0.79 | 278 ms |
| **crop_only** | 0.66 | 0.69 | **212 ms** |

> "**程序运行的实际产出**：在 CUB100 上跑了 9 种模式 × 100 样本，结构化证据模式 EM 达到 0.73，crop_only 模式延迟降到 212 毫秒。"

#### ② 跨数据集 100（TextVQA + GQA + Visual7W）

| Mode | CM | EM | Latency |
|---|---:|---:|---:|
| pred_bbox | 0.41 | 0.26 | 303 ms |
| **structured_evidence** | **0.43** | **0.32** | 414 ms |
| crop_only | 0.32 | 0.25 | **212 ms** |

> "为了验证程序在不同数据集上都能跑通，又在 TextVQA、GQA、Visual7W 上做了跨域 100 样本验证，结构化证据模式同样最优。"

#### ③ 累计实验规模

> "整个程序框架累计跑了大约 **5000 次推理**，覆盖 4 个数据集、16 种推理模式、3 个随机种子，所有原始 JSONL 都保存在 `experiments/thesis_experiments/runs/` 下。"

### 5.3 一句话收尾

> "总结一下，我交付的不是一个'训练好的模型'，而是**一个完整的、能跑多模式对比的视觉语言模型推理实验框架**——包含 1043 行核心代码、30+ CLI 参数、25+ JSONL 字段，以及配套的聚合可视化工具链。"

---

## 6. 演示前准备清单

### 6.1 硬件 / 环境

- [ ] 笔记本充满电 + 带电源线
- [ ] 提前 10 分钟到场，跑 `nvidia-smi` 确认显存可用
- [ ] 终端字号调到 18+，关闭屏保 / 系统通知
- [ ] 模型预先 warm-up 一次（避免首次加载慢）

### 6.2 文件 / 命令

- [ ] PPT 已打开，4 步骤页面准备就绪
- [ ] 演示命令两条已粘贴在终端
- [ ] `/tmp/demo_det.jsonl` 和 `/tmp/demo_ans.jsonl` 路径无残留
- [ ] CUB100、跨数据集结果表格 PDF 已打开

### 6.3 兜底预案

| 风险 | 兜底 |
|---|---|
| 模型加载失败 | 直接打开 `runs/.../raw/*.jsonl` 展示已有结果 |
| 现场跑超时 | 切到提前录好的演示视频或截图 |
| 老师问"再多跑几个" | "完整 100 样本结果在这里"，展开提前的表 |
| 时间超 10 分钟 | 跳过 5.2 ②③，只讲 ① CUB100 主表 |

---

## 7. 可能的提问及应对

### Q1: "你的程序为什么不训练？工作量怎么体现？"

**A**：本课题定位是**推理阶段增强**，目标是验证**零训练成本**也能扩展原项目的能力。工作量主要体现在**代码工程**——把原项目 200 行的单一推理流程扩展到 1043 行的多模式实验框架，新增 30+ CLI 参数、25+ JSONL 字段、16 种推理模式、双阶段 pipeline 解耦、配套工具链 20+ 脚本。同时跑了 ~5000 次推理实验做完整消融。

### Q2: "为什么要支持 16 种模式？是不是过度设计？"

**A**：每种模式都对应一个明确的对比维度——比如 `full / pred_bbox / crop_only / lowres_full_highrescrop` 是为了对比**不同视觉输入压缩策略**的精度-延迟权衡；`structured_evidence / region_caption / answer_verifier` 是为了对比**不同输出结构**的效果。模式数量是消融实验需求驱动的，不是空架子。

### Q3: "代码改了 1043 行，怎么保证没破坏原项目？"

**A**：改造遵循**向后兼容**原则——原版 `model_cot_loader.py` 的 200 行代码逻辑完全保留，新增逻辑用 `--mode full`（默认整图模式）退化到原版行为。所有新参数都有默认值，不传就走原版路径。同时保留了原项目的接口签名，只是把位置参数扩展为关键字参数。

### Q4: "程序跑出来这些数字，怎么证明不是偶然？"

**A**：做了**多种子稳定性测试**——3 个随机种子下 structured_evidence 是 0.443 ± 0.012，pred_bbox 是 0.393 ± 0.025。提升 0.05 大于噪声 std 0.025，统计上是显著的。完整数据在 `runs/20260522_round2_extensions/`。

### Q5: "如果我现在让你改一个参数重新跑，能跑通吗？"

**A**：可以。比如把 `--mode` 从 `structured_evidence` 改成 `crop_only`，再加 `--max-samples 5`，30 秒内能拿到结果。**所有模式共用同一份代码、同一个 CLI 入口、同一份 JSONL schema**。

---

## 8. 临场速记口诀

| 步骤 | 一句话核心 |
|---|---|
| 项目背景 | "基于 Visual-CoT 做多模式推理实验框架" |
| 开发目标 | "16 种模式 + 双阶段 pipeline + 标准化实验目录" |
| 核心技术 | "200 行 → 1043 行 + 30+ CLI + 25+ JSONL 字段" |
| 运行结果 | "现场跑 2 样本 + CUB100 主表 + 跨数据集 100" |

**最关键三个数字**（一定要记住）：**1043 行 / 16 种模式 / ~5000 次推理**。
