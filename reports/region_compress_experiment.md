# Region-Aware Image Compression for Visual-CoT

## 1. 实验目标

本实验在 Visual-CoT 两阶段推理流程基础上，引入区域感知图像压缩方法。该方法复用 detection 阶段预测得到的关键区域 bbox，在不重新训练大模型、不做模型参数剪枝的前提下，从图像输入侧减少背景干扰，并记录更细的推理效率指标。

本阶段重点从 CUB 30 条扩展到 CUB 100 条 predicted bbox 实验，同时加入 `visual_input_policy` 控制和更细粒度的耗时统计，用于分析“区域压缩”和“视觉输入张数变化”对推理效率与回答效果的影响。

## 2. 方法说明

- `none`：原始 Visual-CoT 输入，不做区域压缩。
- `crop_only`：只保留 bbox 对应的关键区域裁剪图作为视觉输入。
- `blur`：bbox 内关键区域保持清晰，bbox 外背景进行高斯模糊。
- `downsample`：bbox 内关键区域保持清晰，bbox 外背景先降采样再上采样。
- `mask`：bbox 内关键区域保持清晰，bbox 外背景灰度化并降低亮度。

新增 `visual_input_policy`：

- `auto`：默认策略；`none` 保持原始双图输入，压缩模式使用单张压缩图。
- `single`：强制只使用一张视觉输入图，用于验证“少一张视觉输入”本身是否带来加速。
- `dual`：保留给后续公平消融，可强制使用主图加局部裁剪图。

新增效率字段：

- `num_visual_images`
- `preprocess_latency_sec`
- `generation_latency_sec`
- `end_to_end_latency_sec`
- `visual_input_policy`

## 3. 实验环境

- 项目路径：`/root/code/Visual-CoT`
- Conda 环境：`viscot`
- Python：`3.10.20`
- GPU：NVIDIA GeForce RTX 3090，24GB
- NVIDIA Driver：`580.105.08`
- nvidia-smi CUDA：`13.0`
- PyTorch：`2.1.2+cu121`
- Torchvision：`0.16.2+cu121`
- Transformers：`4.37.2`
- Accelerate：`0.21.0`
- DeepSpeed：`0.12.6`
- BitsAndBytes：`0.45.5`
- Flash-Attn：`2.3.6`
- 模型 checkpoint：`checkpoints/VisCoT-7b-224`
- 实际加载路径：`checkpoints/llava-VisCoT-7b-224`，该路径为兼容原仓库 builder 逻辑的符号链接。
- 数据集：CUB，`downloads/cub/CUB_200_2011/images` 共 11788 张图片。
- 主实验样本数量：CUB 100 条，predicted bbox。

## 4. CUB100 主结果

当前 `results/region_compress/summary.md` 对应 CUB 100 条 predicted bbox 综合对比结果：

| compress_mode | visual_input_policy | num_samples | avg_latency_sec | avg_preprocess_latency_sec | avg_generation_latency_sec | avg_end_to_end_latency_sec | avg_max_gpu_memory_mb | avg_num_visual_images | bbox_valid_rate | avg_bbox_area | contains_match | normalized_contains_match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blur | auto | 100 | 0.2025 | 0.0748 | 0.2025 | 0.2777 | 13947.0603 | 1.0000 | 0.9100 | 0.1798 | 0.8000 | 0.8000 |
| crop_only | auto | 100 | 0.2109 | 0.0301 | 0.2109 | 0.2414 | 13944.0414 | 1.0000 | 0.9100 | 0.1798 | 0.7000 | 0.7000 |
| downsample | auto | 100 | 0.2093 | 0.0735 | 0.2093 | 0.2831 | 13947.0603 | 1.0000 | 0.9100 | 0.1798 | 0.7900 | 0.7900 |
| mask | auto | 100 | 0.2040 | 0.0588 | 0.2040 | 0.2631 | 13947.0212 | 1.0000 | 0.9100 | 0.1798 | 0.7800 | 0.7800 |
| none | auto | 100 | 0.3633 | 0.0181 | 0.3633 | 0.3820 | 14223.4833 | 2.0000 | 0.9100 | 0.1798 | 0.8000 | 0.8000 |
| none | single | 100 | 0.3639 | 0.0133 | 0.3639 | 0.3778 | 13944.7526 | 1.0000 | 0.9100 | 0.1798 | 0.7300 | 0.7300 |

阶段性结果文件：

- CUB 100 auto 主实验：`results/region_compress/summary_predbbox_cub100_auto.md`
- CUB 100 none-single 对照：`results/region_compress/summary_predbbox_cub100_none_single.md`
- CUB 100 综合对比：`results/region_compress/summary_predbbox_cub100_comparison.md`
- CUB 30 predicted bbox 旧结果：`results/region_compress/summary_predbbox_cub30.md`
- CUB 3 smoke test：`results/region_compress/summary_predbbox_cub3.md`

压缩图示例已保存到：

- `results/region_compress/compressed_images_predbbox_cub100_auto/`

该目录包含 400 张压缩图，来自 `crop_only`、`blur`、`downsample`、`mask` 四种模式各 100 张。

## 5. 初步分析

从 CUB100 predicted bbox 结果看，`blur+auto` 与原始 `none+auto` 的 normalized contains match 均为 0.8000，但 `blur+auto` 的平均生成耗时从 0.3633 秒降到 0.2025 秒，平均端到端耗时从 0.3820 秒降到 0.2777 秒。也就是说，在本轮小样本实验中，背景模糊策略保持了与原始流程相同的规则匹配准确率，同时表现出更低的平均推理耗时。

`downsample+auto` 的准确率为 0.7900，接近原始流程；`mask+auto` 为 0.7800。二者说明背景弱化或降采样也能保留较多有效信息，但稳定性略低于 blur。`crop_only+auto` 的端到端耗时最低，为 0.2414 秒，但准确率降到 0.7000，说明只保留局部裁剪会丢失鸟类属性判断所需的上下文。

`none+single` 对照显示，单纯把原始 Visual-CoT 从双图输入改成单图输入，并没有带来生成耗时下降：`none+auto` 为 0.3633 秒，`none+single` 为 0.3639 秒。但显存从 14223MB 降到 13945MB，同时准确率从 0.8000 降到 0.7300。这说明本实验中的收益不能简单归因于“少一张图”，更可能来自区域压缩后的视觉干扰降低、提示组织变化和输出分布变化。

显存方面，所有单图策略约为 13.94GB，原始双图 `none+auto` 约为 14.22GB。由于视觉编码器通常会将输入处理到固定分辨率，区域压缩不等同于底层 token 级计算剪枝；该方法更适合表述为输入侧区域感知压缩和视觉信息筛选。

## 6. bbox 失败案例

CUB100 predicted bbox 的 bbox 有效率为 0.9100，共有 9 条样本因 bbox 面积过小被判定为无效。典型样本包括：

- question_id `833`：`Does the bird in the picture have olive forehead and olive crown?`
- question_id `354`：`Does the bird in the picture have black eye and solid belly?`
- question_id `976`：`Does the bird in the picture have solid wing and solid tail?`
- question_id `2283`：`Does the bird in the picture have green eye and grey eye?`
- question_id `1699`：`Does the bird in the picture have pink eye and grey forehead?`

这些失败案例集中体现了 detection 阶段 bbox 质量的重要性。当 predicted bbox 过窄时，区域压缩策略会回退到原图，或者在局部信息不足时影响回答稳定性。

## 7. 答案变化案例

与原始 `none+auto` 相比，`blur+auto` 有 7 条样本从错误变为正确，也有 7 条样本从正确变为错误。

改善案例：

- question_id `4191`：问题为 `squared_tail tail and red bill`，`none` 回答 Yes，`blur` 回答 No，标准答案为 No。
- question_id `2416`：问题为 `purple bill and green eye`，`none` 回答 Yes，`blur` 回答 No，标准答案为 No。
- question_id `1608`：问题为 `blue nape and black underparts`，`none` 回答 No，`blur` 回答 Yes，标准答案为 Yes。

退化案例：

- question_id `2119`：问题为 `large size and grey leg`，`none` 回答 No，`blur` 回答 Yes，标准答案为 No。
- question_id `3024`：问题为 `black crown and buff crown`，`none` 回答 Yes，`blur` 回答 No，标准答案为 Yes。
- question_id `3306`：问题为 `solid belly and solid breast`，`none` 回答正确，`blur` 回答 No，标准答案为 Yes。

这说明区域压缩并非对所有问题都单调提升。对局部属性判断，背景模糊可能减少干扰；但当问题需要全局姿态、尺寸、上下文或多个部位联合判断时，压缩可能损失有用信息。

## 8. 当前验证状态

已完成：

- 新增脚本语法检查通过。
- `visual_input_policy` 已支持 `auto|single|dual`。
- metadata 已记录视觉输入张数、预处理耗时、生成耗时和端到端耗时。
- 3 条 CUB smoke test 已验证新增字段。
- 100 条 CUB predicted bbox detection 已完成。
- 100 条 CUB 五模式 auto 主实验已完成。
- 100 条 CUB `none+single` 加速对照已完成。
- `summary.csv` 与 `summary.md` 已更新为 CUB100 综合对比。
- 原始 `llava/eval/model_cot_loader.py` 未修改。

未完成：

- 尚未扩展到 TextVQA、DocVQA 等其他数据集；当前真实可用图片数据仍以 CUB 为主。
- 尚未做人工答案复核；当前答案评测为简单 contains match 与 normalized contains match。

## 9. 复现实验命令

进入项目与环境：

```bash
cd /root/code/Visual-CoT
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate viscot
```

语法检查：

```bash
python -m py_compile tools/region_compress.py
python -m py_compile tools/eval_region_compress_results.py
python -m py_compile tools/build_small_benchmark.py
python -m py_compile llava/eval/model_cot_loader_region_compress.py
bash -n scripts/region_compress/run_region_compress_eval.sh
```

构建 CUB100：

```bash
python tools/build_small_benchmark.py \
  --benchmark-dir ./viscot_benchmark/benchmark \
  --image-folder ./playground/data \
  --datasets cub \
  --num-per-dataset 100 \
  --output ./small_benchmark_cub100.json \
  --det-output ./results/region_compress/detection_input_cub100.jsonl
```

运行 detection：

```bash
python -m llava.eval.model_cot_det_loader \
  --model-path ./checkpoints/llava-VisCoT-7b-224 \
  --question-file ./results/region_compress/detection_input_cub100.jsonl \
  --image-folder ./playground/data \
  --answers-file ./results/region_compress/detection_pred_cub100.jsonl \
  --temperature 0 \
  --conv-mode vicuna_v1
```

运行五模式 auto 主实验：

```bash
SAVE_COMPRESSED_IMAGES=true \
VISUAL_INPUT_POLICY=auto \
COMPRESSED_IMAGE_DIR=./results/region_compress/compressed_images_predbbox_cub100_auto \
bash scripts/region_compress/run_region_compress_eval.sh \
  VisCoT-7b-224_predbbox_cub100_auto \
  ./small_benchmark_cub100.json \
  ./results/region_compress/detection_pred_cub100.jsonl \
  ./playground/data
```

运行 `none+single` 加速对照：

```bash
MODES="none" \
VISUAL_INPUT_POLICY=single \
bash scripts/region_compress/run_region_compress_eval.sh \
  VisCoT-7b-224_predbbox_cub100_none_single \
  ./small_benchmark_cub100.json \
  ./results/region_compress/detection_pred_cub100.jsonl \
  ./playground/data
```

重新生成 CUB100 综合对比：

```bash
python tools/eval_region_compress_results.py \
  --inputs ./results/region_compress/*/VisCoT-7b-224_predbbox_cub100_auto.jsonl \
           ./results/region_compress/none/VisCoT-7b-224_predbbox_cub100_none_single.jsonl \
  --output-csv ./results/region_compress/summary.csv \
  --output-md ./results/region_compress/summary.md
```
