# Visual-CoT CUB100 实验报告

## 1. 实验目标

本阶段实验用于验证 Visual-CoT 改造后的推理增强与诊断平台是否能够在真实 GPU 环境下完成小规模闭环。实验不训练模型，不运行完整 benchmark，而是在 CUB 数据集前 100 条样本上比较不同视觉输入策略和结构化证据提示的效果。

## 2. 实验环境

- GPU：NVIDIA GeForce RTX 3090，24GB
- Python 环境：conda 环境 `viscot`
- 模型 checkpoint：`./checkpoints/VisCoT-7b-224`
- 数据集：CUB 前 100 条样本
- 推理精度：默认 bf16
- 说明：4-bit smoke test 曾进入生成阶段但出现 dtype 不一致，因此本次正式 CUB100 使用 bf16。

## 3. 实验模式

- `full`：只输入完整图像。
- `pred_bbox`：使用 detection 预测 bbox，输入完整图像和局部 crop。
- `random_bbox`：随机 bbox 负对照。
- `center_bbox`：中心 bbox 负对照。
- `woimg`：无图输入，用于观察语言先验。
- `crop_only`：只输入 predicted bbox 的局部 crop。
- `lowres_full_highrescrop`：输入低分辨率完整图和高清局部 crop。
- `structured_evidence`：使用 predicted bbox + crop，并要求模型按区域、证据、推理、答案的结构回答。

## 4. 主结果

| Dataset | Mode | BBox | Crop | Evidence | N | EM | Substr | BBox Acc | Latency | Mem | Inputs | Crop Ratio |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cub | center_bbox | center | full_crop | none | 100 | 0.3700 | 0.7900 | 0.5300 | 446.2526 | 14212.8317 | 2.0000 | 0.2500 |
| cub | crop_only | pred | crop_only | none | 100 | 0.6100 | 0.7600 | 0.6800 | 313.7803 | 13890.8044 | 1.0000 | 0.1542 |
| cub | full | none | full_only | none | 100 | 0.0000 | 0.6400 |  | 701.4467 | 13886.9255 | 1.0000 |  |
| cub | lowres_full_highrescrop | pred | lowres_full_highres_crop | none | 100 | 0.5600 | 0.7300 | 0.6800 | 360.8498 | 14177.1387 | 2.0000 | 0.1542 |
| cub | pred_bbox | pred | full_crop | none | 100 | 0.4900 | 0.8000 | 0.6800 | 389.1029 | 14212.8317 | 2.0000 | 0.1542 |
| cub | random_bbox | random | full_crop | none | 100 | 0.5700 | 0.7800 | 0.0000 | 316.7484 | 14212.8317 | 2.0000 | 0.1137 |
| cub | structured_evidence | pred | full_crop | structured | 100 | 0.8300 | 0.8300 | 0.6800 | 263.8775 | 14272.6801 | 2.0000 | 0.1542 |
| cub | woimg | none | full_only | none | 100 | 0.0000 | 0.4600 |  | 534.9479 | 13617.8978 | 0.0000 |  |

## 5. 结果分析

CUB100 detection 阶段的 bbox 解析成功率为 100%，predicted bbox 在 answer 阶段对应的平均 IoU 为 0.6375，IoU 大于 0.5 的比例为 0.68。说明当前 detection 能够提供可用区域，但仍有一部分 bbox 质量不足。

在回答质量上，`full` 的 normalized contains match 为 0.64，`pred_bbox` 提升到 0.80，说明局部 crop 对 CUB 细粒度属性判断有帮助。`structured_evidence` 达到 0.83，是本次实验中最高的模式，表明结构化区域证据提示在小规模实验中能改善回答稳定性。

从诊断消融看，`random_bbox` 为 0.78，`center_bbox` 为 0.79，与 `pred_bbox` 的 0.80 较接近。这说明 CUB yes/no 属性判断中，模型并不完全依赖 predicted bbox；部分答案可能来自整图、中心区域或语言先验。因此论文中不能把全部提升简单归因于 bbox 精准定位，需要如实说明 bbox 利用仍不稳定。

从效率看，`crop_only` 的视觉输入数量为 1，平均延迟 313.78 ms，低于 `pred_bbox` 的 389.10 ms，但 contains match 从 0.80 降到 0.76。`lowres_full_highrescrop` 平均延迟 360.85 ms，contains match 为 0.73。由于视觉编码器采用固定输入分辨率，区域级压缩对底层视觉 token 计算的降低有限，当前收益主要来自输入组织变化和背景干扰弱化。

## 6. 错误分类

`structured_evidence` 的错误分类统计如下：

| error_tag | count |
|---|---:|
| bbox_correct_answer_correct | 56 |
| bbox_correct_answer_wrong | 12 |
| bbox_wrong_answer_correct | 27 |
| bbox_wrong_answer_wrong | 5 |

其中 bbox 错误但答案正确的样本有 27 个，说明模型在部分样本中仍可能依赖全局图像或语言先验。bbox 正确但答案错误的样本有 12 个，说明仅有正确 bbox 并不保证 answer 阶段正确使用局部证据。

## 7. 论文可写结论

本文完成的是输入侧区域证据增强和推理流程优化，不是模型参数剪枝。CUB100 小规模实验初步表明：

1. predicted bbox + crop 相比 full image baseline 有更好的回答质量；
2. 结构化区域证据提示可以提升回答格式稳定性和小规模准确率；
3. crop-only 可以降低视觉输入数量和部分延迟，但会损失全局上下文；
4. random/center bbox 与 predicted bbox 接近，说明模型对 bbox 的利用仍不充分；
5. 后续需要扩展数据集、改进 top-k region selection，并探索更细粒度视觉 token pruning。
