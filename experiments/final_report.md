# Visual-CoT 毕设实验阶段报告

## 实验环境

待运行后填写 GPU、CUDA、PyTorch、checkpoint、样本数量。

## 模型与数据

当前默认使用 `./checkpoints/VisCoT-7b-224` 和仓库中已配置的 CUB 数据。TextVQA、DocVQA、SROIE 脚本已保留入口，但图片不完整时不强制运行。

## 实验模式

- `full`：只使用完整图像。
- `pred_bbox`：使用 detection 预测 bbox 的完整 Visual-CoT 流程。
- `random_bbox` / `center_bbox`：负对照。
- `woimg`：无图输入，检测语言先验。
- `crop_only`：只看局部区域。
- `structured_evidence`：回答前显式组织局部区域、视觉证据、推理和答案。
- `lowres_full_highrescrop`：低分辨率全局图和高清局部 crop 组合。

## 主结果

运行 `experiments/analysis/summarize_results.py` 后引用 `experiments/tables/diagnostics_summary.md`。

## 消融分析

待实验结果生成后补充。

## 效率分析

待 `run_latency_benchmark.sh` 生成结果后补充。

## 典型案例

待 `run_visualizations.sh` 生成案例图后补充。

## 局限性

当前实现属于免训练的输入侧区域证据增强与视觉输入压缩，不是模型参数剪枝。受限于计算资源和时间，本文优先采用小规模验证实验，尚未覆盖完整 Visual-CoT benchmark。

## 后续工作

后续可扩展到多数据集正式评测，完善 top-k crops，并探索更细粒度的视觉 token pruning。
