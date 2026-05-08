# Paper Tables

## 不同输入压缩策略准确率与效率对比

| mode | N | contains_match_mean | mean_latency_ms_total | mean_peak_gpu_memory_mb | mean_num_visual_inputs | mean_crop_ratio |
|---|---|---|---|---|---|---|
| crop_only | 3.0000 | 0.6667 | 465.4010 | 13889.3444 | 1.0000 | 0.3019 |
| full | 3.0000 | 0.6667 | 820.9412 | 13889.3315 | 1.0000 | NA |
| pred_bbox | 3.0000 | 0.6667 | 473.5177 | 14213.3255 | 2.0000 | 0.3019 |
| structured_evidence | 3.0000 | 0.6667 | 385.6609 | 14273.8494 | 2.0000 | 0.3019 |

## 图像压缩效率对比

| mode | visual_input_reduction | latency_reduction_vs_pred_bbox | memory_reduction_vs_pred_bbox | accuracy_delta_vs_pred_bbox |
|---|---|---|---|---|
| crop_only | 0.5000 | 0.0171 | 0.0228 | 0.0000 |
| full | 0.5000 | -0.7337 | 0.0228 | 0.0000 |
| pred_bbox | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| structured_evidence | 0.0000 | 0.1855 | -0.0043 | 0.0000 |

## 推理优化策略对比

| mode | reasoning_length_reduction | answer_accuracy_delta | generation_latency_delta | extraction_success_rate |
|---|---|---|---|---|
| crop_only | NA | 0.0000 | -8.1540 | 1.0000 |
| full | NA | 0.0000 | 347.4179 | 1.0000 |
| pred_bbox | NA | 0.0000 | 0.0000 | 1.0000 |
| structured_evidence | NA | 0.0000 | -87.8845 | 1.0000 |

## bbox 正确性与 answer 正确性交叉表

| mode | bbox_correct_answer_correct | bbox_correct_answer_wrong | bbox_wrong_answer_correct | bbox_wrong_answer_wrong | bbox_missing_or_parse_failed |
|---|---|---|---|---|---|
| crop_only | 2.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.0000 |
| pred_bbox | 2.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| structured_evidence | 2.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
