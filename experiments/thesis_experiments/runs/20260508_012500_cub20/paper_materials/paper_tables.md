# Paper Tables

## 不同输入压缩策略准确率与效率对比

| mode | N | contains_match_mean | mean_latency_ms_total | mean_peak_gpu_memory_mb | mean_num_visual_inputs | mean_crop_ratio |
|---|---|---|---|---|---|---|
| center_bbox | 20.0000 | 0.9500 | 464.1109 | 14212.5983 | 2.0000 | 0.2500 |
| crop_only | 20.0000 | 0.7500 | 287.1790 | 13889.2340 | 1.0000 | 0.1566 |
| full | 20.0000 | 0.8000 | 708.6136 | 13887.3358 | 1.0000 | NA |
| lowres_full_highrescrop | 20.0000 | 0.9500 | 305.5286 | 14176.9226 | 2.0000 | 0.1566 |
| oracle_bbox | 20.0000 | 0.9000 | 414.3626 | 14212.5983 | 2.0000 | 0.2879 |
| pred_bbox | 20.0000 | 0.8500 | 380.6192 | 14212.5983 | 2.0000 | 0.1566 |
| random_bbox | 20.0000 | 0.9000 | 351.7497 | 14212.5983 | 2.0000 | 0.0912 |
| structured_evidence | 20.0000 | 0.8000 | 277.2532 | 14272.4631 | 2.0000 | 0.1566 |
| woimg | 20.0000 | 0.5500 | 527.6480 | 13616.0478 | 0.0000 | NA |

## 图像压缩效率对比

| mode | visual_input_reduction | latency_reduction_vs_pred_bbox | memory_reduction_vs_pred_bbox | accuracy_delta_vs_pred_bbox |
|---|---|---|---|---|
| center_bbox | 0.0000 | -0.2194 | 0.0000 | 0.1000 |
| crop_only | 0.5000 | 0.2455 | 0.0228 | -0.1000 |
| full | 0.5000 | -0.8617 | 0.0229 | -0.0500 |
| lowres_full_highrescrop | 0.0000 | 0.1973 | 0.0025 | 0.1000 |
| oracle_bbox | 0.0000 | -0.0887 | 0.0000 | 0.0500 |
| pred_bbox | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_bbox | 0.0000 | 0.0758 | 0.0000 | 0.0500 |
| structured_evidence | 0.0000 | 0.2716 | -0.0042 | -0.0500 |
| woimg | 1.0000 | -0.3863 | 0.0420 | -0.3000 |

## 推理优化策略对比

| mode | reasoning_length_reduction | answer_accuracy_delta | generation_latency_delta | extraction_success_rate |
|---|---|---|---|---|
| center_bbox | NA | 0.1000 | 83.4559 | 1.0000 |
| crop_only | NA | -0.1000 | -93.4573 | 1.0000 |
| full | NA | -0.0500 | 327.9847 | 1.0000 |
| lowres_full_highrescrop | NA | 0.1000 | -75.1298 | 1.0000 |
| oracle_bbox | NA | 0.0500 | 33.7427 | 1.0000 |
| pred_bbox | NA | 0.0000 | 0.0000 | 1.0000 |
| random_bbox | NA | 0.0500 | -28.8996 | 1.0000 |
| structured_evidence | NA | -0.0500 | -103.4115 | 1.0000 |
| woimg | NA | -0.3000 | 146.9973 | 1.0000 |

## bbox 正确性与 answer 正确性交叉表

| mode | bbox_correct_answer_correct | bbox_correct_answer_wrong | bbox_wrong_answer_correct | bbox_wrong_answer_wrong | bbox_missing_or_parse_failed |
|---|---|---|---|---|---|
| center_bbox | 14.0000 | 0.0000 | 5.0000 | 1.0000 | 0.0000 |
| crop_only | 5.0000 | 2.0000 | 10.0000 | 3.0000 | 0.0000 |
| full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.0000 |
| lowres_full_highrescrop | 6.0000 | 1.0000 | 13.0000 | 0.0000 | 0.0000 |
| oracle_bbox | 18.0000 | 2.0000 | 0.0000 | 0.0000 | 0.0000 |
| pred_bbox | 5.0000 | 2.0000 | 12.0000 | 1.0000 | 0.0000 |
| random_bbox | 0.0000 | 0.0000 | 18.0000 | 2.0000 | 0.0000 |
| structured_evidence | 6.0000 | 1.0000 | 10.0000 | 3.0000 | 0.0000 |
| woimg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 20.0000 |
