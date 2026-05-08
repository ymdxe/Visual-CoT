# Paper Tables

## 不同输入压缩策略准确率与效率对比

| mode | N | contains_match_mean | mean_latency_ms_total | mean_peak_gpu_memory_mb | mean_num_visual_inputs | mean_crop_ratio |
|---|---|---|---|---|---|---|
| center_bbox | 100.0000 | 0.8300 | 414.3402 | 14212.4493 | 2.0000 | 0.2500 |
| crop_only | 100.0000 | 0.6900 | 211.8008 | 13888.2580 | 1.0000 | 0.1544 |
| full | 100.0000 | 0.7000 | 692.5032 | 13886.5939 | 1.0000 | NA |
| lowres_full_highrescrop | 100.0000 | 0.7900 | 277.8703 | 14176.6949 | 2.0000 | 0.1544 |
| oracle_bbox | 100.0000 | 0.8200 | 437.6364 | 14212.4493 | 2.0000 | 0.2660 |
| pred_bbox | 100.0000 | 0.8000 | 354.4706 | 14212.4493 | 2.0000 | 0.1544 |
| random_bbox | 100.0000 | 0.8200 | 310.6738 | 14212.4493 | 2.0000 | 0.1154 |
| structured_evidence | 100.0000 | 0.7300 | 264.9504 | 14272.1887 | 2.0000 | 0.1544 |
| woimg | 100.0000 | 0.4700 | 514.9730 | 13616.0077 | 0.0000 | NA |

## 图像压缩效率对比

| mode | visual_input_reduction | latency_reduction_vs_pred_bbox | memory_reduction_vs_pred_bbox | accuracy_delta_vs_pred_bbox |
|---|---|---|---|---|
| center_bbox | 0.0000 | -0.1689 | 0.0000 | 0.0300 |
| crop_only | 0.5000 | 0.4025 | 0.0228 | -0.1100 |
| full | 0.5000 | -0.9536 | 0.0229 | -0.1000 |
| lowres_full_highrescrop | 0.0000 | 0.2161 | 0.0025 | -0.0100 |
| oracle_bbox | 0.0000 | -0.2346 | 0.0000 | 0.0200 |
| pred_bbox | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_bbox | 0.0000 | 0.1236 | 0.0000 | 0.0200 |
| structured_evidence | 0.0000 | 0.2525 | -0.0042 | -0.0700 |
| woimg | 1.0000 | -0.4528 | 0.0420 | -0.3300 |

## 推理优化策略对比

| mode | reasoning_length_reduction | answer_accuracy_delta | generation_latency_delta | extraction_success_rate |
|---|---|---|---|---|
| center_bbox | NA | 0.0300 | 59.8757 | 1.0000 |
| crop_only | NA | -0.1100 | -142.6584 | 1.0000 |
| full | NA | -0.1000 | 338.0247 | 1.0000 |
| lowres_full_highrescrop | NA | -0.0100 | -76.5865 | 1.0000 |
| oracle_bbox | NA | 0.0200 | 83.1533 | 1.0000 |
| pred_bbox | NA | 0.0000 | 0.0000 | 1.0000 |
| random_bbox | NA | 0.0200 | -43.7749 | 1.0000 |
| structured_evidence | NA | -0.0700 | -89.5116 | 1.0000 |
| woimg | NA | -0.3300 | 160.5046 | 1.0000 |

## bbox 正确性与 answer 正确性交叉表

| mode | bbox_correct_answer_correct | bbox_correct_answer_wrong | bbox_wrong_answer_correct | bbox_wrong_answer_wrong | bbox_missing_or_parse_failed |
|---|---|---|---|---|---|
| center_bbox | 46.0000 | 8.0000 | 37.0000 | 9.0000 | 0.0000 |
| crop_only | 19.0000 | 8.0000 | 50.0000 | 23.0000 | 0.0000 |
| full | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 |
| lowres_full_highrescrop | 21.0000 | 6.0000 | 58.0000 | 15.0000 | 0.0000 |
| oracle_bbox | 82.0000 | 18.0000 | 0.0000 | 0.0000 | 0.0000 |
| pred_bbox | 20.0000 | 7.0000 | 60.0000 | 13.0000 | 0.0000 |
| random_bbox | 1.0000 | 0.0000 | 81.0000 | 18.0000 | 0.0000 |
| structured_evidence | 20.0000 | 7.0000 | 53.0000 | 20.0000 | 0.0000 |
| woimg | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 |
