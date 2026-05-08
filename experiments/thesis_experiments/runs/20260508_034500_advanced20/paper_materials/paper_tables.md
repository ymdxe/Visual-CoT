# Paper Tables

## 不同输入压缩策略准确率与效率对比

| mode | N | contains_match_mean | mean_latency_ms_total | mean_peak_gpu_memory_mb | mean_num_visual_inputs | mean_crop_ratio |
|---|---|---|---|---|---|---|
| feature_pca_256 | 20.0000 | 0.8500 | 385.8309 | 14212.5983 | 2.0000 | 0.1566 |
| feature_pca_512 | 20.0000 | 0.8500 | 383.8382 | 14212.5983 | 2.0000 | 0.1566 |
| vision_pruned_10 | 20.0000 | 0.6000 | 331.4417 | 14212.5475 | 2.0000 | 0.1566 |
| vision_pruned_20 | 20.0000 | 0.5000 | 232.7996 | 14212.5475 | 2.0000 | 0.1566 |

## 图像压缩效率对比

| mode | visual_input_reduction | latency_reduction_vs_pred_bbox | memory_reduction_vs_pred_bbox | accuracy_delta_vs_pred_bbox |
|---|---|---|---|---|
| feature_pca_256 | NA | NA | NA | NA |
| feature_pca_512 | NA | NA | NA | NA |
| vision_pruned_10 | NA | NA | NA | NA |
| vision_pruned_20 | NA | NA | NA | NA |

## 推理优化策略对比

| mode | reasoning_length_reduction | answer_accuracy_delta | generation_latency_delta | extraction_success_rate |
|---|---|---|---|---|
| feature_pca_256 | NA | NA | NA | 1.0000 |
| feature_pca_512 | NA | NA | NA | 1.0000 |
| vision_pruned_10 | NA | NA | NA | 1.0000 |
| vision_pruned_20 | NA | NA | NA | 1.0000 |

## bbox 正确性与 answer 正确性交叉表

| mode | bbox_correct_answer_correct | bbox_correct_answer_wrong | bbox_wrong_answer_correct | bbox_wrong_answer_wrong | bbox_missing_or_parse_failed |
|---|---|---|---|---|---|
| feature_pca_256 | 5.0000 | 2.0000 | 12.0000 | 1.0000 | 0.0000 |
| feature_pca_512 | 5.0000 | 2.0000 | 12.0000 | 1.0000 | 0.0000 |
| vision_pruned_10 | 4.0000 | 3.0000 | 8.0000 | 5.0000 | 0.0000 |
| vision_pruned_20 | 4.0000 | 3.0000 | 6.0000 | 7.0000 | 0.0000 |
