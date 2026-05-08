# Paper Tables

## 不同输入压缩策略准确率与效率对比

| mode | N | contains_match_mean | mean_latency_ms_total | mean_peak_gpu_memory_mb | mean_num_visual_inputs | mean_crop_ratio |
|---|---|---|---|---|---|---|
| answer_reasoning_compressed | 100.0000 | 0.7300 | 260.3771 | 14272.1887 | 2.0000 | 0.1544 |
| direct_then_explain | 100.0000 | 0.7700 | 776.9798 | 14176.6678 | 2.0000 | 0.1544 |
| heuristic_step_selector | 100.0000 | 0.7300 | 261.2435 | 14272.1887 | 2.0000 | 0.1544 |
| structured_evidence | 100.0000 | 0.7300 | 261.0688 | 14272.1887 | 2.0000 | 0.1544 |

## 图像压缩效率对比

| mode | visual_input_reduction | latency_reduction_vs_pred_bbox | memory_reduction_vs_pred_bbox | accuracy_delta_vs_pred_bbox |
|---|---|---|---|---|
| answer_reasoning_compressed | NA | NA | NA | NA |
| direct_then_explain | NA | NA | NA | NA |
| heuristic_step_selector | NA | NA | NA | NA |
| structured_evidence | NA | NA | NA | NA |

## 推理优化策略对比

| mode | reasoning_length_reduction | answer_accuracy_delta | generation_latency_delta | extraction_success_rate |
|---|---|---|---|---|
| answer_reasoning_compressed | 0.0000 | NA | NA | 1.0000 |
| direct_then_explain | NA | NA | NA | 1.0000 |
| heuristic_step_selector | NA | NA | NA | 1.0000 |
| structured_evidence | NA | NA | NA | 1.0000 |

## bbox 正确性与 answer 正确性交叉表

| mode | bbox_correct_answer_correct | bbox_correct_answer_wrong | bbox_wrong_answer_correct | bbox_wrong_answer_wrong | bbox_missing_or_parse_failed |
|---|---|---|---|---|---|
| answer_reasoning_compressed | 20.0000 | 7.0000 | 53.0000 | 20.0000 | 0.0000 |
| direct_then_explain | 20.0000 | 7.0000 | 57.0000 | 16.0000 | 0.0000 |
| heuristic_step_selector | 20.0000 | 7.0000 | 53.0000 | 20.0000 | 0.0000 |
| structured_evidence | 20.0000 | 7.0000 | 53.0000 | 20.0000 | 0.0000 |
