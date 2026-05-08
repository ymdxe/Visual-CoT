# 实验结果草稿

本轮实验实际产生的模式包括：crop_only, full, pred_bbox, structured_evidence。

所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。

- `crop_only`: N=3, contains_match=0.6667, latency=465.4010 ms, memory=13889.3444 MB。
- `full`: N=3, contains_match=0.6667, latency=820.9412 ms, memory=13889.3315 MB。
- `pred_bbox`: N=3, contains_match=0.6667, latency=473.5177 ms, memory=14213.3255 MB。
- `structured_evidence`: N=3, contains_match=0.6667, latency=385.6609 ms, memory=14273.8494 MB。
