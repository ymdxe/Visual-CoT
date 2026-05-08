# 实验结果草稿

本轮实验实际产生的模式包括：center_bbox, crop_only, full, lowres_full_highrescrop, oracle_bbox, pred_bbox, random_bbox, structured_evidence, woimg。

所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。

- `center_bbox`: N=20, contains_match=0.9500, latency=464.1109 ms, memory=14212.5983 MB。
- `crop_only`: N=20, contains_match=0.7500, latency=287.1790 ms, memory=13889.2340 MB。
- `full`: N=20, contains_match=0.8000, latency=708.6136 ms, memory=13887.3358 MB。
- `lowres_full_highrescrop`: N=20, contains_match=0.9500, latency=305.5286 ms, memory=14176.9226 MB。
- `oracle_bbox`: N=20, contains_match=0.9000, latency=414.3626 ms, memory=14212.5983 MB。
- `pred_bbox`: N=20, contains_match=0.8500, latency=380.6192 ms, memory=14212.5983 MB。
- `random_bbox`: N=20, contains_match=0.9000, latency=351.7497 ms, memory=14212.5983 MB。
- `structured_evidence`: N=20, contains_match=0.8000, latency=277.2532 ms, memory=14272.4631 MB。
- `woimg`: N=20, contains_match=0.5500, latency=527.6480 ms, memory=13616.0478 MB。
