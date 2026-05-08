# 实验结果草稿

本轮实验实际产生的模式包括：center_bbox, crop_only, full, lowres_full_highrescrop, oracle_bbox, pred_bbox, random_bbox, structured_evidence, woimg。

所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。

- `center_bbox`: N=100, contains_match=0.8300, latency=414.3402 ms, memory=14212.4493 MB。
- `crop_only`: N=100, contains_match=0.6900, latency=211.8008 ms, memory=13888.2580 MB。
- `full`: N=100, contains_match=0.7000, latency=692.5032 ms, memory=13886.5939 MB。
- `lowres_full_highrescrop`: N=100, contains_match=0.7900, latency=277.8703 ms, memory=14176.6949 MB。
- `oracle_bbox`: N=100, contains_match=0.8200, latency=437.6364 ms, memory=14212.4493 MB。
- `pred_bbox`: N=100, contains_match=0.8000, latency=354.4706 ms, memory=14212.4493 MB。
- `random_bbox`: N=100, contains_match=0.8200, latency=310.6738 ms, memory=14212.4493 MB。
- `structured_evidence`: N=100, contains_match=0.7300, latency=264.9504 ms, memory=14272.1887 MB。
- `woimg`: N=100, contains_match=0.4700, latency=514.9730 ms, memory=13616.0077 MB。
