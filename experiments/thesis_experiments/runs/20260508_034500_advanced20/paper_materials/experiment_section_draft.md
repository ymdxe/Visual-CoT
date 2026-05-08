# 实验结果草稿

本轮实验实际产生的模式包括：feature_pca_256, feature_pca_512, vision_pruned_10, vision_pruned_20。

所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。

- `feature_pca_256`: N=20, contains_match=0.8500, latency=385.8309 ms, memory=14212.5983 MB。
- `feature_pca_512`: N=20, contains_match=0.8500, latency=383.8382 ms, memory=14212.5983 MB。
- `vision_pruned_10`: N=20, contains_match=0.6000, latency=331.4417 ms, memory=14212.5475 MB。
- `vision_pruned_20`: N=20, contains_match=0.5000, latency=232.7996 ms, memory=14212.5475 MB。
