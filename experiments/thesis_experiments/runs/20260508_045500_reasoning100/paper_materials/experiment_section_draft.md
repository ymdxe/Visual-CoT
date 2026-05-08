# 实验结果草稿

本轮实验实际产生的模式包括：answer_reasoning_compressed, direct_then_explain, heuristic_step_selector, structured_evidence。

所有数字均来自本轮 JSONL 输出经 `tools/thesis/aggregate_results.py` 聚合得到的结果，未运行或失败的高级实验不写作主结果。

- `answer_reasoning_compressed`: N=100, contains_match=0.7300, latency=260.3771 ms, memory=14272.1887 MB。
- `direct_then_explain`: N=100, contains_match=0.7700, latency=776.9798 ms, memory=14176.6678 MB。
- `heuristic_step_selector`: N=100, contains_match=0.7300, latency=261.2435 ms, memory=14272.1887 MB。
- `structured_evidence`: N=100, contains_match=0.7300, latency=261.0688 ms, memory=14272.1887 MB。
