# PLANS.md

## Current Goal

在一周内完成 Visual-CoT 毕业设计的最小实验闭环，包括代码改造、小规模实验、结果分析、图表生成和论文初稿。

## One-Week Plan

| Day | Stage | Status | Deliverable |
|---|---|---|---|
| Day 1 | 代码审查与实验底座 | [x] | experiments/ 目录、CLI 参数、AGENTS.md、PLANS.md |
| Day 2 | 统一日志与 smoke test | [ ] | JSONL schema、smoke test 结果 |
| Day 3 | 诊断实验 | [ ] | diagnostics_summary、error_summary |
| Day 4 | 区域证据增强 | [ ] | structured_evidence 结果 |
| Day 5 | 区域压缩实验 | [ ] | efficiency_summary、tradeoff 图 |
| Day 6 | 图表与案例整理 | [ ] | final_report.md、主结果表、案例图 |
| Day 7 | 论文初稿 | [ ] | 论文初稿、图表引用、局限性分析 |

## Must Finish

- [x] 修复 detection question_id
- [x] 暴露 --load-4bit / --load-8bit
- [x] 扩展 JSONL 输出字段
- [x] 实现 structured evidence prompt
- [ ] 完成至少一个数据集的 20 条样本诊断实验
- [ ] 完成至少一张主结果表
- [ ] 完成至少一张效率对比表
- [ ] 完成至少 3 张可视化案例图
- [ ] 完成论文初稿

## Optional

- [x] oracle bbox
- [x] region caption 最小版本
- [x] answer verifier 最小版本
- [x] lowres_full_highres_crop 初版
- [x] top-k crops 参数脚手架
- [ ] GQA 补充实验

## Cut If Time Is Not Enough

- [ ] 完整 benchmark
- [ ] 全量训练
- [ ] patch/token-level pruning
- [ ] 多数据集大规模实验
- [ ] 外部 API 打分
