# AGENTS.md

本仓库用于 Visual-CoT 毕业设计实验开发。当前阶段为论文修订与图表整合（v6）。
**当前进度快照见 [`STATUS.md`](STATUS.md)**——每次进入仓库先读它确认上下文，不要靠默认假设。

---

## 一、绝对禁止

- 不训练大模型，不启动完整 benchmark。
- 不删除原始代码、模型、数据和历史实验结果。
- 不把 `checkpoints/`、下载缓存、生成图片、日志、pid 文件纳入 Git。
- 不动 `Obsidian/过程文档/` 里的归档版本（除非用户明确要求）。
- 实验结果必须来自实际运行，不编造准确率、延迟或显存数据。
- 缺模型或缺数据时，记录缺失项和下一步命令，不要造数据。

## 二、强制约定

- **原 Visual-CoT 流程保持兼容**：新增能力通过参数、脚本和实验目录扩展，不改动原签名。
- **实验目录**：所有新实验放在 `experiments/thesis_experiments/runs/<时间戳>_<标签>/` 下；不要污染老 run。
- **JSONL 模式**：跟随 `llava/eval/model_cot_loader.py` 已确立的字段（包含 `latency_ms_*`、`crop_ratio`、`bbox_iou`、`metadata.bbox_score` 等）。新增字段加在尾部，不要重排已有字段。
- **图脚本**：Round-2 的论文图统一由 `tools/thesis/round2/render_paper_figures*.py` 生成。中文版用 `_zh` 后缀。不要在其他位置重画。
- **docx 编辑**：用 Anthropic 官方 `docx` skill 的 `unpack → 编辑 → pack`。打包时若新增内容不是 tracked changes，需要 `--validate false` 绕过严格校验。

## 三、Obsidian 仓库（论文与说明）命名约定

- 论文版本号用**前序号**：`5论文初稿…docx` → `6论文初稿…docx`。版本号往后只递增，**正文标题字符串不要改**。
- 修改说明同样用**前序号**：`6 修改说明-<日期>.md` 与 `6论文…docx` 配对。
- 历史版本一律归入 `Obsidian/过程文档/`，根目录只保留"当前最新版"+"导师审阅意见"。
- 论文图表的语言跟随论文语言：本论文是中文，所有图表标注必须用中文（中英对照映射见 `tools/thesis/round2/render_paper_figures_zh.py` 中的 `CN_LABEL` 字典）。

## 四、内容修改边界

- 用户要求"只添加内容不改格式"时：仅在小节末尾追加段落／公式／伪代码／图，**不修改任何已有段落的字号、字体、样式**。
- 章节命名修订统一用 Word tracked changes (`<w:ins>` + `<w:del>`)，由用户在 Word 里逐条接受／拒绝。
- 替换图像保持原 `rId` 与 `media/imageN.png` 名，仅改文件内容；这样章节正文中的引用、caption、TOC 都不受影响。

## 五、关键文件指引

| 用途 | 路径 |
|---|---|
| 论文当前版本 | `Obsidian/6论文初稿第三版-20226451-张恒-马安香.docx` |
| 当前修改说明 | `Obsidian/6 修改说明-20260522.md` |
| 导师审阅意见 | `Obsidian/论文初稿-20226451-张恒-马安香_审阅意见_20260521.md` |
| 中文图源文件 | `Obsidian/assets/round2/paper_figures_zh/` |
| 实验运行根目录 | `experiments/thesis_experiments/runs/20260522_round2_extensions/` |
| Round-2 工具 | `tools/thesis/round2/` |
| 新算法实现 | `llava/eval/model_cot_loader.py::score_pred_bbox` + `select_box` |

## 六、答辩材料

本仓库当前以答辩材料完整性为目标。任何改动都应考虑：
- 是否影响 §4.4.1 边界框评分门控算法（本文的核心算法贡献）
- 是否影响 §3-§5 已注入的公式、伪代码、图表的章节归位
- 是否影响 `STATUS.md` 中列出的下游待办事项

## 七、状态维护责任

- 进入仓库工作时，第一步是 `Read STATUS.md`。
- 完成工作前，第二步是 `Edit STATUS.md` 把进度更新进来（论文版本表、未完成事项、已知问题）。
- 不要把状态信息塞进 `AGENTS.md`——这里只放约定与规则。
