# STATUS.md

> 状态快照，最后更新：2026-05-22 13:30
>
> 任何 AI 助手或新 IDE 会话进入仓库时，**先读这份文档**确认当前位置，然后再决定下一步动作。

---

## 0. 一句话当前位置

毕业设计论文修订进行到 **v6（中文图表版）**，方法贡献章节已注入（§4.4.1 边界框评分门控算法），实验补全完成（Phase 1-3 共 24 个 JSONL run），等待答辩。

---

## 1. 代码仓库状态

- **当前分支**：`thesis/cross-dataset-region-evidence-20260516`
- **主分支**：`main`
- **GitHub 同步**：Round-2 代码与说明文件已整理为代码提交；大型实验输出、Obsidian 论文资料和本地运行产物保留在远端/本地同步目录，不作为代码提交。
- **GitHub**：`https://github.com/ymdxe/Visual-CoT`
- **最近 5 次提交**：
  ```
  1b8c898 feat(exp): add cross-dataset region evidence experiments
  c376957 docs(exp): add thesis experiment report and paper materials
  14f5688 feat(analysis): refine advanced experiment aggregation
  3e4580d feat(analysis): aggregate and visualize thesis experiment results
  bd250a8 feat(eval): add visual input compression modes
  ```

---

## 2. 论文版本演进

| 版本 | 文件名 | 状态 | 关键改动 |
|---|---|---|---|
| 初稿第二版 | `过程文档/论文初稿第二版-...docx` | 归档 | 提交给导师审阅的版本，963 段落、20 张图 |
| 第三版集中式（已废弃）| `过程文档/...Round2增补.docx` | 归档 | 章末"增补 A/B/C"形式注入，与正文割裂 |
| 第三版分散式 | `过程文档/...分散注入.docx` | 归档 | 按 10 个小节边界分散注入，与正文融合 |
| v5 | `过程文档/5论文初稿第三版-...docx` | 归档 | 用户基于分散式手动整理 |
| **v6 当前** | **`6论文初稿第三版-20226451-张恒-马安香.docx`** | **活跃** | 5 张新增图表标注汉化 |

文件大小演进：v2 2.26 MB → v3 增补 2.65 MB → v3 分散 2.44 MB → v5 2.80 MB → v6 2.50 MB

---

## 3. 已完成的实验（Round-2）

输出根目录：`experiments/thesis_experiments/runs/20260522_round2_extensions/`

| 阶段 | 实验 | 数量 | 输出 |
|---|---|---|---|
| Phase 1 零 GPU | A1 IoU 分箱 | 1 | `analysis/cross_region100/A1_iou_bins.{csv,png}` |
| | A2 latency 分阶段 | 1 | `A2_latency_breakdown.{csv,png}` |
| | A3 题型分层 | 1 | `A3_question_types.{csv,md}` |
| | A4 Pareto 前沿 | 1 | `A4_pareto.{csv,png}` |
| | A5 SCS 结构一致性 | 1 | `A5_scs.{csv,md}` |
| | A6 McNemar 显著性 | 1 | `A6_mcnemar.{csv,md}` |
| Phase 2 灵敏性 | B1 lowres 扫描 | 5 GPU runs | `raw/lowres_sweep/answer_lowres{64,96,112,144,196}.jsonl` |
| | B2 padding 扫描 | 4 GPU runs | `raw/padding_sweep/answer_pad{10,12,15,20}.jsonl` |
| Phase 3 稳定性 + 新算法 | C1 3-seed × 3 mode | 9 GPU runs | `raw/seed_stability/` |
| | C2 bbox 评分消融 | 6 GPU runs | `raw/bbox_scoring/` |

**累计**：~1500 次推理 / 24 个新 JSONL / 12 张分析图表 / 1 个新算法实现。

---

## 4. 已注入到论文的内容（v6）

10 个分散注入点（章节边界）：

| 论文位置 | 注入内容 |
|---|---|
| §3.1 末 | 两阶段推理形式化 + Algorithm 1 |
| §3.4 末 | IoU / ContainsMatch / ANLS 三组公式 |
| §3.5.4 末 | 图 3.7 IoU 分箱（中文） |
| §4.2.1 末 | π_struct 提示构造算子形式化 |
| §4.2.2 末 | SCS 结构一致性得分定义 |
| §4.4 末 | **§4.4.1 边界框评分门控算法**：score 公式 + Algorithm 5 + 图 4.6（中文） |
| §5.2.1 末 | 四种视觉输入算子 Φ 形式化 |
| §5.2.2 末 | Algorithm 3 + 压缩率 η 形式化 |
| §5.3 末 | lowres 灵敏性 + 图 5.4 + padding 灵敏性 + 图 5.5（中文） |
| §5.4 末 | 多种子稳定性 + 图 5.6（中文） |

**章节命名 tracked changes**（用户在 Word 中接受/拒绝）：
- §2 "实验基础" → "技术"
- §3 "推理流程复现" → "推理方法复现"
- §7 "总结与后续工作" → "总结与展望"

---

## 5. 新增算法代码（已落地到 `llava/eval/model_cot_loader.py`）

| 函数 / 参数 | 位置 |
|---|---|
| `score_pred_bbox(box, question, args)` | 主评分函数 |
| `_FOCUS_KEYWORDS` | 方位关键词→中心坐标映射表 |
| `_parse_lambdas(text)` | CLI 解析 `"1.0,1.0,0.5"` 形式 |
| `select_box(...)` 改 4-元组返回 | 加 `score_info` |
| CLI `--bbox-scoring` | 总开关 |
| CLI `--bbox-score-threshold` | 默认 0.4 |
| CLI `--bbox-score-lambdas` | 默认 `"1.0,1.0,0.5"` |
| CLI `--bbox-score-target-area` | 默认 0.1 |
| CLI `--bbox-score-area-sigma` | 默认 1.0 |
| CLI `--bbox-score-fallback` | `center` / `oracle` / `none` |
| CLI `--crop-pad` | 暴露 crop padding 系数 α，默认 1.2 |

JSONL 新字段：`metadata.bbox_score = {score, components, lambdas, focus_word, threshold, triggered, used_fallback, original_box, fallback_kind}`。

---

## 6. 工具脚本目录

`tools/thesis/round2/`：
- `phase1_analysis.py` — A1-A6 零 GPU 分析
- `phase2_analysis.py` — B1 lowres + B2 padding 灵敏性聚合
- `phase3_analysis.py` — C1 多 seed + C2 bbox 评分聚合
- `aggregate_round2.py` — 汇总到 `ROUND2_REPORT.md`
- `render_paper_figures.py` — 英文版 nature-figure 渲染（7 张 × 4 格式）
- `render_paper_figures_zh.py` — **中文版图渲染**（5 张 × 3 格式，本次新增）
- `inject_into_thesis.py` — 集中式注入（已废弃）
- `inject_distributed.py` — **分散式注入**（10 个小节边界）

`experiments/thesis_experiments/runs/20260522_round2_extensions/scripts/`：
- `run_phase2_sweeps.sh` — Phase 2 GPU 扫描驱动
- `run_phase3.sh` — Phase 3 GPU 驱动

---

## 7. Obsidian 仓库结构

```
Obsidian/
├── 6论文初稿第三版-20226451-张恒-马安香.docx    ← 当前最新版（中文图）
├── 6 修改说明-20260522.md                       ← v5→v6 变更说明
├── 论文初稿-20226451-张恒-马安香_审阅意见_20260521.md ← 导师审阅原文
├── AGENTS.md（在仓库根，不在 Obsidian/）
├── STATUS.md（本文件，在仓库根）
├── assets/round2/
│   ├── paper_figures/       ← 英文版图（备份）
│   └── paper_figures_zh/    ← 中文版图（v6 使用）
└── 过程文档/                ← 归档
    ├── 5论文初稿第三版-...docx
    ├── 论文初稿第二版-...docx
    ├── 论文初稿第三版-...Round2增补.docx (废弃)
    ├── 论文初稿第三版-...分散注入.docx
    ├── 论文初稿第三版修订说明-20260522.md
    ├── 论文初稿第三版分散注入版说明-20260522.md
    └── Round2-实验补全与方法形式化-20260522.md
```

---

## 8. 关键实验数字（口径：跨数据集 100 样本）

主结果（包含性匹配 = ContainsMatch）：

| Mode | CM | EM | Latency | 备注 |
|---|---:|---:|---:|---|
| woimg | 0.08 | 0.01 | 927 ms | 跨域时语言先验失效 |
| full | 0.37 | 0.10 | 411 ms | 整图基线 |
| pred_bbox | 0.41 | 0.26 | 303 ms | Visual-CoT 主流程 |
| structured_evidence | **0.43** | **0.32** | 414 ms | **本文方法** |
| crop_only | 0.32 | 0.25 | **212 ms** | 最快 |
| lowres+highres | 0.37 | 0.28 | 437 ms | 输入侧正则 |

C1 多 seed 稳定性（3 种子 mean ± std）：
- pred_bbox: 0.393 ± 0.025
- structured_evidence: **0.443 ± 0.012**
- lowres_full_highrescrop: 0.363 ± 0.006

C2 新算法（bbox 评分门控）：
- baseline (无门控): 0.41
- default θ=0.5: 0.42（回退 54%）
- strict θ=0.7: **0.43**（回退 79%，幸存样本 IoU≥0.5 比例从 0.32 升到 0.67）
- oracle fallback: **0.46**（+0.05 vs baseline）

---

## 9. 未完成（用户待办）

> [!warning] 这些是机器无法自动完成的事，需要用户在 Word 里手动操作

- [ ] **TOC 重建**：v6 docx 打开后右键目录 → 更新整个目录（含新增的 §4.4.1）
- [ ] **接受/拒绝 tracked changes**：§2 / §3 / §7 三处章名修订建议
- [ ] **图编号校对**：当前用了占位编号（图 3.7 / 4.6 / 5.4 / 5.5 / 5.6），请按你章内实际编号调整
- [ ] **通读修订不通顺句子**：导师指出的"边界框已经是推理流程的一部分"、"而是利用已有区域定位阶段提供的边界框" 等仍需人工通读
- [ ] **答辩 PPT**：尚未生成（如需要可调用 nature-skills 的 nature-paper2ppt）

---

## 10. 已知问题与注意事项

- **`nature-skills` plugin 未激活**：marketplace 已 add，但 plugin 没有正式 install（用户拒绝过 settings.json 编辑）。当前所有"nature-figure"/"nature-polishing"按规则手动执行，效果等价。重新激活方法：`/plugin install nature-skills@nature-skills`。
- **anthropic `docx` skill 严格 tracked-change 校验**：`pack.py` 默认要求所有新增内容必须 wrap 在 `<w:ins>` 里，否则报"Document text doesn't match"。绕过方法：`--validate false`。
- **中文字体**：系统已装 `fonts-noto-cjk`（Noto Sans CJK），matplotlib 通过 `fm.fontManager.addfont()` 强制加载 TTC 后才能识别。脚本 `render_paper_figures_zh.py` 已包含此逻辑。
- **GPU**：RTX 3090 24GB，bf16，VisCoT-7b-224。**4-bit 路径有 dtype 问题**（detection 阶段 `float != Half`），暂时不要走 4-bit。
- **Pareto 图标签轻度重叠**：A4 Pareto 图在英文版有轻微标签重叠（中文版没用到该图所以未修复），如果以后要用可在 `render_paper_figures.py::figure_pareto` 的 `OFFS_LAT/OFFS_TOK` 字典里再加偏移。

---

## 11. 下一版（v7）触发条件

下一次进入仓库工作时，**如果用户提出以下需求**，按以下步骤推进：

1. **生成答辩 PPT** → 用 `nature-paper2ppt` 风格，基于 v6 主结果 + 图 4.6 / 图 5.6
2. **修改不通顺句子** → 用 docx skill tracked changes，精确替换原句
3. **扩展实验** → 在 `experiments/thesis_experiments/runs/20260522_round2_extensions/` **平级新建**新 timestamp 目录，不污染现有
4. **更换图风格** → 改 `render_paper_figures_zh.py` 的 `PALETTE` 字典即可统一所有图

每次完成后：
1. 把新文件按序号命名（`7论文…`, `7 修改说明…`）
2. v6 移入 `过程文档/`
3. 更新本 `STATUS.md` 的"§2 论文版本演进"、"§9 未完成事项"等表格

---

> **维护提示**：本文件由 AI 助手维护。每次进入仓库时先 `Read` 这份文件，结束工作前 `Edit` 把进度更新进来。
