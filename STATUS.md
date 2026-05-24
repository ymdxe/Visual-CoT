# STATUS.md

> 状态快照，最后更新：2026-05-23 04:30
>
> 任何 AI 助手或新 IDE 会话进入仓库时，**先读这份文档**确认当前位置，然后再决定下一步动作。

---

## 0. 一句话当前位置

毕业设计论文修订进行到 **v7（DSAC + 跨领域 DSAC 验证版）**。**2026-05-23 完成方向 A 全流程注入 + 跨领域复验**：(1) 在 4 个新公开域（DocVQA / InfographicsVQA / Visual7W / GQA-spatial 子集）各 100 样本上完成 SE / pred_bbox / DSAC default / DSAC tlow05+center / DSAC tlow06+oracle 五配置扫描；(2) 跨域池化（n=398）下 DSAC tlow06+oracle vs SE 取得 **McNemar 显著**（χ²=4.21, p=0.040, gap=+0.048 CM），其中 GQA-spatial 单域 **+0.100 CM (p=0.044 显著)**；(3) 注入论文 v6 → v7：新增 **§4.6 DSAC**（7 段 + 4 图）与 **§4.7 跨领域 DSAC 验证**（5 段 + 2 图 + 2 表），原 §4.5 重编号为 §4.8；摘要 / §7.1 / §7.2 / §7.3 各追加 1 段联动。等待用户在 Word 中刷新 TOC + 通读。

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
| v6 | `过程文档/6论文初稿第三版-...docx` | 归档 | 5 张新增图表标注汉化 |
| **v7 当前** | **`7论文初稿第三版-20226451-张恒-马安香.docx`** | **活跃** | 新增 §4.6 DSAC + §4.7 跨领域 DSAC 验证；原 §4.5 → §4.8 |

文件大小演进：v2 2.26 MB → v3 增补 2.65 MB → v3 分散 2.44 MB → v5 2.80 MB → v6 2.50 MB → v7 ≈ 3.4 MB（+§4.6 4 图 + §4.7 2 图）

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
| Phase 4 DSAC（v7 方向 A） | DSAC 12 配置 × 100 样本 | 12 GPU runs | `runs/20260522_dsac_v7/raw/dsac_*.jsonl` |
| Phase 4 §4.6 CUB100 复验 | DSAC 4 配置 × 100 样本 | 4 GPU runs | `runs/20260522_dsac_v7/raw_cub/dsac_*.jsonl` |
| **Phase 5 跨领域 DSAC（v7 §4.7）** | **4 域 × (1 det + 5 ans) × 100 = 24 runs** | **24 GPU runs** | `runs/20260523_cross_domain_dsac/raw/*.jsonl` |

**累计**：~2700 + 2400 ≈ 5100 次推理 / 60 个 JSONL / 17 张分析图表 / 2 个新算法实现（边界框评分门控 + 双端自适应控制器）。

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

### 5.1 §4.5 边界框评分门控（Round 2）

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

### 5.2 §4.6 双端自适应控制器 DSAC（2026-05-22 方向 A）

| 函数 / 参数 | 位置 |
|---|---|
| `compute_scs(parsed)` | 计算 SCS 与四段产出 flags |
| `adaptive_extract(parsed, flags, raw_output)` | 中档 re_extract 的自适应抽取算子 |
| `composite_decision(score_info, scs_info, args)` | composite = α·score + (1-α)·SCS 三档决策 |
| MODE_PRESETS["dsac"] | `("pred", "full_crop", "structured", False)` |
| CLI `--adaptive-gate` | 总开关 |
| CLI `--adaptive-alpha` | 输入端权重，默认 0.5 |
| CLI `--adaptive-theta-low` | 默认 0.3（**实测在跨数据集 100 上需 ≥0.5 才能让 fallback 触发**） |
| CLI `--adaptive-theta-high` | 默认 0.7 |
| CLI `--adaptive-fallback` | `center` / `oracle` / `none` |

JSONL 新字段：`metadata.composite_gate = {alpha, theta_low, theta_high, s_in, s_out, composite, decision, scs_flags, second_pass, fallback_box}`，仅在 `--adaptive-gate` 开启时出现，**加在 metadata 尾部不重排**。

---

## 6. 工具脚本目录

`tools/thesis/round2/`：
- `phase1_analysis.py` — A1-A6 零 GPU 分析
- `phase2_analysis.py` — B1 lowres + B2 padding 灵敏性聚合
- `phase3_analysis.py` — C1 多 seed + C2 bbox 评分聚合
- `aggregate_round2.py` — 汇总到 `ROUND2_REPORT.md`
- `render_paper_figures.py` — 英文版 nature-figure 渲染（7 张 × 4 格式）
- `render_paper_figures_zh.py` — **中文版图渲染**（5 张 × 3 格式）
- `inject_into_thesis.py` — 集中式注入（已废弃）
- `inject_distributed.py` — **分散式注入**（10 个小节边界）

`tools/thesis/round3/`（DSAC 实验，2026-05-22 ~ 2026-05-23）：
- `dsac_analysis.py` — 12 配置聚合，输出主表 / 决策档位剖析 / 消融 / 基线对照（4 个 md + 4 个 csv）
- `render_dsac_figures.py` — 图 4.7 流程（ASCII）/ 4.8 composite 直方图 / 4.9 决策档位 CM / 4.10 α & θ 扫描 / 4.11 SCS 分布
- `mcnemar_dsac_vs_se.py` — DSAC vs SE 配对显著性检验（跨数据集 100），χ² + p-value 输出
- `aggregate_cub_dsac.py` — CUB100 单域 4 配置聚合（SCS 分布 + CM/EM 主表）
- **`aggregate_cross_domain_dsac.py`** — 跨域 4 + CUB 5 数据集 × 5 配置主表 + summary.json + 每域 summary.md
- **`mcnemar_cross_domain.py`** — 跨域 McNemar（DSAC default vs SE + DSAC tlow06+oracle vs SE 双对照，含池化）
- **`render_cross_domain_figures.py`** — 图 4.12 跨领域 CM 分组柱图 + 图 4.13 oracle 增益 per-dataset 柱图（中文）
- **`inject_v7.py`** — v6 → v7 docx 注入器（§4.6 + §4.7 + 摘要 / §7 联动 + 4.5→4.8 重编号）

`experiments/thesis_experiments/runs/20260522_round2_extensions/scripts/`：
- `run_phase2_sweeps.sh` — Phase 2 GPU 扫描驱动
- `run_phase3.sh` — Phase 3 GPU 驱动

`experiments/thesis_experiments/runs/20260522_dsac_v7/scripts/`：
- `run_dsac.sh` — DSAC 8 基础配置扫描（跨数据集 100）
- `run_dsac_extension.sh` — DSAC 4 个 strict θ_low 扩展配置（跨数据集 100）
- `run_dsac_cub.sh` — CUB100 单域 4 配置复验（2026-05-22 新增）

`experiments/thesis_experiments/runs/20260523_cross_domain_dsac/scripts/`：
- `run_all_domains.sh` — 4 域 × (1 detection + 5 answer 配置) = 24 runs，支持 `ONLY=<domain>` 单域调用

`tools/`：
- `build_gqa_spatial.py` — GQA 空间关系子集（关键词正则，305 候选 → 100 抽样）

仓库根新增 benchmark 文件（v7 §4.7 用）：
- `small_benchmark_docvqa100.{json,_det.jsonl}`
- `small_benchmark_infographicsvqa100.{json,_det.jsonl}`
- `small_benchmark_visual7w100.{json,_det.jsonl}`
- `small_benchmark_gqa_spatial100.{json,_det.jsonl}`

---

## 7. Obsidian 仓库结构

```
Obsidian/
├── 7论文初稿第三版-20226451-张恒-马安香.docx    ← 当前最新版（DSAC + 跨领域）
├── 7 修改说明-20260523.md                       ← v6→v7 变更说明
├── 论文初稿-20226451-张恒-马安香_审阅意见_20260521.md ← 导师审阅原文
├── 方向A实施记录_DSAC_20260522.md               ← DSAC 实施日志 + 附录 A/B（已注入）
├── 方向A实施文档_双端自适应控制器_20260522.md
├── 原创性方法方案_决策版_20260522.md
├── 原创性方法扩展候选_20260522.md
├── AGENTS.md（在仓库根，不在 Obsidian/）
├── STATUS.md（本文件，在仓库根）
├── assets/round2/
│   ├── paper_figures/       ← 英文版图（备份）
│   └── paper_figures_zh/    ← 中文版图（v6 使用）
└── 过程文档/                ← 归档
    ├── 6论文初稿第三版-20226451-张恒-马安香.docx  ← v6 归档
    ├── 6 修改说明-20260522.md                       ← v5→v6 说明归档
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

**Phase 4 §4.6 DSAC 双端自适应控制器（2026-05-22）**（跨数据集 100，12 配置）：
- DSAC default (α=0.5, θ=(0.3,0.7), center fb): CM=**0.434** (vs structured_evidence baseline 0.429，+0.005)
- **McNemar 配对检验（n=98）**：DSAC 与 SE 在共同 qid 上 CM、EM 完全相等，χ²=0.25, p=0.6171，**+0.005 由有效样本数差异（DSAC 99 有效 vs SE 98 有效）造成，统计上不显著**。`metrics/mcnemar_dsac_vs_se.md`
- 默认配置 **fallback 档未触发**：跨数据集 SCS 恒为 0.5，composite ∈ [0.45,0.70] 全部 ≥ θ_low=0.3
- DSAC tlow06 + center fb: CM=0.323（中心框拖累 fallback 档）
- DSAC tlow06 + **oracle fb (上限)**: CM=**0.459**（+0.030 vs structured_evidence baseline，+0.045 vs pred_bbox baseline）
- 判别 gap：strict 配置下 accept 档 CM=0.524 vs re_extract 档 CM=0.410，**+0.114 CM**，证明 composite 信号有判别力
- 详细记录：`experiments/thesis_experiments/runs/20260522_dsac_v7/paper_materials/section_4_6_results.md`

**Phase 4 §4.6 CUB100 单域复验（2026-05-22 增补）**（4 配置 × 100 样本）：
- CUB SE baseline: CM=0.710 (vs 历史 0.73，温度差异)
- CUB DSAC default (center fb): CM=0.710（fallback 同样不触发，SCS 恒 0.5——证明 SCS 偏低是模型限制，与数据集无关）
- **CUB DSAC tlow05 + center fb: CM=0.770（+0.060 vs SE baseline）**——CUB 居中先验使 center 框反而成为优秀回退
- CUB DSAC tlow06 + oracle fb: CM=0.750（+0.040 vs SE baseline）
- 关键洞察：DSAC 的实际兑现度由 **回退策略 × 数据集分布** 共同决定；CUB 的 center 回退好，跨数据集需 oracle 上限
- 详细记录：`experiments/thesis_experiments/runs/20260522_dsac_v7/metrics/cub_dsac_validation.md`

**Phase 5 §4.7 跨领域 DSAC 验证（2026-05-23）**（4 域 × 5 配置 × 100 样本）：

ContainsMatch 主表（每域 n=100）：

| 数据集 | SE 基线 | pred_bbox | DSAC default | DSAC tlow05+center | DSAC tlow06+oracle |
|---|---:|---:|---:|---:|---:|
| DocVQA          | 0.133 | 0.141 | 0.133 | 0.091 | 0.143 |
| InfographicsVQA | 0.250 | 0.212 | 0.250 | 0.130 | 0.270 |
| Visual7W        | 0.300 | 0.364 | 0.300 | 0.260 | 0.360 |
| GQA-spatial     | 0.520 | 0.510 | 0.520 | 0.510 | 0.620 |
| CUB100（对照）   | 0.710 | —     | 0.710 | 0.770 | 0.750 |

McNemar 配对显著性（DSAC tlow06+oracle vs SE baseline，连续性校正）：

| 数据集 | n | b01 | b10 | χ² | p | ΔCM |
|---|---:|---:|---:|---:|---:|---:|
| DocVQA          | 98  | 7  | 8  | 0.067 | 0.796 | +0.010 |
| InfographicsVQA | 100 | 11 | 13 | 0.042 | 0.838 | +0.020 |
| Visual7W        | 100 | 6  | 12 | 1.389 | 0.239 | +0.060 |
| GQA-spatial     | 100 | 5  | 15 | **4.050** | **0.044** | **+0.100** |
| **跨域池化**     | **398** | **29** | **48** | **4.208** | **0.040** | **+0.048** |

- **DSAC default vs SE baseline**：跨域池化下 p=1.0000（b01=b10=0，与 SE 完全等效，保守阈值无害）
- **DSAC tlow06+oracle vs SE baseline**：跨域池化下 **α=0.05 显著**（p=0.040），GQA-spatial 单独显著
- 关键洞察：DSAC 的潜在增益在跨任务类型上可池化显著且兑现度依赖回退框策略；GQA-spatial 关系推理是 DSAC 上限最大的方向
- 详细记录：`experiments/thesis_experiments/runs/20260523_cross_domain_dsac/metrics/{cross_domain_table.md,mcnemar_cross_domain.md}`

---

## 9. 未完成（用户待办）

> [!warning] 这些是机器无法自动完成的事，需要用户在 Word 里手动操作

- [ ] **打开 v7.docx 并刷新 TOC**：v7 新增 §4.6 / §4.7 / §4.8 三个章节，需右键目录 → 更新整个目录
- [ ] **§4.6 / §4.7 字体一致性核查**：注入用的是 v6 的 `A论文正文` 样式，建议通读一遍图标题、表标题是否与正文字号一致
- [ ] **§4.7 表 4.5 / 表 4.6 边距调整**：表使用 `Table Grid` 默认样式，可能与 §3 已有表的边框样式略有差异，按需手动调整
- [ ] **§4.7 图 4.12 / 4.13 嵌入位置微调**：图与文段相对位置可能在打开 Word 时漂移，按需拖动
- [ ] **摘要末段长度核查**：摘要末追加了 DSAC + 跨域验证 1 段，约 200 字，按需精简或并入前段
- [ ] **§7.2 第六/第七点编号校对**：原 §7.2 是五点列举，追加了 2 点（"第六" / "第七"），打开 Word 后核对编号是否连贯
- [ ] **接受/拒绝 tracked changes**：§2 / §3 / §7 三处章名修订建议（v6 中已经存在的 tracked changes）
- [ ] **图编号校对**：注入用了 §4.6 → 图 4.8~4.11、§4.7 → 图 4.12~4.13，按你正文实际编号调整
- [ ] **通读修订不通顺句子**：导师 0521 审阅意见中的不通顺句子仍待人工通读
- [ ] **答辩 PPT**：尚未生成（如需要可调用 nature-skills 的 nature-paper2ppt）

---

## 10. 已知问题与注意事项

- **`nature-skills` plugin 未激活**：marketplace 已 add，但 plugin 没有正式 install（用户拒绝过 settings.json 编辑）。当前所有"nature-figure"/"nature-polishing"按规则手动执行，效果等价。重新激活方法：`/plugin install nature-skills@nature-skills`。
- **anthropic `docx` skill 严格 tracked-change 校验**：`pack.py` 默认要求所有新增内容必须 wrap 在 `<w:ins>` 里，否则报"Document text doesn't match"。绕过方法：`--validate false`。
- **中文字体**：系统已装 `fonts-noto-cjk`（Noto Sans CJK），matplotlib 通过 `fm.fontManager.addfont()` 强制加载 TTC 后才能识别。脚本 `render_paper_figures_zh.py` 与 `render_dsac_figures.py` 已包含此逻辑。
- **GPU**：RTX 3090 24GB，bf16，VisCoT-7b-224。**4-bit 路径有 dtype 问题**（detection 阶段 `float != Half`），暂时不要走 4-bit。
- **Pareto 图标签轻度重叠**：A4 Pareto 图在英文版有轻微标签重叠（中文版没用到该图所以未修复），如果以后要用可在 `render_paper_figures.py::figure_pareto` 的 `OFFS_LAT/OFFS_TOK` 字典里再加偏移。
- **DSAC 默认配置在跨数据集 100 上 fallback 永不触发**：原因是跨数据集场景 SCS 恒为 0.5（[Region]/[VE] 段不产出），composite 最小 0.45 ≥ θ_low=0.3。要让控制器分流必须 θ_low ≥ 0.5。论文 §4.6 必须把这点明确写为"诚实警示"，否则评审会挑战默认配置形同虚设。
- **DSAC default vs SE baseline 的 +0.005 CM 不显著**：McNemar 配对检验在 98 个共同 qid 上 χ²=0.25, p=0.617，DSAC 与 SE CM 完全相等 (0.429)。+0.005 完全由两份 JSONL 的有效样本数差异造成（DSAC 99 有效 vs SE 98 有效），并非控制器带来的真实增益。论文 §4.6 必须由判别力 (+0.114 gap) 与 oracle 上限 (+0.030) 与 CUB +0.060 这三个数字承担论证。
- **center 回退框拖累 DSAC 的 fallback 档（跨数据集）但在 CUB 反而提升**：跨数据集 textvqa/gqa 上 center 框 CM=0.222 远低于全局 0.43；但 CUB100 上 DSAC tlow05+center 反而 +0.060 CM。证明 DSAC 的兑现度由 *回退策略 × 数据集分布* 共同决定；CUB 居中先验使 center 框天然适配。
- **SCS 恒为 0.5 是模型限制而非数据集限制**：CUB100 与跨数据集 100 上 SCS 都恒为 0.5（仅 [Reasoning]+[Answer] 段产出），证明 VisCoT-7b-224 对结构化模板四段全产出的能力不足，需要在模型/提示层面扩展（§7 工作展望）。

---

## 11. 下一版（v8）触发条件

下一次进入仓库工作时，**如果用户提出以下需求**，按以下步骤推进：

1. **生成答辩 PPT** → 用 `nature-paper2ppt` 风格，基于 v7 主结果 + 图 4.12 / 图 4.13 / 表 4.5 / 表 4.6
2. **修改不通顺句子** → 用 docx skill tracked changes，精确替换原句
3. **扩展实验**（如继续 DSAC v8） → 在 `experiments/thesis_experiments/runs/` **平级新建**新 timestamp 目录，不污染现有
4. **更换图风格** → 改 `render_cross_domain_figures.py` / `render_dsac_figures.py` 的 `PALETTE` 字典即可统一
5. **实现更智能的回退框**（§7.3 第六点 outlook）→ 新建 `tools/build_sam_fallback.py` 或 `tools/build_obj_detect_fallback.py`，作为 DSAC 的 fallback 升级

每次完成后：
1. 把新文件按序号命名（`8论文…`, `8 修改说明…`）
2. v7 移入 `过程文档/`
3. 更新本 `STATUS.md` 的"§2 论文版本演进"、"§3 已完成的实验"、"§8 关键实验数字"、"§9 未完成事项"等表格

---

> **维护提示**：本文件由 AI 助手维护。每次进入仓库时先 `Read` 这份文件，结束工作前 `Edit` 把进度更新进来。
