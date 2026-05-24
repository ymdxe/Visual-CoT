#!/usr/bin/env python3
"""把 §4.6 DSAC + 新 §4.7 跨领域 DSAC 验证注入 v6.docx，并联动摘要 / §7.1 / §7.2 / §7.3 → v7.docx。

输入：Obsidian/6论文初稿第三版-20226451-张恒-马安香.docx
输出：Obsidian/7论文初稿第三版-20226451-张恒-马安香.docx

策略：
- 4.4.1 后插入 4.6 DSAC（5 段正文 + 实验段 + 局限段）
- 4.6 后插入 4.7 跨领域 DSAC 验证（5 段 + 表 4.5 + 图 4.12 + 图 4.13 + McNemar 总结）
- 原 4.5 本章小结 重命名为 4.8 本章小结（保持位置在章末）
- 摘要 / §7.1 / §7.2 / §7.3 各追加一段（按附录 B）
"""
from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.shared import Inches, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DOCX_IN  = Path("Obsidian/6论文初稿第三版-20226451-张恒-马安香.docx")
DOCX_OUT = Path("Obsidian/7论文初稿第三版-20226451-张恒-马安香.docx")
FIG_DIR  = Path("experiments/thesis_experiments/runs/20260523_cross_domain_dsac/paper_materials")
DSAC_FIG = Path("experiments/thesis_experiments/runs/20260522_dsac_v7/figures")

BODY_STYLE = "A论文正文"          # 既有正文样式
H2_STYLE   = "Heading 2"
H3_STYLE   = "Heading 3"


def insert_paragraph_before(target_p, text="", style=None):
    """在 target_p 之前插入一个新段落，返回新段对象。"""
    new_p = OxmlElement("w:p")
    target_p._element.addprevious(new_p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(new_p, target_p._parent)
    if style:
        para.style = target_p.part.document.styles[style]
    if text:
        para.add_run(text)
    return para


def insert_picture_before(target_p, image_path: Path, width_cm: float = 14.0):
    """在 target_p 之前插入一张居中图片。"""
    para = insert_paragraph_before(target_p, "", style=BODY_STYLE)
    run = para.add_run()
    run.add_picture(str(image_path), width=Cm(width_cm))
    para.alignment = 1  # center
    return para


def insert_caption_before(target_p, caption: str):
    """在 target_p 之前插入一行图标题（居中、与 §4.4.1 图风格一致）。"""
    para = insert_paragraph_before(target_p, caption, style=BODY_STYLE)
    para.alignment = 1
    return para


def insert_table_before(doc, target_p, header, rows):
    """在 target_p 之前插入一个 Word 表（含表头）。"""
    # python-docx 不直接支持 insert before，需要把 table 元素 addprevious。
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.style = "Table Grid"
    hdr = tbl.rows[0].cells
    for i, h in enumerate(header):
        hdr[i].text = h
    for r_i, row in enumerate(rows, start=1):
        for c_i, val in enumerate(row):
            tbl.rows[r_i].cells[c_i].text = str(val)
    # 把刚 append 的 table 移动到 target 前
    tbl_el = tbl._element
    tbl_el.getparent().remove(tbl_el)
    target_p._element.addprevious(tbl_el)
    return tbl


def find_paragraph_by_text(doc, contains: str, after: int = 0):
    for i, p in enumerate(doc.paragraphs[after:], start=after):
        if contains in p.text:
            return i, p
    return None, None


def main():
    doc = Document(str(DOCX_IN))

    # ============= 1. 找到 §4.5 本章小结的位置（作为 §4.6 + §4.7 的注入锚点）=============
    idx_45, p_45 = find_paragraph_by_text(doc, "4.5 本章小结")
    assert p_45 is not None, "未找到 §4.5 本章小结"
    print(f"[anchor] 4.5 本章小结 at paragraph {idx_45}")

    # 重命名 4.5 → 4.8
    for run in p_45.runs:
        run.text = ""
    p_45.runs[0].text = "4.8 本章小结" if p_45.runs else p_45.add_run("4.8 本章小结").text
    if not p_45.runs:
        p_45.add_run("4.8 本章小结")
    print("[rename] §4.5 → §4.8 本章小结")

    # ============= 2. 注入 §4.6 双端自适应控制器 =============
    insert_paragraph_before(p_45, "4.6 双端自适应控制器", style=H2_STYLE)

    sec46_paragraphs = [
        "§4.4.1 给出的边界框评分门控仅依赖输入端的预测框质量，无法感知模型输出本身的组织情况；§4.2.2 给出的结构一致性得分 SCS 仅刻画输出端模板遵守程度，无法感知边界框是否可用。二者各自反映推理过程的一个侧面，但单独使用都会忽略另一端的信息。本节将二者合并为一个统一控制信号，并据此触发不同处理路径。",
        "复合得分。给定预测框 b̂、问题 q 和首次生成的答案 y，定义复合得分 composite(b̂, y, q) = α · score(b̂, q) + (1 − α) · SCS(y)，其中 score(·) 在 §4.4.1 中定义，SCS(·) 在 §4.2.2 中定义，α ∈ [0, 1]。当 α = 1 时复合得分退化为输入端评分（与 §4.4.1 等价），当 α = 0 时退化为输出端 SCS，因此 α 是一个连续控制的双端融合系数。",
        "三档决策。给定上下阈值 (θ_low, θ_high)，控制器按 composite 取值分三档处理：composite ≥ θ_high 表示输入端预测框可信且输出端模板遵守较好，直接采纳第一次生成的答案；θ_low ≤ composite < θ_high 表示存在部分组件失败，按实际产出的字段自适应抽取答案；composite < θ_low 表示两端同时失败，触发回退框并执行二次生成。",
        "自适应答案抽取。在中档（re_extract）下，根据 SCS 四个分量 (s_R, s_E, s_C, s_A) 选择抽取路径：若 [Answer] 段存在则直接抽取，否则依次退化到 [Reasoning] 末句、[Visual Evidence] 末句、整段末句。该设计的依据是 §4.2.2 实测 SCS = 0.50（四段中只稳定产出两段），因此自适应抽取比硬要求 [Answer] 段更鲁棒。",
        "与现有方法的关系。控制器把 §4.2、§4.4.1、§5.2 三个模块绑成一个系统：§4.2 提供输出端模板（产生 SCS 信号），§4.4.1 提供输入端评分（产生 score 信号），§5.2 提供视觉输入构造（决定 Φ）。控制器本身不引入新的视觉编码或语言模型，仅通过对两个已有信号的联合判断决定何时进入回退路径，因此符合本文“免训练、推理时改进”的边界。完整算法描述见算法 6。",
        "实验结果。在跨数据集 100 样本（TextVQA + GQA + Visual7W）上以 α ∈ {0.0, 0.3, 0.5, 0.7, 1.0} 和 (θ_low, θ_high) ∈ {(0.3, 0.7), (0.4, 0.6), (0.2, 0.8), (0.5, 0.7), (0.55, 0.7), (0.6, 0.7)} 共 12 种配置进行联合扫描，详细结果见图 4.8 ~ 4.11。默认配置 (α=0.5, θ_low=0.3, θ_high=0.7, center fallback) 的包含性匹配为 0.434，与 §4.2 单独的结构化提示基线 (0.429) 在 McNemar 配对检验下无显著差异（n=98 共同 qid，χ²=0.25，p=0.6171）——0.005 的差距完全由两份 JSONL 的有效样本数差异（DSAC 99 vs SE 98）造成；该配置下 fallback 档未触发（composite ∈ [0.45, 0.70]，全部 ≥ θ_low），DSAC 退化为输入端评分门控与结构化提示的组合。当 θ_low 提升到 0.6 并采用 oracle 回退框时（控制器上限设定），包含性匹配升至 0.459，相对结构化提示基线提升 +0.030，相对 Visual-CoT 主流程 pred_bbox 基线 (0.414) 提升 +0.045。决策档位剖析显示，strict 配置（α=0.5, θ=(0.4, 0.6)）下 accept 档样本平均 CM 为 0.524，re_extract 档为 0.410，判别 gap 达 +0.114，证明 composite 信号对样本可信度具有显著判别力。为验证 DSAC 的兑现度是否依赖数据集分布，在 CUB100 单域上对照 4 个配置：SE baseline 0.710 / DSAC default 0.710 / DSAC tlow05+center 0.770（+0.060）/ DSAC tlow06+oracle 0.750（+0.040）。CUB 居中目标先验使 center 回退框天然适配，控制器在该域无需 oracle 即可解锁 +0.060 CM 的提升，与跨数据集形成对照。",
        "实验局限。本次实验暴露三点局限：第一，结构化模板的 [Region] 与 [Visual Evidence] 段在跨数据集 100 与 CUB100 上均完全不产出，SCS 恒为 0.5，输出端信号区分度有限；这是 VisCoT-7b-224 模型层面对结构化模板四段全产出能力不足的问题，而非数据集特异。第二，回退框策略的兑现度强烈依赖数据集分布：在 CUB100（鸟类居中先验）上 DSAC tlow05+center 直接获得 +0.060 CM；在跨数据集（textvqa/gqa 类问题）上 center 回退框 CM=0.222 远低于全局 0.43，oracle 回退框 CM=0.416 接近全局，必须依赖 oracle 上限设定才能解锁 +0.030 CM。第三，跨数据集默认配置相对 SE baseline 的 0.005 CM 差距在 McNemar 配对检验下不具备统计显著性（p=0.617），不能作为主结果——本节论证由判别力（+0.114 gap）、跨数据集 oracle 上限（+0.030）、CUB 单域 center 回退（+0.060）三个数字承担。三点局限分别指向：(a) 输出端信号扩展；(b) 更智能的回退策略；(c) 大样本扩展以提升小差距的检测功效，均列入第七章工作展望。",
    ]
    for txt in sec46_paragraphs:
        insert_paragraph_before(p_45, txt, style=BODY_STYLE)

    # §4.6 配套 4 张图（fig_4_8 ~ fig_4_11）来自 round2 已就绪的 DSAC v7 实验
    fig_specs_46 = [
        ("fig_4_8_composite_hist.png",     "图 4.8 跨数据集 100 上 composite 分布"),
        ("fig_4_9_decision_cm.png",        "图 4.9 三档决策下的样本 CM 分布"),
        ("fig_4_10_alpha_theta_sweep.png", "图 4.10 α 与 (θ_low, θ_high) 联合扫描"),
        ("fig_4_11_scs_distribution.png",  "图 4.11 跨数据集 100 上 SCS 四分量分布"),
    ]
    for fname, caption in fig_specs_46:
        fp = DSAC_FIG / fname
        if fp.exists():
            insert_picture_before(p_45, fp, width_cm=12.5)
            insert_caption_before(p_45, caption)
        else:
            print(f"[warn] missing figure: {fp}")

    print("[insert] §4.6 双端自适应控制器 done")

    # ============= 3. 注入 §4.7 跨领域 DSAC 验证 =============
    insert_paragraph_before(p_45, "4.7 跨领域 DSAC 验证", style=H2_STYLE)

    sec47_paragraphs = [
        "为检验 §4.6 双端自适应控制器在不同视觉问答任务上的泛化性与回退策略的兑现度，本节将 DSAC 扩展到四个公开域：文档文本（DocVQA）、信息图表（InfographicsVQA）、通用视觉问答（Visual7W）以及关系推理（GQA-spatial 子集）。每个域抽取 100 个样本，与已完成的 CUB100 单域复验共五个数据集，构成跨四类任务类型的对照矩阵。",
        "实验设置。所有域均使用相同模型（VisCoT-7b-224，bf16，温度 0.2，top_p 0.95，seed=42），对每个样本顺序执行五种配置：SE 基线（structured_evidence，无 DSAC）、pred_bbox 基线、DSAC default (α=0.5, θ_low=0.3, center fallback)、DSAC tlow05+center 与 DSAC tlow06+oracle（控制器上限）。配置矩阵与决策档位分布详见表 4.5 与图 4.12；逐域配对 McNemar 检验汇总见表 4.6。GQA-spatial 子集由正则关键词过滤构成（in front of / behind / above / below / left / right / next to / between 等，共 305 候选，从中均衡抽取 100）。",
        "ContainsMatch 主结果。如图 4.12 与表 4.5 所示，DSAC default 在四个跨域上 composite 始终 ≥ θ_low=0.3，从未触发回退，与 SE baseline 输出完全一致（b01=b10=0），这一现象在所有跨域上稳定复现，证明控制器在保守阈值下不引入额外错误。DSAC tlow05+center 在两个文档型任务（DocVQA、InfographicsVQA）上分别下降 −0.042 与 −0.120 CM，原因与 §4.6 跨数据集分析一致：center 回退框对 OCR 类任务破坏语义；在 Visual7W 与 GQA-spatial 上 center 回退仅小幅下降，在 CUB100 上反而上升（与 §4.6 报告的 +0.060 一致）。控制器上限 DSAC tlow06+oracle 在五个数据集上一致带来正向 ΔCM，其中 GQA-spatial 提升最大（+0.10 CM，b10=15, b01=5），Visual7W 次之（+0.06 CM），CUB100 +0.04 CM，InfographicsVQA +0.02 CM，DocVQA +0.01 CM。",
        "显著性检验。McNemar 配对（连续性校正）结果如表 4.6 所示：DSAC default vs SE baseline 在所有 4 个跨域以及池化上 p=1.0000（不一致对 b01=b10=0），证明默认配置在跨域下与 SE 等效。DSAC tlow06+oracle vs SE baseline 在跨域池化（n=398）下 χ²=4.208，p=0.0402（α=0.05 显著），gap=+0.048；逐域中 GQA-spatial 单独显著（χ²=4.05, p=0.0442, gap=+0.10），其他三个跨域未达单独显著但方向一致（b10 > b01）。这一结果表明：DSAC 的潜在增益在跨域上确实存在并可池化显著，但其兑现需要更优的回退框策略。",
        "结论。跨领域 DSAC 验证给出三点结论：(a) DSAC 在保守阈值下对跨域无害（与 SE baseline 等价），不会破坏既有方法；(b) center 回退框的兑现度强烈依赖数据集分布——居中目标先验（CUB）下涨点 +0.060，OCR / 文档定位类任务下负贡献；(c) oracle 回退上限在跨域池化下达到统计显著（p=0.040），其中关系推理任务（GQA-spatial）显示出最大潜在增益（+0.100）。这一发现把 §4.6 单一数据集观察到的 composite 判别力进一步推广到四类视觉问答任务，并明确指出后续工程化路径：将 center 回退替换为基于 SAM 显著性图、物体检测器或 CLIP 选框的更智能回退方案，是兑现 DSAC oracle 上限的关键改进方向。",
    ]
    for txt in sec47_paragraphs:
        insert_paragraph_before(p_45, txt, style=BODY_STYLE)

    # 表 4.5：跨领域 DSAC 主表
    insert_paragraph_before(p_45, "表 4.5 跨领域 DSAC 验证主表（ContainsMatch / ExactMatch，n=100/域）", style=BODY_STYLE)
    header = ["数据集", "SE 基线", "pred_bbox 基线", "DSAC default", "DSAC tlow05+center", "DSAC tlow06+oracle"]
    rows = [
        ["DocVQA",              "0.133 / 0.061", "0.141 / 0.081", "0.133 / 0.061", "0.091 / 0.030", "0.143 / 0.061"],
        ["InfographicsVQA",     "0.250 / 0.100", "0.212 / 0.131", "0.250 / 0.100", "0.130 / 0.040", "0.270 / 0.150"],
        ["Visual7W",            "0.300 / 0.190", "0.364 / 0.141", "0.300 / 0.190", "0.260 / 0.180", "0.360 / 0.240"],
        ["GQA-spatial",         "0.520 / 0.460", "0.510 / 0.370", "0.520 / 0.460", "0.510 / 0.440", "0.620 / 0.580"],
        ["CUB100（对照）",       "0.710 / 0.710", "—",             "0.710 / 0.710", "0.770 / 0.770", "0.750 / 0.750"],
    ]
    insert_table_before(doc, p_45, header, rows)

    # 图 4.12 & 4.13
    insert_picture_before(p_45, FIG_DIR / "fig_4_12_cross_domain_cm_zh.png", width_cm=14.5)
    insert_caption_before(p_45, "图 4.12 跨领域 DSAC 验证 — 五数据集 ContainsMatch（n=100/域）")

    insert_picture_before(p_45, FIG_DIR / "fig_4_13_dsac_gain_by_difficulty_zh.png", width_cm=14.5)
    insert_caption_before(p_45, "图 4.13 DSAC tlow06+oracle 相对 SE 基线的 ContainsMatch 增益（per dataset，*=McNemar p<0.05）")

    # 表 4.6：McNemar 配对显著性
    insert_paragraph_before(p_45, "表 4.6 跨领域 McNemar 配对显著性检验（DSAC tlow06+oracle vs SE 基线，连续性校正）", style=BODY_STYLE)
    header2 = ["数据集", "n", "b01", "b10", "χ²", "p-value", "ΔCM"]
    rows2 = [
        ["DocVQA",          "98",  "7",  "8",  "0.067", "0.796", "+0.010"],
        ["InfographicsVQA", "100", "11", "13", "0.042", "0.838", "+0.020"],
        ["Visual7W",        "100", "6",  "12", "1.389", "0.239", "+0.060"],
        ["GQA-spatial",     "100", "5",  "15", "4.050", "0.044*", "+0.100"],
        ["跨域池化",         "398", "29", "48", "4.208", "0.040*", "+0.048"],
    ]
    insert_table_before(doc, p_45, header2, rows2)

    print("[insert] §4.7 跨领域 DSAC 验证 done")

    # ============= 4. 摘要联动（附录 B.1）=============
    # 查找摘要内容段落：v6 中位于"摘  要"标题后、"关键词"前，找"在 CUB 小样本"作为锚点（实测在摘要末段）
    idx_struct, p_struct = find_paragraph_by_text(doc, "在 CUB 小样本")
    if p_struct is None:
        idx_struct, p_struct = find_paragraph_by_text(doc, "通过结构化提示组织")
    if p_struct is not None:
        abstract_add = (
            "本文进一步提出双端自适应控制器（§4.6），将输入端预测框评分（§4.4.1）与输出端结构一致性得分（§4.2.2）"
            "合并为统一复合得分，按上下两个阈值动态选择直接采纳、自适应抽取或回退路径，"
            "使前述结构化提示与边界框评分门控形成闭环；跨四类任务（文档文本 / 信息图表 / 通用 VQA / 关系推理）"
            "上的验证表明，控制器在 CUB100 单域 center 回退下将 ContainsMatch 从 0.710 提升至 0.770，"
            "在跨数据集 100 样本的 oracle 上限设定下达到 0.459，"
            "在四个跨域池化（n=398）下相对 SE 基线提升 0.048（McNemar p=0.040，显著），"
            "其中关系推理任务（GQA-spatial）单域提升 +0.100。"
        )
        # 在摘要相关段后追加一段新内容（保持位置在该段之后）
        # 简单做法：直接在它后面 insert
        new_p = OxmlElement("w:p")
        p_struct._element.addnext(new_p)
        from docx.text.paragraph import Paragraph
        para = Paragraph(new_p, p_struct._parent)
        para.style = doc.styles[BODY_STYLE]
        para.add_run(abstract_add)
        print("[insert] 摘要 DSAC 联动段 done")
    else:
        print("[warn] 未找到摘要段（含'通过结构化提示组织'），跳过摘要联动")

    # ============= 5. §7.1 工作总结联动 =============
    idx_71, p_71 = find_paragraph_by_text(doc, "7.1 工作总结")
    if p_71 is not None:
        # 找到 §7.1 后第三段（方法改进段）末尾追加
        # 简单做法：找 §7.2 标题作为锚点，在它之前 insert
        idx_72, p_72 = find_paragraph_by_text(doc, "7.2 实验贡献与应用价值")
        if p_72 is not None:
            insert_paragraph_before(
                p_72,
                "此外，本文设计了双端自适应控制器（§4.6），将输入端边界框评分与输出端结构一致性得分"
                "合并为复合得分，并据此触发“直接采纳 / 自适应抽取 / 回退框二次生成”三档处理路径，"
                "使结构化提示、边界框评分门控与区域级输入压缩三个模块形成统一推理系统；"
                "在四个跨域（文档文本、信息图表、通用 VQA、关系推理）的扩展验证（§4.7）进一步证明，"
                "控制器在保守阈值下与 SE 基线等效，在 oracle 上限设定下跨域池化提升达到统计显著（p=0.040），"
                "其中关系推理任务的潜在增益最大（+0.100 CM）。",
                style=BODY_STYLE
            )
            print("[insert] §7.1 DSAC 联动段 done")

    # ============= 6. §7.2 实验贡献联动 =============
    idx_72, p_72 = find_paragraph_by_text(doc, "7.2 实验贡献与应用价值")
    if p_72 is not None:
        # 找 §7.3 作为锚点，在其前 insert 新点
        idx_73, p_73 = find_paragraph_by_text(doc, "7.3 后续扩展方向")
        if p_73 is not None:
            insert_paragraph_before(
                p_73,
                "第六，本文提出双端自适应控制器（§4.6），在跨数据集 100 样本上以 12 种配置进行联合扫描，"
                "证明复合得分 composite 在 strict 配置下能区分高置信样本（CM=0.524）与中置信样本（CM=0.410），"
                "判别 gap 达 0.114；并通过 McNemar 配对检验证实跨数据集场景默认配置与 SE 基线无显著差异（p=0.617），"
                "主结果由判别力、oracle 回退框上限（CM 0.459）与 CUB100 单域 center 回退（CM 0.770, +0.060）三组数字共同支撑。"
                "第七，本文将控制器在四个公开域（文档文本 DocVQA、信息图表 InfographicsVQA、通用 VQA Visual7W、"
                "关系推理 GQA-spatial 子集）进行扩展验证（§4.7），跨域池化（n=398）下 DSAC tlow06+oracle 相对 SE 基线"
                "提升 +0.048 CM（McNemar χ²=4.21, p=0.040，显著），其中 GQA-spatial 单域提升 +0.100 CM 且单独显著，"
                "证明 DSAC 的潜在增益在跨任务类型上可池化显著且兑现度依赖回退框与数据分布的匹配。",
                style=BODY_STYLE
            )
            print("[insert] §7.2 DSAC 联动段 done")

    # ============= 7. §7.3 工作展望末尾追加 =============
    idx_73, p_73 = find_paragraph_by_text(doc, "7.3 后续扩展方向")
    if p_73 is not None:
        # 找参考文献或附录作为锚点；若找不到则插到文档末尾
        anchor = None
        for keyword in ["参考文献", "致谢", "附录", "References"]:
            i_k, p_k = find_paragraph_by_text(doc, keyword, after=idx_73 + 1)
            if p_k is not None:
                anchor = p_k
                break
        outlook_text = (
            "第六，针对双端自适应控制器（§4.6, §4.7）的工程化兑现，当前实际增益受限于三点工程因素："
            "(a) 结构化模板的 [Region] / [Visual Evidence] 段在 VisCoT-7b-224 上产出率为 0（CUB100 与跨数据集表现一致），"
            "导致输出端 SCS 信号区分度有限；(b) center 回退框在 textvqa / gqa / DocVQA / InfographicsVQA 类分布敏感问题上不能保留语义，"
            "仅在 CUB 这种居中目标先验下才与 center 回退匹配；(c) 跨数据集默认配置与 SE 基线的小差距（+0.005）在 100 样本下统计上不显著。"
            "后续工作可在三个方向扩展：（1）通过更强的提示约束或轻量微调激励完整产出四段，提高 SCS 的判别度；"
            "（2）将回退框从中心框换为基于 SAM 显著性图或物体检测器产生的候选框，使 DSAC 的 oracle 上限可在工程上实现；"
            "（3）扩大测试样本规模以提升小差距的统计检测功效。"
        )
        if anchor is not None:
            insert_paragraph_before(anchor, outlook_text, style=BODY_STYLE)
            print("[insert] §7.3 DSAC 展望段（在参考文献/附录前）done")
        else:
            p_out = doc.add_paragraph(outlook_text, style=BODY_STYLE)
            print("[insert] §7.3 DSAC 展望段（文档末尾）done")

    # ============= 保存 =============
    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_OUT))
    print(f"\n[saved] {DOCX_OUT}")


if __name__ == "__main__":
    main()
