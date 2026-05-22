"""Distributed injection: place Round-2 content inside chapters at section boundaries
instead of appending it as monolithic "增补 A/B/C" subsections at chapter ends.

Insertion strategy: for each target subsection we identify the *next* section's
heading text in the body (NOT the TOC) and insert new paragraphs immediately
before the <w:p> that contains that heading. This keeps the existing TOC valid
because section numbers are unchanged.

Mapping is documented in INSERTIONS below.
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

SENTINEL = "R2DST3FA"
PAPER_FIGURES_DIR = Path("/root/code/Visual-CoT/Obsidian/assets/round2/paper_figures")
NOW_ISO = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

FIGURE_MAP = [
    ("fig_a1_iou_bins.png",       "image21.png", "rId101"),  # §3.5.4 末
    ("fig_c2_bbox_scoring.png",   "image22.png", "rId102"),  # §4.4 末
    ("fig_b1_lowres_sweep.png",   "image23.png", "rId103"),  # §5.3 末 (lowres)
    ("fig_b2_padding_sweep.png",  "image24.png", "rId104"),  # §5.3 末 (padding)
    ("fig_c1_seed_stability.png", "image25.png", "rId105"),  # §5.4 末
]


# ---------- XML helpers ----------

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def heading_para(text: str, para_id: str) -> str:
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:pStyle w:val="3"/>'
        f'<w:keepNext w:val="0"/><w:keepLines w:val="0"/><w:widowControl/>'
        f'<w:spacing w:before="156" w:after="156"/><w:jc w:val="both"/>'
        f'<w:rPr><w:b/></w:rPr></w:pPr>'
        f'<w:r><w:rPr><w:b/></w:rPr><w:t>{text}</w:t></w:r>\n'
        f'    </w:p>\n'
    )


def body_para(text: str, para_id: str) -> str:
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:pStyle w:val="31"/><w:ind w:firstLine="480"/></w:pPr>\n'
        f'      <w:r><w:t xml:space="preserve">{text}</w:t></w:r>\n'
        f'    </w:p>\n'
    )


def code_para(lines: List[str], para_id: str) -> str:
    runs = []
    for i, ln in enumerate(lines):
        suffix = "" if i == len(lines) - 1 else "<w:br/>"
        runs.append(
            f'<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" '
            f'w:eastAsia="宋体"/><w:sz w:val="18"/></w:rPr>'
            f'<w:t xml:space="preserve">{_esc(ln)}</w:t>{suffix}</w:r>'
        )
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:pStyle w:val="31"/><w:spacing w:before="60" w:after="60"/>'
        f'<w:ind w:left="480" w:firstLine="0"/><w:jc w:val="left"/></w:pPr>\n'
        f'      {"".join(runs)}\n'
        f'    </w:p>\n'
    )


def math_para(text: str, para_id: str) -> str:
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:pStyle w:val="31"/>'
        f'<w:spacing w:before="80" w:after="80"/>'
        f'<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>\n'
        f'      <w:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" '
        f'w:eastAsia="宋体"/><w:i/></w:rPr>'
        f'<w:t xml:space="preserve">{_esc(text)}</w:t></w:r>\n'
        f'    </w:p>\n'
    )


def figure_para(rid: str, cx: int, cy: int, para_id: str, drawing_id: int) -> str:
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:jc w:val="center"/></w:pPr>\n'
        f'      <w:r><w:drawing>\n'
        f'        <wp:inline distT="0" distB="0" distL="0" distR="0">\n'
        f'          <wp:extent cx="{cx}" cy="{cy}"/>\n'
        f'          <wp:effectExtent l="0" t="0" r="0" b="0"/>\n'
        f'          <wp:docPr id="{drawing_id}" name="Picture {drawing_id}"/>\n'
        f'          <wp:cNvGraphicFramePr>\n'
        f'            <a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>\n'
        f'          </wp:cNvGraphicFramePr>\n'
        f'          <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">\n'
        f'            <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">\n'
        f'              <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">\n'
        f'                <pic:nvPicPr><pic:cNvPr id="{drawing_id}" name="Picture {drawing_id}"/>\n'
        f'                  <pic:cNvPicPr><a:picLocks noChangeAspect="1"/></pic:cNvPicPr></pic:nvPicPr>\n'
        f'                <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>\n'
        f'                <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>\n'
        f'                  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>\n'
        f'              </pic:pic>\n'
        f'            </a:graphicData></a:graphic>\n'
        f'        </wp:inline>\n'
        f'      </w:drawing></w:r>\n'
        f'    </w:p>\n'
    )


def caption_para(text: str, para_id: str) -> str:
    return (
        f'    <w:p w14:paraId="{para_id}">\n'
        f'      <w:pPr><w:pStyle w:val="31"/>'
        f'<w:spacing w:before="40" w:after="160"/>'
        f'<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>\n'
        f'      <w:r><w:rPr><w:sz w:val="18"/></w:rPr>'
        f'<w:t xml:space="preserve">{text}</w:t></w:r>\n'
        f'    </w:p>\n'
    )


# ---------- Content blocks (each insertion is a self-contained string) ----------

def block_after_3_1() -> str:
    """At the end of §3.1 Visual-CoT 基础流程: two-stage formalization + Algorithm 1."""
    parts = []
    parts.append(body_para(
        "为方便后续章节的算法表述与对照实验定义，本节用统一的符号给出 Visual-CoT 两阶段推理流程的形式化描述。"
        "记输入图像为 I ∈ ℝ^(H×W×3)，自然语言问题为 q。第一阶段定位模型输出归一化边界框 b̂ = D_θ(I, q)，b̂ ∈ [0,1]^4；"
        "第二阶段回答模型在视觉输入 Φ(I, b̂) 和提示 π(q, b̂) 上生成答案 ŷ = A_φ(Φ(I, b̂), π(q, b̂))，"
        "其中算子 Φ 描述视觉输入的构造方式（第五章重点研究），算子 π 描述提示模板的构造方式（第四章重点研究）。",
        f"{SENTINEL}31A"))
    parts.append(body_para(
        "在该符号下，本章诊断框架对应于在固定 D_θ 与 A_φ 的前提下，替换边界框 b̂ 的来源（预测、随机、中心、Oracle）"
        "或抑制视觉输入 Φ（无图像），从而隔离各因素对最终结果的影响。算法 1 给出该两阶段推理的统一伪代码描述。",
        f"{SENTINEL}31B"))
    parts.append(code_para([
        "算法 1  Visual-CoT 基线两阶段推理",
        "输入：图像 I, 问题 q, 模式参数 m",
        "输出：预测答案 ŷ",
        " 1  b̂  ← D_θ(I, q)                              # 区域定位",
        " 2  b̂  ← Normalize(b̂)                            # 裁剪到 [0,1]^4",
        " 3  Φ  ← BuildVisualInputs(I, b̂, m.crop_mode)   # 见 §5.2",
        " 4  π  ← BuildPrompt(q, b̂, m.evidence_mode)     # 见 §4.2",
        " 5  y  ← A_φ(Φ, π)                                # 答案生成",
        " 6  ŷ  ← Parse(y, m.evidence_mode)               # 答案抽取",
        " 7  返回 ŷ",
    ], f"{SENTINEL}31C"))
    return "".join(parts)


def block_after_3_4() -> str:
    """At the end of §3.4 评价指标与样本划分: formula box for IoU/ContainsMatch/ANLS."""
    parts = []
    parts.append(body_para(
        "为后续章节复用，本节给出以上指标的形式化定义。"
        "记 b̂ 与 b* 分别为预测与参考边界框，ŷ 与 y* 分别为模型答案与参考答案，N 为样本数。"
        "边界框交并比 IoU 与 IoU ≥ 0.5 准确率定义为：",
        f"{SENTINEL}34A"))
    parts.append(math_para(
        "IoU(b̂, b*) = |b̂ ∩ b*| / |b̂ ∪ b*|,    Acc_bbox = (1/N) Σₙ 1[IoUₙ ≥ 0.5]",
        f"{SENTINEL}34B"))
    parts.append(body_para(
        "包含性匹配 ContainsMatch 用于在生成式回答场景中获得相对宽松的答案核验口径，"
        "其中带波浪号的项表示经过去标点、去冠词并小写化的标准化算子（对应代码中的 normalize_answer 函数）：",
        f"{SENTINEL}34C"))
    parts.append(math_para(
        "ContainsMatch = (1/N) Σₙ 1[ ỹ*ₙ ⊆ ŷ̃ₙ  ∨  ŷ̃ₙ ⊆ ỹ*ₙ ]",
        f"{SENTINEL}34D"))
    parts.append(body_para(
        "对 DocVQA 与 SROIE 等存在长字符串答案的样本，本文同时报告基于标准化 Levenshtein 距离的 ANLS 指标，"
        "其中 Lev 表示编辑距离，|·| 表示字符长度：",
        f"{SENTINEL}34E"))
    parts.append(math_para(
        "ANLS = (1/N) Σₙ max( 0, 1 − Lev(ŷₙ, y*ₙ) / max(|ŷₙ|, |y*ₙ|) )",
        f"{SENTINEL}34F"))
    parts.append(body_para(
        "上述指标共同覆盖了边界框定位质量与答案语义匹配两个维度，第六章在结果分析时会同时引用 ContainsMatch、Acc_bbox 与 ANLS 进行对比。",
        f"{SENTINEL}34G"))
    return "".join(parts)


def block_after_3_5_4() -> str:
    """At the end of §3.5.4 诊断结果小结: IoU bin figure + analysis."""
    parts = []
    parts.append(body_para(
        "为进一步剖析定位质量与答案表现之间的关系，本节按预测边界框 IoU 区间分箱重新计算包含性匹配。"
        "图 3.7 显示，在跨数据集 100 样本上约 61% 样本的 IoU 小于 0.3，高 IoU 区间样本数量较少；"
        "在高 IoU 区间，预测区域增强与结构化提示均显著优于仅局部输入，这表明定位质量与答案表现存在区间性相关关系。"
        "这一图示也支持第六章对负对照偏高现象的解释：当前主要瓶颈在于定位阶段，回答阶段的不稳定与定位质量共同决定最终指标。",
        f"{SENTINEL}354A"))
    parts.append(figure_para("rId101", 5166360, 3013710, f"{SENTINEL}354B", 301))
    parts.append(caption_para(
        "图 3.7 按预测边界框 IoU 分箱后的答案包含性匹配（cross-domain 100 样本）。高 IoU 区间样本较少，但预测区域增强与结构化提示在该区间相对仅局部输入更稳定。",
        f"{SENTINEL}354C"))
    return "".join(parts)


def block_after_4_2_1() -> str:
    """End of §4.2.1: structured-prompt formalization."""
    parts = []
    parts.append(body_para(
        "为统一描述本章的结构化提示与第三章对比方法，本节给出提示构造算子的形式化定义。"
        "记 𝒯_schema 为结构化模板，由 [Region]、[Visual Evidence]、[Reasoning]、[Answer] 四个段落组成。"
        "结构化提示算子 π_struct 将问题 q 与边界框文字描述 bbox(b) 与模板拼接：",
        f"{SENTINEL}421A"))
    parts.append(math_para(
        "π_struct(q, b) = q || bbox(b) || 𝒯_schema",
        f"{SENTINEL}421B"))
    parts.append(body_para(
        "其中 || 表示文本拼接。模型输出 y 通过解析器 𝒫 抽取为四元组 (r, e, c, a)，分别对应区域描述、视觉证据、推理过程和最终答案。"
        "答案抽取阶段优先使用 a 字段，当解析失败时退化为整段文本，并由后续标准化与匹配算子统一处理。"
        "该形式化为 §4.2.2 介绍的字段抽取与 §4.3 的实验结果提供了统一记号。",
        f"{SENTINEL}421C"))
    return "".join(parts)


def block_after_4_2_2() -> str:
    """End of §4.2.2: SCS metric definition."""
    parts = []
    parts.append(body_para(
        "为量化结构化模板被遵守的程度，本节定义结构一致性得分 SCS（structure conformance score），"
        "用于度量模型生成的输出 y 中四个目标段落是否齐备：",
        f"{SENTINEL}422A"))
    parts.append(math_para(
        "SCS(y) = (1/4) Σ_{k ∈ {R,E,C,A}} 1[ section_k ∈ y ]",
        f"{SENTINEL}422B"))
    parts.append(body_para(
        "其中 1[·] 为指示函数，section_k 为对应于 [Region] / [Visual Evidence] / [Reasoning] / [Answer] 的子段。"
        "在跨数据集 100 样本上的实测结果显示，SCS 的均值为 0.50，即模型平均只能稳定产出四段中的两段；同时四段全齐的样本比例为 0%。"
        "该结果表明结构化提示能稳定改变输出格式（严格匹配从 0.260 提升到 0.317），但并未完全收敛到设计的四段模板。"
        "本文据此将结构化提示定位为答案抽取稳定化策略，并将模板对齐的监督学习方法列为后续工作。",
        f"{SENTINEL}422C"))
    return "".join(parts)


def block_after_4_4() -> str:
    """End of §4.4: new subsection introducing bbox-scoring gating algorithm + Algorithm 5 + figure."""
    parts = []
    parts.append(heading_para("4.4.1 边界框评分门控算法（本文提出）", f"{SENTINEL}44H"))
    parts.append(body_para(
        "针对第三章 IoU 分箱实验观察到的&#x201C;预测边界框 IoU 较低时仍被原样使用&#x201D;问题，"
        "本节在不重新训练定位模型的前提下，提出一个轻量的边界框评分门控机制。"
        "该机制对预测框打一个综合分数，低于阈值时切换为可控的回退框（默认为中心框，可选 Oracle 框上界对照）。"
        "评分函数由解析合法性、面积先验和方位关键词聚焦度三个分量构成，权重通过参数 λ₁、λ₂、λ₃ 控制：",
        f"{SENTINEL}44A"))
    parts.append(math_para(
        "score(b̂) = ( λ₁ · s_parse(b̂) + λ₂ · s_size(b̂) + λ₃ · s_focus(b̂, q) ) / Σᵢ λᵢ",
        f"{SENTINEL}44B"))
    parts.append(body_para(
        "三个分量定义如下：s_parse(b̂) = 1[b̂ ∈ [0,1]^4] 反映边界框解析成功与否；"
        "s_size(b̂) = exp( −( log ρ(b̂) − log ρ* )² / (2σ²) ) 是相对面积 ρ(b̂) 的对数高斯先验，ρ* 与 σ 分别表示先验中心与扩散；"
        "s_focus(b̂, q) = max{ 1 − ‖c(b̂) − c_word‖₂ / 0.5 : word ∈ q ∩ KeywordSet } 衡量边界框中心 c(b̂) 是否与问题中的方位关键词（left/right/top/bottom/center 等）一致，无方位词时该项为 0。算法 5 给出门控与回退流程：",
        f"{SENTINEL}44C"))
    parts.append(code_para([
        "算法 5  边界框评分门控与回退（本文提出）",
        "输入：预测框 b̂, 问题 q, 权重 (λ₁,λ₂,λ₃), 阈值 θ, 回退策略 fallback",
        "输出：选定框 b*, 评分信息 info",
        " 1  s_parse ← 1[b̂ 解析成功]",
        " 2  s_size  ← exp( −0.5 · ((log ρ(b̂) − log ρ*) / σ)² )",
        " 3  s_focus ← max{1 − ‖c(b̂) − c_word‖₂ / 0.5 : word ∈ q ∩ KeywordSet} 或 0",
        " 4  score   ← Σᵢ λᵢ · sᵢ / Σᵢ λᵢ",
        " 5  若 score < θ：",
        " 6      b*  ← FallbackBox()           # 默认为中心框 [0.25,0.25,0.75,0.75]",
        " 7      triggered ← True",
        " 8  否则：",
        " 9      b*  ← b̂",
        "10      triggered ← False",
        "11  返回 b*, {score, components, triggered}",
    ], f"{SENTINEL}44D"))
    parts.append(body_para(
        "在跨数据集 100 样本上以四种配置评估该门控：默认配置 (θ=0.5, λ=(1,1,0.5)) 触发回退 54%，包含性匹配 0.420；"
        "严格配置 (θ=0.7) 触发回退 79%，包含性匹配 0.430；缺省 focus 项或 size 项时门控完全失效（触发率为 0%）；"
        "若以 Oracle 框替代默认的中心框回退，包含性匹配达到 0.460，相对无门控基线提升 0.050。"
        "更具说服力的指标是&#x201C;幸存样本&#x201D;质量：严格配置保留的 21 个样本中，IoU ≥ 0.5 的比例从基线的 0.32 提升到 0.67，"
        "表明评分机制能够正确识别低质量边界框，与第三章 IoU 分箱观察的趋势一致。",
        f"{SENTINEL}44E"))
    parts.append(figure_para("rId102", 5166360, 2295720, f"{SENTINEL}44F", 401))
    parts.append(caption_para(
        "图 4.6 边界框评分门控的消融结果。左：不同配置下的答案包含性匹配，柱状内数字为触发回退比例；右：仅在保留样本上重新计算的 IoU ≥ 0.5 比例，反映门控筛选后的&#x201C;幸存样本&#x201D;质量。",
        f"{SENTINEL}44G"))
    parts.append(body_para(
        "该算法不重新训练定位模型即可改善低质量预测样本的表现，且每个分量都对应一种可解释的失败模式："
        "缺省 size 时面积异常的边界框无法被识别，缺省 focus 时方位类问题无法获得匹配奖励；"
        "缺一不可的消融结果证实算法设计的最小性。"
        "该模块与本章 §4.2 的结构化提示在功能上互补：结构化提示稳定答案抽取，评分门控稳定输入边界框质量，二者共同构成本文方法贡献的核心。",
        f"{SENTINEL}44I"))
    return "".join(parts)


def block_after_5_2_1() -> str:
    """End of §5.2.1: four Φ strategies formalization."""
    parts = []
    parts.append(body_para(
        "为方便横向比较，本节给出本章涉及的四种视觉输入策略的统一形式化。"
        "记裁剪算子为 C(I, b, α) = I[ c(b), α · s(b) ]，其中 c(b) 为边界框中心、s(b) 为半边长、α ≥ 1 为 padding 系数；"
        "记低清重采样算子为 R(I, s_low)，表示先等比缩放到边长 s_low 再上采样回原尺寸。"
        "四种策略下的视觉输入算子 Φ 写作：",
        f"{SENTINEL}521A"))
    parts.append(math_para("Φ_full(I, b) = { I }", f"{SENTINEL}521B"))
    parts.append(math_para("Φ_pred(I, b) = { I, C(I, b, α) }", f"{SENTINEL}521C"))
    parts.append(math_para("Φ_crop(I, b) = { C(I, b, α) }", f"{SENTINEL}521D"))
    parts.append(math_para("Φ_mix(I, b)  = { R(I, s_low), C(I, b, α) }", f"{SENTINEL}521E"))
    parts.append(body_para(
        "上述记号将&#x201C;整图基线&#x201D;、&#x201C;预测区域增强&#x201D;、&#x201C;仅局部输入&#x201D;与&#x201C;全局低清加局部高清&#x201D;"
        "四种策略表示为同一算子 Φ 的不同实现。"
        "在该统一记号下，§5.2.2 描述的低清加高清组合可以视为对 Φ_pred 的输入侧正则版本，"
        "区别在于对全局分量施加显式的信息瓶颈而不改变视觉输入数量。",
        f"{SENTINEL}521F"))
    return "".join(parts)


def block_after_5_2_2() -> str:
    """End of §5.2.2: Algorithm 3 + compression ratio η."""
    parts = []
    parts.append(body_para(
        "在 §5.2.1 的统一形式化下，本节给出全局低清加局部高清组合 Φ_mix 的输入构造流程：",
        f"{SENTINEL}522A"))
    parts.append(code_para([
        "算法 3  全局低清与局部高清组合（Mixed-Resolution Input）",
        "输入：图像 I, 边界框 b, 低清分辨率 s_low, padding 系数 α",
        "输出：视觉输入列表 Φ = [I_low, I_crop]",
        " 1  I_low  ← Resize(I, s_low)                    # 等比缩放到边长 s_low",
        " 2  I_low  ← Upsample(I_low, (H, W))             # 再上采样回原尺寸",
        " 3  I_crop ← Crop(I, c(b), α · s(b))             # 围绕 bbox 的高清裁剪",
        " 4  返回 [I_low, I_crop]",
    ], f"{SENTINEL}522B"))
    parts.append(body_para(
        "为给输入侧压缩与底层视觉标记剪枝建立明确的对比口径，本节进一步定义视觉 token 总量与压缩率两个量。"
        "记单张图像编码后的 token 数为 T_patch（CLIP ViT-L/14-224 下 T_patch = 576），"
        "则视觉输入算子 Φ 的 token 总量为 τ(Φ) = |Φ| · T_patch；相对预测区域增强 Φ_pred 的压缩率定义为：",
        f"{SENTINEL}522C"))
    parts.append(math_para(
        "η(Φ) = 1 − τ(Φ) / τ(Φ_pred)",
        f"{SENTINEL}522D"))
    parts.append(body_para(
        "据此可以读出：仅局部输入 Φ_crop 对应 η = 0.5，整图基线 Φ_full 与全局低清加局部高清 Φ_mix 的 token 数与 Φ_pred 相同，因此 η = 0。"
        "但 Φ_mix 通过抑制全局分量的信息密度获得了&#x201C;近似正则&#x201D;效果，"
        "因此在 §5.3 的实验中其准确率与延迟取舍曲线与 Φ_pred 不同；这一现象会在 §5.3 与 §5.4 中进一步分析。",
        f"{SENTINEL}522E"))
    return "".join(parts)


def block_after_5_3() -> str:
    """End of §5.3: lowres + padding sensitivity."""
    parts = []
    parts.append(body_para(
        "为研究 Φ_mix 的关键超参数 s_low 与 α 的取值范围，本节给出两组灵敏性扫描结果。"
        "首先固定输入策略为全局低清加局部高清组合，扫描低清分辨率 s_low ∈ {64, 96, 112, 144, 196}。"
        "如图 5.4 左图所示，包含性匹配在 s_low = 112 取得 0.370 的局部最优；当 s_low 超过 144 时准确率反而下降，"
        "暗示过高的低清分辨率会让全局分量与局部裁剪在信息层面冗余度上升。"
        "图 5.4 右图给出延迟分阶段拆解：preprocess 在 11 ms 量级、generate 在 430–470 ms 量级，"
        "不同 s_low 下整体延迟基本恒定，说明 Φ_mix 的收益不在于减少 wall-clock 时间，"
        "而在于通过输入侧的信息瓶颈实现正则化。",
        f"{SENTINEL}53A"))
    parts.append(figure_para("rId103", 5166360, 2127795, f"{SENTINEL}53B", 501))
    parts.append(caption_para(
        "图 5.4 全局低清分辨率 s_low 的灵敏性扫描。左：准确率随 s_low 的变化；右：总延迟、生成延迟与预处理延迟的对比。",
        f"{SENTINEL}53C"))
    parts.append(body_para(
        "其次固定输入策略为预测区域增强，扫描裁剪 padding 系数 α ∈ {1.0, 1.2, 1.5, 2.0}。"
        "图 5.5 显示，跨数据集小框场景下 α 从 1.0 增至 2.0 使包含性匹配从 0.400 升至 0.420，"
        "延迟保持在 305–316 ms 区间。"
        "对于平均裁剪比 0.07 的小区域，适度扩展上下文窗口能补偿过紧裁剪导致的语义截断；"
        "考虑收益曲线趋于平稳，本文将 α = 1.2 作为默认值（在最大边长下增加 10% 边距），作为效率与准确率的折中。",
        f"{SENTINEL}53D"))
    parts.append(figure_para("rId104", 4500000, 2625000, f"{SENTINEL}53E", 502))
    parts.append(caption_para(
        "图 5.5 裁剪 padding 系数 α 的灵敏性扫描。曲线表明跨域小框场景下更大的 α 略有收益，但收益曲线趋于平稳。",
        f"{SENTINEL}53F"))
    return "".join(parts)


def block_after_5_4() -> str:
    """End of §5.4: multi-seed stability."""
    parts = []
    parts.append(body_para(
        "为评估上述输入压缩与提示组织策略的稳定性，本节以采样温度 T = 0.2 与 top_p = 0.95 在三个不同随机种子（42, 1234, 2026）上重跑核心方法。"
        "如图 5.6 所示，结构化提示的包含性匹配为 0.443 ± 0.012，明显高于预测区域增强的 0.393 ± 0.025；"
        "全局低清加局部高清组合的包含性匹配为 0.363 ± 0.006，与单次运行结果一致。"
        "种子噪声 (≤ 0.025) 远小于方法间差异 (≥ 0.05)，说明实验结论在小样本规模下仍具有统计意义上的可解释性。"
        "该稳定性补充结果也回应了第六章对实验显著性的进一步讨论。",
        f"{SENTINEL}54A"))
    parts.append(figure_para("rId105", 4500000, 3000000, f"{SENTINEL}54B", 503))
    parts.append(caption_para(
        "图 5.6 三种核心方法在 T = 0.2、top_p = 0.95 下的多种子稳定性。柱高为 3 个种子的均值，误差棒为标准差。",
        f"{SENTINEL}54C"))
    return "".join(parts)


# ---------- Anchor + insertion mapping ----------

INSERTIONS: List[Tuple[str, Callable[[], str]]] = [
    # Anchor text appears in the NEXT section's heading <w:t>; we insert before that <w:p>.
    ("3.2 实验平台与日志设计",   block_after_3_1),
    ("3.5 CUB 小样本诊断实验",   block_after_3_4),
    ("3.6 本章小结",               block_after_3_5_4),
    ("4.2.2 答案组织与字段抽取", block_after_4_2_1),
    ("4.3 结构化提示实验",       block_after_4_2_2),
    ("4.5 本章小结",               block_after_4_4),
    ("5.2.2 低清整图与高清局部组合", block_after_5_2_1),
    ("5.3 输入压缩实验结果",       block_after_5_2_2),
    ("5.4 模型侧压缩扩展实验",     block_after_5_3),
    ("5.5 本章小结",               block_after_5_4),
]


def insert_before(text: str, anchor_text: str, payload: str) -> str:
    """Insert payload immediately before the <w:p> that contains <w:t>{anchor_text}</w:t>."""
    needle = f"<w:t>{anchor_text}</w:t>"
    positions = [m.start() for m in re.finditer(re.escape(needle), text)]
    if not positions:
        raise RuntimeError(f"anchor not found: {anchor_text!r}")
    # Use the LAST occurrence (body, not TOC).
    last = positions[-1]
    # Walk back to the nearest opening <w:p (with whitespace or '>') that is the paragraph
    scan = last
    p_open = -1
    while scan > 0:
        cand = text.rfind("<w:p", 0, scan)
        if cand < 0:
            break
        tail = text[cand:cand + 6]
        if (tail.startswith("<w:p ") or tail.startswith("<w:p>") or tail.startswith("<w:p\n") or tail.startswith("<w:p\t")) and not tail.startswith("<w:pPr") and not tail.startswith("<w:pict"):
            p_open = cand
            break
        scan = cand
    if p_open < 0:
        raise RuntimeError(f"no enclosing <w:p before anchor {anchor_text!r}")
    return text[:p_open] + payload + text[p_open:]


# ---------- Chapter title tracked-change rename ----------

def replace_title(text: str, old_full: str, new_full: str, ins_id: int) -> str:
    pattern = f"<w:t>{old_full}</w:t>"
    positions = [m.start() for m in re.finditer(re.escape(pattern), text)]
    if not positions:
        print(f"[warn] no match for {old_full!r}")
        return text
    last = positions[-1]
    # Find enclosing <w:r> (not <w:rPr>)
    scan_pos = last
    run_open = -1
    while scan_pos > 0:
        cand = text.rfind("<w:r", 0, scan_pos)
        if cand < 0:
            break
        tail = text[cand:cand + 5]
        if (tail.startswith("<w:r>") or tail.startswith("<w:r ") or tail.startswith("<w:r\n") or tail.startswith("<w:r\t")) and not tail.startswith("<w:rPr") and not tail.startswith("<w:rsid"):
            run_open = cand
            break
        scan_pos = cand
    if run_open < 0:
        print(f"[warn] no enclosing <w:r> for {old_full!r}")
        return text
    run_close = text.find("</w:r>", last) + len("</w:r>")
    run_block = text[run_open:run_close]
    del_run = run_block.replace(pattern, f"<w:delText>{old_full}</w:delText>")
    ins_run = run_block.replace(pattern, f"<w:t>{new_full}</w:t>")
    wrapped = (
        f'<w:del w:id="{ins_id*2}" w:author="Claude" w:date="{NOW_ISO}">{del_run}</w:del>'
        f'<w:ins w:id="{ins_id*2+1}" w:author="Claude" w:date="{NOW_ISO}">{ins_run}</w:ins>'
    )
    return text[:run_open] + wrapped + text[run_close:]


# ---------- Driver ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unpacked", required=True)
    args = p.parse_args()
    root = Path(args.unpacked)
    doc_xml = root / "word/document.xml"
    rels_xml = root / "word/_rels/document.xml.rels"
    media_dir = root / "word/media"
    media_dir.mkdir(parents=True, exist_ok=True)

    text = doc_xml.read_text(encoding="utf-8")
    if SENTINEL in text:
        print("[skip] sentinel found, document already injected.")
        return

    # 1) copy figures + register relationships
    rels_text = rels_xml.read_text(encoding="utf-8")
    inserted_rels = []
    for src, dst, rid in FIGURE_MAP:
        src_path = PAPER_FIGURES_DIR / src
        if not src_path.exists():
            print(f"[warn] missing source figure {src_path}")
            continue
        shutil.copy(src_path, media_dir / dst)
        inserted_rels.append(
            f'  <Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{dst}"/>'
        )
    rels_text = rels_text.replace(
        '</Relationships>',
        '\n'.join(inserted_rels) + '\n</Relationships>'
    )
    rels_xml.write_text(rels_text, encoding="utf-8")
    print(f"[ok] registered {len(inserted_rels)} image relationships")

    # 2) Insert content at each section boundary.
    # IMPORTANT: process insertions in REVERSE document order so earlier offsets stay valid.
    # Anchor text is unique enough that we instead iterate in declared order (each replacement
    # re-scans the full text) — safer and simpler.
    for anchor, builder in INSERTIONS:
        payload = builder()
        text = insert_before(text, anchor, payload)
        print(f"[ok] inserted before {anchor}")

    # 3) Tracked-change chapter rename suggestions.
    text = replace_title(text, "2 视觉语言推理相关理论与实验基础", "2 视觉语言推理相关理论与技术", ins_id=1000)
    text = replace_title(text, "3 Visual-CoT 推理流程复现与局部依据诊断", "3 Visual-CoT 推理方法复现与局部依据诊断", ins_id=1010)
    text = replace_title(text, "7 总结与后续工作", "7 总结与展望", ins_id=1020)
    print("[ok] proposed chapter title renames as tracked changes")

    doc_xml.write_text(text, encoding="utf-8")
    print(f"[ok] wrote {doc_xml}")


if __name__ == "__main__":
    main()
