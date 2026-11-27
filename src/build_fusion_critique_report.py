# -*- coding: utf-8 -*-
"""Build the fusion-strategy critique report."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "报告归档"
OUT = ARCHIVE / "06_2026-11-27_融合诊断与Concat" / "Fusion_Strategy_Critique_2026-11-27.pdf"
FIG_SCHEMA = ROOT / "outputs" / "fusion_critique" / "fusion_schematic.png"
FIG_ACC = ROOT / "outputs" / "fusion_critique" / "fusion_accuracy.png"
METRICS = ROOT / "outputs" / "pan18" / "fusion_metrics.json"

TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")
LIGHT = colors.HexColor("#F2F3F5")

pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

title = ParagraphStyle("Title", fontName="SimHei", fontSize=18, leading=24,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=6)
subtitle = ParagraphStyle("Sub", fontName="SimSun", fontSize=11, leading=16,
                          textColor=TEXT, alignment=TA_CENTER, spaceAfter=2)
meta = ParagraphStyle("Meta", fontName="SimSun", fontSize=9, leading=13,
                      textColor=MUTED, alignment=TA_CENTER)
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.5, leading=16,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=10.5, leading=18,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=21, spaceAfter=4)
body_noindent = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
bullet_style = ParagraphStyle("Bullet", parent=body, leftIndent=21,
                              bulletIndent=10, firstLineIndent=0, spaceAfter=3,
                              bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=12, leading=16,
                    textColor=TEXT, spaceBefore=12, spaceAfter=6,
                    keepWithNext=True)
h3 = ParagraphStyle("H3", fontName="SimHei", fontSize=10.5, leading=15,
                    textColor=TEXT, spaceBefore=8, spaceAfter=3,
                    keepWithNext=True)
quote = ParagraphStyle("Quote", fontName="SimSun", fontSize=10, leading=17,
                       textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                       leftIndent=18, rightIndent=18, spaceBefore=4, spaceAfter=6)
table_header = ParagraphStyle("TH", fontName="SimHei", fontSize=8.6, leading=11,
                              textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
table_cell = ParagraphStyle("TC", fontName="SimSun", fontSize=8.5, leading=11,
                            textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
table_cell_c = ParagraphStyle("TCC", parent=table_cell, alignment=TA_CENTER)
table_caption = ParagraphStyle("TCap", fontName="SimSun", fontSize=9, leading=12,
                               textColor=TEXT, alignment=TA_CENTER,
                               spaceBefore=4, spaceAfter=3)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=9, leading=12,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=3, spaceAfter=6)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=8.8, leading=12.5,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK",
                      spaceBefore=2, spaceAfter=4)


def heading(t):
    return [Paragraph(t, h2)]


def bullet(lead, text):
    return Paragraph(f"<b>{lead}</b> {text}", bullet_style, bulletText="•")


def _widths(header, rows, pad=12):
    widths = []
    for j in range(len(header)):
        w = pdfmetrics.stringWidth(str(header[j]), "SimHei", 8.6)
        for row in rows:
            w = max(w, pdfmetrics.stringWidth(str(row[j]), "SimSun", 8.5))
        widths.append(w + pad)
    total = sum(widths)
    return [x * W / total for x in widths] if total > W else widths


def table(header, rows, numeric=(), small=False, widths=None):
    data = [[Paragraph(str(h), table_header) for h in header]]
    for row in rows:
        data.append([Paragraph(str(v), table_cell_c if j in numeric else table_cell)
                     for j, v in enumerate(row)])
    t = Table(data, colWidths=widths or _widths(header, rows),
              hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, TEXT),
        ("LINEBELOW", (0, 0), (-1, 0), .6, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5 if small else 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5 if small else 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def table_block(cap, t):
    return [KeepTogether([Paragraph(cap, table_caption), t, Spacer(1, 8)])]


def figure(path, width, max_h, cap):
    pil = PILImage.open(path)
    ow, oh = pil.size
    h = width * oh / ow
    if h > max_h:
        h = max_h
        width = h * ow / oh
    return [KeepTogether([Image(str(path), width=width, height=h),
                          Spacer(1, 2), Paragraph(cap, caption)])]


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimSun", 9)
    canvas.setFillColor(TEXT)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


def fmt(x):
    return f"{x:.3f}"


def pct(x):
    return f"{x * 100:.1f}"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    global W
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    r = m["results"]
    a = r["all"]
    text_acc = a["text_only"]["accuracy"]
    stack_acc = a["stacked_lr"]["accuracy"]
    gain = stack_acc - text_acc
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=19 * mm, bottomMargin=19 * mm,
        title="当前证据融合策略为何无效：诊断与改进路线",
        author="", creator="VIAS",
        subject="Diagnosis of late-fusion strategy on PAN 2018 gender inference",
    )
    W = doc.width
    story = []

    story.append(Paragraph("当前证据融合策略为何无效：诊断与改进路线", title))
    story.append(Paragraph("基于 PAN 2018 全量性别推断实验的分析", subtitle))
    story.append(Spacer(1, 3))
    story.append(Paragraph("2026 年 11 月 27 日", meta))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        '数据：<link href="https://doi.org/10.5281/zenodo.3746006" color="#1A1A1A">'
        '<u>PAN 2018 Author Profiling · Zenodo 3746006</u></link>　｜　'
        '代码：<link href="https://github.com/Koipoppy/VIAS" color="#1A1A1A">'
        '<u>github.com/Koipoppy/VIAS</u></link>', meta))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=.6, color=TEXT,
                            spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "<b>摘要：</b>本报告回答三个问题：当前融合实际做了什么；为什么没有带来准确率提升；"
        "真正有效的融合应该怎么做。结论是：当前实现属于决策级 late fusion，只对文本、图像两个"
        "独立后验做加权，模型无法利用跨模态交互，权重也不是从验证集或混淆矩阵估计；在 PAN 2018 "
        "全量 5,100 名用户上，文本单模态准确率 0.792、图像 0.573、最佳 late fusion 0.795，"
        "只多正确 16 人（Δ=0.003，McNemar p=0.281），不构成判断增量。要获得增量，必须把融合"
        "前移到特征/表示层，训练联合 P(Y|text,image)，并把证据组织为“证据→中间假设→目标”的"
        "层级结构；来源可靠性应由校准后的混淆矩阵估计，而不是手工六维权重。实施顺序上，先做"
        "embedding concat 作为最小特征级融合，再做 cross-attention；两阶段使用同一数据划分和"
        "同一测试集，只有 cross-attention 相对 concat 有显著增量时才采用更复杂结构。", abstract))
    story.append(HRFlowable(width="100%", thickness=.5, color=TEXT,
                            spaceBefore=6, spaceAfter=8))

    # Summary box
    summary_rows = [
        [Paragraph("<b>一句话诊断</b>", table_cell),
         Paragraph("当前不是“证据融合”，而是“后验加权集成”。它适合做审计、冲突检测和保守更新，"
                   "不适合作为准确率提升手段。", table_cell)],
        [Paragraph("<b>直接证据</b>", table_cell),
         Paragraph(f"文本 {fmt(text_acc)} → 最佳 late fusion {fmt(stack_acc)}，仅多正确 16/5,100 人；"
                   "按分语言堆叠为 0.7965，p≈0.103；统一堆叠口径 p=0.281。两种口径都不显著。", table_cell)],
        [Paragraph("<b>根因</b>", table_cell),
         Paragraph("没有联合 P(Y|text,image)；没有跨模态交互；没有中间假设层；权重是手工规则，"
                   "不是从混淆矩阵估计。", table_cell)],
        [Paragraph("<b>正解</b>", table_cell),
         Paragraph("联合多模态编码器 + 跨模态注意力 + 层级证据推理；先把融合做进模型，再把可靠性"
                   "做进校准。", table_cell)],
    ]
    st = Table(summary_rows, colWidths=[24 * mm, W - 24 * mm], hAlign="CENTER")
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#B8BEC6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(st)
    story.append(Spacer(1, 6))

    story.extend(heading("1　问题与结论"))
    story.append(Paragraph(
        "用户关心的指标是准确率：证据融合之后，结果有没有比单模态更好。对这个问题的回答必须分成"
        "两层。第一层是“当前代码有没有变好”；第二层是“当前方法有没有可能变好”。前者的答案是有，"
        "但只在千分位，且不显著；后者的答案是否定的，因为当前方法在信息上就没有给模型增加任何"
        "新的判断依据。", body))
    story.append(Paragraph(
        "本报告不否定证据单元、可信度、冲突处理和解释链路这套系统工程。它们对审计、溯源、"
        "风险控制有价值，应当保留。本报告否定的是把它当作精度引擎：在当前结构下，继续调六维权重、"
        "继续加弱模态、继续换融合公式，都不会产生稳定的判断增量。", body))
    story.append(Paragraph(
        "报告的结构如下：第 2 节拆解当前融合实际做了什么；第 3 节从信息、统计、结构和评估四个"
        "层面解释为什么它无效；第 4 节给出 PAN 2018 全量实验与单例证据；第 5 节说明它与项目书"
        "“融合”的差距；第 6 节给出真正应该做的联合多模态模型与层级证据路线；第 7 节给出实施"
        "顺序和验收标准。", body))

    story.extend(heading("2　当前融合到底做了什么"))
    story.append(Paragraph("2.1　一条公式", h3))
    story.append(Paragraph(
        "当前实现把每个模态分别送进一个分类器，得到 P<sub>text</sub>(H) 与 P<sub>image</sub>(H)，"
        "再用可信度权重在对数空间相加：", body))
    story.append(Paragraph(
        "log P(H|E) ∝ log P(H) + Σ<sub>i</sub> ω<sub>i</sub> log P<sub>i</sub>(H)，"
        "　ω<sub>i</sub> = (credibility<sub>i</sub> × multiplier<sub>i</sub>)<sup>1.5</sup>", quote))
    story.append(Paragraph(
        "其中 H 取 female/male，先验 P(H)=0.5，后验置信度取 max P(H|E)。这在数学上是合法的"
        "对数意见池（log-opinion pool），也是集成学习里的 product of experts。但它有一个关键"
        "性质：输入只有两个独立模型的后验，融合器看不到文本和图像本身。它做的只是给两个已有"
        "观点调音量，不会产生第三个观点。", body))
    story.append(Paragraph("2.2　证据单元里装的其实不是“证据”", h3))
    story.append(Paragraph(
        "当前每个案例通常只有一个文本证据单元和一个图像证据单元。文本单元的内容是“文本模型"
        "判断为女性的概率”，图像单元的内容是“图像模型判断为女性的概率”。这不是项目书意义上的"
        "语义证据。真正的证据应该是“账号自述‘我是女生’”“昵称使用女性化用字”“头像中检测到"
        "人脸，呈现女性化外观”“多条推文使用女性代词”等可核验的事实。当前实现把模态分类器的"
        "输出直接当作证据，证据粒度太粗，后面无论怎么加权都无法恢复被分类器压掉的信息。", body))
    story.append(Paragraph("2.3　权重是规则，不是从数据学出来的可靠性", h3))
    story.append(Paragraph(
        "六维可信度（来源可靠性、权威性、客观性、时效性、完整性、一致性）是人工设定的评分规则，"
        "不是该来源在验证集上的判别能力。一个来源可以“很权威”但在这项任务上经常错；一张图可以"
        "“很完整”但根本没有正脸。权重幂次 1.5 也是经验值。更关键的是，这套权重没有用混淆矩阵、"
        "敏感度/特异度、ECE 或 Brier 分数估计过，因此它不能回答“这条证据在多大程度上改变后验”"
        "这个概率问题。", body))
    story.append(Paragraph("2.4　后验置信度不是结果可信度", h3))
    story.append(Paragraph(
        "当前把 max P(H|E) 直接当作结果可信度。这个数只表示模型内部的后验峰值，未经过校准；"
        "模型完全可以在准确率 0.79 的情况下给出大量 0.95 以上的自信预测。结果可信度应该来自"
        "校准后的概率与可验证的混淆矩阵指标，而不是 softmax 的峰值。这一点与融合问题是两个"
        "问题，但会共同造成“看起来很有把握、实际没有提升”的错觉。", body))

    story.extend(table_block("表 1　当前实现与真正融合的对照", table(
        ("环节", "当前实现", "真正融合的要求", "差距"),
        [
            ("融合位置", "决策层：两个后验加权", "特征/表示层或语义证据层", "无跨模态交互"),
            ("融合输入", "一个文本概率、一个图像概率", "多条可核验的语义证据", "证据粒度过粗"),
            ("权重来源", "手工六维规则，幂次 1.5", "验证集混淆矩阵/校准估计", "非经验、不可检验"),
            ("中间层", "无，直接融合性别概率", "证据→中间假设→目标", "缺推理层"),
            ("置信度", "max 后验", "校准概率/区间/可信度", "未校准"),
            ("评估", "点准确率", "配对检验、置信区间、ECE/Brier", "增益不显著"),
            ("保留价值", "溯源、冲突、解释、API", "保留为系统审计层", "有价值，但不是精度引擎"),
        ],
    )))

    story.extend(heading("3　为什么不会产生判断增量"))
    story.append(Paragraph("3.1　信息层面：联合信息在融合前就丢了", h3))
    story.append(Paragraph(
        "设文本为 X<sub>t</sub>，图像为 X<sub>i</sub>，目标为 Y。当前融合器能看到的只有"
        "P(Y|X<sub>t</sub>) 和 P(Y|X<sub>i</sub>)，看不到 X<sub>t</sub>、X<sub>i</sub> 本身。"
        "只有当两个模态在给定 Y 下条件独立时，这对边缘后验才是联合后验的充分统计量；现实中的"
        "多模态数据通常不满足这个假设。文本里的反讽、图片中的人脸角度、图文之间的指代关系，"
        "都属于交互信息，在各自模态被压成一个性别概率时已经丢失。融合器再聪明，也无法从两个"
        "标量里恢复这些信息。", body))
    story.append(Paragraph(
        "换句话说，late fusion 只能改变“文本说 female 0.9、图像说 male 0.6”这两个观点的相对"
        "音量，不能发现“文本其实在引用别人的话，而图像是合影中的另一个人”这类新判断。判断增量"
        "来自新的条件信息或新的证据组合，不来自权重重排。这个限制也不是靠“再加一个模态”能解决"
        "的：每多一个模态，只是多一个边缘后验；如果它们都经过各自的分类头，交互信息仍然在进入"
        "融合器之前丢失。要保留交互，必须让融合发生在特征层，或者让证据在中间假设层发生组合。", body))
    story.extend(figure(FIG_SCHEMA, W, 210,
                        "图 1　late fusion 与 joint fusion（cross-attn/Q-Former）的信息流差异"))
    story.append(Paragraph("3.2　统计层面：弱模态 + 启发式权重 = 必然退化", h3))
    story.append(Paragraph(
        "PAN 2018 的图像单模态只有 0.573，文本 0.792。等权融合把弱图像直接注入，降到 0.710；"
        "可信度融合同样降到 0.710，说明六维权重没有识别出“图像在这项任务上没有增量”。校准融合"
        "把文本权重推到 0.95，才回到 0.792；这不是融合的胜利，而是模型自己学会了“不要相信图像”。"
        "堆叠融合加入了交互特征，也只到 0.795，多正确 16 人。换句话说，当前数据下图像提供的"
        "条件信息太少，任何 late fusion 都只能逼近文本，而不能超过文本。", body))
    story.append(Paragraph("3.3　结构层面：没有中间假设，证据链是断的", h3))
    story.append(Paragraph(
        "项目书要求的是“证据→中间假设→目标”的层级推断：先从多条证据形成“性别表达”“年龄段”"
        "“地域”“职业”等中间判断，再在这些判断上推断目标身份。当前实现跳过了中间层，直接把"
        "“文本模型对性别的概率”和“图像模型对性别的概率”相加。这样做的后果是：无法表达证据之间"
        "的依赖，无法处理“文本可靠但图像是合影”的情况，也无法解释冲突到底发生在哪一层。它只有"
        "一个冲突标志，没有冲突的语义。", body))
    story.append(Paragraph("3.4　评估层面：点准确率不是融合增益", h3))
    story.append(Paragraph(
        "融合增益应该定义为“融合结果相对最强单模态的提升”，并且要做配对显著性检验。当前报告"
        "里 +0.003 的差异只对应 16 个用户，McNemar p=0.281；按分语言堆叠口径为 0.7965，"
        "p≈0.103。两种口径都不显著。如果不做检验，只看最高一行，很容易把文本模型本身的水平"
        "误读成融合的贡献。更严谨的做法是报告 Δ 的 bootstrap 置信区间、ECE/Brier，并按语言"
        "分别检验。", body))
    story.append(Paragraph(
        "把四个层面合起来看，当前融合的问题不是“权重还没调好”，而是结构上缺了三样东西："
        "能看到原始特征的联合表示、能承载语义的中间假设、以及能从验证集估计的来源可靠性。"
        "表 2 把这些差距逐项列出。", body))

    story.extend(table_block("表 2　融合有效性诊断清单", table(
        ("检查项", "当前结果", "判断"),
        [
            ("是否存在联合 P(Y|text,image)", "否，只有两个边缘后验", "不满足"),
            ("是否利用跨模态交互", "否，融合器看不到原始特征", "不满足"),
            ("权重是否由验证集/混淆矩阵估计", "否，手工六维 + 固定幂次", "不满足"),
            ("是否有语义证据与中间假设层", "否，只有模态概率", "不满足"),
            ("是否做配对检验与校准误差", "事后 p=0.281；未报告 ECE/Brier", "不满足"),
        ],
    )))
    story.append(Paragraph(
        "注：证据溯源与冲突解释已具备（Schema/API/解释链完整），应保留为系统审计层，"
        "但它不能弥补上面的五项缺口。", note))

    story.extend(heading("4　实验证据：全量 PAN 2018"))
    story.append(Paragraph("4.1　设置", h3))
    story.append(Paragraph(
        "数据为 PAN 2018 Author Profiling 官方训练/测试集：训练 7,500 人、测试 5,100 人，"
        "英语 1,900、西语 2,200、阿语 1,000；每人 100 条推文、10 张图片，标签为账号级性别。"
        "文本用分语言字符 2–5 gram TF-IDF + 逻辑回归；图像用公开 ViT 性别分类器，每用户取首张"
        "可用图；融合在统一证据单元上完成。测试集不参与训练或调参。", body))
    story.extend(table_block("表 3　全量测试集总体结果（n=5,100）", table(
        ("方法", "Accuracy", "Macro-F1", "AUC", "Δ vs 文本", "判断"),
        [
            ("文本单模态", fmt(a["text_only"]["accuracy"]), fmt(a["text_only"]["macro_f1"]),
             fmt(a["text_only"]["auc"]), "—", "最强单模态"),
            ("图像单模态", fmt(a["image_only"]["accuracy"]), fmt(a["image_only"]["macro_f1"]),
             fmt(a["image_only"]["auc"]), f"{a['image_only']['accuracy']-text_acc:+.3f}", "弱信号"),
            ("等权融合", fmt(a["equal_prob"]["accuracy"]), fmt(a["equal_prob"]["macro_f1"]),
             fmt(a["equal_prob"]["auc"]), f"{a['equal_prob']['accuracy']-text_acc:+.3f}", "显著伤害"),
            ("可信度融合", fmt(a["credibility"]["accuracy"]), fmt(a["credibility"]["macro_f1"]),
             fmt(a["credibility"]["auc"]), f"{a['credibility']['accuracy']-text_acc:+.3f}", "显著伤害"),
            ("校准融合", fmt(a["calibrated_late"]["accuracy"]), fmt(a["calibrated_late"]["macro_f1"]),
             fmt(a["calibrated_late"]["auc"]), f"{a['calibrated_late']['accuracy']-text_acc:+.3f}", "退化为文本"),
            ("堆叠融合", fmt(a["stacked_lr"]["accuracy"]), fmt(a["stacked_lr"]["macro_f1"]),
             fmt(a["stacked_lr"]["auc"]), f"{gain:+.3f}", "不显著"),
            ("冲突处理", fmt(a["conflict_handled"]["accuracy"]), fmt(a["conflict_handled"]["macro_f1"]),
             fmt(a["conflict_handled"]["auc"]), f"{a['conflict_handled']['accuracy']-text_acc:+.3f}", "显著伤害"),
        ],
        numeric=(1, 2, 3, 4),
    )))
    story.append(Paragraph(
        "注：Δ 为相对文本单模态的绝对准确率差；堆叠融合多正确 16 人，McNemar p=0.281。"
        "若按分语言堆叠模型合并预测，准确率为 0.7965，p≈0.103；两种口径均不支持显著增益。"
        "校准融合的文本权重在三种语言上均为 0.95。", note))
    story.extend(figure(FIG_ACC, W, 135, "图 2　PAN 2018 全量测试集上的方法对比"))

    story.append(Paragraph("4.2　分语言结果", h3))
    story.extend(table_block("表 4　分语言 Accuracy", table(
        ("方法", "英语 (1,900)", "西语 (2,200)", "阿语 (1,000)", "总体 (5,100)"),
        [
            ("文本", fmt(r["en"]["text_only"]["accuracy"]), fmt(r["es"]["text_only"]["accuracy"]),
             fmt(r["ar"]["text_only"]["accuracy"]), fmt(text_acc)),
            ("图像", fmt(r["en"]["image_only"]["accuracy"]), fmt(r["es"]["image_only"]["accuracy"]),
             fmt(r["ar"]["image_only"]["accuracy"]), fmt(a["image_only"]["accuracy"])),
            ("校准融合", fmt(r["en"]["calibrated_late"]["accuracy"]), fmt(r["es"]["calibrated_late"]["accuracy"]),
             fmt(r["ar"]["calibrated_late"]["accuracy"]), fmt(a["calibrated_late"]["accuracy"])),
            ("堆叠融合", fmt(r["en"]["stacked_lr"]["accuracy"]), fmt(r["es"]["stacked_lr"]["accuracy"]),
             fmt(r["ar"]["stacked_lr"]["accuracy"]), fmt(stack_acc)),
        ],
        numeric=(1, 2, 3, 4),
    )))
    story.append(Paragraph(
        "三种语言呈现同一模式：图像单独使用都在 0.57 左右，等权/可信度融合显著低于文本；校准融合"
        "基本回到文本；堆叠融合在英语上 +0.008、西语 +0.004、阿语 +0.002，方向上有一点点，"
        "但分别对应 p=0.086、0.445、0.902，不能称为稳定提升。", body))

    story.append(Paragraph("4.3　单例解剖：融合后的概率反而更低", h3))
    story.append(Paragraph(
        "取测试集用户 134b59cc1b46626c0cc6465f9317c999（真值 female，100 条推文、5,803 字符）"
        "做一次分解。文本模型给出 female=0.897，图像模型给出 female=0.463；两个证据单元的可信度"
        "都是 0.7525，权重 0.6528。", body))
    story.extend(table_block("表 5　单例融合分解", table(
        ("计算方式", "P(female)", "权重", "结论"),
        [
            ("文本单模态", "0.897", "—", "正确，置信度高"),
            ("图像单模态", "0.463", "—", "错误，接近随机"),
            ("等权 log pool", "0.882", "1 / 1", "正确，但低于文本"),
            ("可信度 log pool", "0.788", "0.653 / 0.653", "正确，但明显低于文本"),
            ("算术平均", "0.680", "0.5 / 0.5", "正确，但已接近边界"),
        ],
        numeric=(1,),
    )))
    story.append(Paragraph(
        "注：log pool 的“等权”指两个证据权重均为 1；“算术平均”对应当前 equal_prob 方法"
        "（0.5/0.5）。两种口径都低于文本单模态。", note))
    story.append(Paragraph(
        "这个例子说明，融合并没有产生“文本 + 图像 = 更准”的判断，它只是把文本的高置信度往下拉。"
        "当图像与文本一致时，融合可能略微增加把握；当图像弱且不一致时，融合只会稀释文本。"
        "这不是证据互补，而是弱证据对强证据的拖累。", body))

    story.extend(heading("5　这不是项目书要求的“融合”"))
    story.append(Paragraph(
        "项目书里的融合至少包含三层含义：多模态特征对齐、跨模态交互、以及证据到中间假设再到"
        "目标的层级推断。当前实现只做了一件事：把两个属性分类器的输出加权。它没有特征对齐，"
        "没有视觉-语言交互，没有中间假设，也没有在 PAN 2018 上涉及候选身份。严格说，它是"
        " model ensemble，不是 evidence fusion。", body))
    story.extend(table_block("表 6　项目书要求与当前实现的差距", table(
        ("项目书要求", "当前实现", "差距"),
        [
            ("多模态特征对齐", "无，文本/图像各自独立", "缺少共享表示空间"),
            ("跨模态交互（CLIP/Q-Former/注意力）", "无", "核心缺口"),
            ("证据→中间假设→目标", "无，直接融合性别概率", "缺少推理层"),
            ("候选身份层", "服务接口与合成案例支持；PAN18 无", "真实数据未验证"),
            ("证据溯源与解释", "有 Schema、LOO、Top-k、冲突链", "可保留"),
            ("来源可靠性", "手工六维", "需混淆矩阵/校准"),
        ],
    )))
    story.append(Paragraph(
        "这也解释了为什么“再加一个模态”不会自动变好。音频、行为、结构化证据如果仍然各自输出"
        "一个属性概率，再做加权，得到的只是更多可被稀释的边缘后验，而不是更强的联合判断。", body))

    story.extend(heading("6　真正应该怎么做"))
    story.append(Paragraph("6.1　先把评估协议定死", h3))
    story.append(Paragraph(
        "在改模型之前，先定义什么叫“融合有效”。建议主指标为 Accuracy，辅以 Macro-F1、AUC、"
        "ECE、Brier；融合增益定义为 Δ = Acc(fusion) - max(Acc(text), Acc(image))，而不是相对"
        "等权融合的提升。统计上使用配对 McNemar 检验和 bootstrap 95% 置信区间；按语言分别报告；"
        "测试集只用于最终评估。没有显著性的正数，一律称为“未观察到增益”。", body))
    story.append(Paragraph("6.2　核心：先 concat，再 cross-attention", h3))
    story.append(Paragraph(
        "<b>第一步：embedding concat。</b>文本编码器（XLM-R/DeBERTa）输出用户级表示 "
        "h<sub>t</sub>，图像编码器（ViT/CLIP）输出多图聚合表示 h<sub>i</sub>；两边各接一个"
        "线性投影到同一维度，做 LayerNorm/L2 归一化后直接拼接 [h<sub>t</sub>; h<sub>i</sub>]，"
        "再接 MLP 分类头，端到端训练。这是最小可用的特征级融合，也是当前约定的第一步：先把"
        "embedding 拼起来，不先上 cross-attention。", body))
    story.append(Paragraph(
        "<b>第二步：cross-attention。</b>保留同一文本/图像编码器，把文本 token 序列 "
        "H<sub>t</sub> 与图像 patch token 序列 H<sub>i</sub> 送入 cross-attention，或用 "
        "Q-Former 的可学习 query 做视觉-语言交互，再接联合分类头。它检验的是：token 级对齐"
        "能否比简单拼接带来额外增量，而不是默认“attention 一定更好”。", body))
    story.append(Paragraph(
        "两阶段必须做受控对比：同一训练/验证/测试划分、同一文本/图像编码器、同一最大 epoch "
        "与早停策略、同一分类头容量，只替换融合模块。结果表至少并列 text-only、image-only、"
        "late fusion、concat、cross-attention 五行；每行报告 Accuracy、Macro-F1、AUC、ECE、"
        "Brier，以及相对 text 和相对 concat 的 Δ 与 McNemar p 值。若 concat 相对最强单模态"
        "没有显著提升，应先查编码器与数据，而不是直接上 cross-attention；若 cross-attention "
        "相对 concat 不显著，就保留 concat，避免为复杂度买单。", body))

    story.extend(table_block("表 7　concat 与 cross-attention 的受控对比", table(
        ("维度", "第一步：embedding concat", "第二步：cross-attention"),
        [
            ("融合位置", "池化后的特征向量层", "token 表示层"),
            ("输入", "h_text、h_image（用户级向量）", "H_text 序列、H_image patch 序列"),
            ("融合模块", "投影 + 拼接 + MLP", "cross-attention / Q-Former + 分类头"),
            ("回答的问题", "简单特征拼接是否足够", "token 级对齐是否带来额外增量"),
            ("控制变量", "同一编码器、划分、训练预算", "同一编码器、划分、训练预算"),
            ("判定规则", "相对最强单模态显著才继续", "相对 concat 显著才采用，否则保留 concat"),
        ],
    )))
    story.append(Paragraph("6.3　把证据组织成“证据→中间假设→目标”", h3))
    story.append(Paragraph(
        "联合模型解决“看不看得到交互”，层级证据解决“能不能推理”。具体做法是：从文本、图像、"
        "结构化字段中抽取多条语义证据（自述、代词、昵称、头像外观、社会角色、年龄线索等），"
        "每条证据保留来源、时间、原文引用和可信度；在中间层形成“性别表达”“年龄段”“地域”"
        "“职业”等假设；最后在假设上推断目标身份或候选身份。冲突检测发生在这个层级，而不是"
        "只在两个最终概率之间。", body))
    story.append(Paragraph("6.4　来源可靠性从混淆矩阵来", h3))
    story.append(Paragraph(
        "每个来源、每个模态、每种语言都应在校准集上统计混淆矩阵，得到敏感度、特异度、似然比，"
        "再做温度缩放或 isotonic 校准。权重应该由这些量估计，而不是由“权威性 0.15、时效性 0.15”"
        "这样的手工维度直接相乘。六维评分可以保留为先验或正则项，但不应该成为主证据权重。"
        "校准后的概率还要报告 ECE/Brier，否则“可信度”仍然只是自我评价。", body))
    story.append(Paragraph("6.5　选择性融合作为过渡基线", h3))
    story.append(Paragraph(
        "在联合模型完成之前，可以先做质量门控：只有检测到人脸、图像质量足够、且图像与文本的"
        "条件置信度都较高时，才让图像参与融合；否则回退文本。这个基线能减少弱图像拖累，但它"
        "仍然属于 late fusion，不应被当作最终方案。它的价值是给联合模型提供一个可比较的强基线。", body))
    story.append(Paragraph("6.6　不要用“更大的单模态”替代联合训练", h3))
    story.append(Paragraph(
        "一个常见误区是：把文本模型换成大模型、把图像模型换成 CLIP，就会自动获得融合收益。"
        "更强的单模态是必要条件，不是充分条件；如果两个模型仍然各自输出后验、再做加权，信息"
        "瓶颈没有改变，late fusion 仍然只能逼近最强的那一路。真正改变上限的是联合训练与跨模态"
        "交互，而不是把两个独立模型各自做大。", body))
    story.append(Paragraph(
        "反过来，也不能因为联合模型难做，就长期停留在 late fusion 上。late fusion 可以作为"
        "系统冷启动、审计和回退路径，但它不是项目书所要求的融合。把这条边界说清楚，才能避免"
        "用“融合”的名义继续优化一个不会产生增量的结构。", body))
    story.append(Paragraph(
        "把两阶段放回实施顺序里，P1a 和 P1b 不是并列选项：P1a 先回答“特征级拼接是否值得做”，"
        "P1b 再回答“更细粒度的交互是否还有额外收益”。因此报告也要按这个顺序写：先报 concat "
        "相对最强单模态的结果，再报 cross-attention 相对 concat 的结果，而不是只报一个“融合后"
        "最好”的数字。这样即使 cross-attention 没有赢，也能清楚知道收益来自特征融合本身，还是"
        "来自更复杂的注意力结构。", body))

    story.extend(table_block("表 8　融合路线的定位", table(
        ("路线", "融合位置", "利用交互", "主要作用", "代价", "建议"),
        [
            ("late fusion（当前）", "决策层", "否", "审计、冲突、保守更新", "低", "保留为基线"),
            ("质量门控 late fusion", "决策层 + 质量", "有限", "减少弱模态拖累", "中", "过渡基线"),
            ("embedding concat", "特征向量层", "是（简单）", "最小特征级基线", "中", "第一步"),
            ("cross-attention", "token 表示层", "是（细粒度）", "检验额外交互增量", "高", "第二步"),
            ("层级证据推理", "语义/假设层", "是 + 推理", "身份推断、解释、冲突", "高", "项目目标"),
        ],
    )))

    story.extend(heading("7　实施顺序与验收标准"))
    story.append(Paragraph(
        "建议按 P0→P1a→P1b→P2 推进，不要同时改模型、改权重、改数据。每一阶段只回答一个问题，"
        "并且用同一套测试集和同一套统计检验。P1a 先做 concat，P1b 再和 cross-attention 做受控"
        "对比；只有后者显著更好时才替换。", body))
    story.extend(table_block("表 9　实施路线", table(
        ("阶段", "时间", "工作", "交付", "验收"),
        [
            ("P0", "1–2 周", "基线加固：文本换 XLM-R/DeBERTa；人脸检测 + 10 图聚合；概率校准；质量门控",
             "强单模态基线 + 门控 late fusion", "总体 Acc ≥ 0.80；ECE ≤ 0.05；相对文本不显著下降"),
            ("P1a", "1–2 周", "embedding concat：文本/图像编码器 → 投影 → 拼接 → MLP；端到端训练",
             "concat 全量结果与消融", "相对最强单模态 ≥ +1.0 点且 McNemar p<0.05；否则先查编码器/数据"),
            ("P1b", "2–3 周", "cross-attention：同一编码器，替换融合模块为 cross-attention/Q-Former",
             "cross-attention 与 concat 的受控对比", "相对 concat ≥ +0.5 点且 p<0.05；否则保留 concat"),
            ("P2", "4–6 周", "层级证据：语义证据抽取；中间假设；候选身份；冲突/解释",
             "证据→假设→目标原型", "Accuracy@1/Recall@3；冲突可解释；真实案例验证"),
        ],
        widths=[14 * mm, 18 * mm, 50 * mm, 38 * mm, W - 120 * mm],
    )))
    story.append(Paragraph(
        "P1a/P1b 的验收要特别防止“用测试集调出来的 +1 点”。联合模型必须在训练集内做划分，"
        "校准集独立，测试集只在最后打开。若 P1a 没有显著增益，先回到数据与编码器质量；若 P1b "
        "没有显著超过 P1a，就保留 concat，不把 cross-attention 当作默认答案。", body))
    story.append(Paragraph("7.1　应该停止做什么", h3))
    story.append(bullet("停止调六维权重：", "在没有验证集证据的情况下，调参只会过拟合校准集；当前权重已经把图像拖到 0.710。"))
    story.append(bullet("停止把弱模态硬融合：", "图像 0.573 时，等权/可信度融合必然伤害文本；先提升单模态或做质量门控。"))
    story.append(bullet("停止把 max 后验当可信度：", "未校准的峰值不能作为结果可信度，也不能作为对外承诺。"))
    story.append(bullet("停止只看点准确率：", "没有配对检验和置信区间的 +0.003，不能称为提升。"))
    story.append(Paragraph("7.2　应该保留什么", h3))
    story.append(bullet("证据单元与 Schema：", "23 字段、来源、时间、PII 标记、冲突字段是系统审计的基础。"))
    story.append(bullet("冲突与解释链路：", "LOO、Top-k、证据链在联合模型上仍然需要，只是冲突要发生在语义层。"))
    story.append(bullet("五接口与服务化：", "上传、评估、贝叶斯、冲突、解释接口可以继续用，后端替换为联合模型/层级推理即可。"))

    story.extend(heading("8　结论"))
    story.append(Paragraph(
        "对“当前融合有没有提高准确率”的直接回答是：没有显著提高。PAN 2018 全量 5,100 人上，"
        f"文本单模态 {fmt(text_acc)}，最佳 late fusion {fmt(stack_acc)}，只多正确 16 人，"
        "McNemar p=0.281；分语言堆叠口径 0.7965，p≈0.103。两个口径都不支持“融合带来稳定增益”。"
        "等权/可信度融合甚至把准确率拉到 0.710，说明当前权重没有识别出图像是弱信号。", body))
    story.append(Paragraph(
        "对“为什么无效”的回答是：当前实现是决策级 late fusion，只对两个独立后验做加权。它既"
        "看不到原始特征，也没有跨模态交互；既没有语义证据，也没有中间假设层；权重是手工规则，"
        "不是从混淆矩阵估计。它最多只能把文本的置信度调高或调低，不能产生新的判断。图像单模态"
        "只有 0.573 时，任何加权都只能逼近文本，不能超过文本。", body))
    story.append(Paragraph(
        "对“真正应该怎么做”的回答是：把融合前移到特征/表示层，训练联合 P(Y|text,image)；"
        "实施顺序是先 concat、再 cross-attention，用同一测试集做受控对比，只有后者显著超过"
        "concat 才采用；同时把证据组织成“证据→中间假设→目标”的层级结构，用校准后的混淆矩阵"
        "估计来源可靠性，用配对检验和校准误差评估。短期先做强单模态、人脸质量门控和概率校准；"
        "中期先 concat 再 cross-attention；长期做层级证据与候选身份。", body))
    story.append(Paragraph(
        "最后需要明确边界：当前这套证据单元、可信度、冲突和解释系统并不是无用的。它作为审计层、"
        "风控层和可解释层是有价值的，应该保留。但它不能同时被当作精度引擎。在联合模型完成之前，"
        "最诚实的表述是：文本单模态 0.792，融合未显著超过文本；当前融合策略不产生判断增量。", body))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=.5, color=TEXT,
                            spaceBefore=2, spaceAfter=5))
    story.append(Paragraph(
        "附：数据与复现。指标文件 outputs/pan18/fusion_metrics.json；逐用户预测 "
        "outputs/pan18/fusion_predictions.csv；对比报告 PAN2018_Multimodal_Gender_Report_2026-11-26.pdf；"
        "代码仓库 github.com/Koipoppy/VIAS。PAN 2018 仅提供性别标签，不含年龄、音频、行为或真实身份；"
        "本报告的机制结论可迁移，具体增益需在项目数据上重新验证。", note))

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
