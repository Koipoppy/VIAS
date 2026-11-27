# -*- coding: utf-8 -*-
"""Build the PAN18 three-way fusion comparison report (PDF)."""
from __future__ import annotations

import json
from pathlib import Path

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
OUT = ARCHIVE / "07_2026-11-27_三种融合对比" / "PAN2018_Three_Fusion_Comparison_2026-11-27.pdf"
BASE = ROOT / "outputs" / "pan18_joint" / "metrics.json"
CROSS = ROOT / "outputs" / "pan18_joint" / "crossattn_2026-11-27" / "metrics_crossattn.json"
FIGS = OUT.parent

TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")
LIGHT = colors.HexColor("#F2F3F5")

pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

title = ParagraphStyle("Title", fontName="SimHei", fontSize=17, leading=23,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=5)
subtitle = ParagraphStyle("Sub", fontName="SimSun", fontSize=10.5, leading=15,
                          textColor=TEXT, alignment=TA_CENTER, spaceAfter=2)
meta = ParagraphStyle("Meta", fontName="SimSun", fontSize=9, leading=13,
                      textColor=MUTED, alignment=TA_CENTER)
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.3, leading=15.5,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=10, leading=17,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=20, spaceAfter=4)
body_ni = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=9,
                        firstLineIndent=0, spaceAfter=3, bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=11.5, leading=15,
                    textColor=TEXT, spaceBefore=10, spaceAfter=5, keepWithNext=True)
h3 = ParagraphStyle("H3", fontName="SimHei", fontSize=10.2, leading=14,
                    textColor=TEXT, spaceBefore=7, spaceAfter=3, keepWithNext=True)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=8.8, leading=11.5,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=2, spaceAfter=5)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=8.5, leading=12,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK",
                      spaceBefore=2, spaceAfter=4)
th = ParagraphStyle("TH", fontName="SimHei", fontSize=8.2, leading=10.5,
                    textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
tc = ParagraphStyle("TC", fontName="SimSun", fontSize=8.1, leading=10.5,
                    textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
tcc = ParagraphStyle("TCC", parent=tc, alignment=TA_CENTER)


def fmt(v, d=4):
    return f"{v:.{d}f}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pstr(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def table(header, rows, widths, numeric=(), cap=None):
    data = [[Paragraph(str(h), th) for h in header]]
    for row in rows:
        data.append([Paragraph(str(v), tcc if j in numeric else tc) for j, v in enumerate(row)])
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, TEXT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    if cap:
        return KeepTogether([Paragraph(cap, caption), t, Spacer(1, 6)])
    return t


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE.read_text(encoding="utf-8"))["results"]
    cross = json.loads(CROSS.read_text(encoding="utf-8"))
    r = {
        "text": base["text"]["test"], "image": base["image"]["test"],
        "late": base["late_stack"]["test"], "concat": base["concat"]["test"],
        "cross": cross["results"]["crossattn_ens"]["test"],
        "ensemble": base["ensemble"]["test"],
    }
    lang = {
        "text": base["text"]["per_lang"], "late": base["late_stack"]["per_lang"],
        "concat": base["concat"]["per_lang"],
        "cross": cross["results"]["crossattn_ens"]["per_lang"],
        "ensemble": base["ensemble"]["per_lang"],
    }
    comp = json.loads(BASE.read_text(encoding="utf-8"))["comparisons"]
    seed_summary = cross["seed_summary"]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=17 * mm, bottomMargin=17 * mm,
                            title="PAN 2018 三种多模态融合方式对比", author="VIAS")
    story = []
    story.append(Paragraph("PAN 2018 三种多模态融合方式对比", title))
    story.append(Paragraph("Late Fusion ｜ Embedding Concat ｜ Cross-Attention", subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("2026 年 11 月 27 日　·　数据：PAN 2018 Author Profiling（EN/ES/AR）", meta))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT, spaceAfter=7))

    story.append(Paragraph(
        f"<b>摘要</b>　在同一份 PAN 2018 数据、同一文本编码器（XLM-R 4×512）和同一冻结图像特征上，"
        f"比较决策级 late fusion、特征级 embedding concat、表示级 cross-attention 三种融合方式。"
        f"三者在 5,100 人测试集上的准确率分别为 {fmt(r['late']['accuracy'])}、{fmt(r['concat']['accuracy'])}、"
        f"{fmt(r['cross']['accuracy'])}，都显著高于文本单模态 {fmt(r['text']['accuracy'])}；"
        f"但两两之间的准确率差异都不显著（concat vs late p=0.272；cross-attn vs concat p=0.213；"
        f"cross-attn vs late p=0.875）。<b>没有架构赢家</b>：late 最便宜、校准最好；concat 居中；"
        f"cross-attention 的 AUC/Brier/Log loss 最好但 ECE 最差、方差最大，且需要 3 seed 集成才达到 0.8610。"
        f"concat + late 的简单集成达到 0.8698（相对 late +0.98 pt，p=0.002），说明当前真正的增量在"
        f"逐用户可靠性，而不是融合层级。", abstract))
    story.append(Spacer(1, 5))

    story.append(table(
        ["方法", "Acc", "Macro-F1", "AUC", "ECE", "Brier", "Log loss", "Δtext"],
        [["文本 XLM-R", fmt(r["text"]["accuracy"]), fmt(r["text"]["macro_f1"]), fmt(r["text"]["auc"]),
          fmt(r["text"]["ece"]), fmt(r["text"]["brier"]), fmt(r["text"]["log_loss"]), "—"],
         ["图像十图线性头", fmt(r["image"]["accuracy"]), fmt(r["image"]["macro_f1"]), fmt(r["image"]["auc"]),
          fmt(r["image"]["ece"]), fmt(r["image"]["brier"]), fmt(r["image"]["log_loss"]), "-7.31"],
         ["Late fusion（堆叠）", fmt(r["late"]["accuracy"]), fmt(r["late"]["macro_f1"]), fmt(r["late"]["auc"]),
          fmt(r["late"]["ece"]), fmt(r["late"]["brier"]), fmt(r["late"]["log_loss"]), "+3.04"],
         ["Embedding concat（门控）", fmt(r["concat"]["accuracy"]), fmt(r["concat"]["macro_f1"]),
          fmt(r["concat"]["auc"]), fmt(r["concat"]["ece"]), fmt(r["concat"]["brier"]),
          fmt(r["concat"]["log_loss"]), "+2.53"],
         ["Cross-attention（3-seed 集成）", fmt(r["cross"]["accuracy"]), fmt(r["cross"]["macro_f1"]),
          fmt(r["cross"]["auc"]), fmt(r["cross"]["ece"]), fmt(r["cross"]["brier"]),
          fmt(r["cross"]["log_loss"]), "+3.14"],
         ["附加：concat + late 集成", fmt(r["ensemble"]["accuracy"]), fmt(r["ensemble"]["macro_f1"]),
          fmt(r["ensemble"]["auc"]), fmt(r["ensemble"]["ece"]), fmt(r["ensemble"]["brier"]),
          fmt(r["ensemble"]["log_loss"]), "+4.02"]],
        [47 * mm, 15 * mm, 16 * mm, 14 * mm, 13 * mm, 14 * mm, 16 * mm, 18 * mm],
        numeric=tuple(range(1, 8)),
        cap="表 1　总体结果（PAN 2018 官方测试集 n=5,100；概率经 calib 温度缩放）"))

    story.extend([Paragraph("1　三种方法", h2)])
    story.append(Paragraph(
        "<b>Late fusion</b>：文本/图像各自输出后验，在 calib 上做校准 log-pool 或堆叠 LR"
        "（logit_text、logit_image、交互项、图像有效标记、语言 one-hot）。编码器完全冻结，只训练一个很小的融合层。", bullet, bulletText="•"))
    story.append(Paragraph(
        "<b>Embedding concat</b>：文本 4 个 chunk 的 CLS 均值与十图 ViT CLS 均值各投影到 256 维，"
        "经 LayerNorm 后用逐用户门控缩放图像分支，拼接为 512 维，接 512→256→2 的 MLP；"
        "XLM-R 从零端到端微调，图像编码器冻结。", bullet, bulletText="•"))
    story.append(Paragraph(
        "<b>Cross-attention</b>：文本 token（4×512，投影 256）作为 query，最多 10 个图像 CLS token"
        "（投影 256）作为 memory，接 2 层 TransformerDecoder（8 heads，d=256，FFN=1024），"
        "null token 处理无图用户，masked mean pooling 后接 256→2 分类头；XLM-R 微调，图像编码器冻结。", bullet, bulletText="•"))

    story.append(Paragraph("2　分语言结果", h2))
    story.append(table(
        ["方法", "英语 (1,900)", "西语 (2,200)", "阿语 (1,000)", "总体"],
        [["文本", fmt(lang["text"]["en"]["accuracy"]), fmt(lang["text"]["es"]["accuracy"]),
          fmt(lang["text"]["ar"]["accuracy"]), fmt(r["text"]["accuracy"])],
         ["Late fusion", fmt(lang["late"]["en"]["accuracy"]), fmt(lang["late"]["es"]["accuracy"]),
          fmt(lang["late"]["ar"]["accuracy"]), fmt(r["late"]["accuracy"])],
         ["Embedding concat", fmt(lang["concat"]["en"]["accuracy"]), fmt(lang["concat"]["es"]["accuracy"]),
          fmt(lang["concat"]["ar"]["accuracy"]), fmt(r["concat"]["accuracy"])],
         ["Cross-attention", fmt(lang["cross"]["en"]["accuracy"]), fmt(lang["cross"]["es"]["accuracy"]),
          fmt(lang["cross"]["ar"]["accuracy"]), fmt(r["cross"]["accuracy"])],
         ["concat + late", fmt(lang["ensemble"]["en"]["accuracy"]), fmt(lang["ensemble"]["es"]["accuracy"]),
          fmt(lang["ensemble"]["ar"]["accuracy"]), fmt(r["ensemble"]["accuracy"])]],
        [52 * mm, 26 * mm, 26 * mm, 26 * mm, 20 * mm],
        numeric=(1, 2, 3, 4),
        cap="表 2　分语言 Accuracy"))
    story.append(Paragraph(
        "没有一种方法在所有语言上最好：cross-attention 英语 0.8779（最高），但西语 0.8527（最低）；"
        "late fusion 西语 0.8814（最高）；concat 较均衡；集成在西语最高（0.8882）。"
        "这说明三种方法学到的是不同语言上的不同决策边界，而不是一种方法全面更优。", body))

    story.append(Paragraph("3　显著性", h2))
    story.append(table(
        ["对比", "Δ Acc (pp)", "bootstrap 95% CI", "McNemar b/c", "p"],
        [["Concat vs Late", "-0.51", "[-1.39, +0.37]", "272/246", "0.272"],
         ["Cross-attn vs Concat", "+0.61", "[-0.29, +1.53]", "275/306", "0.213"],
         ["Cross-attn vs Late", "+0.10", "[-0.86, +1.12]", "323/328", "0.875"],
         ["Concat+late vs Late", "+0.98", "[+0.35, +1.61]", "103/153", "0.002"],
         ["Late vs Text", "+3.04", "[+2.25, +3.86]", "141/296", "<0.001"],
         ["Concat vs Text", "+2.53", "[+1.49, +3.55]", "297/426", "<0.001"],
         ["Cross-attn vs Text", "+3.14", "[+1.98, +4.27]", "366/526", "<0.001"]],
        [48 * mm, 20 * mm, 37 * mm, 26 * mm, 20 * mm],
        numeric=(1, 2, 3, 4),
        cap="表 3　配对比较（b = 前者对/后者错，c = 前者错/后者对；McNemar 双侧）"))
    story.append(Paragraph(
        "三种融合方式之间没有显著差异；三者相对文本都显著。唯一显著的“方法间”差异来自"
        " concat+late 集成相对 late（+0.98 pt，p=0.002）。", body))

    story.append(PageBreak())
    story.append(Paragraph("4　为什么三种方式打平", h2))
    story.append(Paragraph(
        "当前三种方法的图像输入都是每用户一个池化向量或 10 个图像级 CLS token，文本也是池化向量或"
        " chunk token；没有 patch 级视觉信息，也没有“哪条推文对应哪张图”的监督。"
        "此时 late 做的是“两个后验的可靠加权”，concat 做的是“两个池化向量的联合分类头”，"
        "cross-attention 做的是“文本 token 对 10 个图像 token 的注意力”——三者能利用的信息高度重叠，"
        "所以准确率都落在 0.85–0.86。要体现 cross-attention 的优势，需要 token/patch 级交互和更强的"
        "图文对应信号；在当前的图像 token 粒度下，它没有额外的信息可挖。", body))
    story.append(Paragraph(
        "这也解释了为什么 cross-attention 的方差更大、ECE 更差：它多出来的容量主要在学习"
        "语言/样本特定的噪声，而不是稳定的跨模态证据。单 seed 均值 0.8516 ± 0.0095，"
        "反而低于 concat 的 0.8549；只有 3-seed 集成才到 0.8610。", body))

    story.append(Paragraph("5　校准、成本与互补性", h2))
    story.append(Paragraph(
        "Late fusion 的 ECE 最低（0.0137），且只需训练一个很小的 stacker，是最省成本的方案；"
        "concat 的 ECE 0.0149，和 late 接近；cross-attention 的 AUC/Brier/Log loss 最好，"
        "但 ECE 0.0252，且需要 3 seed 和额外 decoder。若对外输出置信度，late/concat 更安全。", body))
    story.append(Paragraph(
        "三种方法之外，文本与图像至少一方正确的 oracle 上限是 0.9347，而三者都在 0.855–0.861；"
        "concat 与 late 只在 518/5,100 人上不一致，oracle 选择可达 0.9082。简单的 concat+late logit 集成"
        "达到 0.8698（+0.98 pt，p=0.002），ECE 0.0069。剩余空间主要在逐用户判断该信谁，"
        "而不是再换一种融合架构。", body))

    for fig, cap, width in [
        ("three_fusion_overall.png", "图 1　三种融合方式的 Accuracy / AUC / ECE", 168),
        ("three_fusion_per_language.png", "图 2　分语言 Accuracy", 150),
        ("three_fusion_crossattn_seeds.png", "图 3　Cross-attention 的 seed 波动与 3-seed 集成", 125),
    ]:
        p = FIGS / fig
        if p.exists():
            img = Image(str(p), width=width * mm, height=width * mm * 0.48)
            story.append(KeepTogether([img, Paragraph(cap, caption)]))

    story.append(Paragraph("6　结论与建议", h2))
    for lead, txt in [
        ("没有架构赢家：", "三种方法 0.8549–0.8610，两两 p=0.213–0.875；都显著高于文本（+2.5～+3.1 pt）。"),
        ("增益来自证据，不是层级：", "十图图像单模态已到 0.7565；融合的增量主要来自图像信息，oracle 还有约 7.5–8 pt 空间。"),
        ("生产默认：", "late fusion——最便宜、最稳、ECE 最好；现有服务可继续用。"),
        ("研究基线：", "embedding concat——P1a 的清晰基线；cross-attention 没有显著超过它（p=0.213），不应仅为“更复杂”而替换。"),
        ("cross-attention 的定位：", "当前输入是每图 CLS token，不是 patch；要做真正的 token 级交互，需要 patch/word token 和图文对应监督，否则额外复杂度买不到准确率。"),
        ("实用最佳：", "concat + late 集成 0.8698；需独立验证/多 seed 后再上线。"),
        ("下一步优先级：", "多 seed 与等价性检验 → 语言条件化融合 → 图像质量/人脸检测/十图 attention pooling → 逐用户可靠性 meta-classifier → 多方法集成。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("7　局限", h2))
    for txt in [
        "Late/concat 为单 seed；cross-attention 为 3 seed，跨方法 seed 预算不对等。",
        "训练预算不同：late 只训 stacker；concat/cross-attn 都微调 XLM-R 4 epoch；cross-attn 额外训 decoder 和 3 个 seed。",
        "Cross-attention 使用的是每图 CLS token，不是 patch token；本报告不能否定 patch 级 cross-attention 的潜力。",
        "PAN18 只有账号级性别标签，没有年龄/真实身份；资料图是 Twitter 头像，不是监控/证件照。",
        "calib 同时用于早停、温度和 stacker/集成；test 5,100 全程未参与训练或调参。",
    ]:
        story.append(Paragraph(txt, bullet, bulletText="•"))

    story.append(Paragraph("8　复现", h2))
    for line in [
        "python src/pan18_joint_fusion.py text --chunks 4",
        "python src/pan18_joint_fusion.py concat --chunks 4 --gate",
        "python src/pan18_joint_fusion.py late",
        "python src/pan18_joint_fusion.py ensemble",
        "python src/pan18_image_tokens.py",
        "python src/pan18_joint_fusion.py crossattn --chunks 4 --epochs 4 --batch-size 8 --lr-encoder 1e-5",
        "python src/eval_crossattn.py",
        "python src/make_three_fusion_figures.py",
        "python src/build_three_fusion_report.py",
    ]:
        story.append(Paragraph(line, body_ni))

    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
