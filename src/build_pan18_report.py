# -*- coding: utf-8 -*-
"""Build the PAN18 full multimodal gender evaluation report."""
from __future__ import annotations

import json
import os
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
from reportlab.platypus import (CondPageBreak, HRFlowable, Image, KeepTogether,
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "报告归档"
OUT = ARCHIVE / "05_2026-11-26_PAN2018全量评估" / "PAN2018_Multimodal_Gender_Report_2026-11-26.pdf"
FIG_COMP = ROOT / "outputs" / "pan18_figures" / "pan18_composition.png"
FIG_METHOD = ROOT / "outputs" / "pan18_figures" / "pan18_methods.png"
FIG_CONF = ROOT / "outputs" / "pan18_figures" / "pan18_confusions.png"
METRICS = ROOT / "outputs" / "pan18" / "fusion_metrics.json"

TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")

pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

title = ParagraphStyle("Title", fontName="SimHei", fontSize=18, leading=24,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=6)
author = ParagraphStyle("Author", fontName="SimSun", fontSize=10.5, leading=16,
                        textColor=TEXT, alignment=TA_CENTER)
link = ParagraphStyle("Link", fontName="SimSun", fontSize=9, leading=13,
                      textColor=TEXT, alignment=TA_CENTER)
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.5, leading=16,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=10.5, leading=18,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=21)
bullet_style = ParagraphStyle("Bullet", parent=body, leftIndent=21,
                              bulletIndent=10, firstLineIndent=0, spaceAfter=3,
                              bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=12, leading=16,
                    textColor=TEXT, spaceBefore=12, spaceAfter=6)
h3 = ParagraphStyle("H3", fontName="SimHei", fontSize=10.5, leading=15,
                    textColor=TEXT, spaceBefore=8, spaceAfter=3)
table_header = ParagraphStyle("TH", fontName="SimHei", fontSize=8.8, leading=11,
                              textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
table_cell = ParagraphStyle("TC", fontName="SimSun", fontSize=8.8, leading=11,
                            textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
table_cell_c = ParagraphStyle("TCC", parent=table_cell, alignment=TA_CENTER)
table_caption = ParagraphStyle("TCap", fontName="SimSun", fontSize=9, leading=12,
                               textColor=TEXT, alignment=TA_CENTER,
                               spaceBefore=4, spaceAfter=2)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=9, leading=12,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=3, spaceAfter=6)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=9, leading=13,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK")


def heading(t):
    return [Paragraph(t, h2)]


def bullet(lead, text):
    return Paragraph(f"<b>{lead}</b> {text}", bullet_style, bulletText="•")


def _widths(header, rows, pad=12):
    widths = []
    for j in range(len(header)):
        w = pdfmetrics.stringWidth(str(header[j]), "SimHei", 8.8)
        for row in rows:
            w = max(w, pdfmetrics.stringWidth(str(row[j]), "SimSun", 8.8))
        widths.append(w + pad)
    total = sum(widths)
    return [x * W / total for x in widths] if total > W else widths


def table(header, rows, numeric=()):
    data = [[Paragraph(str(h), table_header) for h in header]]
    for row in rows:
        data.append([Paragraph(str(v), table_cell_c if j in numeric else table_cell)
                     for j, v in enumerate(row)])
    t = Table(data, colWidths=_widths(header, rows), hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, TEXT),
        ("LINEBELOW", (0, 0), (-1, 0), .6, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    global W
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="PAN 2018 多模态性别推断全量评估",
        author="", creator="VIAS",
        subject="PAN 2018 Author Profiling full test evaluation",
    )
    W = doc.width
    story = []
    allres = m["results"]["all"]
    labels = {"text_only": "文本", "image_only": "图像", "equal_prob": "等权",
              "credibility": "可信度", "calibrated_late": "校准融合",
              "stacked_lr": "堆叠融合", "conflict_handled": "冲突处理"}
    best = max(allres, key=lambda k: allres[k]["accuracy"])
    best_acc = allres[best]["accuracy"]
    best_name = labels.get(best, best)

    story.append(Paragraph("PAN 2018 多模态性别推断全量评估", title))
    story.append(Paragraph("2026 年 11 月 26 日", author))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        '数据：<link href="https://doi.org/10.5281/zenodo.3746006" color="#1A1A1A">'
        '<u>PAN 2018 Author Profiling · Zenodo 3746006</u></link>', link))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=.6, color=TEXT,
                            spaceBefore=0, spaceAfter=8))
    story.append(Paragraph(
        "<b>摘要：</b>本报告在 PAN 2018 Author Profiling 的官方训练/测试集全量数据上，评估文本、图像与"
        "多模态融合的性别推断。数据覆盖英语、西班牙语、阿拉伯语；训练集 7,500 人、测试集 5,100 人，"
        "每人 100 条推文与 10 张图片。文本用分语言字符 n-gram + 逻辑回归，图像用 ViT 性别分类器，"
        "在统一证据单元上比较单模态、等权融合、可信度加权、校准融合与堆叠融合。全量测试上最佳方法为"
        f"{best_name}（准确率 {best_acc:.3f}）。方法链沿用 VIAS pilot 报告（2026-11-25），本报告替换为"
        "PAN 2018 全量数据与分语言模型。", abstract))
    story.append(HRFlowable(width="100%", thickness=.5, color=TEXT,
                            spaceBefore=6, spaceAfter=8))

    story.extend(heading("1　研究对象"))
    story.append(Paragraph(
        "研究对象是 PAN 2018 Author Profiling 的性别推断任务：给定一个社交账号的 100 条推文和 10 张图片，"
        "判断账号主体为 male 或 female。PAN 2018 提供英语、西班牙语、阿拉伯语三个子集，标签为账号级性别；"
        "本报告不涉及年龄、真实身份或候选身份层，只评估性别这一属性目标。", body))
    story.extend(table_block("表 1　研究范围与数据边界", table(
        ("维度", "设置"),
        [
            ("任务", "账号级性别二分类"),
            ("语言", "英语 / 西班牙语 / 阿拉伯语"),
            ("输入", "100 条推文 + 10 张图片 / 用户"),
            ("训练 / 测试", "7,500 / 5,100 用户（官方划分）"),
            ("不覆盖", "年龄、真实身份、候选身份、跨平台账号对齐"),
        ],
    )))

    story.extend(heading("2　系统搭建"))
    story.append(Paragraph(
        "系统沿用 pilot 的五段链路：多模态解析、统一证据单元、可信度、加权融合、冲突/解释。"
        "本报告的重点是把数据换成 PAN 2018 全量，并按语言分别训练文本模型；图像模型为公开预训练 ViT，"
        "不再使用 pilot 的中文文本模型或 AFAD 人脸配对。", body))
    story.extend(table_block("表 2　PAN 2018 上的实现", table(
        ("模块", "实现"),
        [
            ("文本解析", "XML 推文拼接；字符 2–5 gram TF-IDF；分语言 LogisticRegression"),
            ("图像解析", "ViT gender-classification；每用户首张可用图"),
            ("证据单元", "23 字段 Schema v1.1；text/image 两类证据"),
            ("可信度", "六维，文本按长度、图像按质量；权重参与融合"),
            ("融合", "等权、可信度加权、校准权重、堆叠 LR；冲突处理"),
        ],
    )))

    story.extend(heading("3　评估场景"))
    story.append(Paragraph(
        "评估使用官方训练集训练、官方测试集全量测试；文本模型按语言独立训练，图像模型语言无关。"
        "为校准融合权重，从训练集按语言 × 性别分层抽取 20% 作为校准集，其余用于文本拟合；"
        "测试集完全不参与训练或调参。", body))
    story.extend(table_block("表 3　评估方法与指标", table(
        ("方法", "说明"),
        [
            ("文本单模态", "分语言 TF-IDF + LR 的 P(female)"),
            ("图像单模态", "每用户首张图 ViT 概率"),
            ("等权融合", "文本/图像概率 0.5/0.5"),
            ("可信度融合", "证据单元六维权重，weight_power=1.5"),
            ("校准融合", "在校准集上搜索文本权重"),
            ("堆叠融合", "文本/图像概率与交互特征训练 LR"),
            ("冲突处理", "文本/图像冲突时降权，τ=0.65"),
        ],
    )))
    story.append(Paragraph(
        "指标为账号级 Accuracy、Macro-F1 与 AUC；按英语、西班牙语、阿拉伯语及三者合并报告。", body))

    story.extend(heading("4　样本构造"))
    story.append(Paragraph(
        "PAN 2018 官方数据按语言和性别平衡：训练集英语 3,000、西语 3,000、阿语 1,500；"
        "测试集英语 1,900、西语 2,200、阿语 1,000。每名用户有 100 条推文和 10 张图片；"
        "图像和文本均来自公开 PAN 数据，按研究用途使用，不重新分发。", body))
    story.extend(figure(FIG_COMP, W, 165, "图 1　PAN 2018 训练/测试用户构成"))
    story.extend(table_block("表 4　全量数据规模", table(
        ("语言", "训练", "测试", "文本/用户", "图像/用户"),
        [
            ("英语", "3,000", "1,900", "100", "10"),
            ("西班牙语", "3,000", "2,200", "100", "10"),
            ("阿拉伯语", "1,500", "1,000", "100", "10"),
            ("合计", "7,500", "5,100", "100", "10"),
        ],
        numeric=(1, 2, 3, 4),
    )))
    story.append(Paragraph(
        "图像侧 5,097/5,100 用户有可用首图；3 例缺失按中性概率 0.5 处理，不改变用户级全量统计口径。", note))

    story.append(PageBreak())
    story.extend(heading("5　结果与结论"))
    order = ["text_only", "image_only", "equal_prob", "credibility",
             "calibrated_late", "stacked_lr", "conflict_handled"]
    story.extend(table_block("表 5　全量测试集总体结果", table(
        ("方法", "Accuracy", "Macro-F1", "AUC"),
        [(label, fmt(allres[k]["accuracy"]), fmt(allres[k]["macro_f1"]),
          fmt(allres[k]["auc"])) for k, label in [
              ("text_only", "文本"), ("image_only", "图像"), ("equal_prob", "等权"),
              ("credibility", "可信度"), ("calibrated_late", "校准融合"),
              ("stacked_lr", "堆叠融合"), ("conflict_handled", "冲突处理")]],
        numeric=(1, 2, 3),
    )))
    story.extend(table_block("表 6　分语言 Accuracy", table(
        ("方法", "英语", "西班牙语", "阿拉伯语"),
        [(label, fmt(m["results"]["en"][k]["accuracy"]),
          fmt(m["results"]["es"][k]["accuracy"]),
          fmt(m["results"]["ar"][k]["accuracy"])) for k, label in [
              ("text_only", "文本"), ("image_only", "图像"), ("equal_prob", "等权"),
              ("credibility", "可信度"), ("calibrated_late", "校准融合"),
              ("stacked_lr", "堆叠融合"), ("conflict_handled", "冲突处理")]],
        numeric=(1, 2, 3),
    )))
    story.extend(figure(FIG_METHOD, W, 180, "图 2　分语言方法对比"))
    story.extend(figure(FIG_CONF, W * .88, 165, "图 3　堆叠融合的性别混淆矩阵（全量 / 英语）"))

    text_en = m["results"]["en"]["text_only"]["accuracy"]
    text_es = m["results"]["es"]["text_only"]["accuracy"]
    text_ar = m["results"]["ar"]["text_only"]["accuracy"]
    img_en = m["results"]["en"]["image_only"]["accuracy"]
    img_es = m["results"]["es"]["image_only"]["accuracy"]
    img_ar = m["results"]["ar"]["image_only"]["accuracy"]
    story.append(Paragraph("5.1　结果解读", h3))
    story.append(Paragraph(
        f"全量测试上，文本单模态在英语/西语/阿语分别为 {text_en:.3f}/{text_es:.3f}/{text_ar:.3f}，"
        f"图像单模态为 {img_en:.3f}/{img_es:.3f}/{img_ar:.3f}。最佳总体方法为 {best_name}，"
        f"Accuracy={best_acc:.3f}，但相对文本单模态的提升只有约 {best_acc - allres['text_only']['accuracy']:.3f}，"
        "属于 pilot 级波动，不能视为稳定增益。", body))
    story.append(Spacer(1, 3))
    story.append(bullet("语言差异：", "文本模型对英语/西语更强；阿拉伯语数据量只有 1,500 训练用户，且字符形态不同，准确率最低。"))
    story.append(bullet("图像是弱信号：", "每用户首图的 ViT 性别准确率仅约 0.57；PAN 图片不保证是人脸，预训练人脸/性别模型与任务不完全匹配。"))
    story.append(bullet("等权融合会伤害文本：", "等权/可信度融合把弱图像信号直接注入，整体降到 0.710；校准融合把文本权重推到 0.95 后才回到文本水平。"))
    story.append(bullet("冲突处理需谨慎：", "当前冲突处理把文本/图像分歧视为证据冲突，整体降到 0.716；在图像弱、自然分歧多的场景应使用来源感知阈值。"))
    story.append(Paragraph("5.2　边界与下一步", h3))
    story.append(bullet("任务边界：", "本报告只评估性别，不评估项目书的真实身份定位；PAN 2018 没有年龄标签，也没有真实身份真值。"))
    story.append(bullet("数据口径：", "全量测试 5,100 用户；图像侧取每用户首张可用图，未使用 10 图平均。"))
    story.append(bullet("下一步：", "加入人脸检测/质量门控与全部 10 图聚合；对图像概率做校准；按语言学习融合权重；再接入候选身份层。"))
    story.append(Paragraph(
        "结论：PAN 2018 上的文本是主信号，图像是弱辅助信号；多模态融合必须按语言和来源质量校准，"
        "不能假设“文本+图像”必然提升。与 pilot 报告相比，本报告验证了同一套证据/可信度/融合链路在"
        "英文、西班牙文、阿拉伯文全量测试集上的可运行性，但性别属性仍不能等同于真实身份。", body))
    story.append(Paragraph(
        "数据与代码：PAN 2018 Author Profiling（Zenodo DOI 10.5281/zenodo.3746006）；"
        "方法与版式参考 VIAS Pilot Study Report 2026-11-25。", note))

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
