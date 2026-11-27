# -*- coding: utf-8 -*-
"""Build the audio multimodal dataset/fusion report (PDF)."""
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
FOLDER = ARCHIVE / "08_2026-11-28_音频多模态测试"
OUT = FOLDER / "AUDIO_MULTIMODAL_FUSION_REPORT_2026-11-28.pdf"
CASES = ROOT / "data" / "processed" / "audio" / "audio_multimodal_cases.csv"
METRICS = ROOT / "data" / "processed" / "audio" / "audio_final_metrics.json"

TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")
LIGHT = colors.HexColor("#F2F3F5")

pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

title = ParagraphStyle("Title", fontName="SimHei", fontSize=17, leading=23,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=5)
subtitle = ParagraphStyle("Sub", fontName="SimSun", fontSize=10.2, leading=15,
                          textColor=TEXT, alignment=TA_CENTER, spaceAfter=2)
meta = ParagraphStyle("Meta", fontName="SimSun", fontSize=9, leading=13,
                      textColor=MUTED, alignment=TA_CENTER)
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.2, leading=15.2,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=9.8, leading=16.4,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=20, spaceAfter=4)
body_ni = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=9,
                        firstLineIndent=0, spaceAfter=2.5, bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=11.3, leading=15,
                    textColor=TEXT, spaceBefore=9, spaceAfter=4, keepWithNext=True)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=8.7, leading=11.3,
                         textColor=TEXT, alignment=TA_CENTER, spaceBefore=2, spaceAfter=5)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=8.3, leading=11.7,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK", spaceBefore=2, spaceAfter=4)
th = ParagraphStyle("TH", fontName="SimHei", fontSize=8.0, leading=10.2,
                    textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
tc = ParagraphStyle("TC", fontName="SimSun", fontSize=7.9, leading=10.2,
                    textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
tcc = ParagraphStyle("TCC", parent=tc, alignment=TA_CENTER)


def fmt(v, d=4):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def table(header, rows, widths, numeric=(), cap=None):
    data = [[Paragraph(str(h), th) for h in header]]
    for row in rows:
        data.append([Paragraph(str(v), tcc if j in numeric else tc) for j, v in enumerate(row)])
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, TEXT), ("LINEBELOW", (0, 0), (-1, 0), 0.4, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]))
    return KeepTogether([Paragraph(cap, caption), t, Spacer(1, 6)]) if cap else t


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    import pandas as pd
    cases = pd.read_csv(CASES)
    n = m["n"]
    comp = []
    for b in ["18-24", "25-29", "30-34", "35-40"]:
        sub = cases[cases.age_bucket == b]
        comp.append([b, int((sub.gender == "female").sum()), int((sub.gender == "male").sum()), len(sub)])

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=17 * mm, bottomMargin=17 * mm,
                            title="音频多模态测试集与融合评估", author="VIAS")
    story = []
    story.append(Paragraph("音频多模态测试集与融合评估", title))
    story.append(Paragraph("CREMA-D（音频+人脸同人）× SMP 2016（合成文本）", subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("2026 年 11 月 28 日　·　232 例 / 58 位演员 / 按演员隔离", meta))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT, spaceAfter=7))

    g = m["gender"]; a = m["age"]
    story.append(Paragraph(
        f"<b>摘要</b>　我们从 CREMA-D 构建了一个音频+人脸同人的 232 例测试集，并用 SMP 2016 pilot 文本做"
        f"同性别/同年龄段的合成配对；按演员隔离为 fit {n['fit']} / calib {n['calib']} / test {n['test']}。"
        f"音频性别在 test 上校准后达到 {fmt(g['audio_cal']['test']['accuracy'])}，三模态融合也是 "
        f"{fmt(g['fusion_3']['test']['accuracy'])}；但这不是真实场景精度，而是表演语音的易任务。"
        f"年龄是瓶颈：音频年龄 zero-shot 原始 MAE ≈14.7 岁（系统性高估），线性校准后 "
        f"{fmt(a['audio_zero']['test']['mae'], 2)}；文本年龄 {fmt(a['text']['test']['mae'], 2)} 最好；"
        f"人脸年龄头 {fmt(a['face_head']['test']['mae'], 2)} / 桶准确率 "
        f"{fmt(a['face_head']['test']['bucket_accuracy'], 3)}；文本+人脸融合把年龄桶准确率提到 "
        f"{fmt(a['fusion_text_face']['test']['bucket_accuracy'], 3)}（MAE "
        f"{fmt(a['fusion_text_face']['test']['mae'], 2)}）。", abstract))
    story.append(Spacer(1, 5))

    story.append(Paragraph("1　数据集", h2))
    story.append(table(
        ["split", "样本", "演员", "用途"],
        [["fit", n["fit"], 28, "训练音频/人脸年龄头"],
         ["calib", n["calib"], 15, "阈值/温度/融合权重"],
         ["test", n["test"], 15, "最终评估"]],
        [30 * mm, 20 * mm, 20 * mm, 70 * mm], numeric=(1, 2),
        cap="表 1　按演员隔离的划分（同一演员不跨 split）"))
    story.append(table(
        ["年龄段", "女", "男", "合计"],
        [[r[0], r[1], r[2], r[3]] for r in comp],
        [35 * mm, 25 * mm, 25 * mm, 25 * mm], numeric=(1, 2, 3),
        cap="表 2　性别 × 年龄段（232 例）"))
    story.append(Paragraph(
        "音频与人脸来自同一 CREMA-D actor（WAV + 视频三帧）；文本来自 SMP pilot（中文），"
        "同性别、同年龄段，但不是同一人，且语言与音频不同。文本模型训练时已排除 pilot 50 人。", body))

    story.append(Paragraph("2　性别结果（test n=60）", h2))
    rows = []
    for key, label in [("text", "文本（原始）"), ("text_cal", "文本（Platt）"),
                       ("face", "人脸（原始 0.5）"), ("face_cal", "人脸（Platt）"),
                       ("audio", "音频（原始）"), ("audio_cal", "音频（Platt）"),
                       ("fusion_text_face", "文本+人脸"), ("fusion_3", "文本+人脸+音频")]:
        r = g[key]["test"]
        rows.append([label, fmt(r["accuracy"]), fmt(r["macro_f1"]), fmt(r["auc"]),
                     fmt(r["ece"]), fmt(r["brier"])])
    story.append(table(["方法", "Acc", "Macro-F1", "AUC", "ECE", "Brier"], rows,
                       [48 * mm, 16 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm],
                       numeric=(1, 2, 3, 4, 5),
                       cap="表 3　性别指标（Platt 在 calib 上拟合，test 只用一次）"))
    story.append(Paragraph(
        "音频性别几乎满分（AUC 1.0，校准后 1.0）——CREMA-D 是表演语音，性别线索极强，不能外推。"
        "人脸性别 ViT 的 AUC 很高（0.982）但默认 0.5 阈值只有 0.617：模型对 CREMA-D 女性系统性低估，"
        "Platt 校准后到 0.833。这是“排序好、校准差”的典型。文本 0.75/0.833；音频缺失时文本+人脸 0.883。",
        body))
    if (FOLDER / "audio_gender_metrics.png").exists():
        story.append(KeepTogether([Image(str(FOLDER / "audio_gender_metrics.png"),
                                         width=155 * mm, height=155 * mm * 0.47),
                                   Paragraph("图 1　性别 Accuracy / AUC", caption)]))

    story.append(Paragraph("3　年龄结果（test n=60）", h2))
    rows = []
    for key, label in [("text", "文本（线性校准）"), ("face_zero", "人脸 zero-shot"),
                       ("audio_zero", "音频 zero-shot"), ("audio_head", "音频 age head"),
                       ("face_head", "人脸 age head"), ("fusion_text_face", "文本+人脸"),
                       ("fusion_3", "文本+人脸+音频"), ("prob_avg3", "三路概率平均")]:
        r = a[key]["test"]
        rows.append([label, fmt(r["mae"], 2), f"{r['bias']:+.2f}",
                     fmt(r["bucket_accuracy"], 3), fmt(r["pm1"], 3)])
    story.append(table(["方法", "MAE（岁）", "Bias", "桶准确率", "±1 桶"], rows,
                       [50 * mm, 22 * mm, 20 * mm, 25 * mm, 22 * mm],
                       numeric=(1, 2, 3, 4),
                       cap="表 4　年龄指标（线性/ridge 校准在 calib 上拟合）"))
    story.append(Paragraph("注：zero-shot 行的 MAE 已应用 calib 上的线性校准；原始音频年龄 zero-shot MAE≈14.7 岁（bias +14.2）。", note))
    story.append(Paragraph(
        "音频年龄是主要短板：zero-shot 原始 MAE ≈14.7 岁、bias +14.2（严重高估）；线性校准后才到 4.02。"
        "用冻结音频 embedding 训练 4 段年龄头（fit 演员）得到 4.63，没有改善——112 个 fit 片段不足以"
        "训练可靠年龄头。人脸年龄头把桶准确率从 0.283 提到 0.483；文本年龄 MAE 最低 3.63，但桶准确率"
        "只有 0.383；文本+人脸融合把桶准确率提到 0.500，是当前最好的年龄组合。", body))
    if (FOLDER / "audio_age_metrics.png").exists():
        story.append(KeepTogether([Image(str(FOLDER / "audio_age_metrics.png"),
                                         width=165 * mm, height=165 * mm * 0.42),
                                   Paragraph("图 2　年龄 MAE / 桶准确率", caption)]))

    story.append(Paragraph("4　结论与启示", h2))
    for lead, txt in [
        ("音频性别很强，但任务太易：", "CREMA-D 表演语音 + 固定台词，音频性别校准后 1.0；不能当成真实社媒/监控精度。"),
        ("校准比换模型更急：", "人脸性别排序 AUC 0.982、默认阈值 0.617；Platt 后 0.833。直接 0.5 阈值会系统性误判女性。"),
        ("年龄是瓶颈：", "音频年龄原始高估 14 岁；线性校准 4.02；音频 age head 4.63 无改善。人脸 age head 4.38/桶 0.483；文本 3.63 最好。"),
        ("融合的条件性：", "文本+人脸年龄桶 0.500 最好；加音频反而 0.467。融合前必须先校准/筛选模态，否则弱音频会拖累。"),
        ("Fallback 可用：", "音频缺失时文本+人脸性别 0.883、年龄 3.77/0.500；系统不是只能依赖音频。"),
        ("数据仍是合成：", "音频+人脸同人，但文本是跨语言、不同人的合成配对；不能用于真实身份结论。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("5　局限与下一步", h2))
    for txt in [
        "CREMA-D 是 12 句固定台词的表演语音，20–40 岁；不能代表真实社媒/监控音频。",
        "文本为中文 SMP pilot，音频为英文；文本是合成配对，不是同一人。",
        "test 只有 60 段、15 位演员，且每演员 4 段相关；指标方差大，未做置信区间。",
        "音频年龄 head 在 112 个 fit 片段上过拟合；需要更多演员/目标域数据或更强的年龄模型。",
        "人脸模型存在群体偏差；未做人脸检测/质量门控。",
        "下一步：自发中文语音、40+ 年龄、同人文本+语音、噪声/混响/缺失模态测试、服务化音频解析器。",
    ]:
        story.append(Paragraph(txt, bullet, bulletText="•"))

    story.append(Paragraph("6　复现", h2))
    for line in [
        "python src/audio_prepare_crema.py",
        "python src/audio_extract_audio.py",
        "python src/audio_extract_face.py",
        "python src/audio_build_cases.py",
        "python src/audio_train_heads.py",
        "python src/audio_eval_final.py",
        "python src/make_audio_figures.py",
        "python src/build_audio_report.py",
    ]:
        story.append(Paragraph(line, body_ni))

    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
