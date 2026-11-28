# -*- coding: utf-8 -*-
"""Build the audio fusion-scheme comparison report (PDF)."""
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
FOLDER = ARCHIVE / "09_2026-11-28_音频融合方案对比"
OUT = FOLDER / "AUDIO_FUSION_SCHEMES_REPORT_2026-11-28.pdf"
METRICS = ROOT / "data" / "processed" / "audio" / "audio_fusion_schemes_metrics.json"

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


def fmt(v, d=3):
    return f"{v:.{d}f}"


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
        ("TOPPADDING", (0, 0), (-1, -1), 3.0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    return KeepTogether([Paragraph(cap, caption), t, Spacer(1, 5)]) if cap else t


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    age, gender, comp = m["age"], m["gender"], m["comparisons"]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=17 * mm, bottomMargin=17 * mm,
                            title="音频多模态数据集上的融合方案对比", author="VIAS")
    story = []
    story.append(Paragraph("音频多模态数据集上的融合方案对比", title))
    story.append(Paragraph("Late Fusion ｜ Feature Concat ｜ Gated Fusion ｜ Attention Fusion", subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("2026 年 11 月 28 日　·　CREMA-D + SMP 2016　·　232 例 / 58 演员 / test 60", meta))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT, spaceAfter=7))

    story.append(Paragraph(
        f"<b>摘要</b>　在同一音频多模态测试集（CREMA-D 音频+人脸同人 × SMP 2016 合成文本，"
        f"232 例 / 58 演员，按演员隔离 fit 112 / calib 60 / test 60）上比较单模态、late fusion、"
        f"特征 concat、门控融合与轻量 attention 融合。年龄点估计最好的是 attention（MAE "
        f"{fmt(age['attention']['test']['mae'], 2)} 岁）和 concat text+face 的桶准确率"
        f"（{fmt(age['concat_text_face']['test']['bucket_accuracy'], 3)}）；但所有相对文本的 ΔMAE "
        f"95% CI 都包含 0，<b>没有统计上的融合赢家</b>。唯一 CI 不含 0 的比较是 "
        f"late ridge（含音频）比 late text+face <b>差 +0.184 岁</b>（CI [+0.036, +0.421]），"
        f"即音频年龄特征在拖后腿。性别任务被音频饱和（1.000），无法区分融合方案。", abstract))
    story.append(Spacer(1, 5))

    story.append(Paragraph("1　方案", h2))
    story.append(table(
        ["方案", "层级", "输入/训练"],
        [["单模态", "—", "text / face / audio 各自年龄证据；calib 线性校准"],
         ["Late equal", "决策层", "三个校准后年龄的等权平均（calib）"],
         ["Late ridge", "决策层", "三个校准后年龄 → ridge（calib）"],
         ["Late text+face", "决策层", "文本+人脸 → ridge（calib）"],
         ["Concat all", "特征层", "audio/face/text 各 PCA 32 → 96 维 → ridge（fit 训练，calib 选 alpha）"],
         ["Concat text+face", "特征层", "face+text 各 PCA 32 → 64 维 → ridge（fit 训练）"],
         ["Gated", "表示层", "三模态 embedding → 逐样本 3 路 softmax 门控 → 回归头（3 seed）"],
         ["Attention", "表示层", "三模态 token + 1 层 self-attention → 回归头（3 seed）"],
         ["Oracle", "上界", "逐样本选误差最小的单模态（不可部署）"]],
        [32 * mm, 20 * mm, 115 * mm],
        cap="表 1　融合方案与训练/校准口径"))

    story.append(Paragraph("2　年龄结果（test n=60）", h2))
    order = [("single_text", "文本"), ("single_face", "人脸"), ("single_audio", "音频"),
             ("late_equal", "Late equal"), ("late_ridge", "Late ridge"),
             ("late_text_face", "Late text+face"), ("concat_ridge", "Concat all"),
             ("concat_text_face", "Concat text+face"), ("gated", "Gated"),
             ("attention", "Attention"), ("oracle", "Oracle")]
    rows = []
    for key, label in order:
        r = age[key]["test"]
        ci = age[key].get("test_ci")
        ci_s = "—" if not ci else f"[{ci['mae_ci'][0]:.2f}, {ci['mae_ci'][1]:.2f}]"
        if key == "oracle":
            rows.append([label, f"{r['mae']:.3f}", f"{r['bias']:+.3f}", "—", "—", "—"])
        else:
            rows.append([label, f"{r['mae']:.3f}", f"{r['bias']:+.3f}",
                         f"{r['bucket_accuracy']:.3f}", f"{r['pm1']:.3f}", ci_s])
    story.append(table(["方案", "MAE", "Bias", "桶 Acc", "±1 桶", "MAE 95% CI"], rows,
                       [36 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 32 * mm],
                       numeric=(1, 2, 3, 4, 5),
                       cap="表 2　年龄指标（CI 为 actor-level bootstrap，2000 次）"))
    story.append(Paragraph(
        "Gated 三个 seed 的 MAE：3.778 / 4.211 / 3.207；attention：3.348 / 3.429 / 3.757。"
        "seed 方差与方案差异同量级。Oracle 的 MAE 按逐样本最小绝对误差选单模态，其桶准确率不是上界，故不列。", note))

    story.append(PageBreak())
    story.append(Paragraph("3　与文本 / late text+face 的配对比较", h2))
    rows = []
    for key, label in [("late_equal", "Late equal"), ("late_ridge", "Late ridge"),
                       ("late_text_face", "Late text+face"), ("concat_ridge", "Concat all"),
                       ("concat_text_face", "Concat text+face"), ("gated", "Gated"),
                       ("attention", "Attention")]:
        c = comp[key]
        vt, vl = c["vs_text"], c["vs_late_text_face"]
        if key == "late_text_face":
            late_delta, late_ci = "—", "—"
        else:
            late_delta = f"{vl['delta_mae']:+.3f}"
            late_ci = f"[{vl['ci'][0]:+.3f}, {vl['ci'][1]:+.3f}]"
        rows.append([label, f"{vt['delta_mae']:+.3f}", f"[{vt['ci'][0]:+.3f}, {vt['ci'][1]:+.3f}]",
                     late_delta, late_ci])
    story.append(table(["方案", "ΔMAE vs 文本", "95% CI", "ΔMAE vs Late T+F", "95% CI"], rows,
                       [35 * mm, 24 * mm, 36 * mm, 27 * mm, 36 * mm],
                       numeric=(1, 2, 3, 4),
                       cap="表 3　配对 ΔMAE（负值表示更好；CI 为 actor-level bootstrap）"))
    story.append(Paragraph(
        "所有“优于文本”的 ΔMAE CI 都包含 0；唯一 CI 不含 0 的是 Late ridge 比 Late text+face "
        "差 +0.184 岁（CI [+0.036, +0.421]）。因此当前数据不能证明任何复杂融合显著优于文本或简单 late。", body))

    if (FOLDER / "audio_fusion_age_mae.png").exists():
        story.append(KeepTogether([Image(str(FOLDER / "audio_fusion_age_mae.png"),
                                         width=168 * mm, height=168 * mm * 0.37),
                                   Paragraph("图 1　年龄 MAE 与 actor-level 95% CI", caption)]))
    if (FOLDER / "audio_fusion_age_bucket.png").exists():
        story.append(KeepTogether([Image(str(FOLDER / "audio_fusion_age_bucket.png"),
                                         width=168 * mm, height=168 * mm * 0.37),
                                   Paragraph("图 2　年龄桶准确率与 95% CI", caption)]))

    story.append(Paragraph("4　性别 sanity check", h2))
    g_order = [("text", "文本"), ("face", "人脸"), ("audio", "音频"),
               ("late_text_face", "Late text+face"), ("late_3", "Late 3"), ("concat_logreg", "Concat")]
    rows = []
    for key, label in g_order:
        r = gender[key]["test"]
        rows.append([label, f"{r['accuracy']:.3f}", f"{r['auc']:.3f}"])
    story.append(table(["方案", "Acc", "AUC"], rows, [60 * mm, 25 * mm, 25 * mm],
                       numeric=(1, 2), cap="表 4　性别指标（test n=60）"))
    story.append(Paragraph(
        "音频性别在 CREMA-D 上饱和（1.000），任何包含音频的融合都是 1.000；性别无法区分融合方案。"
        "文本+人脸 fallback 0.900。", body))
    if (FOLDER / "audio_fusion_gender.png").exists():
        story.append(KeepTogether([Image(str(FOLDER / "audio_fusion_gender.png"),
                                         width=145 * mm, height=145 * mm * 0.47),
                                   Paragraph("图 3　性别 Accuracy / AUC", caption)]))

    story.append(Paragraph("5　缺失音频 fallback", h2))
    rows = []
    for key, label in [("single_text", "文本"), ("single_face", "人脸"),
                       ("late_text_face", "Late text+face"), ("concat_text_face", "Concat text+face")]:
        r = age[key]["test"]
        rows.append([label, f"{r['mae']:.3f}", f"{r['bucket_accuracy']:.3f}"])
    story.append(table(["方案", "Age MAE", "年龄桶 Acc"], rows,
                       [60 * mm, 30 * mm, 30 * mm], numeric=(1, 2),
                       cap="表 5　音频缺失时的年龄 fallback"))
    story.append(Paragraph(
        "音频缺失时，concat text+face 的年龄桶准确率 0.700 最高；再次说明音频年龄特征是噪声源，"
        "不应无条件硬融。", body))

    story.append(Paragraph("6　解读与建议", h2))
    for lead, txt in [
        ("没有融合赢家：", "所有相对文本的 ΔMAE CI 含 0；点估计 attention 3.37 最好，但 seed 3.35–3.76，CI 宽。"),
        ("音频年龄拖后腿：", "Late ridge 含音频显著差于 text+face（+0.184 岁，CI 不含 0）；concat all 桶 0.617 < text+face 0.700。"),
        ("表示层融合未证明：", "gated/attention 点估计好但方差大；在 232 例/60 test 上不足以替换 late/concat。"),
        ("特征 concat 的亮点：", "concat text+face 桶准确率 0.700，为所有方案最高；但 MAE 3.809，与文本持平。"),
        ("性别已饱和：", "音频 1.000；融合比较只能看 fallback，不能看性别。"),
        ("默认建议：", "年龄用 late text+face / concat text+face；性别用音频；音频缺失用文本+人脸。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("7　局限", h2))
    for txt in [
        "test 仅 60 段 / 15 位演员，且每演员 4 段相关；actor-level bootstrap 仍不能覆盖训练随机性。",
        "文本是中文 SMP pilot 的合成配对，音频英文；不是同一人。",
        "CREMA-D 是表演语音、固定台词、20–40 岁；性别过易、年龄分布窄。",
        "gated/attention 用 3 seed 平均预测；late/concat 是确定性单次拟合。",
        "Oracle 只是诊断上界，不可部署；不含真实身份。",
    ]:
        story.append(Paragraph(txt, bullet, bulletText="•"))

    story.append(Paragraph("8　复现", h2))
    for line in [
        "python src/audio_text_embeddings.py",
        "python src/audio_fusion_schemes.py",
        "python src/make_audio_fusion_schemes_figures.py",
        "python src/build_audio_fusion_schemes_report.py",
    ]:
        story.append(Paragraph(line, body_ni))

    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
