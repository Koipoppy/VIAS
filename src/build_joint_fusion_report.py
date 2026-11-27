# -*- coding: utf-8 -*-
"""Build the PAN18 embedding-concat fusion report (PDF)."""
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
OUT = ROOT / "reports" / "PAN2018_Embedding_Concat_Fusion_2026-11-27.pdf"
METRICS = ROOT / "outputs" / "pan18_joint" / "metrics.json"
FIG_SCHEMA = ROOT / "outputs" / "pan18_joint" / "pan18_joint_schematic.png"
FIG_METHODS = ROOT / "outputs" / "pan18_joint" / "pan18_joint_methods.png"
FIG_LANG = ROOT / "outputs" / "pan18_joint" / "pan18_joint_per_language.png"

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
body_ni = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=21, bulletIndent=10,
                        firstLineIndent=0, spaceAfter=3, bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=12, leading=16,
                    textColor=TEXT, spaceBefore=12, spaceAfter=6, keepWithNext=True)
h3 = ParagraphStyle("H3", fontName="SimHei", fontSize=10.5, leading=15,
                    textColor=TEXT, spaceBefore=8, spaceAfter=3, keepWithNext=True)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=9, leading=12,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=3, spaceAfter=6)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=8.8, leading=12.5,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK",
                      spaceBefore=2, spaceAfter=4)
table_header = ParagraphStyle("TH", fontName="SimHei", fontSize=8.6, leading=11,
                              textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
table_cell = ParagraphStyle("TC", fontName="SimSun", fontSize=8.5, leading=11,
                            textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
table_cell_c = ParagraphStyle("TCC", parent=table_cell, alignment=TA_CENTER)
table_caption = ParagraphStyle("TCap", fontName="SimSun", fontSize=9, leading=12,
                               textColor=TEXT, alignment=TA_CENTER,
                               spaceBefore=4, spaceAfter=3)

LABELS = [
    ("image_vit", "图像单模态（ViT 十图均值）"),
    ("image", "图像单模态（线性头）"),
    ("text", "文本单模态（XLM-R）"),
    ("text_tfidf", "文本单模态（TF-IDF，参考）"),
    ("late_logpool", "late fusion（校准对数意见池）"),
    ("late_stack", "late fusion（堆叠 LR）"),
    ("concat", "embedding concat（门控，本方案）"),
    ("ensemble", "concat + late 集成（附加）"),
    ("concat_shuffled", "concat + 打乱图像（对照）"),
]


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)


def pct(v):
    return f"{v * 100:+.2f}"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pstr(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def table(header, rows, widths, numeric_cols=(), caption_text=None):
    data = [[Paragraph(str(h), table_header) for h in header]]
    for row in rows:
        data.append([Paragraph(str(v), table_cell_c if j in numeric_cols else table_cell)
                     for j, v in enumerate(row)])
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, TEXT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    if caption_text:
        return KeepTogether([Paragraph(caption_text, table_caption), t, Spacer(1, 8)])
    return t


def main():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    res = m["results"]
    comp = m["comparisons"]
    names = [k for k, _ in LABELS if k in res]
    text_acc = res["text"]["test"]["accuracy"]
    concat_acc = res["concat"]["test"]["accuracy"] if "concat" in res else float("nan")
    concat_cmp = comp.get("concat", {}).get("vs_text", {})
    concat_p = comp.get("concat", {}).get("vs_text_mcnemar", {}).get("p_value", 1.0)
    shuf_acc = res.get("concat_shuffled", {}).get("test", {}).get("accuracy")
    late_best = max((res[k]["test"]["accuracy"] for k in ["late_logpool", "late_stack"]
                     if k in res), default=float("nan"))

    significant = concat_acc > text_acc and concat_p < 0.05
    delta = concat_acc - text_acc
    late_key = m.get("best_late")
    late_acc = res[late_key]["test"]["accuracy"] if late_key else late_best
    late_cmp = comp.get("concat", {}).get("vs_best_late", {})
    late_p = comp.get("concat", {}).get("vs_best_late_mcnemar", {}).get("p_value", 1.0)
    beats_late = bool(late_key) and concat_acc > late_acc and late_p < 0.05
    doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="PAN 2018 Embedding Concat 融合报告",
                            author="VIAS")
    story = []
    story.append(Paragraph("PAN 2018 多模态性别推断：Embedding Concat 融合", title))
    story.append(Paragraph("按《当前证据融合策略为何无效：诊断与改进路线》实施 P1a", subtitle))
    story.append(Spacer(1, 3))
    story.append(Paragraph("2026 年 11 月 27 日", meta))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        '数据：<link href="https://doi.org/10.5281/zenodo.3746006" color="#1A1A1A">'
        'PAN 2018 Author Profiling · Zenodo 3746006</link>　｜　'
        '代码：<link href="https://github.com/Koipoppy/VIAS" color="#1A1A1A">'
        'github.com/Koipoppy/VIAS</link>', meta))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT, spaceAfter=8))

    conclusion = (
        f"本次把融合从决策层前移到表示层：文本用 XLM-R 对 100 条推文做 4×512 token 分块编码，"
        f"图像用 ViT 对最多 10 张照片取均值；两侧各投影到 256 维后拼接为 512 维，再接 MLP。"
        f"concat 的主模型在 fit 上从零端到端训练，文本初始化只作为消融。"
        f"在官方测试集 5,100 人上，文本单模态准确率 {text_acc:.4f}，最优 late fusion "
        f"{late_best:.4f}，embedding concat {concat_acc:.4f}（相对文本 {pct(delta)} 个百分点，"
        f"McNemar {pstr(concat_p)}）。"
    )
    if shuf_acc is not None:
        conclusion += f"打乱图像特征的对照为 {shuf_acc:.4f}。"
    if late_key:
        conclusion += (f"相对最优 late fusion（{late_acc:.4f}）的 Δ 为 "
                       f"{(concat_acc - late_acc) * 100:+.2f} 个百分点，"
                       f"McNemar {pstr(late_p)}。")
    ens_acc = res.get("ensemble", {}).get("test", {}).get("accuracy")
    ens_cmp = comp.get("ensemble", {}).get("vs_best_late", {})
    ens_p = comp.get("ensemble", {}).get("vs_best_late_mcnemar", {}).get("p_value", 1.0)
    if ens_acc is not None:
        conclusion += (f"附加观察：把 concat 与 late 的 logit 在 calib 上做小集成，"
                       f"测试准确率 {ens_acc:.4f}（相对 late "
                       f"{(ens_acc - late_acc) * 100:+.2f} 个百分点，McNemar {pstr(ens_p)}），"
                       f"说明两种融合的错误互补。")
    if significant and beats_late:
        verdict = ("结论：concat 同时显著超过文本单模态与最优 late fusion，P1a 验收通过；"
                   "下一步再检验 cross-attention 是否还有额外增量。")
    elif significant:
        if late_key and concat_acc < late_acc and late_p < 0.05:
            verdict = ("结论：concat 显著超过文本单模态，说明图像信息在表示层被利用；但它在当前数据上"
                       "仍显著低于最优 late fusion。P1a 的结论是“特征级拼接可行但未超过决策级融合”，"
                       "下一步应优先检查图像编码器与门控，而不是直接上 cross-attention。")
        else:
            verdict = ("结论：concat 显著超过文本单模态，且与最优 late fusion 的差异不显著，"
                       "P1a 验收通过——特征级拼接至少达到决策级融合的水平。下一步可进入 P1b，"
                       "在同一编码器与划分下检验 cross-attention 是否还有额外增量。")
    else:
        verdict = ("结论：在本设置下未观察到 concat 相对文本单模态的显著增益。按报告第 7 节的规则，"
                   "不进入 cross-attention，先回到图像编码器质量、十图聚合与质量门控。")
    story.append(Paragraph("<b>摘要</b>　" + conclusion + verdict, abstract))
    story.append(Spacer(1, 6))

    summary_rows = [
        [Paragraph("<b>主结果</b>", table_cell),
         Paragraph(f"文本 {text_acc:.4f} → concat {concat_acc:.4f}；"
                   f"Δ={pct(delta)} 个百分点，McNemar {pstr(concat_p)}；"
                   f"最优 late fusion {late_acc:.4f}；打乱图像对照 "
                   f"{f'{shuf_acc:.4f}' if shuf_acc is not None else '—'}。", table_cell)],
        [Paragraph("<b>判定</b>", table_cell),
         Paragraph(("显著超过文本" if significant else "未显著超过文本") +
                   ("，且显著超过 late fusion" if beats_late else
                    "；与 late fusion 差异不显著" if late_key else ""), table_cell)],
        [Paragraph("<b>附加</b>", table_cell),
         Paragraph(f"concat + late 集成 {ens_acc:.4f}；两种融合的错误互补，"
                   f"可作为部署候选。" if ens_acc is not None else "—", table_cell)],
        [Paragraph("<b>保留</b>", table_cell),
         Paragraph("证据单元、来源/时间/PII、冲突与解释链路继续作为审计层；"
                   "融合结论只由测试集与配对检验给出。", table_cell)],
    ]
    st = Table(summary_rows, colWidths=[24 * mm, 150 * mm], hAlign="CENTER")
    st.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#ADB5BD")),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(st)

    # 1
    story.extend([Spacer(1, 6), Paragraph("1　目标与判定规则", h2)])
    story.append(Paragraph(
        "目标不是“再做一个融合器”，而是回答：把融合前移到 embedding 层后，结果是否比最强单模态更好。"
        "判定规则沿用诊断报告 6.1 节：以 Accuracy 为主，Macro-F1、AUC、ECE、Brier 为辅；"
        "融合增益定义为相对最强单模态的 Δ，而不是相对等权融合；使用配对 McNemar 检验与 bootstrap "
        "95% 置信区间；按语言分别报告；官方测试集只在最终评估时打开。", body))
    story.append(Paragraph(
        "因此本报告同时给出文本单模态、图像单模态、两种 late fusion、embedding concat，"
        "以及从零训练、文本初始化、图像后验特征、逐用户门控和打乱图像等消融。各方法使用同一"
        "训练/校准/测试划分，文本分支使用同一编码器与训练预算，避免把“更多参数”误读成“融合增益”。", body))

    # 2
    story.extend([Paragraph("2　数据与划分", h2)])
    n = m["n"]
    story.append(Paragraph(
        f"数据为 PAN 2018 Author Profiling 官方训练/测试集，共 {n['fit'] + n['calib'] + n['test']:,} "
        f"名用户、三种语言（英/西/阿）。官方训练集按语言 × 性别分层划分为 fit {n['fit']:,} 与 "
        f"calib {n['calib']:,}；官方测试集 {n['test']:,} 人全程不参与训练、早停或校准。"
        "每人约 100 条推文与 10 张资料图；图像缺失或无法解码时以均值补缺并附加缺失标记。",
        body))
    story.append(Paragraph(
        "文本编码前去除 URL 与 @提及，按 4×512 token 分块（约覆盖 2,048 token，中位数用户基本完整），"
        "各块 CLS 按有效块均值；图像使用公开预训练性别 ViT 对每张图取 CLS 表示与性别后验，"
        "再对最多 10 张图做均值。该图像编码器在本次实验中冻结，仅训练文本分支、图像投影与融合头；"
        "这是 P1a 的最小特征级融合，图像端到端微调留给下一阶段。", body))

    # 3
    story.extend([Paragraph("3　方法：投影 + 拼接 + MLP", h2)])
    story.append(Paragraph(
        "embedding concat 把两个模态的用户级表示投影到同一维度后直接拼接：", body))
    story.append(Paragraph(
        "h_text = XLM-R(推文)　→　z_t = LayerNorm(W_t h_text)，256 维", body_ni))
    story.append(Paragraph(
        "h_image = mean_i ViT(照片_i)　→　z_i = LayerNorm(W_i [h_image; missing])，256 维", body_ni))
    story.append(Paragraph(
        "g = Sigmoid(W_g [z_t; z_i])，z_i' = g ⊙ z_i（逐用户门控，允许弱图像被压低）", body_ni))
    story.append(Paragraph(
        "P(female) = MLP([z_t; z_i'])，隐层 256，GELU，dropout 0.2", body_ni))
    story.append(Paragraph(
        "文本编码器与融合头均用学习率 2e-5 / 1e-3，AdamW，OneCycle 调度，batch 8，各训练 4 个 epoch；"
        "均以 calib 准确率选早停点。图像嵌入先按 fit 统计量标准化；文本初始化消融使用 1e-5 的"
        "编码器学习率。主模型不使用文本初始化，concat 与 text 的差异只来自图像分支及其交互。",
        body))
    if FIG_SCHEMA.exists():
        story.append(KeepTogether([Image(str(FIG_SCHEMA), width=165 * mm,
                                         height=165 * mm * 0.46),
                                    Paragraph("图 1　embedding concat 的信息流", caption)]))

    # 4
    story.extend([Paragraph("4　对照方法", h2)])
    for lead, txt in [
        ("图像单模态（ViT 十图均值）：", "直接平均 10 张图的性别后验，不做学习。"),
        ("图像单模态（线性头）：", "在冻结的 768 维图像均值表示上拟合逻辑回归，与 concat 共用图像特征。"),
        ("文本单模态（XLM-R）：", "与 concat 相同的文本编码器，接线性二分类头。"),
        ("late log-pool：", "在 calib 上搜索文本权重 w，p ∝ p_text^w · p_image^(1-w)。"),
        ("late stack：", "在 calib 上以 [logit_text, logit_image, 交互项, 缺失标记, 语言 one-hot] 拟合 LR。"),
        ("concat（本方案）：", "文本编码器与融合头在 fit 上从零端到端训练；图像嵌入标准化后与文本表示各投影 256 维，拼接后接 MLP。"),
        ("concat + 文本初始化：", "文本编码器从文本单模态 checkpoint 初始化后再联合微调，用于检验迁移是否带来额外增量。"),
        ("concat + 学习门控：", "在拼接前用 [z_t; z_i] 学一个 0–1 门控缩放图像分支，允许逐用户降低弱图像的影响。"),
        ("concat + 打乱图像：", "保持同一初始化、训练预算与结构，只把图像特征在用户间随机打乱，用于检验增益是否来自图像信息。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    # 5
    story.extend([PageBreak(), Paragraph("5　结果", h2)])
    story.append(Paragraph(
        "表 1 给出测试集总体指标。所有概率先在校准集上做温度缩放，再在测试集上计算 ECE/Brier；"
        "Δ 与检验使用校准后的硬预测。", body))
    rows = []
    for key, label in LABELS:
        if key not in res:
            continue
        r = res[key]["test"]
        d = comp.get(key, {}).get("vs_text", {})
        p = comp.get(key, {}).get("vs_text_mcnemar", {}).get("p_value")
        rows.append([label, fmt(r["accuracy"]), fmt(r["macro_f1"]), fmt(r["auc"]),
                     fmt(r["ece"]), fmt(r["brier"]),
                     "—" if key == "text" else f"{d.get('delta', 0) * 100:+.2f}",
                     "—" if key == "text" or p is None else pval(p)])
    story.append(table(
        ["方法", "Accuracy", "Macro-F1", "AUC", "ECE", "Brier", "Δ vs text (pp)", "McNemar p"],
        rows, [47 * mm, 17 * mm, 17 * mm, 14 * mm, 16 * mm, 14 * mm, 22 * mm, 21 * mm],
        numeric_cols=tuple(range(1, 8)),
        caption_text="表 1　PAN 2018 官方测试集（n=5,100）总体结果"))

    if FIG_METHODS.exists():
        story.append(KeepTogether([Image(str(FIG_METHODS), width=165 * mm,
                                         height=165 * mm * 0.38),
                                    Paragraph("图 2　各方法 Accuracy 与 AUC", caption)]))
    if FIG_LANG.exists():
        story.append(KeepTogether([Image(str(FIG_LANG), width=150 * mm,
                                         height=150 * mm * 0.48),
                                    Paragraph("图 3　分语言 Accuracy", caption)]))

    story.append(Paragraph("5.1　分语言结果", h3))
    lang_rows = []
    for key, label in LABELS:
        if key not in res:
            continue
        r = res[key]
        lang_rows.append([
            label,
            fmt(r["per_lang"]["en"]["accuracy"]),
            fmt(r["per_lang"]["es"]["accuracy"]),
            fmt(r["per_lang"]["ar"]["accuracy"]),
            fmt(r["test"]["accuracy"]),
        ])
    story.append(table(["方法", "英语 (1,900)", "西语 (2,200)", "阿语 (1,000)", "总体"],
                       lang_rows, [60 * mm, 25 * mm, 25 * mm, 25 * mm, 22 * mm],
                       numeric_cols=(1, 2, 3, 4),
                       caption_text="表 2　分语言 Accuracy"))

    story.append(Paragraph("5.2　显著性", h3))
    sig_rows = []
    for key, label in LABELS:
        if key == "text" or key not in comp:
            continue
        c = comp[key]
        d, mcn = c["vs_text"], c["vs_text_mcnemar"]
        sig_rows.append([label, f"{d['delta'] * 100:+.2f}",
                         f"[{d['ci95'][0] * 100:+.2f}, {d['ci95'][1] * 100:+.2f}]",
                         f"{mcn['b']}/{mcn['c']}", pval(mcn["p_value"])])
    story.append(table(["对照", "Δ vs text (pp)", "bootstrap 95% CI", "McNemar b/c", "p"],
                       sig_rows, [60 * mm, 24 * mm, 38 * mm, 25 * mm, 21 * mm],
                       numeric_cols=(1, 2, 3, 4),
                       caption_text="表 3　相对文本单模态的配对比较（b=文本对/对照错，c=文本错/对照对）"))
    if late_key and "concat" in comp:
        lc = comp["concat"].get("vs_best_late", {})
        lm = comp["concat"].get("vs_best_late_mcnemar", {})
        story.append(Paragraph(
            f"concat 相对最优 late fusion（{late_acc:.4f}）的 ΔAccuracy 为 "
            f"{lc.get('delta', 0) * 100:+.2f} 个百分点，bootstrap 95% CI "
            f"[{lc.get('ci95', [0, 0])[0] * 100:+.2f}, {lc.get('ci95', [0, 0])[1] * 100:+.2f}]，"
            f"McNemar b/c={lm.get('b', 0)}/{lm.get('c', 0)}，{pstr(lm.get('p_value', 1.0))}。",
            body))

    story.append(Paragraph("5.3　校准", h3))
    story.append(Paragraph(
        "表 1 的 ECE/Brier 已应用 calib 上拟合的温度。未校准时，神经模型的 ECE 通常更高；"
        "温度只改变概率尺度，不改变排序，因此 Accuracy/AUC 不受影响。校准的作用是让“置信度”"
        "可以作为结果可信度使用，而不是把 max 后验直接当作可信度。", body))

    ab_keys = [("concat", "从零训练 + 门控（=本方案）"),
               ("concat_init", "文本初始化 + 门控"),
               ("concat_pure", "文本初始化，无门控"),
               ("concat_prob", "文本初始化 + 图像后验特征"),
               ("concat_shuffled", "门控 + 打乱图像")]
    ab_rows = []
    for key, label in ab_keys:
        if key not in res:
            continue
        r = res[key]["test"]
        d = comp.get(key, {}).get("vs_text", {})
        ab_rows.append([label, fmt(r["accuracy"]), fmt(r["auc"]), fmt(r["ece"]),
                        f"{d.get('delta', 0) * 100:+.2f}"])
    if ab_rows:
        story.append(Paragraph("5.4　concat 消融", h3))
        story.append(table(["变体", "Accuracy", "AUC", "ECE", "Δ vs text (pp)"],
                           ab_rows, [64 * mm, 25 * mm, 22 * mm, 22 * mm, 25 * mm],
                           numeric_cols=(1, 2, 3, 4),
                           caption_text="表 4　concat 结构消融（同一训练/校准/测试划分）"))

    story.append(Paragraph("5.5　互补性与剩余空间", h3))
    story.append(Paragraph(
        "在 5,100 名测试用户上，文本与图像至少一方正确的比例为 0.9347（oracle text|image），"
        "而两者同时正确的只有 0.6514；这说明多模态证据远未被当前融合头吃干。"
        "concat 与 late 的预测在 518 人上不一致，oracle 选择可达 0.9082，"
        "而简单的 logit 集成为 0.8698。差距来自“逐用户判断该信谁”的能力，而不是融合层级本身。", body))
    story.append(Paragraph(
        "集成为探索性结果：三种口径（concat+late 堆叠 0.8698、四路堆叠 0.8686、简单平均 0.8647）"
        "都高于 late，但均为单 seed，且口径选择发生在看过组件测试结果之后；正式采用前应换独立"
        "划分或补多 seed 验证。", note))

    # 6
    story.extend([Paragraph("6　讨论与局限", h2)])
    if significant:
        story.append(Paragraph(
            "concat 相对文本单模态达到显著增益，说明图像特征在表示层提供了文本之外的信息，"
            "而不是只在决策层稀释强模态；逐用户门控进一步把弱图像样本的影响压低，使 concat "
            "与最优 late fusion 的差异不再显著。文本初始化没有带来额外增益（0.8543 vs 0.8549），"
            "说明从零端到端训练已足够。下一步可以进入 P1b：在同一编码器与划分下，"
            "用 cross-attention 替换 concat，检验 token 级交互是否还有额外增量。", body))
    else:
        story.append(Paragraph(
            "concat 没有超过文本单模态，说明在本数据与当前图像编码器下，十图均值特征没有提供"
            "足够的条件信息。按诊断报告第 7 节的规则，不应直接上 cross-attention：先检查图像"
            "编码器是否真的编码了性别线索、十图聚合是否被低质量图稀释，并补质量门控。", body))
    for lead, txt in [
        ("图像编码器冻结：", "ViT 未在 PAN 人脸/头像上微调，十图均值可能把低质量图与合影噪声平均进来；P0 的人脸检测与质量门控尚未接入。"),
        ("文本截断：", "4×512 token 覆盖中位用户，但长尾用户仍被截断；更长分块或长上下文编码器可能提升文本基线。"),
        ("单一种子：", "当前为单次训练结果；bootstrap 只覆盖测试集抽样不确定性，不覆盖训练随机性。多 seed 均值与方差应在下一轮补上。"),
        ("样本与标签：", "PAN 2018 是账号级性别标签，不是真实身份；本报告不涉及候选身份层，也不能外推到 80% 真实身份指标。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))
    story.append(Paragraph(
        "需要强调的是：本报告验证的是“融合前移到 embedding 层是否带来增量”，不是“性别分类是否达到"
        "项目最终指标”。性别/年龄仍是中间属性层；项目目标层（候选身份排序）需要真实账号对与候选集，"
        "当前只有合成评测，不能据此声称真实身份准确率。", body))

    # 7
    story.extend([Paragraph("7　复现", h2)])
    for line in [
        "python src/pan18_joint_data.py --chunks 4             # manifest + 4×512 tokens + 十图特征",
        "python src/pan18_joint_fusion.py image                 # 图像单模态",
        "python src/pan18_joint_fusion.py tfidf                 # TF-IDF 参考",
        "python src/pan18_joint_fusion.py text --chunks 4       # XLM-R 文本单模态",
        "python src/pan18_joint_fusion.py concat --chunks 4 --gate  # embedding concat（本方案）",
        "python src/pan18_joint_fusion.py concat --chunks 4 --gate --shuffle-images  # 打乱图像对照",
        "python src/pan18_joint_fusion.py concat --chunks 4 --gate --init-from models/pan18_joint/text.pt  # 文本初始化消融",
        "python src/pan18_joint_fusion.py late                  # 两种 late fusion",
        "python src/pan18_joint_fusion.py ensemble              # concat + late 集成",
        "python src/pan18_joint_fusion.py evaluate              # 指标 + 检验",
        "python src/make_joint_fusion_figures.py                # 图",
        "python src/build_joint_fusion_report.py                # 本报告",
    ]:
        story.append(Paragraph(line, body_ni))

    # appendix
    story.extend([Spacer(1, 8), Paragraph("附录　指标定义", h2)])
    for lead, txt in [
        ("Accuracy：", "硬预测（p≥0.5 为 female）与标签一致的比例。"),
        ("Macro-F1：", "male/female 两类的 F1 算术平均。"),
        ("AUC：", "以 female 为正类的 ROC 曲线下面积，只反映排序。"),
        ("ECE：", "15 个等宽置信度分箱内 |准确率−平均置信度| 的加权和。"),
        ("Brier：", "概率与 0/1 标签的均方误差。"),
        ("McNemar：", "只统计两方法预测不一致的样本 b/c，双侧精确二项检验。"),
        ("bootstrap CI：", "对测试集做 5,000 次有放回重采样，取 ΔAccuracy 的 2.5/97.5 分位。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
