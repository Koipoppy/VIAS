# -*- coding: utf-8 -*-
"""Build the PAN18 four-way fusion comparison report (PDF)."""
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
FOLDER = ARCHIVE / "07_2026-11-27_四种融合对比"
OUT = FOLDER / "PAN2018_Four_Fusion_Comparison_2026-11-27.pdf"
BASE = ROOT / "outputs" / "pan18_joint" / "metrics.json"
CROSS = ROOT / "outputs" / "pan18_joint" / "crossattn_2026-11-27" / "metrics_crossattn.json"
ATTR = FOLDER / "four_fusion_attribution.json"

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
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.2, leading=15.3,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=9.9, leading=16.6,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=20, spaceAfter=4)
body_ni = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
bullet = ParagraphStyle("Bullet", parent=body, leftIndent=20, bulletIndent=9,
                        firstLineIndent=0, spaceAfter=3, bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=11.4, leading=15,
                    textColor=TEXT, spaceBefore=9, spaceAfter=4, keepWithNext=True)
caption = ParagraphStyle("Cap", fontName="SimSun", fontSize=8.8, leading=11.5,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=2, spaceAfter=5)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=8.4, leading=11.8,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK",
                      spaceBefore=2, spaceAfter=4)
th = ParagraphStyle("TH", fontName="SimHei", fontSize=8.0, leading=10.2,
                    textColor=TEXT, alignment=TA_CENTER, wordWrap="CJK")
tc = ParagraphStyle("TC", fontName="SimSun", fontSize=7.9, leading=10.2,
                    textColor=TEXT, alignment=TA_LEFT, wordWrap="CJK")
tcc = ParagraphStyle("TCC", parent=tc, alignment=TA_CENTER)


def fmt(v, d=4):
    return f"{v:.{d}f}"


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
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
    ]))
    if cap:
        return KeepTogether([Paragraph(cap, caption), t, Spacer(1, 6)])
    return t


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE.read_text(encoding="utf-8"))["results"]
    cross = json.loads(CROSS.read_text(encoding="utf-8"))
    attr = json.loads(ATTR.read_text(encoding="utf-8"))
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

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                            topMargin=17 * mm, bottomMargin=17 * mm,
                            title="PAN 2018 四种多模态融合方式对比", author="VIAS")
    story = []
    story.append(Paragraph("PAN 2018 四种多模态融合方式对比", title))
    story.append(Paragraph("Late Fusion ｜ Embedding Concat ｜ Cross-Attention ｜ Concat+Late 集成", subtitle))
    story.append(Spacer(1, 2))
    story.append(Paragraph("2026 年 11 月 27 日　·　数据：PAN 2018 Author Profiling（EN/ES/AR）", meta))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT, spaceAfter=7))

    story.append(Paragraph(
        f"<b>摘要</b>　在同一份 PAN 2018 数据、同一文本编码器（XLM-R 4×512）和同一冻结图像特征上，"
        f"比较四种融合方案：① 决策级 late fusion、② 特征级 embedding concat、③ 表示级 cross-attention、"
        f"④ 在 ① 与 ② 的 logit 上再做小集成。四者在 5,100 人测试集上的准确率分别为 "
        f"{fmt(r['late']['accuracy'])}、{fmt(r['concat']['accuracy'])}、{fmt(r['cross']['accuracy'])}、"
        f"{fmt(r['ensemble']['accuracy'])}，都显著高于文本单模态 {fmt(r['text']['accuracy'])}。"
        f"前三者两两之间没有显著差异（② vs ① p=0.272；③ vs ② p=0.213；③ vs ① p=0.875）；"
        f"<b>④ 是唯一显著超过 ① 与 ② 的方案</b>（+0.98 / +1.49 pt，p=0.002 / p<0.001），"
        f"但与 ③ 的差异仍不显著（+0.84 pt，p=0.073）。提升归因显示：旧 late 0.7931 → 新 0.8598 的 "
        f"+6.67 pt 中，文本证据约 +4.14 pt、图像证据约 +3.10 pt、交互约 -0.57 pt；"
        f"<b>绝对收益文本略多，图像是让融合开始有用的那一半</b>。", abstract))
    story.append(Spacer(1, 5))

    story.append(table(
        ["方案", "Acc", "Macro-F1", "AUC", "ECE", "Brier", "Log loss", "Δtext"],
        [["文本 XLM-R", fmt(r["text"]["accuracy"]), fmt(r["text"]["macro_f1"]), fmt(r["text"]["auc"]),
          fmt(r["text"]["ece"]), fmt(r["text"]["brier"]), fmt(r["text"]["log_loss"]), "—"],
         ["图像十图线性头", fmt(r["image"]["accuracy"]), fmt(r["image"]["macro_f1"]), fmt(r["image"]["auc"]),
          fmt(r["image"]["ece"]), fmt(r["image"]["brier"]), fmt(r["image"]["log_loss"]), "-7.31"],
         ["① Late fusion", fmt(r["late"]["accuracy"]), fmt(r["late"]["macro_f1"]), fmt(r["late"]["auc"]),
          fmt(r["late"]["ece"]), fmt(r["late"]["brier"]), fmt(r["late"]["log_loss"]), "+3.04"],
         ["② Embedding concat", fmt(r["concat"]["accuracy"]), fmt(r["concat"]["macro_f1"]),
          fmt(r["concat"]["auc"]), fmt(r["concat"]["ece"]), fmt(r["concat"]["brier"]),
          fmt(r["concat"]["log_loss"]), "+2.53"],
         ["③ Cross-attention", fmt(r["cross"]["accuracy"]), fmt(r["cross"]["macro_f1"]),
          fmt(r["cross"]["auc"]), fmt(r["cross"]["ece"]), fmt(r["cross"]["brier"]),
          fmt(r["cross"]["log_loss"]), "+3.14"],
         ["④ Concat + late 集成", fmt(r["ensemble"]["accuracy"]), fmt(r["ensemble"]["macro_f1"]),
          fmt(r["ensemble"]["auc"]), fmt(r["ensemble"]["ece"]), fmt(r["ensemble"]["brier"]),
          fmt(r["ensemble"]["log_loss"]), "+4.02"]],
        [44 * mm, 15 * mm, 16 * mm, 14 * mm, 13 * mm, 14 * mm, 16 * mm, 17 * mm],
        numeric=tuple(range(1, 8)),
        cap="表 1　总体结果（PAN 2018 官方测试集 n=5,100；概率经 calib 温度缩放）"))

    story.append(Paragraph("1　四种方案", h2))
    for lead, txt in [
        ("① Late fusion：", "文本/图像各自输出后验，在 calib 上做校准 log-pool 或堆叠 LR（logit_text、logit_image、交互项、图像有效标记、语言 one-hot）；编码器冻结，只训练一个很小的融合层。"),
        ("② Embedding concat：", "文本 4 个 chunk 的 CLS 均值与十图 ViT CLS 均值各投影到 256 维，经 LayerNorm 后逐用户门控缩放图像分支，拼接为 512 维，接 512→256→2 MLP；XLM-R 从零端到端微调，图像编码器冻结。"),
        ("③ Cross-attention：", "文本 token（4×512，投影 256）作为 query，最多 10 个图像 CLS token（投影 256）作为 memory，接 2 层 TransformerDecoder（8 heads，d=256，FFN=1024）；null token 处理无图用户，masked mean pooling 后接 256→2 头；XLM-R 微调，图像编码器冻结。"),
        ("④ Concat + late 集成：", "在 calib 上用 5 参数 LR 对 ② 与 ① 的 logit（加语言 one-hot）做集成；不训练任何编码器。它是“融合的融合”，不是新的单模态模型。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("2　分语言结果", h2))
    story.append(table(
        ["方案", "英语 (1,900)", "西语 (2,200)", "阿语 (1,000)", "总体"],
        [["文本", fmt(lang["text"]["en"]["accuracy"]), fmt(lang["text"]["es"]["accuracy"]),
          fmt(lang["text"]["ar"]["accuracy"]), fmt(r["text"]["accuracy"])],
         ["① Late fusion", fmt(lang["late"]["en"]["accuracy"]), fmt(lang["late"]["es"]["accuracy"]),
          fmt(lang["late"]["ar"]["accuracy"]), fmt(r["late"]["accuracy"])],
         ["② Embedding concat", fmt(lang["concat"]["en"]["accuracy"]), fmt(lang["concat"]["es"]["accuracy"]),
          fmt(lang["concat"]["ar"]["accuracy"]), fmt(r["concat"]["accuracy"])],
         ["③ Cross-attention", fmt(lang["cross"]["en"]["accuracy"]), fmt(lang["cross"]["es"]["accuracy"]),
          fmt(lang["cross"]["ar"]["accuracy"]), fmt(r["cross"]["accuracy"])],
         ["④ Concat + late", fmt(lang["ensemble"]["en"]["accuracy"]), fmt(lang["ensemble"]["es"]["accuracy"]),
          fmt(lang["ensemble"]["ar"]["accuracy"]), fmt(r["ensemble"]["accuracy"])]],
        [47 * mm, 26 * mm, 26 * mm, 26 * mm, 20 * mm],
        numeric=(1, 2, 3, 4),
        cap="表 2　分语言 Accuracy"))
    story.append(Paragraph(
        "没有一种方案在所有语言上最好：③ 英语 0.8779（最高），但西语 0.8527（最低）；"
        "① 西语 0.8814（最高）；② 较均衡；④ 在西语最高（0.8882）。"
        "这说明三种基础融合学到的是不同语言上的不同决策边界，而不是一种方案全面更优。", body))

    story.append(Paragraph("3　显著性", h2))
    story.append(table(
        ["对比", "Δ Acc (pp)", "bootstrap 95% CI", "McNemar b/c", "p"],
        [["② vs ①", "-0.51", "[-1.39, +0.37]", "272/246", "0.272"],
         ["③ vs ②", "+0.61", "[-0.29, +1.53]", "275/306", "0.213"],
         ["③ vs ①", "+0.10", "[-0.86, +1.12]", "323/328", "0.875"],
         ["④ vs ①", "+0.98", "[+0.35, +1.61]", "103/153", "0.002"],
         ["④ vs ②", "+1.49", "[+0.88, +2.10]", "93/169", "<0.001"],
         ["④ vs ③*", "+0.84", "[-0.08, +1.73]", "297/254", "0.073"],
         ["① vs 文本", "+3.04", "[+2.25, +3.86]", "141/296", "<0.001"],
         ["② vs 文本", "+2.53", "[+1.49, +3.55]", "297/426", "<0.001"],
         ["③ vs 文本", "+3.14", "[+1.98, +4.27]", "366/526", "<0.001"],
         ["④ vs 文本", "+4.02", "[+3.12, +4.90]", "165/370", "<0.001"]],
        [38 * mm, 20 * mm, 38 * mm, 25 * mm, 19 * mm],
        numeric=(1, 2, 3, 4),
        cap="表 3　配对比较（b = 前者对/后者错，c = 前者错/后者对；McNemar 双侧）"))
    story.append(Paragraph(
        "①–③ 之间没有显著差异；④ 显著超过 ① 与 ②，但与 ③ 的差异仍不显著（p=0.073）。"
        "* ④ vs ③ 使用 ③ 的 3-seed 平均预测（0.8614）；官方 3-seed 集成为 0.8610，结论不变。", note))

    story.append(PageBreak())
    story.append(Paragraph("4　提升归因：文本 vs 图像", h2))
    story.append(Paragraph(
        "为了回答“昨天到今天的提升主要来自哪里”，用旧（2026-11-26）与新（2026-11-27）的组件后验做四组组合："
        "在同一旧 calib（1,499 人）上拟合统一 stacker，在同一官方测试集（5,100 人）上评估。"
        "旧文本 = TF-IDF 全文；旧图像 = 首图 + ViT 分类头（0.573）；"
        "新文本 = XLM-R 4×512（0.8296）；新图像 = 十图均值 + 线性头（0.7565）。", body))
    ar = attr["results"]
    story.append(table(
        ["组合", "测试 Acc", "相对旧 late"],
        [["旧文本 + 旧图像", fmt(ar["old_text_old_image"]["accuracy"]), "—"],
         ["新文本 + 旧图像", fmt(ar["new_text_old_image"]["accuracy"]),
          f"+{ar['new_text_old_image']['delta_vs_old'] * 100:.2f} pp"],
         ["旧文本 + 新图像", fmt(ar["old_text_new_image"]["accuracy"]),
          f"+{ar['old_text_new_image']['delta_vs_old'] * 100:.2f} pp"],
         ["新文本 + 新图像", fmt(ar["new_text_new_image"]["accuracy"]),
          f"+{ar['new_text_new_image']['delta_vs_old'] * 100:.2f} pp"]],
        [58 * mm, 30 * mm, 30 * mm],
        numeric=(1, 2),
        cap="表 4　提升归因（旧/新组件组合；统一 stacker 在旧 calib 上拟合）"))
    story.append(Paragraph(
        "总提升 0.7931 → 0.8598 = +6.67 pt。文本单独贡献 +4.14 pt，图像单独贡献 +3.10 pt，"
        "交互约 -0.57 pt（轻微次可加）。<b>绝对分数提升里文本略多于图像，不是“主要靠图像”。</b>"
        "但“融合相对文本的增量”从 +0.3 pt 变成 +3.0 pt，这部分是图像证据质量带来的："
        "图像从首图分类头 0.573 提升到十图均值 + 线性头 0.7565。", body))
    story.append(Paragraph(
        "更准确的说法：<b>昨天到今天的主要提升来自“证据质量/覆盖度”，文本与图像各占一半左右；"
        "图像是让融合开始有用的那一半，文本是抬高整体基线的那一半。</b>"
        "文本的关键是 4×512 分块（单块 512 截断只有 calib 0.772）；图像的关键是用满 10 张图 + "
        "学一个线性决策头，而不是换了更强的视觉模型。融合架构本身不是主因。", body))
    if (FOLDER / "four_fusion_attribution.png").exists():
        story.append(KeepTogether([
            Image(str(FOLDER / "four_fusion_attribution.png"), width=165 * mm, height=165 * mm * 0.47),
            Paragraph("图 1　文本/图像证据更新的归因", caption)]))

    story.append(Paragraph("5　集成为什么有效，以及它带来的启示", h2))
    story.append(Paragraph(
        "④ 不是“再训练一个更大的模型”，而是在 calib 上用 5 个参数的 LR 对 ② 和 ① 的 logit 做集成。"
        "它有效的直接原因是：② 与 ① 的错误互补。两者在 518/5,100 人上预测不一致；"
        "oracle 选择可达 0.9082，而 ④ 达到 0.8698，相当于解决了约 62% 的分歧。"
        "具体到指标：④ 的 ECE 0.0069、Brier 0.0975、AUC 0.9356，都是四种方案中最好的。", body))
    for lead, txt in [
        ("启示一：不要执着于“选一个架构”。", "①–③ 打平，④ 用几行代码把 ①+② 合成后显著超过两者；模型多样性比单架构复杂度更值钱。"),
        ("启示二：可靠性/选择层是下一桶金。", "oracle 选择 ①/② 可达 0.9082，而 ④ 只到 0.8698；一个更好的逐用户选择器（输入一致性、语言、图像置信度、文本长度）可能继续提升。"),
        ("启示三：集成成本极低。", "④ 不需要重训编码器，只需保留 ① 的后验和 ② 的 checkpoint；代价是推理时要跑两个后端。"),
        ("启示四：语言校准很重要。", "④ 在西语 0.8882（最高），③ 在英语 0.8779（最高）；语言 one-hot 的 stacker 已经在做校准，进一步可以做语言条件化门控。"),
        ("谨慎：④ 是探索性结果。", "三种集成口径 0.8647–0.8698，且口径选择发生在看过组件测试结果之后；上线前需独立划分/多 seed 验证。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("6　为什么 ①–③ 会打平", h2))
    story.append(Paragraph(
        "当前三种方法的图像输入都是每用户一个池化向量或 10 个图像级 CLS token，文本也是池化向量或"
        " chunk token；没有 patch 级视觉信息，也没有“哪条推文对应哪张图”的监督。"
        "此时 ① 做“两个后验的可靠加权”，② 做“两个池化向量的联合分类头”，③ 做“文本 token 对 10 个"
        "图像 token 的注意力”——三者能利用的信息高度重叠，所以准确率都落在 0.85–0.86。"
        "要体现 cross-attention 的优势，需要 patch/word token 和更强的图文对应监督；"
        "在当前图像 token 粒度下，它多出来的容量主要在学语言/样本特定噪声，所以方差更大、ECE 更差。", body))

    for fig, cap, width in [
        ("four_fusion_overall.png", "图 2　四种方案的 Accuracy / AUC / ECE", 168),
        ("four_fusion_per_language.png", "图 3　分语言 Accuracy", 150),
        ("four_fusion_crossattn_seeds.png", "图 4　Cross-attention 的 seed 波动与 3-seed 集成", 125),
    ]:
        p = FOLDER / fig
        if p.exists():
            story.append(KeepTogether([Image(str(p), width=width * mm, height=width * mm * 0.46),
                                       Paragraph(cap, caption)]))

    story.append(Paragraph("7　结论与建议", h2))
    for lead, txt in [
        ("没有单架构赢家：", "①–③ 0.8549–0.8610，两两 p=0.213–0.875；都显著高于文本（+2.5～+3.1 pt）。"),
        ("④ 是目前最好的实用方案：", "0.8698，显著超过 ①/②（p=0.002 / p<0.001），与 ③ 差异不显著（p=0.073）；ECE/Brier/AUC 都最优。"),
        ("生产默认：", "① late fusion——最便宜、最稳、ECE 好；现有服务可继续用。"),
        ("研究基线：", "② embedding concat——P1a 清晰基线；③ 没有显著超过它，不应仅为“更复杂”而替换。"),
        ("③ 的定位：", "当前每图 CLS token 不足以体现 token 级交互；要做 patch 级 cross-attention，需要更细粒度输入和图文对应监督。"),
        ("归因：", "文本证据 +4.14 pt、图像证据 +3.10 pt、交互 -0.57 pt；绝对收益文本略多，图像驱动融合增益。"),
        ("下一步：", "多 seed/等价性检验 → 语言条件化融合 → 图像质量/人脸检测/十图 attention pooling → 逐用户可靠性选择器 → 把 ③ 也纳入集成。"),
    ]:
        story.append(Paragraph(f"<b>{lead}</b>{txt}", bullet, bulletText="•"))

    story.append(Paragraph("8　局限", h2))
    for txt in [
        "①/② 为单 seed，③ 为 3 seed；跨方案 seed 预算不对等。",
        "训练预算不同：① 只训 stacker；②/③ 都微调 XLM-R 4 epoch；③ 额外训 decoder 和 3 个 seed。",
        "③ 使用的是每图 CLS token，不是 patch token；本报告不能否定 patch 级 cross-attention 的潜力。",
        "④ 是探索性结果：三种集成口径 0.8647–0.8698，且口径选择发生在看过组件测试结果之后。",
        "提升归因使用旧 calib（1,499 人）拟合统一 stacker；与各方法原始 stacker 不完全相同，数值是近似分解。",
        "PAN18 只有账号级性别标签，没有年龄/真实身份；资料图是 Twitter 头像，不是监控/证件照。",
    ]:
        story.append(Paragraph(txt, bullet, bulletText="•"))

    story.append(Paragraph("9　复现", h2))
    for line in [
        "python src/pan18_joint_fusion.py text --chunks 4",
        "python src/pan18_joint_fusion.py concat --chunks 4 --gate",
        "python src/pan18_joint_fusion.py late",
        "python src/pan18_joint_fusion.py ensemble",
        "python src/pan18_image_tokens.py",
        "python src/pan18_joint_fusion.py crossattn --chunks 4 --epochs 4 --batch-size 8 --lr-encoder 1e-5",
        "python src/eval_crossattn.py",
        "python src/eval_fusion_attribution.py",
        "python src/make_four_fusion_figures.py",
        "python src/build_four_fusion_report.py",
    ]:
        story.append(Paragraph(line, body_ni))

    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
