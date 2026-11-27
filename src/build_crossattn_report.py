# -*- coding: utf-8 -*-
"""Build the PAN18 concat vs cross-attention report (PDF + Markdown)."""
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
OUT = ROOT / "outputs" / "pan18_joint" / "crossattn_2026-11-27"
METRICS = OUT / "metrics_crossattn.json"
ARCHIVE = ROOT.parent / "报告归档"
PDF = ARCHIVE / "06_2026-11-27_融合诊断与Concat" / "PAN2018_CrossAttention_vs_Concat_2026-11-27.pdf"
MD = ARCHIVE / "06_2026-11-27_融合诊断与Concat" / "PAN2018_CrossAttention_vs_Concat_2026-11-27.md"

TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")
LIGHT = colors.HexColor("#F2F3F5")
ACCENT = colors.HexColor("#2F5D8A")

pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

title = ParagraphStyle("Title", fontName="SimHei", fontSize=18, leading=25,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=6)
subtitle = ParagraphStyle("Sub", fontName="SimSun", fontSize=11, leading=16,
                          textColor=TEXT, alignment=TA_CENTER, spaceAfter=2)
meta = ParagraphStyle("Meta", fontName="SimSun", fontSize=8.5, leading=12,
                      textColor=MUTED, alignment=TA_CENTER)
h1 = ParagraphStyle("H1", fontName="SimHei", fontSize=14, leading=19,
                    textColor=TEXT, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=11.5, leading=16,
                    textColor=TEXT, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("Body", fontName="SimSun", fontSize=9.6, leading=15,
                      textColor=TEXT, alignment=TA_LEFT, spaceAfter=4)
body_ni = ParagraphStyle("BodyNI", parent=body, firstLineIndent=0)
small = ParagraphStyle("Small", fontName="SimSun", fontSize=8.2, leading=12,
                       textColor=MUTED, alignment=TA_LEFT)
cell = ParagraphStyle("Cell", fontName="SimSun", fontSize=8.2, leading=11,
                      textColor=TEXT, alignment=TA_LEFT)
cell_c = ParagraphStyle("CellC", parent=cell, alignment=TA_CENTER)
head = ParagraphStyle("Head", fontName="SimHei", fontSize=8.4, leading=11,
                      textColor=colors.white, alignment=TA_CENTER)


def f4(x: float) -> str:
    return f"{x:.4f}"


def pct(x: float) -> str:
    return f"{x * 100:+.2f}"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def get(results: dict, method: str, key: str = "test", default=None):
    return results.get(method, {}).get(key, default)


def make_table(rows, widths, header_bg=ACCENT, font_size=8.2) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "SimHei"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    table.setStyle(TableStyle(style))
    return table


def main() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    r, seeds, ss = m["results"], m["seeds"], m["seed_summary"]
    concat, cross = r["concat"]["test"], r["crossattn_ens"]["test"]
    text = r["text"]["test"]
    late = r["late_stack"]["test"]
    image = r["image"]["test"]
    cmp = m["comparisons"]["crossattn_ens"]
    c_concat = cmp["vs_concat"]
    c_late = cmp["vs_late_stack"]
    c_text = cmp["vs_text"]
    lang_delta = {
        lang: (r["crossattn_ens"]["per_lang"][lang]["accuracy"]
               - r["concat"]["per_lang"][lang]["accuracy"])
        for lang in ["en", "es", "ar"]
    }

    d_concat = c_concat["delta"]["delta"]
    ci_concat = c_concat["delta"]["ci95"]
    p_concat = c_concat["mcnemar"]["p_value"]
    d_late = c_late["delta"]["delta"]
    p_late = c_late["mcnemar"]["p_value"]
    d_text = c_text["delta"]["delta"]
    p_text = c_text["mcnemar"]["p_value"]

    if cross["accuracy"] > concat["accuracy"] and p_concat < 0.05:
        verdict = (f"cross-attention 集成在测试集上比本机 concat 高 {pct(d_concat)} 个百分点，"
                   f"McNemar p={pval(p_concat)}，达到统计显著。")
        mech_sentence = ("说明在图像信号较弱、图像编码器冻结的条件下，token 级交互仍带来了"
                         "统计显著的增益。")
        conclusion_extra = "融合机制本身带来了可测的增益，但仍需在更强图像编码器上复核。"
    elif cross["accuracy"] > concat["accuracy"]:
        verdict = (f"cross-attention 集成在数值上比 concat 高 {pct(d_concat)} 个百分点，"
                   f"但 McNemar p={pval(p_concat)}，差异不显著。")
        mech_sentence = "说明 token 级交互的数值增益不足以在 5,100 人测试集上稳定显著。"
        conclusion_extra = "融合机制不是主要瓶颈；图像质量、图像编码器和数据规模更值得优先投入。"
    else:
        verdict = (f"cross-attention 集成没有超过本机 concat（{f4(cross['accuracy'])} vs "
                   f"{f4(concat['accuracy'])}，差值 {pct(d_concat)} 个百分点，"
                   f"McNemar p={pval(p_concat)}）。")
        mech_sentence = ("说明在图像信号较弱、图像编码器冻结的条件下，把 10 个图像 token 交给 "
                         "attention 并不优于均值池化。")
        conclusion_extra = "融合机制不是主要瓶颈；图像质量、图像编码器和数据规模更值得优先投入。"

    story = []
    story.append(Paragraph("PAN 2018 性别推断：本机 Concat 与 Cross-Attention 融合对比", title))
    story.append(Paragraph("VIAS 虚实融合身份推断 · 阶段性实验报告", subtitle))
    story.append(Paragraph(
        "2026-11-27 ｜ 数据：PAN 2018 Author Profiling（EN/ES/AR）｜ "
        "本机 RTX 5090 ｜ 服务器 A800 环境已装好，GPU 队列被组内任务占满 ｜ "
        "测试集 5,100 用户，全程隔离", meta))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT))
    story.append(Spacer(1, 8))

    story.append(Paragraph("摘要", h1))
    story.append(Paragraph(
        f"本次把融合从“均值图像向量 + 文本表示拼接”推进到“文本 token 对多张图像 token 做"
        f"cross-attention”。本机 concat 在官方测试集上准确率 {f4(concat['accuracy'])}、"
        f"AUC {f4(concat['auc'])}；cross-attention 用 3 个随机种子在本机 RTX 5090 上训练，"
        f"单模型准确率均值 {f4(ss['accuracy_mean'])} ± {f4(ss['accuracy_std'])}，3-seed 概率集成"
        f"准确率 {f4(cross['accuracy'])}、AUC {f4(cross['auc'])}。{verdict}", body))
    story.append(Paragraph(
        f"作为参照，文本单模态 {f4(text['accuracy'])}、图像单模态 {f4(image['accuracy'])}、"
        f"late stack {f4(late['accuracy'])}。因此本次实验回答的是“在冻结图像编码器和相同文本"
        f"编码器的前提下，attention 融合是否比 concat 更好”，而不是“多模态是否优于单模态”。", body))

    story.append(Paragraph("1. 实验设置", h1))
    story.append(Paragraph(
        "PAN 2018 Author Profiling 提供英语、西班牙语、阿拉伯语三种语言的 Twitter 用户，"
        "每位用户最多 100 条推文和 10 张资料图。官方训练集 7,500 人按语言与性别分层划分为"
        " fit 6,000 / calib 1,500；官方测试集 5,100 人只用于最终评估，不参与训练、早停或校准。"
        "文本侧对 100 条推文做 4×512 token 分块；图像侧用冻结的 ViT 提取每张图最多 10 个 CLS"
        " token。所有模型使用同一份切分和同一套评估代码。", body))
    story.append(Paragraph(
        "本机 concat 与 cross-attention 的训练配方一致：文本编码器 XLM-R base 端到端"
        "微调，图像编码器冻结；AdamW，编码器学习率 1e-5、融合头 1e-3，OneCycle，batch 8，"
        "4 个 epoch，以 calib 准确率选早停点。concat 使用图像均值向量（主结果不带门控，"
        "门控仅作为消融）；cross-attention 使用每张图的 CLS token 和 2 层 TransformerDecoder"
        "（文本 token 作 query，图像 token 作 key/value），并加入一个可学习的空图像 token"
        "处理无图用户。", body))
    story.append(Paragraph(
        "组内服务器为 Slurm 单节点，8×NVIDIA A800 80GB PCIe，单分区 a800；环境与 3-seed 作业"
        "脚本已备好，但提交时 8 张卡已被组内其他任务占满（4+4 卡，时限 2 天）。按用户指示，"
        "cross-attention 改在本机 RTX 5090 上完成，代码、数据切分和超参与服务器作业完全一致；"
        "服务器排队作业已取消，环境保留供后续复现。", body))

    story.append(Paragraph("2. 结果", h1))
    rows = [[Paragraph("方法", head), Paragraph("Acc", head), Paragraph("Macro-F1", head),
             Paragraph("AUC", head), Paragraph("ECE", head), Paragraph("Brier", head),
             Paragraph("Log loss", head)]]
    for label, key in [("文本单模态（XLM-R）", "text"), ("图像单模态（ViT 均值）", "image"),
                       ("Late stack", "late_stack"), ("Concat（本机）", "concat"),
                       ("Cross-attn（3-seed 集成）", "crossattn_ens")]:
        rr = r[key]["test"]
        rows.append([Paragraph(label, cell), Paragraph(f4(rr["accuracy"]), cell_c),
                     Paragraph(f4(rr["macro_f1"]), cell_c), Paragraph(f4(rr["auc"]), cell_c),
                     Paragraph(f4(rr["ece"]), cell_c), Paragraph(f4(rr["brier"]), cell_c),
                     Paragraph(f4(rr["log_loss"]), cell_c)])
    story.append(make_table(rows, [58 * mm, 18 * mm, 21 * mm, 18 * mm, 17 * mm, 18 * mm, 21 * mm]))
    story.append(Paragraph(
        f"Cross-attention 单模型 3 个种子的准确率为 "
        f"{', '.join(f4(seeds[s]['test']['accuracy']) for s in seeds)}；"
        f"均值 {f4(ss['accuracy_mean'])}，标准差 {f4(ss['accuracy_std'])}。"
        f"集成把 3 个种子的校准后概率取平均，再在 calib 上拟合温度。", small))

    if (OUT / "fig_methods.png").exists():
        story.append(Spacer(1, 5))
        story.append(KeepTogether([Image(str(OUT / "fig_methods.png"), width=155 * mm,
                                         height=86 * mm),
                                   Paragraph("图 1  各方法在 PAN18 官方测试集上的准确率；"
                                             "cross-attention 误差棒为 3 个种子的标准差。", small)]))

    story.append(Paragraph("2.1 分语言结果", h2))
    lang_rows = [[Paragraph("语言", head), Paragraph("人数", head),
                  Paragraph("Concat Acc", head), Paragraph("Cross-attn Acc", head),
                  Paragraph("差值（pp）", head)]]
    for lang, name in [("en", "英语"), ("es", "西班牙语"), ("ar", "阿拉伯语")]:
        a = r["concat"]["per_lang"][lang]
        b = r["crossattn_ens"]["per_lang"][lang]
        lang_rows.append([Paragraph(name, cell), Paragraph(str(a["n"]), cell_c),
                          Paragraph(f4(a["accuracy"]), cell_c), Paragraph(f4(b["accuracy"]), cell_c),
                          Paragraph(pct(b["accuracy"] - a["accuracy"]), cell_c)])
    story.append(make_table(lang_rows, [30 * mm, 20 * mm, 32 * mm, 40 * mm, 30 * mm]))
    if (OUT / "fig_per_language.png").exists():
        story.append(Spacer(1, 4))
        story.append(KeepTogether([Image(str(OUT / "fig_per_language.png"), width=130 * mm,
                                         height=77 * mm),
                                   Paragraph("图 2  分语言测试准确率。", small)]))

    comp_rows = [[Paragraph("对比", head), Paragraph("Δ Acc（pp）", head),
                  Paragraph("95% CI", head), Paragraph("b", head), Paragraph("c", head),
                  Paragraph("McNemar p", head)]]
    for label, c in [("Cross-attn vs Concat", c_concat), ("Cross-attn vs Late stack", c_late),
                     ("Cross-attn vs Text", c_text)]:
        d = c["delta"]["delta"]
        ci = c["delta"]["ci95"]
        mc = c["mcnemar"]
        comp_rows.append([Paragraph(label, cell), Paragraph(pct(d), cell_c),
                          Paragraph(f"[{pct(ci[0])}, {pct(ci[1])}]", cell_c),
                          Paragraph(str(mc["b"]), cell_c), Paragraph(str(mc["c"]), cell_c),
                          Paragraph(pval(mc["p_value"]), cell_c)])
    story.append(KeepTogether([
        Paragraph("3. 统计检验", h1),
        Paragraph(
            "对同一批 5,100 个测试用户做配对 McNemar 检验，并在测试集上做 5,000 次 bootstrap，"
            "报告准确率差值的 95% 置信区间。b 表示“concat 对、cross-attn 错”的用户数，c 表示"
            "“concat 错、cross-attn 对”的用户数。", body),
        make_table(comp_rows, [52 * mm, 28 * mm, 32 * mm, 14 * mm, 14 * mm, 26 * mm]),
    ]))

    story.append(Paragraph("3.1 种子方差", h2))
    seed_rows = [[Paragraph("种子", head), Paragraph("Acc", head), Paragraph("Log loss", head),
                  Paragraph("温度", head)]]
    for s in seeds:
        seed_rows.append([Paragraph(s.replace("seed", ""), cell_c),
                          Paragraph(f4(seeds[s]["test"]["accuracy"]), cell_c),
                          Paragraph(f4(seeds[s]["test"]["log_loss"]), cell_c),
                          Paragraph(f"{seeds[s]['temperature']:.3f}", cell_c)])
    story.append(make_table(seed_rows, [30 * mm, 30 * mm, 35 * mm, 30 * mm]))
    if (OUT / "fig_seeds.png").exists():
        story.append(Spacer(1, 4))
        story.append(KeepTogether([Image(str(OUT / "fig_seeds.png"), width=150 * mm,
                                         height=64 * mm),
                                   Paragraph("图 3  3 个种子的测试准确率与 log loss；虚线为 3-seed 集成。",
                                             small)]))

    story.append(Paragraph("4. 讨论", h1))
    story.append(Paragraph(
        f"这组实验中最强的单模型仍是 late stack（{f4(late['accuracy'])}），concat 为 "
        f"{f4(concat['accuracy'])}。Cross-attention 的 3-seed 集成为 {f4(cross['accuracy'])}；"
        f"相对 concat 的差值为 {pct(d_concat)} 个百分点（95% CI "
        f"[{pct(ci_concat[0])}, {pct(ci_concat[1])}]，McNemar p={pval(p_concat)}）。"
        f"相对 late stack 的差值为 {pct(d_late)} 个百分点（p={pval(p_late)}）。"
        f"分语言看，集成相对 concat 的变化为英语 {pct(lang_delta['en'])}、西班牙语 "
        f"{pct(lang_delta['es'])}、阿拉伯语 {pct(lang_delta['ar'])} 个百分点，说明融合收益"
        f"与语言/数据域相关。"
        f"{mech_sentence}文本仍是主要信息源。"
        f"从概率质量看，集成 log loss {f4(cross['log_loss'])}、Brier {f4(cross['brier'])}，"
        f"优于 concat 的 {f4(concat['log_loss'])} / {f4(concat['brier'])}；但 ECE "
        f"{f4(cross['ece'])} 高于 concat 的 {f4(concat['ece'])}，分箱校准仍有改进空间。", body))
    story.append(Paragraph(
        "需要强调，本实验改变的是融合机制，而不是编码器质量：两侧使用相同的 XLM-R 和相同的"
        "冻结 ViT 特征，训练预算、切分和早停口径也一致，因此差值可以归因于融合方式。"
        "Cross-attention 的参数量略多（2 层 decoder，约 1.6M），但在 6,000 个 fit 用户上并没有"
        "转化为稳定增益。下一步若继续这条线，优先做两件事：一是解冻/微调图像编码器或使用更强的"
        "人脸模型；二是引入可学习的 query（Q-Former）并做属性-specific 的注意力，而不是直接在"
        "2048 个文本 token 上做全量 cross-attention。", body))
    story.append(Paragraph(
        "局限：只评估性别属性；只覆盖 PAN18 的英/西/阿三语；图像编码器未在 PAN18 上微调；"
        "3 个种子只能给出粗略的方差估计；本实验在单张 RTX 5090 上完成，服务器 GPU 队列被"
        "组内任务占满，未做跨硬件复核。"
        "所有结论不应外推为“真实身份”识别——本项目输出的是性别后验，不是自然人身份。", body))

    story.append(Paragraph("5. 复现与运维", h1))
    story.append(Paragraph(
        "本机 concat：python src/pan18_joint_fusion.py concat --chunks 4 --max-len 512 "
        "--epochs 4 --batch-size 8 --lr-encoder 1e-5 --lr-head 1e-3 --seed 20261127。"
        "本机图像 token：python src/pan18_image_tokens.py --data data/processed/pan18_joint "
        "--device cuda。本机 cross-attention：对 3 个种子分别运行 python "
        "src/pan18_joint_fusion.py crossattn（同超参）；服务器同配置脚本 slurm/crossattn.sbatch "
        "保留，本次因 GPU 被占未执行。汇总：python src/eval_crossattn.py。",
        body_ni))
    story.append(Paragraph(
        "服务器环境已装好：/data5/$USER/vias/envs/vias-gpu，Python 3.11 + PyTorch cu126 + "
        "transformers；代码、特征缓存和 Slurm 脚本已上传，运行目录 "
        "/data5/$USER/vias/runs/crossattn。但 HuggingFace 直连超时，XLM-R/ViT 权重未预下载；"
        "若后续在服务器复现，需先补齐模型缓存（从本机上传或配置镜像）。本次最终数字来自"
        "本机 RTX 5090；由于原始图像上传带宽约 2 MB/s，只上传冻结特征缓存，图像 token 也在"
        "本机一次性提取（约 12.5 万张图）。",
        body_ni))
    story.append(Paragraph(
        "全部重要服务器操作（登录、建目录、装环境、上传、提交作业、清理）记录在 "
        "SERVER_OPERATIONS.md；该文件不含地址、账号或口令。", small))

    story.append(Paragraph("6. 结论", h1))
    story.append(Paragraph(verdict + " " + conclusion_extra, body))

    PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="PAN18 concat vs cross-attention", author="VIAS")
    doc.build(story)
    print("wrote", PDF)

    # Markdown summary
    lines = [
        "# PAN 2018 性别推断：本机 Concat 与 Cross-Attention 对比",
        "",
        f"- 日期：2026-11-27",
        f"- 数据：PAN 2018 Author Profiling（EN/ES/AR），fit 6,000 / calib 1,500 / test 5,100",
        f"- 本机 concat：Acc **{f4(concat['accuracy'])}**，AUC {f4(concat['auc'])}",
        f"- cross-attention（本机 RTX 5090）：3 seed 均值 **{f4(ss['accuracy_mean'])} ± {f4(ss['accuracy_std'])}**，"
        f"集成 **{f4(cross['accuracy'])}**，AUC {f4(cross['auc'])}",
        f"- 参照：text {f4(text['accuracy'])}，image {f4(image['accuracy'])}，late stack {f4(late['accuracy'])}",
        f"- 概率质量：cross-attn log loss {f4(cross['log_loss'])}、Brier {f4(cross['brier'])}、ECE {f4(cross['ece'])}；"
        f"concat log loss {f4(concat['log_loss'])}、Brier {f4(concat['brier'])}、ECE {f4(concat['ece'])}",
        "",
        f"**结论**：{verdict}",
        "",
        "| 方法 | Acc | Macro-F1 | AUC | ECE | Brier | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("文本单模态", "text"), ("图像单模态", "image"), ("Late stack", "late_stack"),
                       ("Concat（本机）", "concat"), ("Cross-attn（3-seed 集成）", "crossattn_ens")]:
        rr = r[key]["test"]
        lines.append(f"| {label} | {f4(rr['accuracy'])} | {f4(rr['macro_f1'])} | {f4(rr['auc'])} | "
                     f"{f4(rr['ece'])} | {f4(rr['brier'])} | {f4(rr['log_loss'])} |")
    lines += [
        "",
        "## 显著性检验（vs concat）",
        "",
        "| 对比 | Δ Acc (pp) | 95% CI | b | c | McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, c in [("vs Concat", c_concat), ("vs Late stack", c_late), ("vs Text", c_text)]:
        d = c["delta"]["delta"]
        ci = c["delta"]["ci95"]
        mc = c["mcnemar"]
        lines.append(f"| {label} | {pct(d)} | [{pct(ci[0])}, {pct(ci[1])}] | {mc['b']} | {mc['c']} | {pval(mc['p_value'])} |")
    lines += [
        "",
        "## 种子",
        "",
        "| Seed | Acc | Log loss | Temperature |",
        "|---|---:|---:|---:|",
    ]
    for s in seeds:
        lines.append(f"| {s} | {f4(seeds[s]['test']['accuracy'])} | "
                     f"{f4(seeds[s]['test']['log_loss'])} | {seeds[s]['temperature']:.3f} |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", MD)


if __name__ == "__main__":
    main()
