# -*- coding: utf-8 -*-
"""Academic-style pilot report v3: centered title, abstract, keywords,
first-line indent, 1.5 line spacing, three-line tables with captions,
figure captions, centered page numbers, repo link on page 1."""
import os

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

# ── Palette ──
TEXT = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#5B6770")
ACCENT = colors.HexColor("#2F4F6F")

# ── Fonts ──
pdfmetrics.registerFont(TTFont("SimSun", "C:/Windows/Fonts/simsun.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
registerFontFamily("SimSun", normal="SimSun", bold="SimHei")
registerFontFamily("SimHei", normal="SimHei", bold="SimHei")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "VIAS_Pilot_Study_Report_2026-11-25.pdf")
FIG_PIPE = os.path.join(ROOT, "outputs", "figures", "pipeline_diagram.png")
FIG_SAMPLE = os.path.join(ROOT, "outputs", "figures", "sample_composition.png")
FIG_METHOD = os.path.join(ROOT, "outputs", "figures", "method_comparison.png")
FIG_CONF = os.path.join(ROOT, "outputs", "figures", "confusions.png")

# ── Styles ──
title = ParagraphStyle("Title", fontName="SimHei", fontSize=18, leading=24,
                       textColor=TEXT, alignment=TA_CENTER, spaceAfter=6)
author = ParagraphStyle("Author", fontName="SimSun", fontSize=10.5, leading=16,
                        textColor=TEXT, alignment=TA_CENTER)
repo = ParagraphStyle("Repo", fontName="SimSun", fontSize=9, leading=13,
                      textColor=TEXT, alignment=TA_CENTER)
abstract = ParagraphStyle("Abstract", fontName="SimSun", fontSize=9.5, leading=16,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
keywords = ParagraphStyle("Keywords", fontName="SimSun", fontSize=9.5, leading=16,
                          textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK")
body = ParagraphStyle("Body", fontName="SimSun", fontSize=10.5, leading=18,
                      textColor=TEXT, alignment=TA_JUSTIFY, wordWrap="CJK",
                      firstLineIndent=21)
body_bullet = ParagraphStyle("BodyBullet", parent=body, leftIndent=21,
                             bulletIndent=10, firstLineIndent=0, spaceAfter=3,
                             bulletFontName="SimSun")
h2 = ParagraphStyle("H2", fontName="SimHei", fontSize=12, leading=16,
                    textColor=TEXT, spaceBefore=12, spaceAfter=6,
                    alignment=TA_LEFT)
h3 = ParagraphStyle("H3", fontName="SimHei", fontSize=10.5, leading=15,
                    textColor=TEXT, spaceBefore=8, spaceAfter=3,
                    alignment=TA_LEFT)
table_header = ParagraphStyle("TableHeader", fontName="SimHei", fontSize=9,
                              leading=11.5, textColor=TEXT,
                              alignment=TA_CENTER, wordWrap="CJK")
table_cell = ParagraphStyle("TableCell", fontName="SimSun", fontSize=9,
                            leading=11.5, textColor=TEXT,
                            alignment=TA_LEFT, wordWrap="CJK")
table_cell_center = ParagraphStyle("TableCellCenter", parent=table_cell,
                                   alignment=TA_CENTER)
table_caption = ParagraphStyle("TableCaption", fontName="SimSun", fontSize=9,
                               leading=12, textColor=TEXT,
                               alignment=TA_CENTER, spaceBefore=4, spaceAfter=2)
caption = ParagraphStyle("Caption", fontName="SimSun", fontSize=9, leading=12,
                         textColor=TEXT, alignment=TA_CENTER,
                         spaceBefore=3, spaceAfter=6)
note = ParagraphStyle("Note", fontName="SimSun", fontSize=9, leading=13,
                      textColor=MUTED, alignment=TA_LEFT, wordWrap="CJK")


def heading(text):
    return [Paragraph(text, h2)]


def bullet(lead, text):
    return Paragraph(f"<b>{lead}</b> {text}", body_bullet, bulletText="•")


def _natural_widths(header, rows, pad=14):
    widths = []
    for j in range(len(header)):
        nat = pdfmetrics.stringWidth(str(header[j]), "SimHei", 9)
        for row in rows:
            nat = max(nat, pdfmetrics.stringWidth(str(row[j]), "SimSun", 9))
        widths.append(nat + pad)
    total = sum(widths)
    if total > W:
        widths = [x * W / total for x in widths]
    return widths


def three_line_table(header, rows, numeric_cols=(), widths=None):
    data = [[Paragraph(str(h), table_header) for h in header]]
    for row in rows:
        cells = []
        for j, v in enumerate(row):
            style = table_cell_center if j in numeric_cols else table_cell
            cells.append(Paragraph(str(v), style))
        data.append(cells)
    col_widths = list(widths) if widths else _natural_widths(header, rows)
    total = sum(col_widths)
    if total > W:
        col_widths = [x * W / total for x in col_widths]
    t = Table(data, colWidths=col_widths,
              hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, TEXT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEXT),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, TEXT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def table_block(caption_text, table):
    return [KeepTogether([Paragraph(caption_text, table_caption), table,
                          Spacer(1, 7)])]


def figure(path, width, max_height=None, caption_text=None):
    pil = PILImage.open(path)
    ow, oh = pil.size
    h = width * oh / ow
    if max_height and h > max_height:
        h = max_height
        width = h * ow / oh
    elems = [Image(path, width=width, height=h)]
    if caption_text:
        elems += [Spacer(1, 2), Paragraph(caption_text, caption)]
    return [KeepTogether(elems)]


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("SimSun", 9)
    canvas.setFillColor(TEXT)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


def main():
    global W
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
        title="VIAS 虚实融合身份推断 Pilot Study",
        author="", creator="VIAS",
        subject="研究对象、系统搭建、评估场景、样本构造与结论",
        keywords="虚实融合身份推断, 多模态证据, 可信度, 贝叶斯融合, 冲突处理",
    )
    W = doc.width
    story = []

    # ── Title block ──
    story.append(Paragraph("虚实融合身份推断 Pilot Study", title))
    story.append(Paragraph("2026 年 11 月", author))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        '代码仓库：<link href="https://github.com/Koipoppy/VIAS" color="#1A1A1A">'
        '<u>https://github.com/Koipoppy/VIAS</u></link>', repo))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.6, color=TEXT,
                            spaceBefore=0, spaceAfter=8))

    # ── Abstract & keywords ──
    story.append(Paragraph(
        "<b>摘要：</b>此处研究的对象是“虚实融合身份推断”：把文本、人脸、自述等多模态证据统一为证据单元，"
        "经六维可信度加权、贝叶斯融合和冲突处理，输出目标主体的性别/年龄段后验。"
        "Pilot 用 SMP 2016 与 AFAD-Lite 构造50个样本，验证这条链路能否跑通", abstract))
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEXT,
                            spaceBefore=6, spaceAfter=8))

    # ── 1 研究对象 ──
    story.extend(heading("1　研究对象"))
    story.append(Paragraph(
        "研究对象不是单一分类器，而是一条从多模态证据到属性后验的推断链路。最终目标是真实身份定位；"
        "pilot 先用性别与年龄段做代理变量，因为这两类标签在公开数据上可得、可评、可解释。", body))
    story.append(Spacer(1, 3))
    story.append(bullet("问题一：", "多源证据能否统一表示并加权？"))
    story.append(bullet("问题二：", "可信度加权是否优于等权融合？"))
    story.append(bullet("问题三：", "冲突处理能否在矛盾证据下保持稳健？"))
    story.append(Spacer(1, 3))
    story.extend(table_block("表 1　研究范围与边界", three_line_table(
        ("维度", "设置"),
        [
            ("输入", "中文文本、亚洲人脸、自述"),
            ("输出", "性别、年龄段后验"),
            ("方法", "证据单元、六维可信度、加权贝叶斯、冲突、可解释"),
            ("不覆盖", "真实身份、音频/行为模型、候选身份"),
        ],
    )))

    # ── 2 系统搭建 ──
    story.append(CondPageBreak(100))
    story.extend(heading("2　系统搭建"))
    story.append(Paragraph(
        "系统分五段：多模态解析、统一证据单元、六维可信度、加权贝叶斯、冲突处理与可解释。"
        "每段可运行，证据以统一 JSON 流转；技术方案与 Schema 见附录 A。", body))
    story.append(Spacer(1, 3))
    story.extend(figure(FIG_PIPE, W, max_height=95,
                        caption_text="图 1　五段式证据推断流水线"))
    story.extend(table_block("表 2　系统模块与实现", three_line_table(
        ("模块", "实现"),
        [
            ("解析", "文本 TF-IDF + LR；人脸 ViT；自述正则"),
            ("证据单元", "22 字段；JSON Schema v1.0；442 条校验"),
            ("可信度", "六维；冲突后更新 consistency；权重"),
            ("融合", "加权意见池；证据 → 假设 → 目标"),
            ("冲突/解释", "D-S，阈值 0.50；LOO Top-3；证据链"),
        ],
    )))
    story.append(Spacer(1, 2))

    # ── 3 评估场景 ──
    story.append(CondPageBreak(100))
    story.extend(heading("3　评估场景"))
    story.append(Paragraph(
        "评估分两个场景：clean 用于测基线，conflict 用于测稳健性。重点不是刷 SOTA，"
        "而是看每一段是否有增量、冲突处理是否值得。", body))
    story.extend(table_block("表 3　评估场景", three_line_table(
        ("场景", "设置与目的"),
        [
            ("Clean", "50 例，无注入；测性别/年龄基线"),
            ("Conflict", "50 例中 10 例注入 1-2 条矛盾证据；测冲突检出/FPR/恢复"),
        ],
    )))
    story.extend(table_block("表 4　评估指标", three_line_table(
        ("指标", "定义"),
        [
            ("性别 Acc/F1", "二分类"),
            ("年龄 Acc/F1", "4 段（18-24 / 25-29 / 30-34 / 35-40）"),
            ("年龄 MAE", "以分段中点计算"),
            ("冲突检出", "注入案例中最大冲突 > 0.50"),
            ("干净误报", "干净案例被标记冲突"),
        ],
    )))
    story.append(Paragraph(
        "对比五种方法：文本单模态、人脸单模态、等权融合、可信度加权、可信度 + 冲突处理。", body))

    # ── 4 样本构造 ──
    story.append(CondPageBreak(100))
    story.extend(heading("4　样本构造"))
    story.append(Paragraph(
        "样本从两个公开数据集构造，按年龄 × 性别分层，避免模型只学多数类。", body))
    story.append(Paragraph(
        "公开数据已摸底（27 个数据集），示范单位样本待补充。", body))
    story.extend(table_block("表 5　样本构造", three_line_table(
        ("项目", "设置"),
        [
            ("文本", "SMP 2016；2,874 个 18-40 岁用户；2,824 训练 / 50 pilot"),
            ("人脸", "AFAD-Lite；22/24/28/32/36 × 男/女 × 5；60 采样，50 使用"),
            ("配对", "同年龄/同性别；合成，不是同一人"),
            ("30 例案件包", "每格 3 例；20 干净 + 10 注入；文本脱敏"),
        ],
    )))
    story.append(Spacer(1, 2))
    story.extend(figure(FIG_SAMPLE, W, max_height=175,
                        caption_text="图 2　样本构成：按年龄 × 性别分层"))
    story.append(Paragraph(
        "分层保证每个年龄/性别单元都有样本。配对保证文本与人脸属性一致，但无法验证“同一人”。"
        "注入冲突用于测试冲突处理；限制是合成配对，不是真实案件。", body))

    # ── 5 结论 ──
    story.append(CondPageBreak(120))
    story.extend(heading("5　结论"))
    story.extend(table_block("表 6　方法对比", three_line_table(
        ["方法", "性别 Acc", "年龄 Acc", "年龄 MAE"],
        [
            ("文本单模态", "0.780", "0.360", "4.62"),
            ("人脸单模态", "0.960", "0.460", "6.36"),
            ("融合（可信度）", "0.960", "0.500", "3.94"),
            ("融合 + 冲突处理", "0.920", "0.500", "3.98"),
        ],
        numeric_cols=(1, 2, 3),
    )))
    story.append(Spacer(1, 4))
    story.extend(figure(FIG_METHOD, W * 0.94, max_height=155,
                        caption_text="图 3　方法对比：clean 50 例与 conflict 10 例"))
    story.append(Spacer(1, 2))
    story.extend(figure(FIG_CONF, W * 0.86, max_height=175,
                        caption_text="图 4　融合方法的性别 / 年龄段混淆矩阵"))
    story.append(Spacer(1, 3))
    story.append(bullet("问题一（统一表示/加权）：能。", "22 字段证据单元与 Schema v1.0 已跑通 589 条证据，性别/年龄后验可复现。"))
    story.append(bullet("问题二（可信度 vs 等权）：未显示优势。", "可信度加权与等权结果相同（性别 0.96、年龄 0.50、MAE 3.94）；性别主要靠人脸，文本 0.78 未带来增量。"))
    story.append(bullet("问题三（冲突稳健性）：部分有效。", "检出 9/10、注入年龄 0.30 升至 0.40，但干净误报 7/50、干净性别 0.96 降至 0.92；有效但有副作用。"))
    story.append(bullet("年龄瓶颈：", "融合年龄 0.50、MAE 3.94；人脸年龄偏粗，文本年龄 0.36。"))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "边界：属性代理，不是真实身份；合成配对、50 例；没有 LLM 解析、音频/行为模型、候选身份。"
        "下一步：年龄模型微调与校准、来源可靠度学习、属性-specific 冲突阈值、真实 30 例。", body))
    story.append(Paragraph(
        "数据与代码：SMP CUP 2016、AFAD-Lite；代码仓库见首页链接。", note))

    # ── Appendix A ──
    story.append(PageBreak())
    story.extend(heading("附录 A　技术方案与数据规范"))
    story.append(Paragraph("A.1　技术方案与数据规范", h3))
    story.append(Paragraph(
        "技术方案覆盖标签体系、证据类型、六维可信度、层次结构、融合/冲突与可解释；"
        "数据规范覆盖来源许可、样本配对、PII 脱敏、质量检查、存储与访问。", body))
    story.extend(table_block("表 A1　技术方案与数据规范要点", three_line_table(
        ("类别", "规范内容"),
        [
            ("标签", "性别 male/female；年龄 18-24/25-29/30-34/35-40；unknown 暂不处理"),
            ("证据", "text_model、face_model、self_report、synthetic_conflict；audio/behavior 预留"),
            ("可信度", "来源 .20、权威 .15、客观 .20、时效 .15、完整 .15、一致 .15"),
            ("层次", "证据 → 性别/年龄假设 → (性别, 年龄段) 目标"),
            ("融合/冲突", "加权意见池；D-S C=K×min(max p,max q)，τ=0.50，低权 ×0.25"),
            ("可解释", "LOO 贡献、Top-3、证据链图"),
            ("实现环境", "Python 3.12、PyTorch/Transformers；PostgreSQL 14 + MinIO + Redis compose 已配置"),
            ("数据治理", "公开数据仅研究、不二次分发；文本脱敏；合成配对非同一人；原始数据不进公共仓库"),
        ],
    )))

    story.append(Paragraph("A.2　统一证据单元 Schema v1.0", h3))
    story.append(Paragraph(
        "v1.0 已冻结为 22 个字段；Pydantic 与 JSON Schema 双重校验，"
        "枚举、范围、概率和、额外字段均受约束。", body))
    story.extend(table_block("表 A2　Schema v1.0 字段分组", three_line_table(
        ("组", "字段", "约束/说明"),
        [
            ("标识", "schema_version, evidence_id, case_id",
             "版本 1.0；ID 案件内唯一；仅字母/数字/_.:-"),
            ("来源", "modality, source_type, source_name",
             "枚举；source_type 7 类；名称非空"),
            ("内容", "content, raw_ref, time_range, language",
             "按 modality 约定必填键；time_range 仅 start/end"),
            ("概率", "gender_probs, age_probs",
             "键精确；值 0–1；和为 1"),
            ("可信度", "credibility",
             "六维 + score + weight；均 0–1"),
            ("关系/冲突", "supports, contradicts, conflict_flag, conflict_degree",
             "冲突度 0–1；flag 默认 false"),
            ("合规/权重", "injected, synthetic, pii_redacted, weight_multiplier",
             "标记真实性质；权重 (0,1]"),
            ("说明", "note", "处理备注"),
        ],
        widths=(62, 190, W - 252),
    )))

    story.append(Paragraph("A.3　定型与校验", h3))
    story.append(Paragraph(
        "定型结果：JSON Schema 22 properties、additionalProperties=false；示例通过；"
        "442 条输出证据与 147 条案件证据（共 589 条）迁移后全部通过；音频/行为仍为 reserved。"
        "v0.1→v1.0 的主要变化是枚举化 source_type、增加范围/概率和校验、禁止额外字段、"
        "默认版本 1.0，并移除持久化 effective_weight。团队评审待确认。", body))
    story.append(Paragraph(
        "完整规范：reports/technical_scheme_and_data_spec_v1.0.md；"
        "机器可读：schema/evidence_unit.schema.json。", note))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
