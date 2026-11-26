# -*- coding: utf-8 -*-
"""Build the solution DOCX from the Markdown specification.

This is a lightweight Markdown -> DOCX renderer tuned for the project's
technical documents: cover page, Chinese fonts, headings, tables, code blocks,
header/footer and page numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "VIAS_Full_System_API_Solution.md"
OUT = ROOT / "docs" / "VIAS_Full_System_API_Solution.docx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5B, 0x67, 0x70)


def set_cell_shading(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_paragraph_shading(paragraph, fill: str):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def set_font(run, size=10.5, bold=False, name="宋体", latin="Times New Roman",
             color=INK):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fldChar = OxmlElement("w:fldChar")
    fldChar.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar)
    run._r.append(instrText)
    run._r.append(fldChar2)


def setup_styles(doc: Document):
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(4)

    for name, size in [("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 11.5)]:
        style = doc.styles[name]
        style.font.name = "黑体"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = INK
        style.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)


def add_header_footer(doc: Document):
    section = doc.sections[0]
    header = section.header
    p = header.paragraphs[0]
    p.text = ""
    r = p.add_run("VIAS 虚实融合身份推断系统 · API 方案说明")
    set_font(r, 8.5, color=MUTED)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(p)


def add_cover(doc: Document):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("VIAS 虚实融合身份推断系统")
    set_font(r, 24, bold=True, name="黑体")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("能力 API 技术方案")
    set_font(r, 20, bold=True, name="黑体")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("完整系统实现 · v1.0")
    set_font(r, 12, color=MUTED)

    doc.add_paragraph()
    table = doc.add_table(rows=5, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    rows = [
        ("文档版本", "v1.0"),
        ("日期", "2026-11-25"),
        ("定位", "从 pilot 脚本升级为可部署、可持久化、可审计的完全体服务"),
        ("适用范围", "多模态证据接入、可信度评估、层次贝叶斯推断、冲突处理、可解释溯源"),
        ("参考", "《课题一能力 API 技术方案 v0.4》接口设计原则"),
    ]
    for i, (k, v) in enumerate(rows):
        table.rows[i].cells[0].text = k
        table.rows[i].cells[1].text = v
        for j in range(2):
            for p in table.rows[i].cells[j].paragraphs:
                for r in p.runs:
                    set_font(r, 9.5, bold=(j == 0))
        set_cell_shading(table.rows[i].cells[0], "F2F4F6")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("\n本方案描述完全体服务，不替代项目实施方案、示范单位数据协议或伦理审批文件。")
    set_font(r, 9, color=MUTED)
    doc.add_page_break()


def strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = text.replace("**", "").replace("`", "")
    text = text.replace("→", "→")
    return text.strip()


def add_table(doc: Document, rows: list[list[str]]):
    if not rows:
        return
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        trPr = table.rows[i]._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        for j in range(cols):
            cell = table.rows[i].cells[j]
            cell.text = strip_inline(row[j]) if j < len(row) else ""
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    set_font(r, 8.8, bold=(i == 0), name="宋体")
            if i == 0:
                set_cell_shading(cell, "E8EDF2")
    doc.add_paragraph()


def add_code(doc: Document, lines: list[str]):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.left_indent = Cm(0.3)
    p.paragraph_format.keep_together = True
    set_paragraph_shading(p, "F4F5F7")
    r = p.add_run("\n".join(lines))
    r.font.name = "Consolas"
    r.font.size = Pt(8.2)
    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def build():
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()
    setup_styles(doc)
    add_header_footer(doc)
    add_cover(doc)

    i = 0
    first_h1 = True
    front_matter = True
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if front_matter:
            plain = line.replace("**", "")
            if plain.startswith(("版本：", "日期：", "定位：", "参考：", "配套：")):
                i += 1
                continue
        if line.startswith("## "):
            front_matter = False
        if line.startswith("```"):
            lang = line[3:].strip().lower()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            if lang == "mermaid":
                p = doc.add_paragraph()
                r = p.add_run("架构流程：调用方 → FastAPI 网关 → 五个能力接口 → 服务层 → SQLite/文件存储；"
                              "证据解析、可信度、融合、冲突、解释分别落库并可查询。")
                set_font(r, 9.5, color=MUTED)
            else:
                add_code(doc, code)
            continue
        if line.startswith("# "):
            if first_h1:
                first_h1 = False  # title already on cover
            else:
                doc.add_heading(strip_inline(line[2:]), level=1)
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(strip_inline(line[3:]), level=2)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(strip_inline(line[4:]), level=3)
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            continue
        if re.match(r"^\d+\.\s+", line):
            p = doc.add_paragraph(style="List Number")
            r = p.add_run(strip_inline(re.sub(r"^\d+\.\s+", "", line)))
            set_font(r, 10.5)
            i += 1
            continue
        if line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(strip_inline(line[2:]))
            set_font(r, 10.5)
            i += 1
            continue
        if line.strip() == "---":
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)
        r = p.add_run(strip_inline(line))
        set_font(r, 10.5)
        i += 1

    doc.core_properties.title = "VIAS 虚实融合身份推断系统能力 API 技术方案"
    doc.core_properties.subject = "五个能力接口的完整系统实现"
    doc.core_properties.author = ""
    doc.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    build()
