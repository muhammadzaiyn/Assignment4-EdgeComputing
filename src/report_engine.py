"""Small document engine used by scripts/12_make_report.py.

The report is described once as a list of blocks and rendered twice:
  * PDF  with reportlab   (the submitted documentation)
  * DOCX with python-docx (editable copy, e.g. to paste extra screenshots)

Supported blocks
  ("title", dict)                    title page
  ("h1", text) / ("h2", text) / ("h3", text)
  ("p", text)                        paragraph; may contain <b>, <i>, <sub>, <super> tags
  ("bullets", [text, ...])
  ("eq", text)                       centred equation line (plain text / simple tags)
  ("code", text)                     monospaced block (commands, configs)
  ("table", header, rows, caption, col_widths or None)
  ("img", path, width_fraction, caption)
  ("placeholder", caption)           boxed spot for a manual screenshot
  ("pagebreak",)
"""
import re
from pathlib import Path

TAG_RE = re.compile(r"(<b>|</b>|<i>|</i>|<sub>|</sub>|<super>|</super>)")


# ============================================================================ PDF (reportlab)
def render_pdf(blocks, path: Path, header_text: str):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                    Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    ink, ink2, accent = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#1f5fae")
    st = {
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14,
                               alignment=TA_JUSTIFY, textColor=ink, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19,
                             textColor=accent, spaceBefore=10, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                             textColor=ink, spaceBefore=8, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-BoldOblique", fontSize=10.5,
                             leading=13, textColor=ink, spaceBefore=6, spaceAfter=3),
        "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5,
                              leading=11, textColor=ink2, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
        "eq": ParagraphStyle("eq", parent=ss["BodyText"], fontName="Times-Italic", fontSize=11, leading=15,
                             alignment=TA_CENTER, spaceBefore=4, spaceAfter=8),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontName="Helvetica", fontSize=8, leading=10),
        "cellh": ParagraphStyle("cellh", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=8,
                                leading=10, textColor=colors.white),
        "code": ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=8, leading=10,
                               backColor=colors.HexColor("#f2f1ee"), borderPadding=5, spaceBefore=4, spaceAfter=8),
    }
    page_w = A4[0] - 4 * cm
    story = []
    for b in blocks:
        kind = b[0]
        if kind == "title":
            d = b[1]
            story += [Spacer(1, 3.2 * cm),
                      Paragraph(d["course"], ParagraphStyle("t0", parent=st["body"], alignment=TA_CENTER, fontSize=12, textColor=ink2)),
                      Spacer(1, 0.4 * cm),
                      Paragraph(d["title"], ParagraphStyle("t1", parent=st["h1"], alignment=TA_CENTER, fontSize=22, leading=27)),
                      Spacer(1, 0.3 * cm),
                      Paragraph(d["subtitle"], ParagraphStyle("t2", parent=st["body"], alignment=TA_CENTER, fontSize=12, leading=16)),
                      Spacer(1, 1.5 * cm)]
            rows = [[Paragraph(f"<b>{k}</b>", st["body"]), Paragraph(v, st["body"])] for k, v in d["fields"]]
            t = Table(rows, colWidths=[4.5 * cm, 8 * cm])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0cfca")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story += [t, Spacer(1, 1.5 * cm),
                      Paragraph(d.get("note", ""), ParagraphStyle("t3", parent=st["cap"], fontSize=9)), PageBreak()]
        elif kind in ("h1", "h2", "h3"):
            story.append(Paragraph(b[1], st[kind]))
        elif kind == "p":
            story.append(Paragraph(b[1], st["body"]))
        elif kind == "bullets":
            items = [ListItem(Paragraph(t, st["body"]), leftIndent=12) for t in b[1]]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14, bulletFontSize=8))
        elif kind == "eq":
            story.append(Paragraph(b[1], st["eq"]))
        elif kind == "code":
            story.append(Preformatted(b[1], st["code"]))
        elif kind == "table":
            _, header, rows, caption, widths = b
            wide = len(header) > 9                       # many columns: smaller font, tighter padding
            cs = ParagraphStyle("cs", parent=st["cell"], fontSize=7, leading=8.5) if wide else st["cell"]
            chs = ParagraphStyle("chs", parent=st["cellh"], fontSize=7, leading=8.5) if wide else st["cellh"]
            data = [[Paragraph(str(h), chs) for h in header]]
            data += [[Paragraph(str(c), cs) for c in r] for r in rows]
            if widths:
                cw = [w * page_w for w in widths]
            else:
                cw = [page_w / len(header)] * len(header)
            t = Table(data, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a5b9a")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f3f0")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 if wide else 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 if wide else 6)]))
            story.append(KeepTogether([t, Paragraph(caption, st["cap"])]) if len(rows) < 25 else t)
            if len(rows) >= 25:
                story.append(Paragraph(caption, st["cap"]))
        elif kind == "img":
            _, p, frac, caption = b
            if not Path(p).exists():
                story.append(Paragraph(f"[missing figure: {Path(p).name}]", st["cap"]))
                continue
            from PIL import Image as PILImage
            w, h = PILImage.open(p).size
            width = page_w * frac
            height = width * h / w
            max_h = 21 * cm
            if height > max_h:
                width, height = width * max_h / height, max_h
            story.append(KeepTogether([Image(str(p), width=width, height=height), Paragraph(caption, st["cap"])]))
        elif kind == "placeholder":
            t = Table([[Paragraph(f"<i>Space for a screenshot: {b[1]}</i>", st["cap"])]], colWidths=[page_w * 0.9],
                      rowHeights=[3.2 * cm])
            t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9a998f")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story += [t, Spacer(1, 0.3 * cm)]
        elif kind == "pagebreak":
            story.append(PageBreak())

    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(ink2)
        canvas.drawString(2 * cm, 1.2 * cm, header_text)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title=header_text, author="")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# ============================================================================ DOCX (python-docx)
def _add_runs(par, text):
    """Add text with <b>/<i>/<sub>/<super> tags as formatted runs."""
    bold = ital = sub = sup = False
    for part in TAG_RE.split(text.replace("<br/>", "\n")):
        if part == "<b>": bold = True
        elif part == "</b>": bold = False
        elif part == "<i>": ital = True
        elif part == "</i>": ital = False
        elif part == "<sub>": sub = True
        elif part == "</sub>": sub = False
        elif part == "<super>": sup = True
        elif part == "</super>": sup = False
        elif part:
            part = re.sub(r"<[^>]+>", "", part).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            r = par.add_run(part)
            r.bold, r.italic = bold, ital
            r.font.subscript, r.font.superscript = sub, sup


def render_docx(blocks, path: Path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    # neutral document properties (python-docx would otherwise put its own name as the author)
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.title = "Assignment 4 - Knowledge Distillation and Hardware-Aware NAS"
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    style = doc.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(10.5)
    usable = sec.page_width - sec.left_margin - sec.right_margin

    def caption(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_runs(p, f"<i>{text}</i>")
        for r in p.runs:
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(0x52, 0x51, 0x4e)

    for b in blocks:
        kind = b[0]
        if kind == "title":
            d = b[1]
            for _ in range(5):
                doc.add_paragraph()
            for text, size, bold in ((d["course"], 12, False), (d["title"], 22, True), (d["subtitle"], 12, False)):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _add_runs(p, text)
                for r in p.runs:
                    r.font.size, r.bold = Pt(size), bold or r.bold
            doc.add_paragraph()
            t = doc.add_table(rows=0, cols=2)
            for k, v in d["fields"]:
                cells = t.add_row().cells
                _add_runs(cells[0].paragraphs[0], f"<b>{k}</b>")
                _add_runs(cells[1].paragraphs[0], v)
            doc.add_paragraph()
            caption(d.get("note", ""))
            doc.add_page_break()
        elif kind in ("h1", "h2", "h3"):
            h = doc.add_heading(level={"h1": 1, "h2": 2, "h3": 3}[kind])
            _add_runs(h, b[1])
        elif kind == "p":
            _add_runs(doc.add_paragraph(), b[1])
        elif kind == "bullets":
            for t in b[1]:
                _add_runs(doc.add_paragraph(style="List Bullet"), t)
        elif kind == "eq":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(p, f"<i>{b[1]}</i>")
        elif kind == "code":
            p = doc.add_paragraph()
            r = p.add_run(b[1])
            r.font.name, r.font.size = "Consolas", Pt(8)
        elif kind == "table":
            _, header, rows, cap, _w = b
            t = doc.add_table(rows=1, cols=len(header))
            t.style = "Light Grid Accent 1"
            for c, h in zip(t.rows[0].cells, header):
                _add_runs(c.paragraphs[0], f"<b>{h}</b>")
            for r in rows:
                cells = t.add_row().cells
                for c, v in zip(cells, r):
                    _add_runs(c.paragraphs[0], str(v))
                    for run in c.paragraphs[0].runs:
                        run.font.size = Pt(8.5)
            caption(cap)
        elif kind == "img":
            _, p, frac, cap = b
            if Path(p).exists():
                doc.add_picture(str(p), width=int(usable * frac))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption(cap)
        elif kind == "placeholder":
            t = doc.add_table(rows=1, cols=1)
            t.style = "Table Grid"
            t.rows[0].height = Cm(3.2)
            _add_runs(t.rows[0].cells[0].paragraphs[0], f"<i>Paste screenshot here: {b[1]}</i>")
            doc.add_paragraph()
        elif kind == "pagebreak":
            doc.add_page_break()
    doc.save(str(path))
