# -*- coding: utf-8 -*-
"""Small helper layer over python-docx for building the walkthrough document.

python-docx has no high-level API for paragraph shading, borders or CJK font
binding, so those are done by injecting the raw OOXML elements here once and
reused by the document script.
"""
import os

from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Pt, RGBColor

CJK = "Microsoft JhengHei"
MONO = "Consolas"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x28, 0x52, 0x7A)
STEEL = RGBColor(0x3B, 0x6E, 0xA5)
RED = RGBColor(0xB5, 0x49, 0x5B)
GREY = RGBColor(0x66, 0x66, 0x66)
AMBER = RGBColor(0x9A, 0x6B, 0x10)
BLACK = RGBColor(0x11, 0x11, 0x11)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# --------------------------------------------------------------- low level
def _rpr(run):
    rpr = run._element.get_or_add_rPr()
    return rpr


def set_font(run, name=CJK, size=10.5, bold=False, color=None, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = _rpr(run)
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


# ECMA-376 child orders. OOXML validates element order inside pPr / tblPr /
# tcPr / trPr, so new children must be inserted before their successors
# instead of simply appended.
_PPR_ORDER = [
    "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr",
    "widowControl", "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs",
    "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct",
    "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
    "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
    "suppressOverlap", "jc", "textDirection", "textAlignment",
    "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr",
    "pPrChange",
]
_TBLPR_ORDER = [
    "tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
    "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd",
    "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption",
    "tblDescription", "tblPrChange",
]
_TCPR_ORDER = [
    "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd",
    "noWrap", "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark",
    "headers", "cellIns", "cellDel", "cellMerge", "tcPrChange",
]
_TRPR_ORDER = [
    "cnfStyle", "divId", "gridBefore", "gridAfter", "wBefore", "wAfter",
    "cantSplit", "trHeight", "tblHeader", "tblCellSpacing", "jc", "hidden",
    "ins", "del", "trPrChange",
]


def _insert_ordered(parent, child, order):
    """Insert child into parent at the position the schema demands."""
    tag = child.tag.split("}")[-1]
    try:
        idx = order.index(tag)
    except ValueError:
        parent.append(child)
        return child
    successors = order[idx + 1:]
    for existing in parent:
        name = existing.tag.split("}")[-1]
        if name in successors:
            existing.addprevious(child)
            return child
    parent.append(child)
    return child


def _shd(fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), fill)
    return el


def para_shade(par, fill):
    _insert_ordered(par._p.get_or_add_pPr(), _shd(fill), _PPR_ORDER)


def para_border(par, left=None, around=None, size=12, space=6):
    ppr = par._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    color = around or left
    sides = ["top", "left", "bottom", "right"] if around else ["left"]
    for side in sides:
        el = OxmlElement("w:" + side)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size if (side == "left" or around) else 4))
        el.set(qn("w:space"), str(space))
        el.set(qn("w:color"), color)
        bdr.append(el)
    _insert_ordered(ppr, bdr, _PPR_ORDER)


def cell_shade(cell, fill):
    _insert_ordered(cell._tc.get_or_add_tcPr(), _shd(fill), _TCPR_ORDER)


def set_cell_margins(table, top=60, bottom=60, left=90, right=90):
    mar = OxmlElement("w:tblCellMar")
    for name, val in (("top", top), ("left", left),
                      ("bottom", bottom), ("right", right)):
        el = OxmlElement("w:" + name)
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    _insert_ordered(table._tbl.tblPr, mar, _TBLPR_ORDER)


def repeat_header(row):
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    _insert_ordered(row._tr.get_or_add_trPr(), el, _TRPR_ORDER)


def keep_with_next(par):
    _insert_ordered(par._p.get_or_add_pPr(), OxmlElement("w:keepNext"),
                    _PPR_ORDER)


# -------------------------------------------------------------- high level
class Doc:
    def __init__(self, document):
        self.d = document

    # ---- text -----------------------------------------------------------
    def para(self, parts, size=10.5, align=None, before=3, after=6,
             color=None, line=1.5):
        """parts: str, or list of dicts {t, b(old), mono, c(olor), size}."""
        p = self.d.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = line
        if align is not None:
            p.alignment = align
        if isinstance(parts, str):
            parts = [{"t": parts}]
        for r in parts:
            run = p.add_run(r["t"])
            set_font(run, MONO if r.get("mono") else CJK,
                     r.get("size", size - (0.8 if r.get("mono") else 0)),
                     r.get("b", False), r.get("c", color))
            if r.get("mono"):
                _rpr(run).append(_shd("F1F3F5"))
        return p

    def bullet(self, parts, size=10.5):
        p = self.d.add_paragraph(style="List Bullet")
        pf = p.paragraph_format
        pf.space_before = Pt(2)
        pf.space_after = Pt(3)
        pf.line_spacing = 1.45
        pf.left_indent = Pt(22)
        if isinstance(parts, str):
            parts = [{"t": parts}]
        for r in parts:
            run = p.add_run(r["t"])
            set_font(run, MONO if r.get("mono") else CJK,
                     r.get("size", size - (0.8 if r.get("mono") else 0)),
                     r.get("b", False), r.get("c"))
            if r.get("mono"):
                _rpr(run).append(_shd("F1F3F5"))
        return p

    def h(self, text, level=1, page_break=False):
        if page_break:
            br = self.d.add_paragraph()
            br.add_run().add_break(WD_BREAK.PAGE)
        p = self.d.add_heading("", level=level)
        run = p.add_run(text)
        sizes = {1: 16, 2: 12.5, 3: 11}
        colors = {1: NAVY, 2: BLUE, 3: STEEL}
        set_font(run, CJK, sizes[level], True, colors[level])
        pf = p.paragraph_format
        pf.space_before = Pt(16 if level == 1 else 14 if level == 2 else 10)
        pf.space_after = Pt(9 if level == 1 else 6 if level == 2 else 5)
        keep_with_next(p)
        return p

    # ---- code -----------------------------------------------------------
    def code(self, lines, caption=None):
        if caption:
            p = self.d.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            set_font(p.add_run(caption), CJK, 9, True, BLUE)
            keep_with_next(p)
        n = len(lines)
        for i, ln in enumerate(lines):
            p = self.d.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(2 if i == 0 else 0)
            pf.space_after = Pt(7 if i == n - 1 else 0)
            pf.line_spacing = 1.15
            pf.left_indent = Pt(10)
            para_shade(p, "F6F8FA")
            para_border(p, left="3B6EA5", size=18, space=5)
            set_font(p.add_run(ln if ln else " "), MONO, 8.8)
            if i < n - 1:
                keep_with_next(p)

    # ---- callout --------------------------------------------------------
    def note(self, label, text):
        p = self.d.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(8)
        pf.space_after = Pt(9)
        pf.line_spacing = 1.45
        pf.left_indent = Pt(8)
        pf.right_indent = Pt(8)
        para_shade(p, "FFF8E6")
        para_border(p, around="E0A030", size=10, space=6)
        set_font(p.add_run(label + "　"), CJK, 10, True, AMBER)
        set_font(p.add_run(text), CJK, 10)
        return p

    # ---- figure ---------------------------------------------------------
    def figure(self, path, caption=None, width_in=6.1):
        from PIL import Image
        w, h = Image.open(path).size
        p = self.d.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Emu(int(width_in * 914400)))
        if caption:
            c = self.d.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.paragraph_format.space_before = Pt(1)
            c.paragraph_format.space_after = Pt(10)
            set_font(c.add_run(caption), CJK, 8.6, False, GREY)

    # ---- table ----------------------------------------------------------
    def table(self, headers, rows, widths, font=9.0):
        t = self.d.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.autofit = False
        total = float(sum(widths))
        avail = 6.3  # inches
        for i, hdr in enumerate(headers):
            cell = t.rows[0].cells[i]
            cell.width = Emu(int(widths[i] / total * avail * 914400))
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell_shade(cell, "1F3A5F")
            par = cell.paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            set_font(par.add_run(hdr), CJK, font, True, WHITE)
        repeat_header(t.rows[0])
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].width = Emu(int(widths[i] / total * avail * 914400))
                cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if ri % 2 == 1:
                    cell_shade(cells[i], "EEF2F7")
                par = cells[i].paragraphs[0]
                par.paragraph_format.space_before = Pt(1)
                par.paragraph_format.space_after = Pt(1)
                par.paragraph_format.line_spacing = 1.25
                for j, piece in enumerate(str(val).split("@@")):
                    if j:
                        par = cells[i].add_paragraph()
                        par.paragraph_format.space_before = Pt(0)
                        par.paragraph_format.space_after = Pt(0)
                        par.paragraph_format.line_spacing = 1.25
                    set_font(par.add_run(piece), CJK, font, False, BLACK)
        set_cell_margins(t)
        self.d.add_paragraph().paragraph_format.space_after = Pt(4)
        return t

    def page_break(self):
        p = self.d.add_paragraph()
        p.add_run().add_break(WD_BREAK.PAGE)
