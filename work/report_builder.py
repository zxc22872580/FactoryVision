"""Minimal ReportLab helper for building the Traditional-Chinese PDF reports."""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\msjh.ttc", "MSJH", 0),
    (r"C:\Windows\Fonts\msjhbd.ttc", "MSJH-Bold", 0),
]
BODY_FONT, BOLD_FONT = "MSJH", "MSJH-Bold"


def register_fonts():
    for path, name, idx in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
            except Exception:
                pdfmetrics.registerFont(TTFont(name, path))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily(BODY_FONT, normal=BODY_FONT, bold=BOLD_FONT,
                       italic=BODY_FONT, boldItalic=BOLD_FONT)


register_fonts()


# --- glyph safety -------------------------------------------------------
# Microsoft JhengHei has no glyph for a few symbols we would naturally reach
# for; they would silently render as tofu boxes in the PDF.  Everything that
# goes through P()/H()/bullets()/table() is rewritten to a covered glyph.
GLYPH_FIX = {
    "✓": "●",   # check mark      -> black circle
    "✔": "●",
    "✕": "○",   # cross           -> white circle
    "✗": "○",
    "✘": "○",
    "−": "-",        # minus sign      -> hyphen
    "≥": "≧",   # >=
    "≤": "≦",   # <=
}


def _font_charset():
    chars = set()
    for path, _n, _i in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            from fontTools.ttLib import TTCollection, TTFont
            if path.lower().endswith(".ttc"):
                for f in TTCollection(path).fonts:
                    chars |= set(f.getBestCmap())
            else:
                chars |= set(TTFont(path).getBestCmap())
        except Exception:                                   # noqa: BLE001
            return None
    return chars or None


_CHARSET = _font_charset()
_WARNED = set()


def fix_glyphs(text):
    """Replace glyphs the CJK font cannot draw, and warn about leftovers."""
    text = str(text)
    for a, b in GLYPH_FIX.items():
        text = text.replace(a, b)
    if _CHARSET:
        for ch in set(text):
            if ord(ch) > 0x7F and ord(ch) not in _CHARSET                     and ch not in _WARNED:
                _WARNED.add(ch)
                print("  [font warning] no glyph for U+{:04X} ({!r})".format(
                    ord(ch), ch))
    return text


_ss = getSampleStyleSheet()

ST = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontName=BOLD_FONT,
                            fontSize=22, leading=28, spaceAfter=6),
    "subtitle": ParagraphStyle("st", parent=_ss["Normal"], fontName=BODY_FONT,
                               fontSize=12, leading=17, alignment=TA_CENTER,
                               textColor=colors.HexColor("#555555")),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontName=BOLD_FONT,
                         fontSize=16, leading=21, spaceBefore=14,
                         spaceAfter=7, textColor=colors.HexColor("#1f3a5f")),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontName=BOLD_FONT,
                         fontSize=13, leading=18, spaceBefore=10,
                         spaceAfter=5, textColor=colors.HexColor("#28527a")),
    "h3": ParagraphStyle("h3", parent=_ss["Heading3"], fontName=BOLD_FONT,
                         fontSize=11.5, leading=16, spaceBefore=8,
                         spaceAfter=4, textColor=colors.HexColor("#3b6ea5")),
    "body": ParagraphStyle("b", parent=_ss["Normal"], fontName=BODY_FONT,
                           fontSize=10, leading=15.5, alignment=TA_LEFT,
                           spaceAfter=4, wordWrap="CJK"),
    "bullet": ParagraphStyle("bu", parent=_ss["Normal"], fontName=BODY_FONT,
                             fontSize=10, leading=15.5, leftIndent=12,
                             bulletIndent=3, spaceAfter=2, wordWrap="CJK"),
    "cap": ParagraphStyle("c", parent=_ss["Normal"], fontName=BODY_FONT,
                          fontSize=8.5, leading=12, alignment=TA_CENTER,
                          textColor=colors.HexColor("#666666"),
                          spaceBefore=2, spaceAfter=8),
    "code": ParagraphStyle("co", parent=_ss["Normal"], fontName="Courier",
                           fontSize=8.5, leading=11.5,
                           backColor=colors.HexColor("#f3f4f6"),
                           borderPadding=5, spaceBefore=3, spaceAfter=6),
    "note": ParagraphStyle("n", parent=_ss["Normal"], fontName=BODY_FONT,
                           fontSize=9, leading=13.5,
                           textColor=colors.HexColor("#444444"),
                           leftIndent=8, spaceAfter=5, wordWrap="CJK"),
}

TABLE_STYLE = TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
    ("FONTNAME", (0, 1), (-1, -1), BODY_FONT),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("LEADING", (0, 0), (-1, -1), 11.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c3cf")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
     [colors.white, colors.HexColor("#eef2f7")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
])


def P(text, style="body"):
    return Paragraph(fix_glyphs(text), ST[style])


def H(text, level=1):
    return Paragraph(fix_glyphs(text), ST["h{}".format(level)])


def bullets(items, style="bullet"):
    return [Paragraph("• " + fix_glyphs(t), ST[style]) for t in items]


def table(data, col_widths=None, highlight_rows=(), font_size=8.5,
          align_left_cols=(0,)):
    wrapped = [[Paragraph(fix_glyphs(c), ParagraphStyle(
        "cell", fontName=BOLD_FONT if r == 0 else BODY_FONT,
        fontSize=font_size, leading=font_size * 1.35,
        textColor=colors.white if r == 0 else colors.black,
        alignment=0 if ci in align_left_cols else 1))
        for ci, c in enumerate(row)] for r, row in enumerate(data)]
    t = Table(wrapped, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    st = TableStyle(TABLE_STYLE.getCommands())
    for r in highlight_rows:
        st.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff3cd"))
        st.add("LINEBELOW", (0, r), (-1, r), 1.0,
               colors.HexColor("#d39e00"))
    t.setStyle(st)
    return t


def figure(path, width_mm=160, caption=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = width_mm * mm
    img = Image(path, width=width, height=width * h / w)
    out = [img]
    if caption:
        out.append(Paragraph(caption, ST["cap"]))
    return out


def figure_block(path, width_mm=160, caption=None):
    return KeepTogether(figure(path, width_mm, caption))


def build(path, story, title="FactoryVision", author="TeamXX"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(18 * mm, 10 * mm, title)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm,
                               "- {} -".format(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=title, author=author)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF ->", path, "({:.2f} MB)".format(
        os.path.getsize(path) / 1024 ** 2))
    return path


def fmt(v, n=4):
    try:
        return ("{:." + str(n) + "f}").format(float(v))
    except (TypeError, ValueError):
        return str(v)


def pct(v, n=2):
    return ("{:." + str(n) + "f}%").format(float(v) * 100)


__all__ = ["P", "H", "bullets", "table", "figure", "figure_block", "build",
           "ST", "Spacer", "PageBreak", "fmt", "pct", "mm", "Paragraph"]
