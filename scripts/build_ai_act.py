"""Render docs/RESPONSIBLE_AI.md as the one-page AI Act self-assessment PDF (brief §9/§10).

Run after editing the Markdown:  .venv/bin/python -m scripts.build_ai_act   (needs: pip install reportlab)
Handles the subset the page uses: #/## headings, paragraphs, '-' bullets (one nested level),
pipe tables, **bold**, *italic* and [links](url).
"""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'RESPONSIBLE_AI.md'
TARGET = ROOT / 'docs' / 'ForceCRM-AI-Act-Self-Assessment.pdf'
FONTS = {'Card': ['C:/Windows/Fonts/segoeui.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
         'CardBold': ['C:/Windows/Fonts/segoeuib.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']}


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', text)
    return re.sub(r'`([^`]+)`', r'\1', text)


def build():
    for name, candidates in FONTS.items():
        font = next((Path(c) for c in candidates if Path(c).is_file()), None)
        if font is None:
            raise SystemExit(f'No Greek-capable font found for {name}; tried: {candidates}')
        pdfmetrics.registerFont(TTFont(name, str(font)))
    pdfmetrics.registerFontFamily('Card', normal='Card', bold='CardBold', italic='Card', boldItalic='CardBold')
    green, body = colors.HexColor('#15594b'), colors.HexColor('#283c36')
    base = dict(fontName='Card', textColor=body)
    styles = {'h1': ParagraphStyle('h1', fontName='CardBold', fontSize=15, leading=18, textColor=green, spaceAfter=3),
              'h2': ParagraphStyle('h2', fontName='CardBold', fontSize=9.6, leading=11.5, textColor=green, spaceBefore=4, spaceAfter=1),
              'p': ParagraphStyle('p', fontSize=7.9, leading=9.9, spaceAfter=2.5, **base),
              'li': ParagraphStyle('li', fontSize=7.9, leading=9.6, leftIndent=9, bulletIndent=1, spaceAfter=0.8, **base),
              'li2': ParagraphStyle('li2', fontSize=7.9, leading=9.6, leftIndent=18, bulletIndent=10, spaceAfter=0.8, **base),
              'cell': ParagraphStyle('cell', fontSize=7.5, leading=9, **base),
              'th': ParagraphStyle('th', fontName='CardBold', fontSize=7.5, leading=9, textColor=colors.white)}
    flow, table = [], []

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not re.fullmatch(r'\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?', r)]
        cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
        content = [[Paragraph(inline(c), styles['th' if i == 0 else 'cell']) for c in row] for i, row in enumerate(cells)]
        width = A4[0] - 28 * mm
        t = Table(content, colWidths=[width * .32, width * .68], hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), green), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f5ef'), colors.white]),
                               ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                               ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]))
        flow.append(t)
        table.clear()

    for line in SOURCE.read_text('utf-8').splitlines():
        if line.startswith('|'):
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            continue
        if line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]), styles['h1']))
        elif line.startswith('## '):
            flow.append(Paragraph(inline(line[3:]), styles['h2']))
        elif line.startswith('  - '):
            flow.append(Paragraph(inline(line[4:]), styles['li2'], bulletText='–'))
        elif line.startswith('- '):
            flow.append(Paragraph(inline(line[2:]), styles['li'], bulletText='•'))
        else:
            flow.append(Paragraph(inline(line), styles['p']))
    flush_table()
    doc = SimpleDocTemplate(str(TARGET), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=11 * mm, bottomMargin=10 * mm, title='ForceCRM EU AI Act self-assessment')
    doc.build(flow)
    print(TARGET)


if __name__ == '__main__':
    build()
