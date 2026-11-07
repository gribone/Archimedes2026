"""Optional publishing step. Run after analysis and benchmarking with reportlab installed."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
TARGET = ROOT / 'docs' / 'ForceCRM-Model-Data-Card.pdf'


def read(path):
    return json.loads((OUT / path).read_text('utf-8-sig'))


def build():
    run = 'enriched' if (OUT / 'enriched/report.json').is_file() else 'real'
    profiles = read(f'{run}/profiles.json')
    report = read(f'{run}/report.json')
    embedding = read('embedding-comparison.json')['results']
    language = read('llm-comparison.json')['results']
    for path, name in [('segoeui.ttf', 'Card'), ('segoeuib.ttf', 'CardBold')]:
        font = Path('C:/Windows/Fonts') / path
        if not font.exists():
            font = Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans.ttf' if name == 'Card' else 'DejaVuSans-Bold.ttf')
        pdfmetrics.registerFont(TTFont(name, str(font)))
    green = colors.HexColor('#15594b')
    body = colors.HexColor('#283c36')
    styles = dict(title=ParagraphStyle('title', fontName='CardBold', fontSize=24, leading=30, textColor=green, spaceAfter=6),
                  sub=ParagraphStyle('sub', fontName='Card', fontSize=11, leading=16, textColor=body, spaceAfter=12),
                  h=ParagraphStyle('h', fontName='CardBold', fontSize=12, leading=16, textColor=green, spaceBefore=8, spaceAfter=4),
                  p=ParagraphStyle('p', fontName='Card', fontSize=9.3, leading=13.3, textColor=body, spaceAfter=5),
                  cell=ParagraphStyle('cell', fontName='Card', fontSize=8.5, leading=11.5, textColor=body),
                  th=ParagraphStyle('th', fontName='CardBold', fontSize=8.5, leading=11.5, textColor=colors.white))
    flow = []

    def add(text, style='p'):
        flow.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        content = [[Paragraph(escape(str(value)), styles['th' if i == 0 else 'cell']) for value in row] for i, row in enumerate(rows)]
        obj = Table(content, colWidths=widths, hAlign='LEFT')
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),green), ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5ef'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'), ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7)]))
        flow.extend([obj, Spacer(1,5)])

    add('Model &amp; Data Card', 'title')
    add('ForceCRM | Local profile discovery for Greek foodservice businesses', 'sub')
    add('Inputs and intended use', 'h')
    add('A sales representative uploads customer VAT numbers and revenue. ForceCRM groups customer venues, describes statistically supported traits in Greek and ranks one prospective venue per non-customer VAT. A person verifies suitability before contacting a business.')
    audit = report['data_audit']
    add(f'The supplied GEMI extract contains {audit["venues"]:,} venues, {audit["businesses"]:,} distinct VATs and 3,736 Greek activity descriptions. GPS coverage is {audit["gps_coverage"]:.1%}. All 300 portfolio VATs matched. The competition scope covers businesses established during 2020-2024 with a primary activity in division 56.')
    add('Profiles in the supplied portfolio', 'h')
    table([['Profile','Customers','Revenue share','Stability¹','Fit confidence²']] + [[p['profile_id'],p['customer_count'],f'{p["revenue_share"]:.1%}',f'{p["stability"]:.1%}',f'{p["confidence_score"]:.1%}'] for p in profiles], [80,83,107,104,125])
    add('¹ Mean membership Jaccard across 20 VAT bootstraps with matched profiles and fixed cluster count. ² Mean relative fit to the profile radius. Sales conversion requires separate outcome data.')
    add('Internal retrieval evaluation', 'h')
    metrics = report['validation']['metrics']
    table([['Rank cutoff','ForceCRM recall','NACE baseline','Difference']] + [[f'Top {m["k"]:,}',f'{m["recall"]:.1%}',f'{m["baseline_recall"]:.1%}',f'{100*m["absolute_uplift"]:+.1f} percentage points'] for m in metrics], [100,110,110,179])
    add('Seed 2026 held out all venues of 60 customers before fitting on the other 240. The baseline ranks primary-code frequencies in the training portfolio. The organizer retains its separate hidden evaluation. A single development split gives a limited estimate of future sales performance.')
    add('Grounding, oversight and data rights', 'h')
    add('Fisher tests compare trait prevalence at VAT level. Global Benjamini-Hochberg correction, at least five supporting VATs and lift of at least 1.2 limit chance findings. Clustering on the same data makes these associations exploratory. Missing GPS, self-declared activities and multiple branches can distort similarity.')
    add('GEMI competition data. Geocoding and venue nodes © OpenStreetMap contributors (ODbL), via Geofabrik. ForceCRM divided ELSTAT 2024 hotel nights by 2021 census residents. ELSTAT bears no responsibility for this transformation. Sales research only.')
    flow.append(PageBreak())
    add('Model selection and cost', 'title')
    add('Measurements on the local Intel Core i3-1215U CPU with 7.7 GiB installed RAM', 'sub')
    add('Greek activity embeddings', 'h')
    add('Multilingual E5-small encodes activity descriptions into 384 dimensions. The US-based Microsoft Research E5 team trained the model on multilingual web and translation pairs, then supervised retrieval data. The benchmark encoded 75 Greek foodservice descriptions and checked top-five retrieval against their four-digit activity families.')
    table([['Embedding model','Family hit@5','ms / description','Process RSS³']] + [[r['model_path'].split('/')[-1],f'{r["family_hit_at_5"]:.1%}',f'{r["seconds_per_description"]*1000:.1f}',f'{r["rss_mb_after_encoding"]:.0f} MiB'] for r in embedding], [160,100,125,114])
    add('³ Sampled after encoding and inclusive of Python allocations. E5-small matched E5-base on this code-family proxy and required less time and memory. Expert judgments would provide a direct test of semantic relevance.')
    add('Local Greek narrative drafting', 'h')
    add('Mistral AI in France developed Ministral 3 3B Instruct. The Q4_K_M package occupies approximately 3 GB. Full corpus provenance remains undisclosed. The LLM rewrites one templated fact sentence at a time under a JSON schema; numbers, percentages and names are checked against the source fact, and failing sentences keep their verified template. Each displayed sentence links to its source statistic and records its author. Generated descriptions carry an AI label.')
    rows = [['LLM','Valid drafts / attempted','Generation time⁴']]
    for r in language:
        attempts = r.get('profiles', [])
        valid = [a for a in attempts if a['valid_grounded_output']]
        latency = sum(a['generation_seconds'] for a in valid) / len(valid) if valid else None
        rows.append([r['model'],f'{len(valid)} / {len(attempts)}',f'{latency:.1f} s' if latency is not None else 'No valid draft'])
    table(rows, [165,166,168])
    add('⁴ Each model receives the same profiles and evidence. Times are means over valid drafts and retain the original generation latency for cached output. The sample measures schema completion and grounding under the laptop load at run time. Human Greek fluency ratings remain uncollected.')
    times = [p['llm']['seconds'] for p in profiles]
    add(f'Ministral generated all three final descriptions in {min(times):.0f}-{max(times):.0f} seconds per profile. Ollama reported 2.34 GiB allocated during CPU inference. The full run took {report["performance"]["seconds"]/60:.1f} minutes with cached activity embeddings. Temperature 0, seed 42, top-k 1 and context 2,048 are fixed. Each export records model identity and source hashes.')
    add('Licenses and reproducibility', 'h')
    add('The <a href="https://ollama.com/library/ministral-3:3b/blobs/43070e2d4e53" color="#15594b">distributed Ministral license</a> is Apache 2.0 and allows commercial use under its terms. The <a href="https://huggingface.co/intfloat/multilingual-e5-small/tree/main" color="#15594b">E5-small weight repository</a> declares MIT in its metadata and lacks a standalone license file. The <a href="https://github.com/microsoft/unilm/blob/master/LICENSE" color="#15594b">official E5 code repository</a> supplies the MIT license. The weight-specific licensing evidence is limited to the publisher’s declaration.')
    add('OpenAI Codex assisted implementation, documentation and testing. The repository includes source links, an EU AI Act self-assessment, benchmark commands and the temporary Colab workflow.')

    def footer(canvas, doc):
        canvas.setFont('CardBold',9); canvas.setFillColor(green)
        canvas.drawString(48, A4[1]-35, 'FORCECRM   /   TECHNICAL BRIEF')
        canvas.setFont('Card',8); canvas.setFillColor(body)
        canvas.drawRightString(A4[0]-48, A4[1]-35, '7 November 2026')
        canvas.setStrokeColor(colors.HexColor('#d9e2db')); canvas.line(48,40,A4[0]-48,40)
        canvas.drawString(48,26,'NKUA Hackathon   ·   Competition use')
        canvas.drawRightString(A4[0]-48,26,f'{doc.page} / 2')
    doc = SimpleDocTemplate(str(TARGET), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=19*mm, bottomMargin=18*mm, title='ForceCRM Model & Data Card')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(TARGET)


if __name__ == '__main__':
    build()
