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
    # Greek-capable fonts on Windows, macOS and Linux, in order of preference.
    fonts = {'Card': ['C:/Windows/Fonts/segoeui.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
             'CardBold': ['C:/Windows/Fonts/segoeuib.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                          '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']}
    for name, candidates in fonts.items():
        font = next((Path(c) for c in candidates if Path(c).is_file()), None)
        if font is None:
            raise SystemExit(f'No Greek-capable font found for {name}; tried: {candidates}')
        pdfmetrics.registerFont(TTFont(name, str(font)))
    machine = read('llm-comparison.json').get('hardware') or {}
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
        content = [[value if isinstance(value, Paragraph) else Paragraph(escape(str(value)), styles['th' if i == 0 else 'cell'])
                    for value in row] for i, row in enumerate(rows)]
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
    add('GEMI competition data. Geocoding and venue nodes © OpenStreetMap contributors (ODbL), via Geofabrik. ForceCRM divided ELSTAT 2024 hotel nights by 2021 census residents. ELSTAT bears no responsibility for this transformation. Map outline: Natural Earth (public domain). Optional live registry-status check: ΓΕΜΗ Open Data API (ODC-BY 1.0), opt-in, sends prospect VATs only. Sales research only.')
    flow.append(PageBreak())
    add('Model selection and cost', 'title')
    add(f'Measurements on {escape(machine.get("cpu", "the local CPU"))} with {machine.get("ram_gib", 0):.0f} GiB RAM, CPU/GPU as scheduled by Ollama', 'sub')
    add('Greek activity embeddings', 'h')
    add('Multilingual E5-small (Microsoft Research, US; MIT) encodes activity descriptions into 384 dimensions. It is compared with E5-base, MiniLM and two models from the brief: granite-embedding multilingual r2 311M (IBM, US; Apache-2.0) and BGE-M3 (BAAI, China; MIT). Licences are declared in each repository\'s model-card metadata (no separate LICENSE files). Measured on challenge data: top-five retrieval of 75 Greek descriptions against their 4-digit activity family (proxy), and the real task, 5-split holdout recall with only the embedding swapped.')
    task = {Path(r['model_path']).name: r for r in (read('embedding-recall.json')['results'] if (OUT / 'embedding-recall.json').is_file() else [])}
    def recall(name):
        r = task.get(name)
        return f'{r["recall_at_500_mean"]:.1%} / {r["recall_at_2000_mean"]:.1%}' if r and r.get('status') == 'measured' else '-'
    table([['Embedding model','Family hit@5','Recall@500 / @2000³','ms / description','Size on disk']] +
          [[Path(r['model_path']).name, f'{r["family_hit_at_5"]:.1%}', recall(Path(r['model_path']).name),
            f'{r["seconds_per_description"]*1000:.1f}', f'{r["model_bytes_on_disk"]/1e9:.2f} GB'] for r in embedding if r.get('status') == 'measured'],
          [118,80,125,90,86])
    add('³ Mean of the same 5 seeded VAT splits as the ranking audit, shipped pipeline. With 60 held-out VATs per split, differences of 2-3 points are noise. BGE-M3 and granite lead on the proxy but none beats E5-small on the task, at 4-10 times the cost, so the smallest model ships.')
    add('Local Greek narrative drafting', 'h')
    add('Mistral AI in France developed Ministral 3 3B Instruct. The Q4_K_M package occupies approximately 3 GB. Full corpus provenance remains undisclosed. The LLM rewrites one templated fact sentence at a time under a JSON schema; numbers, percentages, names and wording are checked against the source fact, and a failing sentence keeps its verified template. It also picks profile titles from data-backed phrases. Computed sentences state activity mix, quality, stability and confidence. All AI text is labelled.')
    # Arial on macOS lacks U+2074, so footnote 4 uses real superscript markup.
    rows = [['LLM', Paragraph('LLM-written sentences<super>4</super>', styles['th']), 'Time per profile', 'Ollama memory']]
    for r in language:
        attempts = [a for a in r.get('profiles', []) if a['valid_grounded_output']]
        if r.get('status') != 'measured' or not attempts:
            rows.append([r['model'], 'Unavailable', '-', '-'])
            continue
        written = sum(a['llm_sentences'] for a in attempts)
        total = written + sum(a['template_sentences'] for a in attempts)
        latency = sum(a['generation_seconds'] for a in attempts) / len(attempts)
        memory = r.get('ollama_resident_bytes')
        rows.append([r['model'], f'{written} / {total} ({written / total:.0%})', f'{latency:.1f} s',
                     f'{memory / 1024**3:.2f} GiB' if memory else '-'])
    table(rows, [140,150,104,105])
    add('<super>4</super> Sentences passing every check within two attempts; same profiles, prompt and checks for all models. Measures grounded rewriting, not fluency; human fluency ratings remain uncollected.')
    times = [p['llm']['seconds'] for p in profiles]
    share = sum(p['llm']['llm_sentences'] for p in profiles) / sum(p['llm']['llm_sentences'] + p['llm']['template_sentences'] for p in profiles)
    add(f'In the submitted run Ministral wrote {share:.0%} of fact sentences and took {min(times):.0f}-{max(times):.0f} seconds per profile. The pass rate varies with the local Ollama state; failing sentences keep their verified template. The full run took {report["performance"]["seconds"]/60:.1f} minutes with cached activity embeddings. Temperature 0, seed 42, top-k 1 and context 2,048 are fixed. Each export records model identity, prompt hash, rejected drafts and source hashes.')
    add('Licenses and reproducibility', 'h')
    add('The <a href="https://ollama.com/library/ministral-3:3b/blobs/43070e2d4e53" color="#15594b">distributed Ministral license</a> is Apache 2.0 and allows commercial use under its terms. The <a href="https://huggingface.co/intfloat/multilingual-e5-small/tree/main" color="#15594b">E5-small weight repository</a> declares MIT in its metadata and lacks a standalone license file. The <a href="https://github.com/microsoft/unilm/blob/master/LICENSE" color="#15594b">official E5 code repository</a> supplies the MIT license. The weight-specific licensing evidence is limited to the publisher’s declaration.')
    add('AI coding assistants (development only): OpenAI Codex and Claude Code (Anthropic) assisted implementation, review, testing and documentation, incl. the grounded narration, SQLite store, stage pipeline and demo login. The repository holds sources, the AI Act self-assessment and benchmark commands.')

    def footer(canvas, doc):
        canvas.setFont('CardBold',9); canvas.setFillColor(green)
        canvas.drawString(48, A4[1]-35, 'FORCECRM   /   MODEL & DATA CARD')
        canvas.setFont('Card',8); canvas.setFillColor(body)
        canvas.drawRightString(A4[0]-48, A4[1]-35, '8 November 2026')
        canvas.setStrokeColor(colors.HexColor('#d9e2db')); canvas.line(48,40,A4[0]-48,40)
        canvas.drawString(48,26,'NKUA Hackathon   ·   Competition use')
        canvas.drawRightString(A4[0]-48,26,f'{doc.page} / 2')
    doc = SimpleDocTemplate(str(TARGET), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=19*mm, bottomMargin=18*mm, title='ForceCRM Model & Data Card')
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(TARGET)


if __name__ == '__main__':
    build()
