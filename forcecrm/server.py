from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
import json
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import db
from .data import load_universe, read_portfolio, demo_portfolio
from .pipeline import run_analysis, export_result
from .narrative import DEFAULT_LLM
from .features import DEFAULT_WEIGHTS, normalize_weights
from .live import ATTRIBUTION, LiveCheckError, check_vats, export_live, requests_per_minute
import os

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='ForceCRM Local Intelligence', docs_url='/api/docs')
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.Lock()
jobs = {}
live_runs = {}  # job_id -> live GEMI check state; separate from the single analysis worker


@app.middleware('http')
async def local_security(request: Request, call_next):
    if request.method == 'POST' and request.headers.get('x-forcecrm') != 'local':
        return Response('Local application header required', status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
    return response


@app.get('/api/status')
def status():
    c, _, _, audit = load_universe(ROOT)
    return {'data': audit, 'embedding_ready': (ROOT / 'models' / 'e5-small').is_dir(),
            'llm': DEFAULT_LLM, 'default_weights': DEFAULT_WEIGHTS, 'portfolios_found': [p.name for p in ROOT.glob('*.xlsx')]}


@app.get('/api/template')
def template():
    import pandas as pd
    buffer = BytesIO()
    pd.DataFrame(columns=['vat', 'revenue']).to_excel(buffer, index=False)
    return Response(buffer.getvalue(), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition': 'attachment; filename="portfolio-template.xlsx"'})


@app.get('/api/latest')
def latest():
    """Restore the latest complete local export, including CLI-created runs."""
    with lock:
        if jobs:
            return {'job_id': next(iter(jobs))}
    required = ['profiles.json', 'report.json', 'assignments.csv', 'lookalikes.csv']
    folders = sorted((ROOT / 'outputs').glob('*/report.json'), key=lambda path: path.stat().st_mtime, reverse=True)
    for report_path in folders:
        directory = report_path.parent
        if not all((directory / name).is_file() for name in required):
            continue
        try:
            profiles = json.loads((directory / 'profiles.json').read_text('utf-8'))
            report = json.loads(report_path.read_text('utf-8'))
            with closing(db.connect(ROOT)) as con:
                if not db.has_run(con, directory.name):
                    # CLI runs and runs from before the database: import the export folder once.
                    c, _, _, _ = load_universe(ROOT)
                    db.save_run_from_directory(con, directory.name, directory, c)
        except (ValueError, OSError, KeyError):
            continue
        with lock:
            if not jobs:
                jobs[directory.name] = {'status': 'complete', 'message': 'Restored local analysis',
                    'directory': directory, 'result': {'profiles': profiles, 'report': report}}
            return {'job_id': next(iter(jobs))}
    return {'job_id': None}


def worker(job_id, payload, demo, weights=None):
    try:
        c, a, _, _ = load_universe(ROOT)
        portfolio = demo_portfolio(c, a) if demo else read_portfolio(BytesIO(payload), c)
        def progress(message):
            with lock:
                jobs[job_id]['message'] = message
        result = run_analysis(ROOT, portfolio, demo=demo, progress=progress, weights=weights)
        directory = ROOT / 'outputs' / job_id
        export_result(result, directory)
        with closing(db.connect(ROOT)) as con:
            db.save_run(con, job_id, result['profiles'], result['report'], result['assignments'],
                        result['lookalikes'], c, params={'demo': demo, 'weights': weights})
        result = {'profiles': result['profiles'], 'report': result['report']}  # rows now live in SQLite
        with lock:
            jobs[job_id].update(status='complete', message='Analysis complete', result=result, directory=directory)
    except Exception as exc:
        with lock:
            jobs[job_id].update(status='error', message=f'{type(exc).__name__}: {exc}')


@app.post('/api/analyze', status_code=202)
async def analyze(file: UploadFile | None = File(None), demo: bool = Form(False), weights: str | None = Form(None)):
    payload = b''
    parsed_weights = None
    if weights:
        try:
            parsed_weights = normalize_weights(json.loads(weights))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f'Invalid weights: {exc}')
    if not demo:
        if file is None or not (file.filename or '').lower().endswith('.xlsx'):
            raise HTTPException(400, 'Upload an .xlsx customer portfolio.')
        payload = await file.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(413, 'Workbook exceeds 10 MB.')
        try:
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 50 * 1024 * 1024:
                    raise HTTPException(413, 'Expanded workbook exceeds 50 MB.')
        except zipfile.BadZipFile:
            raise HTTPException(400, 'Invalid .xlsx file.')
    with lock:
        if any(j['status'] == 'running' for j in jobs.values()):
            raise HTTPException(409, 'An analysis is already running.')
        # Retain only the latest in-memory job; exported artifacts remain on disk.
        jobs.clear()
        for run in live_runs.values():
            run['stop'].set()
        live_runs.clear()
        job_id = uuid.uuid4().hex
        jobs[job_id] = {'status': 'running', 'message': 'Validating portfolio'}
    executor.submit(worker, job_id, payload, demo, parsed_weights)
    return {'job_id': job_id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'Analysis not found.')
        response = {k: job[k] for k in ['status', 'message']}
        if job['status'] != 'complete':
            return response
        result = job['result']
    # The table pages through /api/runs/{id}/lookalikes; only the map's top points come inline.
    with closing(db.connect(ROOT)) as con:
        points = db.map_points(con, job_id)
        total = con.execute('SELECT candidate_count FROM runs WHERE run_id = ?', (job_id,)).fetchone()
    response.update(profiles=result['profiles'], report=result['report'], candidates=points,
                    candidate_total=total[0] if total else len(points))
    return response


@app.get('/api/runs')
def runs(limit: int = 20):
    with closing(db.connect(ROOT)) as con:
        return {'runs': db.list_runs(con, max(1, min(limit, 100)))}


@app.get('/api/runs/{run_id}/lookalikes')
def run_lookalikes(run_id: str, profile: str | None = None, q: str | None = None, live: str | None = None,
                   offset: int = 0, limit: int = 50):
    if offset < 0 or not 1 <= limit <= 200:
        raise HTTPException(400, 'offset must be ≥ 0 and limit between 1 and 200.')
    with closing(db.connect(ROOT)) as con:
        if not db.has_run(con, run_id):
            raise HTTPException(404, 'Run not found.')
        try:
            return db.query_lookalikes(con, run_id, profile=profile, q=q, live=live, offset=offset, limit=limit)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@app.get('/api/companies')
def companies(q: str, limit: int = 20):
    if not q.strip():
        raise HTTPException(400, 'Empty search.')
    load_universe(ROOT)  # ensures the universe is imported
    with closing(db.connect(ROOT)) as con:
        return {'companies': db.search_companies(con, q.strip(), max(1, min(limit, 100)))}


@app.get('/api/companies/{company_id}')
def company(company_id: str):
    load_universe(ROOT)
    with closing(db.connect(ROOT)) as con:
        detail = db.company_detail(con, company_id)
    if detail is None:
        raise HTTPException(404, 'Company not found.')
    return detail


@app.get('/api/jobs/{job_id}/export/{filename}')
def download(job_id: str, filename: str):
    if filename not in ['profiles.json', 'assignments.csv', 'lookalikes.csv', 'enrichment.csv', 'report.json', 'live-status.csv', 'all.zip']:
        raise HTTPException(404)
    with lock:
        job = jobs.get(job_id)
        if not job or job['status'] != 'complete':
            raise HTTPException(404, 'Completed analysis not found.')
        directory = job['directory']
    if filename == 'all.zip':
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in directory.iterdir():
                if file.suffix in ['.json', '.csv']:
                    archive.write(file, file.name)
        return Response(buffer.getvalue(), media_type='application/zip', headers={'Content-Disposition': 'attachment; filename="forcecrm-results.zip"'})
    if not (directory / filename).is_file():
        raise HTTPException(404, 'Export unavailable for this run')
    return FileResponse(directory / filename, filename=filename)


def live_view(run):
    return {k: run[k] for k in ['status', 'message', 'checked', 'total', 'results', 'requests_per_minute', 'attribution']}


def live_worker(job_id, run, vats, api_key, candidates, directory):
    def on_result(vat, result):
        with lock:
            run['results'][vat] = result
            run['checked'] = len(run['results'])
            run['message'] = f"Έλεγχος ΓΕΜΗ: {run['checked']}/{run['total']}"
    try:
        check_vats(vats, api_key, ROOT, on_result, should_stop=run['stop'].is_set)
        with lock:
            stopped = run['stop'].is_set()
            run['status'] = 'stopped' if stopped else 'complete'
            run['message'] = f"Ελέγχθηκαν {run['checked']}/{run['total']} επιχειρήσεις στο ΓΕΜΗ"
            results = dict(run['results'])
        if results and not stopped:
            export_live(results, candidates, directory)
    except LiveCheckError as exc:
        with lock:
            run.update(status='error', message=str(exc))
    except Exception as exc:  # never let the thread die silently
        with lock:
            run.update(status='error', message=f'{type(exc).__name__}: {exc}')


@app.post('/api/jobs/{job_id}/live-check', status_code=202)
def start_live_check(job_id: str, api_key: str = Form(''), limit: int = Form(50)):
    if not 1 <= limit <= 200:
        raise HTTPException(400, 'Ο αριθμός επιχειρήσεων πρέπει να είναι από 1 έως 200.')
    key = api_key.strip() or os.environ.get('GEMI_API_KEY', '').strip()
    if not key:
        raise HTTPException(400, 'Απαιτείται προσωπικό api_key του ΓΕΜΗ (ή η μεταβλητή GEMI_API_KEY).')
    with lock:
        job = jobs.get(job_id)
        if not job or job['status'] != 'complete':
            raise HTTPException(404, 'Completed analysis not found.')
        existing = live_runs.get(job_id)
        if existing and existing['status'] == 'running':
            raise HTTPException(409, 'Ο έλεγχος ΓΕΜΗ εκτελείται ήδη.')
        with closing(db.connect(ROOT)) as con:
            candidates = db.query_lookalikes(con, job_id, limit=limit)['rows']
        vats = list(dict.fromkeys(c['vat'] for c in candidates))
        run = {'status': 'running', 'message': f'Έλεγχος ΓΕΜΗ: 0/{len(vats)}', 'checked': 0, 'total': len(vats),
               'results': {}, 'stop': threading.Event(), 'requests_per_minute': requests_per_minute(),
               'attribution': ATTRIBUTION}
        live_runs[job_id] = run
        directory = job['directory']
    threading.Thread(target=live_worker, args=(job_id, run, vats, key, candidates, directory), daemon=True).start()
    with lock:
        return live_view(run)


@app.get('/api/jobs/{job_id}/live')
def live_status(job_id: str):
    with lock:
        run = live_runs.get(job_id)
        if run is None:
            raise HTTPException(404, 'No live check for this analysis.')
        return live_view(run)


@app.post('/api/jobs/{job_id}/live-check/stop')
def stop_live_check(job_id: str):
    with lock:
        run = live_runs.get(job_id)
        if run is None:
            raise HTTPException(404, 'No live check for this analysis.')
        run['stop'].set()
        return live_view(run)


app.mount('/', StaticFiles(directory=ROOT / 'web', html=True), name='web')
