"""End-to-end HTTP test of a running ForceCRM server: the same calls the browser makes.

Start the app (python run.py, or uvicorn forcecrm.server:app --port 8000), then:
    .venv/bin/python -m scripts.e2e_server --url http://127.0.0.1:8000 --portfolio ForceCRM_Hackathon_Pelatologio.xlsx
Modes: full (upload, analysis, every export, security checks), weights (same with custom weights),
restore (run after restarting the server: the latest run must be restored and downloadable).
Note: weights mode leaves a custom-weights run as the "latest" one; delete its outputs/<id> folder before a demo.
"""
import argparse, io, json, sys, time, zipfile
import httpx

_p = argparse.ArgumentParser()
_p.add_argument('--url', default='http://127.0.0.1:8000')
_p.add_argument('--portfolio', required=True)
_p.add_argument('--mode', choices=['full', 'weights', 'restore'], default='full')
_a = _p.parse_args()
BASE, XLSX, MODE = _a.url, _a.portfolio, _a.mode
H = {'X-ForceCRM': 'local'}
c = httpx.Client(base_url=BASE, timeout=60, trust_env=False)
ok = lambda cond, msg: print(('PASS ' if cond else 'FAIL ') + msg) or cond
results = []

for _ in range(120):
    try:
        if c.get('/api/status').status_code in (200, 401): break
    except httpx.HTTPError: pass
    time.sleep(1)
# Demo login (on unless the server runs with FORCECRM_AUTH=0): unauthenticated API calls get 401,
# a new username creates its account, and the session cookie carries the rest of the run.
if c.get('/api/status').status_code == 401:
    results.append(ok(c.get('/', follow_redirects=False).status_code == 303, 'app page redirects to login (303)'))
    lg = c.post('/api/auth/login', headers=H, data={'username': 'e2e-test', 'password': 'e2e-test'})
    results.append(ok(lg.status_code == 200, f"login as e2e-test (created={lg.json().get('created')})"))
    wrong = c.post('/api/auth/login', headers=H, data={'username': 'e2e-test', 'password': 'wrong'}).status_code
    results.append(ok(wrong == 401, 'wrong password for an existing user rejected (401)'))
st = c.get('/api/status').json()
results.append(ok(st['embedding_ready'], f"status: embeddings ready, llm={st['llm']}, venues={st['data']['venues']}"))

def exports(job, expect_live=False):
    for name in ['profiles.json', 'assignments.csv', 'lookalikes.csv', 'enrichment.csv', 'report.json']:
        r = c.get(f'/api/jobs/{job}/export/{name}')
        results.append(ok(r.status_code == 200 and len(r.content) > 100, f'export {name} ({len(r.content):,} bytes)'))
    r = c.get(f'/api/jobs/{job}/export/all.zip')
    names = sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) if r.status_code == 200 else []
    results.append(ok({'profiles.json', 'assignments.csv', 'lookalikes.csv', 'report.json'} <= set(names), f'all.zip contains {names}'))
    results.append(ok(c.get(f'/api/jobs/{job}/export/../../run.py').status_code == 404, 'path traversal rejected'))

if MODE == 'restore':
    job = c.get('/api/latest').json()['job_id']
    results.append(ok(bool(job), f'restored latest run after restart: {job}'))
    j = c.get(f'/api/jobs/{job}').json()
    results.append(ok(j['status'] == 'complete' and len(j['candidates']) >= 2000, f"restored job complete, {len(j['candidates']):,} candidates, {len(j['profiles'])} profiles"))
    exports(job)
else:
    for path in ['/', '/login.html', '/app.js', '/onboarding.js', '/login.js', '/style.css', '/theme.js', '/greece-map.json']:
        results.append(ok(c.get(path).status_code == 200, f'static {path}'))
    t = c.get('/api/template')
    results.append(ok(t.status_code == 200 and t.content[:2] == b'PK', 'template download is an .xlsx'))
    results.append(ok(c.post('/api/analyze', files={'file': ('p.xlsx', b'x')}).status_code == 403, 'POST without local header rejected (403)'))
    results.append(ok(c.get('/api/status', headers={'Host': 'evil.example'}).status_code == 400, 'foreign Host header rejected (400)'))
    bad = c.post('/api/analyze', headers=H, files={'file': ('notes.txt', b'hello')})
    results.append(ok(bad.status_code == 400, f'non-xlsx upload rejected ({bad.status_code})'))
    data = {'weights': json.dumps({'activity': .5, 'geography': .3, 'legal_type': .15, 'age': .05})} if MODE == 'weights' else {}
    r = c.post('/api/analyze', headers=H, files={'file': ('ForceCRM_Hackathon_Pelatologio.xlsx', open(XLSX, 'rb').read())}, data=data)
    results.append(ok(r.status_code == 202, f'upload accepted (202) mode={MODE}'))
    job = r.json()['job_id']; start = time.time(); last = ''
    while True:
        j = c.get(f'/api/jobs/{job}').json()
        if j.get('message') != last: last = j.get('message'); print('   …', last)
        if j['status'] in ('complete', 'error') or time.time() - start > 900: break
        time.sleep(3)
    results.append(ok(j['status'] == 'complete', f"analysis {j['status']} in {time.time()-start:.0f} s"))
    if j['status'] == 'complete':
        P, R, K = j['profiles'], j['report'], j['candidates']
        results.append(ok(1 <= len(P) <= 6 and all(p.get('title') and p.get('description') for p in P), f'{len(P)} profiles with title + description'))
        results.append(ok(all(p['ai_generated'] in (True, False) and 'claims' in p and 'activity_mix' in p for p in P), 'AI labels, claims and activity_mix present'))
        results.append(ok('portfolio_traits' in R and len(K) >= 2000, f"portfolio_traits in report; {len(K):,} candidates"))
        for p in P: print(f"   {p['profile_id']} | {p['title']} | {p['title_source']} | {p['description_status'][:55]}")
        if MODE == 'weights':
            m = R['validation']['metrics']
            results.append(ok(all('default_weights_recall' in x for x in m) and R['feature_configuration']['custom_weights'], 'custom weights flagged; holdout re-run with defaults'))
            for x in m: print(f"   R@{x['k']}: custom {x['recall']:.1%} vs default {x['default_weights_recall']:.1%}")
        else:
            for x in R['validation']['metrics']: print(f"   R@{x['k']}: {x['recall']:.1%} (baseline {x['baseline_recall']:.1%})")
        exports(job)
        lc = c.post(f'/api/jobs/{job}/live-check', headers=H, data={'api_key': '', 'limit': 5})
        results.append(ok(lc.status_code == 400, f'live ΓΕΜΗ check without key refused (400): {lc.json().get("detail","")[:60]}'))
        results.append(ok(c.get('/api/latest').json()['job_id'] == job, 'latest points to this job'))
print(f'\n{sum(results)}/{len(results)} checks passed')
sys.exit(0 if all(results) else 1)
