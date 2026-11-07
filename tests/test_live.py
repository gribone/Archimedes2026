import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from forcecrm.live import LiveCheckError, check_vats, classify, export_live

ACTIVE = {'searchMetadata': {'totalCount': 1}, 'searchResults': [
    {'arGemi': 123, 'afm': '000000001', 'status': {'id': 1, 'descr': 'Ενεργή', 'isActive': True}}]}
DELETED = {'searchMetadata': {'totalCount': 1}, 'searchResults': [
    {'arGemi': 456, 'afm': '000000002', 'status': {'id': 3, 'descr': 'Διαγραμμένη', 'isActive': False},
     'lastStatusChange': '2024-03-01'}]}


def fake_gemi(responses, calls):
    def handler(request):
        calls.append(request)
        afm = request.url.params['afm']
        code, body = responses[afm]
        return httpx.Response(code, json=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def run(vats, responses, tmp_path, key='k'):
    calls, results = [], {}
    client = fake_gemi(responses, calls)
    check_vats(vats, key, tmp_path, lambda vat, r: results.__setitem__(vat, r), client=client, sleep=lambda s: None)
    return results, calls


def test_classify_active_inactive_and_missing():
    assert classify(ACTIVE)['status'] == 'active'
    assert classify(DELETED) | {} == {'status': 'inactive', 'description': 'Διαγραμμένη', 'ar_gemi': 456,
                                      'last_status_change': '2024-03-01'}
    assert classify({'searchResults': []})['status'] == 'not_found'
    # Any active registration for the VAT counts as active.
    both = {'searchResults': DELETED['searchResults'] + ACTIVE['searchResults']}
    assert classify(both)['ar_gemi'] == 123


def test_check_sends_only_vat_with_api_key_header_and_caches(tmp_path):
    responses = {'000000001': (200, ACTIVE), '000000002': (200, DELETED), '000000003': (200, {'searchResults': []})}
    results, calls = run(['000000001', '000000002', '000000003'], responses, tmp_path)
    assert [r['status'] for r in results.values()] == ['active', 'inactive', 'not_found']
    assert all(c.headers['api_key'] == 'k' and set(c.url.params) == {'afm', 'resultsSize'} for c in calls)
    assert calls[0].url.path == '/api/opendata/v1/companies'
    # Second run is served from the cache without network calls.
    results, calls = run(['000000001', '000000002'], responses, tmp_path)
    assert calls == [] and all(r['cached'] for r in results.values())


def test_stale_cache_is_rechecked(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    (tmp_path / '.cache').mkdir()
    (tmp_path / '.cache' / 'gemi-status.json').write_text(json.dumps({'000000001': {'status': 'inactive', 'checked_at': old}}))
    results, calls = run(['000000001'], {'000000001': (200, ACTIVE)}, tmp_path)
    assert len(calls) == 1 and results['000000001']['status'] == 'active'


def test_rejected_key_stops_the_run(tmp_path):
    with pytest.raises(LiveCheckError, match='api_key'):
        run(['000000001'], {'000000001': (401, {'message': 'Unauthorized'})}, tmp_path)
    with pytest.raises(LiveCheckError):
        run(['000000001'], {}, tmp_path, key='')


def test_rate_limited_requests_are_retried(tmp_path):
    attempts = []
    def handler(request):
        attempts.append(request)
        return httpx.Response(429) if len(attempts) < 3 else httpx.Response(200, json=ACTIVE)
    results = {}
    check_vats(['000000001'], 'k', tmp_path, lambda v, r: results.__setitem__(v, r),
               client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    assert len(attempts) == 3 and results['000000001']['status'] == 'active'


def test_export_live_writes_csv(tmp_path):
    export_live({'000000002': {'status': 'inactive', 'description': 'Διαγραμμένη', 'ar_gemi': 456, 'checked_at': 'x'}},
                [{'vat': '000000002', 'rank': 4, 'company_name_el': 'Α'}], tmp_path)
    text = (tmp_path / 'live-status.csv').read_text('utf-8-sig')
    assert 'inactive' in text and 'Διαγραμμένη' in text and 'ΓΕΜΗ' in text


def test_live_check_api_requires_key_and_completed_job(monkeypatch):
    from fastapi.testclient import TestClient
    from forcecrm.server import app
    monkeypatch.delenv('GEMI_API_KEY', raising=False)
    with TestClient(app) as client:
        headers = {'X-ForceCRM': 'local'}
        assert client.post('/api/jobs/missing/live-check', data={'api_key': ''}, headers=headers).status_code == 400
        assert client.post('/api/jobs/missing/live-check', data={'api_key': 'k'}, headers=headers).status_code == 404
        assert client.post('/api/jobs/missing/live-check', data={'api_key': 'k', 'limit': 500}, headers=headers).status_code == 400
        assert client.get('/api/jobs/missing/live').status_code == 404
