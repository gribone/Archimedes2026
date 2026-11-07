"""Opt-in live check of prospect status against the GEMI Open Data API.

This is the only feature that contacts an external service. It runs server-side, only when the
user enables it and supplies a personal GEMI api_key, and it sends nothing but the VAT numbers of
the prospects being checked (never the customer portfolio or revenue).
API spec: https://opendata-api.businessportal.gr/opendata/docs/ (Swagger 2.0, header `api_key`).
Data licence: ODC-BY-1.0, © Κεντρική Υπηρεσία ΓΕΜΗ, Κεντρική Ένωση Επιμελητηρίων Ελλάδος.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import json
import os
import threading
import time
from pathlib import Path

import httpx

GEMI_API = os.environ.get('GEMI_API_URL', 'https://opendata-api.businessportal.gr/api/opendata/v1')
CACHE_DAYS = 7
ATTRIBUTION = '© ODC-BY-1.0 – Κεντρική Υπηρεσία ΓΕΜΗ, Κεντρική Ένωση Επιμελητηρίων Ελλάδος'
_cache_lock = threading.Lock()


class LiveCheckError(Exception):
    """Stops the whole run (bad key, API unavailable)."""


def requests_per_minute():
    try:
        return max(1.0, float(os.environ.get('GEMI_REQUESTS_PER_MINUTE', '8')))
    except ValueError:
        return 8.0


def classify(payload):
    """Map a /companies search response to active / inactive / not_found."""
    results = (payload or {}).get('searchResults') or []
    if not results:
        return {'status': 'not_found', 'description': None, 'ar_gemi': None}
    statuses = [r.get('status') or {} for r in results]
    active = [r for r, s in zip(results, statuses) if s.get('isActive') is True]
    chosen = active[0] if active else results[0]
    status = chosen.get('status') or {}
    return {'status': 'active' if active else 'inactive', 'description': status.get('descr'),
            'ar_gemi': chosen.get('arGemi'), 'last_status_change': chosen.get('lastStatusChange')}


def lookup(client, vat, api_key, retries=3, sleep=time.sleep):
    for attempt in range(retries + 1):
        try:
            response = client.get(GEMI_API + '/companies', params={'afm': vat, 'resultsSize': 10},
                                  headers={'api_key': api_key, 'Accept': 'application/json'})
        except httpx.HTTPError as exc:
            if attempt == retries:
                raise LiveCheckError(f'Δεν ήταν δυνατή η σύνδεση με το ΓΕΜΗ: {exc}') from exc
            sleep(5 * (attempt + 1))
            continue
        if response.status_code in (401, 403):
            raise LiveCheckError('Το ΓΕΜΗ απέρριψε το api_key (μη εξουσιοδοτημένο).')
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == retries:
                raise LiveCheckError(f'Το ΓΕΜΗ δεν απαντά (HTTP {response.status_code}). Δοκιμάστε αργότερα.')
            wait = response.headers.get('Retry-After')
            sleep(float(wait) if wait and wait.isdigit() else 15 * (attempt + 1))
            continue
        if response.status_code == 404:
            return classify({})
        if response.status_code >= 400:
            return {'status': 'error', 'description': f'HTTP {response.status_code}', 'ar_gemi': None}
        return classify(response.json())


def load_cache(path: Path):
    try:
        return json.loads(path.read_text('utf-8'))
    except (OSError, ValueError):
        return {}


def save_cache(path: Path, cache):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache, ensure_ascii=False), 'utf-8')
    tmp.replace(path)


def check_vats(vats, api_key, root: Path, on_result, should_stop=lambda: False, client=None, sleep=time.sleep):
    """Check each VAT, reusing results younger than CACHE_DAYS. Calls on_result(vat, result)."""
    if not api_key:
        raise LiveCheckError('Απαιτείται προσωπικό api_key του ΓΕΜΗ.')
    cache_path = root / '.cache' / 'gemi-status.json'
    with _cache_lock:
        cache = load_cache(cache_path)
    fresh_after = datetime.now(timezone.utc) - timedelta(days=CACHE_DAYS)
    interval = 60.0 / requests_per_minute()
    own_client = client is None
    client = client or httpx.Client(timeout=30, trust_env=True)
    last_call = None
    try:
        for vat in vats:
            if should_stop():
                break
            cached = cache.get(vat)
            if cached and datetime.fromisoformat(cached['checked_at']) > fresh_after:
                on_result(vat, {**cached, 'cached': True})
                continue
            if last_call is not None:
                sleep(max(0.0, interval - (time.monotonic() - last_call)))
            last_call = time.monotonic()
            result = lookup(client, vat, api_key, sleep=sleep)
            result['checked_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            if result['status'] != 'error':
                cache[vat] = result
                with _cache_lock:
                    save_cache(cache_path, cache)
            on_result(vat, {**result, 'cached': False})
    finally:
        if own_client:
            client.close()


def export_live(results, candidates, directory: Path):
    """Write live-status.csv next to the other exports."""
    rows = {c['vat']: c for c in candidates}
    with (directory / 'live-status.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['vat', 'rank', 'company_name', 'gemi_status', 'gemi_description', 'ar_gemi', 'checked_at', 'source'])
        for vat, result in results.items():
            candidate = rows.get(vat, {})
            writer.writerow([vat, candidate.get('rank'), candidate.get('company_name_el'), result['status'],
                             result.get('description') or '', result.get('ar_gemi') or '', result.get('checked_at', ''),
                             ATTRIBUTION])
