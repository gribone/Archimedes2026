"""Embedded SQLite store: the GEMI universe, analysis runs, live ΓΕΜΗ statuses and (later) users.

Serverless by design: one file (`.cache/forcecrm.sqlite`), the standard-library `sqlite3`
driver, no daemon. The three source CSVs are imported once and re-imported only when their
SHA-256 changes. Universe rows are stored exactly as pandas reads the CSVs (all text, original
row and column order), and `data.load_universe` applies the same normalisation on top, so the
analysis sees identical frames and the file exports stay byte-identical.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import threading
import unicodedata
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = 1
DB_FILE = Path('.cache') / 'forcecrm.sqlite'
# Source CSV -> table holding its raw rows (text columns, original order).
UNIVERSE_TABLES = {'vwHackCompany.csv': 'companies', 'vwHackCompanyActivity.csv': 'activities',
                   'vwHackActivities.csv': 'activity_descriptions'}
UNIVERSE_INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_companies_company ON companies(company_id)',
    'CREATE INDEX IF NOT EXISTS idx_companies_vat ON companies(vat)',
    'CREATE INDEX IF NOT EXISTS idx_companies_municipality ON companies(municipality_name)',
    'CREATE INDEX IF NOT EXISTS idx_activities_company ON activities(company_id)',
    'CREATE INDEX IF NOT EXISTS idx_descriptions_activity ON activity_descriptions(activity_id)',
]
APP_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, demo INTEGER NOT NULL,
    portfolio_sha256 TEXT, universe_sha256 TEXT, params TEXT NOT NULL,
    profile_count INTEGER, candidate_count INTEGER, report TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, profile_id TEXT NOT NULL,
    title TEXT, customer_count INTEGER, revenue_share REAL, stability REAL, confidence REAL,
    data TEXT NOT NULL, PRIMARY KEY (run_id, profile_id));
CREATE TABLE IF NOT EXISTS profile_traits (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, profile_id TEXT NOT NULL,
    feature TEXT NOT NULL, label TEXT, count INTEGER, total INTEGER, profile_rate REAL,
    universe_rate REAL, lift REAL, p_value REAL, q_value REAL);
CREATE INDEX IF NOT EXISTS idx_traits_run ON profile_traits(run_id, profile_id);
CREATE TABLE IF NOT EXISTS assignments (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, company_id TEXT NOT NULL,
    profile_id TEXT NOT NULL, score REAL, PRIMARY KEY (run_id, company_id));
CREATE TABLE IF NOT EXISTS lookalikes (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, rank INTEGER NOT NULL,
    profile_id TEXT NOT NULL, company_id TEXT NOT NULL, score REAL, rationale TEXT,
    vat TEXT, company_name_el TEXT, city TEXT, municipality_name TEXT, latitude REAL, longitude REAL,
    search_text TEXT NOT NULL, PRIMARY KEY (run_id, rank));
CREATE INDEX IF NOT EXISTS idx_lookalikes_profile ON lookalikes(run_id, profile_id, rank);
CREATE INDEX IF NOT EXISTS idx_lookalikes_vat ON lookalikes(run_id, vat);
CREATE TABLE IF NOT EXISTS gemi_status (
    vat TEXT PRIMARY KEY, status TEXT NOT NULL, description TEXT, ar_gemi INTEGER,
    last_status_change TEXT, checked_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL);
"""
_import_lock = threading.Lock()


def enabled():
    return os.environ.get('FORCECRM_DB', '1') != '0'


def db_path(root: Path):
    return Path(root) / DB_FILE


def fold(text):
    """Case- and accent-insensitive key for Greek/Latin search (SQLite lower() is ASCII-only)."""
    if text is None:
        return ''
    decomposed = unicodedata.normalize('NFD', str(text).casefold())
    return ''.join(ch for ch in decomposed if not unicodedata.combining(ch))


def connect(root: Path):
    path = db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA foreign_keys=ON')
    con.create_function('fold', 1, fold, deterministic=True)
    con.executescript(APP_SCHEMA)
    con.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    return con


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(folder: Path):
    return {name: file_sha256(Path(folder) / name) for name in UNIVERSE_TABLES}


def read_csv(path: Path):
    """The single definition of how a source CSV is parsed (shared by the CSV and DB paths)."""
    return pd.read_csv(path, sep=';', dtype=str, encoding='utf-8-sig',
                       na_values=['NULL'], keep_default_na=True).fillna('')


def _meta(con, key):
    row = con.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
    return json.loads(row['value']) if row else None


def _set_meta(con, key, value):
    con.execute('INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                (key, json.dumps(value, ensure_ascii=False)))


def quote(name):
    return '"' + str(name).replace('"', '""') + '"'


def ensure_universe(con, folder: Path):
    """Import the CSVs if the DB is empty or their hashes changed. Returns True when it imported."""
    hashes = source_hashes(folder)
    if _meta(con, 'universe_sha256') == hashes:
        return False
    with _import_lock:
        con.execute('BEGIN IMMEDIATE')  # serialises concurrent importers across threads/processes
        try:
            if _meta(con, 'universe_sha256') == hashes:
                con.rollback()
                return False
            columns = {}
            for name, table in UNIVERSE_TABLES.items():
                frame = read_csv(Path(folder) / name)
                columns[table] = list(frame.columns)
                con.execute(f'DROP TABLE IF EXISTS {table}')
                con.execute(f'CREATE TABLE {table} ({", ".join(quote(c) + " TEXT" for c in frame.columns)})')
                placeholders = ', '.join('?' * len(frame.columns))
                con.executemany(f'INSERT INTO {table} VALUES ({placeholders})', frame.itertuples(index=False, name=None))
            for statement in UNIVERSE_INDEXES:
                try:
                    con.execute(statement)
                except sqlite3.OperationalError:
                    pass  # an optional column is missing; required columns are validated by data.py
            _set_meta(con, 'universe_columns', columns)
            _set_meta(con, 'universe_sha256', hashes)
            _set_meta(con, 'universe_imported_at', now())
            con.commit()
        except BaseException:
            con.rollback()
            raise
    return True


def read_universe(con):
    """Raw frames, identical to `read_csv` on each source file."""
    columns = _meta(con, 'universe_columns')
    frames = []
    cursor = con.cursor()
    cursor.row_factory = None  # plain tuples: much faster than sqlite3.Row for bulk reads
    for table in UNIVERSE_TABLES.values():
        names = columns[table]
        rows = cursor.execute(f'SELECT {", ".join(quote(c) for c in names)} FROM {table} ORDER BY rowid').fetchall()
        frames.append(pd.DataFrame.from_records(rows, columns=names) if rows else pd.DataFrame(columns=names, dtype=str))
    return frames


def load_raw(folder: Path):
    """Raw frames via the DB (importing on first use or when the CSVs changed)."""
    with closing(connect(folder)) as con:
        ensure_universe(con, folder)
        return read_universe(con)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, 'item') else value


def save_run(con, run_id, profiles, report, assignments, lookalikes, companies, params=None, status='complete'):
    """Store (or replace) one analysis. `companies` supplies display fields for lookalikes."""
    details = companies.set_index('company_id')[['vat', 'company_name_el', 'city', 'municipality_name',
                                                  'latitude', 'longitude']]
    with con:
        con.execute('DELETE FROM runs WHERE run_id = ?', (run_id,))
        con.execute('INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (
            run_id, now(), status, int(bool(report.get('demo'))), report.get('portfolio_sha256'),
            json.dumps(report.get('input_sha256'), ensure_ascii=False),
            json.dumps(params or {}, ensure_ascii=False), len(profiles), len(lookalikes),
            json.dumps(report, ensure_ascii=False, allow_nan=False)))
        con.executemany('INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)', [
            (run_id, p['profile_id'], p.get('title'), p.get('customer_count'), p.get('revenue_share'),
             p.get('stability'), p.get('confidence_score'), json.dumps(p, ensure_ascii=False, allow_nan=False))
            for p in profiles])
        con.executemany('INSERT INTO profile_traits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
            (run_id, p['profile_id'], t['feature'], t.get('label'), t.get('count'), t.get('total'),
             t.get('profile_rate'), t.get('universe_rate'), t.get('lift'), t.get('p_value'), t.get('q_value'))
            for p in profiles for t in p.get('significant_features', [])])
        con.executemany('INSERT INTO assignments VALUES (?, ?, ?, ?)', [
            (run_id, str(r.company_id), r.profile_id, _clean(r.score)) for r in assignments.itertuples(index=False)])
        rows = []
        for r in lookalikes.itertuples(index=False):
            company = str(r.company_id)
            d = details.loc[company] if company in details.index else None
            values = [None] * 6 if d is None else [_clean(d[k]) for k in details.columns]
            rows.append((run_id, int(r.rank), r.profile_id, company, _clean(r.score), r.rationale, *values,
                         fold(' '.join(str(v) for v in values[:4] if v))))
        con.executemany('INSERT INTO lookalikes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)


def save_run_from_directory(con, run_id, directory: Path, companies):
    """Import an export folder (CLI runs, runs from before the DB existed)."""
    directory = Path(directory)
    profiles = json.loads((directory / 'profiles.json').read_text('utf-8'))
    report = json.loads((directory / 'report.json').read_text('utf-8'))
    assignments = pd.read_csv(directory / 'assignments.csv', dtype={'company_id': str})
    lookalikes = pd.read_csv(directory / 'lookalikes.csv', dtype={'company_id': str})
    save_run(con, run_id, profiles, report, assignments, lookalikes, companies, params={'imported_from': str(directory)})


def has_run(con, run_id):
    return con.execute('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)).fetchone() is not None


def list_runs(con, limit=20):
    rows = con.execute('SELECT run_id, created_at, status, demo, profile_count, candidate_count, params '
                       'FROM runs ORDER BY created_at DESC, run_id LIMIT ?', (limit,)).fetchall()
    return [{**dict(r), 'demo': bool(r['demo']), 'params': json.loads(r['params'])} for r in rows]


LIVE_FILTERS = {
    None: '', '': '',
    'hide-inactive': " AND COALESCE(g.status, '') != 'inactive'",
    'active': " AND g.status = 'active'",
}


def query_lookalikes(con, run_id, profile=None, q=None, live=None, offset=0, limit=50):
    """One page of a run's lookalikes in global rank order, with total count after filters."""
    if live not in LIVE_FILTERS:
        raise ValueError(f'Unknown live filter: {live!r}')
    where, args = 'l.run_id = ?', [run_id]
    if profile:
        where += ' AND l.profile_id = ?'
        args.append(profile)
    if q:
        where += " AND l.search_text LIKE ? ESCAPE '\\'"
        term = fold(q).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        args.append(f'%{term}%')
    where += LIVE_FILTERS[live]
    source = 'lookalikes l LEFT JOIN gemi_status g ON g.vat = l.vat'
    total = con.execute(f'SELECT COUNT(*) FROM {source} WHERE {where}', args).fetchone()[0]
    rows = con.execute(
        f'SELECT l.rank, l.profile_id, l.company_id, l.score, l.rationale, l.vat, l.company_name_el, l.city, '
        f'l.municipality_name, l.latitude, l.longitude, g.status AS gemi_status, g.description AS gemi_description '
        f'FROM {source} WHERE {where} ORDER BY l.rank LIMIT ? OFFSET ?', [*args, limit, offset]).fetchall()
    return {'total': total, 'offset': offset, 'limit': limit, 'rows': [dict(r) for r in rows]}


def map_points(con, run_id, limit=2000):
    rows = con.execute('SELECT rank, profile_id, vat, company_name_el, city, latitude, longitude FROM lookalikes '
                       'WHERE run_id = ? ORDER BY rank LIMIT ?', (run_id, limit)).fetchall()
    return [dict(r) for r in rows]


def top_vats(con, run_id, limit):
    rows = con.execute('SELECT vat FROM lookalikes WHERE run_id = ? ORDER BY rank LIMIT ?', (run_id, limit)).fetchall()
    return list(dict.fromkeys(r['vat'] for r in rows))


def company_detail(con, company_id):
    from .data import normalize_vat
    row = con.execute('SELECT * FROM companies WHERE company_id = ?', (str(company_id),)).fetchone()
    if row is None:
        return None
    company = dict(row)
    company['vat'] = normalize_vat(company['vat'])
    for key in ('latitude', 'longitude'):
        try:
            company[key] = float(str(company.get(key, '')).replace(',', '.'))
        except ValueError:
            company[key] = None
    activities = con.execute(
        'SELECT a.activity_id, a.activity_type, a.date_from, d.description FROM activities a '
        'LEFT JOIN activity_descriptions d ON d.activity_id = a.activity_id '
        'WHERE a.company_id = ? ORDER BY a.rowid', (str(company_id),)).fetchall()
    company['activities'] = [dict(a) for a in activities]
    status = con.execute('SELECT * FROM gemi_status WHERE vat = ?', (company['vat'],)).fetchone()
    company['gemi_status'] = dict(status) if status else None
    return company


def search_companies(con, q, limit=20):
    term = '%' + fold(q).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
    rows = con.execute(
        "SELECT company_id, vat, company_name_el, city, municipality_name FROM companies "
        "WHERE fold(company_name_el || ' ' || city || ' ' || municipality_name || ' ' || vat) LIKE ? ESCAPE '\\' "
        "ORDER BY rowid LIMIT ?", (term, limit)).fetchall()
    return [dict(r) for r in rows]


# ---- Live ΓΕΜΗ status cache (used by live.py) ----

def get_gemi_status(con, vat):
    row = con.execute('SELECT * FROM gemi_status WHERE vat = ?', (vat,)).fetchone()
    return dict(row) if row else None


def put_gemi_status(con, vat, result):
    with con:
        con.execute('INSERT INTO gemi_status VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(vat) DO UPDATE SET '
                    'status = excluded.status, description = excluded.description, ar_gemi = excluded.ar_gemi, '
                    'last_status_change = excluded.last_status_change, checked_at = excluded.checked_at',
                    (vat, result['status'], result.get('description'), result.get('ar_gemi'),
                     result.get('last_status_change'), result['checked_at']))
