from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

FILES = ['vwHackCompany.csv', 'vwHackCompanyActivity.csv', 'vwHackActivities.csv']


def normalize_vat(value):
    text = str(value).strip().upper().removeprefix('EL')
    text = re.sub(r'\.0$', '', text)
    if not text.isdigit() or len(text) > 9:
        raise ValueError(f'Invalid VAT / ΑΦΜ: {value!r}. Expected up to 9 digits.')
    return text.zfill(9)


def load_universe(folder: Path, use_db=None):
    """Load the GEMI universe. By default via the embedded SQLite store (`forcecrm.db`), which
    imports the CSVs once and re-imports only when their SHA-256 changes; FORCECRM_DB=0 or
    use_db=False reads the CSVs directly. Both paths yield identical frames."""
    from . import db
    folder = Path(folder)
    if db.enabled() if use_db is None else use_db:
        frames = db.load_raw(folder)
    else:
        frames = [db.read_csv(folder / name) for name in FILES]
    c, a, d = frames
    required = [set('company_id vat company_name_el legal_type is_branch city municipality_name prefecture_name incorporation_date latitude longitude'.split()),
                {'company_id', 'activity_id', 'activity_type', 'date_from'}, {'activity_id', 'description'}]
    for name, frame, fields in zip(FILES, frames, required):
        if fields - set(frame):
            raise ValueError(f'{name}: missing columns {sorted(fields - set(frame))}')
    if c.company_id.duplicated().any() or d.activity_id.duplicated().any():
        raise ValueError('Duplicate company_id or activity description ID in source data.')
    c['vat'] = c.vat.map(normalize_vat)
    c = c.sort_values('company_id').reset_index(drop=True)
    for col, low, high in [('latitude', -90, 90), ('longitude', -180, 180)]:
        c[col] = pd.to_numeric(c[col].str.replace(',', '.', regex=False), errors='coerce')
        c.loc[~c[col].between(low, high), col] = np.nan
    types = {'κύρια': 'primary', 'πρωτεύουσα': 'primary', 'primary': 'primary',
             'δευτερεύουσα': 'secondary', 'secondary': 'secondary',
             'βοηθητική': 'auxiliary', 'auxiliary': 'auxiliary'}
    a['kind'] = a.activity_type.str.strip().str.lower().map(types).fillna('unknown')
    a = a.drop_duplicates(['company_id', 'activity_id', 'kind']).reset_index(drop=True)
    report = {'venues': len(c), 'businesses': c.vat.nunique(), 'activity_rows': len(a),
              'gps_coverage': float(c[['latitude', 'longitude']].notna().all(axis=1).mean()),
              'unknown_activity_types': a.loc[a.kind.eq('unknown'), 'activity_type'].unique().tolist(),
              'orphan_activity_rows': int((~a.company_id.isin(c.company_id)).sum()),
              'missing_activity_descriptions': int((~a.activity_id.isin(d.activity_id)).sum())}
    a = a[a.company_id.isin(c.company_id)].copy()
    return c, a, d, report


def read_portfolio(source, companies):
    try:
        frame = pd.read_excel(source, dtype=str)
    except Exception as exc:
        raise ValueError('Cannot read workbook. Upload a valid .xlsx with VAT and revenue columns.') from exc
    aliases = {'vat': 'vat', 'αφμ': 'vat', 'α.φ.μ.': 'vat', 'revenue': 'revenue',
               'έσοδα': 'revenue', 'εσοδα': 'revenue', 'τζίρος': 'revenue', 'τζιρος': 'revenue'}
    def column_name(col):
        cleaned = re.sub(r'\s*\([€\s]*\)\s*', '', str(col).strip().lower()).replace('€', '').strip()
        return aliases.get(cleaned, cleaned)
    frame = frame.rename(columns={col: column_name(col) for col in frame})
    if not {'vat', 'revenue'}.issubset(frame.columns) or frame.columns.duplicated().any():
        raise ValueError('Expected distinct VAT/ΑΦΜ and revenue/έσοδα columns.')
    frame = frame[['vat', 'revenue']].dropna(how='all').copy()
    frame['vat'] = frame.vat.map(normalize_vat)

    def money(value):
        s = str(value).strip().replace('€', '').replace(' ', '').replace('\u00a0', '')
        if ',' in s:
            s = s.replace('.', '').replace(',', '.')
        try:
            result = float(s)
        except ValueError as exc:
            raise ValueError(f'Invalid revenue: {value!r}') from exc
        if not np.isfinite(result) or result < 0:
            raise ValueError('Revenue must be finite and non-negative.')
        return result

    frame['revenue'] = frame.revenue.map(money)
    if frame.vat.duplicated().any():
        raise ValueError('Duplicate VAT rows: combine each customer’s revenue into one row before uploading.')
    unmatched = frame.loc[~frame.vat.isin(companies.vat), 'vat'].tolist()
    if unmatched:
        raise ValueError(f'{len(unmatched)} VATs are outside the universe. First values: {unmatched[:5]}')
    if len(frame) < 5:
        raise ValueError('At least 5 distinct customers are required for profile analysis.')
    if frame.revenue.sum() <= 0:
        raise ValueError('Portfolio revenue must have a positive total.')
    return frame.sort_values('vat').reset_index(drop=True)


def demo_portfolio(companies, activities, seed=42):
    """Synthetic selection and revenue; never an evaluation of the real portfolio."""
    primary = activities[activities.kind.eq('primary')].copy()
    common = primary.activity_id.value_counts().head(3).index
    ids = primary.loc[primary.activity_id.isin(common), 'company_id']
    pool = companies[companies.company_id.isin(ids)].drop_duplicates('vat')
    if len(pool) < 10:
        pool = companies.drop_duplicates('vat')
    sample = pool.sample(min(300, len(pool)), random_state=seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'vat': sample.vat, 'revenue': np.round(rng.lognormal(8, .65, len(sample)), 2)}).reset_index(drop=True)
