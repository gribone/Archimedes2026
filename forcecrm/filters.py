"""User-chosen exclusions: drop businesses by activity (ΚΑΔ) or location, before or after the analysis.

Two modes, chosen by the user:
- 'pre'  (raw data): excluded venues are removed from the universe before features, clustering,
         statistics and ranking; customers whose venues are all excluded leave the portfolio.
         Profiles, base rates and the holdout are computed on the filtered market.
- 'post' (results only): the analysis runs on the full data and profiles are unchanged; excluded
         businesses are removed from the suggestions, which are re-ranked 1..n. The internal holdout
         applies the same exclusion to the ranked lists and the baseline.

A venue is excluded when its primary ΚΑΔ (or, with activity_scope='any', any declared ΚΑΔ) starts with
one of the chosen codes, or when it lies in a chosen Περιφερειακή Ενότητα or Δήμος. Exclusion is per
venue (shop); in suggestions a VAT appears only through its non-excluded venues.
"""
from __future__ import annotations

import re

import numpy as np

MODES = ('pre', 'post')
SCOPES = ('primary', 'any')
MAX_ITEMS = 500
LABELS = {'pre': 'πριν από την ανάλυση (ακατέργαστα δεδομένα)', 'post': 'μετά την ανάλυση (μόνο στις προτάσεις)'}


def normalize_filters(raw):
    """Validate user input. Returns None when nothing is excluded (the default analysis)."""
    if raw in (None, '', {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError('filters must be an object')
    mode = raw.get('mode', 'post')
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    scope = raw.get('activity_scope', 'primary')
    if scope not in SCOPES:
        raise ValueError(f'activity_scope must be one of {SCOPES}')

    def items(name, check=None):
        values = raw.get(name) or []
        if not isinstance(values, list) or len(values) > MAX_ITEMS:
            raise ValueError(f'{name} must be a list of at most {MAX_ITEMS} values')
        cleaned = set()
        for value in values:
            text = str(value).strip()
            if not text or len(text) > 120:
                raise ValueError(f'invalid {name} value: {value!r}')
            if check:
                text = check(text)
            cleaned.add(text)
        return sorted(cleaned)

    def code(text):
        text = text.replace('.', '')
        if not re.fullmatch(r'\d{2,8}', text):
            raise ValueError(f'ΚΑΔ must be 2–8 digits: {text!r}')
        return text

    result = {'mode': mode, 'activity_scope': scope, 'activities': items('activities', code),
              'prefectures': items('prefectures', str.upper), 'municipalities': items('municipalities', str.upper)}
    if not (result['activities'] or result['prefectures'] or result['municipalities']):
        return None
    return result


def excluded_mask(companies, activities, filters):
    """Boolean array aligned with `companies`: True for venues the filters exclude."""
    mask = np.zeros(len(companies), dtype=bool)
    if not filters:
        return mask
    if filters['prefectures']:
        mask |= companies.prefecture_name.fillna('').str.upper().isin(filters['prefectures']).to_numpy()
    if filters['municipalities']:
        mask |= companies.municipality_name.fillna('').str.upper().isin(filters['municipalities']).to_numpy()
    if filters['activities']:
        rows = activities if filters['activity_scope'] == 'any' else activities[activities.kind.eq('primary')]
        hit = rows.activity_id.astype(str).str.startswith(tuple(filters['activities']))
        mask |= companies.company_id.isin(set(rows.loc[hit, 'company_id'])).to_numpy()
    return mask


def apply_to_universe(companies, activities, filters):
    """'pre' mode: the universe without excluded venues (and their activity rows)."""
    keep = ~excluded_mask(companies, activities, filters)
    kept = companies[keep].reset_index(drop=True)
    # A business (VAT) leaves the universe only when every one of its venues is excluded.
    gone = set(companies.vat[~keep]) - set(kept.vat)
    return kept, activities[activities.company_id.isin(kept.company_id)].reset_index(drop=True), {
        'excluded_venues': int((~keep).sum()), 'excluded_businesses': len(gone)}


def remaining_portfolio(portfolio, companies, activities, filters):
    """'pre' mode: customers keep only non-excluded venues; a customer with none left is dropped."""
    keep = ~excluded_mask(companies, activities, filters)
    eligible = set(companies.vat[keep])
    dropped = sorted(set(portfolio.vat) - eligible)
    return portfolio[portfolio.vat.isin(eligible)].reset_index(drop=True), dropped


def summary(filters):
    """One Greek sentence describing the filters, for the report and the UI."""
    parts = []
    if filters['activities']:
        scope = 'κύρια δραστηριότητα' if filters['activity_scope'] == 'primary' else 'οποιαδήποτε δηλωμένη δραστηριότητα'
        parts.append(f"ΚΑΔ {', '.join(filters['activities'])} ({scope})")
    if filters['prefectures']:
        parts.append('Π.Ε. ' + ', '.join(filters['prefectures']))
    if filters['municipalities']:
        parts.append('Δήμοι ' + ', '.join(filters['municipalities']))
    return f"Εξαιρέθηκαν {LABELS[filters['mode']]}: " + ' · '.join(parts) + '.'
