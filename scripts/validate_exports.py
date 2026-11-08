"""Validate actual grading files against source data and customer exclusion rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from forcecrm.data import load_universe, read_portfolio
from forcecrm.filters import excluded_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/real'))
    parser.add_argument('--portfolio', type=Path, default=Path('ForceCRM_Hackathon_Pelatologio.xlsx'))
    args = parser.parse_args()
    c, a, _, _ = load_universe(Path('.'))
    p = read_portfolio(args.portfolio, c)
    report_path = args.output / 'report.json'
    filters = json.loads(report_path.read_text('utf-8')).get('filters') if report_path.exists() else None
    if filters:
        # A user-filtered run: excluded venues are absent ('pre') or never suggested ('post').
        mask = excluded_mask(c, a, filters)
        if filters['mode'] == 'pre':
            assert mask.sum() == filters['excluded_venues']
            p = p[~p.vat.isin(filters['dropped_customer_vats'])]
            c_all, c = c, c[~mask]
        excluded_ids = set(c_all.company_id[mask]) if filters['mode'] == 'pre' else set(c.company_id[mask])
    profiles = json.loads((args.output / 'profiles.json').read_text('utf-8'))
    assignments = pd.read_csv(args.output / 'assignments.csv', dtype={'company_id': str})
    candidates = pd.read_csv(args.output / 'lookalikes.csv', dtype={'company_id': str})
    assert 1 <= len(profiles) <= 6
    assert abs(sum(row['revenue_share'] for row in profiles) - 1) < 1e-8
    assert assignments.company_id.is_unique
    assert set(assignments.company_id) == set(c.company_id)
    assert assignments.columns.tolist() == ['company_id', 'profile_id', 'score']
    assert candidates.columns.tolist() == ['profile_id', 'company_id', 'rank', 'score', 'rationale']
    assert len(candidates) >= 2000
    assert candidates.company_id.is_unique
    assert candidates['rank'].tolist() == list(range(1, len(candidates) + 1))
    assert candidates.score.is_monotonic_decreasing
    assert np.isfinite(candidates.score).all() and candidates.score.between(0, 1).all()
    vats = candidates.company_id.map(c.set_index('company_id').vat)
    assert vats.notna().all() and vats.is_unique
    assert not set(vats) & set(p.vat)
    if filters:
        assert not set(candidates.company_id) & excluded_ids
    ids = {row['profile_id'] for row in profiles}
    assert set(assignments.profile_id) <= ids | {'NONE'}
    assert set(candidates.profile_id) <= ids
    for profile in profiles:
        assert profile['ai_generated'] is True
        assert profile['description'] == ' '.join(claim['sentence'] for claim in profile['claims'])
        assert 0 <= profile['stability'] <= 1 and 0 <= profile['confidence_score'] <= 1
        assert profile['confidence'] == profile['confidence_score']  # brief §10 name alongside the original
        for key in ('profile_id', 'title', 'description', 'customer_count', 'revenue_share', 'stability', 'confidence'):
            assert key in profile, key
        assert all({'lift', 'p_value'} <= set(f) for f in profile['significant_features'])
        assert all(f['q_value'] <= .05 and f['count'] >= 5 and f['lift'] >= 1.2 for f in profile['significant_features'])
    print(f'PASS: {len(profiles)} grounded profiles, {len(assignments)} venue assignments, {len(candidates)} unique non-customer VAT recommendations.')
    if filters:
        print(f"NOTE: user-filtered run ({filters['mode']}): {filters['summary']} "
              'Submission exports must come from an unfiltered run.')


if __name__ == '__main__':
    main()
