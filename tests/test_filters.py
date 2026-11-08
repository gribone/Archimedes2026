"""User exclusions by activity or location, applied before ('pre') or after ('post') the analysis."""
import pandas as pd
import pytest

from forcecrm.data import load_universe
from forcecrm.filters import excluded_mask, normalize_filters
from tests.test_stages import universe, analyze  # noqa: F401  (module-scoped synthetic universe)


def test_normalize_filters():
    assert normalize_filters(None) is None
    assert normalize_filters({'mode': 'pre', 'activities': [], 'prefectures': []}) is None
    f = normalize_filters({'mode': 'post', 'activities': ['5630.00.07', '5630'], 'municipalities': ['θηρασ', 'ΘΗΡΑΣ']})
    assert f == {'mode': 'post', 'activity_scope': 'primary', 'activities': ['5630', '56300007'],
                 'prefectures': [], 'municipalities': ['ΘΗΡΑΣ']}
    for bad in [{'mode': 'later', 'activities': ['56']}, {'activities': ['ab']}, {'activities': '5630'},
                {'activities': ['5'], 'mode': 'pre'}, {'activity_scope': 'all', 'activities': ['56']}, []]:
        with pytest.raises(ValueError):
            normalize_filters(bad)


def test_mask_matches_primary_or_any_activity_and_locations(universe):
    root, _ = universe
    c, a, _, _ = load_universe(root, use_db=False)
    cafe = normalize_filters({'activities': ['5630']})
    assert excluded_mask(c, a, cafe).sum() == 60  # one primary code in four
    # 56210000 is every venue's secondary code: excluded only with activity_scope='any'.
    assert excluded_mask(c, a, normalize_filters({'activities': ['5621']})).sum() == 60
    assert excluded_mask(c, a, normalize_filters({'activities': ['5621'], 'activity_scope': 'any'})).all()
    places = normalize_filters({'prefectures': ['ΚΥΚΛΑΔΩΝ'], 'municipalities': ['ΑΘΗΝΑΙΩΝ']})
    assert excluded_mask(c, a, places).sum() == 160


def companies_of(root):
    c, a, _, _ = load_universe(root, use_db=False)
    return c, a


def test_post_filter_keeps_profiles_and_removes_suggestions(universe):
    root, portfolio = universe
    base = analyze(root, portfolio)
    f = {'mode': 'post', 'prefectures': ['ΘΕΣΣΑΛΟΝΙΚΗΣ']}
    result = analyze(root, portfolio, filters=f)
    c, _ = companies_of(root)
    excluded = set(c.company_id[c.prefecture_name.eq('ΘΕΣΣΑΛΟΝΙΚΗΣ')])
    assert not set(result['lookalikes'].company_id) & excluded
    assert result['lookalikes']['rank'].tolist() == list(range(1, len(result['lookalikes']) + 1))
    # Same order as the unfiltered list with the excluded rows taken out.
    kept = base['lookalikes'][~base['lookalikes'].company_id.isin(excluded)]
    assert result['lookalikes'].company_id.tolist() == kept.company_id.tolist()
    assert [p['customer_count'] for p in result['profiles']] == [p['customer_count'] for p in base['profiles']]
    report = result['report']['filters']
    assert report['mode'] == 'post' and report['excluded_venues'] == 80 and report['dropped_customers'] == 0
    assert any('μόνο από τις προτάσεις' in w for w in result['report']['warnings'])
    assert 'held_out_excluded' in result['report']['validation']


def test_pre_filter_removes_venues_before_the_analysis(universe):
    root, portfolio = universe
    f = {'mode': 'pre', 'prefectures': ['ΚΥΚΛΑΔΩΝ']}
    result = analyze(root, portfolio, filters=f)
    c, _ = companies_of(root)
    island = c[c.prefecture_name.eq('ΚΥΚΛΑΔΩΝ')]
    assert not set(result['lookalikes'].company_id) & set(island.company_id)
    assert not set(result['assignments'].company_id) & set(island.company_id)
    dropped = set(portfolio.vat) & set(island.vat)
    report = result['report']['filters']
    assert dropped and report['dropped_customers'] == len(dropped)
    assert result['report']['portfolio_customers'] == len(portfolio) - len(dropped)
    assert sum(p['customer_count'] for p in result['profiles']) <= len(portfolio) - len(dropped)


def test_pre_filter_needs_five_customers_left(universe):
    root, portfolio = universe
    with pytest.raises(ValueError, match='τουλάχιστον 5'):
        analyze(root, portfolio, filters={'mode': 'pre', 'prefectures': ['ΚΥΚΛΑΔΩΝ', 'ΑΤΤΙΚΗΣ']})


def test_no_filters_is_the_default_analysis(universe):
    root, portfolio = universe
    base, same = analyze(root, portfolio), analyze(root, portfolio, filters={'mode': 'pre', 'activities': []})
    pd.testing.assert_frame_equal(base['lookalikes'], same['lookalikes'])
    assert same['report']['filters'] is None


def test_filter_api(universe, monkeypatch):
    from fastapi.testclient import TestClient
    from forcecrm import server
    root, _ = universe
    monkeypatch.setattr(server, 'ROOT', root)
    H = {'X-ForceCRM': 'local'}
    with TestClient(server.app) as client:
        options = client.get('/api/filters/options').json()
        assert options['total_venues'] == 240
        assert {a['id'] for a in options['activities']} >= {'56300007', '56210000'}
        assert [p['name'] for p in options['prefectures']] == ['ΑΤΤΙΚΗΣ', 'ΘΕΣΣΑΛΟΝΙΚΗΣ', 'ΚΥΚΛΑΔΩΝ']
        preview = client.post('/api/filters/preview', headers=H, json={'prefectures': ['ΚΥΚΛΑΔΩΝ']}).json()
        assert preview['excluded_venues'] == 80 and preview['excluded_businesses'] == 80
        assert client.post('/api/filters/preview', headers=H, json={'mode': 'x', 'activities': ['56']}).status_code == 400
        bad = client.post('/api/analyze', headers=H, data={'demo': 'true', 'filters': '{"activities": "56"}'})
        assert bad.status_code == 400 and 'Invalid filters' in bad.json()['detail']
