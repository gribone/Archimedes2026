from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import normalize

from forcecrm.data import normalize_vat, read_portfolio
from forcecrm.clustering import discover, stability
from forcecrm.evidence import bh_adjust, significant_features
from forcecrm.narrative import fact_catalog, validate_draft
from forcecrm.pipeline import fit_portfolio, rank_candidates, export_result


def workbook(rows):
    stream = BytesIO()
    pd.DataFrame(rows).to_excel(stream, index=False)
    stream.seek(0)
    return stream


def test_vat_and_greek_decimal_import():
    companies = pd.DataFrame({'vat': [f'{i:09}' for i in range(1, 6)]})
    source = workbook({'ΑΦΜ': [1, 2, 3, 4, 5], 'Τζίρος (€)': ['1.234,50', '0', '12.5', '1,25', '50']})
    result = read_portfolio(source, companies)
    assert result.iloc[0].vat == '000000001'
    assert result.iloc[0].revenue == 1234.5
    assert result.revenue.sum() == 1298.25
    assert normalize_vat('EL047067049') == '047067049'
    with pytest.raises(ValueError):
        normalize_vat('12bad')


@pytest.mark.parametrize('revenues', [[-1, 1, 2, 3, 4], ['NaN', 1, 2, 3, 4], [0]*5])
def test_invalid_revenue_rejected(revenues):
    companies = pd.DataFrame({'vat': [f'{i:09}' for i in range(1, 6)]})
    with pytest.raises(ValueError):
        read_portfolio(workbook({'vat': [1, 2, 3, 4, 5], 'revenue': revenues}), companies)


def test_duplicates_and_unknown_customers_rejected():
    companies = pd.DataFrame({'vat': [f'{i:09}' for i in range(1, 6)]})
    for vats in [[1, 1, 2, 3, 4], [1, 2, 3, 4, 99]]:
        with pytest.raises(ValueError):
            read_portfolio(workbook({'vat': vats, 'revenue': [100]*5}), companies)


def test_homogeneous_portfolio_returns_one_profile():
    X = normalize(np.ones((30, 4))).astype('float32')
    centers, labels, _ = discover(X, np.ones(30), np.arange(30))
    assert len(centers) == 1
    assert set(labels) == {0}
    scores, ari = stability(X, np.ones(30), np.arange(30), centers, labels, repetitions=3)
    assert scores[0]['mean'] == 1
    assert ari == 1


def test_separated_groups_are_recovered():
    rng = np.random.default_rng(3)
    X = normalize(np.r_[rng.normal([1, 0, 0], .015, (20, 3)), rng.normal([0, 1, 0], .015, (20, 3))])
    centers, labels, _ = discover(X, np.ones(40), np.arange(40))
    assert len(centers) == 2
    assert len(set(labels[:20])) == 1
    assert labels[0] != labels[20]


def test_branch_weights_and_candidate_exclusion():
    c = pd.DataFrame({'company_id': [str(i) for i in range(9)], 'vat': ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'f', 'g']})
    portfolio = pd.DataFrame({'vat': list('abcde'), 'revenue': [100]*5})
    X = normalize(np.array([[1, 0]]*6 + [[.99, .01], [.98, .02], [0, 1]], dtype=float))
    fit = fit_portfolio(X, c, portfolio)
    assert fit['weights'][0] + fit['weights'][1] == pytest.approx(fit['weights'][2])
    assert not fit['accepted'][-1]
    ranked = rank_candidates(c, fit, set(portfolio.vat))
    assert len(ranked) == 2
    assert set(ranked.company_id) == {'6', '8'}
    assert ranked['rank'].tolist() == [1, 2]
    assert 'Διερευνητική' in ranked.iloc[-1].rationale


def test_global_bh_and_minimum_independent_support():
    assert np.allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])
    c = pd.DataFrame({'vat': ['same']*10 + [str(i) for i in range(100)]})
    traits = [{'legal_type:rare'}]*10 + [{'legal_type:common'}]*100
    mask = np.array([True]*10 + [False]*100)
    features, _ = significant_features(c, traits, [mask], pd.DataFrame(columns=['activity_id','description']))
    assert not features[0]  # Ten branches still represent only one independent customer.


def test_grounding_rejects_fabrication():
    p = {'customer_count': 10, 'revenue_share': .2, 'significant_features': []}
    facts = fact_catalog(p)
    valid = {'sentences': [{'fact_id': f['id'], 'variant': 0} for f in facts]}
    text, claims = validate_draft(valid, facts)
    assert len(claims) == 3 and '10' in text
    valid['sentences'][-1]['fact_id'] = 'high_creditworthiness'
    with pytest.raises(ValueError):
        validate_draft(valid, facts)


def test_export_contract(tmp_path):
    result = {'profiles': [{'profile_id': 'ICP-01', 'title': 'Εστίαση'}], 'report': {'demo': True},
              'assignments': pd.DataFrame([['1', 'NONE', .2]], columns=['company_id', 'profile_id', 'score']),
              'lookalikes': pd.DataFrame([['ICP-01', '1', 1, .2, 'Δοκιμή']], columns=['profile_id','company_id','rank','score','rationale'])}
    export_result(result, tmp_path)
    assert 'Εστίαση' in (tmp_path / 'profiles.json').read_text('utf-8')
    assert pd.read_csv(tmp_path / 'lookalikes.csv').columns.tolist() == ['profile_id','company_id','rank','score','rationale']


def test_local_api_rejects_cross_site_mutations():
    from fastapi.testclient import TestClient
    from forcecrm.server import app
    with TestClient(app) as client:
        assert client.post('/api/analyze', data={'demo': 'true'}).status_code == 403
        assert client.post('/api/analyze', data={'demo': 'false'}, headers={'X-ForceCRM':'local'}).status_code == 400
        assert client.get('/').status_code == 200
        assert client.get('/api/template').status_code == 200


def test_many_irrelevant_secondary_codes_do_not_penalize_primary(monkeypatch, tmp_path):
    from forcecrm import features
    d = pd.DataFrame({'activity_id':['56110000','56999999'], 'description':['restaurant','noise']})
    monkeypatch.setattr(features, 'description_embeddings', lambda *args: (np.eye(2,dtype='float32'), {'model':'test'}))
    c = pd.DataFrame({'company_id':['1','2'], 'vat':['000000001','000000002'],
        'legal_type':['IKE','IKE'], 'prefecture_name':['ATTICA','ATTICA'], 'municipality_name':['ATHENS','ATHENS'],
        'incorporation_date':['2021-01-01','2021-01-01'], 'latitude':[38.,38.], 'longitude':[23.,23.], 'is_branch':['0','0']})
    a = pd.DataFrame({'company_id':['1','2']+['2']*30, 'activity_id':['56110000']*2+['56999999']*30,
                       'kind':['primary']*2+['secondary']*30})
    X,*_ = features.build_features(c,a,d,tmp_path)
    assert np.allclose(X[0],X[1])
