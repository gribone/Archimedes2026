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


def test_sentence_checks_reject_known_llm_errors():
    from forcecrm.narrative import check_sentence
    fact = ('30 από τους 126 πελάτες (24%) βρίσκονται στην Περιφερειακή Ενότητα ΗΡΑΚΛΕΙΟΥ, '
            '8,6 φορές συχνότερα από ό,τι στο σύνολο των επιχειρήσεων εστίασης.')
    good = 'Στην Περιφερειακή Ενότητα Ηρακλείου βρίσκονται 30 από τους 126 πελάτες (24%), 8,6 φορές συχνότερα.'
    assert check_sentence(good, fact).startswith('Στην')
    for bad in ['Το 30% των πελατών βρίσκεται στην Π.Ε. ΗΡΑΚΛΕΙΟΥ, 8,6 φορές συχνότερα.',   # invented percentage
                'Στο Ηράκλειο βρίσκονται 30 από τους 126 πελάτες (24%).',                     # changed name
                'Στην Περιφερειακή Ενότητα ΗΡΑΚΛΕΙΟΥ βρίσκονται περίπου 30 από 126 άτομα.',  # hedging, «άτομα»
                'Στην Περιφερειακή Ενότητα ΗΡΑΚΛΕΙΟΥ κυριαρχούν οι πελάτες, 8,6 φορές συχνότερα.',  # dropped count
                'Στα 126 πελάτες, 30 βρίσκονται στην Περιφερειακή Ενότητα ΗΡΑΚΛΕΙΟΥ.']:      # wrong gender
        with pytest.raises(ValueError):
            check_sentence(bad, fact)


def test_description_states_quality_stability_and_confidence():
    from forcecrm.narrative import narrate
    p = {'customer_count': 10, 'venue_count': 10, 'revenue_share': .2, 'significant_features': [],
         'stability': .9, 'stability_detail': {'mean': .9, 'p10': .7}, 'confidence_score': .55, 'quality': 'limited'}
    result = narrate(p, None, demo=True)
    assert result['description'] == ' '.join(c['sentence'] for c in result['claims'])
    last = result['claims'][-1]
    assert last['fact_ids'] == ['quality'] and 'περιορισμένη' in last['sentence']
    assert '90%' in last['sentence'] and '70%' in last['sentence'] and '55%' in last['sentence']


def test_feature_selection_avoids_restating_the_same_place():
    from forcecrm.narrative import select_features
    row = lambda f: {'feature': f, 'label': f, 'count': 5, 'total': 10, 'lift': 2.0, 'q_value': .01}
    picked = select_features([row('prefecture_name:ΧΑΝΙΩΝ'), row('municipality_name:ΧΑΝΙΩΝ / ΧΑΝΙΩΝ'),
                              row('prefecture_name:ΚΥΚΛΑΔΩΝ'), row('prefecture_name:ΡΟΔΟΥ'),
                              row('hotel_band:≥10'), row('legal_type:ΙΚΕ')])
    names = [f['feature'] for f in picked]
    assert 'municipality_name:ΧΑΝΙΩΝ / ΧΑΝΙΩΝ' not in names and 'prefecture_name:ΡΟΔΟΥ' not in names
    assert names[-1] == 'legal_type:ΙΚΕ' and len(names) == 4


def test_portfolio_traits_and_activity_mix_count_vats_not_venues():
    from forcecrm.evidence import portfolio_traits, activity_mix
    vats = [f'{i:09d}' for i in range(200)]
    c = pd.DataFrame({'vat': vats + ['000000000'] * 5})  # five extra branches of one customer
    codes = [['56300007'] if i < 40 else ['56110100'] for i in range(200)] + [['56300007']] * 5
    traits = [{f'primary_activity:{x[0]}', 'prefecture_name:A'} for x in codes]
    d = pd.DataFrame({'activity_id': ['56300007', '56110100'], 'description': ['ΚΑΦΕΤΕΡΙΑ', 'ΕΣΤΙΑΤΟΡΙΟ']})
    customers = set(vats[:30]) | set(vats[100:110])  # 30 of 40 cafes, 10 of 160 restaurants
    rows, tests = portfolio_traits(c, traits, customers, d)
    assert [r['feature'] for r in rows] == ['primary_activity:56300007']  # geography is excluded
    assert rows[0]['count'] == 30 and rows[0]['total'] == 40
    mask = c.vat.isin(customers).to_numpy()
    mix = activity_mix(c, codes, [mask], d)[0]
    assert mix[0]['activity_id'] == '56300007' and mix[0]['count'] == 30 and mix[0]['total'] == 40
    assert abs(mix[0]['universe_rate'] - 40 / 200) < 1e-9


def test_activity_sentence_precedes_quality_sentence():
    from forcecrm.narrative import narrate
    p = {'customer_count': 10, 'venue_count': 10, 'revenue_share': .2, 'significant_features': [],
         'stability': .9, 'stability_detail': {'mean': .9, 'p10': .7}, 'confidence_score': .55, 'quality': 'strong',
         'activity_mix': [{'label': 'ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ΚΑΦΕΤΕΡΙΑ', 'profile_rate': .31, 'universe_rate': .126}]}
    claims = narrate(p, None, demo=True)['claims']
    assert claims[-2]['fact_ids'] == ['activity_mix'] and '«ΚΑΦΕΤΕΡΙΑ» 31%' in claims[-2]['sentence']
    assert claims[-1]['fact_ids'] == ['quality']


def test_singular_legal_form_with_plural_count_is_rejected():
    from forcecrm.narrative import check_sentence
    fact = ('74 από τους 100 πελάτες (74%) είναι Ιδιωτικές Κεφαλαιουχικές Εταιρείες (ΙΚΕ), '
            '4,8 φορές συχνότερα από ό,τι στο σύνολο των επιχειρήσεων εστίασης.')
    with pytest.raises(ValueError):
        check_sentence('Η Ιδιωτική Κεφαλαιουχική Εταιρεία (ΙΚΕ) είναι 74 από τους 100 πελάτες (74%), 4,8 φορές συχνότερα.', fact)
    assert check_sentence('Ιδιωτικές Κεφαλαιουχικές Εταιρείες (ΙΚΕ) είναι 74 από τους 100 πελάτες (74%), 4,8 φορές συχνότερα.', fact)


def test_cluster_export_lists_prospects_and_customers_per_profile(tmp_path):
    import json
    import openpyxl
    from forcecrm.clusters import build_clusters, export_clusters
    companies = pd.DataFrame({'company_id': ['1', '2', '3', '4', '5', '6'],
                              'vat': ['000000001', '000000001', '000000002', '000000003', '000000004', '000000005'],
                              'company_name_el': ['A', 'A2', 'B', 'C', 'D', 'E'], 'city': ['X'] * 6,
                              'municipality_name': ['M'] * 6})
    profiles = [{'profile_id': 'ICP-01', 'title': 'T1', 'description': 'D1', 'ai_generated': True},
                {'profile_id': 'ICP-02', 'title': 'T2', 'description': 'D2', 'ai_generated': True}]
    # Customer 000000001 has two branches in ICP-01; customer 000000002 is in ICP-02.
    fit = {'indices': np.array([0, 1, 2]), 'labels': np.array([0, 0, 1]),
           'vats': np.array(['000000001', '000000001', '000000002'])}
    assignments = pd.DataFrame({'company_id': ['4', '5', '6'], 'profile_id': ['ICP-01', 'NONE', 'ICP-02'], 'score': [.9, .5, .8]})
    lookalikes = pd.DataFrame({'profile_id': ['ICP-01', 'ICP-02', 'ICP-01'], 'company_id': ['4', '6', '5'],
                               'rank': [1, 2, 3], 'score': [.9, .8, .5], 'rationale': ['r1', 'r2', 'r3']})
    table = build_clusters(companies, profiles, fit, assignments, lookalikes)
    prospects = table[table.role == 'prospect'].set_index('vat')
    assert prospects.loc['000000003', 'profile_rank'] == 1 and prospects.loc['000000004', 'profile_rank'] == 2
    assert bool(prospects.loc['000000003', 'within_profile']) and not bool(prospects.loc['000000004', 'within_profile'])
    customers = table[table.role == 'customer']
    assert customers.vat.tolist().count('000000001') == 1  # one row per ΑΦΜ despite two branches
    export_clusters({'clusters': table, 'profiles': profiles}, tmp_path)
    data = json.loads((tmp_path / 'clusters.json').read_text('utf-8'))
    assert data[0]['customers'] == ['000000001'] and [p['vat'] for p in data[0]['prospects']] == ['000000003', '000000004']
    sheet = openpyxl.load_workbook(tmp_path / 'clusters.xlsx')['ICP-01']
    assert sheet['C5'].value == '000000003' and sheet['A2'].value == 'D1'
