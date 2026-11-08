from contextlib import closing

import pandas as pd
import pytest

from forcecrm import db
from forcecrm.data import load_universe

COMPANIES = (
    'company_id;vat;company_name_el;legal_type;is_branch;street;street_number;zip_code;city;'
    'municipality_name;prefecture_name;incorporation_date;latitude;longitude\n'
    '10;047067049;Καφετέρια Άλφα;ΙΚΕ;0;ΟΔΟΣ;1;10000;ΑΘΗΝΑ;ΑΘΗΝΑΙΩΝ / ΚΕΝΤΡΙΚΟΥ;ΑΤΤΙΚΗΣ;2020-01-01;37,98;23,72\n'
    '11;000000002;Ταβέρνα Βήτα;ΑΤΟΜΙΚΗ;1;;;54000;ΘΕΣΣΑΛΟΝΙΚΗ;ΘΕΣΣΑΛΟΝΙΚΗΣ / ΘΕΣΣΑΛΟΝΙΚΗΣ;ΘΕΣΣΑΛΟΝΙΚΗΣ;2019-05-05;NULL;NULL\n'
    '12;000000003;Snack Γάμμα;ΟΕ;0;ΠΛΑΤΕΙΑ;2;84100;ΣΥΡΟΣ;ΣΥΡΟΥ / ΚΥΚΛΑΔΩΝ;ΚΥΚΛΑΔΩΝ;2021-06-01;37,44;24,94\n')
ACTIVITIES = ('company_id;activity_id;activity_type;date_from\n'
              '10;56300007;Κύρια;2020-01-01\n11;56110100;Κύρια;2019-05-05\n12;56110111;Κύρια;2021-06-01\n'
              '12;56300007;Δευτερεύουσα;2021-06-01\n')
DESCRIPTIONS = ('activity_id;description\n56300007;ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ΚΑΦΕΤΕΡΙΑ\n'
                '56110100;ΥΠΗΡΕΣΙΕΣ ΠΑΡΟΧΗΣ ΓΕΥΜΑΤΩΝ ΜΕ ΠΛΗΡΗ ΕΞΥΠΗΡΕΤΗΣΗ ΕΣΤΙΑΤΟΡΙΟΥ\n'
                '56110111;ΥΠΗΡΕΣΙΕΣ ΠΑΡΟΧΗΣ ΓΕΥΜΑΤΩΝ ΚΑΙ ΠΟΤΩΝ ΑΠΟ ΣΝΑΚ ΜΠΑΡ\n')


@pytest.fixture
def universe(tmp_path):
    for name, text in [('vwHackCompany.csv', COMPANIES), ('vwHackCompanyActivity.csv', ACTIVITIES),
                       ('vwHackActivities.csv', DESCRIPTIONS)]:
        (tmp_path / name).write_text(text, encoding='utf-8-sig')
    return tmp_path


def test_import_is_idempotent_and_follows_csv_hashes(universe):
    with closing(db.connect(universe)) as con:
        assert db.ensure_universe(con, universe) is True
        assert db.ensure_universe(con, universe) is False  # unchanged CSVs: no re-import
        (universe / 'vwHackActivities.csv').write_text(DESCRIPTIONS + '56210000;ΤΡΟΦΟΔΟΣΙΑ\n', encoding='utf-8-sig')
        assert db.ensure_universe(con, universe) is True
        assert con.execute('SELECT COUNT(*) FROM activity_descriptions').fetchone()[0] == 4


def test_db_and_csv_paths_yield_identical_frames(universe):
    from_csv = load_universe(universe, use_db=False)
    from_db = load_universe(universe, use_db=True)
    for x, y in zip(from_csv[:3], from_db[:3]):
        pd.testing.assert_frame_equal(x, y)
    assert from_csv[3] == from_db[3]
    assert from_db[0].latitude.isna().sum() == 1  # NULL survives the round trip


def saved_run(universe):
    companies = load_universe(universe)[0]
    profiles = [{'profile_id': 'ICP-01', 'title': 'Τίτλος', 'customer_count': 2, 'revenue_share': .6,
                 'stability': .9, 'confidence_score': .7,
                 'significant_features': [{'feature': 'legal_type:ΙΚΕ', 'label': 'Νομική μορφή: ΙΚΕ', 'count': 2,
                                           'total': 2, 'profile_rate': 1.0, 'universe_rate': .3, 'lift': 3.3,
                                           'p_value': .01, 'q_value': .02}]},
                {'profile_id': 'ICP-02', 'title': 'Άλλο', 'significant_features': []}]
    report = {'demo': False, 'portfolio_sha256': 'abc', 'input_sha256': {'x': 'y'}}
    assignments = pd.DataFrame({'company_id': ['10', '11', '12'], 'profile_id': ['ICP-01', 'NONE', 'ICP-02'],
                                'score': [.9, .1, .8]})
    lookalikes = pd.DataFrame({'profile_id': ['ICP-01', 'ICP-02', 'ICP-01'], 'company_id': ['11', '12', '10'],
                               'rank': [1, 2, 3], 'score': [.9, .8, .7], 'rationale': ['a', 'b', 'c']})
    con = db.connect(universe)
    db.save_run(con, 'run1', profiles, report, assignments, lookalikes, companies, params={'demo': False})
    return con


def test_run_store_pagination_filters_and_search(universe):
    with closing(saved_run(universe)) as con:
        page = db.query_lookalikes(con, 'run1', offset=0, limit=2)
        assert page['total'] == 3 and [r['rank'] for r in page['rows']] == [1, 2]
        assert [r['rank'] for r in db.query_lookalikes(con, 'run1', offset=2, limit=2)['rows']] == [3]
        assert db.query_lookalikes(con, 'run1', profile='ICP-01')['total'] == 2
        # Rank within the profile (A7) is the same whether or not the page is filtered.
        assert [r['profile_rank'] for r in db.query_lookalikes(con, 'run1')['rows']] == [1, 1, 2]
        assert [r['profile_rank'] for r in db.query_lookalikes(con, 'run1', profile='ICP-01')['rows']] == [1, 2]
        # Case- and accent-insensitive Greek search over name, city, municipality and ΑΦΜ.
        assert [r['vat'] for r in db.query_lookalikes(con, 'run1', q='ταβερνα')['rows']] == ['000000002']
        assert db.query_lookalikes(con, 'run1', q='σύρος')['total'] == 1
        assert db.query_lookalikes(con, 'run1', q='047067049')['total'] == 1
        assert db.query_lookalikes(con, 'run1', q='%')['total'] == 0  # LIKE wildcards are escaped
        assert db.list_runs(con)[0]['run_id'] == 'run1'
        assert con.execute('SELECT COUNT(*) FROM profile_traits').fetchone()[0] == 1
        with pytest.raises(ValueError):
            db.query_lookalikes(con, 'run1', live='bogus')
    # Saving the same run again replaces it instead of duplicating rows.
    with closing(saved_run(universe)) as con:
        assert con.execute("SELECT COUNT(*) FROM lookalikes WHERE run_id = 'run1'").fetchone()[0] == 3
        assert con.execute('SELECT COUNT(*) FROM profiles').fetchone()[0] == 2


def test_live_filters_join_stored_gemi_statuses(universe):
    with closing(saved_run(universe)) as con:
        db.put_gemi_status(con, '000000002', {'status': 'inactive', 'description': 'Διαγραμμένη', 'checked_at': 'x'})
        db.put_gemi_status(con, '000000003', {'status': 'active', 'checked_at': 'x'})
        assert [r['rank'] for r in db.query_lookalikes(con, 'run1', live='hide-inactive')['rows']] == [2, 3]
        assert [r['rank'] for r in db.query_lookalikes(con, 'run1', live='active')['rows']] == [2]
        first = db.query_lookalikes(con, 'run1', limit=1)['rows'][0]
        assert first['gemi_status'] == 'inactive' and first['gemi_description'] == 'Διαγραμμένη'


def test_company_detail_includes_activities_and_status(universe):
    with closing(saved_run(universe)) as con:
        detail = db.company_detail(con, '12')
        assert detail['vat'] == '000000003' and detail['latitude'] == 37.44
        assert [a['activity_id'] for a in detail['activities']] == ['56110111', '56300007']
        assert detail['activities'][0]['description'].startswith('ΥΠΗΡΕΣΙΕΣ')
        assert db.company_detail(con, '999') is None
        assert [c['company_id'] for c in db.search_companies(con, 'καφετερια')] == ['10']


def test_api_pages_runs_and_companies(universe, monkeypatch):
    from fastapi.testclient import TestClient
    from forcecrm import server
    saved_run(universe).close()
    monkeypatch.setattr(server, 'ROOT', universe)
    with TestClient(server.app) as client:
        body = client.get('/api/runs/run1/lookalikes', params={'limit': 2, 'offset': 1}).json()
        assert body['total'] == 3 and [r['rank'] for r in body['rows']] == [2, 3]
        assert client.get('/api/runs/run1/lookalikes', params={'profile': 'ICP-02'}).json()['total'] == 1
        assert client.get('/api/runs/missing/lookalikes').status_code == 404
        assert client.get('/api/runs/run1/lookalikes', params={'limit': 0}).status_code == 400
        assert client.get('/api/runs/run1/lookalikes', params={'live': 'bogus'}).status_code == 400
        assert client.get('/api/runs').json()['runs'][0]['run_id'] == 'run1'
        assert client.get('/api/companies/10').json()['company_name_el'] == 'Καφετέρια Άλφα'
        assert client.get('/api/companies/404').status_code == 404
        assert client.get('/api/companies', params={'q': 'θεσσαλονικη'}).json()['companies'][0]['company_id'] == '11'


def test_export_matches_the_filtered_table(universe):
    with closing(saved_run(universe)) as con:
        for kwargs in [{}, {'profile': 'ICP-01'}, {'q': 'ταβερνα'}, {'q': '%'}]:
            table = db.query_lookalikes(con, 'run1', offset=0, limit=200, **kwargs)
            export = db.export_lookalikes(con, 'run1', **kwargs)
            assert len(export) == table['total']
            assert [(r['rank'], r['profile_rank']) for r in export] == \
                   [(r['rank'], r['profile_rank']) for r in table['rows']]
        with pytest.raises(ValueError):
            db.export_lookalikes(con, 'run1', live='bogus')
