"""The staged pipeline: deterministic keys, cache reuse, and identical results with or without cache.
Runs on a small synthetic universe in demo mode (lexical embeddings, template text, no Ollama)."""
import numpy as np
import pandas as pd
import pytest

from forcecrm.pipeline import run_analysis, make_runner
from forcecrm.stages import STAGES, ORDER

CODES = {'56101100': 'ΥΠΗΡΕΣΙΕΣ ΕΣΤΙΑΤΟΡΙΟΥ ΜΕ ΠΛΗΡΗ ΕΞΥΠΗΡΕΤΗΣΗ',
         '56300007': 'ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ΚΑΦΕΤΕΡΙΑ',
         '56110111': 'ΥΠΗΡΕΣΙΕΣ ΠΑΡΟΧΗΣ ΓΕΥΜΑΤΩΝ ΚΑΙ ΠΟΤΩΝ ΑΠΟ ΣΝΑΚ ΜΠΑΡ',
         '56210000': 'ΥΠΗΡΕΣΙΕΣ ΤΡΟΦΟΔΟΣΙΑΣ ΓΙΑ ΕΚΔΗΛΩΣΕΙΣ'}
PLACES = [('ΑΤΤΙΚΗΣ', 'ΑΘΗΝΑΙΩΝ', 37.98, 23.73), ('ΚΥΚΛΑΔΩΝ', 'ΘΗΡΑΣ', 36.42, 25.43),
          ('ΘΕΣΣΑΛΟΝΙΚΗΣ', 'ΘΕΣΣΑΛΟΝΙΚΗΣ', 40.64, 22.94)]


@pytest.fixture(scope='module')
def universe(tmp_path_factory):
    root = tmp_path_factory.mktemp('universe')
    rng = np.random.default_rng(7)
    companies, activities = [], []
    for i in range(240):
        prefecture, municipality, lat, lon = PLACES[i % 3]
        code = list(CODES)[(i // 3) % 4]
        companies.append({'company_id': str(1000 + i), 'vat': f'{100000 + i:09d}',
                          'company_name_el': f'ΚΑΤΑΣΤΗΜΑ {i}', 'legal_type': ['ΙΚΕ', 'ΑΤΟΜΙΚΗ', 'ΟΕ'][i % 7 % 3],
                          'is_branch': '0', 'street': 'ΟΔΟΣ', 'street_number': '1', 'zip_code': '10000',
                          'city': municipality, 'municipality_name': municipality, 'prefecture_name': prefecture,
                          'incorporation_date': f'{2020 + i % 5}-03-01',
                          'latitude': f'{lat + rng.normal(0, .05):.5f}'.replace('.', ','),
                          'longitude': f'{lon + rng.normal(0, .05):.5f}'.replace('.', ',')})
        activities.append({'company_id': str(1000 + i), 'activity_id': code, 'activity_type': 'Κύρια', 'date_from': '2020-01-01'})
        activities.append({'company_id': str(1000 + i), 'activity_id': '56210000', 'activity_type': 'Δευτερεύουσα', 'date_from': '2020-01-01'})
    tables = {'vwHackCompany.csv': pd.DataFrame(companies), 'vwHackCompanyActivity.csv': pd.DataFrame(activities),
              'vwHackActivities.csv': pd.DataFrame({'activity_id': list(CODES), 'description': list(CODES.values())})}
    for name, frame in tables.items():
        frame.to_csv(root / name, sep=';', index=False, encoding='utf-8-sig')
    # Customers: mostly the island and Attica venues, with uneven revenue.
    vats = [c['vat'] for i, c in enumerate(companies) if i % 3 in (0, 1) and i % 4 == 0]
    portfolio = pd.DataFrame({'vat': sorted(vats), 'revenue': [1000.0 + 37 * k for k in range(len(vats))]})
    return root, portfolio


def analyze(root, portfolio, **options):
    return run_analysis(root, portfolio, demo=True, bootstrap=3, **options)


def statuses(result):
    return {s['stage']: s['status'] for s in result['report']['stages']}


def test_stage_graph_is_ordered_and_complete():
    seen = set()
    for name in ORDER:
        assert set(STAGES[name].needs) <= seen, name  # every dependency comes earlier
        seen.add(name)
    assert [s for s in ORDER if not STAGES[s].deterministic] == ['narrate']


def test_two_runs_give_identical_keys_and_outputs(universe):
    root, portfolio = universe
    a, b = analyze(root, portfolio, cache=False), analyze(root, portfolio, cache=False)
    assert [s['key'] for s in a['report']['stages']] == [s['key'] for s in b['report']['stages']]
    pd.testing.assert_frame_equal(a['lookalikes'], b['lookalikes'])
    pd.testing.assert_frame_equal(a['assignments'], b['assignments'])
    assert a['profiles'] == b['profiles']


def test_cached_run_matches_and_skips_every_stage(tmp_path, universe):
    root, portfolio = universe
    fresh = analyze(root, portfolio, cache=False)
    first = analyze(root, portfolio)
    second = analyze(root, portfolio)
    assert set(statuses(second).values()) == {'cached'}
    for result in (first, second):
        pd.testing.assert_frame_equal(result['lookalikes'], fresh['lookalikes'])
        pd.testing.assert_frame_equal(result['assignments'], fresh['assignments'])
        assert result['profiles'] == fresh['profiles']
    assert second['report']['stage_notes']['narrate'] == {'profiles': len(fresh['profiles']),
                                                          'reused_text': len(fresh['profiles'])}


def test_new_portfolio_reuses_the_universe_side(universe):
    root, portfolio = universe
    analyze(root, portfolio)
    changed = portfolio.assign(revenue=portfolio.revenue[::-1].to_numpy())
    status = statuses(analyze(root, changed))
    assert all(status[s] == 'cached' for s in ('load', 'enrich', 'represent'))
    assert status['discover'] == 'computed'


def test_forcing_narration_recomputes_only_narration(universe):
    root, portfolio = universe
    analyze(root, portfolio)
    result = analyze(root, portfolio, force={'narrate'})
    status = statuses(result)
    assert status['narrate'] == 'computed'
    assert all(v == 'cached' for k, v in status.items() if k != 'narrate')
    assert result['report']['stage_notes']['narrate']['reused_text'] == 0


def test_custom_weights_compare_against_cached_default_representation(universe):
    root, portfolio = universe
    analyze(root, portfolio)
    custom = {'activity': .4, 'geography': .4, 'legal_type': .1, 'age': .1}
    result = analyze(root, portfolio, weights=custom)
    represents = [s for s in result['report']['stages'] if s['stage'] == 'represent']
    assert len(represents) == 2 and represents[0]['key'] != represents[1]['key']
    assert represents[1]['status'] == 'cached'  # the default-weights variant from the earlier run
    assert result['report']['stages'][0]['status'] == 'cached'  # load is shared by both variants
    assert all('default_weights_recall' in m for m in result['report']['validation']['metrics'])


def test_stage_runs_alone_without_downstream_work(universe):
    root, portfolio = universe
    runner = make_runner(root, portfolio, demo=True, bootstrap=3, cache=False)
    ranked = runner.get('rank')
    assert len(ranked) and 'narrate' not in {r.stage for r in runner.records}
