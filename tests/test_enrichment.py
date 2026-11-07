import pandas as pd
from forcecrm.enrichment import link_venues, market_context


def test_nearby_duplicates_abstain_and_address_conflict_excludes_candidate():
    c = pd.DataFrame([dict(company_id='1', latitude=38., longitude=24.,
                           company_name_el='Albatross', street='Αθηνάς', street_number='10')])
    first = [1, 38., 24., ['Albatross'], 'Αθηνάς', '10', 'cafe']
    second = [2, 38., 24., ['Albatross'], 'Αθηνάς', '10', 'restaurant']
    assert link_venues(c, [first, second]).status.item() == 'ambiguous_neighbours'
    second[5] = '12'
    assert link_venues(c, [first, second]).osm_id.item() == 'node/1'
    both = pd.concat([c, c.assign(company_id='2')], ignore_index=True)
    assert set(link_venues(both, [first]).status) == {'ambiguous_shared_poi'}
    assert link_venues(c.assign(company_name_el='Επιχείρηση Εστίασης'), [first]).osm_id.isna().all()


def test_municipal_join_skips_pooled_totals():
    c = pd.DataFrame({'municipality_name': ['ΔΡΑΜΑΣ / ΔΡΑΜΑΣ', 'ΔΟΞΑΤΟΥ / ΔΡΑΜΑΣ']})
    result = market_context(c)
    assert result.hotel_nights_2024.iloc[0] == 71170
    assert result.residents_2021.iloc[0] == 55679
    assert pd.isna(result.hotel_band.iloc[1])

