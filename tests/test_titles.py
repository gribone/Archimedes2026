import pytest

from forcecrm.titles import activity_families, default_choice, family_name, title_options, validate_choice


def feature(name, rate, lift=3.0):
    return {'feature': name, 'profile_rate': rate, 'lift': lift}


PROFILE = {
    'significant_features': [
        feature('prefecture_name:ΧΑΝΙΩΝ', .17, 8.3),
        feature('hotel_band:≥10', .70, 2.9),
        feature('municipality_name:ΓΛΥΦΑΔΑΣ / ΝΟΤΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ', .07, 9.0),
        feature('legal_type:ΙΚΕ', .74, 4.8),
    ],
    'activity_families': [{'family': 'καφετέριες', 'share': .31}, {'family': 'αναψυκτήρια', 'share': .19},
                     {'family': 'μπαρ', 'share': .05}],
}


def test_options_lead_with_widest_coverage_and_drop_small_traits():
    places, activities, legal = title_options(PROFILE)
    assert [p['phrase'] for p in places] == ['Τουριστικοί δήμοι', 'Π.Ε. ΧΑΝΙΩΝ']
    assert [a['phrase'] for a in activities] == ['καφετέριες', 'αναψυκτήρια']
    assert [l['phrase'] for l in legal] == ['κυρίως ΙΚΕ']


def test_valid_choice_assembles_grounded_title():
    options = title_options(PROFILE)
    choice = {'second_place': 'Π.Ε. ΧΑΝΙΩΝ', 'activities': ['καφετέριες'], 'legal': 'κυρίως ΙΚΕ'}
    assert validate_choice(choice, *options) == 'Τουριστικοί δήμοι & Π.Ε. ΧΑΝΙΩΝ · καφετέριες · κυρίως ΙΚΕ'
    assert validate_choice(default_choice(*options), *options) == \
        'Τουριστικοί δήμοι · καφετέριες, αναψυκτήρια · κυρίως ΙΚΕ'


@pytest.mark.parametrize('choice', [
    {'second_place': 'Δήμος ΓΛΥΦΑΔΑΣ', 'activities': ['καφετέριες'], 'legal': 'none'},
    {'second_place': 'none', 'activities': ['μπαρ'], 'legal': 'none'},
    {'second_place': 'none', 'activities': ['καφετέριες', 'καφετέριες'], 'legal': 'none'},
    {'second_place': 'none', 'activities': [], 'legal': 'none'},
    {'second_place': 'none', 'activities': ['καφετέριες'], 'legal': 'κυρίως ΑΕ'},
    {'second_place': 'Τουριστικοί δήμοι', 'activities': ['καφετέριες'], 'legal': 'none'},
    'free text title',
])
def test_unsupported_choices_are_rejected(choice):
    with pytest.raises(ValueError):
        validate_choice(choice, *title_options(PROFILE))


def test_no_significant_place_or_dominant_activity_has_safe_defaults():
    profile = {'significant_features': [], 'activity_families': [{'family': 'μπαρ', 'share': .05}]}
    options = title_options(profile)
    assert validate_choice(default_choice(*options), *options) == 'Πανελλαδική κατανομή · μικτή εστίαση'


def test_activity_mix_counts_each_vat_once():
    mix = activity_families([['56300007'], ['56300008'], ['56110108']], ['1', '1', '2'], {})
    assert mix[0] == {'family': 'καφετέριες', 'share': .5, 'count': 1}
    assert family_name('99999999', 'ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ΚΑΤΙ, ΜΕ ΑΛΛΟ') == 'κατι'
