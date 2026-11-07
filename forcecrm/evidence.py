from collections import defaultdict
import numpy as np
from scipy.stats import fisher_exact

LABELS = {'hotel_band': 'Ετήσιες διανυκτερεύσεις ξενοδοχείων ανά κάτοικο στον δήμο',
          'osm_amenity': 'Κατηγορία καταστήματος στο OpenStreetMap',
          'primary_activity': 'Κύρια δραστηριότητα', 'legal_type': 'Νομική μορφή',
          'prefecture_name': 'Περιφερειακή ενότητα', 'municipality_name': 'Δήμος',
          'incorporation_year': 'Έτος ίδρυσης', 'is_branch': 'Υποκατάστημα'}


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(1, adjusted)
    return result


def significant_features(companies, traits, member_masks, descriptions):
    # Independent observational units are VATs. Multiple venues cannot multiply evidence.
    universe = set(companies.vat)
    feature_vats = defaultdict(set)
    for vat, values in zip(companies.vat, traits):
        for value in values:
            if value.partition(':')[2]:
                feature_vats[value].add(vat)
    names = dict(zip(descriptions.activity_id, descriptions.description))
    tests = []
    for profile, mask in enumerate(member_masks):
        members = set(companies.loc[mask, 'vat'])
        other = universe - members
        for feature, vats in feature_vats.items():
            if len(vats) < 5 or not other:
                continue
            inside, outside = len(members & vats), len(other & vats)
            p = fisher_exact([[inside, len(members) - inside], [outside, len(other) - outside]], alternative='greater').pvalue
            lift = (inside / max(1, len(members))) / (len(vats) / len(universe))
            kind, _, value = feature.partition(':')
            label = names.get(value, value) if kind == 'primary_activity' else value
            if kind == 'is_branch':
                label = 'Ναι' if value == '1' else 'Όχι'
            tests.append({'profile': profile, 'feature': feature, 'label': f'{LABELS[kind]}: {label}',
                          'count': inside, 'total': len(members), 'profile_rate': inside / max(1, len(members)),
                          'universe_rate': len(vats) / len(universe), 'lift': float(lift), 'p_value': float(p)})
    qvalues = bh_adjust([t['p_value'] for t in tests])
    output = [[] for _ in member_masks]
    for row, q in zip(tests, qvalues):
        row['q_value'] = float(q)
        profile = row.pop('profile')
        if q <= .05 and row['count'] >= 5 and row['lift'] >= 1.2:
            output[profile].append(row)
    for rows in output:
        rows.sort(key=lambda row: (row['q_value'], -row['lift']))
    return output, len(tests)


BUSINESS_KINDS = ('primary_activity', 'legal_type', 'incorporation_year', 'is_branch', 'osm_amenity')


def portfolio_traits(companies, traits, customer_vats, descriptions):
    """Non-geographic traits over-represented in the whole portfolio versus the universe.

    Profiles in this data split mostly by place, so business traits shared by customers across
    regions (e.g. a legal form or an activity such as school canteens) would otherwise stay hidden.
    Uses the same VAT-level Fisher test, BH correction and thresholds as the per-profile evidence.
    """
    business = [{t for t in values if t.partition(':')[0] in BUSINESS_KINDS} for values in traits]
    rows, tests = significant_features(companies, business, [companies.vat.isin(customer_vats).to_numpy()], descriptions)
    return rows[0], tests


def activity_mix(companies, primary_codes, member_masks, descriptions, top=3):
    """Descriptive share of each profile's customers (VATs) per primary activity, versus the universe."""
    names = dict(zip(descriptions.activity_id, descriptions.description))
    codes_by_vat = defaultdict(set)
    for vat, codes in zip(companies.vat, primary_codes):
        codes_by_vat[vat].update(codes)
    universe = defaultdict(int)
    for codes in codes_by_vat.values():
        for code in codes:
            universe[code] += 1
    total_vats = len(codes_by_vat)
    result = []
    for mask in member_masks:
        members = set(companies.loc[mask, 'vat'])
        counts = defaultdict(int)
        for vat in members:
            for code in codes_by_vat[vat]:
                counts[code] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]
        result.append([{'activity_id': code, 'label': names.get(code, code), 'count': n, 'total': len(members),
                        'profile_rate': n / max(1, len(members)), 'universe_rate': universe[code] / total_vats,
                        'lift': (n / max(1, len(members))) / (universe[code] / total_vats)}
                       for code, n in ranked])
    return result
