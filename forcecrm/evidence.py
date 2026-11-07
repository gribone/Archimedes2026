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
