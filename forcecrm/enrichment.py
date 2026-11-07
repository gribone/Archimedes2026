"""Offline venue linkage and municipal hotel-market context."""
import gzip
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SNAPSHOT = Path(__file__).resolve().parents[1] / 'open_data'
EARTH_M = 6371009
LEGAL = {'μονοπροσωπη', 'ικε', 'οε', 'εε', 'αε', 'επε', 'υποκαταστημα',
         'ιδιωτικη', 'κεφαλαιουχικη', 'εταιρεια', 'ανωνυμη', 'ετερορρυθμη', 'ομορρυθμη'}
GENERIC = {'επιχειρηση εστιασησ', 'foodservice business', 'cafe', 'bar', 'restaurant',
           'καφε', 'εστιατοριο', 'ταβερνα', 'καφενειο'}


def clean(value):
    plain = ''.join(c for c in unicodedata.normalize('NFD', str(value).casefold())
                    if not unicodedata.combining(c))
    return ' '.join(re.findall(r'[^\W_]+', plain.replace('.', ''), re.UNICODE))


def municipal_key(value):
    return clean(str(value).split('/')[0]).removeprefix('δημοσ ')


def trade_name(value):
    return ' '.join(t for t in clean(value).split() if t not in LEGAL)


def link_venues(companies, pois):
    gps = np.radians([[p[1], p[2]] for p in pois])
    tree = BallTree(gps, metric='haversine')
    normalized = [{trade_name(n) for n in p[3]} - GENERIC - {''} for p in pois]
    rows = []
    for venue in companies.itertuples():
        name = trade_name(venue.company_name_el)
        street = clean(getattr(venue, 'street', ''))
        number = clean(getattr(venue, 'street_number', ''))
        result = {'company_id': venue.company_id, 'status': 'unmatched'}
        rows.append(result)
        if not np.isfinite([venue.latitude, venue.longitude]).all():
            result['status'] = 'missing_gps'
            continue
        if len(name) < 3 or name in GENERIC:
            result['status'] = 'uninformative_name'
            continue
        ids, distances = tree.query_radius(np.radians([[venue.latitude, venue.longitude]]),
                                            r=120/EARTH_M, return_distance=True)
        candidates = []
        for i, distance in zip(ids[0], distances[0] * EARTH_M):
            p = pois[i]
            similarity = max((SequenceMatcher(None, name, alias).ratio() for alias in normalized[i]), default=0)
            street_agrees = bool(street and street == clean(p[4]))
            number_agrees = bool(number and number == clean(p[5]))
            if street_agrees and number and p[5] and not number_agrees:
                continue
            score = .78 * similarity + .16 * (1-distance/120) + .06 * street_agrees
            candidates.append((score, i, distance, similarity, street_agrees, number_agrees))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        score, i, distance, similarity, street_ok, number_ok = candidates[0]
        if similarity < .85 or (distance > 65 and not street_ok) or score < .80:
            continue
        if len(candidates) > 1 and score - candidates[1][0] < .07:
            result['status'] = 'ambiguous_neighbours'
            continue
        p = pois[i]
        result.update(status='matched', osm_id=f'node/{p[0]}', osm_name=p[3][0], amenity=p[6],
                      distance_m=round(float(distance), 1), name_similarity=round(similarity, 3),
                      street_agrees=street_ok, house_number_agrees=number_ok)
    links = pd.DataFrame(rows).reindex(columns=['company_id', 'status', 'osm_id', 'osm_name',
        'amenity', 'distance_m', 'name_similarity', 'street_agrees', 'house_number_agrees'])
    collision = links.osm_id.notna() & links.osm_id.duplicated(keep=False)
    links.loc[collision, 'status'] = 'ambiguous_shared_poi'
    for col in links.columns[2:]:
        links[col] = links[col].mask(collision)
    return links


def market_context(companies):
    data = json.loads((SNAPSHOT / 'hotel-market.json').read_text('utf-8'))
    keys = companies.municipality_name.map(municipal_key)
    nights = keys.map({k: v[0] for k, v in data.items()})
    residents = keys.map({k: v[1] for k, v in data.items()})
    ratio = nights / residents
    bands = pd.cut(ratio, [-np.inf, 1, 10, np.inf], right=False,
                   labels=['<1', '1–<10', '≥10']).astype(object)
    return pd.DataFrame({'hotel_nights_2024': nights, 'residents_2021': residents,
                         'hotel_nights_per_resident': ratio, 'hotel_band': bands})


def enrich(companies):
    with gzip.open(SNAPSHOT / 'osm-food.json.gz', 'rt', encoding='utf-8') as f:
        links = link_venues(companies, json.load(f))
    combined = pd.concat([links, market_context(companies).reset_index(drop=True)], axis=1)
    counts = links.status.value_counts().to_dict()
    matched = counts.get('matched', 0)
    eligible = len(links) - counts.get('missing_gps', 0) - counts.get('uninformative_name', 0)
    report = {'venue_count': len(links), 'osm_status_counts': counts, 'eligible': eligible,
              'match_rate_all': matched/max(1,len(links)), 'match_rate_eligible': matched/max(1,eligible),
              'tourism_covered': int(combined.hotel_band.notna().sum()),
              'snapshot': json.loads((SNAPSHOT / 'manifest.json').read_text('utf-8'))}
    return combined, report

