"""Prepare offline snapshots from Geofabrik Greece and two ELSTAT spreadsheets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import osmium
import pandas as pd

from forcecrm.enrichment import municipal_key

CENSUS = 'https://www.statistics.gr/documents/20181/17286366/mon_plith_2021.xlsx/7adf2ca4-0baf-7b19-2860-1e7caea81e7f'
TOURISM = 'https://www.statistics.gr/el/statistics/-/publication/STO12/2024'
OSM = 'https://download.geofabrik.de/europe/greece-latest.osm.pbf'
AMENITIES = {'restaurant', 'fast_food', 'ice_cream', 'cafe', 'bar', 'pub', 'food_court'}


def tourism_records(census, hotels):
    population = pd.read_excel(census, header=None)
    hotels = pd.read_excel(hotels, header=None)
    if '2021' not in str(population.iloc[0, 4]):
        raise ValueError('Unexpected census table')
    if 'ΔΙΑΝΥΚΤΕΡΕΥΣΕΙΣ ΣΥΝΟΛΟ' != str(hotels.iloc[4, 9]).strip() or int(hotels.iloc[1, 0]) != 2024:
        raise ValueError('Unexpected ELSTAT tourism table')
    pop = population.loc[population[0].eq(5), [2, 4]].copy()
    pop.columns = ['name', 'residents']
    hotel = hotels.loc[hotels[3].astype(str).str.strip().str.startswith('ΔΗΜΟΣ '), [3, 9]].copy()
    hotel.columns = ['name', 'nights']
    for frame in [pop, hotel]:
        frame['key'] = frame.name.map(municipal_key)
        frame.drop_duplicates('key', keep=False, inplace=True)
    merged = pop.merge(hotel, on='key', validate='one_to_one')
    merged['nights'] = pd.to_numeric(merged.nights, errors='coerce')
    return {r.key: [int(r.nights), int(r.residents)] for r in merged.dropna().itertuples()
            if r.residents > 0 and r.nights >= 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Contains greece.osm.pbf, census.xlsx and tourism.xls (Table 15)')
    args = parser.parse_args()
    src = args.folder
    out = Path(__file__).resolve().parents[1] / 'open_data'
    out.mkdir(exist_ok=True)
    pois = []
    reader = osmium.FileProcessor(src / 'greece.osm.pbf', osmium.osm.NODE)
    timestamp = reader.header.get('osmosis_replication_timestamp')
    for node in reader.with_filter(osmium.filter.TagFilter(*[('amenity', v) for v in sorted(AMENITIES)])):
        t = node.tags
        names = sorted({t.get(k, '') for k in ('name', 'name:el', 'name:en', 'brand', 'alt_name')} - {''})
        if names and node.location.valid():
            pois.append([node.id, node.location.lat, node.location.lon, names,
                         t.get('addr:street', ''), t.get('addr:housenumber', ''), t['amenity']])
    if not pois:
        raise ValueError('Empty OSM foodservice extract')
    market = tourism_records(src / 'census.xlsx', src / 'tourism.xls')
    if not market:
        raise ValueError('Empty ELSTAT join')
    (out / 'osm-food.json.gz').write_bytes(gzip.compress(json.dumps(pois, ensure_ascii=False).encode(), mtime=0))
    (out / 'hotel-market.json').write_text(json.dumps(market, ensure_ascii=False, indent=1), 'utf-8')
    manifest = {'osm_source': OSM, 'osm_timestamp': timestamp, 'osm_named_nodes': len(pois),
                'census': CENSUS, 'hotels': TOURISM, 'hotel_table': '15 (document 581118)',
                'municipalities': len(market), 'source_sha256': {}}
    for name in ['greece.osm.pbf', 'tourism.xls', 'census.xlsx']:
        with (src / name).open('rb') as f:
            manifest['source_sha256'][name] = hashlib.file_digest(f, 'sha256').hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), 'utf-8')


if __name__ == '__main__':
    main()


