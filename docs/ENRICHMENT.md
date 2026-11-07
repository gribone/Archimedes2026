# Offline venue and market enrichment

Every analysis reads the compact snapshots in `open_data/` and produces `enrichment.csv` with one row per registry company_id. The validation tab shows coverage and provides the download. The three competition export schemas are unchanged.

## Venue linkage (EP2)

Source: [© OpenStreetMap contributors, ODbL 1.0](https://www.openstreetmap.org/copyright), [Geofabrik Greece](https://download.geofabrik.de/europe/greece.html). The bundled derived OSM database `osm-food.json.gz` is distributed under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Each row contains the OSM node ID, latitude, longitude, name aliases, street, house number and amenity. Snapshot timestamp and original file hashes are in the manifest.

The extract includes named foodservice nodes. Venues mapped only as buildings or relations are outside its scope. Matching uses GPS within 120 m, normalizes case, accents, punctuation and common legal suffixes, and compares registered names against mapped aliases. Missing GPS and generic names abstain. A conflicting house number on the same street rules out that candidate. Score = 78% name similarity + 16% proximity (linear decline to zero at 120 m) + 6% street agreement. Acceptance requires name similarity ≥0.85, score ≥0.80 and street agreement beyond 65 m. If the best two candidate scores differ by less than 0.07, neither is accepted. If multiple registry venues select the same node, all claims on it are withheld. Local records carry the decision, accepted node ID, distance and agreement fields. A blank address agreement is not evidence for an address match. Coverage is measured, identity precision has no independently labelled ground truth. Legal names that differ from storefront signage and incomplete mapping reduce recall.

Matched OSM amenity categories enter the same VAT-level significance tests as registry traits. Unmatched venues retain baseline information and get no inferred amenity.

## Municipal hotel market (EP3)

[ELSTAT 2024 hotels, table 15](https://www.statistics.gr/el/statistics/-/publication/STO12/2024) supplies annual guest-nights in hotels by municipality. [ELSTAT 2021 census](https://www.statistics.gr/en/2021-census-res-pop-results) supplies the resident population. We select municipality-level census rows and individual municipality hotel rows. Combined municipalities and regional totals are excluded. Exact normalized municipal names must match uniquely in both tables. We join each venue by its municipality before the slash in the registry field.

The derived ratio measures annual hotel nights per census resident. It excludes campsites and short stays and cannot identify the clientele of an individual venue. Bands are below 1, 1 to below 10, and at least 10 nights per resident. Available bands enter profile evidence tests and contribute 0.10 times their shrinkage-adjusted log concentration to prospect ranking. The concentration uses only training customers. Missing ratios contribute zero log lift. OSM coverage itself is never used to prioritize sales calls.

[ELSTAT reuse terms](https://www.statistics.gr/documents/20181/1412250/Copyright_Reuse_Policy_EN.pdf/dfacb7d1-3d9b-471f-851a-8b8b09994a74): ForceCRM joined and transformed the published tables. ELSTAT bears no responsibility for the derived results.

## Reproduce

Snapshots ship with the app. To rebuild them, install `osmium==4.3.2` in the project Python environment. Download the Greece PBF and the two workbooks listed in `open_data/manifest.json` to one folder as `greece.osm.pbf`, `census.xlsx`, and `tourism.xls` (the last file contains XLSX bytes). Run `python -m scripts.prepare_open_data FOLDER` from the project root. A newer PBF changes the source hash and may change matches.

Use `python -m scripts.audit_ranking` and `python -m scripts.audit_ranking --with-tourism` on the supplied portfolio for paired VAT holdouts. These compare the same ranker with the hotel-market component disabled and enabled.


## Measured on the competition registry

The matcher accepted 482 of 24,424 venues (1.97%), or 3.23% of 14,908 eligible venues. Median accepted distance was 7.65 m. Exact normalized street names agreed on 153 accepted links. Six venues had near-tied candidates and four claimed a shared OSM node. The other decisions were 975 missing GPS, 8,541 uninformative names after the GPS check and 14,416 unmatched. These are coverage figures, not accuracy estimates.

The municipal ratio covers 15,092 venues (61.8%). Across five internal splits of 240 training and 60 withheld VATs, mean Recall@500 was 18.0% without tourism and 21.3% with it. Recall@2,000 was 50.3% and 51.3%, respectively. See the [paired enriched results](tourism-audit.json) and [unenriched results](ranking-audit.json). Splits overlap and the organizer controls the independent hidden evaluation.
