from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .clustering import discover, stability
from .data import FILES, load_universe
from .evidence import significant_features
from .features import build_features, is_custom, normalize_weights
from .narrative import narrate, DEFAULT_LLM
from .titles import activity_mix, make_title
from .affinity import market_affinity
from .enrichment import enrich


def fit_portfolio(X, companies, portfolio):
    indices = np.flatnonzero(companies.vat.isin(portfolio.vat).to_numpy())
    vats = companies.vat.to_numpy()[indices]
    revenues = portfolio.set_index('vat').revenue
    # Saturated weights prevent one unusually large account dominating clustering.
    cap = float(revenues.quantile(.95))
    by_vat = np.sqrt(np.minimum(revenues, cap) + 1)
    by_vat /= by_vat.mean()
    branch_counts = pd.Series(vats).value_counts()
    weights = np.array([by_vat[v] / branch_counts[v] for v in vats])
    centers, labels, diagnostics = discover(X[indices], weights, vats)
    distances = np.clip(1 - X @ centers.T, 0, 2)
    nearest = distances.argmin(axis=1)
    radii = np.array([max(.015, float(np.quantile(distances[indices[labels == j], j], .95)) + .015)
                      for j in range(len(centers))])
    selected_dist = distances[np.arange(len(companies)), nearest]
    accepted = selected_dist <= radii[nearest]
    similarity = np.clip(1 - selected_dist, 0, 1)
    return {'indices': indices, 'vats': vats, 'weights': weights, 'centers': centers, 'labels': labels,
            'diagnostics': diagnostics, 'distances': distances, 'nearest': nearest,
            'radii': radii, 'accepted': accepted, 'similarity': similarity}


def rank_candidates(companies, fit, customer_vats, traits=None, profiles=None, limit=None, primary_codes=None):
    idx = np.flatnonzero(~companies.vat.isin(customer_vats).to_numpy())
    score = fit['similarity']
    if primary_codes is not None:
        lift = market_affinity(companies, customer_vats, primary_codes)
        score = score * (1 / (1 + np.exp(-lift)))
    table = pd.DataFrame({'index': idx, 'vat': companies.vat.to_numpy()[idx],
                          'company_id': companies.company_id.to_numpy()[idx], 'score': score[idx]})
    # One best venue per VAT. Stable tie-break makes the global evaluation order reproducible.
    table = table.sort_values(['score', 'company_id'], ascending=[False, True]).drop_duplicates('vat')
    if limit is not None:
        table = table.head(limit)
    rows = []
    for rank, row in enumerate(table.itertuples(), 1):
        i = row.index
        j = int(fit['nearest'][i])
        accepted = bool(fit['accepted'][i])
        reasons = []
        if traits is not None and profiles is not None:
            reasons = [f['label'] for f in profiles[j]['significant_features'] if f['feature'] in traits[i]][:2]
        rationale = '; '.join(reasons) if reasons else 'Ομοιότητα δραστηριότητας, γεωγραφίας, νομικής μορφής και ηλικίας'
        if primary_codes is not None:
            rationale += f' · Ομοιότητα {fit["similarity"][i]:.1%}, σχετική συγκέντρωση πελατών {np.exp(lift[i]):.2f}×.'
        if not accepted:
            rationale = 'Διερευνητική πρόταση — εκτός ορίου αντιστοίχισης. ' + rationale
        rows.append({'profile_id': f'ICP-{j + 1:02}', 'company_id': row.company_id, 'rank': rank,
                     'score': float(row.score), 'rationale': rationale})
    return pd.DataFrame(rows, columns=['profile_id', 'company_id', 'rank', 'score', 'rationale'])


def evaluate_holdout(X, companies, portfolio, primary_codes, seed=2026):
    rng = np.random.default_rng(seed)
    count = max(1, round(len(portfolio) * .2))
    withheld = set(rng.choice(portfolio.vat.to_numpy(), count, replace=False))
    train = portfolio[~portfolio.vat.isin(withheld)]
    if len(train) < 5:
        return {'status': 'too_few_customers'}
    fitted = fit_portfolio(X, companies, train)
    ranked = rank_candidates(companies, fitted, set(train.vat), primary_codes=primary_codes)
    original = rank_candidates(companies, fitted, set(train.vat), limit=2000)
    # Baseline primary-code prevalence counts each train VAT at most once per code.
    frequencies = {}
    for vat in train.vat:
        idx = np.flatnonzero(companies.vat.eq(vat))
        for code in set(code for i in idx for code in primary_codes[i]):
            frequencies[code] = frequencies.get(code, 0) + 1
    candidate = companies.loc[~companies.vat.isin(train.vat), ['company_id', 'vat']].copy()
    candidate['baseline_score'] = [max([frequencies.get(code, 0) for code in primary_codes[i]], default=0) for i in candidate.index]
    baseline = candidate.sort_values(['baseline_score', 'company_id'], ascending=[False, True]).drop_duplicates('vat')
    lookup = companies.set_index('company_id').vat
    result = {'status': 'internal VAT holdout; not the organizer’s hidden holdout', 'seed': seed,
              'train_customers': len(train), 'held_out_customers': len(withheld), 'metrics': []}
    for k in [500, 2000]:
        recall = len(set(ranked.head(k).company_id.map(lookup)) & withheld) / len(withheld)
        base = len(set(baseline.head(k).vat) & withheld) / len(withheld)
        old = len(set(original.head(k).company_id.map(lookup)) & withheld) / len(withheld)
        result['metrics'].append({'k': k, 'recall': recall, 'previous_recall': old, 'baseline_recall': base,
                                  'absolute_uplift': recall - base, 'relative_uplift': recall / base - 1 if base else None})
    return result


@threadpool_limits.wrap(limits=2, user_api='blas')
def run_analysis(root: Path, portfolio, demo=False, model=DEFAULT_LLM, progress=lambda message: None,
                 bootstrap=20, evaluate=True, model_path=None, weights=None):
    start = time.perf_counter()
    progress('Reading and validating the business universe')
    c, a, d, audit = load_universe(root)
    progress('Building activity embeddings and geographic features')
    feature_start = time.perf_counter()
    weights = normalize_weights(weights)
    X, traits, primary, feature_meta = build_features(c, a, d, root, demo, model_path, weights)
    progress('Linking OpenStreetMap venues and municipal hotel statistics')
    enrichment, coverage = enrich(c)
    c['hotel_band'] = enrichment.hotel_band
    for values, venue in zip(traits, enrichment.itertuples()):
        if pd.notna(venue.hotel_band):
            values.add(f'hotel_band:{venue.hotel_band}')
        if venue.status == 'matched':
            values.add(f'osm_amenity:{venue.amenity}')
    feature_seconds = time.perf_counter() - feature_start
    progress('Discovering 1–6 revenue-weighted customer profiles')
    fit = fit_portfolio(X, c, portfolio)
    indices, labels, centers = fit['indices'], fit['labels'], fit['centers']
    masks = []
    for j in range(len(centers)):
        mask = np.zeros(len(c), dtype=bool)
        mask[indices[labels == j]] = True
        masks.append(mask)
    progress('Measuring bootstrap stability and statistical evidence')
    stable, ari = stability(X[indices], fit['weights'], fit['vats'], centers, labels, repetitions=bootstrap)
    evidence, tests = significant_features(c, traits, masks, d)
    revenue_map = portfolio.set_index('vat').revenue
    branches = c[c.vat.isin(portfolio.vat)].vat.value_counts()
    activity_names = dict(zip(d.activity_id, d.description))
    profiles = []
    for j, mask in enumerate(masks):
        members = c[mask]
        allocated_revenue = sum(revenue_map[v] / branches[v] for v in members.vat)
        distance = fit['distances'][mask, j]
        # A relative fit score, not a calibrated probability of purchase.
        confidence = float(np.mean(np.exp(-distance / max(fit['radii'][j], 1e-6))))
        profile = {'profile_id': f'ICP-{j + 1:02}', 'title': f'Προφίλ {j + 1}',
                   'customer_count': int(members.vat.nunique()), 'venue_count': len(members),
                   'revenue_share': float(allocated_revenue / portfolio.revenue.sum()),
                   'stability': stable[j]['mean'], 'stability_detail': stable[j], 'confidence_score': confidence,
                   'quality': 'strong' if stable[j]['mean'] >= .75 and len(evidence[j]) >= 2 else 'limited',
                   'significant_features': evidence[j], 'acceptance_cosine_distance': float(fit['radii'][j])}
        positions = np.flatnonzero(mask)
        profile['activity_mix'] = activity_mix([primary[i] for i in positions], members.vat, activity_names)
        progress(f'Writing grounded Greek title and description {j + 1}/{len(centers)}')
        profile.update(make_title(profile, root, model=model, demo=demo))
        profile.update(narrate(profile, root, model=model, demo=demo))
        profiles.append(profile)
    ids = np.array([f'ICP-{i + 1:02}' for i in fit['nearest']], dtype=object)
    ids[~fit['accepted']] = 'NONE'
    assignments = pd.DataFrame({'company_id': c.company_id, 'profile_id': ids, 'score': fit['similarity']})
    progress('Ranking non-customer businesses')
    lookalikes = rank_candidates(c, fit, set(portfolio.vat), traits, profiles, primary_codes=primary)
    validation = None
    if evaluate:
        progress('Running an internal 20% VAT holdout against the primary-code baseline')
        validation = evaluate_holdout(X, c, portfolio, primary)
        if is_custom(weights) and 'metrics' in validation:
            # Same holdout with the validated default weights, so the user sees the accuracy trade-off.
            progress('Re-running the holdout with default weights for comparison')
            default_X = build_features(c, a, d, root, demo, model_path)[0]
            default = {m['k']: m['recall'] for m in evaluate_holdout(default_X, c, portfolio, primary)['metrics']}
            for metric in validation['metrics']:
                metric['default_weights_recall'] = default.get(metric['k'])
    warnings = []
    if is_custom(weights):
        shares = ', '.join(f'{name} {value:.0%}' for name, value in weights.items())
        warnings.append(f'Custom similarity weights ({shares}). These are not the validated defaults '
                        '(activity 65%, geography 20%, legal_type 10%, age 5%); results may be less accurate. '
                        'Compare Recall@k with the default weights in the validation view.')
    if demo:
        warnings.append('DEMO: synthetic customer selection/revenue and lexical embeddings; no LLM. Not a challenge submission.')
    if len(lookalikes) < 2000:
        warnings.append('Fewer than 2,000 eligible distinct prospective VATs exist. Export cannot meet the candidate minimum.')
    if feature_meta['missing_primary_venues']:
        warnings.append(f"{feature_meta['missing_primary_venues']} venues have no usable primary activity embedding.")
    report = {'demo': demo, 'data_audit': audit, 'portfolio_customers': len(portfolio),
              'portfolio_venues': len(indices), 'portfolio_revenue': float(portfolio.revenue.sum()),
              'profile_count': len(profiles), 'candidate_count': len(lookalikes),
              'rejected_venues': int((~fit['accepted']).sum()), 'feature_configuration': feature_meta,
              'cluster_selection': fit['diagnostics'], 'bootstrap_ari': ari, 'hypotheses_tested': tests,
              'statistics_note': 'Exploratory Fisher tests after data-driven clustering; global BH q≤0.05, ≥5 VAT support, lift≥1.2. Not independent confirmatory inference.',
              'confidence_note': 'Similarity and fit confidence are not probabilities of conversion or creditworthiness.',
              'rank_note': 'Global rank, one representative venue per VAT; profile_id indicates nearest profile, including explicitly marked exploratory candidates outside threshold.',
              'validation': validation, 'warnings': warnings, 'seed': 42,
              'input_sha256': {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in FILES},
              'portfolio_sha256': hashlib.sha256(portfolio.to_csv(index=False).encode()).hexdigest(),
              'performance': {'seconds': time.perf_counter() - start, 'feature_seconds': feature_seconds,
                              'rss_mb_at_finish': psutil.Process().memory_info().rss / 1024 ** 2,
                              'platform': platform.platform(), 'processor': platform.processor()},
              'sources': ['GEMI Open Data — competition-only data', 'Geocoding powered by OpenStreetMap contributors, ODbL']}
    report['enrichment'] = coverage
    report['sources'] += ['OpenStreetMap contributors via Geofabrik (ODbL)', 'ELSTAT: municipal hotel nights 2024 and resident population 2021. ForceCRM calculated the ratios. ELSTAT bears no responsibility for the transformation.']
    preview = lookalikes.merge(c[['company_id', 'vat', 'company_name_el', 'city', 'latitude', 'longitude']], on='company_id')
    return {'profiles': profiles, 'assignments': assignments, 'lookalikes': lookalikes, 'report': report,
            'enrichment': enrichment, 'preview': preview.replace({np.nan: None}).to_dict('records')}


def export_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'profiles.json').write_text(json.dumps(result['profiles'], ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    (directory / 'report.json').write_text(json.dumps(result['report'], ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    files = ['assignments', 'lookalikes']
    if 'enrichment' in result:
        files.append('enrichment')
    for name in files:
        result[name].to_csv(directory / f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.8f')
