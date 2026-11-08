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

from .clusters import build_clusters, export_clusters
from .clustering import discover
from .data import FILES, load_universe
from .filters import excluded_mask, normalize_filters, remaining_portfolio, summary as filter_summary
from .features import is_custom, normalize_weights
from .narrative import DEFAULT_LLM, feature_phrase, greek_number, short_activity
from .affinity import market_affinity


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


def rank_candidates(companies, fit, customer_vats, traits=None, profiles=None, limit=None, primary_codes=None,
                    exclude=None):
    eligible = ~companies.vat.isin(customer_vats).to_numpy()
    if exclude is not None:  # user filters applied to the results ('post' mode), before one venue per VAT is kept
        eligible &= ~exclude
    idx = np.flatnonzero(eligible)
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
        profile_id = f'ICP-{j + 1:02}'
        reasons = []
        if traits is not None and profiles is not None:
            # Short Greek names of the profile's significant traits that this business shares.
            reasons = [short_activity(feature_phrase(f)[1]) for f in profiles[j]['significant_features']
                       if f['feature'] in traits[i]][:3]
        rationale = (f'Κοινά με το {profile_id}: ' + ' · '.join(reasons) + '.') if reasons else \
            f'Παρόμοια δραστηριότητα, γεωγραφία, νομική μορφή και ηλικία με τους πελάτες του {profile_id}.'
        if primary_codes is not None:
            rationale += (f' Ομοιότητα {greek_number(100 * fit["similarity"][i])}%, συγκέντρωση πελατών '
                          f'{greek_number(np.exp(lift[i]), 2)}× του αναμενόμενου.')
        if not accepted:
            rationale = 'Διερευνητική πρόταση — εκτός ορίου αντιστοίχισης. ' + rationale
        rows.append({'profile_id': profile_id, 'company_id': row.company_id, 'rank': rank,
                     'score': float(row.score), 'rationale': rationale})
    return pd.DataFrame(rows, columns=['profile_id', 'company_id', 'rank', 'score', 'rationale'])


def evaluate_holdout(X, companies, portfolio, primary_codes, seed=2026, exclude=None):
    rng = np.random.default_rng(seed)
    count = max(1, round(len(portfolio) * .2))
    withheld = set(rng.choice(portfolio.vat.to_numpy(), count, replace=False))
    train = portfolio[~portfolio.vat.isin(withheld)]
    if len(train) < 5:
        return {'status': 'too_few_customers'}
    excluded_withheld = set()
    if exclude is not None:
        # With result filters, a held-out customer whose venues are all excluded can never be suggested;
        # recall is measured over the held-out customers the filtered list could still contain.
        excluded_withheld = withheld - set(companies.vat[~exclude])
        withheld = withheld - excluded_withheld
        if not withheld:
            return {'status': 'all_held_out_customers_excluded', 'held_out_excluded': len(excluded_withheld)}
    fitted = fit_portfolio(X, companies, train)
    ranked = rank_candidates(companies, fitted, set(train.vat), primary_codes=primary_codes, exclude=exclude)
    original = rank_candidates(companies, fitted, set(train.vat), limit=2000, exclude=exclude)
    # Baseline primary-code prevalence counts each train VAT at most once per code.
    frequencies = {}
    for vat in train.vat:
        idx = np.flatnonzero(companies.vat.eq(vat))
        for code in set(code for i in idx for code in primary_codes[i]):
            frequencies[code] = frequencies.get(code, 0) + 1
    eligible = ~companies.vat.isin(train.vat).to_numpy() if exclude is None else ~companies.vat.isin(train.vat).to_numpy() & ~exclude
    candidate = companies.loc[eligible, ['company_id', 'vat']].copy()
    candidate['baseline_score'] = [max([frequencies.get(code, 0) for code in primary_codes[i]], default=0) for i in candidate.index]
    baseline = candidate.sort_values(['baseline_score', 'company_id'], ascending=[False, True]).drop_duplicates('vat')
    lookup = companies.set_index('company_id').vat
    result = {'status': 'internal VAT holdout; not the organizer’s hidden holdout', 'seed': seed,
              'train_customers': len(train), 'held_out_customers': len(withheld), 'metrics': []}
    if exclude is not None:
        result['held_out_excluded'] = len(excluded_withheld)
    for k in [500, 2000]:
        recall = len(set(ranked.head(k).company_id.map(lookup)) & withheld) / len(withheld)
        base = len(set(baseline.head(k).vat) & withheld) / len(withheld)
        old = len(set(original.head(k).company_id.map(lookup)) & withheld) / len(withheld)
        result['metrics'].append({'k': k, 'recall': recall, 'previous_recall': old, 'baseline_recall': base,
                                  'absolute_uplift': recall - base, 'relative_uplift': recall / base - 1 if base else None})
    return result


def fingerprints(root: Path, portfolio, demo=False, model_path=None, weights=None, filters=None):
    """The root inputs of the stage graph, hashed once. Every cache key chains from these."""
    from importlib import metadata
    from . import db
    from .enrichment import SNAPSHOT

    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    if demo:
        embedding = 'demo-lexical'
    else:
        path = Path(model_path or Path(root) / 'models' / 'e5-small')
        embedding = [(str(f.relative_to(path)), f.stat().st_size, f.stat().st_mtime_ns)
                     for f in sorted(path.rglob('*')) if f.is_file()] if path.is_dir() else 'missing'
    return {'data_sha256': db.source_hashes(root),
            'portfolio_sha256': hashlib.sha256(portfolio.to_csv(index=False).encode()).hexdigest(),
            'open_data_sha256': {f.name: db.file_sha256(f) for f in sorted(SNAPSHOT.iterdir()) if f.is_file()},
            'embedding_fingerprint': embedding,
            'libraries': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'scipy',
                                                  'sentence-transformers', 'torch')} | {'python': platform.python_version()},
            'weights': normalize_weights(weights), 'demo': bool(demo),
            # User exclusions: 'pre' filters the universe in `load`, 'post' filters `rank` and `evaluate`.
            'prefilter': filters if filters and filters['mode'] == 'pre' else None,
            'postfilter': filters if filters and filters['mode'] == 'post' else None}


def make_runner(root: Path, portfolio, demo=False, model=DEFAULT_LLM, progress=lambda message: None,
                bootstrap=20, model_path=None, weights=None, cache=True, force=(), filters=None):
    """A stage runner for one analysis. `cache=False` keeps artifacts in memory only;
    `force` names stages to recompute even when a cached artifact exists."""
    from .stages import Runner, STAGES
    params = fingerprints(root, portfolio, demo, model_path, weights, filters) | {'model': model, 'bootstrap': bootstrap}
    return Runner(STAGES, params, {'root': Path(root), 'portfolio': portfolio, 'model_path': model_path},
                  Path(root) / '.cache' / 'stages' if cache else None, frozenset(force), progress)


def prepare_filters(root: Path, portfolio, filters):
    """Validate user filters; in 'pre' mode, customers keep only their non-excluded venues and a
    customer with none left leaves the analysis. Returns (filters or None, portfolio, dropped VATs)."""
    filters = normalize_filters(filters)
    dropped = []
    if filters and filters['mode'] == 'pre':
        companies, activities, _, _ = load_universe(root)
        portfolio, dropped = remaining_portfolio(portfolio, companies, activities, filters)
        if len(portfolio) < 5:
            raise ValueError(f'Μετά τα φίλτρα έμειναν {len(portfolio)} πελάτες· χρειάζονται τουλάχιστον 5. '
                             'Αφαιρέστε κάποιες εξαιρέσεις ή χρησιμοποιήστε φιλτράρισμα μετά την ανάλυση.')
    return filters, portfolio, dropped


@threadpool_limits.wrap(limits=2, user_api='blas')
def run_analysis(root: Path, portfolio, demo=False, model=DEFAULT_LLM, progress=lambda message: None,
                 bootstrap=20, evaluate=True, model_path=None, weights=None, cache=True, force=(), filters=None):
    start = time.perf_counter()
    filters, portfolio, dropped = prepare_filters(root, portfolio, filters)
    runner = make_runner(root, portfolio, demo, model, progress, bootstrap, model_path, weights, cache, force, filters)
    weights = runner.params['weights']
    # Deterministic stages first; ranking needs no AI text, so it is ready before the LLM starts.
    universe, enriched, represented = runner.get('load'), runner.get('enrich'), runner.get('represent')
    c = enriched['companies']
    fit = runner.get('discover')['fit']
    stable, found = runner.get('stability'), runner.get('evidence')
    assignments, lookalikes = runner.get('discover')['assignments'], runner.get('rank')
    profiles = runner.get('narrate')
    validation = runner.get('evaluate') if evaluate else None
    feature_meta = represented['meta']
    warnings = []
    if is_custom(weights):
        shares = ', '.join(f'{name} {value:.0%}' for name, value in weights.items())
        warnings.append(f'Custom similarity weights ({shares}). These are not the validated defaults '
                        '(activity 65%, geography 20%, legal_type 10%, age 5%); results may be less accurate. '
                        'Compare Recall@k with the default weights in the validation view.')
    filter_report = None
    if filters:
        if filters['mode'] == 'pre':
            counts = universe['filtered']
        else:
            mask = excluded_mask(c, universe['activities'], filters)
            counts = {'excluded_venues': int(mask.sum()),
                      'excluded_businesses': len(set(c.vat[mask]) - set(c.vat[~mask]))}
        filter_report = {**filters, **counts, 'summary': filter_summary(filters),
                         'dropped_customers': len(dropped), 'dropped_customer_vats': dropped}
        effect = ('Τα προφίλ, οι στατιστικές και οι προτάσεις υπολογίστηκαν χωρίς αυτές τις επιχειρήσεις'
                  + (f'· {len(dropped)} πελάτες σας αφαιρέθηκαν επειδή όλα τα καταστήματά τους εξαιρέθηκαν.'
                     if dropped else '.')) if filters['mode'] == 'pre' else \
                 'Τα προφίλ δεν άλλαξαν· οι εξαιρεμένες επιχειρήσεις αφαιρέθηκαν μόνο από τις προτάσεις.'
        venues = f"{counts['excluded_venues']:,}".replace(',', '.')  # Greek thousands separator
        warnings.append(f"{filter_summary(filters)} {venues} καταστήματα εξαιρέθηκαν. {effect}")
    if demo:
        warnings.append('DEMO: synthetic customer selection/revenue and lexical embeddings; no LLM. Not a challenge submission.')
    if len(lookalikes) < 2000:
        warnings.append('Fewer than 2,000 eligible distinct prospective VATs exist. Export cannot meet the candidate minimum.')
    if feature_meta['missing_primary_venues']:
        warnings.append(f"{feature_meta['missing_primary_venues']} venues have no usable primary activity embedding.")
    seconds = {r.stage: r.seconds for r in runner.records}
    report = {'demo': demo, 'data_audit': universe['audit'], 'portfolio_customers': len(portfolio),
              'portfolio_venues': len(fit['indices']), 'portfolio_revenue': float(portfolio.revenue.sum()),
              'profile_count': len(profiles), 'candidate_count': len(lookalikes),
              'rejected_venues': int((~fit['accepted']).sum()), 'feature_configuration': feature_meta,
              'cluster_selection': fit['diagnostics'], 'bootstrap_ari': stable['ari'], 'hypotheses_tested': found['tests'],
              'portfolio_traits': found['portfolio_traits'], 'portfolio_traits_tested': found['portfolio_traits_tested'],
              'statistics_note': 'Exploratory Fisher tests after data-driven clustering; global BH q≤0.05, ≥5 VAT support, lift≥1.2. Not independent confirmatory inference.',
              'confidence_note': 'Similarity and fit confidence are not probabilities of conversion or creditworthiness.',
              'rank_note': 'Global rank, one representative venue per VAT; profile_id indicates nearest profile, including explicitly marked exploratory candidates outside threshold.',
              'validation': validation, 'warnings': warnings, 'seed': 42, 'filters': filter_report,
              'input_sha256': {f: runner.params['data_sha256'][f] for f in FILES},
              'portfolio_sha256': runner.params['portfolio_sha256'],
              'performance': {'seconds': time.perf_counter() - start,
                              'feature_seconds': seconds.get('enrich', 0) + seconds.get('represent', 0),
                              'rss_mb_at_finish': psutil.Process().memory_info().rss / 1024 ** 2,
                              'platform': platform.platform(), 'processor': platform.processor()},
              'stages': runner.summary(), 'stage_notes': runner.notes,
              'sources': ['GEMI Open Data — competition-only data', 'Geocoding powered by OpenStreetMap contributors, ODbL']}
    report['enrichment'] = enriched['coverage']
    report['sources'] += ['OpenStreetMap contributors via Geofabrik (ODbL)', 'ELSTAT: municipal hotel nights 2024 and resident population 2021. ForceCRM calculated the ratios. ELSTAT bears no responsibility for the transformation.']
    preview = lookalikes.merge(c[['company_id', 'vat', 'company_name_el', 'city', 'latitude', 'longitude']], on='company_id')
    clusters = build_clusters(c, profiles, fit, assignments, lookalikes)
    return {'profiles': profiles, 'assignments': assignments, 'lookalikes': lookalikes, 'report': report,
            'clusters': clusters,
            'enrichment': enriched['enrichment'], 'preview': preview.replace({np.nan: None}).to_dict('records')}


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
    if 'clusters' in result:
        export_clusters(result, directory)
