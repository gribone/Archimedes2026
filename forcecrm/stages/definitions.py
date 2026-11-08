"""The analysis as explicit stages. Each function is the code that used to sit inline in
`pipeline.run_analysis`, unchanged in logic; only the hand-over between steps is now explicit.

    universe side (independent of the customer list)
      load ─► enrich ─► represent
    portfolio side
      represent ─► discover ─► stability ┐
                     │  └────► evidence ─┼─► profiles ─► narrate (LLM)
                     └──────────────────►rank           evaluate (holdout)

Stages are deterministic except `narrate`, whose wording comes from the local LLM and is
fact-checked sentence by sentence. Ranking never reads anything the LLM wrote.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from .engine import Stage, sha256, source_hash
from ..clustering import stability as bootstrap_stability
from ..data import load_universe
from ..enrichment import enrich as link_and_contextualise
from ..filters import apply_to_universe, excluded_mask
from ..evidence import significant_features, portfolio_traits, activity_mix
from ..features import build_features, count_secondaries, secondary_band, DEFAULT_WEIGHTS
from ..narrative import narrate as describe
from ..pipeline import fit_portfolio, rank_candidates, evaluate_holdout
from ..titles import activity_families, make_title


def load(ctx):
    companies, activities, descriptions, audit = load_universe(ctx.obj('root'))
    filtered = None
    if ctx.param('prefilter'):
        # 'pre' mode: the excluded venues are gone before any feature, profile or statistic is computed.
        companies, activities, filtered = apply_to_universe(companies, activities, ctx.param('prefilter'))
    return {'companies': companies, 'activities': activities, 'descriptions': descriptions, 'audit': audit,
            'filtered': filtered}


def result_exclusions(ctx):
    """'post' mode: venues removed from the suggestions (None when no result filter is set)."""
    filters = ctx.param('postfilter')
    return excluded_mask(ctx['enrich']['companies'], ctx['load']['activities'], filters) if filters else None


def enrich(ctx):
    enrichment, coverage = link_and_contextualise(ctx['load']['companies'])
    # A new frame instead of a column added in place: the tourism band feeds market affinity.
    companies = ctx['load']['companies'].copy()
    companies['hotel_band'] = enrichment.hotel_band
    return {'companies': companies, 'enrichment': enrichment, 'coverage': coverage}


def represent(ctx):
    universe = ctx['load']
    X, traits, primary, meta = build_features(universe['companies'], universe['activities'], universe['descriptions'],
                                              ctx.obj('root'), ctx.param('demo'), ctx.obj('model_path'),
                                              ctx.param('weights'))
    for values, venue in zip(traits, ctx['enrich']['enrichment'].itertuples()):
        if pd.notna(venue.hotel_band):
            values.add(f'hotel_band:{venue.hotel_band}')
        if venue.status == 'matched':
            values.add(f'osm_amenity:{venue.amenity}')
    return {'X': X, 'traits': traits, 'primary': primary, 'meta': meta}


def discover(ctx):
    companies = ctx['enrich']['companies']
    fit = fit_portfolio(ctx['represent']['X'], companies, ctx.obj('portfolio'))
    masks = []
    for j in range(len(fit['centers'])):
        mask = np.zeros(len(companies), dtype=bool)
        mask[fit['indices'][fit['labels'] == j]] = True
        masks.append(mask)
    # A3: every venue is assigned with the same cosine metric that formed the profiles.
    ids = np.array([f'ICP-{i + 1:02}' for i in fit['nearest']], dtype=object)
    ids[~fit['accepted']] = 'NONE'
    assignments = pd.DataFrame({'company_id': companies.company_id, 'profile_id': ids, 'score': fit['similarity']})
    return {'fit': fit, 'masks': masks, 'assignments': assignments}


def stability(ctx):
    fit = ctx['discover']['fit']
    per_profile, ari = bootstrap_stability(ctx['represent']['X'][fit['indices']], fit['weights'], fit['vats'],
                                           fit['centers'], fit['labels'], repetitions=ctx.param('bootstrap'))
    return {'per_profile': per_profile, 'ari': ari}


def evidence(ctx):
    companies, d = ctx['enrich']['companies'], ctx['load']['descriptions']
    traits, masks = ctx['represent']['traits'], ctx['discover']['masks']
    features, tests = significant_features(companies, traits, masks, d)
    # Portfolio-wide only: how many secondary ΚΑΔ a business declares (customers declare more than
    # the market). Kept out of the per-profile tests and out of the ranking.
    counts = count_secondaries(companies, ctx['load']['activities'])
    shared_traits = [values | {f'secondary_count:{secondary_band(n)}'} for values, n in zip(traits, counts)]
    shared, shared_tests = portfolio_traits(companies, shared_traits, set(ctx.obj('portfolio').vat), d)
    return {'features': features, 'tests': tests, 'activity_mix': activity_mix(companies, ctx['represent']['primary'], masks, d),
            'portfolio_traits': shared, 'portfolio_traits_tested': shared_tests}


def profiles(ctx):
    """Every number a profile reports, before any text is written."""
    c, d, portfolio = ctx['enrich']['companies'], ctx['load']['descriptions'], ctx.obj('portfolio')
    fit, masks, primary = ctx['discover']['fit'], ctx['discover']['masks'], ctx['represent']['primary']
    stable, found = ctx['stability']['per_profile'], ctx['evidence']
    revenue_map = portfolio.set_index('vat').revenue
    branches = c[c.vat.isin(portfolio.vat)].vat.value_counts()
    activity_names = dict(zip(d.activity_id, d.description))
    result = []
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
                   'confidence': confidence,  # brief §10 name ("εμπιστοσύνη"); same value as confidence_score
                   'quality': 'strong' if stable[j]['mean'] >= .75 and len(found['features'][j]) >= 2 else 'limited',
                   'significant_features': found['features'][j], 'activity_mix': found['activity_mix'][j],
                   'acceptance_cosine_distance': float(fit['radii'][j])}
        positions = np.flatnonzero(mask)
        profile['activity_families'] = activity_families([primary[i] for i in positions], members.vat, activity_names)
        result.append(profile)
    return result


def rank(ctx):
    # Reads the statistical traits only, never the AI text, so it runs before narration.
    reasons = [{'significant_features': features} for features in ctx['evidence']['features']]
    return rank_candidates(ctx['enrich']['companies'], ctx['discover']['fit'], set(ctx.obj('portfolio').vat),
                           ctx['represent']['traits'], reasons, primary_codes=ctx['represent']['primary'],
                           exclude=result_exclusions(ctx))


def narrate(ctx):
    """Titles and descriptions, cached per profile on that profile's facts and the model digest.
    Unchanged profiles reuse their text; `force` regenerates it."""
    model, demo, root = ctx.param('model'), ctx.param('demo'), ctx.obj('root')
    cache_dir = ctx.runner.cache_dir
    digest = None if demo else ollama_digest(model)
    code = source_hash(('narrative.py', 'titles.py'))
    written, hits = [], 0
    source = ctx['profiles']
    for j, base in enumerate(source):
        key = sha256({'profile': base, 'model': model, 'digest': digest, 'demo': demo, 'code': code})
        path = cache_dir / f'narrate-{key[:24]}.json' if cache_dir else None
        if path is not None and 'narrate' not in ctx.forced and path.exists():
            text = json.loads(path.read_text('utf-8'))
            hits += 1
        else:
            ctx.progress(f'Writing grounded Greek title and description {j + 1}/{len(source)}')
            profile = copy.deepcopy(base)  # the LLM steps never touch the stage's own artifact
            text = make_title(profile, root, model=model, demo=demo)
            profile.update(text)
            text = {'title': text, 'description': describe(profile, root, model=model, demo=demo)}
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(text, ensure_ascii=False, indent=1, allow_nan=False), 'utf-8')
        profile = dict(base)
        profile.update(text['title'])
        profile.update(text['description'])
        written.append(profile)
    ctx.runner.notes['narrate'] = {'profiles': len(source), 'reused_text': hits}
    ctx.status = 'cached' if source and hits == len(source) else 'partial' if hits else 'computed'
    return written


def evaluate(ctx):
    companies, primary = ctx['enrich']['companies'], ctx['represent']['primary']
    portfolio = ctx.obj('portfolio')
    exclude = result_exclusions(ctx)
    result = evaluate_holdout(ctx['represent']['X'], companies, portfolio, primary, exclude=exclude)
    if ctx['represent']['meta']['custom_weights'] and 'metrics' in result:
        # Same holdout with the validated default weights, so the user sees the accuracy trade-off.
        ctx.progress('Re-running the holdout with default weights for comparison')
        default_X = ctx.runner.variant(weights=dict(DEFAULT_WEIGHTS)).get('represent')['X']
        default = {m['k']: m['recall'] for m in evaluate_holdout(default_X, companies, portfolio, primary, exclude=exclude)['metrics']}
        for metric in result['metrics']:
            metric['default_weights_recall'] = default.get(metric['k'])
    return result


def ollama_digest(model):
    """The pulled model's digest, so re-pulling a model invalidates its cached text."""
    import httpx
    from ..narrative import OLLAMA
    with httpx.Client(trust_env=False, timeout=30) as client:
        tags = client.get(OLLAMA + '/api/tags')
        tags.raise_for_status()
        available = {m['name']: m['digest'] for m in tags.json().get('models', [])}
    if model not in available:
        raise ValueError(f'Local Ollama model {model} not found. Run: ollama pull {model}')
    return available[model]


UNIVERSE = ('data_sha256', 'libraries')
PORTFOLIO = ('portfolio_sha256',)

STAGES = {s.name: s for s in [
    Stage('load', 'A8 data', (), UNIVERSE + ('prefilter',), ('data.py', 'db.py', 'filters.py'), load,
          message='Reading and validating the business universe'),
    Stage('enrich', 'X2/X3', ('load',), UNIVERSE + ('open_data_sha256',), ('enrichment.py',), enrich,
          message='Linking OpenStreetMap venues and municipal hotel statistics'),
    Stage('represent', 'A1', ('load', 'enrich'), ('weights', 'demo', 'embedding_fingerprint', 'libraries'),
          ('features.py',), represent, message='Building activity embeddings and geographic features'),
    Stage('discover', 'A2/A3', ('enrich', 'represent'), PORTFOLIO, ('clustering.py', 'pipeline.py'), discover,
          message='Discovering 1–6 revenue-weighted customer profiles'),
    Stage('stability', 'A5', ('represent', 'discover'), ('bootstrap',), ('clustering.py',), stability,
          message='Measuring bootstrap stability'),
    Stage('evidence', 'A4', ('load', 'enrich', 'represent', 'discover'), PORTFOLIO, ('evidence.py', 'features.py'), evidence,
          message='Testing which traits are over-represented (Fisher + BH)'),
    Stage('profiles', 'A2–A5', ('load', 'enrich', 'represent', 'discover', 'stability', 'evidence'), PORTFOLIO,
          ('titles.py',), profiles),
    Stage('rank', 'A7', ('load', 'enrich', 'represent', 'discover', 'evidence'), PORTFOLIO + ('postfilter',),
          ('pipeline.py', 'affinity.py', 'narrative.py', 'filters.py'), rank, message='Ranking non-customer businesses'),
    Stage('narrate', 'A6', ('profiles',), ('model', 'demo'), ('narrative.py', 'titles.py'), narrate,
          persist=False, deterministic=False),
    Stage('evaluate', 'holdout', ('load', 'enrich', 'represent'), PORTFOLIO + ('postfilter',),
          ('pipeline.py', 'affinity.py', 'features.py', 'filters.py'),
          evaluate, message='Running an internal 20% VAT holdout against the primary-code baseline'),
]}
ORDER = list(STAGES)
