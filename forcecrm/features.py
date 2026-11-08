from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
AS_OF = '2026-11-01'
# Validated defaults; custom weights are an exploratory option and are re-checked by the holdout.
DEFAULT_WEIGHTS = {'activity': .65, 'geography': .20, 'legal_type': .10, 'age': .05}


def normalize_weights(weights):
    """Return weights over exactly the four blocks, non-negative and summing to 1."""
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f'Weights must contain exactly: {", ".join(DEFAULT_WEIGHTS)}.')
    values = {}
    for name in DEFAULT_WEIGHTS:
        value = weights[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Weight {name} must be a non-negative number.')
        values[name] = float(value)
    total = sum(values.values())
    if total <= 0:
        raise ValueError('At least one weight must be positive.')
    return {name: value / total for name, value in values.items()}


def description_embeddings(descriptions, root: Path, demo=False, model_path=None):
    texts = descriptions.description.tolist()
    if demo:
        tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=12000)
        sparse = tf.fit_transform(texts)
        dim = min(96, sparse.shape[0] - 1, sparse.shape[1] - 1)
        values = TruncatedSVD(dim, random_state=42).fit_transform(sparse) if dim >= 2 else sparse.toarray()
        return normalize(values).astype('float32'), {'model': 'DEMO lexical TF-IDF/SVD; not semantic AI', 'demo': True}
    path = Path(model_path or root / 'models' / 'e5-small')
    if not path.is_dir():
        raise ValueError('Local embedding model is missing. Run: python run.py --setup-models. Or select demo mode.')
    manifest = path / 'forcecrm-model.json'
    identity = json.loads(manifest.read_text('utf-8')) if manifest.exists() else {'local_path': str(path)}
    # Local file fingerprints invalidate the cache after a model replacement.
    stamps = [(str(f.relative_to(path)), f.stat().st_size, f.stat().st_mtime_ns)
              for f in sorted(path.rglob('*')) if f.is_file()]
    # E5 models need the 'query: ' prefix; other encoders (used in benchmarks) get raw text.
    prefix = 'query: ' if 'e5' in str(identity.get('model', path)).lower() or 'e5' in path.name.lower() else ''
    fingerprint = [texts, identity, stamps] + ([] if prefix == 'query: ' else [prefix])  # shipped E5 cache key unchanged
    key = hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False).encode()).hexdigest()
    cache = root / '.cache' / f'embeddings-{key}.npy'
    if cache.exists():
        return np.load(cache, allow_pickle=False), identity
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    encoder = SentenceTransformer(str(path), local_files_only=True, trust_remote_code=False, device='cpu')
    encoder.max_seq_length = 256
    values = encoder.encode([prefix + t for t in texts], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False).astype('float32')
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, values)
    return values, identity


def is_custom(weights):
    return any(abs(weights[name] - value) > 1e-6 for name, value in DEFAULT_WEIGHTS.items())


# How secondary activities enter the activity vector (A1). The primary activity always dominates.
# weight: share of the secondary part when a business declares few secondaries;
# full_until / ignore_above: trust in secondaries falls as full_until / n beyond full_until declared
# secondaries, and is zero above ignore_above (declaring very many codes is usually noise);
# non56_weight: share for related secondaries outside food service (bakery, retail, lodging...);
# max_codes / min_similarity: at most this many codes per group, each this similar to the primary.
# Measured on 5 holdout splits (2026-11-08): recall-neutral against the old 85/15, ≤3-ΚΑΔ-56 rule
# (21.7% / 51.0% vs 21.3% / 51.3%), and it follows the challenge owner's guidance: primary dominates,
# secondaries take part (also outside food service), very many secondaries are treated as noise.
SECONDARY_DEFAULT = {'weight': .15, 'full_until': 5, 'ignore_above': 20, 'non56_weight': .05,
                     'max_codes': 3, 'min_similarity': .70}
SECONDARY_BANDS = [(0, '0'), (5, '1–5'), (10, '6–10'), (20, '11–20')]


def secondary_band(n):
    return next((label for limit, label in SECONDARY_BANDS if n <= limit), '>20')


def secondary_trust(n, policy):
    """Share of the secondary weight a business keeps, given how many secondaries it declares."""
    if policy['ignore_above'] is not None and n > policy['ignore_above']:
        return 0.0
    if policy['full_until'] is not None and n > policy['full_until']:
        return policy['full_until'] / n
    return 1.0


def count_secondaries(c, a):
    """Number of declared secondary activities per venue (primary codes excluded), in c's order."""
    secondary = a[a.kind.eq('secondary')].groupby('company_id').activity_id.apply(set)
    primary = a[a.kind.eq('primary')].groupby('company_id').activity_id.apply(set)
    return np.array([len(secondary.get(cid, set()) - primary.get(cid, set())) for cid in c.company_id])


def build_features(c, a, d, root, demo=False, model_path=None, weights=None, secondary=None):
    weights = normalize_weights(weights)
    policy = {**SECONDARY_DEFAULT, **(secondary or {})}
    wanted = a.kind.eq('secondary') & (a.activity_id.str.startswith('56') | (policy['non56_weight'] > 0))
    relevant_ids = set(a.loc[a.kind.eq('primary') | wanted, 'activity_id'])
    used = d[d.activity_id.isin(relevant_ids)].reset_index(drop=True)
    if used.empty:
        raise ValueError('No usable activity descriptions were found for the supplied businesses.')
    embeddings, identity = description_embeddings(used, root, demo, model_path)
    lookup = {code: embeddings[i] for i, code in enumerate(used.activity_id)}
    n, dim = len(c), embeddings.shape[1]
    semantic = np.zeros((n, dim), dtype='float32')
    groups = {cid: group for cid, group in a.groupby('company_id', sort=False)}
    primary_codes, retained, secondary_counts = [], [], []
    for i, row in enumerate(c.itertuples()):
        group = groups.get(row.company_id)
        codes = [] if group is None else sorted(set(group.loc[group.kind.eq('primary'), 'activity_id']))
        primary_codes.append(codes)
        candidates = [] if group is None else sorted(set(group.loc[group.kind.eq('secondary'), 'activity_id']) - set(codes))
        secondary_counts.append(len(candidates))
        valid = [lookup[code] for code in codes if code in lookup]
        if not valid:
            retained.append([])
            continue
        p = normalize(np.mean(valid, axis=0).reshape(1, -1))[0]
        # Auxiliary codes never enter; secondaries count only if related to the primary activity, and
        # count less the more of them a business declares (A3: the business itself is not penalised,
        # its secondaries are trusted less, so its vector falls back towards its primary activity).
        trust = secondary_trust(len(candidates), policy)

        def related(food):
            scored = sorted([(float(p @ lookup[code]), code) for code in candidates
                             if code.startswith('56') == food and code in lookup], reverse=True)
            return [code for similarity, code in scored if similarity >= policy['min_similarity']][:policy['max_codes']]
        keep = related(True)
        other = related(False) if policy['non56_weight'] > 0 else []
        w_food, w_other = policy['weight'] * trust, policy['non56_weight'] * trust
        food = np.mean([lookup[code] for code in keep], axis=0) if keep else p
        vector = round(1 - w_food - w_other, 12) * p + w_food * food
        if w_other:
            vector = vector + w_other * (np.mean([lookup[code] for code in other], axis=0) if other else p)
        semantic[i] = vector
        retained.append(keep + other)
    semantic = normalize(semantic)
    # Fixed geographic RBF anchors in Greece; no network tiles or external enrichment.
    anchors = np.array([[lat, lon] for lat in [35., 37., 39., 41.] for lon in [20., 22., 24., 26., 28.]])
    gps = c[['latitude', 'longitude']].to_numpy(float)
    available = np.isfinite(gps).all(axis=1)
    delta = (np.nan_to_num(gps[:, None, :]) - anchors[None, :, :]) * np.array([111., 85.])
    geo = np.exp(-np.sum(delta ** 2, axis=2) / (2 * 130 ** 2))
    geo[~available] = 0
    regions = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(c[['prefecture_name']])
    geography = normalize(np.c_[normalize(geo) * np.sqrt(.7), regions * np.sqrt(.3), (~available).astype(float)])
    legal = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(c[['legal_type']])
    years = (pd.Timestamp(AS_OF) - pd.to_datetime(c.incorporation_date, errors='coerce')).dt.days / 365.25
    age = np.clip(years.fillna(years.median() if years.notna().any() else 4).to_numpy(), 0, 15)
    age_block = normalize(np.c_[np.exp(-((age[:, None] - np.array([2, 4, 6, 8])) ** 2) / 8), years.isna()])
    X = normalize(np.c_[semantic * np.sqrt(weights['activity']), geography * np.sqrt(weights['geography']),
                         legal * np.sqrt(weights['legal_type']), age_block * np.sqrt(weights['age'])]).astype('float32')
    traits = []
    for i, row in enumerate(c.itertuples()):
        values = {f'legal_type:{row.legal_type}', f'prefecture_name:{row.prefecture_name}',
                  f'municipality_name:{row.municipality_name}', f'is_branch:{row.is_branch}',
                  f'incorporation_year:{str(row.incorporation_date)[:4]}'}
        values |= {f'primary_activity:{code}' for code in primary_codes[i]}
        traits.append(values)
    return X, traits, primary_codes, {'embedding': identity, 'as_of': AS_OF,
        'missing_primary_venues': int(np.count_nonzero(np.linalg.norm(semantic, axis=1) == 0)),
        'secondary_codes_retained': sum(map(len, retained)), 'embedded_description_count': len(used),
        'secondary_policy': policy,
        'block_weights': weights, 'custom_weights': is_custom(weights)}
