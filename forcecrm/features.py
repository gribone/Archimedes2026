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
    key = hashlib.sha256(json.dumps([texts, identity, stamps], ensure_ascii=False).encode()).hexdigest()
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
    values = encoder.encode(['query: ' + t for t in texts], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False).astype('float32')
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, values)
    return values, identity


def build_features(c, a, d, root, demo=False, model_path=None):
    relevant_ids = set(a.loc[a.kind.eq('primary') | (a.kind.eq('secondary') & a.activity_id.str.startswith('56')), 'activity_id'])
    used = d[d.activity_id.isin(relevant_ids)].reset_index(drop=True)
    if used.empty:
        raise ValueError('No usable activity descriptions were found for the supplied businesses.')
    embeddings, identity = description_embeddings(used, root, demo, model_path)
    lookup = {code: embeddings[i] for i, code in enumerate(used.activity_id)}
    n, dim = len(c), embeddings.shape[1]
    semantic = np.zeros((n, dim), dtype='float32')
    groups = {cid: group for cid, group in a.groupby('company_id', sort=False)}
    primary_codes, retained = [], []
    for i, row in enumerate(c.itertuples()):
        group = groups.get(row.company_id)
        codes = [] if group is None else sorted(set(group.loc[group.kind.eq('primary'), 'activity_id']))
        primary_codes.append(codes)
        valid = [lookup[code] for code in codes if code in lookup]
        if not valid:
            retained.append([])
            continue
        p = normalize(np.mean(valid, axis=0).reshape(1, -1))[0]
        candidates = [] if group is None else sorted(set(group.loc[group.kind.eq('secondary'), 'activity_id']) - set(codes))
        # Auxiliary and unrelated self-declared secondary codes never enter representation.
        scored = sorted([(float(p @ lookup[code]), code) for code in candidates
                         if code.startswith('56') and code in lookup], reverse=True)
        keep = [code for similarity, code in scored if similarity >= .70][:3]
        secondary = np.mean([lookup[code] for code in keep], axis=0) if keep else p
        semantic[i] = .85 * p + .15 * secondary
        retained.append(keep)
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
    X = normalize(np.c_[semantic * np.sqrt(.65), geography * np.sqrt(.20),
                         legal * np.sqrt(.10), age_block * np.sqrt(.05)]).astype('float32')
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
        'block_weights': {'activity': .65, 'geography': .20, 'legal_type': .10, 'age': .05}}
