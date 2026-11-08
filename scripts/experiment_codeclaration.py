"""T5 experiment: ΚΑΔ co-declaration vectors (PPMI + SVD over which codes businesses declare together),
alone or mixed 50/50 with the E5 description embeddings, as the activity representation.

Reports 5-split holdout recall (tourism on), profile count/sizes and the kinds of significant traits.
Results (2026-11-08) are recorded in docs/ARCHITECTURE.md §3. Not shipped: a user decision (TODO T5).
    .venv/bin/python -m scripts.experiment_codeclaration --portfolio ForceCRM_Hackathon_Pelatologio.xlsx
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import forcecrm.features as F
from forcecrm.data import load_universe, read_portfolio
from forcecrm.enrichment import market_context, enrich
from forcecrm.pipeline import evaluate_holdout, fit_portfolio
from forcecrm.evidence import significant_features

_p = argparse.ArgumentParser()
_p.add_argument('--data', type=Path, default=Path('.'))
_p.add_argument('--portfolio', type=Path, required=True)
_a = _p.parse_args()
root = _a.data; c, a, d, _ = load_universe(root); p = read_portfolio(_a.portfolio, c)
enr, _ = enrich(c); c['hotel_band'] = enr.hotel_band
# Co-declaration matrix over primary + secondary codes (auxiliary ignored), one row per venue.
use = a[a.kind.isin(['primary', 'secondary'])]
codes = sorted(set(use.activity_id)); col = {k: i for i, k in enumerate(codes)}
rows = {cid: i for i, cid in enumerate(sorted(set(use.company_id)))}
B = sp.csr_matrix((np.ones(len(use)), ([rows[x] for x in use.company_id], [col[x] for x in use.activity_id])),
                  shape=(len(rows), len(codes)))
B.data[:] = 1
C = (B.T @ B).toarray().astype(float); n = C.diagonal().copy(); N = B.shape[0]
with np.errstate(divide='ignore', invalid='ignore'):
    pmi = np.log(C * N / np.outer(n, n))
ppmi = np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0); np.fill_diagonal(ppmi, 0)
vec = normalize(TruncatedSVD(64, random_state=42).fit_transform(ppmi))
PPMI = {k: vec[i] for k, i in col.items()}
original = F.description_embeddings

def patched(alpha):
    def emb(descriptions, root, demo=False, model_path=None):
        values, identity = original(descriptions, root, demo, model_path)
        co = np.array([PPMI.get(k, np.zeros(64)) for k in descriptions.activity_id])
        mixed = normalize(np.c_[values * np.sqrt(alpha), co * np.sqrt(1 - alpha)]).astype('float32')
        return mixed, dict(identity, ppmi_alpha=alpha)
    return emb

traits_base = None
def run(name, alpha):
    global traits_base
    F.description_embeddings = original if alpha is None else patched(alpha)
    with threadpool_limits(limits=2):
        X, traits, primary, meta = F.build_features(c, a, d, root)
        res = [evaluate_holdout(X, c, p, primary, s) for s in (2026, 17, 43, 101, 713)]
        fit = fit_portfolio(X, c, p)
    for t, v in zip(traits, enr.itertuples()):
        if isinstance(v.hotel_band, str): t.add(f'hotel_band:{v.hotel_band}')
    r = {k: np.mean([m['recall'] for x in res for m in x['metrics'] if m['k'] == k]) for k in (500, 2000)}
    masks = []
    for j in range(len(fit['centers'])):
        m = np.zeros(len(c), bool); m[fit['indices'][fit['labels'] == j]] = True; masks.append(m)
    ev, _ = significant_features(c, traits, masks, d)
    kinds = lambda rows: Counter({'primary_activity': 'activity', 'legal_type': 'legal', 'hotel_band': 'tourism'}.get(f['feature'].split(':')[0], 'geo') for f in rows)
    sizes = [len(set(fit['vats'][fit['labels'] == j])) for j in range(len(fit['centers']))]
    print(f'\n== {name}: R@500 {r[500]:.1%} | R@2000 {r[2000]:.1%} | k={len(sizes)} sizes={sizes}', flush=True)
    names = dict(zip(d.activity_id, d.description))
    for j, rows_ in enumerate(ev):
        top_act = [f"{names.get(f['feature'].split(':')[1], '')[:40]} {f['count']}/{f['total']} {f['lift']:.1f}x" for f in rows_ if f['feature'].startswith('primary_activity')][:2]
        print(f'   ICP-{j+1}: {dict(kinds(rows_))} | first traits: {[f["label"][:38] for f in rows_[:3]]} | activity: {top_act}', flush=True)

run('A shipped (E5 descriptions)', None)
run('B co-declaration only (PPMI)', 0.0)
run('C 50% E5 + 50% co-declaration', 0.5)
