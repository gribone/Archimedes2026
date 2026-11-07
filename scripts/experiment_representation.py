"""Representation experiment: does centering activity embeddings or finer geography make profiles less geographic?

Reports 5-split holdout recall, chosen k, each block's share of the separation between profile centres,
and the kinds of statistically supported traits per profile. Results are recorded in docs/ARCHITECTURE.md.

    .venv/bin/python -m scripts.experiment_representation --portfolio ForceCRM_Hackathon_Pelatologio.xlsx [--sweep]
"""
import argparse
import sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, OneHotEncoder
from threadpoolctl import threadpool_limits
from forcecrm.data import load_universe, read_portfolio
from forcecrm.features import build_features
from forcecrm.enrichment import enrich
from forcecrm.pipeline import evaluate_holdout, fit_portfolio
from forcecrm.evidence import significant_features

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=Path('.'))
parser.add_argument('--portfolio', type=Path, required=True)
parser.add_argument('--sweep', action='store_true', help='Also sweep partial centering x activity weight')
args = parser.parse_args()
root, xlsx = args.data, args.portfolio
c, a, d, _ = load_universe(root); p = read_portfolio(xlsx, c)
X0, traits, primary, _ = build_features(c, a, d, root)
enr, _ = enrich(c); c['hotel_band'] = enr.hotel_band
for t, v in zip(traits, enr.itertuples()):
    if pd.notna(v.hotel_band): t.add(f'hotel_band:{v.hotel_band}')
    if v.status == 'matched': t.add(f'osm_amenity:{v.amenity}')
ng = 20 + c.prefecture_name.nunique() + 1; nl = c.legal_type.nunique()
S = X0[:, :384] / np.sqrt(.65)
G = X0[:, 384:384+ng] / np.sqrt(.20)
L = X0[:, 384+ng:384+ng+nl] / np.sqrt(.10)
A = X0[:, 384+ng+nl:] / np.sqrt(.05)
has = np.linalg.norm(S, axis=1) > 0

def centered(S, alpha=1.0):
    out = S.copy(); out[has] = normalize(S[has] - alpha * S[has].mean(axis=0)); return out

def fine_geo(sigma_km=25, step=.35, muni_w=.3):
    gps = c[['latitude', 'longitude']].to_numpy(float); ok = np.isfinite(gps).all(axis=1)
    lat = np.arange(34.6, 41.9, step); lon = np.arange(19.3, 29.8, step)
    anchors = np.array([[x, y] for x in lat for y in lon])
    delta = (np.nan_to_num(gps[:, None, :]) - anchors[None]) * np.array([111., 85.])
    rbf = np.exp(-np.sum(delta**2, axis=2) / (2 * sigma_km**2)); rbf[~ok] = 0
    keep = rbf.max(axis=0) > .05; rbf = rbf[:, keep]
    muni = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(c[['municipality_name']])
    return normalize(np.c_[normalize(rbf) * np.sqrt(1 - muni_w), muni * np.sqrt(muni_w), (~ok).astype(float)])

def assemble(S, G, w=(.65, .20, .10, .05)):
    return normalize(np.c_[S*np.sqrt(w[0]), G*np.sqrt(w[1]), L*np.sqrt(w[2]), A*np.sqrt(w[3])]).astype('float32'), \
           [S.shape[1], G.shape[1], L.shape[1], A.shape[1]]

def kind(f):
    k = f.split(':')[0]
    return {'prefecture_name': 'geo', 'municipality_name': 'geo', 'hotel_band': 'tourism',
            'primary_activity': 'activity', 'osm_amenity': 'activity', 'legal_type': 'legal'}.get(k, k)

def run(name, X, dims):
    with threadpool_limits(limits=2):
        res = [evaluate_holdout(X, c, p, primary, s) for s in (2026, 17, 43, 101, 713)]
        fit = fit_portfolio(X, c, p)
    r = {k: np.mean([m['recall'] for x in res for m in x['metrics'] if m['k'] == k]) for k in (500, 2000)}
    cen = fit['centers']; bounds = np.cumsum([0] + dims)
    tb = sum(np.sum((cen[i]-cen[j])**2) for i in range(len(cen)) for j in range(i+1, len(cen))) or 1
    sep = [sum(np.sum((cen[i, bounds[b]:bounds[b+1]]-cen[j, bounds[b]:bounds[b+1]])**2)
               for i in range(len(cen)) for j in range(i+1, len(cen))) / tb for b in range(4)]
    masks = []
    for j in range(len(cen)):
        m = np.zeros(len(c), bool); m[fit['indices'][fit['labels'] == j]] = True; masks.append(m)
    ev, _ = significant_features(c, traits, masks, d)
    sizes = [len(set(fit['vats'][fit['labels'] == j])) for j in range(len(cen))]
    print(f'\n== {name}: R@500 {r[500]:.1%}  R@2000 {r[2000]:.1%}  k={len(cen)} sizes={sizes}')
    print('   separation activity/geo/legal/age: ' + ' / '.join(f'{x:.0%}' for x in sep))
    for j, rows in enumerate(ev):
        top = [f"{x['label'][:48]} ({x['count']}/{x['total']}, {x['lift']:.1f}x)" for x in rows[:4]]
        print(f'   ICP-{j+1} kinds={dict(Counter(kind(x["feature"]) for x in rows))}')
        for t in top: print('      ', t)
    sys.stdout.flush()

Gf = fine_geo()
variants = {'A baseline': (S, G), 'B centered activity': (centered(S), G),
            'C finer geography': (S, Gf), 'D centered + finer': (centered(S), Gf)}
for name, (s, g) in variants.items():
    run(name, *assemble(s, g))

if args.sweep:
    for alpha in (0, .5, .8, 1.0):
        for wa in (.65, .45, .30):
            rest = (1 - wa) / .35
            w = (wa, .20 * rest, .10 * rest, .05 * rest)
            X, dims = assemble(centered(S, alpha), G, w)
            run(f'alpha={alpha} activity_w={wa}', X, dims)
