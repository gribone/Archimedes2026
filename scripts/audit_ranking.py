"""Paired VAT holdouts comparing the shipped ranker, its predecessor and NACE."""
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from forcecrm.data import load_universe, read_portfolio
from forcecrm.features import build_features
from forcecrm.pipeline import evaluate_holdout

if __name__ == '__main__':
    root = Path('.')
    c, a, d, _ = load_universe(root)
    p = read_portfolio(root / 'ForceCRM_Hackathon_Pelatologio.xlsx', c)
    with threadpool_limits(limits=2):
        X, _, primary, _ = build_features(c, a, d, root)
        results = [evaluate_holdout(X, c, p, primary, seed) for seed in (2026, 17, 43, 101, 713)]
    out = root / 'outputs' / 'ranking-audit.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), 'utf-8')
    for k in (500, 2000):
        rows = [m for split in results for m in split['metrics'] if m['k'] == k]
        print(k, {name: round(sum(r[name] for r in rows)/len(rows), 4)
                  for name in ('recall','previous_recall','baseline_recall')})
