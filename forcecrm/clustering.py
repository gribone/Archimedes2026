from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize


def fit_spherical(X, weights, k, seed=42, restarts=6):
    """Revenue-weighted spherical k-means; one cosine distance for fit and retrieval."""
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        chosen = [int(rng.choice(len(X), p=weights / weights.sum()))]
        for __ in range(1, k):
            distances = np.maximum(0, 1 - (X @ X[chosen].T).max(axis=1)) * weights
            distances[chosen] = 0
            if distances.sum() < 1e-10:
                remaining = np.setdiff1d(np.arange(len(X)), chosen)
                chosen.append(int(rng.choice(remaining)))
            else:
                chosen.append(int(rng.choice(len(X), p=distances / distances.sum())))
        centers = X[chosen].copy()
        previous = None
        for __ in range(100):
            labels = np.argmax(X @ centers.T, axis=1)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
            for j in range(k):
                members = labels == j
                if members.any():
                    centers[j] = normalize(np.average(X[members], weights=weights[members], axis=0).reshape(1, -1))[0]
                else:
                    centers[j] = X[np.argmax((1 - (X @ centers.T).max(axis=1)) * weights)]
        labels = np.argmax(X @ centers.T, axis=1)
        loss = float(np.average(1 - np.sum(X * centers[labels], axis=1), weights=weights))
        if best is None or loss < best[0]:
            best = loss, centers.copy(), labels.copy()
    return best[1], best[2], best[0]


def discover(X, weights, vats, max_k=6):
    centers, labels, loss = fit_spherical(X, weights, 1)
    chosen = (centers, labels)
    best_score = .18
    diagnostics = [{'k': 1, 'loss': loss, 'silhouette': None, 'selection_score': .18, 'eligible': True}]
    for k in range(2, min(max_k, len(set(vats)) // 5, len(X) - 1) + 1):
        candidate, assignment, loss = fit_spherical(X, weights, k)
        sizes = [len(set(vats[assignment == j])) for j in range(k)]
        eligible = min(sizes) >= 5
        sil = float(silhouette_score(X, assignment, metric='cosine', sample_size=min(1000, len(X)), random_state=42)) if len(set(assignment)) > 1 else -1.
        score = sil - .025 * (k - 1)
        diagnostics.append({'k': k, 'loss': loss, 'silhouette': sil, 'selection_score': score, 'eligible': eligible})
        if eligible and score > best_score:
            chosen, best_score = (candidate, assignment), score
    return *chosen, diagnostics


def stability(X, weights, vats, centers, labels, repetitions=20):
    rng = np.random.default_rng(123)
    unique = np.unique(vats)
    per_profile = [[] for _ in centers]
    aris = []
    for b in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values, counts = np.unique(sampled, return_counts=True)
        count = dict(zip(values, counts))
        multipliers = np.array([count.get(v, 0) for v in vats])
        mask = multipliers > 0
        boot, _, _ = fit_spherical(X[mask], weights[mask] * multipliers[mask], len(centers), seed=500 + b, restarts=3)
        predicted = np.argmax(X @ boot.T, axis=1)
        overlap = np.zeros((len(centers), len(centers)))
        # Membership is compared at VAT level to prevent large chains dominating stability.
        for i in range(len(centers)):
            left = set(vats[labels == i])
            for j in range(len(centers)):
                right = set(vats[predicted == j])
                overlap[i, j] = len(left & right) / max(1, len(left | right))
        rows, cols = linear_sum_assignment(-overlap)
        for i, j in zip(rows, cols):
            per_profile[i].append(float(overlap[i, j]))
        aris.append(float(adjusted_rand_score(labels, predicted)))
    return [{'mean': float(np.mean(v)), 'p10': float(np.quantile(v, .10)), 'method': 'VAT bootstrap matched Jaccard',
             'repetitions': repetitions} for v in per_profile], float(np.mean(aris))
