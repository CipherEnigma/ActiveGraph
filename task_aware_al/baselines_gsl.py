"""Graph Structure Learning (GSL) baselines — "why query at all?" comparison.

Two GSL methods that add edges WITHOUT querying an oracle:
  1. Feature-kNN: add the B most feature-similar non-edges
  2. VGAE-GSL: use our VGAE to add the B highest-probability non-edges

We add exactly B edges (matching the active querying budget) and compare
downstream accuracy.  The key question: can you match active querying by
just hallucinating edges from features / embeddings?
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cora_exp import (
    DATASET_LOADERS, Instance, lp_solve, lp_accuracy,
    gcn_accuracy, train_vgae,
)


def gsl_feature_knn(A_obs, X, B, targets=None):
    """Add top-B new edges ranked by cosine feature similarity.
    If targets is provided, only consider edges touching at least one target."""
    if sp.issparse(X):
        X_dense = np.asarray(X.todense())
    else:
        X_dense = np.asarray(X)

    A_dense = np.asarray(A_obs.todense())

    if targets is not None:
        sim = cosine_similarity(X_dense[targets], X_dense)
        mask = np.zeros_like(sim, dtype=bool)
        for ti, t in enumerate(targets):
            mask[ti] = (A_dense[t] == 0)
            mask[ti, t] = False
        sim[~mask] = -np.inf
        flat = sim.ravel()
        top_idx = np.argsort(-flat)[:B]
        rows_local = top_idx // sim.shape[1]
        cols = top_idx % sim.shape[1]
        rows = np.array(targets)[rows_local]
    else:
        sim = cosine_similarity(X_dense)
        np.fill_diagonal(sim, -np.inf)
        sim[A_dense > 0] = -np.inf
        sim = np.triu(sim, k=1)
        sim[sim == 0] = -np.inf
        flat = sim.ravel()
        top_idx = np.argsort(-flat)[:B]
        n = A_obs.shape[0]
        rows = top_idx // n
        cols = top_idx % n

    edges = list(zip(rows.tolist(), cols.tolist()))
    return edges


def gsl_vgae(A_obs, X, B, seed=0):
    """Add top-B new edges ranked by VGAE link probability."""
    _, Z = train_vgae(A_obs, X, seed=seed, epochs=200)
    sim = Z @ Z.T
    A_dense = np.asarray(A_obs.todense())
    np.fill_diagonal(sim, -np.inf)
    sim[A_dense > 0] = -np.inf
    sim = np.triu(sim, k=1)
    sim[sim == 0] = -np.inf
    flat = sim.ravel()
    top_idx = np.argsort(-flat)[:B]
    n = A_obs.shape[0]
    rows = top_idx // n
    cols = top_idx % n
    return list(zip(rows.tolist(), cols.tolist()))


def evaluate_gsl(method, seed, data, n_targets, budget, per_class=20,
                 model_type="gcn"):
    """Run a GSL method, add B edges, measure LP & GCN accuracy."""
    inst = Instance(seed, data, n_targets=n_targets, per_class=per_class)
    A_obs = inst.A_obs()
    X = inst.X

    t0 = time.perf_counter()

    if method == "feature_knn":
        edges = gsl_feature_knn(A_obs, X, budget, targets=inst.targets)
    elif method == "feature_knn_global":
        edges = gsl_feature_knn(A_obs, X, budget, targets=None)
    elif method == "vgae_gsl":
        edges = gsl_vgae(A_obs, X, budget, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    t_gsl = time.perf_counter() - t0

    A_aug = A_obs.tolil()
    true_count = 0
    for i, j in edges:
        A_aug[i, j] = 1.0
        A_aug[j, i] = 1.0
        if data[0][i, j] > 0:
            true_count += 1
    A_aug = A_aug.tocsr()

    F_obs = lp_solve(A_obs, inst.y, inst.lab, inst.k)
    lp_obs = lp_accuracy(inst, F_obs)
    F_aug = lp_solve(A_aug, inst.y, inst.lab, inst.k)
    lp_aug = lp_accuracy(inst, F_aug)

    gcn_obs = gcn_accuracy(A_obs, X, inst.y, inst.train, inst.targets, inst.k,
                           seed=seed, model_type=model_type)
    gcn_aug = gcn_accuracy(A_aug, X, inst.y, inst.train, inst.targets, inst.k,
                           seed=seed, model_type=model_type)

    return {
        "seed": seed, "method": method, "budget": budget,
        "edges_added": len(edges),
        "true_edges": true_count,
        "precision": true_count / max(len(edges), 1),
        "lp_obs": lp_obs, "lp_aug": lp_aug, "lp_gain": lp_aug - lp_obs,
        "gcn_obs": gcn_obs, "gcn_aug": gcn_aug, "gcn_gain": gcn_aug - gcn_obs,
        "time": t_gsl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora", choices=list(DATASET_LOADERS))
    ap.add_argument("--method", default="feature_knn",
                    choices=["feature_knn", "feature_knn_global", "vgae_gsl"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--n_targets", type=int, default=200)
    ap.add_argument("--model_type", default="gcn", choices=["gcn", "gat", "sage"])
    ap.add_argument("-o", "--out", default="gsl_results.json")
    args = ap.parse_args()

    data = DATASET_LOADERS[args.dataset]()
    print(f"Dataset: {args.dataset}, method: {args.method}, n={data[0].shape[0]}")

    results = []
    for s in range(args.seed0, args.seed0 + args.seeds):
        print(f"\n=== Seed {s} ===")
        r = evaluate_gsl(args.method, s, data, args.n_targets, args.budget,
                         model_type=args.model_type)
        results.append(r)
        print(f"  LP: {r['lp_obs']:.4f} -> {r['lp_aug']:.4f} (gain={r['lp_gain']:+.4f})")
        print(f"  GCN: {r['gcn_obs']:.4f} -> {r['gcn_aug']:.4f} (gain={r['gcn_gain']:+.4f})")
        print(f"  True edges: {r['true_edges']}/{r['edges_added']} ({r['precision']:.1%})")
        print(f"  Time: {r['time']:.1f}s")

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
