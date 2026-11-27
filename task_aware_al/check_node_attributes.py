"""Do node attributes help classification here?

Compares, on the initially observed graph (no queries yet), accuracy on all unlabelled nodes of:
  features   logistic regression on X, trained on the labelled nodes only
  lp         label propagation on the observed graph (unreached nodes get a fixed random class)
  lp+fallback  lp, but unreached nodes take the features-only prediction
  lp+prior   propagation seeded with the features-only probabilities:
             F <- a * S F + (1 - a) * P0, labelled rows clamped

Swept over feature strength, label density and graph density. 30 seeds, paired.
"""
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from data import make_csbm
from env import GraphEnv
from labelprop import propagate, predict

N_SEEDS = 30
N, K = 90, 3
FEAT_SIGNALS = [0.0, 0.1, 0.3, 0.5, 1.0]
LABEL_FRACS = [0.15, 0.40]
GRAPHS = {"locked(p_in=.13,p_out=.006)": (0.13, 0.006), "dense(p_in=.25,p_out=.02)": (0.25, 0.02)}
ALPHA = 0.8


def prior_propagate(A, P0, train_idx, y, k, alpha=ALPHA, iters=200):
    n = A.shape[0]
    deg = A.sum(axis=1)
    S = np.divide(A, deg[:, None], out=np.zeros((n, n)), where=deg[:, None] > 0)
    Y = np.zeros((n, k))
    Y[train_idx, y[train_idx]] = 1.0
    F = P0.copy()
    F[train_idx] = Y[train_idx]
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * P0
        F[train_idx] = Y[train_idx]
    return F


def one_run(p_in, p_out, feat_signal, train_frac, seed):
    y, A_true, X = make_csbm(n=N, k=K, p_in=p_in, p_out=p_out, feat_signal=feat_signal, seed=seed)
    env = GraphEnv(y, A_true, X, observe_frac=0.15, train_frac=train_frac, val_frac=0.0, seed=seed)
    tr = env.train_idx
    ev = np.setdiff1d(np.arange(N), tr)
    A = env.observed_graph().astype(float)

    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
    P0 = np.zeros((N, K))
    P0[:, lr.classes_] = lr.predict_proba(X)
    feat_pred = P0.argmax(axis=1)

    F = propagate(A, y, tr, K)
    lp_pred, unreached = predict(F)
    fb_pred = np.where(unreached, feat_pred, lp_pred)
    prior_pred = prior_propagate(A, P0, tr, y, K).argmax(axis=1)

    acc = lambda p: float((p[ev] == y[ev]).mean())
    return {"features": acc(feat_pred), "lp": acc(lp_pred), "lp+fallback": acc(fb_pred),
            "lp+prior": acc(prior_pred), "unreached": float(unreached[ev].mean())}


def main():
    methods = ["features", "lp", "lp+fallback", "lp+prior"]
    for gname, (p_in, p_out) in GRAPHS.items():
        for tf in LABEL_FRACS:
            print(f"\n=== {gname}, labelled={int(tf * 100)}%, {N_SEEDS} seeds (chance = 0.333) ===")
            print(f"{'feat_signal':>11} {'unreached':>9} " + " ".join(f"{m:>12}" for m in methods)
                  + f" {'prior-lp':>16} {'features-chance':>16}")
            for fs in FEAT_SIGNALS:
                runs = [one_run(p_in, p_out, fs, tf, s) for s in range(N_SEEDS)]
                col = {m: np.array([r[m] for r in runs]) for m in methods}
                un = np.mean([r["unreached"] for r in runs])
                d = col["lp+prior"] - col["lp"]
                p1 = stats.ttest_rel(col["lp+prior"], col["lp"]).pvalue if d.std() > 0 else 1.0
                p2 = stats.ttest_1samp(col["features"], 1 / 3).pvalue
                print(f"{fs:>11.1f} {un:>9.2f} " + " ".join(f"{col[m].mean():>12.3f}" for m in methods)
                      + f" {d.mean():>+8.3f}{'*' if p1 < .05 else ' '}({p1:.3f})"
                      + f" {col['features'].mean() - 1 / 3:>+8.3f}{'*' if p2 < .05 else ' '}({p2:.3f})")
    print("\n* = p < 0.05 (prior-lp: paired t-test; features-chance: one-sample t-test vs 1/3)")


if __name__ == "__main__":
    main()
