"""E0: shared setup for the task-aware active-learning experiments.

One Instance = one seeded problem: a cSBM graph, a partly observed copy of it (GraphEnv), a
labelled set, and a TARGET set (the user's query: ~15 unlabelled nodes to classify).

What the algorithm may use: observed edges, labels of train nodes, the target set, the budget,
and p_in/p_out ESTIMATED from the initial observed graph. What it may not use: the true graph,
any non-train label, or the true p_in/p_out (only the *_knownp control uses those).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from data import make_csbm  # noqa: E402
from env import GraphEnv  # noqa: E402

N, K = 90, 3
GRAPHS = {"locked": dict(p_in=0.13, p_out=0.006), "dense": dict(p_in=0.25, p_out=0.02)}
OBSERVE_FRAC = 0.15
N_TARGETS = 15
ALPHA = 0.99


def solve_lp(A, y, train_idx, k, alpha=ALPHA):
    """Damped label propagation in closed form, labelled rows clamped:
    F_U = (I - a S_UU)^-1 a S_UL Y_L with S = D^-1 A. Nodes no label reaches get F = 0."""
    n = A.shape[0]
    d = A.sum(axis=1)
    S = np.divide(A, d[:, None], out=np.zeros_like(A), where=d[:, None] > 0)
    lab = np.zeros(n, dtype=bool)
    lab[train_idx] = True
    U = ~lab
    YL = np.zeros((int(lab.sum()), k))
    YL[np.arange(YL.shape[0]), y[lab]] = 1.0
    F = np.zeros((n, k))
    F[lab] = YL
    F[U] = np.linalg.solve(np.eye(int(U.sum())) - alpha * S[np.ix_(U, U)], alpha * S[np.ix_(U, lab)] @ YL)
    return F


_FALLBACK = np.random.default_rng(0).integers(0, K, size=N)


def predict(F):
    """argmax; unreached nodes keep a fixed per-node guess, so a prediction changes only when
    a label actually reaches the node."""
    unreached = np.abs(F).sum(axis=1) < 1e-12
    return np.where(unreached, _FALLBACK, F.argmax(axis=1))


def beliefs(F):
    mass = F.sum(axis=1, keepdims=True)
    return np.divide(F, mass, out=np.full_like(F, 1.0 / F.shape[1]), where=mass > 1e-12)


def reach(A, train_idx, idx):
    """Fraction of idx connected (by any path) to at least one labelled node."""
    _, comp = connected_components(csr_matrix(A > 0), directed=False)
    return float(np.isin(comp[idx], comp[train_idx]).mean())


def estimate_densities(A_obs, y, train_idx, observe_frac=OBSERVE_FRAC):
    """p_in / p_out from the INITIAL observed graph only. Overall density from all observed
    edges (corrected for the known observe fraction); the within-class share of edges from
    observed edges between labelled nodes, add-one smoothed. Noisy at 15% labels (about 13
    labelled nodes), and biased toward less homophily -- a stated limitation."""
    n = A_obs.shape[0]
    d = (A_obs.sum() / 2) / (observe_frac * n * (n - 1) / 2)
    lab = np.asarray(train_idx)
    iu = np.triu_indices(len(lab), 1)
    same = (y[lab][:, None] == y[lab][None, :])[iu]
    e = (A_obs[np.ix_(lab, lab)] > 0)[iu]
    h = ((e & same).sum() + 1.0) / (e.sum() + 2.0)
    pi = 1.0 / (int(y.max()) + 1)
    p_in = float(np.clip(d * h / pi, 1e-3, 0.5))
    p_out = float(np.clip(d * (1 - h) / (1 - pi), 1e-3, 0.5))
    return p_in, p_out


REMOVALS = ("random", "target_hidden", "hub_observed")


def apply_removal(env, scheme, targets):
    """Re-pick WHICH true edges are initially observed (same count as GraphEnv's random pick).
    target_hidden: no observed edge touches a target (targets start cold).
    hub_observed : observation probability proportional to deg_i * deg_j (low-degree nodes start cold)."""
    edges = env.true_edges
    n_init = int(len(edges) * OBSERVE_FRAC)
    if scheme == "target_hidden":
        tset = set(targets.tolist())
        ok = [k for k, (i, j) in enumerate(edges) if i not in tset and j not in tset]
        pool = ok if len(ok) >= n_init else list(range(len(edges)))
        chosen = env.rng.choice(pool, size=n_init, replace=False)
    else:
        deg = env.A_true.sum(axis=1)
        w = np.array([deg[i] * deg[j] for i, j in edges], dtype=float)
        chosen = env.rng.choice(len(edges), size=n_init, replace=False, p=w / w.sum())
    env.A_obs[:] = 0.0
    env.revealed = set()
    for k in chosen:
        i, j = edges[k]
        env._set_edge(i, j)
        env.revealed.add((i, j))


class Instance:
    def __init__(self, seed, train_frac, graph="locked", feat_signal=0.1, n_targets=N_TARGETS,
                 removal="random"):
        gp = GRAPHS[graph]
        self.y, self.A_true, self.X = make_csbm(n=N, k=K, feat_signal=feat_signal, seed=seed, **gp)
        self.env = GraphEnv(self.y, self.A_true, self.X, observe_frac=OBSERVE_FRAC,
                            train_frac=train_frac, val_frac=0.0, seed=seed)
        self.k = K
        rng = np.random.default_rng(10_000 + seed)
        pool = self.env.test_idx
        self.targets = np.sort(rng.choice(pool, size=min(n_targets, len(pool)), replace=False))
        if removal != "random":
            apply_removal(self.env, removal, self.targets)
        self.p_true = (gp["p_in"], gp["p_out"])
        self.p_est = estimate_densities(self.env.A_obs.astype(float), self.y, self.env.train_idx)
        self.ceiling = self.target_accuracy(self.A_true.astype(float))

    def A_obs(self):
        return self.env.A_obs.astype(np.float64)

    def target_accuracy(self, A):
        pred = predict(solve_lp(A, self.y, self.env.train_idx, self.k))
        return float((pred[self.targets] == self.y[self.targets]).mean())


def paired(a, b):
    """mean difference, sd, p-value for paired samples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = stats.ttest_rel(a, b).pvalue if d.std() > 0 else 1.0
    return float(d.mean()), float(d.std(ddof=1)), float(p)
