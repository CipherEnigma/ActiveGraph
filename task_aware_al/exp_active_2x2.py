"""E1 + E2: does ADAPTIVE (active) selection beat STATIC, and does TASK-AWARE beat task-agnostic?

Strategies (each run static = all queries chosen once from the initial state, and adaptive =
re-scored after every answer; random has no state so it is adaptive only):
  likely-link heuristics : common_nbrs, adamic_adar
  ALPINE-style / AL      : maxdeg, pagerank, uncertainty
  value (task-aware)     : P(real) x expected gain in self-estimated accuracy on the TARGET nodes
  value_all              : same objective over ALL unlabelled nodes (task-agnostic ablation)
  value_knownp           : value with the true p_in/p_out (sensitivity control for the estimates)

Task: classify ~15 target nodes; classifier = label propagation. Per checkpoint we record
accuracy on targets, reach (targets connected to any label), confidence, real edges found.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import Instance, solve_lp, predict, beliefs, reach, paired  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

SHORTLIST = 100


def _IJ(cands):
    a = np.array(cands)
    return a[:, 0], a[:, 1]


def score_random(inst, cands, **kw):
    return inst.env.rng.random(len(cands))


def score_cn(inst, cands, **kw):
    A = inst.A_obs()
    I, J = _IJ(cands)
    return (A @ A)[I, J]


def score_aa(inst, cands, **kw):
    A = inst.A_obs()
    deg = A.sum(axis=1)
    w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    I, J = _IJ(cands)
    return ((A * w) @ A)[I, J]


def score_maxdeg(inst, cands, **kw):
    deg = inst.A_obs().sum(axis=1)
    I, J = _IJ(cands)
    return deg[I] + deg[J]


def score_pagerank(inst, cands, damping=0.85, iters=100, **kw):
    A = inst.A_obs()
    n = A.shape[0]
    deg = A.sum(axis=1)
    dangling = deg == 0
    P = np.divide(A, deg[:, None], out=np.zeros_like(A), where=~dangling[:, None])
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        pr = damping * (pr @ P + pr[dangling].sum() / n) + (1 - damping) / n
    I, J = _IJ(cands)
    return pr[I] + pr[J]


def score_uncertainty(inst, cands, **kw):
    q = beliefs(solve_lp(inst.A_obs(), inst.y, inst.env.train_idx, inst.k))
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    I, J = _IJ(cands)
    return ent[I] + ent[J]


def score_value(inst, cands, objective="targets", known_p=False, static_budget=None, **kw):
    env = inst.env
    A = inst.A_obs()
    F = solve_lp(A, inst.y, env.train_idx, inst.k)
    q = beliefs(F)
    I, J = _IJ(cands)
    ps = (q[I] * q[J]).sum(axis=1)
    p_in, p_out = inst.p_true if known_p else inst.p_est
    p_real = p_in * ps + p_out * (1 - ps)
    conf = q.max(axis=1)

    if objective == "targets":
        obj = inst.targets
        _, comp = connected_components(csr_matrix(A > 0), directed=False)
        in_target_comp = np.isin(comp, comp[obj])
        relevant = in_target_comp[I] | in_target_comp[J]  # exact: other components cannot change
    else:
        obj = np.setdiff1d(np.arange(A.shape[0]), env.train_idx)
        relevant = np.ones(len(cands), dtype=bool)

    # shortlist by P(real) x confidence gap: an edge informs only from a node that knows
    # to one that doesn't
    key = p_real * np.abs(conf[I] - conf[J]) * relevant
    n_short = SHORTLIST if static_budget is None else max(SHORTLIST, 2 * static_budget)
    top = np.argsort(-(key + 1e-12 * env.rng.random(len(cands))))[:n_short]

    base = conf[obj].mean()
    scores = 1e-9 * p_real * relevant
    for c in top:
        if key[c] <= 0:
            break
        A2 = A.copy()
        A2[I[c], J[c]] = A2[J[c], I[c]] = 1.0
        conf2 = beliefs(solve_lp(A2, inst.y, env.train_idx, inst.k))[obj].max(axis=1).mean()
        scores[c] += p_real[c] * (conf2 - base)
    return scores


SCORERS = {
    "random": score_random,
    "common_nbrs": score_cn,
    "adamic_adar": score_aa,
    "maxdeg": score_maxdeg,
    "pagerank": score_pagerank,
    "uncertainty": score_uncertainty,
    "value": lambda inst, cands, **kw: score_value(inst, cands, "targets", False, **kw),
    "value_all": lambda inst, cands, **kw: score_value(inst, cands, "all", False, **kw),
    "value_knownp": lambda inst, cands, **kw: score_value(inst, cands, "targets", True, **kw),
}
COMBOS = [("random", "adaptive")] + [(s, m) for s in SCORERS if s != "random" for m in ("static", "adaptive")]


def frontier_filter(inst, cands):
    """Frontier-only exploration: a pair may be queried only if an endpoint is already reachable
    from the target nodes through observed edges (the target nodes themselves count)."""
    A = inst.A_obs()
    _, comp = connected_components(csr_matrix(A > 0), directed=False)
    explored = np.isin(comp, comp[inst.targets])
    I, J = _IJ(cands)
    keep = explored[I] | explored[J]
    return [c for c, k in zip(cands, keep) if k] if keep.any() else cands


def checkpoints(budget):
    return sorted(set(range(0, min(50, budget) + 1, 5)) | set(range(60, budget + 1, 10)) | {budget})


def run_policy(seed, train_frac, strategy, mode, budget, graph, feat_signal, removal="random"):
    inst = Instance(seed, train_frac, graph, feat_signal, removal=removal)
    env = inst.env
    t0 = time.perf_counter()
    e0 = int(env.A_obs.sum() / 2)
    cps = set(checkpoints(budget))
    rec = {m: {} for m in ("acc", "reach", "conf", "edges")}

    def snap(step):
        A = inst.A_obs()
        F = solve_lp(A, inst.y, env.train_idx, inst.k)
        rec["acc"][step] = float((predict(F)[inst.targets] == inst.y[inst.targets]).mean())
        rec["conf"][step] = float(beliefs(F)[inst.targets].max(axis=1).mean())
        rec["reach"][step] = reach(A, env.train_idx, inst.targets)
        rec["edges"][step] = int(env.A_obs.sum() / 2) - e0

    snap(0)
    queue = None
    if mode == "static" and strategy != "random":
        cands = env.candidates()
        s = SCORERS[strategy](inst, cands, static_budget=budget)
        order = np.lexsort((env.rng.random(len(s)), -s))
        queue = [cands[k] for k in order[:budget]]
    for step in range(1, budget + 1):
        if queue is not None:
            edge = queue[step - 1]
        else:
            cands = env.candidates()
            if mode == "frontier":
                cands = frontier_filter(inst, cands)
            s = SCORERS[strategy](inst, cands)
            edge = cands[int(env.rng.choice(np.flatnonzero(s == s.max())))]
        env.reveal(edge)
        if step in cps:
            snap(step)
    return {"rec": rec, "time": time.perf_counter() - t0, "ceiling": inst.ceiling,
            "p_est": inst.p_est, "p_true": inst.p_true}


def unit(args):
    seed, train_frac, budget, graph, feat_signal, combos, removal = args
    return seed, train_frac, {f"{s}/{m}": run_policy(seed, train_frac, s, m, budget, graph, feat_signal, removal)
                               for s, m in combos}


def collect(units, labels, seeds):
    res = {}
    for tf in labels:
        res[str(tf)] = {}
        by_seed = {sd: out for sd, t, out in units if t == tf}
        for key in by_seed[seeds[0]]:
            runs = [by_seed[sd][key] for sd in seeds]
            res[str(tf)][key] = {
                "acc": {str(c): [r["rec"]["acc"][c] for r in runs] for c in runs[0]["rec"]["acc"]},
                "reach": {str(c): [r["rec"]["reach"][c] for r in runs] for c in runs[0]["rec"]["reach"]},
                "conf": {str(c): [r["rec"]["conf"][c] for r in runs] for c in runs[0]["rec"]["conf"]},
                "edges": {str(c): [r["rec"]["edges"][c] for r in runs] for c in runs[0]["rec"]["edges"]},
                "time": [r["time"] for r in runs],
                "ceiling": [r["ceiling"] for r in runs],
                "p_est": [r["p_est"] for r in runs],
                "p_true": runs[0]["p_true"],
            }
    return res


def gain(r, b):
    return np.array(r["acc"][str(b)]) - np.array(r["acc"]["0"])


def summarize(res, budget):
    marks = sorted({50, 100, 200, budget} & set(range(budget + 1)))
    for tf, r in res.items():
        any_r = next(iter(r.values()))
        pe = np.array(any_r["p_est"])
        print(f"\n######## labelled = {float(tf):.0%}   targets=15   p_true={any_r['p_true']}  "
              f"p_est mean=({pe[:, 0].mean():.3f}, {pe[:, 1].mean():.3f})")
        a0 = np.mean(any_r["acc"]["0"])
        ceil = np.mean(any_r["ceiling"])
        print(f"start acc={a0:.3f}  ceiling (full graph)={ceil:.3f}  start reach={np.mean(any_r['reach']['0']):.2f}")
        hdr = f"{'strategy/mode':>24} " + " ".join(f"{'gain@' + str(b):>10}" for b in marks) \
              + f" {'gapclosed':>9} {'reach':>6} {'edges':>6} {'sec':>6}"
        print(hdr)
        for key, v in r.items():
            gs = [gain(v, b).mean() for b in marks]
            gc = gs[-1] / (ceil - a0) if ceil - a0 > 1e-9 else float("nan")
            print(f"{key:>24} " + " ".join(f"{g:>+10.3f}" for g in gs)
                  + f" {gc:>9.2f} {np.mean(v['reach'][str(budget)]):>6.2f}"
                  + f" {np.mean(v['edges'][str(budget)]):>6.1f} {np.mean(v['time']):>6.1f}")

        def cmp(label, ka, kb):
            if ka not in r or kb not in r:
                return
            cells = []
            for b in marks:
                m, s, p = paired(gain(r[ka], b), gain(r[kb], b))
                cells.append(f"b={b}: {m:+.3f}+-{s:.3f}{'*' if p < .05 else ' '}(p={p:.3f})")
            print(f"  {label:<46} " + "  ".join(cells))

        print("-- active learning: adaptive minus static (same strategy)")
        for s in ["common_nbrs", "adamic_adar", "uncertainty", "value", "value_all"]:
            cmp(s, f"{s}/adaptive", f"{s}/static")
        print("-- task-aware: value (adaptive) minus ...")
        for k in ["random/adaptive", "common_nbrs/adaptive", "adamic_adar/adaptive", "maxdeg/adaptive",
                  "pagerank/adaptive", "uncertainty/adaptive", "value_all/adaptive", "value_knownp/adaptive"]:
            cmp(k, "value/adaptive", k)
        print("-- static value minus static common_nbrs")
        cmp("value/static - common_nbrs/static", "value/static", "common_nbrs/static")
    print("\n* = paired t-test p < 0.05; gain = accuracy on targets minus accuracy before any query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--labels", type=float, nargs="+", default=[0.15, 0.40])
    ap.add_argument("--graph", default="locked")
    ap.add_argument("--feat", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="e1_active_2x2")
    ap.add_argument("--only", nargs="+", default=None, help="subset of strategy/mode, e.g. value/static")
    ap.add_argument("--removal", default="random", choices=["random", "target_hidden", "hub_observed"])
    ap.add_argument("--seed-start", type=int, default=0, help="first seed; use >=100 for unseen seeds")
    a = ap.parse_args()

    combos = COMBOS if a.only is None else [tuple(c.split("/")) for c in a.only]
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    jobs = [(sd, tf, a.budget, a.graph, a.feat, combos, a.removal) for tf in a.labels for sd in seeds]
    print(f"graph={a.graph} feat={a.feat} removal={a.removal} seeds={a.seeds} budget={a.budget} "
          f"labels={a.labels} workers={a.workers} combos={len(combos)}", flush=True)
    t0 = time.perf_counter()
    units = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, out in enumerate(ex.map(unit, jobs), 1):
            units.append(out)
            print(f"  done {i}/{len(jobs)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    res = collect(units, a.labels, seeds)
    with open(os.path.join(HERE, a.out + ".json"), "w") as f:
        json.dump(res, f)
    summarize(res, a.budget)


if __name__ == "__main__":
    main()
