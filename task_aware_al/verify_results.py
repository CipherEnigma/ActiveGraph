"""Verification of the E1-E7 results.

(a) Leakage test: replace the labels of all NON-train nodes with random labels. If any scorer
    used them, its chosen queries would change. They must be identical.
(b) Pairing integrity: files that are compared pairwise must hold identical instances
    (same start accuracy and ceiling per seed), and E7-random must reproduce E1 exactly.
(c) Statistics: for the headline comparisons, paired t-test AND Wilcoxon signed-rank, a
    bootstrap 95% CI of the mean difference, and Holm-adjusted p-values across the family.
"""
import json
import os

import numpy as np
from scipy import stats

from common import Instance, N
from exp_active_2x2 import SCORERS

HERE = os.path.dirname(os.path.abspath(__file__))


def chosen_edges(inst, strategy, steps):
    env, picked = inst.env, []
    for _ in range(steps):
        cands = env.candidates()
        s = SCORERS[strategy](inst, cands)
        edge = cands[int(env.rng.choice(np.flatnonzero(s == s.max())))]
        env.reveal(edge)
        picked.append(edge)
    return picked


def leakage_test(steps=25, seeds=(0, 1, 2), frac=0.4):
    print("(a) LEAKAGE TEST: non-train labels randomised; chosen queries must not change")
    ok_all = True
    for strat in ["random", "common_nbrs", "adamic_adar", "maxdeg", "pagerank", "uncertainty",
                  "value", "value_all"]:
        same, acc_changed = 0, 0
        for sd in seeds:
            a = Instance(sd, frac)
            b = Instance(sd, frac)
            nontrain = np.setdiff1d(np.arange(N), b.env.train_idx)
            y2 = b.y.copy()
            y2[nontrain] = np.random.default_rng(999 + sd).integers(0, 3, size=len(nontrain))
            b.y = y2
            acc_changed += int(abs(a.target_accuracy(a.A_true.astype(float)) -
                                   b.target_accuracy(b.A_true.astype(float))) > 1e-9)
            same += int(chosen_edges(a, strat, steps) == chosen_edges(b, strat, steps))
        ok = same == len(seeds)
        ok_all &= ok
        print(f"  {strat:>12}: identical query sequences on {same}/{len(seeds)} seeds "
              f"{'OK' if ok else '*** LEAK ***'}   (test sensitive: evaluation changed on {acc_changed}/{len(seeds)})")
    print("  ->", "no scorer uses non-train labels" if ok_all else "LEAKAGE FOUND")


def load(name):
    return json.load(open(os.path.join(HERE, name)))


def pairing_checks():
    print("\n(b) PAIRING / REPRODUCIBILITY")
    e1 = load("e1_active_2x2.json")
    for other in ["e5_frontier.json", "e3_long_budget.json", "e7_random.json", "e7_target_hidden.json",
                  "e7_hub_observed.json"]:
        o = load(other)
        for tf in e1:
            k1, k2 = next(iter(e1[tf])), next(iter(o[tf]))
            same_ceil = np.allclose(e1[tf][k1]["ceiling"], o[tf][k2]["ceiling"])
            same_start = np.allclose(e1[tf][k1]["acc"]["0"], o[tf][k2]["acc"]["0"])
            tag = "same instances" if (same_ceil and same_start) else "DIFFERENT start/ceiling"
            if "target_hidden" in other or "hub_observed" in other:
                tag = ("same graph+labels+targets (ceiling equal)" if same_ceil else "ceilings DIFFER") + \
                      ("" if same_start else "; start differs by design (other edges observed)")
            print(f"  {other:>24} @ {float(tf):.0%}: {tag}")
    o = load("e7_random.json")
    worst = 0.0
    for tf in e1:
        for key in o[tf]:
            for metric in ("acc", "reach", "edges"):
                for st in o[tf][key][metric]:
                    worst = max(worst, np.abs(np.array(o[tf][key][metric][st]) - np.array(e1[tf][key][metric][st])).max())
    print(f"  E7-random vs E1, every strategy/checkpoint/metric: max |difference| = {worst:.2e}")


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def stats_checks(budget=200, n_boot=10000):
    print(f"\n(c) STATISTICS on headline comparisons (gain @ {budget} queries, 30 paired seeds)")
    e1 = load("e1_active_2x2.json")
    comps = [("value/adaptive", "random/adaptive"), ("value/adaptive", "common_nbrs/adaptive"),
             ("value/adaptive", "maxdeg/adaptive"), ("value/adaptive", "value_all/adaptive"),
             ("value/adaptive", "value/static"), ("value/static", "common_nbrs/static")]
    rng = np.random.default_rng(0)
    rows = []
    for tf in e1:
        for ka, kb in comps:
            a = np.array(e1[tf][ka]["acc"][str(budget)]) - np.array(e1[tf][ka]["acc"]["0"])
            b = np.array(e1[tf][kb]["acc"][str(budget)]) - np.array(e1[tf][kb]["acc"]["0"])
            d = a - b
            pt = stats.ttest_rel(a, b).pvalue
            pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append((tf, ka, kb, d.mean(), lo, hi, pt, pw, float((d > 0).mean()), float((d < 0).mean())))
    ph = holm(np.array([max(r[6], r[7]) for r in rows]))
    print(f"{'labels':>6} {'comparison':>42} {'mean diff':>10} {'95% bootstrap CI':>20} {'t p':>8} {'wilcoxon p':>11}"
          f" {'Holm p':>8} {'seeds +/-':>10}")
    for r, p in zip(rows, ph):
        print(f"{float(r[0]):>6.0%} {r[1] + ' - ' + r[2]:>42} {r[3]:>+10.3f} {f'[{r[4]:+.3f}, {r[5]:+.3f}]':>20} "
              f"{r[6]:>8.4f} {r[7]:>11.4f} {p:>8.4f} {f'{r[8]:.0%}/{r[9]:.0%}':>10}")
    print("  Holm p uses the larger of the t and Wilcoxon p-values, adjusted over this family of "
          f"{len(rows)} tests.")


if __name__ == "__main__":
    leakage_test()
    pairing_checks()
    stats_checks()
