"""Summarise a cora_exp.py run: accuracy gain and gap closed (GCN and LP), real edges found, paired tests."""
import json
import sys

import numpy as np
from scipy import stats

path = sys.argv[1]
step = sys.argv[2] if len(sys.argv) > 2 else None
R = json.load(open(path))
runs, ceil = R["runs"], R["ceiling"]
seeds = sorted(ceil)
step = step or str(R["args"]["budget"])
print(f"{path}: {len(seeds)} seeds, step {step}, per_class {R['args']['per_class']}, targets {R['args']['targets']}")
print(f"pool recall {np.mean([next(iter(runs.values()))[s]['pool_recall'] for s in seeds]):.2f}, "
      f"precision {np.mean([next(iter(runs.values()))[s]['pool_precision'] for s in seeds]):.3f}\n")


def gain(strat, metric, s):
    r = runs[strat][s]["rec"]
    return r[step][metric] - r["0"][metric]


def gap(strat, metric, s):
    room = ceil[s][metric] - runs[strat][s]["rec"]["0"][metric]
    return gain(strat, metric, s) / room if room > 1e-9 else np.nan


print(f"{'strategy':13s} {'GCN gain':>9s} {'GCN gap':>8s} {'LP gain':>8s} {'LP gap':>7s} {'reach':>6s} {'found':>6s}")
for st in runs:
    g = [gain(st, "gcn", s) for s in seeds]
    l = [gain(st, "lp", s) for s in seeds]
    print(f"{st:13s} {np.mean(g):+9.3f} {np.nanmean([gap(st, 'gcn', s) for s in seeds]):8.2f} "
          f"{np.mean(l):+8.3f} {np.nanmean([gap(st, 'lp', s) for s in seeds]):7.2f} "
          f"{np.mean([runs[st][s]['rec'][step]['reach'] for s in seeds]):6.2f} "
          f"{np.mean([runs[st][s]['rec'][step]['found'] for s in seeds]):6.1f}")

print("\nstart / ceiling (mean): GCN %.3f / %.3f   LP %.3f / %.3f" % (
    np.mean([next(iter(runs.values()))[s]["rec"]["0"]["gcn"] for s in seeds]),
    np.mean([ceil[s]["gcn"] for s in seeds]),
    np.mean([next(iter(runs.values()))[s]["rec"]["0"]["lp"] for s in seeds]),
    np.mean([ceil[s]["lp"] for s in seeds])))

if "value" in runs:
    print("\npaired: value minus X  (mean diff, p)")
    for st in runs:
        if st == "value":
            continue
        out = []
        for m in ("gcn", "lp"):
            d = np.array([gain("value", m, s) - gain(st, m, s) for s in seeds])
            p = stats.ttest_1samp(d, 0).pvalue if len(d) > 2 and d.std() > 0 else float("nan")
            out.append(f"{m.upper()} {d.mean():+.3f} (p={p:.3f})")
        print(f"  vs {st:13s} " + "   ".join(out))

print("\nGCN accuracy over budget (mean):")
steps = sorted({k for st in runs for s in seeds for k, v in runs[st][s]["rec"].items() if "gcn" in v}, key=int)
print(f"{'strategy':13s} " + " ".join(f"{k:>6s}" for k in steps))
for st in runs:
    print(f"{st:13s} " + " ".join(f"{np.mean([runs[st][s]['rec'][k]['gcn'] for s in seeds]):6.3f}" for k in steps))
