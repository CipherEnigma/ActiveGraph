"""E3: how much exploration is enough, and what does it cost?

From a long-budget run (e.g. e3_long_budget.json). Per label density and strategy, on the mean
curves: gap closed = (acc - acc_start) / (ceiling - acc_start); queries needed to close 25/50/75%
of the gap (None = never within the budget); accuracy gain added by each successive block of
queries (saturation: blocks that add ~0 mean the curve has flattened); wall-clock seconds.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def mean_curve(v, metric):
    steps = sorted(int(s) for s in v[metric])
    return np.array(steps), np.array([np.mean(v[metric][str(s)]) for s in steps])


def first_reaching(steps, gc, frac):
    hit = np.flatnonzero(gc >= frac)
    return int(steps[hit[0]]) if len(hit) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="e3_long_budget.json")
    ap.add_argument("--block", type=int, default=100)
    a = ap.parse_args()
    res = json.load(open(os.path.join(HERE, a.json)))
    for tf, r in res.items():
        any_v = next(iter(r.values()))
        ceil = float(np.mean(any_v["ceiling"]))
        print(f"\n=== {float(tf):.0%} labelled   ceiling (full-graph accuracy) = {ceil:.3f} ===")
        steps, _ = mean_curve(any_v, "acc")
        edges = list(range(0, int(steps[-1]) + 1, a.block))
        hdr = f"{'strategy':>22} {'gap@end':>8} {'q->25%':>7} {'q->50%':>7} {'q->75%':>7}  " \
              + " ".join(f"{'+' + str(e) + '-' + str(e + a.block):>10}" for e in edges[:-1]) + f" {'sec/run':>8}"
        print(hdr)
        for key, v in r.items():
            steps, acc = mean_curve(v, "acc")
            gc = (acc - acc[0]) / (ceil - acc[0])
            blocks = []
            for e in edges[:-1]:
                i0, i1 = np.searchsorted(steps, e), np.searchsorted(steps, e + a.block)
                blocks.append(acc[i1] - acc[i0])
            q = [first_reaching(steps, gc, f) for f in (0.25, 0.5, 0.75)]
            qs = " ".join(f"{(str(x) if x is not None else 'never'):>7}" for x in q)
            print(f"{key:>22} {gc[-1]:>8.2f} {qs}  " + " ".join(f"{b:>+10.3f}" for b in blocks)
                  + f" {np.mean(v['time']):>8.1f}")
    print("\nq->X% = first checkpoint where the MEAN curve has closed X% of the gap to full-graph accuracy.")


if __name__ == "__main__":
    main()
