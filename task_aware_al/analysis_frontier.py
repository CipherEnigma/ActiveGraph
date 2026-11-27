"""E5: global vs frontier-only exploration.

Frontier results come from e5_frontier.json; the global (adaptive) rows come from
e1_active_2x2.json. Both use seeds 0..29 and budget 200 on identical instances, so the
comparisons are paired across the two files.
"""
import json
import os

import numpy as np

from common import paired

HERE = os.path.dirname(os.path.abspath(__file__))
STRATS = ["random", "common_nbrs", "maxdeg", "pagerank", "value"]
MARKS = [50, 100, 200]


def gain(r, key, b):
    return np.array(r[key]["acc"][str(b)]) - np.array(r[key]["acc"]["0"])


def cell(a, b):
    m, s, p = paired(a, b)
    return f"{m:+.3f}+-{s:.3f}{'*' if p < .05 else ' '}(p={p:.3f})"


def main():
    e1 = json.load(open(os.path.join(HERE, "e1_active_2x2.json")))
    e5 = json.load(open(os.path.join(HERE, "e5_frontier.json")))
    for tf in e5:
        g, f = e1[tf], e5[tf]
        ceil = float(np.mean(g["random/adaptive"]["ceiling"]))
        a0 = float(np.mean(g["random/adaptive"]["acc"]["0"]))
        print(f"\n=== {float(tf):.0%} labelled ===")
        print(f"{'strategy':>14} {'mode':>9} " + " ".join(f"{'gain@' + str(b):>9}" for b in MARKS)
              + f" {'gapclosed':>9} {'reach':>6} {'edges':>6}")
        for s in STRATS:
            for mode, src, key in (("global", g, f"{s}/adaptive"), ("frontier", f, f"{s}/frontier")):
                gs = [gain(src, key, b).mean() for b in MARKS]
                print(f"{s:>14} {mode:>9} " + " ".join(f"{x:>+9.3f}" for x in gs)
                      + f" {gs[-1] / (ceil - a0):>9.2f} {np.mean(src[key]['reach']['200']):>6.2f}"
                      + f" {np.mean(src[key]['edges']['200']):>6.1f}")
        print("-- frontier minus global (same strategy)")
        for s in STRATS:
            print(f"  {s:<12} " + "  ".join(
                f"b={b}: {cell(gain(f, f'{s}/frontier', b), gain(g, f'{s}/adaptive', b))}" for b in MARKS))
        print("-- locality control: value minus other strategies, both frontier-restricted")
        for s in STRATS[:-1]:
            print(f"  value - {s:<10} " + "  ".join(
                f"b={b}: {cell(gain(f, 'value/frontier', b), gain(f, f'{s}/frontier', b))}" for b in MARKS))
        print("-- frontier-restricted random minus global random (how much is just locality?)")
        print("  " + "  ".join(f"b={b}: {cell(gain(f, 'random/frontier', b), gain(g, 'random/adaptive', b))}"
                               for b in MARKS))
    print("\n* = paired t-test p < 0.05")


if __name__ == "__main__":
    main()
