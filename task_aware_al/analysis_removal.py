"""E7: random vs targeted edge removal. Reads e7_<scheme>.json for the three schemes.

Instances (graph, labels, targets) are identical across schemes for a given seed; only WHICH
edges are initially observed differs, so scheme comparisons are paired by seed.
"""
import json
import os

import numpy as np

from common import paired

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = ["random", "target_hidden", "hub_observed"]
STRATS = ["random/adaptive", "common_nbrs/adaptive", "maxdeg/adaptive", "pagerank/adaptive",
          "value/static", "value/adaptive"]
MARKS = [50, 100, 200]


def gain(r, key, b):
    return np.array(r[key]["acc"][str(b)]) - np.array(r[key]["acc"]["0"])


def cell(a, b):
    m, s, p = paired(a, b)
    return f"{m:+.3f}+-{s:.3f}{'*' if p < .05 else ' '}(p={p:.3f})"


def main():
    data = {s: json.load(open(os.path.join(HERE, f"e7_{s}.json"))) for s in SCHEMES}
    for tf in data["random"]:
        print(f"\n######## {float(tf):.0%} labelled")
        for s in SCHEMES:
            r = data[s][tf]
            k0 = next(iter(r))
            ceil = float(np.mean(r[k0]["ceiling"]))
            a0 = float(np.mean(r[k0]["acc"]["0"]))
            print(f"\n== {s}: start acc={a0:.3f}  start reach={np.mean(r[k0]['reach']['0']):.2f}  ceiling={ceil:.3f}")
            print(f"{'strategy':>22} " + " ".join(f"{'gain@' + str(b):>9}" for b in MARKS)
                  + f" {'gapclosed':>9} {'reach@200':>9} {'edges@200':>9}")
            for k in STRATS:
                gs = [gain(r, k, b).mean() for b in MARKS]
                print(f"{k:>22} " + " ".join(f"{x:>+9.3f}" for x in gs)
                      + f" {gs[-1] / (ceil - a0):>9.2f} {np.mean(r[k]['reach']['200']):>9.2f}"
                      + f" {np.mean(r[k]['edges']['200']):>9.1f}")
            print("  value/adaptive minus:")
            for k in STRATS[:-1]:
                print(f"    {k:<22} " + "  ".join(f"b={b}: {cell(gain(r, 'value/adaptive', b), gain(r, k, b))}"
                                                  for b in MARKS))
        print("\n== effect of the removal scheme on each strategy (scheme minus random removal), gain@200")
        for k in STRATS:
            print(f"  {k:<22} " + "  ".join(
                f"{s}: {cell(gain(data[s][tf], k, 200), gain(data['random'][tf], k, 200))}"
                for s in SCHEMES[1:]))
    print("\n* = paired t-test p < 0.05")


if __name__ == "__main__":
    main()
