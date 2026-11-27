"""Extended analysis for the paper: AUC metric, Holm correction, Wilcoxon,
bootstrap CI, effect sizes, and failure mode evolution.

Usage:
  python analysis_paper.py results/4b_main_cora.json
  python analysis_paper.py results/4b_main_cora.json --step 500
  python analysis_paper.py results/4c_obs*.json --sweep obs  # compare across files
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats


def load_result(path):
    with open(path) as f:
        return json.load(f)


def auc_gain(run_data, metric, steps):
    """Area under the (accuracy gain vs queries) curve, trapezoidal rule."""
    vals = []
    xs = []
    base = run_data["rec"]["0"][metric]
    for s in sorted(steps, key=int):
        if s in run_data["rec"] and metric in run_data["rec"][s]:
            xs.append(int(s))
            vals.append(run_data["rec"][s][metric] - base)
    if len(xs) < 2:
        return 0.0
    return float(np.trapz(vals, xs))


def bootstrap_ci(data, n_boot=5000, alpha=0.05, rng=None):
    """95% bootstrap CI of the mean."""
    if rng is None:
        rng = np.random.default_rng(42)
    data = np.array(data)
    means = np.array([rng.choice(data, len(data), replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def holm_correction(pvals):
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(pvals[idx] * (n - rank), 1.0)
    for i in range(1, n):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i-1]])
    return adjusted


def analyze_single(R, step=None, reference="value"):
    """Full analysis of one result file."""
    runs, ceil = R["runs"], R["ceiling"]
    seeds = sorted(ceil)
    args = R["args"]
    step = step or str(args["budget"])
    strategies = list(runs.keys())

    print(f"Dataset: {args.get('dataset', 'cora')}, seeds: {len(seeds)}, "
          f"budget: {args['budget']}, obs: {args.get('observe', '?')}, "
          f"link_model: {args.get('link_model', '?')}")

    # Common LP steps
    lp_steps = sorted({k for st in runs for s in seeds
                       for k in runs[st][s]["rec"]
                       if "lp" in runs[st][s]["rec"][k]}, key=int)

    print(f"\n{'strategy':18s} {'LP gain':>8s} {'LP AUC':>8s} {'reach':>6s} {'found':>6s} "
          f"{'struct':>7s} {'info':>5s} {'intrinsic':>9s}")
    print("-" * 90)

    strategy_aucs = {}
    strategy_gains = {}

    for st in strategies:
        gains = []
        aucs = []
        reaches = []
        founds = []
        structs = []
        infos = []
        intrinsics = []

        for s in seeds:
            rd = runs[st][s]
            g = rd["rec"][step]["lp"] - rd["rec"]["0"]["lp"]
            gains.append(g)
            aucs.append(auc_gain(rd, "lp", lp_steps))
            reaches.append(rd["rec"][step].get("reach", 0))
            founds.append(rd["rec"][step].get("found", 0))
            f = rd["rec"][step].get("failures", {})
            structs.append(f.get("structural", 0))
            infos.append(f.get("informational", 0))
            intrinsics.append(f.get("intrinsic", 0))

        strategy_gains[st] = gains
        strategy_aucs[st] = aucs

        print(f"{st:18s} {np.mean(gains):+8.4f} {np.mean(aucs):8.1f} "
              f"{np.mean(reaches):6.3f} {np.mean(founds):6.1f} "
              f"{np.mean(structs):7.1f} {np.mean(infos):5.1f} {np.mean(intrinsics):9.1f}")

    # Starting accuracy and ceiling
    first_st = strategies[0]
    lp_start = np.mean([runs[first_st][s]["rec"]["0"]["lp"] for s in seeds])
    lp_ceil = np.mean([ceil[s]["lp"] for s in seeds])
    print(f"\nLP start: {lp_start:.4f}, ceiling: {lp_ceil:.4f}")

    # Statistical tests: reference strategy vs all others
    if reference in runs and len(seeds) >= 5:
        print(f"\n--- Statistical tests: {reference} vs others ---")
        print(f"{'vs':18s} {'mean diff':>10s} {'t-test p':>9s} {'wilcox p':>9s} "
              f"{'holm-t':>7s} {'holm-w':>7s} {'CI low':>7s} {'CI high':>8s} {'d':>6s}")
        print("-" * 100)

        comparisons = [st for st in strategies if st != reference]
        t_pvals = []
        w_pvals = []
        diffs_all = []

        for st in comparisons:
            d = np.array(strategy_aucs[reference]) - np.array(strategy_aucs[st])
            diffs_all.append(d)
            t_stat, t_p = stats.ttest_1samp(d, 0) if d.std() > 1e-12 else (0, 1.0)
            try:
                w_stat, w_p = stats.wilcoxon(d, alternative="two-sided")
            except ValueError:
                w_p = 1.0
            t_pvals.append(t_p)
            w_pvals.append(w_p)

        t_holm = holm_correction(np.array(t_pvals))
        w_holm = holm_correction(np.array(w_pvals))

        for i, st in enumerate(comparisons):
            d = diffs_all[i]
            ci_lo, ci_hi = bootstrap_ci(d)
            cohens_d = d.mean() / d.std() if d.std() > 1e-12 else 0.0
            sig = "*" if t_holm[i] < 0.05 and w_holm[i] < 0.05 else ""
            print(f"{st:18s} {d.mean():+10.2f} {t_pvals[i]:9.4f} {w_pvals[i]:9.4f} "
                  f"{t_holm[i]:7.4f} {w_holm[i]:7.4f} {ci_lo:7.2f} {ci_hi:8.2f} {cohens_d:6.2f} {sig}")

    # LP accuracy over budget
    print(f"\nLP accuracy curve (mean over seeds):")
    header = f"{'strategy':18s} " + " ".join(f"{s:>6s}" for s in lp_steps)
    print(header)
    for st in strategies:
        vals = []
        for s_step in lp_steps:
            v = np.mean([runs[st][s]["rec"][s_step]["lp"]
                        for s in seeds if s_step in runs[st][s]["rec"]])
            vals.append(f"{v:6.3f}")
        print(f"{st:18s} " + " ".join(vals))

    # Failure mode evolution
    if any("failures" in runs[strategies[0]][seeds[0]]["rec"].get(s, {}) for s in lp_steps):
        print(f"\nFailure mode evolution (structural count, mean over seeds):")
        print(f"{'strategy':18s} " + " ".join(f"{s:>6s}" for s in lp_steps))
        for st in strategies:
            vals = []
            for s_step in lp_steps:
                fc = [runs[st][s]["rec"][s_step].get("failures", {}).get("structural", 0)
                      for s in seeds if s_step in runs[st][s]["rec"]]
                vals.append(f"{np.mean(fc):6.1f}" if fc else "   N/A")
            print(f"{st:18s} " + " ".join(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Result JSON file(s)")
    ap.add_argument("--step", default=None, help="Step to analyze (default: budget)")
    ap.add_argument("--reference", default="value", help="Reference strategy for tests")
    args = ap.parse_args()

    for path in args.files:
        print(f"\n{'#' * 70}")
        print(f"# {os.path.basename(path)}")
        print(f"{'#' * 70}")
        R = load_result(path)
        analyze_single(R, step=args.step, reference=args.reference)


if __name__ == "__main__":
    main()
