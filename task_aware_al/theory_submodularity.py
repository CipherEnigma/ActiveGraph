"""Phase 6: Empirical submodularity analysis.

Tests whether the edge-query value function exhibits diminishing returns:
  f(S) = LP accuracy on targets after adding edge set S to G_obs

Submodularity means: for S subset of S', adding one more edge to S gives at
least as much gain as adding it to S'. Equivalently, marginal gains are
non-increasing as more edges are added.

We check:
  1. Plot marginal gain of the k-th query vs k (averaged over seeds)
  2. Compute the submodularity ratio gamma = min marginal gain ratio
  3. Test monotonicity: is f(S) always non-decreasing? (not guaranteed)

Usage:
  python theory_submodularity.py results/4a_dev_cora.json
  python theory_submodularity.py results/4b_main_cora.json --strategy value
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load(path):
    with open(path) as f:
        return json.load(f)


def marginal_gains(run_data, metric="lp"):
    """Extract marginal gains between consecutive checkpoints."""
    rec = run_data["rec"]
    steps = sorted(rec.keys(), key=int)
    gains = []
    for i in range(1, len(steps)):
        prev = rec[steps[i-1]].get(metric, None)
        curr = rec[steps[i]].get(metric, None)
        if prev is not None and curr is not None:
            delta_queries = int(steps[i]) - int(steps[i-1])
            marginal = (curr - prev) / max(delta_queries, 1)
            gains.append({
                "step": int(steps[i]),
                "from_step": int(steps[i-1]),
                "delta_queries": delta_queries,
                "total_gain": curr - prev,
                "marginal_per_query": marginal,
                "accuracy": curr,
            })
    return gains


def analyze_submodularity(R, strategy="value", metric="lp"):
    """Empirical submodularity analysis."""
    runs = R["runs"]
    seeds = sorted(R["ceiling"])

    if strategy not in runs:
        print(f"Strategy '{strategy}' not in results. Available: {list(runs.keys())}")
        return

    print(f"\n=== Submodularity Analysis: {strategy}, metric={metric} ===\n")

    all_gains = {}
    for s in seeds:
        mg = marginal_gains(runs[strategy][s], metric)
        for g in mg:
            step = g["step"]
            all_gains.setdefault(step, []).append(g["marginal_per_query"])

    steps = sorted(all_gains.keys())
    means = [np.mean(all_gains[s]) for s in steps]
    stds = [np.std(all_gains[s]) for s in steps]

    print(f"{'Step':>6s} {'Mean marginal':>14s} {'Std':>8s} {'n':>3s}")
    for s, m, sd in zip(steps, means, stds):
        print(f"{s:6d} {m:14.6f} {sd:8.6f} {len(all_gains[s]):3d}")

    # Test: are marginal gains non-increasing?
    violations = 0
    total = 0
    for i in range(1, len(means)):
        total += 1
        if means[i] > means[i-1] + 1e-9:
            violations += 1

    print(f"\nMonotonicity of marginal gains:")
    print(f"  Violations: {violations}/{total} steps have INCREASING marginals")

    if total > 0:
        pct = violations / total * 100
        if pct < 10:
            print(f"  Assessment: APPROXIMATELY SUBMODULAR ({pct:.0f}% violations)")
        elif pct < 30:
            print(f"  Assessment: WEAKLY SUBMODULAR ({pct:.0f}% violations)")
        else:
            print(f"  Assessment: NOT SUBMODULAR ({pct:.0f}% violations)")

    # Submodularity ratio
    if len(means) >= 3:
        ratios = []
        for i in range(1, len(means)):
            if means[i-1] > 1e-9:
                ratios.append(means[i] / means[i-1])
        if ratios:
            gamma = min(ratios)
            print(f"\n  Submodularity ratio gamma = {gamma:.4f}")
            print(f"  (gamma=1 is exactly submodular, gamma>0 gives greedy guarantee)")
            if gamma > 0:
                bound = gamma * (1 - 1/np.e)
                print(f"  Greedy approximation bound: gamma*(1-1/e) = {bound:.4f}")

    # Spearman correlation of step vs marginal (should be negative if diminishing returns)
    if len(steps) >= 4:
        rho, p = stats.spearmanr(steps, means)
        print(f"\n  Spearman(step, marginal): rho={rho:.4f}, p={p:.4f}")
        if rho < -0.5 and p < 0.05:
            print("  Strong evidence for diminishing returns")
        elif rho < 0:
            print("  Weak evidence for diminishing returns")
        else:
            print("  No evidence for diminishing returns")

    return steps, means, stds


def fig_marginal_gains(R, strategy="value", metric="lp", outdir=None):
    """Plot marginal gains vs queries."""
    runs = R["runs"]
    seeds = sorted(R["ceiling"])
    if outdir is None:
        outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)

    strategies_to_plot = [s for s in ["value", "value_diag", "linkprob", "random"] if s in runs]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"value": "#d62728", "value_diag": "#e377c2", "linkprob": "#1f77b4", "random": "#7f7f7f"}
    labels = {"value": "Value (ours)", "value_diag": "Value+Diag", "linkprob": "Link prob", "random": "Random"}

    for st in strategies_to_plot:
        all_gains = {}
        for s in seeds:
            mg = marginal_gains(runs[st][s], metric)
            for g in mg:
                all_gains.setdefault(g["step"], []).append(g["total_gain"])

        steps = sorted(all_gains.keys())
        means = [np.mean(all_gains[s]) for s in steps]
        stds = [np.std(all_gains[s]) for s in steps]

        ax.plot(steps, means, color=colors.get(st, "grey"), lw=2,
                label=labels.get(st, st))
        ax.fill_between(steps, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.15, color=colors.get(st, "grey"))

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Queries", fontsize=12)
    ax.set_ylabel("Marginal LP gain (per interval)", fontsize=12)
    ax.set_title("Diminishing Returns in Edge Querying", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(outdir, "marginal_gains.pdf")
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Result JSON file")
    ap.add_argument("--strategy", default="value")
    ap.add_argument("--metric", default="lp")
    ap.add_argument("--plot", action="store_true", help="Generate figure")
    args = ap.parse_args()

    R = load(args.file)
    analyze_submodularity(R, args.strategy, args.metric)

    if args.plot:
        fig_marginal_gains(R, args.strategy, args.metric)


if __name__ == "__main__":
    main()
