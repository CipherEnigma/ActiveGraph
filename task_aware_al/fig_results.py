"""Phase 5: Publication-quality figures for the paper.

Usage:
  python fig_results.py results/4b_main_cora.json --fig accuracy_curves
  python fig_results.py results/4b_main_cora.json --fig failure_evolution
  python fig_results.py results/4c_obs*.json --fig obs_sweep
  python fig_results.py results/4d_budget*.json --fig budget_scaling
  python fig_results.py results/4b_main_cora.json --fig all
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")

STYLE = {
    "value":          {"color": "#d62728", "ls": "-",  "lw": 2.5, "label": "Value (ours)"},
    "value_diag":     {"color": "#e377c2", "ls": "-",  "lw": 2.0, "label": "Value+Diag (ours)"},
    "value_all":      {"color": "#ff7f0e", "ls": "--", "lw": 1.5, "label": "Value (all nodes)"},
    "value_gcn":      {"color": "#8c564b", "ls": "-.", "lw": 1.5, "label": "Value (GCN scorer)"},
    "linkprob":       {"color": "#1f77b4", "ls": "-",  "lw": 1.5, "label": "Link probability"},
    "vgae_topk":      {"color": "#17becf", "ls": "--", "lw": 1.5, "label": "VGAE top-k"},
    "random":         {"color": "#7f7f7f", "ls": ":",  "lw": 1.5, "label": "Random"},
    "common_nbrs":    {"color": "#2ca02c", "ls": "--", "lw": 1.2, "label": "Common neighbours"},
    "adamic_adar":    {"color": "#bcbd22", "ls": "-.", "lw": 1.2, "label": "Adamic-Adar"},
    "maxdeg":         {"color": "#9467bd", "ls": ":",  "lw": 1.2, "label": "Max degree"},
    "uncertainty_edge":{"color": "#aec7e8","ls": "-.", "lw": 1.5, "label": "Uncertainty"},
    "qbc":            {"color": "#98df8a", "ls": "--", "lw": 1.5, "label": "QBC"},
    "infogain":       {"color": "#ff9896", "ls": "-.", "lw": 1.5, "label": "Info gain"},
    "oracle":         {"color": "#000000", "ls": "-",  "lw": 1.0, "label": "Oracle (upper bound)"},
}


def load(path):
    with open(path) as f:
        return json.load(f)


def get_lp_curve(runs, strategy, seeds):
    """Get mean and std of LP accuracy over steps."""
    all_steps = set()
    for s in seeds:
        all_steps |= {k for k in runs[strategy][s]["rec"] if "lp" in runs[strategy][s]["rec"][k]}
    steps = sorted(all_steps, key=int)
    means, stds, xs = [], [], []
    for step in steps:
        vals = [runs[strategy][s]["rec"][step]["lp"]
                for s in seeds if step in runs[strategy][s]["rec"]]
        if vals:
            xs.append(int(step))
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


def fig_accuracy_curves(R, outdir):
    """Main result: LP accuracy vs queries for all strategies."""
    runs, ceil = R["runs"], R["ceiling"]
    seeds = sorted(ceil)
    strategies = list(runs.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    for st in strategies:
        if st not in STYLE:
            continue
        xs, means, stds = get_lp_curve(runs, st, seeds)
        s = STYLE[st]
        ax.plot(xs, means, color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])
        ax.fill_between(xs, means - stds, means + stds, alpha=0.1, color=s["color"])

    ceil_mean = np.mean([ceil[s]["lp"] for s in seeds])
    ax.axhline(ceil_mean, color="grey", ls="--", lw=0.8, alpha=0.5, label=f"Ceiling ({ceil_mean:.3f})")

    ax.set_xlabel("Queries", fontsize=12)
    ax.set_ylabel("LP accuracy on targets", fontsize=12)
    ax.set_title(f"Target Classification vs Edge Queries ({R['args'].get('dataset','cora').title()})",
                 fontsize=13)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f"accuracy_curves_{R['args'].get('dataset','cora')}.pdf")
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def fig_failure_evolution(R, outdir):
    """Stacked area chart of failure modes over queries."""
    runs, ceil = R["runs"], R["ceiling"]
    seeds = sorted(ceil)
    strategies = [s for s in ["value", "value_diag", "linkprob", "random"] if s in runs]

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4), sharey=True)
    if len(strategies) == 1:
        axes = [axes]

    colors = {"structural": "#d62728", "informational": "#ff7f0e", "intrinsic": "#1f77b4"}

    for ax, st in zip(axes, strategies):
        all_steps = sorted({k for s in seeds for k in runs[st][s]["rec"]
                           if "failures" in runs[st][s]["rec"][k]}, key=int)
        xs = [int(s) for s in all_steps]
        struct_vals, info_vals, intr_vals = [], [], []
        for step in all_steps:
            fc = [runs[st][s]["rec"][step]["failures"] for s in seeds if step in runs[st][s]["rec"]]
            struct_vals.append(np.mean([f.get("structural", 0) for f in fc]))
            info_vals.append(np.mean([f.get("informational", 0) for f in fc]))
            intr_vals.append(np.mean([f.get("intrinsic", 0) for f in fc]))

        ax.stackplot(xs,
                     struct_vals, info_vals, intr_vals,
                     labels=["Structural", "Informational", "Intrinsic"],
                     colors=[colors["structural"], colors["informational"], colors["intrinsic"]],
                     alpha=0.7)
        label = STYLE.get(st, {}).get("label", st)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Queries", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Target nodes", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Failure Mode Evolution", fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, f"failure_evolution_{R['args'].get('dataset','cora')}.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


def fig_reach_vs_found(R, outdir):
    """Scatter: real edges found vs target reach at final step."""
    runs, ceil = R["runs"], R["ceiling"]
    seeds = sorted(ceil)
    step = str(R["args"]["budget"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for st in runs:
        if st not in STYLE:
            continue
        found = [runs[st][s]["rec"][step].get("found", 0) for s in seeds]
        reach = [runs[st][s]["rec"][step].get("reach", 0) for s in seeds]
        s = STYLE[st]
        ax.scatter(np.mean(found), np.mean(reach), color=s["color"], s=80,
                   label=s["label"], zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Real edges found", fontsize=12)
    ax.set_ylabel("Target reach (fraction reachable)", fontsize=12)
    ax.set_title("Finding Edges vs Reaching Targets", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f"reach_vs_found_{R['args'].get('dataset','cora')}.pdf")
    fig.savefig(path, dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


ALL_FIGS = {
    "accuracy_curves": fig_accuracy_curves,
    "failure_evolution": fig_failure_evolution,
    "reach_vs_found": fig_reach_vs_found,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fig", default="all", help="Figure to generate (or 'all')")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    for path in args.files:
        R = load(path)
        if args.fig == "all":
            for name, fn in ALL_FIGS.items():
                try:
                    fn(R, FIG_DIR)
                except Exception as e:
                    print(f"  SKIP {name}: {e}")
        elif args.fig in ALL_FIGS:
            ALL_FIGS[args.fig](R, FIG_DIR)
        else:
            print(f"Unknown figure: {args.fig}")
            print(f"Available: {', '.join(ALL_FIGS)}, all")


if __name__ == "__main__":
    main()
