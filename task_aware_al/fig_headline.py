"""Headline figure: accuracy gain on the target nodes (top) and targets reached by a label
(bottom) vs queries spent, one column per label density. Value variants are coloured
(identity = colour + dash); the six heuristic baselines are one grey group, since they are
all near-flat -- the per-strategy numbers are in the results table.

Usage: python fig_headline.py --json e1_active_2x2.json --out fig_headline_e1.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, MUTED, GRID, AXIS, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
SERIES = [
    ("value/adaptive", "Value, targets (adaptive)", "#2a78d6", "-"),
    ("value/static", "Value, targets (static)", "#eb6834", (0, (5, 2.5))),
    ("value_all/adaptive", "Value, all nodes (adaptive)", "#1baf7a", "-"),
]
BASELINES = ["random/adaptive", "common_nbrs/adaptive", "adamic_adar/adaptive", "maxdeg/adaptive",
             "pagerank/adaptive", "uncertainty/adaptive"]
BASE_LABEL = "Heuristic baselines"


def curve(r, metric, key, as_gain):
    steps = sorted(int(s) for s in r[key][metric])
    arr = np.array([r[key][metric][str(s)] for s in steps])
    if as_gain:
        arr = arr - arr[0]
    return np.array(steps), arr.mean(axis=1), arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="e1_active_2x2.json")
    ap.add_argument("--out", default="fig_headline_e1.png")
    a = ap.parse_args()
    res = json.load(open(os.path.join(HERE, a.json)))
    labels = list(res)
    n_seeds = len(next(iter(next(iter(res.values())).values()))["time"])

    fig, axes = plt.subplots(2, len(labels), figsize=(9.4, 6.9), facecolor=SURFACE, sharex="col")
    axes = np.array(axes).reshape(2, len(labels))
    for c, tf in enumerate(labels):
        r = res[tf]
        for row, (metric, as_gain, ylabel) in enumerate([("acc", True, "Accuracy gain on target nodes"),
                                                         ("reach", False, "Targets reached by a label")]):
            ax = axes[row, c]
            ax.set_facecolor(SURFACE)
            base_end = []
            for k in BASELINES:
                if k in r:
                    x, m, _ = curve(r, metric, k, as_gain)
                    ax.plot(x, m, color=GREY, lw=1.0, alpha=0.85, zorder=2)
                    base_end.append(m[-1])
            for k, lab, col, ls in SERIES:
                if k not in r:
                    continue
                x, m, se = curve(r, metric, k, as_gain)
                ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, lw=0, zorder=2)
                ax.plot(x, m, color=col, lw=2.0, ls=ls, zorder=3)
                ax.annotate(lab.replace("Value, ", "").replace(" (", ", ").rstrip(")"),
                            (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                            fontsize=9, color=INK, annotation_clip=False)
            if base_end:
                ax.annotate("baselines", (x[-1], float(np.mean(base_end))), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=9, color=MUTED,
                            annotation_clip=False)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=9)
            ax.set_xlim(0, x[-1])
            if row == 0:
                ax.set_title(f"{float(tf):.0%} of nodes labelled", loc="left", fontsize=11.5, color=INK)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=10, color=INK)
            if row == 1:
                ax.set_xlabel("Queries spent", fontsize=10, color=INK)
    legend = [Line2D([], [], color=col, lw=2.0, ls=ls, label=lab) for k, lab, col, ls in SERIES
              if k in next(iter(res.values()))]
    legend.append(Line2D([], [], color=GREY, lw=1.0, label="Heuristic baselines (grey lines)"))
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Task-aware value queries close the gap; heuristic queries barely move it",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.925, f"{n_seeds} seeds, 15 target nodes, label propagation, locked graph. Bands = +-1 standard error.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 0.93, 0.92))
    out = os.path.join(HERE, a.out)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
