"""Builds the self-contained figures folder for the LaTeX report: report/figures/.

New figures: pipeline diagram, links-found-vs-accuracy scatter, attributes sweep.
Copied/regenerated: cluster formation, headline curves (unseen seeds), 400-query curves.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report", "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, MUTED, GRID, AXIS, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#898781"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, NEUTRAL_FILL = "#cde2fb", "#f0efec"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 3.0), facecolor=SURFACE)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    w, h, gap, x0, y0 = 1.9, 1.1, 0.375, 0.1, 1.7
    steps = [("Partly observed\nnetwork", False, ""),
             ("Choose a pair\nto ask about", True, "query strategy"),
             ("Ask the oracle:\nare they linked?", False, "costs 1 query"),
             ("Add the link\nif it is real", False, ""),
             ("Classify the\ntarget nodes", False, "label propagation")]
    centers = []
    for i, (label, accent, sub) in enumerate(steps):
        x = x0 + i * (w + gap)
        centers.append(x + w / 2)
        ax.add_patch(plt.Rectangle((x, y0), w, h, fc=BLUE_LIGHT if accent else NEUTRAL_FILL,
                                   ec=BLUE if accent else INK, lw=1.6 if accent else 1.1))
        ax.text(x + w / 2, y0 + h / 2, label, ha="center", va="center", fontsize=11, color=INK)
        if sub:
            ax.text(x + w / 2, y0 - 0.2, sub, ha="center", va="top", fontsize=9, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, y0 + h / 2), xytext=(x + w + 0.03, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ly = 0.65
    ax.plot([centers[4], centers[4]], [y0 - 0.55, ly], color=INK, lw=1.3)
    ax.plot([centers[4], centers[1]], [ly, ly], color=INK, lw=1.3)
    ax.annotate("", xy=(centers[1], y0 - 0.55), xytext=(centers[1], ly),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.text((centers[1] + centers[4]) / 2, ly - 0.12, "repeat until the budget of B queries is used",
            ha="center", va="top", fontsize=10, color=INK)
    ax.text(centers[0], 3.05, "Input: observed links, a few labels,\ntarget nodes, budget B", ha="center",
            va="center", fontsize=9, color=MUTED)
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_edges_vs_accuracy():
    res = json.load(open(os.path.join(HERE, "r1_fresh_seeds.json")))
    keys = [("random/adaptive", "random", GREY),
            ("common_nbrs/adaptive", "common neighbours", GREY),
            ("maxdeg/adaptive", "max-degree", GREY),
            ("value_all/adaptive", "value, all nodes", AQUA),
            ("value/static", "value, targets (static)", ORANGE),
            ("value/adaptive", "value, targets (adaptive)", BLUE)]
    # labels sit above-right of the point so the horizontal error bar does not run through them;
    # the two crowded grey points swap sides between panels
    flips = {"0.15": {"random/adaptive": (6, -13)}, "0.4": {"maxdeg/adaptive": (6, -13)}}
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), facecolor=SURFACE, sharey=True)
    for ax, tf in zip(axes, res):
        style(ax)
        for k, lab, col in keys:
            off = flips.get(tf, {}).get(k, (6, 13))
            r = res[tf][k]
            e = np.array(r["edges"]["200"], float)
            g = np.array(r["acc"]["200"]) - np.array(r["acc"]["0"])
            n = len(e)
            ax.errorbar(e.mean(), g.mean(), xerr=e.std(ddof=1) / np.sqrt(n), yerr=g.std(ddof=1) / np.sqrt(n),
                        fmt="o", ms=8, color=col, ecolor=col, elinewidth=1.0, capsize=0, zorder=3)
            ax.annotate(lab, (e.mean(), g.mean()), xytext=off, textcoords="offset points", fontsize=9.5,
                        color=INK, va="center")
        ax.set_xlim(6.5, 13.2)
        ax.set_ylim(-0.03, 0.28)
        ax.set_title(f"{float(tf):.0%} of nodes labelled", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("Real links found in 200 queries (average)", fontsize=10, color=INK)
    axes[0].set_ylabel("Accuracy gain on target nodes", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_edges_vs_accuracy.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_attributes():
    secs, cur = {}, None
    for line in open(os.path.join(HERE, "check_node_attributes_log.txt")):
        m = re.match(r"=== locked\(.*\), labelled=(\d+)%", line)
        if m:
            cur = int(m.group(1))
            secs[cur] = []
            continue
        if line.startswith("==="):
            cur = None
            continue
        m = re.match(r"\s+(\d\.\d)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)", line)
        if cur and m:
            secs[cur].append(tuple(map(float, m.groups())))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), facecolor=SURFACE, sharey=True)
    for ax, tf in zip(axes, sorted(secs)):
        style(ax)
        d = np.array(secs[tf])
        x = d[:, 0]
        ax.axhline(1 / 3, color=GREY, lw=1.0, ls=(0, (4, 3)))
        ax.text(1.0, 1 / 3 - 0.012, "guessing", ha="right", va="top", fontsize=9, color=MUTED)
        ax.axvline(0.1, color=AXIS, lw=1.0)
        ax.text(0.115, 1.0, "our setting", fontsize=9, color=MUTED, va="top")
        for col, vals, lab, off in [(BLUE, d[:, 3], "label propagation only", (6, 0)),
                                    (ORANGE, d[:, 2], "attributes only", (6, -9)),
                                    (AQUA, d[:, 5], "both combined", (6, 9))]:
            ax.plot(x, vals, color=col, lw=2.0, marker="o", ms=5, zorder=3)
            ax.annotate(lab, (x[-1], vals[-1]), xytext=off, textcoords="offset points", fontsize=9.5,
                        color=INK, va="center", annotation_clip=False)
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(0.2, 1.02)
        ax.set_title(f"{tf}% of nodes labelled", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("How informative the node attributes are", fontsize=10, color=INK)
    axes[0].set_ylabel("Accuracy on unlabelled nodes (no queries)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig.savefig(os.path.join(OUT, "fig_attributes.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _load(name):
    return json.load(open(os.path.join(HERE, name)))


def _gain(v, b="200"):
    return float(np.mean(np.array(v["acc"][b]) - np.array(v["acc"]["0"])))


def fig_reach_vs_gain():
    res = _load("e1_active_2x2.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), facecolor=SURFACE, sharey=True)
    out = {}
    for ax, tf in zip(axes, res):
        style(ax)
        xs, ys = [], []
        for k, v in res[tf].items():
            if k.startswith("value_knownp"):
                continue
            method, mode = k.split("/")
            reach, gain = float(np.mean(v["reach"]["200"])), _gain(v)
            xs.append(reach)
            ys.append(gain)
            col = BLUE if method == "value" else AQUA if method == "value_all" else GREY
            ax.scatter([reach], [gain], s=46, marker="o" if mode == "adaptive" else "s",
                       facecolors=col if mode == "adaptive" else "none", edgecolors=col, linewidths=1.6, zorder=3)
        r = float(np.corrcoef(xs, ys)[0, 1])
        out[tf] = (r, len(xs))
        ax.text(0.03, 0.95, f"correlation r = {r:.3f}", transform=ax.transAxes, fontsize=9.5, color=INK, va="top")
        ax.set_title(f"{float(tf):.0%} of nodes labelled", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("Targets reached by a label after 200 queries", fontsize=10, color=INK)
    axes[0].set_ylabel("Accuracy gain on target nodes", fontsize=10, color=INK)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, label="value, targets"),
           Line2D([], [], marker="o", ls="", mfc=AQUA, mec=AQUA, label="value, all nodes"),
           Line2D([], [], marker="o", ls="", mfc=GREY, mec=GREY, label="simple strategies"),
           Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, label="adaptive (filled)"),
           Line2D([], [], marker="s", ls="", mfc="none", mec=INK, label="static (open)")]
    fig.legend(handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_reach_vs_gain.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("reach-vs-gain correlations (r, n points):", out)


def fig_frontier():
    g, f = _load("e1_active_2x2.json"), _load("e5_frontier.json")
    strats = [("random", "random"), ("common_nbrs", "common neighbours"), ("maxdeg", "max-degree"),
              ("pagerank", "PageRank"), ("value", "value, targets")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), facecolor=SURFACE, sharey=True)
    for ax, tf in zip(axes, g):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        for i, (s, lab) in enumerate(strats):
            xg, xf = _gain(g[tf][f"{s}/adaptive"]), _gain(f[tf][f"{s}/frontier"])
            ax.plot([xg, xf], [i, i], color=AXIS, lw=2.0, zorder=1)
            ax.scatter([xg], [i], s=62, facecolors=SURFACE, edgecolors=INK, linewidths=1.6, zorder=3)
            ax.scatter([xf], [i], s=62, color=BLUE, zorder=3)
        ax.set_yticks(range(len(strats)))
        ax.set_yticklabels([lab for _, lab in strats], fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.28)
        ax.set_title(f"{float(tf):.0%} of nodes labelled", loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel("Accuracy gain on target nodes after 200 queries", fontsize=10, color=INK)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK, mew=1.6, label="queries about any pair"),
           Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, label="queries only near the targets")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_frontier.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_removal():
    schemes = [("random", "visible links chosen at random", BLUE),
               ("target_hidden", "no visible link touches a target", ORANGE),
               ("hub_observed", "visible links favour hub nodes", AQUA)]
    data = {s: _load(f"e7_{s}.json") for s, _, _ in schemes}
    strats = [("random/adaptive", "random"), ("common_nbrs/adaptive", "common\nneighbours"),
              ("maxdeg/adaptive", "max\ndegree"), ("pagerank/adaptive", "Page\nRank"),
              ("value/static", "value\n(static)"), ("value/adaptive", "value\n(adaptive)")]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), facecolor=SURFACE, sharey=True)
    for ax, tf in zip(axes, data["random"]):
        style(ax)
        for j, (s, lab, col) in enumerate(schemes):
            r = data[s][tf]
            gcs = []
            for k, _ in strats:
                v = r[k]
                a0, a200 = np.mean(v["acc"]["0"]), np.mean(v["acc"]["200"])
                gcs.append((a200 - a0) / (np.mean(v["ceiling"]) - a0))
            ax.scatter(np.arange(len(strats)) + (j - 1) * 0.2, gcs, s=50, color=col, zorder=3)
        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels([lab for _, lab in strats], fontsize=8.5, color=INK)
        ax.set_ylim(0, 0.72)
        ax.set_title(f"{float(tf):.0%} of nodes labelled", loc="left", fontsize=11.5, color=INK)
    axes[0].set_ylabel("Share of possible improvement achieved\n(gap closed, 200 queries)", fontsize=10, color=INK)
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", mfc=c, mec=c, label=l) for _, l, c in schemes]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_removal.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def copy_and_regenerate():
    shutil.copy(os.path.join(HERE, "fig_e4_cluster_formation.png"), os.path.join(OUT, "fig_cluster_formation.png"))
    shutil.copy(os.path.join(HERE, "fig_headline_e3_400.png"), os.path.join(OUT, "fig_headline_400.png"))
    subprocess.run([sys.executable, os.path.join(HERE, "fig_headline.py"), "--json", "r1_fresh_seeds.json",
                    "--out", os.path.join("report", "figures", "fig_headline_200.png")], check=True)


if __name__ == "__main__":
    fig_pipeline()
    fig_edges_vs_accuracy()
    fig_attributes()
    fig_reach_vs_gain()
    fig_frontier()
    fig_removal()
    copy_and_regenerate()
    print(sorted(os.listdir(OUT)))
