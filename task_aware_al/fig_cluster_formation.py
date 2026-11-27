"""E4: do clusters form as queries are spent, and are the TARGET nodes getting classified?

One illustrative run (seed 0, locked graph, 40% labelled, 15 targets) -- NOT a typical-case
claim; the statistics are in E1. Top row: random queries. Bottom row: value (task-aware,
adaptive) queries. Both start from the identical state.

Reading a panel:
  fill colour     predicted class (label propagation); grey = no label reaches the node yet
  square          labelled node (class known)
  thick ring      target node;  x on it = target currently classified WRONG
  thin outline    true class of every other unlabelled node (fill != outline = wrong)
  grey edges      initially observed;  dark edges = revealed by queries
Layout is a spring layout of the CURRENTLY observed graph, warm-started from the previous
snapshot, so clusters appear only as edges are revealed.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import Instance, solve_lp, predict, reach  # noqa: E402
from exp_active_2x2 import SCORERS  # noqa: E402

SEED, TRAIN_FRAC, GRAPH = 0, 0.40, "locked"
SNAPS = [0, 50, 100, 200]
ROWS = [("random", "Random queries"), ("value", "Value queries (task-aware, adaptive)")]

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
CLASS_COLOR = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
UNREACHED = "#d9d8d2"
EDGE_INIT, EDGE_NEW = "#c3c2b7", "#0b0b0b"


def run(strategy):
    inst = Instance(SEED, TRAIN_FRAC, GRAPH)
    env = inst.env
    init_edges = {tuple(sorted(e)) for e in zip(*np.nonzero(np.triu(env.A_obs)))}
    snaps = {}

    def snap(step):
        A = inst.A_obs()
        F = solve_lp(A, inst.y, env.train_idx, inst.k)
        unreached = np.abs(F).sum(axis=1) < 1e-12
        snaps[step] = dict(A=A.copy(), pred=predict(F), unreached=unreached,
                           acc=float((predict(F)[inst.targets] == inst.y[inst.targets]).mean()),
                           reach=reach(A, env.train_idx, inst.targets))

    snap(0)
    for step in range(1, max(SNAPS) + 1):
        cands = env.candidates()
        s = SCORERS[strategy](inst, cands)
        env.reveal(cands[int(env.rng.choice(np.flatnonzero(s == s.max())))])
        if step in SNAPS:
            snap(step)
    return inst, init_edges, snaps


COLS = 24


def layout(inst, A, prev):
    """Spring layout of nodes that have an observed edge (warm-started from the previous
    snapshot, rescaled to the unit square); nodes with no observed edge go in a grid strip
    below, labelled first, then targets, then the rest."""
    n = len(inst.y)
    deg = A.sum(axis=1)
    conn = [v for v in range(n) if deg[v] > 0]
    G = nx.Graph()
    G.add_nodes_from(conn)
    G.add_edges_from(zip(*np.nonzero(np.triu(A))))
    init = {v: prev[v] for v in conn if prev is not None and v in prev}
    p = nx.spring_layout(G, k=0.6, pos=init or None, seed=SEED, iterations=60 if init else 150)
    xy = np.array([p[v] for v in conn])
    xy = (xy - xy.min(axis=0)) / max(np.ptp(xy, axis=0).max(), 1e-9)
    pos = {v: tuple(xy[i]) for i, v in enumerate(conn)}
    lab, tgt = set(inst.env.train_idx.tolist()), set(inst.targets.tolist())
    iso = sorted((v for v in range(n) if deg[v] == 0), key=lambda v: (v not in lab, v not in tgt, v))
    for k, v in enumerate(iso):
        pos[v] = ((k % COLS) / (COLS - 1), -0.42 - 0.09 * (k // COLS))
    return pos, len(iso)


def draw(ax, inst, init_edges, snap, pos, n_iso):
    n = len(inst.y)
    A = snap["A"]
    ax.text(0.0, -0.285, f"{n_iso} nodes with no observed edge (below)", fontsize=8.5, color=MUTED, va="bottom")
    edges = [tuple(sorted(e)) for e in zip(*np.nonzero(np.triu(A)))]
    for (i, j) in edges:
        new = (i, j) not in init_edges
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=EDGE_NEW if new else EDGE_INIT,
                lw=1.3 if new else 0.8, zorder=1, solid_capstyle="round")
    lab = set(inst.env.train_idx.tolist())
    tgt = set(inst.targets.tolist())
    for v in range(n):
        x, y = pos[v]
        fill = UNREACHED if snap["unreached"][v] else CLASS_COLOR[snap["pred"][v]]
        if v in lab:
            ax.scatter([x], [y], s=46, marker="s", c=CLASS_COLOR[inst.y[v]], edgecolors=INK, linewidths=0.9, zorder=3)
        elif v in tgt:
            ax.scatter([x], [y], s=88, marker="o", c=fill, edgecolors=INK, linewidths=2.0, zorder=4)
            if snap["pred"][v] != inst.y[v]:
                ax.scatter([x], [y], s=60, marker="x", c=INK, linewidths=1.6, zorder=5)
        else:
            ax.scatter([x], [y], s=30, marker="o", c=fill, edgecolors=CLASS_COLOR[inst.y[v]], linewidths=1.1, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    # portrait layout: one row per snapshot, one column per strategy, so it fits an upright page
    fig, axes = plt.subplots(len(SNAPS), len(ROWS), figsize=(7.2, 9.6), facecolor=SURFACE)
    for r, (strategy, title) in enumerate(ROWS):
        inst, init_edges, snaps = run(strategy)
        pos = None
        for c, step in enumerate(SNAPS):
            pos, n_iso = layout(inst, snaps[step]["A"], pos)
            ax = axes[c, r]
            ax.set_facecolor(SURFACE)
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.66, 1.08)
            draw(ax, inst, init_edges, snaps[step], pos, n_iso)
            s = snaps[step]
            ax.set_title(f"{step} queries" if step else "start (no queries)", fontsize=10, color=INK, loc="left")
            ax.text(0.0, -0.03, f"target accuracy {s['acc']:.2f}   targets reached {s['reach']:.2f}",
                    transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

    legend = [
        Line2D([], [], marker="o", ls="", mfc=CLASS_COLOR[0], mec=CLASS_COLOR[0], label="predicted class 1"),
        Line2D([], [], marker="o", ls="", mfc=CLASS_COLOR[1], mec=CLASS_COLOR[1], label="predicted class 2"),
        Line2D([], [], marker="o", ls="", mfc=CLASS_COLOR[2], mec=CLASS_COLOR[2], label="predicted class 3"),
        Line2D([], [], marker="o", ls="", mfc=UNREACHED, mec=UNREACHED, label="no label reaches it"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=INK, label="labelled node"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=2, ms=9, label="target node"),
        Line2D([], [], marker="x", ls="", mec=INK, mew=1.6, label="target classified wrong"),
        Line2D([], [], color=EDGE_INIT, lw=1.2, label="initially observed edge"),
        Line2D([], [], color=EDGE_NEW, lw=1.5, label="edge revealed by a query"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Do clusters form, and do the targets get classified?", fontsize=11.5, color=INK,
                 x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.953, "One run (seed 0, 40% labelled). Grey nodes get a fixed random guess, which counts "
             "in target accuracy.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 0.925))
    for r, (_, title) in enumerate(ROWS):
        p = axes[0, r].get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.035, title, ha="center", fontsize=10.5, color=INK)
    out = os.path.join(HERE, "fig_e4_cluster_formation.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("saved", out)


if __name__ == "__main__":
    main()
