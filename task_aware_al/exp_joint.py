"""Joint label + edge active acquisition experiment.

Unified framework where each oracle call can either:
  (a) reveal whether a candidate edge exists (same as cora_exp), or
  (b) reveal an unlabeled node's true class label.

The joint scorer picks whichever single action maximises expected LP accuracy
gain on the target set, using the same probe logic for both action types.

Includes monotonicity tracking (per-step accuracy delta) and regime logging
(graph state at each step where joint picks label over edge).

Usage:
  python exp_joint.py --seeds 5 --budget 200 --subgraph 150 --dataset cora \
      --strategies joint_value,edge_only,label_only,alternating,random \
      --out results/joint_cora_s42.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cora_exp import (
    Instance, lp_solve, beliefs, lp_accuracy, gcn_accuracy, snapshot,
    load_cora_subgraph, DATASET_LOADERS, OBSERVE_FRAC, SHORTLIST,
    sc_pagerank, sc_linkprob, _train_gcn, _gcn_conf,
)

LABEL_SHORTLIST = 60


# ---------------------------------------------------------------- label probing
def _labelable(inst):
    """Indices of nodes that can be labeled: unlabeled and not a target."""
    tgt_set = set(inst.targets.tolist())
    return np.array([i for i in range(inst.n) if not inst.lab[i] and i not in tgt_set])


def score_label_random(inst):
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool):
        scores[pool] = inst.rng.random(len(pool))
    return scores


def score_label_uncertainty(inst):
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    scores[pool] = ent[pool] + 1e-9 * inst.rng.random(len(pool))
    return scores


def score_label_value(inst):
    """LP-probe label scorer: for each candidate node, probe the LP accuracy gain
    on targets if that node's true label were added to the training set."""
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores

    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)

    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets].tolist())
    connected_to_tgt = np.array([comp[v] in tgt_comps for v in pool])

    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    key = ent[pool] * connected_to_tgt
    order = np.argsort(-key)[:LABEL_SHORTLIST]

    for rank in order:
        v = pool[rank]
        sub_nodes = np.flatnonzero(comp == comp[v])
        tgt_local = np.isin(sub_nodes, inst.targets)
        if not tgt_local.any():
            scores[v] = 1e-9
            continue
        Asub = A[sub_nodes][:, sub_nodes]
        lab_sub = inst.lab[sub_nodes].copy()
        lv = np.searchsorted(sub_nodes, v)
        if lv < len(sub_nodes) and sub_nodes[lv] == v:
            lab_sub[lv] = True
        F2 = lp_solve(Asub, inst.y[sub_nodes], lab_sub, inst.k)
        q2 = beliefs(F2)
        gain = (q2.max(axis=1)[tgt_local] - q[sub_nodes][tgt_local].max(axis=1)).sum()
        scores[v] = max(gain / len(inst.targets), 1e-9)

    return scores


def score_label_infogain(inst):
    """Entropy-reduction label scorer: pick the node whose labeling reduces
    target entropy the most."""
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores

    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    tgt_ent = -(q[inst.targets] * np.log(q[inst.targets] + 1e-12)).sum(axis=1)

    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets].tolist())
    connected_to_tgt = np.array([comp[v] in tgt_comps for v in pool])

    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    key = ent[pool] * connected_to_tgt
    order = np.argsort(-key)[:LABEL_SHORTLIST]

    for rank in order:
        v = pool[rank]
        sub_nodes = np.flatnonzero(comp == comp[v])
        tgt_local = np.isin(sub_nodes, inst.targets)
        if not tgt_local.any():
            scores[v] = 1e-9
            continue
        Asub = A[sub_nodes][:, sub_nodes]
        lab_sub = inst.lab[sub_nodes].copy()
        lv = np.searchsorted(sub_nodes, v)
        if lv < len(sub_nodes) and sub_nodes[lv] == v:
            lab_sub[lv] = True
        F2 = lp_solve(Asub, inst.y[sub_nodes], lab_sub, inst.k)
        q2 = beliefs(F2)
        ent_before = -(q[sub_nodes][tgt_local] * np.log(q[sub_nodes][tgt_local] + 1e-12)).sum(axis=1)
        ent_after = -(q2[tgt_local] * np.log(q2[tgt_local] + 1e-12)).sum(axis=1)
        scores[v] = max((ent_before - ent_after).sum() / len(inst.targets), 1e-9)

    return scores


# ---------------------------------------------------------------- GCN-embedding edge scorer
def _gcn_embeddings(inst):
    """Train GCN, return hidden-layer embeddings (n × 64) as numpy array."""
    import torch
    A = inst.A_obs()
    W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k,
                              seed=inst.seed, epochs=100)
    n = A.shape[0]
    Ah = (A + sp.identity(n)).tocoo()
    deg = np.asarray(Ah.sum(axis=1)).ravel()
    w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
    Ahat = torch.sparse_coo_tensor(
        np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
    with torch.no_grad():
        Z = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
    return Z.numpy()


def _best_edge_gcn_emb(inst, tau=1.0):
    """LP-probe edge scorer re-ranked by GCN embedding proximity.

    1. Train GCN once → get embeddings Z.
    2. Build two shortlists: LP-probe heuristic + embedding-proximity.
    3. LP-probe all candidates in the union.
    4. Final score = lp_gain(e) × exp(-‖z_u - z_v‖² / τ).

    The exponential weight boosts edges the GCN "expects" (close in embedding
    space) and suppresses ones it doesn't need (far apart).
    """
    Z = _gcn_embeddings(inst)
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & alive

    # Embedding distances for all candidates
    diff = Z[I] - Z[J]
    dist_sq = (diff * diff).sum(axis=1)
    emb_weight = np.exp(-dist_sq / tau)

    # --- LP-probe shortlist (standard) ---
    key_lp = p * np.abs(conf[I] - conf[J]) * rel
    order_lp = np.argsort(-key_lp)[:SHORTLIST]
    order_lp = order_lp[key_lp[order_lp] > 0]

    # --- Embedding-proximity shortlist: close in GCN space, alive, relevant ---
    key_emb = emb_weight * p * rel
    order_emb = np.argsort(-key_emb)[:SHORTLIST]
    order_emb = order_emb[key_emb[order_emb] > 0]

    to_probe = np.unique(np.concatenate([order_lp, order_emb]))

    edge_scores = 1e-9 * p * rel
    edge_scores[~alive] = -np.inf
    n_obj = len(inst.targets)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    for c in to_probe:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        lp_gain = p[c] * (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
        edge_scores[c] += lp_gain * emb_weight[c]
    return edge_scores


def _gcn_emb_joint_step(inst, tau=1.0):
    """Joint step: GCN-embedding re-ranking of edges, LP-probe for labels.

    Embedding weight only re-ranks edges. Edge-vs-label decision uses the
    raw LP gain of the winning edge so embeddings can't inflate edges above labels.
    """
    emb_edge_scores = _best_edge_gcn_emb(inst, tau=tau)
    best_e = int(np.argmax(emb_edge_scores))
    raw_edge_scores = _best_edge_value(inst)
    label_scores = score_label_value(inst)
    best_l = int(np.argmax(label_scores))
    if raw_edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


# ---------------------------------------------------------------- edge scorers (wrappers for joint use)
def _best_edge_value(inst):
    """LP-probe edge scorer (same as sc_value in cora_exp)."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & alive
    key = p * np.abs(conf[I] - conf[J]) * rel
    edge_scores = 1e-9 * p * rel
    edge_scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(inst.targets)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        edge_scores[c] += p[c] * (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
    return edge_scores


def _best_edge_infogain(inst):
    """Entropy-reduction edge scorer."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    I, J = inst.PI, inst.PJ
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & inst.alive
    key = p * rel
    scores = 1e-9 * p * rel
    scores[~inst.alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        q2 = beliefs(F2)
        ent_before = -(q[nodes][sub_tgt] * np.log(q[nodes][sub_tgt] + 1e-12)).sum(axis=1)
        ent_after = -(q2[sub_tgt] * np.log(q2[sub_tgt] + 1e-12)).sum(axis=1)
        scores[c] += p[c] * (ent_before - ent_after).sum() / len(inst.targets)
    return scores


# ---------------------------------------------------------------- GCN-based joint scorer
def _gcn_edge_scores(inst, W1, W2, Xt, conf_base):
    """Score each candidate edge by GCN confidence gain on targets."""
    A = inst.A_obs()
    I, J = inst.PI, inst.PJ
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & inst.alive
    key = p * np.abs(conf_base[I] - conf_base[J]) * rel
    scores = 1e-9 * p * rel
    scores[~inst.alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(inst.targets)
    for c in order:
        i, j = I[c], J[c]
        E = sp.csr_matrix(([1.0, 1.0, 1.0, 1.0],
                           ([i, j, j, i], [j, i, i, j])), shape=A.shape)
        A_mod = A + E
        A_mod.data = np.minimum(A_mod.data, 1.0)
        conf2 = _gcn_conf(W1, W2, Xt, A_mod)
        scores[c] += p[c] * (conf2[inst.targets] - conf_base[inst.targets]).sum() / n_obj
    return scores


def _gcn_label_scores(inst, W1, W2, Xt, conf_base):
    """Score each candidate label by GCN confidence gain on targets."""
    A = inst.A_obs()
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores
    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets].tolist())
    connected_to_tgt = np.array([comp[v] in tgt_comps for v in pool])
    # Use GCN confidence as proxy for uncertainty
    ent_proxy = 1.0 - conf_base[pool]
    key = ent_proxy * connected_to_tgt
    order = np.argsort(-key)[:LABEL_SHORTLIST]
    n_obj = len(inst.targets)
    for rank in order:
        v = pool[rank]
        # Temporarily add label: retrain is too expensive, so we re-evaluate
        # with the label clamped in the GCN input
        lab_new = inst.lab.copy()
        lab_new[v] = True
        train_new = np.append(inst.train, v)
        W1n, W2n, Xtn = _train_gcn(A, inst.X, inst.y, train_new, inst.k,
                                     seed=inst.seed, epochs=50)
        conf2 = _gcn_conf(W1n, W2n, Xtn, A)
        gain = (conf2[inst.targets] - conf_base[inst.targets]).sum() / n_obj
        scores[v] = max(gain, 1e-9)
    return scores


def score_label_gcn_uncertainty(inst):
    """Score labels by GCN prediction uncertainty on targets.
    Picks the unlabeled node whose labeling would most reduce GCN uncertainty,
    approximated by: nodes in target-connected components with high GCN entropy."""
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores
    A = inst.A_obs()
    W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k,
                              seed=inst.seed, epochs=100)
    conf_base = _gcn_conf(W1, W2, Xt, A)
    gcn_ent = 1.0 - conf_base
    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets].tolist())
    connected_to_tgt = np.array([comp[v] in tgt_comps for v in pool])
    scores[pool] = gcn_ent[pool] * connected_to_tgt + 1e-9 * inst.rng.random(len(pool))
    return scores


def joint_split_step(inst):
    """Split scorer: LP-probe decides edge vs label (balanced allocation),
    but when label is chosen, pick WHICH label by GCN uncertainty."""
    edge_scores = _best_edge_value(inst)
    lp_label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l_lp = int(np.argmax(lp_label_scores))
    if edge_scores[best_e] >= lp_label_scores[best_l_lp]:
        return "edge", best_e
    gcn_label_scores = score_label_gcn_uncertainty(inst)
    return "label", int(np.argmax(gcn_label_scores))


def joint_gcn_step(inst):
    """Joint scorer using GCN probes instead of LP probes."""
    A = inst.A_obs()
    W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k,
                              seed=inst.seed, epochs=100)
    conf_base = _gcn_conf(W1, W2, Xt, A)
    edge_scores = _gcn_edge_scores(inst, W1, W2, Xt, conf_base)
    label_scores = _gcn_label_scores(inst, W1, W2, Xt, conf_base)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def _gcn_label_scores_fast(inst, W1, W2, Xt, conf_base, Ahat_base):
    """Score labels by fine-tuning from base weights (5 steps) instead of retraining."""
    import torch, copy
    A = inst.A_obs()
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores
    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets].tolist())
    connected_to_tgt = np.array([comp[v] in tgt_comps for v in pool])
    ent_proxy = 1.0 - conf_base[pool]
    key = ent_proxy * connected_to_tgt
    order = np.argsort(-key)[:LABEL_SHORTLIST]
    n_obj = len(inst.targets)
    yt = torch.from_numpy(inst.y).long()
    for rank in order:
        v = pool[rank]
        train_new = np.append(inst.train, v)
        tr = torch.from_numpy(train_new).long()
        W1c = copy.deepcopy(W1)
        W2c = copy.deepcopy(W2)
        W1c.train(); W2c.train()
        opt = torch.optim.Adam(list(W1c.parameters()) + list(W2c.parameters()),
                               lr=0.01, weight_decay=5e-4)
        for _ in range(5):
            opt.zero_grad()
            h = torch.relu(torch.sparse.mm(Ahat_base, W1c(Xt)))
            torch.nn.functional.cross_entropy(
                torch.sparse.mm(Ahat_base, W2c(h))[tr], yt[tr]).backward()
            opt.step()
        W1c.eval(); W2c.eval()
        conf2 = _gcn_conf(W1c, W2c, Xt, A)
        gain = (conf2[inst.targets] - conf_base[inst.targets]).sum() / n_obj
        scores[v] = max(gain, 1e-9)
    return scores


def _make_ahat(A):
    """Build normalised adjacency tensor for GCN forward pass."""
    import torch
    n = A.shape[0]
    Ah = (A + sp.identity(n)).tocoo()
    deg = np.asarray(Ah.sum(axis=1)).ravel()
    w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
    return torch.sparse_coo_tensor(
        np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()


def _gcn_pred(W1, W2, Xt, A):
    """GCN forward pass returning predicted class per node."""
    import torch
    n = A.shape[0]
    Ah = (A + sp.identity(n)).tocoo()
    deg = np.asarray(Ah.sum(axis=1)).ravel()
    w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
    Ahat = torch.sparse_coo_tensor(
        np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
    with torch.no_grad():
        h = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
        logits = torch.sparse.mm(Ahat, W2(h))
    return logits.argmax(dim=1).numpy()


def _gcn_edge_scores_acc(inst, W1, W2, Xt, pred_base):
    """Score edges by GCN accuracy change on targets (not confidence change)."""
    A = inst.A_obs()
    I, J = inst.PI, inst.PJ
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & inst.alive
    conf_base = np.zeros(inst.n)
    conf_base[inst.targets] = (pred_base[inst.targets] == inst.y[inst.targets]).astype(float)
    base_acc = conf_base[inst.targets].sum()
    key = p * rel
    scores = 1e-9 * p * rel
    scores[~inst.alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(inst.targets)
    for c in order:
        i, j = I[c], J[c]
        E = sp.csr_matrix(([1.0, 1.0, 1.0, 1.0],
                           ([i, j, j, i], [j, i, i, j])), shape=A.shape)
        A_mod = A + E
        A_mod.data = np.minimum(A_mod.data, 1.0)
        pred2 = _gcn_pred(W1, W2, Xt, A_mod)
        acc2 = (pred2[inst.targets] == inst.y[inst.targets]).sum()
        scores[c] += p[c] * (acc2 - base_acc) / n_obj
    return scores


def joint_gcn_fast_step(inst):
    """Fast GNN-in-the-loop joint scorer.

    Edge scoring: forward-pass probe (add edge, same weights, measure acc change).
    Label scoring: fine-tune probe (5 gradient steps from base weights).
    """
    import torch
    A = inst.A_obs()
    W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k,
                              seed=inst.seed, epochs=100)
    conf_base = _gcn_conf(W1, W2, Xt, A)
    pred_base = _gcn_pred(W1, W2, Xt, A)
    Ahat_base = _make_ahat(A)
    edge_scores = _gcn_edge_scores(inst, W1, W2, Xt, conf_base)
    label_scores = _gcn_label_scores_fast(inst, W1, W2, Xt, conf_base, Ahat_base)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


# ---------------------------------------------------------------- 2-step lookahead joint scorer
def joint_lookahead_step(inst):
    """Joint scorer with synergy-aware 2-step lookahead.

    Finds the best edge and best label candidate (using LP probes as usual),
    then checks which ordering gives higher 2-step gain:
      (a) edge first → then best label on the modified graph
      (b) label first → then best edge on the modified graph
    Picks whichever first action leads to higher combined gain.
    Falls back to single-step if only one action type is available.
    """
    edge_scores = _best_edge_value(inst)
    label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))

    e_gain = edge_scores[best_e]
    l_gain = label_scores[best_l]

    # If one type has no viable candidate, pick the other
    if e_gain <= 0 and l_gain <= 0:
        return "edge", best_e  # fallback
    if e_gain <= 0:
        return "label", best_l
    if l_gain <= 0:
        return "edge", best_e

    # 2-step lookahead: simulate each ordering and measure combined gain
    A = inst.A_obs()

    # Path A: edge first, then label
    # Simulate adding the edge
    i, j = inst.PI[best_e], inst.PJ[best_e]
    p_edge = inst.p_real()[best_e]
    E = sp.csr_matrix(([1.0, 1.0], ([i, j], [j, i])), shape=A.shape)
    A_after_edge = A + E

    F_ae = lp_solve(A_after_edge, inst.y, inst.lab, inst.k)
    q_ae = beliefs(F_ae)
    # Check what the best label would gain on this modified graph
    _, comp_ae = connected_components(A_after_edge, directed=False)
    tgt_comps_ae = set(comp_ae[inst.targets].tolist())
    pool = _labelable(inst)
    best_label_gain_after_edge = 0.0
    if len(pool) > 0:
        ent_ae = -(q_ae * np.log(q_ae + 1e-12)).sum(axis=1)
        connected = np.array([comp_ae[v] in tgt_comps_ae for v in pool])
        key = ent_ae[pool] * connected
        top_v = pool[np.argmax(key)]
        sub_nodes = np.flatnonzero(comp_ae == comp_ae[top_v])
        tgt_local = np.isin(sub_nodes, inst.targets)
        if tgt_local.any():
            Asub = A_after_edge[sub_nodes][:, sub_nodes]
            lab_tmp = inst.lab[sub_nodes].copy()
            lv = np.searchsorted(sub_nodes, top_v)
            if lv < len(sub_nodes) and sub_nodes[lv] == top_v:
                lab_tmp[lv] = True
            F2 = lp_solve(Asub, inst.y[sub_nodes], lab_tmp, inst.k)
            q2 = beliefs(F2)
            best_label_gain_after_edge = (q2.max(axis=1)[tgt_local] -
                                          q_ae[sub_nodes][tgt_local].max(axis=1)).sum()
    path_a = e_gain + best_label_gain_after_edge / max(len(inst.targets), 1)

    # Path B: label first, then edge
    lab_tmp_b = inst.lab.copy()
    lab_tmp_b[best_l] = True
    F_al = lp_solve(A, inst.y, lab_tmp_b, inst.k)
    q_al = beliefs(F_al)
    conf_al = q_al.max(axis=1)
    # Check what the best edge would gain on this modified label set
    _, comp_b = connected_components(A, directed=False)
    relevant_b = np.isin(comp_b, comp_b[inst.targets])
    I, J = inst.PI, inst.PJ
    p = inst.p_real()
    rel_b = (relevant_b[I] | relevant_b[J]) & inst.alive
    key_b = p * np.abs(conf_al[I] - conf_al[J]) * rel_b
    top_e_b = np.argsort(-key_b)[:3]
    best_edge_gain_after_label = 0.0
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    for c in top_e_b:
        if key_b[c] <= 0:
            break
        ci, cj = I[c], J[c]
        nodes = np.flatnonzero((comp_b == comp_b[ci]) | (comp_b == comp_b[cj]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [ci, cj])
        Ee = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2b = lp_solve(Asub + Ee, inst.y[nodes], lab_tmp_b[nodes], inst.k)
        g = p[c] * (beliefs(F2b).max(axis=1)[sub_tgt] - conf_al[nodes][sub_tgt]).sum()
        if g > best_edge_gain_after_label:
            best_edge_gain_after_label = g
    path_b = l_gain + best_edge_gain_after_label / max(len(inst.targets), 1)

    if path_a >= path_b:
        return "edge", best_e
    return "label", best_l


# ---------------------------------------------------------------- strategy step functions
def joint_value_step(inst):
    edge_scores = _best_edge_value(inst)
    label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def joint_infogain_step(inst):
    edge_scores = _best_edge_infogain(inst)
    label_scores = score_label_infogain(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def edge_only_step(inst):
    return "edge", int(np.argmax(_best_edge_value(inst)))


def label_only_step(inst):
    return "label", int(np.argmax(score_label_value(inst)))


def label_uncertainty_step(inst):
    return "label", int(np.argmax(score_label_uncertainty(inst)))


def label_random_step(inst):
    return "label", int(np.argmax(score_label_random(inst)))


def edge_pagerank_step(inst):
    s = sc_pagerank(inst)
    s[~inst.alive] = -np.inf
    return "edge", int(np.argmax(s))


def edge_linkprob_step(inst):
    s = sc_linkprob(inst)
    s[~inst.alive] = -np.inf
    return "edge", int(np.argmax(s))


def alternating_step(inst, step):
    if step % 2 == 1:
        return "edge", int(np.argmax(_best_edge_value(inst)))
    return "label", int(np.argmax(score_label_value(inst)))


def random_joint_step(inst):
    if inst.rng.random() < 0.5:
        pool = _labelable(inst)
        if len(pool) > 0:
            return "label", int(inst.rng.choice(pool))
    s = inst.rng.random(len(inst.PI))
    s[~inst.alive] = -np.inf
    return "edge", int(np.argmax(s))


def fixed_ratio_step(inst, step, budget, edge_frac=0.5):
    edge_budget = int(budget * edge_frac)
    if step <= edge_budget:
        return "edge", int(np.argmax(_best_edge_value(inst)))
    return "label", int(np.argmax(score_label_value(inst)))


def random_edge_step(inst):
    s = inst.rng.random(len(inst.PI))
    s[~inst.alive] = -np.inf
    return "edge", int(np.argmax(s))


# ---------------------------------------------------------------- NEW BASELINES

def _embedding_edge_scores(inst):
    """ALPINE-style baseline: score edges by node2vec-like embedding proximity.
    Uses spectral embedding of the observed graph (fast, no training).
    Score = p(e) × exp(-‖z_u - z_v‖²) for the top candidates by embedding score.
    Then LP-probe the shortlist to get the final selection (but the shortlist
    is chosen by embedding, not by LP heuristic)."""
    from scipy.sparse.linalg import eigsh
    A = inst.A_obs()
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    L_norm = sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    k_emb = min(32, n - 2)
    try:
        vals, vecs = eigsh(L_norm, k=k_emb, which='SM', tol=1e-4)
        Z = vecs[:, 1:]  # skip the trivial eigenvector
    except Exception:
        Z = np.random.default_rng(inst.seed).standard_normal((n, 16))

    I, J = inst.PI, inst.PJ
    diff = Z[I] - Z[J]
    dist_sq = (diff * diff).sum(axis=1)
    emb_score = np.exp(-dist_sq)
    p = inst.p_real()
    alive = inst.alive
    key = emb_score * p * alive
    scores = 1e-9 * p * alive
    scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]

    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    n_obj = len(inst.targets)
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        scores[c] += p[c] * (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
    return scores


def alpine_joint_step(inst):
    """ALPINE-style baseline: spectral-embedding shortlist for edges,
    LP-probe for labels, joint decision by raw LP gain."""
    edge_scores = _embedding_edge_scores(inst)
    label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def age_label_step(inst):
    """AGE-style label baseline: entropy × PageRank centrality.
    Combines uncertainty with graph centrality for label selection."""
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return "label", 0
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    # PageRank centrality
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        S = sp.diags(np.where(deg > 0, 1.0 / deg, 0.0)) @ A
    alpha_pr = 0.85
    pr = np.ones(n) / n
    for _ in range(30):
        pr = alpha_pr * S.T @ pr + (1 - alpha_pr) / n
    pr /= pr.max() + 1e-12
    scores[pool] = ent[pool] * pr[pool] + 1e-9 * inst.rng.random(len(pool))
    return "label", int(np.argmax(scores))


def oracle_step(inst):
    """Oracle upper bound: try every shortlisted candidate, pick the one
    that ACTUALLY gives the best LP accuracy gain on targets.
    Cheats by using true labels/edges. Expensive but only for reference."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    baseline = conf[inst.targets].mean()
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    n_obj = len(inst.targets)

    best_gain, best_action, best_idx = -np.inf, "edge", 0

    # Try top edge candidates (by link prob, no LP heuristic needed — oracle knows truth)
    p = inst.p_real()
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & alive
    # Only try edges that actually exist (oracle knows)
    true_alive = inst.is_true & alive
    true_idx = np.flatnonzero(true_alive)
    if len(true_idx) > SHORTLIST:
        key = np.abs(conf[I] - conf[J]) * true_alive
        true_idx = np.argsort(-key)[:SHORTLIST]
        true_idx = true_idx[key[true_idx] > 0]
    for c in true_idx:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        gain = (beliefs(F2).max(axis=1)[sub_tgt] - conf[nodes][sub_tgt]).sum() / n_obj
        if gain > best_gain:
            best_gain, best_action, best_idx = gain, "edge", c

    # Try top label candidates
    pool = _labelable(inst)
    if len(pool) > 0:
        tgt_comps = set(comp[inst.targets].tolist())
        connected = np.array([comp[v] in tgt_comps for v in pool])
        ent = -(q * np.log(q + 1e-12)).sum(axis=1)
        key = ent[pool] * connected
        order = np.argsort(-key)[:LABEL_SHORTLIST]
        for rank in order:
            v = pool[rank]
            sub_nodes = np.flatnonzero(comp == comp[v])
            tgt_local = np.isin(sub_nodes, inst.targets)
            if not tgt_local.any():
                continue
            Asub = A[sub_nodes][:, sub_nodes]
            lab_sub = inst.lab[sub_nodes].copy()
            lv = np.searchsorted(sub_nodes, v)
            if lv < len(sub_nodes) and sub_nodes[lv] == v:
                lab_sub[lv] = True
            F2 = lp_solve(Asub, inst.y[sub_nodes], lab_sub, inst.k)
            gain = (beliefs(F2).max(axis=1)[tgt_local] - q[sub_nodes][tgt_local].max(axis=1)).sum() / n_obj
            if gain > best_gain:
                best_gain, best_action, best_idx = gain, "label", v

    return best_action, best_idx


# ---------------------------------------------------------------- ABLATIONS

def _best_edge_no_linkprob(inst):
    """Ablation: LP-probe edge scorer WITHOUT link probability weighting.
    Sets all p̂ = 1 so the shortlisting and scoring ignore existence probability."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    _, comp = connected_components(A, directed=False)
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & alive
    key = np.abs(conf[I] - conf[J]) * rel  # no p multiplication
    edge_scores = 1e-9 * rel.astype(float)
    edge_scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(inst.targets)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    p = inst.p_real()
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        # Score WITHOUT p weighting — raw LP gain
        edge_scores[c] += (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
    return edge_scores


def ablation_no_linkprob_step(inst):
    """Ablation: joint scorer without link probability weighting."""
    edge_scores = _best_edge_no_linkprob(inst)
    label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def _best_edge_random_shortlist(inst):
    """Ablation: LP-probe on a RANDOM shortlist of 100 candidates
    instead of the heuristic top-100. Tests whether shortlisting matters."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    relevant = np.isin(comp, comp[inst.targets])
    rel = (relevant[I] | relevant[J]) & alive
    edge_scores = 1e-9 * p * rel
    edge_scores[~alive] = -np.inf

    alive_idx = np.flatnonzero(alive & rel)
    if len(alive_idx) == 0:
        return edge_scores
    rng = inst.rng
    order = rng.choice(alive_idx, min(SHORTLIST, len(alive_idx)), replace=False)

    n_obj = len(inst.targets)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        edge_scores[c] += p[c] * (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
    return edge_scores


def ablation_rand_shortlist_step(inst):
    """Ablation: joint scorer with random shortlist."""
    edge_scores = _best_edge_random_shortlist(inst)
    label_scores = score_label_value(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


def _best_edge_all_targets(inst):
    """Ablation: LP-probe edge scorer optimising for ALL unlabelled nodes,
    not just the 50 designated targets."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    all_unlabelled = np.flatnonzero(~inst.lab)
    relevant = np.isin(comp, comp[all_unlabelled])
    rel = (relevant[I] | relevant[J]) & alive
    key = p * np.abs(conf[I] - conf[J]) * rel
    edge_scores = 1e-9 * p * rel
    edge_scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(all_unlabelled)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[all_unlabelled] = True
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        edge_scores[c] += p[c] * (conf2 - conf[nodes][sub_tgt]).sum() / n_obj
    return edge_scores


def score_label_all_targets(inst):
    """Ablation: LP-probe label scorer optimising for ALL unlabelled nodes."""
    scores = np.full(inst.n, -np.inf)
    pool = _labelable(inst)
    if len(pool) == 0:
        return scores
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    all_unlabelled = np.flatnonzero(~inst.lab)
    _, comp = connected_components(A, directed=False)
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    key = ent[pool]
    order = np.argsort(-key)[:LABEL_SHORTLIST]
    n_obj = len(all_unlabelled)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[all_unlabelled] = True
    for rank in order:
        v = pool[rank]
        sub_nodes = np.flatnonzero(comp == comp[v])
        sub_tgt = obj_mask[sub_nodes]
        if not sub_tgt.any():
            scores[v] = 1e-9
            continue
        Asub = A[sub_nodes][:, sub_nodes]
        lab_sub = inst.lab[sub_nodes].copy()
        lv = np.searchsorted(sub_nodes, v)
        if lv < len(sub_nodes) and sub_nodes[lv] == v:
            lab_sub[lv] = True
        F2 = lp_solve(Asub, inst.y[sub_nodes], lab_sub, inst.k)
        q2 = beliefs(F2)
        gain = (q2.max(axis=1)[sub_tgt] - q[sub_nodes][sub_tgt].max(axis=1)).sum()
        scores[v] = max(gain / n_obj, 1e-9)
    return scores


def ablation_all_targets_step(inst):
    """Ablation: joint scorer optimising for all unlabelled nodes, not just targets."""
    edge_scores = _best_edge_all_targets(inst)
    label_scores = score_label_all_targets(inst)
    best_e = int(np.argmax(edge_scores))
    best_l = int(np.argmax(label_scores))
    if edge_scores[best_e] >= label_scores[best_l]:
        return "edge", best_e
    return "label", best_l


# Static planning state: computed once per seed, cached globally
_STATIC_PLAN = {}

def _compute_static_plan(inst, budget):
    """Pre-compute all acquisition decisions at step 0, then execute in order.
    This is a 'plan everything upfront' ablation."""
    plan = []
    edge_scores = _best_edge_value(inst)
    label_scores = score_label_value(inst)
    all_edges = np.argsort(-edge_scores)
    all_labels = np.argsort(-label_scores)
    ei, li = 0, 0
    for _ in range(budget):
        while ei < len(all_edges) and edge_scores[all_edges[ei]] <= -np.inf:
            ei += 1
        while li < len(all_labels) and label_scores[all_labels[li]] <= -np.inf:
            li += 1
        e_ok = ei < len(all_edges)
        l_ok = li < len(all_labels)
        if e_ok and (not l_ok or edge_scores[all_edges[ei]] >= label_scores[all_labels[li]]):
            plan.append(("edge", int(all_edges[ei])))
            ei += 1
        elif l_ok:
            plan.append(("label", int(all_labels[li])))
            li += 1
        else:
            break
    return plan


def ablation_static_step(inst, step, budget):
    """Ablation: static planning — score once at step 1, execute the fixed plan."""
    key = inst.seed
    if key not in _STATIC_PLAN or step == 1:
        _STATIC_PLAN[key] = _compute_static_plan(inst, budget)
    plan = _STATIC_PLAN[key]
    idx = step - 1
    if idx < len(plan):
        return plan[idx]
    return "label", int(np.argmax(score_label_random(inst)))


def reach_joint_step(inst):
    """Reach-aware joint scorer: interleave edge queries (to connect unreached targets)
    with label queries (for supervised signal). Strategy: for every 1 edge query attempt,
    do 2 label queries. Edge queries target the highest-probability candidate that would
    connect an unreached target to a labeled component."""
    A = inst.A_obs()
    _, comp = connected_components(A, directed=False)
    labeled_comps = set(comp[inst.train].tolist())
    tgt_unreached = ~np.isin(comp[inst.targets], list(labeled_comps))
    n_unreached = tgt_unreached.sum()
    n_total = int(inst.lab.sum()) - len(inst.train) + (len(inst.obs) - int(A.sum() / 2))

    do_edge = (n_unreached > 0 and n_total % 3 == 0)

    if do_edge:
        I, J = inst.PI, inst.PJ
        p = inst.p_real()
        ci, cj = comp[I], comp[J]
        li = np.isin(ci, list(labeled_comps))
        lj = np.isin(cj, list(labeled_comps))
        tgt_comp_counts = np.zeros(comp.max() + 1)
        for t_idx, t in enumerate(inst.targets):
            if tgt_unreached[t_idx]:
                tgt_comp_counts[comp[t]] += 1
        gain_i = tgt_comp_counts[ci] * lj
        gain_j = tgt_comp_counts[cj] * li
        both_unlabeled = tgt_comp_counts[ci] * (~li & ~lj).astype(float) * 0.1
        edge_scores = p * (gain_i + gain_j + both_unlabeled) * inst.alive
        edge_scores[~inst.alive] = -np.inf
        best_e = int(np.argmax(edge_scores))

        if edge_scores[best_e] > 0:
            return "edge", best_e

    label_scores = score_label_value(inst)
    best_l = int(np.argmax(label_scores))
    return "label", best_l


def _hop_distance(A, sources):
    """BFS shortest distance from any source node. Returns array of distances (inf if unreachable)."""
    from scipy.sparse.csgraph import shortest_path
    n = A.shape[0]
    dist = np.full(n, np.inf)
    if len(sources) == 0:
        return dist
    rows = shortest_path(A, directed=False, indices=sources, unweighted=True)
    dist = rows.min(axis=0)
    return dist


def gcn_struct_step(inst):
    """GCN-structural scorer: score edges by how they reduce 2-hop distance from
    targets to labeled nodes (the GCN's receptive field). Adaptively decides
    edge vs label based on fraction of targets within 2-hop reach."""
    A = inst.A_obs()
    n = A.shape[0]
    train_set = inst.train

    dist_to_label = _hop_distance(A, train_set)
    tgt_dist = dist_to_label[inst.targets]
    n_unreachable = np.isinf(tgt_dist).sum()
    n_far = (tgt_dist > 2).sum()  # outside 2-layer GCN receptive field
    frac_covered = 1.0 - n_far / len(inst.targets)

    # Adaptive ratio: query edges when many targets are outside GCN reach
    # Scale by uncovered fraction but account for low pool precision (~10% hit rate)
    # Effective edge value = edge_prob * hit_rate * structural_gain
    # vs label value which is ~1.0 (always informative)
    # So edge_prob should be modest: at most ~30% even when fully uncovered
    raw_need = max(0.0, 1.0 - frac_covered)
    edge_prob = raw_need * 0.35  # 0 when all covered, 0.35 when none covered

    do_edge = (edge_prob > 0 and inst.rng.random() < edge_prob)

    if do_edge:
        I, J = inst.PI, inst.PJ
        p = inst.p_real()
        alive = inst.alive

        di, dj = dist_to_label[I], dist_to_label[J]
        # Score: how much would this edge help targets reach labeled nodes?
        # An edge (i,j) creates path: labeled -> ... -> i -- j -> ... -> target
        # New distance for nodes near j = di + 1 (and vice versa)
        # For each target t, new dist = min(current dist, dist_to_i + 1 + dist_from_j_to_t, ...)
        # Approximate: edge is valuable if one endpoint is close to labels and the other is close to far targets

        _, comp = connected_components(A, directed=False)

        # For each candidate edge, estimate how many targets it would bring within 2-hop
        edge_scores = np.full(len(I), -np.inf)
        tgt_set = set(inst.targets.tolist())

        # Compute distance from each target to all nodes (expensive but accurate)
        # Optimization: only compute for targets that are far
        far_targets = inst.targets[tgt_dist > 2]
        if len(far_targets) > 0:
            dist_from_tgt = _hop_distance(A, far_targets)  # min dist from any far target
            # For edge (i,j): a far target t gains path t->...->j--i->...->labeled
            # new dist(t, labeled) ~= dist(t, j) + 1 + dist(i, labeled) or dist(t, i) + 1 + dist(j, labeled)
            new_dist_via_ij = np.minimum(dist_from_tgt[J] + 1 + di, dist_from_tgt[I] + 1 + dj)
            # This edge helps if new_dist < current min dist for far targets AND new_dist <= 2
            gain = np.maximum(0, (tgt_dist[tgt_dist > 2].min() - np.minimum(new_dist_via_ij, 999)))
            # Simpler: count how the edge reduces total target distance
            # For each far target, estimate dist reduction
            brings_within_reach = (new_dist_via_ij <= 2).astype(float)
            edge_scores = p * brings_within_reach * alive.astype(float)
            edge_scores[~alive] = -np.inf

        # Fallback: if no far targets, use reach-based scoring
        if len(far_targets) == 0 or edge_scores.max() <= 0:
            labeled_comps = set(comp[train_set].tolist())
            tgt_unreached = ~np.isin(comp[inst.targets], list(labeled_comps))
            if tgt_unreached.any():
                ci, cj = comp[I], comp[J]
                li = np.isin(ci, list(labeled_comps))
                lj = np.isin(cj, list(labeled_comps))
                tgt_comp_counts = np.zeros(comp.max() + 1)
                for t_idx, t in enumerate(inst.targets):
                    if tgt_unreached[t_idx]:
                        tgt_comp_counts[comp[t]] += 1
                gain_i = tgt_comp_counts[ci] * lj
                gain_j = tgt_comp_counts[cj] * li
                edge_scores = p * (gain_i + gain_j) * alive.astype(float)
                edge_scores[~alive] = -np.inf

        best_e = int(np.argmax(edge_scores))
        if edge_scores[best_e] > 0:
            return "edge", best_e

    label_scores = score_label_value(inst)
    best_l = int(np.argmax(label_scores))
    return "label", best_l


_PATH_STATE = {}

def _build_edge_lookup(inst):
    """Build node→candidate-edge-indices lookup for path search."""
    adj = {}
    for idx in range(len(inst.PI)):
        if not inst.alive[idx]:
            continue
        i, j = int(inst.PI[idx]), int(inst.PJ[idx])
        adj.setdefault(i, []).append((j, idx))
        adj.setdefault(j, []).append((i, idx))
    return adj


def _find_paths_to_labeled(inst, target, adj, labeled_nodes, comp, max_hops=3, max_paths=20):
    """BFS through candidate edges to find short paths from target to a labeled component.
    Returns list of (path_prob, [edge_idx1, edge_idx2, ...], target_node)."""
    p = inst.p_real()
    paths = []
    # BFS: (current_node, edge_indices_so_far, cumulative_log_prob, visited_nodes)
    queue = [(target, [], 0.0, {target})]
    while queue and len(paths) < max_paths:
        next_queue = []
        for node, edge_path, log_p, visited in queue:
            if node in labeled_nodes and len(edge_path) > 0:
                paths.append((np.exp(log_p), edge_path, target))
                continue
            if len(edge_path) >= max_hops:
                continue
            for neighbor, eidx in adj.get(node, []):
                if neighbor in visited:
                    continue
                if not inst.alive[eidx]:
                    continue
                new_log_p = log_p + np.log(p[eidx] + 1e-12)
                if new_log_p < -10:  # probability too low
                    continue
                next_queue.append((neighbor, edge_path + [eidx], new_log_p, visited | {neighbor}))
        # Keep only the most promising branches
        next_queue.sort(key=lambda x: x[2], reverse=True)
        queue = next_queue[:max_paths * 3]
    return paths


def _edge_path_betweenness(inst, max_hops=3):
    """Score each candidate edge by how many target→labeled paths it participates in.
    An edge on many short paths to labels is a structural bottleneck worth querying."""
    A = inst.A_obs()
    _, comp = connected_components(A, directed=False)
    labeled_comps = set(comp[inst.train].tolist())
    tgt_unreached = [t for t in inst.targets if comp[t] not in labeled_comps]

    if not tgt_unreached:
        return np.full(len(inst.PI), -np.inf)

    labeled_comp_nodes = set(np.flatnonzero(np.isin(comp, list(labeled_comps))).tolist())
    adj = _build_edge_lookup(inst)
    p = inst.p_real()

    # Count path participation for each candidate edge
    edge_counts = np.zeros(len(inst.PI))
    edge_path_prob = np.zeros(len(inst.PI))

    for t in tgt_unreached[:30]:
        # BFS from target through candidate edges
        # (node, edge_indices_on_path, log_prob)
        queue = [(t, [], 0.0, {t})]
        for hop in range(max_hops):
            next_queue = []
            for node, epath, log_p, visited in queue:
                if node in labeled_comp_nodes and epath:
                    # Found a path — credit all edges on it
                    path_prob = np.exp(log_p)
                    for eidx in epath:
                        edge_counts[eidx] += 1
                        edge_path_prob[eidx] += path_prob
                    continue
                for neighbor, eidx in adj.get(node, []):
                    if neighbor in visited or not inst.alive[eidx]:
                        continue
                    new_lp = log_p + np.log(p[eidx] + 1e-12)
                    if new_lp < -8:
                        continue
                    next_queue.append((neighbor, epath + [eidx], new_lp, visited | {neighbor}))
            next_queue.sort(key=lambda x: x[2], reverse=True)
            queue = next_queue[:200]

    # Score = path_count × mean_path_prob × link_probability
    scores = edge_counts * p * inst.alive.astype(float)
    # Tie-break by individual link probability
    scores += 1e-9 * p * inst.alive.astype(float)
    scores[~inst.alive] = -np.inf
    return scores


_ADAPTIVE_STATE = {}

def path_joint_step(inst):
    """Feature-adaptive joint scorer: trains GCN once at step 0 to detect whether
    the model depends on structure. If initial GCN confidence on targets is low
    (features uninformative), allocate more budget to edges. If high, do labels.
    Uses LP-probe for edge selection (proven best hit rate)."""
    global _ADAPTIVE_STATE
    key = (inst.seed, id(inst))

    if key not in _ADAPTIVE_STATE:
        import torch
        A = inst.A_obs()
        W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k, seed=inst.seed, epochs=50)
        conf = _gcn_conf(W1, W2, Xt, A)
        tgt_conf = float(conf[inst.targets].mean())
        _, comp = connected_components(A, directed=False)
        labeled_comps = set(comp[inst.train].tolist())
        frac_reached = float(np.isin(comp[inst.targets], list(labeled_comps)).mean())
        random_conf = 1.0 / inst.k
        feature_signal = max(0.0, min(1.0, (tgt_conf - random_conf) / (0.8 - random_conf)))
        edge_frac = 0.5 * (1.0 - feature_signal) * max(0.3, 1.0 - frac_reached)
        _ADAPTIVE_STATE[key] = {"edge_frac": edge_frac, "tgt_conf": tgt_conf,
                                 "frac_reached": frac_reached}

    state = _ADAPTIVE_STATE[key]
    do_edge = inst.rng.random() < state["edge_frac"]

    if do_edge:
        edge_scores = _best_edge_value(inst)
        best_e = int(np.argmax(edge_scores))
        if edge_scores[best_e] > -np.inf:
            return "edge", best_e

    label_scores = score_label_value(inst)
    best_l = int(np.argmax(label_scores))
    return "label", best_l


STEP_FN = {
    "joint_value": lambda inst, step, budget: joint_value_step(inst),
    "joint_infogain": lambda inst, step, budget: joint_infogain_step(inst),
    "joint_gcn": lambda inst, step, budget: joint_gcn_step(inst),
    "joint_split": lambda inst, step, budget: joint_split_step(inst),
    "joint_emb": lambda inst, step, budget: _gcn_emb_joint_step(inst, tau=1.0),
    "joint_emb_t5": lambda inst, step, budget: _gcn_emb_joint_step(inst, tau=5.0),
    "joint_emb_t01": lambda inst, step, budget: _gcn_emb_joint_step(inst, tau=0.1),
    "joint_lookahead": lambda inst, step, budget: joint_lookahead_step(inst),
    "edge_only": lambda inst, step, budget: edge_only_step(inst),
    "label_only": lambda inst, step, budget: label_only_step(inst),
    "label_uncertainty": lambda inst, step, budget: label_uncertainty_step(inst),
    "label_random": lambda inst, step, budget: label_random_step(inst),
    "edge_pagerank": lambda inst, step, budget: edge_pagerank_step(inst),
    "edge_linkprob": lambda inst, step, budget: edge_linkprob_step(inst),
    "alternating": lambda inst, step, budget: alternating_step(inst, step),
    "random_joint": lambda inst, step, budget: random_joint_step(inst),
    "edge70_label30": lambda inst, step, budget: fixed_ratio_step(inst, step, budget, 0.7),
    "edge30_label70": lambda inst, step, budget: fixed_ratio_step(inst, step, budget, 0.3),
    "random": lambda inst, step, budget: random_edge_step(inst),
    # --- New baselines ---
    "alpine_joint": lambda inst, step, budget: alpine_joint_step(inst),
    "age_label": lambda inst, step, budget: age_label_step(inst),
    "oracle": lambda inst, step, budget: oracle_step(inst),
    # --- Ablations ---
    "abl_no_linkprob": lambda inst, step, budget: ablation_no_linkprob_step(inst),
    "abl_rand_shortlist": lambda inst, step, budget: ablation_rand_shortlist_step(inst),
    "abl_all_targets": lambda inst, step, budget: ablation_all_targets_step(inst),
    "abl_static": lambda inst, step, budget: ablation_static_step(inst, step, budget),
    # --- GNN-in-the-loop ---
    "gcn_probe": lambda inst, step, budget: joint_gcn_fast_step(inst),
    "reach_joint": lambda inst, step, budget: reach_joint_step(inst),
    "gcn_struct": lambda inst, step, budget: gcn_struct_step(inst),
    "path_joint": lambda inst, step, budget: path_joint_step(inst),
}


# ---------------------------------------------------------------- monotonicity & regime tracking
def _graph_state(inst):
    """Snapshot of graph state for regime analysis."""
    A = inst.A_obs()
    _, comp = connected_components(A, directed=False)
    tgt_reach = float(np.isin(comp[inst.targets], comp[inst.train]).mean())
    n_components = int(len(set(comp)))
    n_edges = len(inst.obs)
    F = inst.lp()
    q = beliefs(F)
    tgt_ent = -(q[inst.targets] * np.log(q[inst.targets] + 1e-12)).sum(axis=1)
    return {
        "tgt_reach": tgt_reach,
        "n_components": n_components,
        "n_edges": n_edges,
        "n_labeled": int(inst.lab.sum()),
        "mean_tgt_entropy": float(tgt_ent.mean()),
        "mean_tgt_conf": float(q[inst.targets].max(axis=1).mean()),
    }


# ---------------------------------------------------------------- run loop
def label_node(inst, v):
    inst.lab[v] = True
    inst.train = np.append(inst.train, v)


def joint_snapshot(inst, with_gcn, n_edge_q, n_label_q):
    s = snapshot(inst, with_gcn)
    s["n_edge_queries"] = n_edge_q
    s["n_label_queries"] = n_label_q
    s["n_labeled"] = int(inst.lab.sum())
    return s


def run_joint(seed, strategy, budget, checkpoints, gcn_checkpoints, n_targets, data, per_class=20):
    inst = Instance(seed, data, n_targets=n_targets, per_class=per_class)
    inst.X = _gcn_features(inst.X)
    step_fn = STEP_FN[strategy]
    t0 = time.perf_counter()
    n_edge_q, n_label_q, edges_found = 0, 0, 0
    rec = {0: joint_snapshot(inst, 0 in gcn_checkpoints, 0, 0)}
    action_log = []
    mono_violations = []
    regime_log = []

    prev_lp = rec[0]["lp"]

    for step in range(1, budget + 1):
        action_type, action_idx = step_fn(inst, step, budget)

        if action_type == "edge":
            hit = inst.reveal(action_idx)
            edges_found += int(hit)
            n_edge_q += 1
            action_log.append(("e", int(action_idx), bool(hit)))
        else:
            # log regime state when a label query is chosen (for joint strategies)
            if strategy.startswith("joint"):
                regime_log.append({"step": step, **_graph_state(inst)})
            label_node(inst, action_idx)
            n_label_q += 1
            action_log.append(("l", int(action_idx), True))

        if step in checkpoints:
            snap = joint_snapshot(inst, step in gcn_checkpoints, n_edge_q, n_label_q)
            snap["edges_found"] = edges_found
            rec[step] = snap
            cur_lp = snap["lp"]
            if cur_lp < prev_lp - 1e-9:
                mono_violations.append({
                    "step": step, "action": action_type,
                    "before": prev_lp, "after": cur_lp,
                    "delta": cur_lp - prev_lp,
                })
            prev_lp = cur_lp

    return {
        "rec": rec,
        "time": time.perf_counter() - t0,
        "pool_recall": inst.pool_recall,
        "pool_precision": inst.pool_precision,
        "link_auc": inst.link_auc,
        "n_edge_queries": n_edge_q,
        "n_label_queries": n_label_q,
        "edges_found": edges_found,
        "action_log": action_log,
        "mono_violations": mono_violations,
        "regime_log": regime_log,
    }


def ceilings_joint(seed, data, n_targets, per_class=20, model_type="gcn"):
    inst = Instance(seed, data, n_targets=n_targets, per_class=per_class)
    inst.X = _gcn_features(inst.X)
    A = data[0]
    lab_all = np.ones(inst.n, bool)
    return {
        "lp": lp_accuracy(inst, lp_solve(A, inst.y, inst.lab, inst.k)),
        "lp_full_labels": lp_accuracy(inst, lp_solve(A, inst.y, lab_all, inst.k)),
        "lp_full_graph": lp_accuracy(inst, lp_solve(data[0], inst.y, inst.lab, inst.k)),
        "lp_full_both": lp_accuracy(inst, lp_solve(data[0], inst.y, lab_all, inst.k)),
        "gcn": gcn_accuracy(A, inst.X, inst.y, inst.train, inst.targets, inst.k,
                            seed=seed, model_type=model_type),
    }


# ---------------------------------------------------------------- parallel harness
_DATA = None
_SUBGRAPH = 0
_DATASET = "cora"
_MODEL_TYPE = "gcn"


_GCN_FEATURES = "original"


def _gcn_features(X):
    n = X.shape[0]
    if _GCN_FEATURES == "original":
        return X
    elif _GCN_FEATURES == "onehot":
        return np.eye(n, dtype=np.float32)
    elif _GCN_FEATURES == "random16":
        rng = np.random.default_rng(999)
        return rng.standard_normal((n, 16)).astype(np.float32)
    elif _GCN_FEATURES.startswith("pca"):
        k = int(_GCN_FEATURES[3:])
        from sklearn.decomposition import PCA
        return PCA(n_components=k, random_state=42).fit_transform(X).astype(np.float32)
    return X


def _init_worker(subgraph=0, observe=OBSERVE_FRAC, link_model="logistic", dataset="cora",
                 model_type="gcn", subseed=42, features="original"):
    global _DATA, _SUBGRAPH, _DATASET, _MODEL_TYPE, _GCN_FEATURES
    import cora_exp
    _SUBGRAPH = subgraph
    _DATASET = dataset
    _MODEL_TYPE = model_type
    _GCN_FEATURES = features
    cora_exp.OBSERVE_FRAC = observe
    cora_exp.LINK_MODEL = link_model
    cora_exp.MODEL_TYPE = model_type
    if subgraph > 0:
        _DATA = load_cora_subgraph(n_nodes=subgraph, subseed=subseed, dataset=dataset)
    else:
        _DATA = DATASET_LOADERS[dataset]()


def job(args):
    global _DATA
    if _DATA is None:
        _init_worker(_SUBGRAPH)
    seed, strat, budget, cps, gcps, nt, pc = args
    if strat == "__ceiling__":
        return seed, strat, ceilings_joint(seed, _DATA, nt, pc, model_type=_MODEL_TYPE)
    return seed, strat, run_joint(seed, strat, budget, set(cps), set(gcps), nt, _DATA, pc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--targets", type=int, default=50)
    ap.add_argument("--per_class", type=int, default=0,
                    help="Labels per class (0 = auto: fit to subgraph size)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--strategies",
                    default="joint_value,joint_infogain,edge_only,label_only,label_uncertainty,"
                            "label_random,edge_pagerank,edge_linkprob,alternating,"
                            "edge70_label30,edge30_label70,random_joint,random")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "joint_pilot.json"))
    ap.add_argument("--observe", type=float, default=OBSERVE_FRAC)
    ap.add_argument("--subgraph", type=int, default=150)
    ap.add_argument("--subseed", type=int, default=42)
    ap.add_argument("--gcn_endpoints_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--link_model", default="logistic", choices=["logistic", "vgae"])
    ap.add_argument("--dataset", default="cora", choices=list(DATASET_LOADERS))
    ap.add_argument("--model_type", default="gcn", choices=["gcn", "gat", "sage"])
    ap.add_argument("--features", default="original")
    a = ap.parse_args()

    # auto per_class: ensure enough unlabeled nodes for targets + label queries,
    # and don't exceed the smallest class (minus 1 for target sampling headroom)
    if a.per_class == 0:
        if a.subgraph > 0:
            tmp_data = load_cora_subgraph(a.subgraph, a.subseed, a.dataset)
        else:
            tmp_data = DATASET_LOADERS[a.dataset]()
        y_tmp = tmp_data[2]
        k_tmp = int(y_tmp.max()) + 1
        n_tmp = tmp_data[0].shape[0]
        max_labeled = n_tmp - a.targets - 30
        _, counts = np.unique(y_tmp, return_counts=True)
        min_class = int(counts.min()) - 1  # leave 1 node per class for targets
        a.per_class = max(1, min(20, max_labeled // k_tmp, min_class))
        print(f"Auto per_class={a.per_class} (k={k_tmp}, n={n_tmp}, min_class={min_class+1})", flush=True)
        del tmp_data

    cps = sorted(set([0, 25, 50, 100, 150, 200]) & set(range(a.budget + 1)) | {a.budget})
    gcps = [0, a.budget] if a.gcn_endpoints_only else cps
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    strats = a.strategies.split(",")
    jobs_list = [(sd, s, a.budget, cps, gcps, a.targets, a.per_class)
                 for sd in seeds for s in strats]
    jobs_list += [(sd, "__ceiling__", 0, [], [], a.targets, a.per_class) for sd in seeds]

    out = {"args": vars(a), "runs": {}, "ceiling": {}}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker,
                             initargs=(a.subgraph, a.observe, a.link_model, a.dataset,
                                       a.model_type, a.subseed, a.features)) as ex:
        for sd, s, r in ex.map(job, jobs_list):
            if s == "__ceiling__":
                out["ceiling"][str(sd)] = r
            else:
                out["runs"].setdefault(s, {})[str(sd)] = r
            elapsed = time.time() - t0
            print(f"[{elapsed:6.0f}s] seed {sd} {s} done", flush=True)
            json.dump(out, open(a.out, "w"))

    print(f"\nDone. {len(seeds)} seeds x {len(strats)} strategies in {time.time() - t0:.0f}s")
    print(f"Results: {a.out}")
