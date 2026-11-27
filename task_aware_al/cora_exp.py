"""Real-data experiment: task-aware active link acquisition on Cora.

Same task as the synthetic study: edges are mostly hidden, a few nodes are labelled, a TARGET set
must be classified, and a budget of B link queries is spent on an oracle that answers whether a
pair is an edge. What differs from the synthetic setup:
  * 2,708 nodes / 3.7M pairs, so candidates come from a POOL (feature kNN + 2-hop + target-focused
    kNN) built only from features and the observed graph.
  * P(edge) is a logistic model fitted on the initial observed graph (features + structure), not
    the p_in/p_out formula.
  * the scorer is label propagation (sparse, solved per affected component), but the EVALUATOR is
    a GCN that sees node features, so scorer and evaluator are different models. LP accuracy is
    reported too, for comparability with the synthetic results.

The scorer may use: observed edges, labels of labelled nodes, node features, the target set.
It may not use: the true graph, non-labelled nodes' labels.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn.functional as Ft
from torch_geometric.nn import GCNConv, VGAE as PyGVGAE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "planetoid")
ALPHA = 0.99
OBSERVE_FRAC = 0.15
LINK_MODEL = "logistic"  # "logistic" or "vgae"
MODEL_TYPE = "gcn"  # "gcn", "gat", or "sage"


# ---------------------------------------------------------------- VGAE link model
class _VGAEEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden=256, latent=128, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv_mu = GCNConv(hidden, latent)
        self.conv_logvar = GCNConv(hidden, latent)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = Ft.dropout(Ft.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)


def _sp_to_edge_index(A):
    """Sparse scipy matrix → PyG edge_index tensor (2, E)."""
    coo = sp.triu(A, 1).tocoo()
    row = np.concatenate([coo.row, coo.col])
    col = np.concatenate([coo.col, coo.row])
    return torch.from_numpy(np.stack([row, col])).long()


def train_vgae(A_obs, X, seed=0, epochs=200, lr=0.005, hidden=256, latent=128, dropout=0.4):
    """Train a VGAE on the observed graph. Returns the model (eval mode) and node embeddings."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n = A_obs.shape[0]
    edge_index = _sp_to_edge_index(A_obs)
    x = torch.from_numpy(X).float()

    encoder = _VGAEEncoder(X.shape[1], hidden, latent, dropout)
    model = PyGVGAE(encoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    num_edges = edge_index.shape[1] // 2
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(x, edge_index)
        neg_ei = torch.randint(0, n, (2, num_edges))
        loss = model.recon_loss(z, edge_index, neg_edge_index=neg_ei)
        loss = loss + (1.0 / n) * model.kl_loss()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return model, z.numpy()


# ---------------------------------------------------------------- data
def _load_planetoid(name):
    from torch_geometric.datasets import Planetoid
    d = Planetoid(DATA_DIR, name)[0]
    n = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    X = d.x.numpy().astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return A, X, d.y.numpy()


def load_cora():
    return _load_planetoid("Cora")


def load_citeseer():
    return _load_planetoid("CiteSeer")


def load_amazon_computers():
    from torch_geometric.datasets import Amazon
    d = Amazon(os.path.join(HERE, "..", "data", "amazon"), "Computers")[0]
    n = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    X = d.x.numpy().astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return A, X, d.y.numpy()


def load_coauthor_cs():
    from torch_geometric.datasets import Coauthor
    d = Coauthor(os.path.join(HERE, "..", "data", "coauthor"), "CS")[0]
    n = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    X = d.x.numpy().astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return A, X, d.y.numpy()


def load_actor():
    from torch_geometric.datasets import Actor
    d = Actor(os.path.join(HERE, "..", "data", "actor"))[0]
    n = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    X = d.x.numpy().astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return A, X, d.y.numpy()


def load_cora_subgraph(n_nodes=400, subseed=42, dataset="cora"):
    """Extract an n_nodes subgraph from Cora via BFS from a random start node.
    This gives a connected, real-world subgraph where 200 queries are a meaningful
    fraction of the edge set, matching the synthetic regime."""
    A_full, X_full, y_full = DATASET_LOADERS[dataset]()
    rng = np.random.default_rng(subseed)
    n_full = A_full.shape[0]
    # BFS from a random node, stop at n_nodes
    start = rng.integers(n_full)
    visited = set()
    frontier = [start]
    while len(visited) < n_nodes and frontier:
        rng.shuffle(frontier)
        nxt = []
        for v in frontier:
            if len(visited) >= n_nodes:
                break
            if v in visited:
                continue
            visited.add(v)
            nbs = A_full[v].nonzero()[1]
            nxt.extend(nbs)
        frontier = nxt
    nodes = np.sort(list(visited))[:n_nodes]
    A = A_full[nodes][:, nodes].tocsr()
    X = X_full[nodes]
    y = y_full[nodes]
    return A, X, y


# ---------------------------------------------------------------- label propagation (sparse)
def lp_solve(A, y, lab, k, alpha=ALPHA):
    """Damped LP, labelled rows clamped: F_U = (I - a S_UU)^-1 a S_UL Y_L. Nodes no label reaches get 0."""
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    S = sp.diags(np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)) @ A
    U = np.flatnonzero(~lab)
    L = np.flatnonzero(lab)
    F = np.zeros((n, k))
    F[L, y[L]] = 1.0
    if len(U) and len(L):
        M = (sp.identity(len(U), format="csc") - alpha * S[U][:, U]).tocsc()
        rhs = alpha * (S[U][:, L] @ F[L])
        F[U] = splu(M).solve(np.asarray(rhs))
    return F


def beliefs(F):
    m = F.sum(axis=1, keepdims=True)
    return np.divide(F, m, out=np.full_like(F, 1.0 / F.shape[1]), where=m > 1e-12)


# ---------------------------------------------------------------- GNN evaluators
def _sp_to_pyg_edge_index(A):
    """Sparse scipy → (edge_index, num_nodes) for PyG models."""
    coo = A.tocoo()
    return torch.from_numpy(np.vstack([coo.row, coo.col])).long(), A.shape[0]


class _GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=8, heads=8, out_dim=7, dropout=0.6):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = Ft.dropout(x, p=self.dropout, training=self.training)
        x = Ft.elu(self.conv1(x, edge_index))
        x = Ft.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class _SAGEModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=7, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = Ft.dropout(x, p=self.dropout, training=self.training)
        x = Ft.relu(self.conv1(x, edge_index))
        x = Ft.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def gcn_accuracy(A, X, y, train_idx, eval_idx, k, seed=0, epochs=150, model_type="gcn"):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n = A.shape[0]
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    tr = torch.from_numpy(np.asarray(train_idx)).long()

    if model_type == "gcn":
        Ah = (A + sp.identity(n)).tocoo()
        deg = np.asarray(Ah.sum(axis=1)).ravel()
        w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
        Ahat = torch.sparse_coo_tensor(np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
        W1 = torch.nn.Linear(X.shape[1], 64)
        W2 = torch.nn.Linear(64, k)
        params = list(W1.parameters()) + list(W2.parameters())

        def fwd(train):
            h = torch.dropout(Xt, 0.5, train)
            h = torch.relu(torch.sparse.mm(Ahat, W1(h)))
            h = torch.dropout(h, 0.5, train)
            return torch.sparse.mm(Ahat, W2(h))
    elif model_type == "gat":
        ei, _ = _sp_to_pyg_edge_index(A + sp.identity(n))
        model = _GATModel(X.shape[1], out_dim=k)
        params = model.parameters()

        def fwd(train):
            model.train(train)
            return model(Xt, ei)
    elif model_type == "sage":
        ei, _ = _sp_to_pyg_edge_index(A + sp.identity(n))
        model = _SAGEModel(X.shape[1], out_dim=k)
        params = model.parameters()

        def fwd(train):
            model.train(train)
            return model(Xt, ei)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    opt = torch.optim.Adam(params, lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(fwd(True)[tr], yt[tr]).backward()
        opt.step()
    with torch.no_grad():
        pred = fwd(False).argmax(1).numpy()
    return float((pred[eval_idx] == y[eval_idx]).mean())


# ---------------------------------------------------------------- instance
def _key(i, j, n):
    return np.minimum(i, j).astype(np.int64) * n + np.maximum(i, j)


class Instance:
    def __init__(self, seed, data, n_targets=50, per_class=20, knn=30, tgt_knn=100):
        A_true, X, y = data
        self.seed, self.X, self.y, self.A_true = seed, X, y, A_true
        self.n, self.k = A_true.shape[0], int(y.max()) + 1
        n = self.n
        self.rng = np.random.default_rng(seed)
        rng = np.random.default_rng(1000 + seed)

        self.train = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                                min(per_class, (y == c).sum() - 1), replace=False)
                                     for c in range(self.k) if (y == c).sum() > 1])
        self.lab = np.zeros(n, bool)
        self.lab[self.train] = True
        self.targets = np.sort(rng.choice(np.flatnonzero(~self.lab), n_targets, replace=False))
        self.not_target = np.setdiff1d(np.flatnonzero(~self.lab), self.targets)

        tu = sp.triu(A_true, 1).tocoo()
        self.true_edges = np.stack([tu.row, tu.col], 1)
        pick = rng.choice(len(self.true_edges), int(OBSERVE_FRAC * len(self.true_edges)), replace=False)
        self.obs = [tuple(e) for e in self.true_edges[pick]]
        self._A = None

        # ---- candidate pool (features + observed graph only)
        sim = X @ X.T
        np.fill_diagonal(sim, -np.inf)
        nb = np.argpartition(-sim, knn, axis=1)[:, :knn]
        a = np.repeat(np.arange(n), knn)
        ks = [_key(a, nb.ravel(), n)]
        knn_keys = np.unique(ks[0])
        for t in self.targets:
            top = np.argpartition(-sim[t], tgt_knn)[:tgt_knn]
            ks.append(_key(np.full(tgt_knn, t), top, n))
        A0 = self.A_obs()
        A2 = sp.triu(A0 @ A0, 1).tocoo()
        ks.append(_key(A2.row, A2.col, n))
        keys = np.unique(np.concatenate(ks))
        obs_keys = _key(np.array([e[0] for e in self.obs]), np.array([e[1] for e in self.obs]), n)
        keys = np.setdiff1d(keys, obs_keys)
        self.PI, self.PJ = keys // n, keys % n
        self.cos = np.asarray(sim[self.PI, self.PJ]).ravel()
        self.is_true = np.asarray(A_true[self.PI, self.PJ]).ravel() > 0
        self.alive = np.ones(len(keys), bool)
        hidden = len(self.true_edges) - len(self.obs)
        self.pool_recall = float(self.is_true.sum() / hidden)
        self.pool_precision = float(self.is_true.mean())

        # ---- link-probability model
        self._link_model_type = LINK_MODEL
        fc = LogisticRegression(C=10.0, max_iter=300).fit(X[self.train], y[self.train])
        self.qf = fc.predict_proba(X)

        if LINK_MODEL == "vgae":
            _, self._vgae_z = train_vgae(A0, X, seed=seed)
            self.linkmodel = None
            self._mu = self._sd = None
        else:
            pos_mask = np.isin(obs_keys, knn_keys)
            pi, pj = obs_keys[pos_mask] // n, obs_keys[pos_mask] % n
            in_knn = np.isin(keys, knn_keys)
            Fpos = self._pair_feats(pi, pj, A0, sim)
            Fneg = self._pair_feats(self.PI[in_knn], self.PJ[in_knn], A0, sim)
            Xtr = np.vstack([Fpos, Fneg])
            self._mu, self._sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            ytr = np.r_[np.ones(len(Fpos)), np.zeros(len(Fneg))]
            self.linkmodel = LogisticRegression(C=1.0, max_iter=500).fit((Xtr - self._mu) / self._sd, ytr)
            self._vgae_z = None
        # ---- link model quality metrics
        p_scores = self.p_real()
        from sklearn.metrics import roc_auc_score
        self.link_auc = float(roc_auc_score(self.is_true, p_scores))
        self.sim_cos = None
        self._init_acc = None

    # -- observed graph
    def A_obs(self):
        if self._A is None:
            e = np.array(self.obs)
            M = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(self.n, self.n))
            self._A = (M + M.T).tocsr()
        return self._A

    def _pair_feats(self, I, J, A, sim=None):
        cn = np.asarray(A[I].multiply(A[J]).sum(axis=1)).ravel()
        deg = np.asarray(A.sum(axis=1)).ravel()
        cos = self.cos if sim is None else np.asarray(sim[I, J]).ravel()
        agree = (self.qf[I] * self.qf[J]).sum(axis=1)
        return np.c_[cos, agree, np.log1p(cn), np.log1p(deg[I]) + np.log1p(deg[J])]

    def p_real(self, idx=None):
        """P(pair is an edge) from the fitted model, on the CURRENT observed graph."""
        I, J = (self.PI, self.PJ) if idx is None else (self.PI[idx], self.PJ[idx])
        if self._link_model_type == "vgae":
            z = self._vgae_z
            logits = (z[I] * z[J]).sum(axis=1)
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -15, 15)))
        A = self.A_obs()
        cn = np.asarray(A[I].multiply(A[J]).sum(axis=1)).ravel()
        deg = np.asarray(A.sum(axis=1)).ravel()
        cos = self.cos if idx is None else self.cos[idx]
        agree = (self.qf[I] * self.qf[J]).sum(axis=1)
        f = np.c_[cos, agree, np.log1p(cn), np.log1p(deg[I]) + np.log1p(deg[J])]
        return self.linkmodel.predict_proba((f - self._mu) / self._sd)[:, 1]

    def reveal(self, idx):
        self.alive[idx] = False
        if self.is_true[idx]:
            self.obs.append((int(self.PI[idx]), int(self.PJ[idx])))
            self._A = None
            return True
        return False

    def lp(self, A=None):
        return lp_solve(self.A_obs() if A is None else A, self.y, self.lab, self.k)


# ---------------------------------------------------------------- scorers (over the full pool)
def _noise(inst, s):
    return s + 1e-9 * inst.rng.random(len(s))


def sc_random(inst):
    return inst.rng.random(len(inst.PI))


def sc_cn(inst):
    A = inst.A_obs()
    return _noise(inst, np.asarray(A[inst.PI].multiply(A[inst.PJ]).sum(axis=1)).ravel())


def sc_aa(inst):
    A = inst.A_obs()
    d = np.asarray(A.sum(axis=1)).ravel()
    w = np.where(d > 1, 1.0 / np.log(np.maximum(d, 2)), 0.0)
    return _noise(inst, np.asarray((A[inst.PI] @ sp.diags(w)).multiply(A[inst.PJ]).sum(axis=1)).ravel())


def sc_maxdeg(inst):
    d = np.asarray(inst.A_obs().sum(axis=1)).ravel()
    return _noise(inst, d[inst.PI] + d[inst.PJ])


def sc_pagerank(inst, damping=0.85, iters=60):
    A = inst.A_obs()
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    P = sp.diags(np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)) @ A
    pr = np.full(n, 1.0 / n)
    dang = d == 0
    for _ in range(iters):
        pr = damping * (P.T @ pr + pr[dang].sum() / n) + (1 - damping) / n
    return _noise(inst, pr[inst.PI] + pr[inst.PJ])


def sc_uncertainty(inst):
    q = beliefs(inst.lp())
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    return _noise(inst, ent[inst.PI] + ent[inst.PJ])


def sc_linkprob(inst):
    return _noise(inst, inst.p_real())


SHORTLIST = 60


def _train_gcn(A, X, y, train_idx, k, seed=0, epochs=100):
    """Train a GCN and return the frozen model components for fast re-evaluation."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n = A.shape[0]
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    W1 = torch.nn.Linear(X.shape[1], 64)
    W2 = torch.nn.Linear(64, k)
    tr = torch.from_numpy(np.asarray(train_idx)).long()

    Ah = (A + sp.identity(n)).tocoo()
    deg = np.asarray(Ah.sum(axis=1)).ravel()
    w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
    Ahat = torch.sparse_coo_tensor(np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
    opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        opt.zero_grad()
        h = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
        torch.nn.functional.cross_entropy(torch.sparse.mm(Ahat, W2(h))[tr], yt[tr]).backward()
        opt.step()
    W1.eval()
    W2.eval()
    return W1, W2, Xt


def _gcn_conf(W1, W2, Xt, A):
    """Forward pass with frozen weights on a (possibly modified) adjacency. Returns per-node max confidence."""
    n = A.shape[0]
    Ah = (A + sp.identity(n)).tocoo()
    deg = np.asarray(Ah.sum(axis=1)).ravel()
    w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
    Ahat = torch.sparse_coo_tensor(np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
    with torch.no_grad():
        h = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
        logits = torch.sparse.mm(Ahat, W2(h))
        probs = torch.softmax(logits, dim=1)
    return probs.max(dim=1).values.numpy()


def sc_value_gcn(inst, objective="targets"):
    """Value scorer using GCN forward passes instead of LP re-solves."""
    A = inst.A_obs()
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    obj_nodes = inst.targets if objective == "targets" else np.concatenate([inst.targets, inst.not_target])
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[obj_nodes] = True
    if objective == "targets":
        relevant = np.isin(comp, comp[obj_nodes])
        rel = relevant[I] | relevant[J]
    else:
        rel = np.ones(len(I), bool)

    W1, W2, Xt = _train_gcn(A, inst.X, inst.y, inst.train, inst.k, seed=inst.seed)
    conf_base = _gcn_conf(W1, W2, Xt, A)

    key = p * np.abs(conf_base[I] - conf_base[J]) * rel * alive
    scores = 1e-9 * p * rel
    scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(obj_nodes)
    for c in order:
        i, j = I[c], J[c]
        E = sp.csr_matrix(([1.0, 1.0, 1.0, 1.0], ([i, j, j, i], [j, i, i, j])), shape=A.shape)
        A_mod = A + E
        A_mod.data = np.minimum(A_mod.data, 1.0)
        conf2 = _gcn_conf(W1, W2, Xt, A_mod)
        scores[c] += p[c] * (conf2[obj_nodes] - conf_base[obj_nodes]).sum() / n_obj
    return scores


def sc_value(inst, objective="targets"):
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    conf = q.max(axis=1)
    I, J = inst.PI, inst.PJ
    alive = inst.alive
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    obj_nodes = inst.targets if objective == "targets" else np.concatenate([inst.targets, inst.not_target])
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[obj_nodes] = True
    if objective == "targets":
        relevant = np.isin(comp, comp[obj_nodes])
        rel = relevant[I] | relevant[J]
    else:
        rel = np.ones(len(I), bool)
    key = p * np.abs(conf[I] - conf[J]) * rel * alive
    scores = 1e-9 * p * rel
    scores[~alive] = -np.inf
    order = np.argsort(-key)[:SHORTLIST]
    order = order[key[order] > 0]
    n_obj = len(obj_nodes)
    for c in order:
        i, j = I[c], J[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_obj = obj_mask[nodes]
        if not sub_obj.any():
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_obj]
        scores[c] += p[c] * (conf2 - conf[nodes][sub_obj]).sum() / n_obj
    return scores


# ---- Phase 2 baselines ----

def sc_vgae_topk(inst):
    """2a: Query highest link probability, ignoring targets entirely."""
    return _noise(inst, inst.p_real())


def sc_uncertainty_edge(inst):
    """2b: Score each candidate edge by the classifier uncertainty on adjacent target nodes."""
    q = beliefs(inst.lp())
    ent = -(q * np.log(q + 1e-12)).sum(axis=1)
    tgt_set = set(inst.targets)
    I, J = inst.PI, inst.PJ
    scores = np.zeros(len(I))
    for idx in range(len(I)):
        if not inst.alive[idx]:
            scores[idx] = -np.inf
            continue
        s = 0.0
        if I[idx] in tgt_set:
            s += ent[I[idx]]
        if J[idx] in tgt_set:
            s += ent[J[idx]]
        scores[idx] = s
    return _noise(inst, scores)


def sc_qbc(inst, n_models=5):
    """2c: Query by Committee — train multiple GCNs with different seeds, score by vote entropy reduction on targets."""
    A = inst.A_obs()
    preds = []
    for m in range(n_models):
        torch.manual_seed(inst.seed * 100 + m)
        torch.set_num_threads(1)
        n = A.shape[0]
        Xt = torch.from_numpy(inst.X).float()
        yt = torch.from_numpy(inst.y).long()
        Ah = (A + sp.identity(n)).tocoo()
        deg = np.asarray(Ah.sum(axis=1)).ravel()
        w = Ah.data / np.sqrt(deg[Ah.row] * deg[Ah.col])
        Ahat = torch.sparse_coo_tensor(np.vstack([Ah.row, Ah.col]), w.astype(np.float32), (n, n)).coalesce()
        W1 = torch.nn.Linear(inst.X.shape[1], 64)
        W2 = torch.nn.Linear(64, inst.k)
        opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01, weight_decay=5e-4)
        tr = torch.from_numpy(np.asarray(inst.train)).long()
        for _ in range(80):
            opt.zero_grad()
            h = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
            torch.nn.functional.cross_entropy(torch.sparse.mm(Ahat, W2(h))[tr], yt[tr]).backward()
            opt.step()
        with torch.no_grad():
            h = torch.relu(torch.sparse.mm(Ahat, W1(Xt)))
            pred = torch.sparse.mm(Ahat, W2(h)).argmax(1).numpy()
        preds.append(pred)
    preds = np.stack(preds)  # (n_models, n_nodes)
    vote_counts = np.zeros((inst.n, inst.k))
    for p in preds:
        vote_counts[np.arange(inst.n), p] += 1
    vote_probs = vote_counts / n_models
    vote_ent = -(vote_probs * np.log(vote_probs + 1e-12)).sum(axis=1)
    tgt_ent = vote_ent[inst.targets]
    I, J = inst.PI, inst.PJ
    _, comp = connected_components(A, directed=False)
    tgt_comps = set(comp[inst.targets])
    relevant = np.array([comp[I[i]] in tgt_comps or comp[J[i]] in tgt_comps for i in range(len(I))])
    p = inst.p_real()
    scores = p * relevant * inst.alive
    tgt_set = set(inst.targets)
    for idx in np.argsort(-scores)[:SHORTLIST]:
        if scores[idx] <= 0:
            break
        s = 0.0
        if I[idx] in tgt_set:
            s += vote_ent[I[idx]]
        if J[idx] in tgt_set:
            s += vote_ent[J[idx]]
        nbrs_i = set(A[I[idx]].nonzero()[1])
        nbrs_j = set(A[J[idx]].nonzero()[1])
        for t in inst.targets:
            if t in nbrs_i or t in nbrs_j:
                s += vote_ent[t] * 0.5
        scores[idx] += p[idx] * s
    scores[~inst.alive] = -np.inf
    return scores


def sc_infogain(inst):
    """2d: Information gain — score each candidate by expected entropy reduction on target beliefs."""
    A = inst.A_obs()
    F = inst.lp()
    q = beliefs(F)
    tgt_ent = -(q[inst.targets] * np.log(q[inst.targets] + 1e-12)).sum(axis=1)
    H_before = tgt_ent.sum()
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
        ent2 = -(q2[sub_tgt] * np.log(q2[sub_tgt] + 1e-12)).sum(axis=1)
        ent_before = -(q[nodes][sub_tgt] * np.log(q[nodes][sub_tgt] + 1e-12)).sum(axis=1)
        ig = (ent_before - ent2).sum()
        scores[c] += p[c] * ig
    return scores


def sc_oracle(inst):
    """2e: Oracle greedy — sees true edges, picks the one with max actual LP accuracy gain on targets.
    Only evaluates true edges in target-reachable components for speed."""
    A = inst.A_obs()
    F = inst.lp()
    I, J = inst.PI, inst.PJ
    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[inst.targets] = True
    tgt_comps = set(comp[inst.targets])
    scores = np.full(len(I), -np.inf)
    candidates = np.flatnonzero(inst.alive & inst.is_true)
    conf_base = beliefs(F)
    for idx in candidates:
        i, j = I[idx], J[idx]
        if comp[i] not in tgt_comps and comp[j] not in tgt_comps:
            scores[idx] = 1e-9
            continue
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            scores[idx] = 1e-9
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        conf2 = beliefs(F2).max(axis=1)[sub_tgt]
        conf1 = conf_base[nodes][sub_tgt].max(axis=1)
        scores[idx] = (conf2 - conf1).sum()
    return scores


# ---------------------------------------------------------------- failure diagnostic
def diagnose_failures(inst, entropy_threshold=0.5):
    """Classify each target node's failure mode.

    Returns dict mapping target node → one of:
      "structural"    — no path to any labeled node (unreachable)
      "informational" — reachable but neighbourhood labels disagree (high entropy)
      "intrinsic"     — reachable, neighbourhood agrees, but prediction still wrong
                        OR reachable and correctly classified (no failure)

    Also returns counts: {"structural": int, "informational": int, "intrinsic": int}
    """
    A = inst.A_obs()
    _, comp = connected_components(A, directed=False)
    lab_comps = set(comp[inst.train])

    F = inst.lp()
    B = beliefs(F)
    pred = F.argmax(axis=1)

    diag = {}
    counts = {"structural": 0, "informational": 0, "intrinsic": 0}

    for t in inst.targets:
        if comp[t] not in lab_comps:
            diag[int(t)] = "structural"
            counts["structural"] += 1
            continue
        ent = -np.sum(B[t] * np.log(np.clip(B[t], 1e-12, 1.0)))
        max_ent = np.log(inst.k)
        if ent / max_ent > entropy_threshold:
            diag[int(t)] = "informational"
            counts["informational"] += 1
        else:
            diag[int(t)] = "intrinsic"
            counts["intrinsic"] += 1

    return diag, counts


def sc_value_diag(inst, scope="targets", entropy_threshold=0.5):
    """Diagnostic-guided value scorer: skip intrinsic failures, focus budget on
    structural and informational targets only."""
    diag, _ = diagnose_failures(inst, entropy_threshold)
    actionable = np.array([t for t in inst.targets if diag[int(t)] != "intrinsic"])
    if len(actionable) == 0:
        return sc_value(inst, scope)

    A = inst.A_obs()
    F = inst.lp()
    p = inst.p_real()
    _, comp = connected_components(A, directed=False)
    obj_mask = np.zeros(inst.n, bool)
    obj_mask[actionable] = True

    scores = np.full(len(inst.PI), -np.inf)
    for c in inst.rng.choice(len(inst.PI), min(500, inst.alive.sum()), replace=False):
        if not inst.alive[c]:
            continue
        i, j = inst.PI[c], inst.PJ[c]
        nodes = np.flatnonzero((comp == comp[i]) | (comp == comp[j]))
        sub_tgt = obj_mask[nodes]
        if not sub_tgt.any():
            scores[c] = 1e-9 * p[c]
            continue
        Asub = A[nodes][:, nodes]
        li, lj = np.searchsorted(nodes, [i, j])
        E = sp.csr_matrix(([1.0, 1.0], ([li, lj], [lj, li])), shape=Asub.shape)
        F2 = lp_solve(Asub + E, inst.y[nodes], inst.lab[nodes], inst.k)
        b2 = beliefs(F2)
        b1 = beliefs(F)[nodes]
        gain = (b2.max(axis=1) - b1.max(axis=1))[sub_tgt].sum()
        scores[c] = p[c] * max(gain, 1e-9)
    return scores


SCORERS = {
    "random": sc_random, "common_nbrs": sc_cn, "adamic_adar": sc_aa, "maxdeg": sc_maxdeg,
    "pagerank": sc_pagerank, "uncertainty": sc_uncertainty, "linkprob": sc_linkprob,
    "value": lambda inst: sc_value(inst, "targets"),
    "value_all": lambda inst: sc_value(inst, "all"),
    "value_gcn": lambda inst: sc_value_gcn(inst, "targets"),
    "value_gcn_all": lambda inst: sc_value_gcn(inst, "all"),
    "vgae_topk": sc_vgae_topk,
    "uncertainty_edge": sc_uncertainty_edge,
    "qbc": sc_qbc,
    "infogain": sc_infogain,
    "oracle": sc_oracle,
    "value_diag": sc_value_diag,
}


# ---------------------------------------------------------------- one run
def lp_accuracy(inst, F):
    unreached = np.abs(F).sum(axis=1) < 1e-12
    fb = np.random.default_rng(0).integers(0, inst.k, size=inst.n)
    pred = np.where(unreached, fb, F.argmax(axis=1))
    return float((pred[inst.targets] == inst.y[inst.targets]).mean())


def snapshot(inst, with_gcn):
    A = inst.A_obs()
    F = inst.lp()
    _, comp = connected_components(A, directed=False)
    r = {"lp": lp_accuracy(inst, F),
         "reach": float(np.isin(comp[inst.targets], comp[inst.train]).mean()),
         "edges": len(inst.obs)}
    _, fc = diagnose_failures(inst)
    r["failures"] = fc
    if with_gcn:
        r["gcn"] = gcn_accuracy(A, inst.X, inst.y, inst.train, inst.targets, inst.k,
                                seed=inst.seed, model_type=MODEL_TYPE)
    return r


def run(seed, strategy, budget, checkpoints, gcn_checkpoints, n_targets, data, per_class=20):
    inst = Instance(seed, data, n_targets=n_targets, per_class=per_class)
    t0 = time.perf_counter()
    rec = {0: snapshot(inst, 0 in gcn_checkpoints)}
    found = 0
    for step in range(1, budget + 1):
        s = SCORERS[strategy](inst)
        s = np.where(inst.alive, s, -np.inf)
        found += inst.reveal(int(np.argmax(s)))
        if step in checkpoints:
            rec[step] = snapshot(inst, step in gcn_checkpoints)
            rec[step]["found"] = found
    return {"rec": rec, "time": time.perf_counter() - t0, "pool_recall": inst.pool_recall,
            "pool_precision": inst.pool_precision, "link_auc": inst.link_auc}


def ceilings(seed, data, n_targets, per_class=20, model_type="gcn"):
    inst = Instance(seed, data, n_targets=n_targets, per_class=per_class)
    A = data[0]
    return {"lp": lp_accuracy(inst, lp_solve(A, inst.y, inst.lab, inst.k)),
            "gcn": gcn_accuracy(A, inst.X, inst.y, inst.train, inst.targets, inst.k,
                                seed=seed, model_type=model_type)}


DATASET_LOADERS = {
    "cora": load_cora,
    "citeseer": load_citeseer,
    "amazon": load_amazon_computers,
    "coauthor_cs": load_coauthor_cs,
    "actor": load_actor,
}

_DATA = None
_SUBGRAPH = 0
_DATASET = "cora"


def _init_worker(subgraph=0, observe=OBSERVE_FRAC, link_model="logistic", dataset="cora", model_type="gcn", subseed=42):
    global _DATA, _SUBGRAPH, OBSERVE_FRAC, LINK_MODEL, _DATASET, MODEL_TYPE
    _SUBGRAPH = subgraph
    OBSERVE_FRAC = observe
    LINK_MODEL = link_model
    _DATASET = dataset
    MODEL_TYPE = model_type
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
        return seed, strat, ceilings(seed, _DATA, nt, pc, model_type=MODEL_TYPE)
    return seed, strat, run(seed, strat, budget, set(cps), set(gcps), nt, _DATA, pc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--targets", type=int, default=50)
    ap.add_argument("--per_class", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strategies", default="random,common_nbrs,maxdeg,pagerank,uncertainty,linkprob,value,value_all")
    ap.add_argument("--out", default=os.path.join(HERE, "cora_run.json"))
    ap.add_argument("--observe", type=float, default=OBSERVE_FRAC)
    ap.add_argument("--subgraph", type=int, default=0,
                    help="If >0, extract a BFS subgraph of this many nodes from Cora")
    ap.add_argument("--subseed", type=int, default=42, help="BFS start seed for the subgraph")
    ap.add_argument("--gcn_endpoints_only", action="store_true", default=True,
                    help="GCN eval only at step 0 and final step (fast)")
    ap.add_argument("--link_model", default="logistic", choices=["logistic", "vgae"],
                    help="Link probability model: logistic (hand-crafted features) or vgae (GCN autoencoder)")
    ap.add_argument("--dataset", default="cora", choices=list(DATASET_LOADERS),
                    help="Dataset to use")
    ap.add_argument("--model_type", default="gcn", choices=["gcn", "gat", "sage"],
                    help="GNN architecture for evaluation")
    a = ap.parse_args()
    cps = sorted(set([0, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]) & set(range(a.budget + 1)) | {a.budget})
    if a.gcn_endpoints_only:
        gcps = [0, a.budget]
    else:
        gcps = [c for c in (0, 50, 100, 200, 300, 400, 500) if c <= a.budget] + ([a.budget] if a.budget not in (0, 50, 100, 200, 300, 400, 500) else [])
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    jobs = [(sd, s, a.budget, cps, gcps, a.targets, a.per_class) for sd in seeds for s in a.strategies.split(",")]
    jobs += [(sd, "__ceiling__", 0, [], [], a.targets, a.per_class) for sd in seeds]
    out = {"args": vars(a), "runs": {}, "ceiling": {}}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(a.subgraph, a.observe, a.link_model, a.dataset, a.model_type, a.subseed)) as ex:
        for sd, s, r in ex.map(job, jobs):
            if s == "__ceiling__":
                out["ceiling"][str(sd)] = r
            else:
                out["runs"].setdefault(s, {})[str(sd)] = r
            print(f"[{time.time() - t0:6.0f}s] seed {sd} {s} done", flush=True)
            json.dump(out, open(a.out, "w"))
