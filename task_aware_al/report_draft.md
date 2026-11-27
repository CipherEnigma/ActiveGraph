# Mid-term report draft: content blocks

Reword everything in your own voice. Numbers are from the runs in `task_aware_al/`; where two
runs exist, the unseen-seed run (seeds 100-129) is the confirmatory one.
[V] = describe this paper only as far as you have read it; verify before leaning on a claim.

---

## Title (suggestion)

Task-Aware Active Link Acquisitivon for Node Classification on Partially Observed Graphs

## Abstract (about 150 words)

Real networks are rarely fully observed, and confirming whether two nodes are linked is
costly. We study how to spend a fixed query budget on hidden links so that a chosen set of
target nodes is classified correctly. We compare likely-link heuristics (common neighbours,
Adamic-Adar), ALPINE-style heuristics (max-degree, PageRank), uncertainty sampling, and a
task-aware value scorer that ranks each candidate link by the expected improvement in the
classifier's confidence on the target nodes. On synthetic community graphs with label
propagation as the classifier, heuristics that find more real links do not improve target
accuracy, whereas the value scorer closes about 45-53% of the gap to full-graph accuracy in
200 queries, against at most 13% for the heuristics (30 unseen seeds, paired tests).
Choosing queries adaptively adds a smaller, later gain. The result holds when edges are
hidden around the targets. We discuss limits and planned work on real data and a GCN.

---

## 1. Introduction

**Context.** Networks used for classification (social, citation, biological) are often only
partly observed; verifying a link needs an expert, an experiment or an API call.
**Problem.** Given a partly observed graph, a few labelled nodes, a set of target nodes to
classify and a budget of B link queries, which pairs should be queried?
**Why it is not ordinary link prediction.** Link prediction asks which links probably exist.
A link can exist and still be useless for the target nodes (for example, it joins nodes that
are already classified, or nodes no label can reach). Our observation, from the first phase
of this project: the strategy that finds the most real links gave the worst classification.
**Aim.** Show the effect of active learning (sequential, feedback-driven selection) and of
task awareness (selection aimed at the target nodes) on link acquisition.
**Contributions so far** (state only what is measured):
1. A controlled benchmark with hidden edges, a target set, a query budget and paired
   evaluation.
2. Evidence that link-finding rate and target accuracy diverge.
3. A task-aware value scorer, and a static vs adaptive and task-aware vs task-agnostic
   comparison against standard baselines.
4. Robustness checks: cold-start targets, hub-biased visibility, frontier-only exploration,
   unseen seeds.

---

## 2. Literature review (thread by thread)

**2.1 Active link prediction (closest work).**
ALPINE (Chen, Kang, Lijffijt, De Bie; arXiv:2002.01227, 2020; Applied Sciences 11(11), 2021)
actively queries links using a network-embedding model of link probability and a closed-form
V-optimality criterion [V]. Its simple selection rules (max-probability, max-degree,
PageRank) are the origin of our heuristic baselines. *Difference:* we optimise downstream
**node classification on a target set**, not link-prediction quality, and we do not use its
embedding-based method (a stated limitation).
ActiveLink (Ostapuk, Yang, Cudre-Mauroux; WWW 2019) applies deep active learning to link
prediction in knowledge graphs [V].

**2.2 Active learning on graphs, and learned selection.**
GPA (Hu et al.; NeurIPS 2020, arXiv:2006.13463) learns a transferable reinforcement-learning
policy for choosing which nodes to label [V]. *Difference:* it queries labels; we query links.
Bevilacqua et al. (ICLR 2024) learn selection policies for subgraph GNNs; it is not active
learning, but it supports learning what to select [V]. Both point to learning a policy that
amortises the cost of our scorer (future work).
ActiveReach (Raghebi, Banaei-Kashani; Frontiers in Big Data 2024) uses active learning to
answer approximate reachability queries in large graphs [V]. Reachability is also a quantity
we measure (targets reachable from a label), so it is worth reading for the connection.

**2.3 Perturbation and value-of-information signals.**
PerbALGraph (Sun et al.; arXiv:2410.19176, 2024) uses a perturbation-based, cheap value
signal for graph active learning [V]. Our scorer is also perturbation-based: it tentatively
adds a candidate link and measures the change in the classifier's confidence. Expected error
reduction (Roy & McCallum, 2001) and Bayesian experimental design (Lindley, 1956; Chaloner &
Verdinelli, 1995) are the classical foundations; BALD (Houlsby et al., 2011; Gal et al.,
ICML 2017) and influence functions (Koh & Liang, ICML 2017) give cheaper approximations.

**2.4 Surveys and framing.**
Li et al. (IEEE TNNLS 2024, arXiv:2405.00334) group deep active learning query strategies
into uncertainty, representative, influence, Bayesian and hybrid families [V]; Ren et al.
(ACM Computing Surveys 2021) and Settles (2009) provide the general vocabulary. Deep Adaptive
Design (Foster et al., ICML 2021) treats sequential design as amortised policy learning, and
decision-focused learning (Wilder et al., AAAI 2019) argues for optimising for the
downstream decision. The latter is the motivation for "task-aware" selection.

**2.5 Gap.** Active selection of *links* in order to improve *node classification on a
chosen target set*, evaluated against both link-prediction heuristics and generic
uncertainty sampling.

**Citations used in our code but missing from your list (add and verify):**
Zhu, Ghahramani and Lafferty (2003), label propagation; Adamic and Adar (2003); Deshpande
et al. (2018), contextual stochastic block models; Roy and McCallum (2001); Kipf and Welling
(2017), GCN (for future work).

---

## 3. Methodology

### 3.1 Problem formulation (Figure 1: pipeline diagram, to draw)
- Hidden graph G = (V, E). Observed edges E_obs, a uniform random 15% of E.
- Labelled set L (15% or 40% of nodes); **target set T**: 15 random unlabelled nodes.
- Budget B queries. A query asks whether a pair (i, j) is an edge; the oracle answers yes or
  no; a real edge joins the observed graph.
- Classifier: label propagation on the observed graph (labelled rows clamped, damping 0.99,
  closed-form solve). Nodes no label reaches get a fixed per-node guess, which counts in the
  accuracy.
- Goal: maximise accuracy on T after B queries.

### 3.2 Data
Contextual SBM: n = 90 nodes, 3 balanced classes, p_in = 0.13, p_out = 0.006, 12-dimensional
node attributes with class signal 0.10 (almost uninformative by design, so the links carry
the information). About 186 true edges, 15% observed. A new graph per seed.

### 3.3 Query strategies
- **random.**
- **common neighbours; Adamic-Adar:** likely-link heuristics.
- **max-degree; PageRank:** ALPINE-style heuristics.
- **uncertainty sampling:** the two endpoints' entropy.
- **value (task-aware):** score(i,j) = P(edge ij) x [A_T(G + ij) - A_T(G)], where A_T is the
  mean over targets of the classifier's maximum class probability. P(edge ij) =
  p_in P_same + p_out (1 - P_same), with P_same the probability the endpoints share a class
  under current beliefs; p_in and p_out are **estimated** from the initial observed graph.
  Exact computation for a shortlist of 100 pairs ranked by P(edge) x confidence gap, limited
  to pairs that can change a target's prediction.
- **value_all:** the same objective over all unlabelled nodes (task-agnostic ablation).
- **Static vs adaptive:** static scores once on the initial state and asks the top B;
  adaptive re-scores after each answer.

### 3.4 Evaluation protocol
Accuracy gain on T (accuracy after queries minus before), gap closed = gain divided by
(full-graph accuracy minus starting accuracy), reach (targets connected to a label), real edges
found. 30 seeds, identical instances across strategies, paired t-test, Wilcoxon, bootstrap
CI, Holm correction. **Confirmatory run on unseen seeds 100-129**, because the scorer was
tuned on pilot seeds 0-2.
**Verification:** a leakage test (non-train labels randomised: no scorer's queries changed);
reproduction of earlier runs exactly (difference 0).

---

## 4. Preliminary results

**4.1 Link-finding rate is the wrong target** (Figure: edges found vs accuracy).
Phase 1 (50 seeds): common neighbours found significantly more real edges than random
(budget 25: 1.94 vs 0.78, p < 0.0001), yet downstream accuracy did not differ. In the main
runs, common neighbours finds about 10 real edges in 200 queries against about 8 for value,
but closes 7-13% of the gap against 45-53% for value.

**4.2 Node attributes (Table).** At class signal 0.1 a features-only classifier is at
chance (0.317 at 15% labels, chance 0.333), and mixing features into propagation lowers
accuracy (0.434 to 0.379). Attributes help from signal 0.3 (0.41) and up (0.87 at 1.0). On
the locked graph at 15% labels, 85% of unlabelled nodes are reachable from no label.

**4.3 Main comparison (Figure: headline; Table).** Accuracy gain on targets after 200 queries
(unseen seeds; original seeds in brackets):

| strategy | 15% labels | 40% labels |
|---|---|---|
| random | +0.007 (+0.013) | +0.044 (+0.040) |
| common neighbours | +0.033 (+0.031) | +0.060 (+0.049) |
| max-degree | +0.047 (+0.038) | +0.031 (+0.036) |
| value, all nodes, adaptive | +0.087 (+0.100) | +0.111 (+0.113) |
| value, targets, static | +0.156 (+0.182) | +0.187 (+0.164) |
| **value, targets, adaptive** | **+0.209 (+0.227)** | **+0.238 (+0.242)** |

Gap closed (value, adaptive): 0.45 and 0.53 (original seeds 0.52 and 0.59); every heuristic at
most 0.13. Paired differences, value minus heuristic: +0.16 to +0.21 (p < 0.001, both seed
sets). Task-aware minus all-nodes value: +0.12 to +0.13 (p < 0.001). Adaptive minus static:
+0.05 (p = 0.02 and 0.006 on unseen seeds); not significant at 100 queries or fewer.

**4.4 How much exploration is enough (Figure: 400-query curves).** Value-adaptive reaches 50%
of the gap after about 200 queries (15% labels) and about 160 (40% labels); static needs
about 290 and 310. Returns diminish: the gain added per 100 queries goes +0.13, +0.09, +0.05,
+0.02 (15% labels). Nobody reaches 75% of the gap within 400 queries. Targets reached by a
label: 0.90 and 0.96 for value against 0.33 and 0.46 for random, so reach is necessary but not
sufficient.

**4.5 Frontier-only exploration.** Restricting queries to pairs touching nodes reachable from
the targets leaves value unchanged (+0.229 vs +0.227; +0.231 vs +0.242, not significant) and
helps random (+0.04 to +0.06). Value still beats frontier-restricted random by +0.13 to +0.17.

**4.6 Different ways of hiding edges (Table).** Gap closed at 200 queries, value vs best
heuristic: random hiding 0.52 vs 0.09 and 0.59 vs 0.12; targets-hidden 0.47 vs 0.07 and 0.56
vs 0.12; hub-biased 0.46 vs 0.07 and 0.56 vs 0.13. Adaptive beats static more when targets
start with no edges (+0.138 at 40% labels, p < 0.001).

**4.7 Cluster formation (Figure: cluster panels).** One illustrative run: with value queries
the revealed edges connect target nodes to labelled nodes and target accuracy rises from
0.60 to 0.80; with random queries it stays at 0.60.

**4.8 Side study: edge weights (state the leak).** Homophily-based weights (0.8 within a class,
0.2 across) let scorers find about 15% more edges but did not improve accuracy. A "signed"
classifier gained 0.07-0.19, but the weights were derived from the true classes and so leak
the answer; treat that as an upper bound, not a result.

---

## 5. Limitations (put these in the report)

- Synthetic graphs of one family and size; a label-propagation classifier that the value
  scorer also uses internally.
- The estimates of p_in and p_out are biased (p_out too high) but made no difference to the
  queries chosen here.
- "Task-aware" value also prunes its shortlist to pairs that can affect the targets; the
  separate effect of the objective is not isolated.
- The adaptive gain is modest and appears only after about 200 queries.
- ALPINE's embedding-based method was not implemented; only the simple heuristic rules were
  used, so no claim about outperforming ALPINE.
- Run-time numbers are unreliable, so no cost claim beyond "static is far cheaper than adaptive".

## 6. Planned work (priority order)

1. A real dataset (for example Cora or Citeseer) and a GCN as the classifier.
2. An oracle upper bound and a shortlist ablation.
3. Different target sets (one community, nodes far from any label) and sizes.
4. Stronger baselines: the full ALPINE method, query by committee.
5. Cost models (different action costs), noisy edge weights and informative attributes.
6. A learned selection policy to amortise the scorer (motivated by GPA and the subgraph
   selection work); scaling to larger graphs.

## 7. Conclusion (three sentences, only what is measured)

On partially observed synthetic graphs, finding more real links did not improve node
classification, while choosing links by their expected value for the target nodes closed
roughly half the accuracy gap within 200 queries. Task awareness gave the main gain and
adaptivity a smaller, later one, and both held on unseen seeds and under several ways of
hiding edges. Real data, a GCN and stronger baselines are the next steps.

---

## Figure and table list

| item | file |
|---|---|
| Fig. 1 pipeline diagram | to draw |
| Cluster formation | `fig_e4_cluster_formation.png` |
| Headline curves (200 queries) | `fig_headline_e1.png` |
| 400-query curves | `fig_headline_e3_400.png` |
| Table 4.3, 4.6 | numbers above; sources `e1_active_2x2_log.txt`, `r1_fresh_seeds_log.txt`, `analysis_removal.py` |
| Attributes table | `check_node_attributes_log.txt` |
