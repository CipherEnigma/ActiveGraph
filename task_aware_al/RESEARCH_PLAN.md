# Research Plan: Active Structure Acquisition for Node Classification

**Working title:** *What Should a GNN Ask? Task-Aware Active Edge Querying on Partially Observed Graphs*

**Author:** Pragunie Aditya  
**Started:** November 2026  
**Target venue:** LoG 2026 / NeurIPS 2026 workshop (stretch: NeurIPS/ICML main)

---

## 1. The Problem

### One-sentence version
When a graph is only partially observed, which edges should you query an oracle about to maximise node classification accuracy on a set of target nodes?

### Why this matters
- GNNs propagate information along edges. If target nodes are disconnected from labeled nodes, **no model improvement can help** — the information is unreachable.
- Real graphs are often incomplete: social networks have unobserved friendships, citation networks miss references, biological networks have untested interactions.
- The entire field of "active learning on graphs" asks "which **node** should I label next?" Nobody asks "which **edge** should I query?" — this is a blind spot.

### Formal definition

```
Input:
  - Partially observed graph  G_obs = (V, E_obs)  where  E_obs ⊂ E_true
  - Node features  X ∈ R^{n×d}
  - Small labeled set  L ⊂ V  with known classes
  - Target set  T ⊂ V \ L  (nodes we want to classify)
  
Oracle:
  - O(i, j) → {0, 1}   (does edge (i,j) exist in E_true?)
  - Each call costs one unit of budget

Objective:
  - Choose a sequence of queries  q_1, q_2, ..., q_B  that maximises
    f(T, G_final) = classification accuracy on T
  - where G_final = G_obs ∪ {discovered true edges}

Constraints:
  - Budget B is fixed in advance
  - The learner never sees E_true \ E_obs except through oracle responses
  - Each pair is queried at most once
```

### What this is NOT
| Related problem | Key difference |
|---|---|
| Active node labeling | Budget buys node labels, not edge information |
| Link prediction | No budget, predict all edges at once, no downstream task |
| Graph structure learning (Pro-GNN, IDGL) | No oracle — hallucinate edges from features |
| Graph rewiring (SDRF, FoSR) | Adds virtual edges for message passing, not real edges |

---

## 2. The Core Idea

### Task-aware acquisition

Not all edges are equally useful. An edge between two nodes in a dense, well-classified cluster adds nothing. An edge that bridges an isolated target node to a labeled neighborhood can flip dozens of classifications.

**Our acquisition strategy scores each candidate edge by its expected improvement on the target set:**

```
score(i, j) = P(edge exists | i, j) × E[Δ accuracy on T | edge (i,j) added]
```

This has two components:
1. **Link probability** — how likely is this edge to be real? (VGAE)
2. **Task value** — if it's real, how much does it help the targets? (GNN forward pass)

### The "why is the GNN failing?" diagnostic

For each uncertain target node, classify the failure mode:
- **Structural failure:** No path to any labeled node → needs an edge query
- **Informational failure:** Path exists but labels in neighborhood disagree → needs more edges or a label
- **Intrinsic failure:** Node features are ambiguous, more edges won't help → skip, save budget

This diagnostic is novel — no existing paper distinguishes between these failure modes.

---

## 3. Method Architecture

Three GNN modules working together:

```
                    Observed graph G_obs
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Module A │ │  Module B │ │  Module C │
        │   VGAE   │ │  Value   │ │ Downstream│
        │ Link Prob │ │  Scorer  │ │ Classifier│
        └────┬─────┘ └────┬─────┘ └─────┬─────┘
             │             │             │
             ▼             ▼             ▼
        Candidate     Score each     Evaluate
        pool of      candidate by   accuracy on
        likely edges  target impact  targets
             │             │             │
             └──────┬──────┘             │
                    ▼                    │
              Query oracle               │
              with top edge              │
                    │                    │
                    ▼                    │
              Add edge if real ──────────┘
              (update G_obs, repeat)
```

### Module A: VGAE Candidate Generator
- **Model:** Variational Graph Autoencoder (Kipf & Welling 2016)
- **Input:** Observed adjacency A_obs, node features X
- **Output:** P(edge | i, j) for all unobserved pairs
- **Purpose:** Filter the O(n²) possible pairs down to a manageable candidate pool
- **Implementation:** 2-layer GCN encoder → 64-dim latent → inner-product decoder
- **Retrain:** Every R queries (R is a hyperparameter, default 100)
- **Replaces:** Current logistic regression link model (AUC 0.63 → target 0.80+)
- **File:** New module in `cora_exp.py`, reusable for all datasets

### Module B: GNN Value Scorer
- **Purpose:** For each candidate edge, estimate how much it helps target classification
- **Three versions (compare all):**

| Version | How it scores | Speed | Accuracy |
|---|---|---|---|
| **LP-proxy** (current) | Closed-form LP re-solve per candidate | Fast | Lower bound |
| **GCN-probe** | One GCN forward pass with the candidate edge added | Medium | Good |
| **Learned** | Small GNN trained to predict accuracy delta from (graph state, candidate edge) | Fastest at inference | Best if enough training data |

- **Target-awareness:** Only score edges that could reach target nodes (within k hops of T in the candidate graph). This is the "task-aware" part that distinguishes our method.
- **File:** Extend `sc_value()` in `cora_exp.py` with new scoring modes

### Module C: Downstream Classifier
- **Models:** GCN, GAT, GraphSAGE (run all three to show architecture-agnostic gains)
- **Purpose:** The model we're actually trying to help — it classifies target nodes
- **Different from the scorer** — this is important: the scorer (Module B) uses a proxy; the evaluator uses a full GNN. This prevents circularity.
- **Retrain:** At evaluation checkpoints (not every query — too expensive)
- **Architecture:** 2 layers, 64 hidden, trained for 150-200 epochs
- **File:** Extend `gcn_accuracy()` in `cora_exp.py` to support GAT and GraphSAGE

### Module D: Failure Diagnostic
- **Purpose:** Classify why each target node is misclassified, to decide whether an edge query can help
- **Implementation:**
  ```
  For each target t ∈ T:
    1. Compute shortest path distance to nearest labeled node in G_obs
    2. If distance = ∞: STRUCTURAL failure (unreachable)
    3. If distance < ∞:
       a. Collect labels in k-hop neighborhood
       b. If labels disagree (entropy > threshold): INFORMATIONAL failure
       c. If labels agree but prediction still wrong: INTRINSIC failure
  ```
- **Use in acquisition:** Prioritise edges that fix STRUCTURAL failures first, then INFORMATIONAL. Don't waste budget on INTRINSIC failures.
- **Metric:** Track fraction of each failure type over the query sequence — should see STRUCTURAL → 0 as budget is spent
- **File:** New function `diagnose_failures()` in `cora_exp.py`

---

## 4. Baselines (10 total)

### Group 1: Link-prediction heuristics (query order = link score ranking)
These find real edges efficiently but ignore the downstream task.

| # | Baseline | Description | Status |
|---|---|---|---|
| 1 | **Random** | Random selection within the candidate pool | Done (`sc_random`) |
| 2 | **Common Neighbours** | |CN(i) ∩ CN(j)| | Done (`sc_cn`) |
| 3 | **Adamic-Adar** | Σ 1/log(deg(k)) for common neighbours k | Done (`sc_aa`) |
| 4 | **VGAE top-k** | Query highest P(edge) from the VGAE, ignoring targets | New — use Module A's output directly |

### Group 2: Active learning strategies (adapted from node→edge querying)
These use principled acquisition functions from the AL literature.

| # | Baseline | Description | Status |
|---|---|---|---|
| 5 | **Uncertainty sampling** | Query edges adjacent to targets where classifier entropy is highest | New |
| 6 | **Query by Committee** | Ensemble of k GNNs (different dropout masks); query edge that maximally reduces committee disagreement on targets | New |
| 7 | **Information Gain** | Adapt IGP (Franchi et al. 2022) from node→edge: estimate information gain on target beliefs from each edge query | New |

### Group 3: Graph completion methods (the "why query at all?" control)
These add/remove edges from features alone, no oracle needed.

| # | Baseline | Description | Status |
|---|---|---|---|
| 8 | **Pro-GNN** | Jointly learns graph structure and GNN parameters, removes noisy edges and adds likely ones | New |
| 9 | **Feature-kNN graph** | Build a kNN graph from node features, union with observed graph | Partially done (used in pool construction) |

### Group 4: Upper bound

| # | Baseline | Description | Status |
|---|---|---|---|
| 10 | **Oracle greedy** | Sees all true edges; at each step picks the edge that gives maximum actual accuracy gain on targets | New |

---

## 5. Datasets

| Dataset | Nodes | Edges | Classes | Features | Domain | Why included |
|---|---|---|---|---|---|---|
| **Cora** | 2,708 | 5,429 | 7 | 1,433 (BoW) | Citation | Standard benchmark, existing infrastructure |
| **CiteSeer** | 3,327 | 4,732 | 6 | 3,703 (BoW) | Citation | Sparser than Cora, harder link prediction |
| **Amazon Computers** | 13,752 | 245,861 | 10 | 767 | Co-purchase | Different domain, much denser |
| **OGB-arxiv subgraph** | ~20,000 | ~100,000 | 40 | 128 | Citation | Scale test, well-known benchmark |

### Dataset preparation
- **Observation rate:** Randomly hide (1 - obs_frac) of edges. Default obs_frac = 0.15 (matching synthetic setup). Sweep: {0.1, 0.2, 0.3, 0.5, 0.7}.
- **Labels:** k labels per class, randomly selected. Default k=20 for Cora/CiteSeer, k=50 for larger graphs.
- **Targets:** 50 random non-labeled nodes (Cora/CiteSeer), 100 for larger graphs.
- **Seeds:** 0-29 for tuning/development; 200-229 reserved for confirmatory run (never used for tuning). See `CLAIM.md`.
- **Loader pattern:** `load_<dataset>()` returns (A, X, y). Follow `load_cora()` in `cora_exp.py`.

---

## 6. Experiments

### Experiment 1: Main result (all strategies, all datasets)
- **Goal:** Show our method beats all baselines on all datasets
- **Settings:** obs_frac=0.3, budget=500, targets=50, 30 seeds
- **Metric:** AUC of accuracy-vs-queries curve (primary), final accuracy, real edges found
- **Statistics:** Paired t-test + Wilcoxon, Holm correction, 95% bootstrap CI
- **Output:** Table (one row per strategy, one column per dataset) + accuracy-vs-queries figure

### Experiment 2: Observation rate sweep
- **Goal:** Show when edge queries matter most
- **Settings:** obs_frac ∈ {0.1, 0.2, 0.3, 0.5, 0.7}, budget=500, 20 seeds
- **Hypothesis:** Biggest gain at 10-30% where graph is fragmented; diminishes past 50%
- **Output:** Heatmap of (observation rate × strategy) → accuracy gain

### Experiment 3: Budget scaling
- **Goal:** Show efficiency — our method's marginal gain per query is higher
- **Settings:** budget ∈ {50, 100, 200, 500, 1000}, obs_frac=0.3, 20 seeds
- **Output:** Accuracy-vs-budget curves for each strategy

### Experiment 4: Architecture comparison
- **Goal:** Method works regardless of downstream GNN architecture
- **Settings:** GCN, GAT, GraphSAGE as Module C, best settings from Exp 1, 20 seeds
- **Output:** Table showing gains are consistent across architectures

### Experiment 5: Ablation study
- **Goal:** Isolate which component causes the gain
- **Ablations:**

| Variant | What changes | Tests |
|---|---|---|
| No target-awareness | Score all candidates equally, not just target-reachable ones | Is task-awareness needed? |
| No VGAE pool | Random candidate pool instead of VGAE-filtered | Does pool quality matter? |
| LP scorer instead of GCN scorer | Use LP proxy (current method) instead of GCN probe | Does GCN scoring help? |
| No diagnostic | Don't classify failure modes, treat all targets equally | Does the diagnostic help? |
| Random within shortlist | Random selection from same shortlisted candidates | Is it the scoring or the shortlisting? |
| Static (score once) vs. adaptive (re-score each step) | Fix scores at step 0 vs. update after each query | Does adaptivity matter? |

### Experiment 6: Active querying vs. graph completion
- **Goal:** Show that querying an oracle beats hallucinating edges from features
- **Settings:** Our method with B queries vs. Pro-GNN adding B edges automatically, 20 seeds
- **Key insight:** Pro-GNN adds more edges but some are wrong; we add fewer but they're verified
- **Output:** Head-to-head accuracy comparison + analysis of false edges added by Pro-GNN

### Experiment 7: Joint label + edge budget (stretch)
- **Goal:** When you can spend each budget unit on a label OR an edge, what's optimal?
- **Settings:** Total budget = 100, vary allocation: (100 labels, 0 edges) to (0 labels, 100 edges)
- **Hypothesis:** Optimal split depends on observation rate — low obs → more edges, high obs → more labels
- **Output:** Accuracy vs. allocation ratio at different observation rates

---

## 7. Visualisations

### Figure 1: Problem illustration
- A real Cora subgraph at 15% observation
- Color targets (red), labeled nodes (blue), other (grey)
- Show unreachable targets — nodes with no path to any labeled node
- Caption: "At 15% observation, X% of targets are unreachable from labeled nodes. No classifier can help them."

### Figure 2: Graph growing under different strategies
- 2×3 grid: rows = {our method, VGAE top-k, random}; columns = {step 0, step 100, step 200}
- Node positions fixed (spring layout from true graph)
- Observed edges in grey, newly queried edges in color (green = real, red = not real)
- Show how our method grows toward targets while others grow randomly

### Figure 3: The critical edge
- One specific query that bridges two components
- Before: 30 targets in an isolated cluster, all misclassified
- After: one bridge edge connects them to labeled nodes, 25 flip to correct class
- Accuracy jump visible in the curve

### Figure 4: Main result — accuracy vs. queries
- One subplot per dataset (2×2 grid)
- All strategies as colored lines with 95% CI bands
- Our method clearly above baselines, approaching oracle

### Figure 5: Failure mode evolution
- Stacked area chart over query steps
- Three areas: structural (red), informational (yellow), intrinsic (blue)
- Show structural failures dropping to near zero as budget is spent
- Show that our method eliminates structural failures faster than baselines

### Figure 6: When do edges vs. labels matter?
- X-axis: observation rate; Y-axis: accuracy gain from budget
- Two curves: "budget spent on edges" vs. "budget spent on labels"
- Crossover point is a key finding

---

## 8. Paper Structure

```
Abstract                              (~150 words)
1. Introduction                       (~1.5 pages)
   - GNNs fail silently on incomplete graphs
   - Existing active learning only queries labels
   - We introduce active structure acquisition
   - Contributions list (problem, method, experiments, findings)

2. Related Work                       (~1 page)
   2.1 Active learning on graphs      (node label queries)
   2.2 Graph structure learning       (automatic, no oracle)
   2.3 Link prediction                (no budget, no downstream task)
   2.4 Graph rewiring                 (virtual edges, not real)

3. Problem Formulation                (~1 page)
   3.1 Setup and notation
   3.2 Relationship to active node labeling
   3.3 Theoretical properties         (submodularity analysis)

4. Method                             (~2 pages)
   4.1 VGAE candidate generation      (Module A)
   4.2 Task-aware value scoring       (Module B)
   4.3 Failure diagnosis              (Module D)
   4.4 The full acquisition loop

5. Experiments                        (~3 pages)
   5.1 Setup (datasets, baselines, metrics, statistics)
   5.2 Main result
   5.3 When do edge queries matter?   (observation rate sweep)
   5.4 Ablation study
   5.5 Active querying vs. graph completion
   5.6 Architecture comparison

6. Analysis                           (~1 page)
   6.1 What kinds of edges matter?    (bridges vs. within-cluster)
   6.2 Failure mode evolution
   6.3 When to query edges vs. labels

7. Limitations and Future Work        (~0.5 page)
   - Assumes oracle is perfect (no noisy responses)
   - Candidate pool limits recall
   - Computational cost of re-scoring
   - Future: joint label+edge budget, learned policies, noisy oracles

8. Conclusion                         (~0.5 page)

Appendix
   A. Proof of submodularity / counter-example
   B. Full experimental tables
   C. Hyperparameter sensitivity
   D. Additional visualisations
```

---

## 9. Theoretical Component

### Question: Is the edge-query value function submodular?

Let S ⊂ E_unobserved be a set of queried edges (those found to be real), and f(S) = accuracy on targets after adding S to G_obs.

**Submodularity** means: for S ⊂ S', adding one more edge to S gives at least as much gain as adding it to S'. In other words, diminishing returns.

**Investigation plan:**
1. **Empirical check:** Plot marginal gain of the k-th edge query vs. k across many seeds. If marginal gain is non-increasing on average → evidence for submodularity.
2. **Formal analysis:** 
   - For LP classifier: f(S) involves the graph Laplacian inverse, which changes non-linearly with added edges. Likely NOT exactly submodular.
   - For GCN: even less likely, since GCN is a non-linear function of adjacency.
3. **Possible results:**
   - If submodular → greedy gives (1-1/e) approximation. Clean theorem.
   - If approximately submodular (submodularity ratio γ > 0) → greedy gives γ(1-1/e) approximation. Still useful.
   - If not submodular → prove an approximation bound under weaker conditions (e.g., graph connectivity constraints). Or show empirically that greedy still works well despite non-submodularity.

**Any of these three outcomes is publishable.**

---

## 10. Implementation Plan

### What exists (in `task_aware_al/`)

| File | What it does | Status |
|---|---|---|
| `common.py` | Synthetic graph setup (90-node cSBM), LP solver, Instance class | Done, stable |
| `exp_active_2x2.py` | Synthetic experiment runner with SCORERS dict | Done, stable |
| `cora_exp.py` | Cora experiment: data loading, LP scorer, GCN evaluator, logistic link model | Done, needs upgrades |
| `analysis_cora.py` | Basic result analysis (gain, gap, paired tests) | Done, needs extension |
| `verify_results.py` | Leakage and reproduction checks | Done, reuse |

### What to build (in order)

#### Phase 1: Core upgrades (weeks 1-2)

| Task | File | What to do | Days |
|---|---|---|---|
| **1a. VGAE link model** | `cora_exp.py` | Replace `LogisticRegression` with VGAE. Encoder: 2-layer GCN (1433→256→64). Decoder: inner product. Train on observed edges with negative sampling. Target: AUC > 0.80, pool recall > 0.50. | 3-4 |
| **1b. GCN value scorer** | `cora_exp.py` | New scoring mode `sc_value_gcn`: for each candidate, do one GCN forward pass with edge added, measure accuracy change on targets. Batch the forward passes for speed. | 3-4 |
| **1c. GAT + GraphSAGE** | `cora_exp.py` | Extend `gcn_accuracy()` to accept a model type parameter. Add GAT (2 heads, 8 dim each) and GraphSAGE (mean aggregator, 64 hidden). | 2 |
| **1d. CiteSeer loader** | `cora_exp.py` | `load_citeseer()` — same pattern as `load_cora()`, swap dataset name. | 0.5 |
| **1e. Amazon Computers loader** | `cora_exp.py` | `load_amazon()` — use `torch_geometric.datasets.Amazon`. Adjust label/target counts for 10 classes. | 1 |

#### Phase 2: New baselines (weeks 3-4)

| Task | File | What to do | Days |
|---|---|---|---|
| **2a. VGAE top-k baseline** | `cora_exp.py` | `sc_vgae_topk`: query highest P(edge) from VGAE, ignore targets. | 0.5 |
| **2b. Uncertainty sampling** | `cora_exp.py` | `sc_uncertainty_edge`: for each target, find adjacent candidate edges, score by classifier entropy on that target. Query the edge adjacent to the most uncertain target. | 2 |
| **2c. Query by Committee** | `cora_exp.py` | `sc_qbc`: train k=5 GCNs with different dropout seeds. Score each candidate by reduction in committee disagreement (vote entropy) on targets. | 3 |
| **2d. Information gain** | `cora_exp.py` | `sc_infogain`: adapt IGP (Franchi et al. 2022) from node selection to edge selection. Estimate expected reduction in posterior entropy on target labels. | 3 |
| **2e. Oracle greedy** | `cora_exp.py` | `sc_oracle`: sees true adjacency; at each step, tries every candidate, picks the one with max actual accuracy gain. Expensive but only needs to run once per seed for the upper bound. | 2 |
| **2f. Pro-GNN baseline** | New file: `baselines_gsl.py` | Run Pro-GNN on G_obs, extract the learned graph, evaluate downstream accuracy. Compare "B edges added by Pro-GNN" vs "B edges added by our queries". | 3-4 |

#### Phase 3: Diagnostic module (week 4)

| Task | File | What to do | Days |
|---|---|---|---|
| **3a. Failure diagnosis** | `cora_exp.py` | `diagnose_failures(inst)`: for each target, compute reachability, neighborhood label entropy, classify as structural/informational/intrinsic. | 2 |
| **3b. Diagnostic-guided acquisition** | `cora_exp.py` | `sc_value_diag`: score only targets with structural/informational failures; skip intrinsic. Redistribute budget. | 2 |
| **3c. Failure tracking** | `cora_exp.py` | At each checkpoint, record failure mode counts. Add to `snapshot()` output. | 1 |

#### Phase 4: Experiments (weeks 5-6)

| Task | Compute | Seeds | Days |
|---|---|---|---|
| **4a. Development runs** (all strategies, Cora, seeds 0-4) | CPU, ~2h | 5 | 1 |
| **4b. Main result** (all strategies, 4 datasets, seeds 0-29) | CPU, ~24-48h per dataset | 30 | 4 (parallel) |
| **4c. Observation rate sweep** (5 rates, Cora+CiteSeer, seeds 0-19) | CPU, ~12h | 20 | 2 |
| **4d. Budget scaling** (5 budgets, Cora, seeds 0-19) | CPU, ~8h | 20 | 1 |
| **4e. Architecture comparison** (3 GNNs, Cora, seeds 0-19) | CPU/GPU, ~6h | 20 | 1 |
| **4f. Ablation** (6 variants, Cora, seeds 0-19) | CPU, ~12h | 20 | 2 |
| **4g. vs. Pro-GNN** (Cora+CiteSeer, seeds 0-19) | CPU, ~8h | 20 | 1 |
| **4h. Confirmatory run** (best settings, seeds 200-229, all datasets) | CPU, ~48h | 30 | 2 |

#### Phase 5: Analysis + figures (weeks 7-8)

| Task | File | Days |
|---|---|---|
| **5a. Extended analysis script** | `analysis_cora.py` | Add AUC metric, Holm correction, Wilcoxon, bootstrap CI, effect size. | 2 |
| **5b. Visualisation: accuracy vs queries** | New: `fig_main_result.py` | Publication-quality matplotlib with CI bands, 2×2 dataset grid. | 2 |
| **5c. Visualisation: graph growth** | New: `fig_graph_growth.py` | networkx + matplotlib, 2×3 grid showing strategy comparison. | 2 |
| **5d. Visualisation: critical edge** | New: `fig_critical_edge.py` | Before/after graph snapshots with accuracy heatmap. | 1 |
| **5e. Visualisation: failure modes** | New: `fig_failure_modes.py` | Stacked area chart of failure types over query steps. | 1 |
| **5f. Visualisation: edge vs label value** | New: `fig_edge_vs_label.py` | If Experiment 7 is done, crossover plot. | 1 |
| **5g. Submodularity check** | New: `analysis_submodularity.py` | Plot marginal gains, compute submodularity ratio. | 2 |

#### Phase 6: Writing (weeks 9-11)

| Task | Days |
|---|---|
| **6a. Abstract + introduction** | 3 |
| **6b. Related work** | 2 |
| **6c. Problem formulation + theory** | 3 |
| **6d. Method section** | 3 |
| **6e. Experiments + analysis** | 4 |
| **6f. Conclusion + limitations** | 1 |
| **6g. Appendix** | 2 |
| **6h. Revision after feedback** | 3-5 |

---

## 11. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| VGAE doesn't beat logistic model on Cora | Low | High | Tune architecture; fallback to GAE (non-variational) or Node2Vec |
| GCN accuracy stays flat with more edges (Cora at 15%) | Medium | High | Use higher observation rates (30-50%) or subgraphs; change primary metric to LP; report null result honestly |
| Pro-GNN beats our method | Medium | Medium | This is a valid finding — means the oracle isn't worth the cost. Identify the regime where querying wins (low obs, sparse features). |
| Oracle greedy is barely better than our method | Low | Positive | Great news — means our heuristic is near-optimal. Report the gap. |
| Oracle greedy is *much* better than our method | Medium | Medium | Shows room for improvement. Motivates the learned policy (Module B learned version). |
| Compute time too long for 30-seed runs | Medium | Medium | Reduce shortlist, use LP-proxy instead of GCN-probe for scoring, reduce GCN epochs, parallelize across machines. |
| Submodularity doesn't hold | High | Low | Expected. Report the submodularity ratio and empirical evidence that greedy still works. This is fine for a paper. |
| Method doesn't transfer to Amazon Computers | Medium | Low | Report it. "The method works on citation networks but not co-purchase" is still a paper with a caveated claim. |

---

## 12. Compute Requirements

| Resource | Minimum | Preferred |
|---|---|---|
| **CPU** | 8-core laptop | 16+ core workstation or cluster |
| **RAM** | 16 GB | 32 GB (for OGB-arxiv subgraph) |
| **GPU** | Not required for core experiments | 1× consumer GPU (RTX 3060+) speeds up VGAE training and GCN evaluation by 2-3× |
| **Time** | ~1 week wall-clock for all experiments (with parallelisation) | ~3 days with GPU + 16 cores |
| **Storage** | ~5 GB for all datasets + results | — |

---

## 13. Success Criteria

### Minimum for a paper (workshop / LoG)
- [ ] Our method beats random, CN, AA, and VGAE top-k on at least 3/4 datasets
- [ ] Statistically significant (p < 0.05 Holm-corrected) on at least 2 datasets
- [ ] Ablation shows target-awareness contributes (not just pool quality)
- [ ] 20+ seeds per setting, proper statistical testing
- [ ] At least one compelling visualisation (graph growth or critical edge)

### Strong paper (main conference)
- [ ] Everything above, plus:
- [ ] Beats all 9 baselines including QBC and Info Gain on 3+ datasets
- [ ] Theoretical result (submodularity bound or approximation guarantee)
- [ ] Active querying beats Pro-GNN in the low-observation regime
- [ ] Failure diagnostic shown to improve budget efficiency
- [ ] Architecture-agnostic (works with GCN, GAT, GraphSAGE)
- [ ] Joint label+edge budget experiment shows non-trivial optimal split

---

## 14. Key References to Cite

### Active learning on graphs (node queries)
- Robust Offline Active Learning on Graphs (Wu & Yuan, NeurIPS 2024)
- Algorithms and Hardness for Active Learning on Graphs (ICML 2025)
- Information Gain Propagation (Franchi et al., 2022)
- DiffusAL (2023)
- AGE: Active Learning on Attributed Graphs (Cai et al., 2017)

### Graph structure learning
- Pro-GNN (Jin et al., NeurIPS 2020)
- IDGL (Chen et al., NeurIPS 2020)
- GAUG: Data Augmentation for GNNs (Zhao et al., ICLR 2021)
- Graph Structure Learning survey (Zhu et al., 2021)

### Active link prediction
- ALPINE (Chen et al., 2020) — the closest prior work to ours

### GNN architectures
- GCN (Kipf & Welling, ICLR 2017)
- GAT (Velickovic et al., ICLR 2018)
- GraphSAGE (Hamilton et al., NeurIPS 2017)
- VGAE (Kipf & Welling, NeurIPS 2016 workshop)

### Graph rewiring
- Graph Rewiring Survey (2025)
- SDRF (Topping et al., ICLR 2022)

---

## 15. Rules While This Plan Is Active

1. **CLAIM.md stays in force.** Seeds 200-229 are never used for tuning. Report is frozen until Phase 4h (confirmatory run).
2. **No cherry-picking.** All experiments report all seeds, all strategies. No dropping seeds that "look weird."
3. **Pre-register settings.** Before the confirmatory run (Phase 4h), write all hyperparameters into CLAIM.md. No changes after seeing results.
4. **Report failures.** If a baseline beats us, report it. If the method doesn't transfer to a dataset, report it. Honest null results are part of the paper.
5. **One commit per phase.** Keep the git history clean and reproducible.
