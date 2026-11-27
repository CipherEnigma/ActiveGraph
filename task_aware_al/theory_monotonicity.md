# Monotonicity Analysis for Joint Active Acquisition

## Definitions

Let G = (V, E) be a graph with adjacency A. Let L ⊆ V be a set of nodes with known
(correct) labels y_L. Let LP(A, L) denote the clamped label propagation solution:

    F = (I - α·S_UU)^{-1} · α · S_UL · Y_L

where S = D^{-1}A is the row-normalized transition, U = V \ L, and α ∈ (0,1).

Let acc_T(A, L) = (1/|T|) Σ_{t∈T} 𝟙[argmax F_t = y_t] be the target-set accuracy.

---

## Proposition 1 (Empirical Label Near-Monotonicity)

**Statement.** In practice, adding a correctly labeled node v to the labeled set L
is near-monotone in target-set LP accuracy:

    acc_T(A, L ∪ {v}) ≥ acc_T(A, L) holds in >99.97% of observed cases.

**Evidence.** Across 4 real-world graph datasets (Cora, CiteSeer, Amazon Computers,
Coauthor CS), 10 seeds each, 200-step budgets, we observe:

| Query type         | Total steps | Violations | Rate    | Max |Δ| |
|--------------------|-------------|------------|---------|---------|
| Label (all strats) | ~8,200      | 2          | 0.024%  | 0.04    |
| Edge (joint_value) | ~32,000     | 3          | 0.009%  | 0.04    |
| Edge (edge_only)   | ~8,000      | 14         | 0.175%  | 0.14    |
| Edge (pagerank)    | ~8,000      | 20         | 0.250%  | 0.34    |

**Remark.** Full monotonicity does not hold theoretically, even on trees.
A correctly labeled node v of class c ≠ y_t can, via clamping, block the propagation
of a helpful signal from a more distant labeled node of class y_t. However, this
requires a specific topological configuration (v lies between t and a helpful ℓ)
AND class disagreement (y_v ≠ y_t), which is rare in homophilous graphs.

**Practical implication.** The near-monotonicity of label queries, combined with
the more frequent violations of edge queries under heuristic strategies, provides
empirical justification for the joint scorer's preference: when both action types
offer similar expected gains, label queries are the safer choice.

---

## Proposition 2 (Edge Non-Monotonicity)

**Statement.** There exist graphs where adding a true edge to the observed graph
strictly decreases target-set LP accuracy:

    ∃ G, e ∈ E: acc_T(A + e, L) < acc_T(A, L)

**Proof.** By construction. Consider a path graph with labeled node ℓ (class 0)
connected to target t through intermediate node m. Before adding edge (m, u):
- m has degree 2, sends weight 1/2 of ℓ's signal to t.

After adding edge (m, u) where u is unlabeled:
- m has degree 3, sends weight 1/3 of ℓ's signal to t.
- The signal at t decreases because m's attention is diluted.

If t's correct prediction depended on receiving sufficient signal from ℓ,
this dilution can flip t's prediction. ∎

(See task_aware_al/theory_setfunction.py for the full computational verification.)

**Practical implication.** Pure edge acquisition strategies face non-negligible
violation rates (up to 0.25% per step, with individual drops up to 0.34 accuracy).
The joint formulation mitigates this by allowing the scorer to choose label queries
when edge queries would be risky.

---

## Proposition 3 (Joint Safety)

**Statement.** The joint_value scorer achieves a monotonicity violation rate
strictly lower than pure edge acquisition strategies across all tested settings.

**Evidence.** joint_value: 3 violations in ~40,000 steps (0.008%).
edge_only: 14 violations in ~8,000 steps (0.175%). edge_pagerank: 20 violations
in ~8,000 steps (0.250%).

**Interpretation.** The joint scorer's ability to substitute a label query for a
risky edge query acts as an implicit safety mechanism. When the edge probe shows
marginal gain (suggesting possible non-monotonicity), the label probe often offers
a safer alternative with guaranteed near-monotonicity.
