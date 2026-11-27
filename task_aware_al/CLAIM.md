# CLAIM

Fixed before the real-data confirmatory run. Do not edit after that run starts.

## The claim (two sentences)

When a graph is only partly observed and link checks are costly, choosing queries by their expected
improvement in the classifier's confidence on the target nodes (the *value* strategy) gives higher
target accuracy per query than link-finding heuristics, even when the heuristics find more real
links. The advantage comes from aiming at the targets, not from finding more links, and it appears
once the budget is large enough, relative to the graph, to connect a meaningful share of targets
to labelled nodes.

## Status of the evidence

- Synthetic (90-node cSBM, label propagation): supported, 30 original and 30 unseen seeds.
- Real data (Cora, 3-seed pilot, 200 queries, 5 labels/class, obs=0.15): **no effect**. The budget
  added about 5% to the visible edges, so this pilot cannot confirm or refute the claim.
- Real data (Cora, 3-seed sweep, 1500 queries, 5 labels/class, obs=0.30): **value leads on LP**
  (+0.047 vs +0.013 for linkprob, p=0.038 vs random). GCN stays flat at all observation rates
  (0.15, 0.30, 0.50) — node features dominate GCN accuracy on Cora.
- **Metric gate triggered (Nov 2026):** GCN is insensitive to structure changes on Cora. Primary
  metric changed from GCN-based AUC to LP-based AUC before any seed 200-229 run. GCN remains
  as a secondary metric. This change is made before seeing confirmatory results, per the rules.
- The claim is therefore **partially supported on real data (LP only), unconfirmed on GCN**.

## Success criterion (decided now)

All of the following, on **at least 20 fresh seeds** (200-229, never used for tuning), on a real
dataset (Cora, scale-matched: budget fixed in advance as a stated fraction of the visible edges):

1. **Primary metric:** area under the curve of target *LP accuracy gain vs queries* (0 to B).
   LP is the natural evaluator for structure changes since it propagates information only along
   edges. GCN accuracy (same AUC) is reported as a **secondary metric**.
2. **Test:** value beats each of common neighbours, max-degree and the plain link predictor
   (`linkprob`) on that AUC. Paired t-test and Wilcoxon both agree, Holm-adjusted p < 0.05 across
   the three comparisons, and the 95% bootstrap CI of the mean paired difference excludes 0.
3. **Practical size:** mean AUC difference of at least 0.02 accuracy (normalised by budget)
   against the best baseline. A significant but tiny gap does not count.
4. **Frozen settings:** pool construction, shortlist size, link model, number of targets, labels per
   class, budget and checkpoints are fixed before the run. No tuning on the 200-229 seeds.
5. **Secondary (needed for the "aiming at the targets" part):** value on targets beats `value_all` on
   the same metric (paired p < 0.05) and an ablation shows the gain does not come only from the
   shortlist restriction.

## What would make me drop or narrow it

| Result | Consequence |
|---|---|
| Value is not better than the best baseline on 20+ fresh seeds at matched scale (CI includes 0, or gap < 0.02) | Drop the real-data claim. Report that the effect did not transfer, and restrict the claim to synthetic community graphs. |
| Value wins under label propagation but not under the GCN | Report honestly: the effect is structure-sensitive but does not transfer to feature-dominant classifiers. |
| `linkprob` or an embedding baseline (ALPINE-style) matches or beats value | Drop "beats link-finding heuristics". Value is then only one of several competitive methods. |
| Value on targets does not beat `value_all`, or the shortlist-only ablation matches it | Drop the "task-aware objective" explanation. Keep only "target-restricted candidate selection helps". |
| Adaptive does not beat static (p >= 0.05) | Drop the adaptive claim. State that adaptivity gives no reliable gain. |
| Win appears only at one label level or one budget | Narrow to that regime and state it explicitly. |
| A leakage or pairing check (`verify_results.py` style) fails | Discard the result until fixed. |

## Rules while this is open

- **Report frozen.** No new report versions until a real-data result exists that meets or fails the
  criterion above. Existing files (v1-v4, drafts) are not edited.
- Report the outcome either way, including the pilot null result.
- Any change to the criterion after seeing 200-229 results is a new, separately labelled analysis,
  not a confirmation.
