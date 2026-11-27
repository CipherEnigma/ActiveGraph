# ActiveGraph

**Joint Active Edge and Label Acquisition for Node Classification on Partially Observed Graphs**

> When a graph is only partially observed, should you spend your budget querying missing edges or labelling nodes? EdgeQuery introduces a unified framework where each oracle call reveals either an edge or a label, and an LP-probe scorer decides which action maximises downstream classification accuracy.

## Key Findings

- **+9.9pp GCN accuracy** on Cora with one-hot features (p < 0.001, 20 seeds) from joint edge+label acquisition vs label-only
- **Feature regime matters**: edges help when features are uninformative (one-hot, low-dim PCA) but hurt when features are rich (original bag-of-words)
- **Homophily required**: significant gains on citation networks (Cora, CiteSeer), neutral on co-purchase/co-authorship (Amazon, Coauthor CS), neutral on heterophilic graphs (Actor)
- **Feature-adaptive scorer**: a one-time GCN probe detects the feature regime and adjusts edge/label allocation — never significantly hurts on any of 5 datasets

## Method

1. **LP-Probe Scorer**: evaluates edge and label candidates on a common scale using the closed-form label propagation solution. At each step, the best edge competes against the best label — whichever offers higher expected accuracy gain wins.
2. **Feature-Adaptive Extension**: trains a GCN once at step 0 to estimate feature informativeness (φ). High φ → allocate mostly labels. Low φ → mix edges and labels.
3. **Oracle Simulation**: experiments use ground-truth edges to simulate oracle responses on BFS subgraphs (300 nodes, 5% observation rate, budget of 200 queries).

## Datasets

| Dataset | Type | Nodes | Edges | Classes | Edge Benefit |
|---------|------|------:|------:|--------:|:-------------|
| Cora | Citation | 2,708 | 5,278 | 7 | +9.9pp (p=0.002) |
| CiteSeer | Citation | 3,327 | 4,552 | 6 | +1.7pp (p=0.002) |
| Amazon Computers | Co-purchase | 13,752 | 245,861 | 10 | +1.1pp (n.s.) |
| Coauthor CS | Co-authorship | 18,333 | 81,894 | 15 | -1.7pp (p=0.037) |
| Actor | Co-appearance | 7,600 | 26,659 | 5 | -1.2pp (n.s.) |

## Project Structure

```
task_aware_al/
├── cora_exp.py          # Dataset loaders, LP solver, GCN training, Instance class
├── exp_joint.py         # Main experiment runner with all strategies
├── analysis_joint.py    # Result analysis utilities
├── results/             # Experiment result JSONs (5 datasets, 20 seeds)
└── final_report/
    ├── main.tex         # BTP report (1850+ lines, 7 chapters, 19 figures)
    └── figures/         # All generated figures
```

## Running Experiments

```bash
# Core comparison: Cora, one-hot features, 10 seeds
python exp_joint.py --seeds 10 --budget 200 --subgraph 300 \
  --dataset cora --features onehot --observe 0.05 \
  --strategies joint_value,alternating,label_only,edge_only \
  --out results/my_experiment.json

# Feature-adaptive scorer
python exp_joint.py --seeds 10 --budget 200 --subgraph 300 \
  --dataset cora --features onehot --observe 0.05 \
  --strategies path_joint,joint_value,alternating,label_only \
  --out results/adaptive_experiment.json

# Available datasets: cora, citeseer, amazon, coauthor_cs, actor
# Available features: original, onehot, pca2, pca8, pca32
```

## Requirements

- Python 3.8+
- PyTorch, PyTorch Geometric
- NumPy, SciPy, scikit-learn

## Citation

This work is part of a B.Tech Project (BTP) targeting PAKDD 2027.
