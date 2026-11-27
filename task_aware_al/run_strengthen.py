"""Strengthened experiment runs: 300-node subgraphs, 20 seeds, GCN at all checkpoints.

Addresses three reviewer weaknesses:
  1. 150-node subgraphs too small  → 300 nodes
  2. 10 seeds too few              → 20 seeds
  3. GCN only at endpoints         → GCN at every checkpoint

Usage:
  python run_strengthen.py                     # run all 4 datasets (overnight)
  python run_strengthen.py --phase smoke       # 2 seeds, budget 50 (quick check)
  python run_strengthen.py --phase cora        # Cora only
  python run_strengthen.py --datasets cora citeseer  # specific datasets
  python run_strengthen.py --seeds 30          # override seed count
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

DATASETS = ["cora", "citeseer", "amazon", "coauthor_cs"]
# Subseeds chosen so BFS from that start node reaches 300 nodes with all classes
# represented. CiteSeer is very disconnected (subseed=42 gives only 9 nodes);
# Coauthor CS has 15 classes spread unevenly.
SUBSEEDS = {"cora": 42, "citeseer": 1, "amazon": 2, "coauthor_cs": 85}

# 7 core strategies for the paper (drop the 6 minor variants)
CORE_STRATEGIES = (
    "joint_value,joint_infogain,edge_only,label_only,"
    "label_uncertainty,alternating,random"
)

# Full 13 strategies (if --all-strategies)
ALL_STRATEGIES = (
    "joint_value,joint_infogain,edge_only,label_only,label_uncertainty,"
    "label_random,edge_pagerank,edge_linkprob,alternating,"
    "edge70_label30,edge30_label70,random_joint,random"
)


def run_one(dataset, seeds, budget, subgraph, workers, subseed, strategies,
            observe=0.15, tag=None):
    if tag is None:
        tag = f"v2_{dataset}"
    out = os.path.join(RESULTS, f"{tag}.json")

    # Check if already complete
    if os.path.exists(out):
        try:
            d = json.load(open(out))
            done_seeds = set()
            for st in d.get("runs", {}).values():
                done_seeds.update(st.keys())
            if len(done_seeds) >= seeds:
                print(f"  SKIP {out} (already has {len(done_seeds)} seeds)")
                return out
        except Exception:
            pass

    cmd = [
        sys.executable, "-I", os.path.join(HERE, "exp_joint.py"),
        "--seeds", str(seeds),
        "--budget", str(budget),
        "--subgraph", str(subgraph),
        "--subseed", str(subseed),
        "--dataset", dataset,
        "--observe", str(observe),
        "--strategies", strategies,
        "--workers", str(workers),
        "--out", out,
        # GCN at ALL checkpoints (not just endpoints)
        "--no-gcn_endpoints_only",
    ]
    print(f"\n{'='*60}")
    print(f"  {tag}: {dataset} sub={subgraph} seeds={seeds} budget={budget}")
    print(f"  strategies: {strategies}")
    print(f"  GCN at all checkpoints: YES")
    print(f"{'='*60}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/3600:.1f}h) -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["smoke", "cora", "all", "sparsity"])
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=DATASETS)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--subgraph", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--all-strategies", action="store_true",
                    help="Run all 13 strategies (slower)")
    a = ap.parse_args()

    strategies = ALL_STRATEGIES if a.all_strategies else CORE_STRATEGIES
    results = []

    if a.phase == "smoke":
        seeds = a.seeds or 2
        for ds in (a.datasets or ["cora"]):
            tag = f"v2_smoke_{ds}"
            results.append(run_one(
                ds, seeds, min(a.budget, 50), a.subgraph, a.workers,
                SUBSEEDS[ds], strategies, tag=tag))

    elif a.phase == "cora":
        seeds = a.seeds or 20
        results.append(run_one(
            "cora", seeds, a.budget, a.subgraph, a.workers,
            SUBSEEDS["cora"], strategies))

    elif a.phase == "sparsity":
        seeds = a.seeds or 20
        sparsity_strats = "joint_value,edge_only,label_only,alternating,random"
        for obs in [0.05, 0.10, 0.15, 0.25]:
            tag = f"v2_sparsity_cora_obs{int(obs*100):02d}"
            results.append(run_one(
                "cora", seeds, a.budget, a.subgraph, a.workers,
                SUBSEEDS["cora"], sparsity_strats, observe=obs, tag=tag))

    elif a.phase == "all":
        seeds = a.seeds or 20
        datasets = a.datasets or DATASETS
        for ds in datasets:
            results.append(run_one(
                ds, seeds, a.budget, a.subgraph, a.workers,
                SUBSEEDS[ds], strategies))

    print(f"\n\nAll runs complete. Result files:")
    for r in results:
        print(f"  {r}")
    print(f"\nTo analyse: python analysis_joint.py {' '.join(results)}")


if __name__ == "__main__":
    main()
