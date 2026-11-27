"""Full experiment matrix for the PAKDD submission.

Runs joint label+edge experiments across:
  - 3 datasets: Cora, CiteSeer, Amazon Computers (150-node subgraphs)
  - 4 sparsity levels: OBSERVE_FRAC = 0.05, 0.10, 0.15, 0.25
  - 7 strategies: joint_value, edge_only, label_only, label_uncertainty,
                  alternating, random_joint, random
  - 10 seeds per cell

Usage:
  python run_pakdd.py                    # run everything
  python run_pakdd.py --phase pilot      # 3 seeds, budget 50 (quick check)
  python run_pakdd.py --phase dataset    # all datasets, default sparsity
  python run_pakdd.py --phase sparsity   # sparsity sweep on Cora only
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
OBSERVE_FRACS = [0.05, 0.10, 0.15, 0.25]
STRATEGIES = ("joint_value,joint_infogain,edge_only,label_only,label_uncertainty,"
              "label_random,edge_pagerank,edge_linkprob,alternating,"
              "edge70_label30,edge30_label70,random_joint,random")
SUBSEEDS = {"cora": 42, "citeseer": 42, "amazon": 42, "coauthor_cs": 42}


def run_one(dataset, observe, seeds, budget, workers, subseed, tag):
    out = os.path.join(RESULTS, f"joint_{tag}.json")
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
        "--subgraph", "150",
        "--subseed", str(subseed),
        "--dataset", dataset,
        "--observe", str(observe),
        "--strategies", STRATEGIES,
        "--workers", str(workers),
        "--out", out,
    ]
    print(f"\n{'='*60}")
    print(f"  {tag}: {dataset} observe={observe} seeds={seeds} budget={budget}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  Done in {time.time()-t0:.0f}s -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["pilot", "dataset", "sparsity", "all"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    results = []

    if a.phase == "pilot":
        seeds = a.seeds or 3
        for ds in DATASETS:
            tag = f"pilot_{ds}_obs15"
            results.append(run_one(ds, 0.15, seeds, min(a.budget, 50), a.workers, SUBSEEDS[ds], tag))

    elif a.phase == "dataset":
        seeds = a.seeds or 10
        for ds in DATASETS:
            tag = f"{ds}_obs15"
            results.append(run_one(ds, 0.15, seeds, a.budget, a.workers, SUBSEEDS[ds], tag))

    elif a.phase == "sparsity":
        seeds = a.seeds or 10
        for obs in OBSERVE_FRACS:
            tag = f"cora_obs{int(obs*100):02d}"
            results.append(run_one("cora", obs, seeds, a.budget, a.workers, SUBSEEDS["cora"], tag))

    elif a.phase == "all":
        seeds = a.seeds or 10
        for ds in DATASETS:
            for obs in OBSERVE_FRACS:
                tag = f"{ds}_obs{int(obs*100):02d}"
                results.append(run_one(ds, obs, seeds, a.budget, a.workers, SUBSEEDS[ds], tag))

    print(f"\n\nAll runs complete. Result files:")
    for r in results:
        print(f"  {r}")
    print(f"\nTo analyse: python analysis_joint.py {' '.join(results)}")


if __name__ == "__main__":
    main()
