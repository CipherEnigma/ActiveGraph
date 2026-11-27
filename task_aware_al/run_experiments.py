"""Phase 4: Run all experiments for the paper.

Usage:
  python run_experiments.py --exp 4a          # Dev runs (5 seeds, core strategies)
  python run_experiments.py --exp 4b          # Main result (20 seeds, all strategies, all datasets)
  python run_experiments.py --exp 4c          # Observation rate sweep
  python run_experiments.py --exp 4d          # Budget scaling
  python run_experiments.py --exp 4e          # Architecture comparison
  python run_experiments.py --exp 4f          # Ablation study
  python run_experiments.py --exp 4h          # Confirmatory run (seeds 200-229)
  python run_experiments.py --exp all         # Everything (very long!)
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CORA_EXP = os.path.join(HERE, "cora_exp.py")
GSL_EXP = os.path.join(HERE, "baselines_gsl.py")


def run_cmd(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.0f}s)\n")
    return proc.returncode == 0


def exp_4a():
    """Dev runs: 5 seeds, core strategies, Cora only, budget 500."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,linkprob,value,value_all,value_gcn,vgae_topk,uncertainty_edge,value_diag"
    return run_cmd([
        sys.executable, CORA_EXP,
        "--seeds", "5", "--seed0", "0",
        "--budget", "500", "--targets", "200",
        "--strategies", strategies,
        "--observe", "0.30", "--link_model", "vgae",
        "--dataset", "cora", "--workers", "2",
        "--out", os.path.join(RESULTS_DIR, "4a_dev_cora.json"),
    ], "4a: Dev runs (Cora, 5 seeds, budget=500)")


def exp_4b_dataset(dataset, seeds=20):
    """Main result for one dataset."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,common_nbrs,adamic_adar,maxdeg,linkprob,value,value_all,value_gcn,vgae_topk,uncertainty_edge,qbc,value_diag"
    return run_cmd([
        sys.executable, CORA_EXP,
        "--seeds", str(seeds), "--seed0", "0",
        "--budget", "500", "--targets", "200",
        "--strategies", strategies,
        "--observe", "0.30", "--link_model", "vgae",
        "--dataset", dataset, "--workers", "2",
        "--out", os.path.join(RESULTS_DIR, f"4b_main_{dataset}.json"),
    ], f"4b: Main result ({dataset}, {seeds} seeds)")


def exp_4b():
    """Main result: 20 seeds, all strategies, all datasets."""
    for ds in ["cora", "citeseer"]:
        exp_4b_dataset(ds)


def exp_4c():
    """Observation rate sweep: 5 rates, Cora, 10 seeds."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,linkprob,value,vgae_topk,value_diag"
    for obs in [0.10, 0.20, 0.30, 0.50, 0.70]:
        run_cmd([
            sys.executable, CORA_EXP,
            "--seeds", "10", "--seed0", "0",
            "--budget", "500", "--targets", "200",
            "--strategies", strategies,
            "--observe", str(obs), "--link_model", "vgae",
            "--dataset", "cora", "--workers", "2",
            "--out", os.path.join(RESULTS_DIR, f"4c_obs{obs:.2f}_cora.json"),
        ], f"4c: Obs sweep (obs={obs}, Cora)")


def exp_4d():
    """Budget scaling: 5 budgets, Cora, 10 seeds."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,linkprob,value,vgae_topk,value_diag"
    for budget in [50, 100, 200, 500, 1000]:
        run_cmd([
            sys.executable, CORA_EXP,
            "--seeds", "10", "--seed0", "0",
            "--budget", str(budget), "--targets", "200",
            "--strategies", strategies,
            "--observe", "0.30", "--link_model", "vgae",
            "--dataset", "cora", "--workers", "2",
            "--out", os.path.join(RESULTS_DIR, f"4d_budget{budget}_cora.json"),
        ], f"4d: Budget scaling (B={budget}, Cora)")


def exp_4e():
    """Architecture comparison: GCN/GAT/SAGE, Cora, 10 seeds."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,linkprob,value,vgae_topk,value_diag"
    for arch in ["gcn", "gat", "sage"]:
        run_cmd([
            sys.executable, CORA_EXP,
            "--seeds", "10", "--seed0", "0",
            "--budget", "500", "--targets", "200",
            "--strategies", strategies,
            "--observe", "0.30", "--link_model", "vgae",
            "--dataset", "cora", "--model_type", arch,
            "--workers", "2",
            "--out", os.path.join(RESULTS_DIR, f"4e_arch_{arch}_cora.json"),
        ], f"4e: Architecture ({arch}, Cora)")


def exp_4f():
    """Ablation study: Cora, 10 seeds."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Compare: value vs value_all vs value_diag vs random vs linkprob
    # Also logistic vs vgae link model
    strategies = "random,linkprob,value,value_all,value_gcn,value_diag"
    for lm in ["logistic", "vgae"]:
        run_cmd([
            sys.executable, CORA_EXP,
            "--seeds", "10", "--seed0", "0",
            "--budget", "500", "--targets", "200",
            "--strategies", strategies,
            "--observe", "0.30", "--link_model", lm,
            "--dataset", "cora", "--workers", "2",
            "--out", os.path.join(RESULTS_DIR, f"4f_ablation_{lm}_cora.json"),
        ], f"4f: Ablation (link_model={lm}, Cora)")


def exp_4h():
    """Confirmatory run: seeds 200-229, Cora."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    strategies = "random,common_nbrs,linkprob,value,value_all,vgae_topk,value_diag"
    return run_cmd([
        sys.executable, CORA_EXP,
        "--seeds", "30", "--seed0", "200",
        "--budget", "500", "--targets", "200",
        "--strategies", strategies,
        "--observe", "0.30", "--link_model", "vgae",
        "--dataset", "cora", "--workers", "2",
        "--out", os.path.join(RESULTS_DIR, "4h_confirmatory_cora.json"),
    ], "4h: Confirmatory (Cora, seeds 200-229)")


EXPERIMENTS = {
    "4a": exp_4a, "4b": exp_4b, "4c": exp_4c, "4d": exp_4d,
    "4e": exp_4e, "4f": exp_4f, "4h": exp_4h,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="Experiment to run: 4a|4b|4c|4d|4e|4f|4h|all")
    args = ap.parse_args()

    if args.exp == "all":
        for name in sorted(EXPERIMENTS):
            EXPERIMENTS[name]()
    elif args.exp in EXPERIMENTS:
        EXPERIMENTS[args.exp]()
    else:
        print(f"Unknown experiment: {args.exp}")
        print(f"Available: {', '.join(sorted(EXPERIMENTS))}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
