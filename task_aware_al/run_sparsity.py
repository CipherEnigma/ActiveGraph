"""Sparsity sweep: run joint experiment at different observation fractions on Cora.
Usage: python run_sparsity.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

OBSERVE_FRACS = [0.05, 0.10, 0.15, 0.25]
STRATEGIES = "joint_value,edge_only,label_only,alternating,random"

for obs in OBSERVE_FRACS:
    tag = f"sparsity_cora_obs{int(obs*100):02d}"
    out = os.path.join(RESULTS, f"{tag}.json")
    cmd = [
        sys.executable, "-I", os.path.join(HERE, "exp_joint.py"),
        "--seeds", "10",
        "--budget", "200",
        "--subgraph", "150",
        "--per_class", "20",
        "--dataset", "cora",
        "--observe", str(obs),
        "--strategies", STRATEGIES,
        "--workers", "2",
        "--out", out,
    ]
    print(f"\n{'='*60}")
    print(f"  {tag}: observe={obs}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  Done in {time.time()-t0:.0f}s -> {out}", flush=True)

print("\nAll sparsity sweeps complete.")
print(f"Analyse: python analysis_joint.py {' '.join(os.path.join(RESULTS, f'sparsity_cora_obs{int(o*100):02d}.json') for o in OBSERVE_FRACS)}")
