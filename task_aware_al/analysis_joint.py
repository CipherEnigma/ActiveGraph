"""Analysis for joint label+edge acquisition experiments.
Usage: python analysis_joint.py results/joint_pilot.json
"""
import json
import sys

import numpy as np
from scipy import stats

files = sys.argv[1:]
data = {}
for f in files:
    data[f] = json.load(open(f))

BUDGET = None

for f, R in data.items():
    args = R.get("args", {})
    BUDGET = str(args.get("budget", 200))
    print(f"\n{'='*80}")
    print(f"File: {f}")
    print(f"Args: seeds={args.get('seeds')}, budget={BUDGET}, subgraph={args.get('subgraph')}, "
          f"dataset={args.get('dataset')}")
    print(f"{'='*80}")

    strats = sorted(R["runs"].keys())
    seeds = sorted(R["runs"][strats[0]].keys()) if strats else []

    # Ceilings
    if R.get("ceiling"):
        cs = R["ceiling"]
        for key in ["lp", "lp_full_labels", "lp_full_graph", "lp_full_both", "gcn"]:
            vals = [cs[s].get(key, 0) for s in cs]
            if any(v > 0 for v in vals):
                print(f"  ceiling {key:>16}: {np.mean(vals):.3f}")

    # Per-strategy summary
    print(f"\n{'strategy':<20} {'LP gain':>8} {'GCN gain':>9} {'edge_q':>7} {'label_q':>8} "
          f"{'edges_found':>12} {'sec':>6}")
    print("-" * 80)

    for st in strats:
        runs = R["runs"][st]
        lp_gains, gcn_gains, eq, lq, ef, times = [], [], [], [], [], []
        for sd in sorted(runs.keys()):
            r = runs[sd]
            rec = r["rec"]
            start = rec["0"]["lp"]
            end = rec[BUDGET]["lp"]
            lp_gains.append(end - start)
            if "gcn" in rec.get(BUDGET, {}) and "gcn" in rec.get("0", {}):
                gcn_gains.append(rec[BUDGET]["gcn"] - rec["0"]["gcn"])
            eq.append(r.get("n_edge_queries", int(BUDGET)))
            lq.append(r.get("n_label_queries", 0))
            ef.append(r.get("edges_found", rec.get(BUDGET, {}).get("found", 0)))
            times.append(r["time"])
        gcn_str = f"{np.mean(gcn_gains):>+9.3f}" if gcn_gains else f"{'n/a':>9}"
        print(f"{st:<20} {np.mean(lp_gains):>+8.3f} {gcn_str} {np.mean(eq):>7.0f} "
              f"{np.mean(lq):>8.0f} {np.mean(ef):>12.1f} {np.mean(times):>6.0f}")

    # Pairwise comparisons: joint_value vs each other
    ref = "joint_value"
    if ref not in R["runs"]:
        ref = strats[0]

    common_seeds = set(R["runs"][ref].keys())
    for st in strats:
        common_seeds &= set(R["runs"][st].keys())
    common_seeds = sorted(common_seeds)

    if len(common_seeds) < 3:
        print("\nToo few common seeds for statistical tests.")
        continue

    print(f"\nPairwise vs {ref} (LP gain @{BUDGET}, {len(common_seeds)} seeds):")
    print(f"{'comparison':<30} {'diff':>7} {'t-p':>7} {'win/loss':>10}")
    print("-" * 60)

    ref_gains = []
    for sd in common_seeds:
        rec = R["runs"][ref][sd]["rec"]
        ref_gains.append(rec[BUDGET]["lp"] - rec["0"]["lp"])
    ref_gains = np.array(ref_gains)

    for st in strats:
        if st == ref:
            continue
        st_gains = []
        for sd in common_seeds:
            rec = R["runs"][st][sd]["rec"]
            st_gains.append(rec[BUDGET]["lp"] - rec["0"]["lp"])
        st_gains = np.array(st_gains)
        d = ref_gains - st_gains
        if d.std() > 0:
            t_p = stats.ttest_rel(ref_gains, st_gains).pvalue
        else:
            t_p = 1.0
        wins = (d > 0).sum()
        losses = (d < 0).sum()
        print(f"{ref + ' - ' + st:<30} {d.mean():>+7.3f} {t_p:>7.3f} {wins}/{losses}")

    # Budget allocation curve for joint strategies
    print(f"\nBudget allocation (edge% at each checkpoint):")
    checkpoints = sorted(set(int(c) for c in R["runs"][strats[0]][common_seeds[0]]["rec"].keys()) - {0})
    header = f"{'strategy':<20}" + "".join(f" @{c:<5}" for c in checkpoints[:6])
    print(header)
    for st in strats:
        row = f"{st:<20}"
        for cp in checkpoints[:6]:
            fracs = []
            for sd in common_seeds:
                rec = R["runs"][st][sd]["rec"]
                cp_key = str(cp)
                if cp_key in rec and "n_edge_queries" in rec[cp_key]:
                    total = rec[cp_key]["n_edge_queries"] + rec[cp_key]["n_label_queries"]
                    if total > 0:
                        fracs.append(rec[cp_key]["n_edge_queries"] / total)
            if fracs:
                row += f" {np.mean(fracs)*100:>5.0f}%"
            else:
                row += f" {'n/a':>5}"
        print(row)

    # Monotonicity violations
    print(f"\nMonotonicity violations:")
    print(f"{'strategy':<20} {'seeds':>5} {'total_violations':>16} {'max_drop':>9}")
    for st in strats:
        total_v = 0
        max_drop = 0.0
        for sd in R["runs"][st]:
            vs = R["runs"][st][sd].get("mono_violations", [])
            total_v += len(vs)
            for v in vs:
                max_drop = min(max_drop, v["delta"])
        print(f"{st:<20} {len(R['runs'][st]):>5} {total_v:>16} {max_drop:>+9.4f}")

    # Regime analysis (when does joint pick labels?)
    print(f"\nRegime analysis (graph state when joint_value picks a label query):")
    for st in strats:
        if not st.startswith("joint"):
            continue
        all_regime = []
        for sd in R["runs"][st]:
            all_regime.extend(R["runs"][st][sd].get("regime_log", []))
        if not all_regime:
            continue
        steps = [r["step"] for r in all_regime]
        reaches = [r["tgt_reach"] for r in all_regime]
        ents = [r["mean_tgt_entropy"] for r in all_regime]
        confs = [r["mean_tgt_conf"] for r in all_regime]
        print(f"  {st}: {len(all_regime)} label queries across {len(R['runs'][st])} seeds")
        print(f"    avg step: {np.mean(steps):.0f}  "
              f"avg tgt_reach: {np.mean(reaches):.2f}  "
              f"avg tgt_entropy: {np.mean(ents):.3f}  "
              f"avg tgt_conf: {np.mean(confs):.3f}")
