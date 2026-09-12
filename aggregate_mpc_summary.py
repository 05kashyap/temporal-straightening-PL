#!/usr/bin/env python
"""Aggregate per-seed closed-loop MPC logs.json into a mean +/- std summary file.

run_mpc.sh runs one plan.py per eval seed, producing
    plan_outputs_<planner>/<model>_s<seed>_gH<goal_H>/logs.json
(one `final_eval/...` JSON line per seed). This script reads those and writes a
single machine-readable summary to
    plan_outputs_<planner>/summaries/<model>_gH<goal_H>.json
containing per-seed values and mean/std across seeds for every final_eval metric
(success_rate, mean_state_dist, mean_visual_dist, mean_proprio_dist, ...).

Usage:
    python aggregate_mpc_summary.py <planner> <model> <goal_H> <seed> [<seed> ...]

Prints the human-readable summary line run_mpc.sh used to print to the console
(for backwards compatibility) and exits 0 if >=1 seed had a final_eval, 1 otherwise.

std is the population std over seeds (biased), matching run_mpc.sh's awk formula:
    var = mean(x^2) - mean(x)^2  (clamped at 0),  std = sqrt(var)
"""

import argparse
import json
import os
import sys


def _load_final_eval(logs_path):
    """Return the final `final_eval/...` dict from a plan logs.json, or None."""
    if not os.path.isfile(logs_path):
        return None
    final = None
    with open(logs_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            evals = {k: v for k, v in obj.items() if k.startswith("final_eval/")}
            if evals:
                final = evals
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("planner", help="e.g. gd_mpc or mpc_cem")
    parser.add_argument("model", help="checkpoint / run-dir model name")
    parser.add_argument("goal_h", type=int, help="planning horizon used for the run dir")
    parser.add_argument("seeds", nargs="+", type=int, help="eval seeds to aggregate")
    args = parser.parse_args()

    per_seed = []          # [{seed, final_eval}] for seeds that produced results
    for seed in args.seeds:
        rundir = os.path.join(
            "plan_outputs_" + args.planner,
            f"{args.model}_s{seed}_gH{args.goal_h}",
        )
        evals = _load_final_eval(os.path.join(rundir, "logs.json"))
        if not evals:
            print(f"  [warn] no final_eval in {os.path.join(rundir, 'logs.json')}", file=sys.stderr)
            continue
        per_seed.append({"seed": seed, "final_eval": evals})

    if not per_seed:
        return 1

    # metric name -> list of numeric values across seeds (insertion order = logs order)
    metric_values = {}
    for rec in per_seed:
        for key, val in rec["final_eval"].items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                metric_values.setdefault(key, []).append(float(val))

    metrics = {}
    for key, vals in metric_values.items():
        n = len(vals)
        mean = sum(vals) / n
        var = (sum(v * v for v in vals) / n) - mean * mean
        if var < 0:
            var = 0.0
        metrics[key] = {"values": vals, "mean": mean, "std": var ** 0.5}

    summary = {
        "planner": args.planner,
        "model": args.model,
        "goal_H": args.goal_h,
        "n_seeds": len(per_seed),
        "seeds": [rec["seed"] for rec in per_seed],
        "metrics": metrics,
        "per_seed": per_seed,
    }

    out_dir = os.path.join("plan_outputs_" + args.planner, "summaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}_gH{args.goal_h}.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    sr = metrics.get("final_eval/success_rate")
    if sr is not None:
        print(
            "  === {} / {}: success_rate = {:.4f} +/- {:.4f}  (n={} seeds) ===".format(
                args.planner, args.model, sr["mean"], sr["std"], len(per_seed)
            )
        )
    else:
        print(
            "  === {} / {}: success_rate missing from seed logs (found {} metrics) ===".format(
                args.planner, args.model, list(metrics)
            )
        )
    print(f"      summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
