#!/usr/bin/env python
"""analysis/div_emb_tables.py
===========================
Render markdown tables of the planner's latent-divergence metrics

    mean_div_visual_emb  = || z_env.visual  - z_imagined.visual  ||_F
    mean_div_proprio_emb = || z_env.proprio - z_imagined.proprio ||_F

(planning/evaluator.py:152-153, aggregated in eval_actions at lines 168-169 and
208-209) for the four training arms, one table per (loop, env), as mean +/- std
across eval seeds.

Where the numbers come from
---------------------------
For every run dir `plan_outputs_<planner>/<model>_s<seed>_gH<H>/logs.json` this
script takes the **last `final_eval/...` record**, exactly like
`helpers/aggregate_mpc_summary.py` (the planner's final evaluation, i.e. after
the MPC loop has finished replanning), and averages over seeds with the
population (biased) std -- the convention documented in that script and used by
run_mpc.sh's awk formula. `plan_outputs_*/summaries/*.json` are only used as a
cross-check, never as the source, because the stale
`plan_outputs_gd_mpc/summaries/*_openloop_gH25.json` files disagree with the real
open-loop logs (5.09 vs 39.61 for umaze/baseline visual divergence).

The four arms' run-dir names come from `helpers/extract_planner_curves.py`
(MODELS / ARM_ORDER), the repo's single source of truth for the arm vocabulary,
so a table can never disagree with the CSVs and figures about which run is which
arm. The display labels are paper-style ("PReg" for the two-thirds / p-reg arm)
and therefore defined here rather than reused from ARM_LABEL.

Reading the numbers
-------------------
`mean_div_*_emb` is a **batch-level** Frobenius norm over
`(n_evals, 1, num_patches, emb_dim)`, so it scales as ~RMS_per_element x sqrt(N)
and is only comparable within one table (same n_evals, same encoder). For the
maze runs (global 1x384 embedder, n_evals=50) the visual divisor is
sqrt(50*1*384)=138.6; for PushT (196x8 channel embedder) it is sqrt(50*196*8)=280;
proprio is sqrt(50*10)=22.4 everywhere. Closed- and open-loop values live in
different regimes (4 replans of 5 actions vs one plan of 25), so they are kept in
separate sections on purpose.

Usage
-----
    python analysis/div_emb_tables.py                    # both loops, seeds 100-102
    python analysis/div_emb_tables.py --seeds 100 101 102 103 104 105
    python analysis/div_emb_tables.py --outdir analysis_outputs
"""

import argparse
import csv
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from helpers.extract_planner_curves import MODELS, ARM_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# Presentation. Arms stay in ARM_ORDER (the CSVs'/figures' order); only the
# labels differ from ARM_LABEL, at the user's request for "PReg".
# ---------------------------------------------------------------------------
ARM_DISPLAY = {
    "baseline": "Baseline",
    "straighten": "Straighten",
    "p_reg": "PReg",
    "both": "Both",
}

ENV_DISPLAY = {
    "point_maze": ("PointMaze (umaze)", "`umaze`, global 1x384 embedder"),
    "point_maze_medium": ("PointMaze (medium)", "`medium`, global 1x384 embedder"),
    "pusht": ("PushT", "`pusht`, 196x8 channel embedder"),
}

# loop dir -> (section title, one-line protocol description)
LOOP_META = {
    "plan_outputs_gd_mpc": (
        "Closed loop",
        "MPC: `max_iter=20`, replans every `n_taken_actions=5` for `goal_H=25`; "
        "divergence accumulates over the replans.",
    ),
    "plan_outputs_gd_ol": (
        "Open loop",
        "single plan for the whole `goal_H=25` (`max_iter=1`, "
        "`n_taken_actions=25`); no replanning.",
    ),
}

METRICS = [
    ("mean_div_visual_emb", "`mean_div_visual_emb`"),
    ("mean_div_proprio_emb", "`mean_div_proprio_emb`"),
    ("success_rate", "success rate"),
]


def last_final_eval(logs_path):
    """Return the final `final_eval/...` dict of a plan logs.json, or None.

    Same rule as helpers/aggregate_mpc_summary.py:_load_final_eval (the planner
    appends one `final_eval/...` line per plan.py invocation; the last one wins).
    """
    last = None
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
                last = evals
    return last


def collect_arm(loop_dir, model, seeds, goal_h):
    """Per-seed final-eval metrics for one arm.

    Returns (per_seed, missing): per_seed is {seed: {metric: value}} for the seeds
    that produced a final_eval, missing lists the seeds that did not.
    """
    per_seed, missing = {}, []
    for seed in seeds:
        run_dir = os.path.join(loop_dir, "%s_s%d_gH%d" % (model, seed, goal_h))
        logs = os.path.join(run_dir, "logs.json")
        if not os.path.isfile(logs):
            missing.append(seed)
            continue
        evals = last_final_eval(logs)
        if not evals:
            missing.append(seed)
            continue
        per_seed[seed] = evals
    return per_seed, missing


def mean_std(values):
    """Population (biased) mean/std, matching helpers/aggregate_mpc_summary.py."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    mean = sum(vals) / n
    var = (sum(v * v for v in vals) / n) - mean * mean
    return mean, (max(0.0, var) ** 0.5)


def cell(values, ndigits):
    """`mean +- std` for a list of per-seed values, or `n/a`."""
    mean, std = mean_std(values)
    if mean is None:
        return "n/a"
    return "%.*f ± %.*f" % (ndigits, mean, ndigits, std)


def render_env_table(env, arms_data, ndigits, notes):
    """One markdown table for one (loop, env)."""
    title, subtitle = ENV_DISPLAY.get(env, (env, ""))
    out = ["### %s" % title, ""]
    if subtitle:
        out += ["%s  " % subtitle, ""]
    header = ["Method"] + [label for _, label in METRICS] + ["n"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for arm in ARM_ORDER:
        row = arms_data.get(arm, {})
        n_used = row.get("n", 0)
        if n_used == 0:
            notes.append("no %s result for arm `%s` (n=0)" % (env, arm))
        cells = [cell(row.get(key, []), ndigits)
                 for key, _ in METRICS]
        out.append("| **%s** | %s | %d |" % (ARM_DISPLAY[arm],
                                             " | ".join(cells), n_used))
    out.append("")
    return out


def render_section(loop_dir, loop_rows, seeds, ndigits, notes):
    """All env tables for one loop (closed / open)."""
    title, protocol = LOOP_META.get(loop_dir, (loop_dir, ""))
    out = ["## %s — `%s/`" % (title, loop_dir), ""]
    if protocol:
        out += [protocol, ""]
    out.append("Cells are `mean ± std` across %d eval seed%s (%s); both metrics are "
               "lower-is-better and comparable within this table only."
               % (len(seeds), "" if len(seeds) == 1 else "s",
                  ", ".join(str(s) for s in seeds)))
    out.append("")
    for env in loop_rows:
        out += render_env_table(env, loop_rows[env], ndigits, notes)
    return out



ENV_ORDER = ["point_maze", "point_maze_medium", "pusht", "wall"]

HEADER = """# Latent divergence between imagined and executed rollouts

`mean_div_visual_emb` / `mean_div_proprio_emb` measure how far the world model's
**imagined** terminal latent drifts from the encoder embedding of the frame the
environment **actually** reaches after the same planned actions
(`planning/evaluator.py:152-153`; aggregated across eval chunks at lines 168-169
and 208-209). Lower is better.

* source: the **last `final_eval/...` record** of every
  `plan_outputs_<planner>/<model>_s<seed>_gH<H>/logs.json` (the planner's final
  evaluation, after the MPC loop), the same rule
  `helpers/aggregate_mpc_summary.py` uses;
* aggregation: mean ± **population (biased) std** across eval seeds, as in
  `helpers/aggregate_mpc_summary.py` / `run_mpc.sh`;
* arms and run-dir names come from `helpers/extract_planner_curves.py`
  (`MODELS`), the repo's single source of truth for the four-arm vocabulary;
* regenerate with `python analysis/div_emb_tables.py` (no GPU, no planning).
"""

NOTES = """## Notes

* **These are batch totals, not per-episode means.** `torch.norm(tensor)` with no
  `dim` is the Frobenius norm over the whole `(n_evals, 1, num_patches, emb_dim)`
  tensor, so a cell is ~`RMS_per_element x sqrt(N)` with
  `N = n_evals x num_patches x emb_dim` (`n_evals = 50`). The `mean_` in the key
  name is historical. Divisors: mazes (global 1x384 embedder) `sqrt(50*1*384) =
  138.6`, PushT (196x8 channel embedder) `sqrt(50*196*8) = 280`, proprio (10-dim)
  `sqrt(50*10) = 22.4`. Divide before comparing across envs or embedders -- the
  visual and proprio columns are in different units and must not be compared with
  each other.
* **Closed and open loop are separate regimes.** A closed-loop episode accumulates
  latent divergence while the planner replans (`max_iter=20`, taking 5 actions at
  a time), an open-loop one comes from a single plan over the whole horizon, so
  the two sections are not interchangeable -- compare rows within a section only
  (the ordering is not even consistent: `medium`/baseline is 84.5 closed vs 91.0
  open, while PushT/baseline is 68.6 vs 55.7).
* **Medium closed loop**: `straighten` and `both` also have seeds 103-105
  available; they are excluded so that every cell has `n = 3` (seeds 100-102),
  matching the question this table answers.
* `plan_outputs_mpc_cem/` is not tabulated: only one seed per arm exists there.
* The `plan_outputs_gd_mpc/summaries/*_openloop_gH25.json` files are **not** used
  for the open-loop section: they disagree with the real open-loop logs (e.g.
  umaze/baseline visual divergence 5.09 vs 39.61), so the raw
  `plan_outputs_gd_ol/*/logs.json` are the source of truth here.
"""


# Run-dir prefix per env (mirrors train_server.sh's ENV_SEL naming: point_maze -> umaze,
# point_maze_medium -> medium, ...). Needed because the built-in MODELS names below are
# the dev machine's; a machine retrained with different recipes (projchannel instead of
# projglobal, ttaggtwothirds, aggflatten, ...) has the same four arms under other names.
ENV_PREFIX = {
    "point_maze": "umaze",
    "point_maze_medium": "medium",
    "pusht": "pusht",
    "wall": "wall",
}


def _classify_arm(name):
    """Same token rules as run_mpc.sh's arm discovery."""
    cos = "cos" in name
    tt = ("wothirds" in name) or ("twothirds" in name)
    if cos and tt:
        return "both"
    if cos:
        return "straighten"
    if tt:
        return "p_reg"
    if "_False_" in name:
        return "baseline"
    return None


def discover_arms(loop_dir, env, goal_h):
    """Resolve arm -> run-dir name by scanning <loop_dir>/<prefix>_*_s*_gH<goal_h>.

    Used when none of the built-in MODELS names exist on disk (the dev machine's names,
    which differ from a retrained machine's). Returns {} unless the mapping is
    unambiguous -- one candidate per arm -- so a wrong mapping can never be averaged into
    a table; otherwise the candidates are reported and the caller falls back.
    """
    prefix = ENV_PREFIX.get(env)
    if not prefix or not os.path.isdir(loop_dir):
        return {}
    pattern = re.compile(r"^%s_(?P<arm>.+)_s(?P<seed>\d+)_gH%d$" % (re.escape(prefix), goal_h))
    found = {}
    for name in sorted(os.listdir(loop_dir)):
        m = pattern.match(name)
        if not m or not os.path.isdir(os.path.join(loop_dir, name)):
            continue
        # classify the FULL dir name (like run_mpc.sh): the baseline marker is
        # "_False_", which only appears once the env prefix is in front.
        arm = _classify_arm(name)
        if arm:
            # store the run-dir name WITHOUT the _s<seed>_gH<H> suffix, i.e. exactly
            # what collect_arm() prefixes with "_s<seed>_gH<H>" again.
            suffix = "_s%s_gH%d" % (m.group("seed"), goal_h)
            found.setdefault(arm, set()).add(name[: -len(suffix)])
    resolved = {}
    for arm in ARM_ORDER:
        cands = sorted(found.get(arm, ()))
        if len(cands) == 1:
            resolved[arm] = cands[0]
        elif cands:
            print("  [warn] %s / %s: %d candidates for arm '%s' (%s)"
                  % (loop_dir, env, len(cands), arm, ", ".join(cands)), file=sys.stderr)
            return {}
    return resolved if len(resolved) == len(ARM_ORDER) else {}


def build(loop_dirs, seeds, goal_h, envs):
    """Collect every (loop, env, arm) cell plus the per-run provenance rows."""
    data, run_rows, notes = {}, [], []
    for loop_dir in loop_dirs:
        loop_rows = {}
        for env in envs:
            # Prefer the repo's built-in (dev machine) arm names; if none of them exist
            # for this env, resolve the arms from the directories actually present.
            models = dict(MODELS[env])
            if not any(os.path.isdir(os.path.join(loop_dir, "%s_s%d_gH%d" % (models[a], s, goal_h)))
                       for a in ARM_ORDER for s in seeds):
                discovered = discover_arms(loop_dir, env, goal_h)
                if discovered:
                    models = discovered
                    print("  [arms] %s / %s: %s"
                          % (loop_dir, env,
                             ", ".join("%s=%s" % (a, discovered[a]) for a in ARM_ORDER)))
                    notes.append("%s / %s: built-in arm names absent, used the four run dirs "
                                 "found on disk (%s)"
                                 % (loop_dir, env,
                                    ", ".join("%s=`%s`" % (ARM_DISPLAY[a], discovered[a])
                                             for a in ARM_ORDER)))
            arms, env_rows = {}, []
            for arm in ARM_ORDER:
                model = models[arm]
                per_seed, missing = collect_arm(loop_dir, model, seeds, goal_h)
                if missing and per_seed:
                    notes.append("%s / %s / `%s`: no final_eval for seed(s) %s"
                                 % (loop_dir, env, arm,
                                    ", ".join(str(s) for s in missing)))
                entry = {"model": model, "n": len(per_seed),
                         "seeds": sorted(per_seed)}
                for key, _ in METRICS:
                    entry[key] = [per_seed[s].get("final_eval/" + key)
                                  for s in sorted(per_seed)]
                arms[arm] = entry
                env_rows.append({
                    "loop": loop_dir,
                    "loop_label": LOOP_META.get(loop_dir, (loop_dir, ""))[0],
                    "env": env,
                    "env_label": ENV_DISPLAY.get(env, (env, ""))[0],
                    "arm": arm,
                    "label": ARM_DISPLAY[arm],
                    "model": model,
                    "n": len(per_seed),
                    "seeds": " ".join(str(s) for s in sorted(per_seed)),
                    "values": {key: entry[key] for key, _ in METRICS},
                })
            if not any(arms[a]["n"] for a in ARM_ORDER):
                # No run at all for this env in this loop: no table, one note.
                print("  [warn] %s: no runs for env '%s' (%s)"
                      % (loop_dir, env, models[ARM_ORDER[0]]), file=sys.stderr)
                notes.append("%s: env `%s` skipped, no final_eval runs found."
                             % (loop_dir, env))
                continue
            loop_rows[env] = arms
            run_rows += env_rows
        data[loop_dir] = loop_rows
    return data, run_rows, notes


def write_csv(path, run_rows, ndigits):
    """One row per (loop, env, arm) with mean/std per metric."""
    cols = ["loop", "loop_label", "env", "env_label", "arm", "label", "model",
            "n", "seeds"]
    for key, _ in METRICS:
        cols += ["%s_mean" % key, "%s_std" % key]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in run_rows:
            row = {k: r.get(k) for k in ("loop", "loop_label", "env", "env_label",
                                         "arm", "label", "model", "n", "seeds")}
            for key, _ in METRICS:
                mean, std = mean_std(r["values"].get(key, []))
                row["%s_mean" % key] = "" if mean is None else "%.*f" % (ndigits, mean)
                row["%s_std" % key] = "" if std is None else "%.*f" % (ndigits, std)
            writer.writerow(row)



def write_markdown(path, data, run_rows, notes, seeds, ndigits):
    """Assemble the whole report: header, one section per loop, notes, provenance."""
    out = [HEADER]
    n_cells = 0
    for loop_dir, loop_rows in data.items():
        out += render_section(loop_dir, loop_rows, seeds, ndigits, notes)
        n_cells += sum(1 for env in loop_rows for arm in ARM_ORDER
                       if loop_rows[env][arm]["n"] > 0)
    out += [NOTES]
    out += ["## Runs behind the tables", "",
            "| Loop | Env | Method | Run name (`plan_outputs_<planner>/`) | Seeds | n |",
            "|---|---|---|---|---|---|"]
    for r in run_rows:
        out.append("| %s | %s | %s | `%s` | %s | %d |"
                   % (r["loop_label"], r["env_label"], r["label"], r["model"],
                      r["seeds"] or "-", r["n"]))
    out.append("")
    out.append("Generated by `analysis/div_emb_tables.py` from %d (loop, env, arm) "
               "cells." % len(run_rows))
    out.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    return n_cells


def main():
    parser = argparse.ArgumentParser(
        description="Markdown tables of mean_div_visual_emb / mean_div_proprio_emb "
                    "per env, method and loop type (closed vs open loop).")
    parser.add_argument("--loop-dirs", nargs="+", default=list(LOOP_META),
                        help="plan output dirs, in report order "
                             "(default: %(default)s)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102],
                        help="eval seeds to aggregate (default: %(default)s)")
    parser.add_argument("--goal-h", type=int, default=25,
                        help="planning horizon of the run dirs (default: %(default)s)")
    parser.add_argument("--envs", nargs="+",
                        default=[e for e in ENV_ORDER if e in MODELS],
                        help="envs to tabulate (default: %(default)s)")
    parser.add_argument("--outdir", default="analysis_outputs",
                        help="where the .md/.csv are written (default: %(default)s)")
    parser.add_argument("--md-name", default="div_emb_tables.md")
    parser.add_argument("--csv-name", default="div_emb_summary.csv")
    parser.add_argument("--ndigits", type=int, default=3,
                        help="decimals in the rendered cells (default: %(default)s)")
    args = parser.parse_args()

    os.chdir(REPO)  # run dirs and --outdir are relative to the repo root
    for env in args.envs:
        if env not in MODELS:
            print("  [warn] unknown env '%s' (known: %s)" % (env, list(MODELS)),
                  file=sys.stderr)

    data, run_rows, notes = build(args.loop_dirs, args.seeds, args.goal_h,
                                 [e for e in args.envs if e in MODELS])

    print("  latent divergence tables: seeds=%s goal_H=%d"
          % (" ".join(str(s) for s in args.seeds), args.goal_h))
    for r in run_rows:
        if r["n"]:
            print("    %-20s %-20s %-11s n=%d  seeds=%s"
                  % (r["loop_label"], r["env_label"], r["label"], r["n"],
                     r["seeds"]))

    os.makedirs(args.outdir, exist_ok=True)
    md_path = os.path.join(args.outdir, args.md_name)
    csv_path = os.path.join(args.outdir, args.csv_name)
    n_cells = write_markdown(md_path, data, run_rows, notes, args.seeds,
                             args.ndigits)
    write_csv(csv_path, run_rows, args.ndigits)
    print("  wrote %s (%d cells with data)" % (md_path, n_cells))
    print("  wrote %s" % csv_path)
    if notes:
        print("  notes:")
        for n in notes:
            print("    - %s" % n)
    return 0 if n_cells else 1


if __name__ == "__main__":
    sys.exit(main())

