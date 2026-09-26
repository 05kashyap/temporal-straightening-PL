#!/usr/bin/env python
"""analysis/run_probes.py
========================
Run BOTH probes for every (env, recipe) checkpoint group under a checkpoint root,
then build the markdown report:

    grounded  ("encoded")  `analysis/linear_probe.py --feature-source encoded`
    ungrounded ("rollout") `analysis/linear_probe.py --feature-source rollout`

Groups come from analysis/probe_ckpts.py: the recipe is read from each run dir's
own hydra.yaml (encoder.projector + encoder.agg_type -> global / flatten / aggmlp)
and the arm from the run-dir name tokens, so every checkpoint of every recipe is
covered and nothing is labelled twice. The output names carry env + recipe
(`rollout_probe_point_maze_medium_aggmlp.png`), which is what stops the global /
flatten / aggmlp groups from overwriting each other.

DATASET_DIR is set per env for each probe invocation, mirroring
run_scripts/dataset_paths.sh (`point_maze` -> <root>/point_maze, `point_maze_medium`
-> <root>, `pusht` -> <root>/pusht), because the probe redirects a checkpoint's
stored dataset path to $DATASET_DIR/<basename> when that stored path is absent --
which is exactly the case on a machine that did not train the checkpoint.

Usage:
    python analysis/run_probes.py                                  # every group
    python analysis/run_probes.py --only point_maze_medium:aggmlp   # one group
    python analysis/run_probes.py --dry-run                         # print the commands
    python analysis/run_probes.py --limit-episodes 2 --tmax 8       # quick smoke
    python analysis/run_probes.py --methods encoded                 # one method only
"""

import argparse
import os
import subprocess
import sys

try:                                                     # `python analysis/run_probes.py`
    import probe_ckpts                                   # noqa: E402
except ImportError:                                      # `from analysis import run_probes`
    from analysis import probe_ckpts                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENVS = ("point_maze", "point_maze_medium", "pusht")
PROBE = os.path.join(REPO, "analysis", "linear_probe.py")
REPORT = os.path.join(REPO, "analysis", "probe_report.py")
METHODS = ("encoded", "rollout")
# dataset directory name per env, i.e. the basename a checkpoint's stored data_path
# ends in -- used to locate the per-env parent for $DATASET_DIR (see env_dataset_dir).
ENV_DATA_NAME = {"point_maze": "point_maze", "point_maze_medium": "point_maze_medium",
                 "pusht": "pusht_noise", "wall": "wall_single",
                 "deformable_env": "deformable"}


def env_dataset_dir(env, data_root):
    """$DATASET_DIR for one env, found by probing the filesystem.

    The probe redirects a checkpoint's stored dataset path to
    ``$DATASET_DIR/<basename>``, so DATASET_DIR has to be a parent of the dataset
    directory -- but which parent differs per machine and per env:

        server  $DATA_ROOT/point_maze/point_maze   -> DATASET_DIR = $DATA_ROOT/point_maze
        laptop  data/datasets/point_maze           -> DATASET_DIR = data/datasets
        server  $DATA_ROOT/pusht/pusht_noise       -> DATASET_DIR = $DATA_ROOT/pusht

    run_scripts/dataset_paths.sh hard-codes the server answers; here both layouts are
    tried and the one that EXISTS wins, so the same command works on either machine
    instead of silently pointing at a directory that is not there.
    """
    name = ENV_DATA_NAME.get(env, env)
    for cand in (os.path.join(data_root, env), data_root):
        if os.path.isdir(os.path.join(cand, name)):
            return cand
    return data_root


def build_cmd(env, recipe, arms, args):
    ckpts = ",".join("%s=%s" % (a, os.path.relpath(d, REPO))
                     for a, d in sorted(arms.items()))
    cmd = [args.py, PROBE, "--env", env, "--label", recipe,
           "--outdir", args.outdir, "--ckpts", ckpts,
           "--seed", str(args.seed), "--test-frac", str(args.test_frac),
           "--device", args.device]
    if args.tmax:
        cmd += ["--tmax", str(args.tmax)]
    if args.limit_episodes:
        cmd += ["--limit-episodes", str(args.limit_episodes)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Run both probes for every checkpoint group")
    ap.add_argument("--ckpt-root", default=os.path.join(REPO, "checkpoints", "test"),
                    help="directory of <run dir>s (server: $CKPT_ROOT/test)")
    ap.add_argument("--data-root", default=os.environ.get(
        "DATA_ROOT", os.path.join(REPO, "data", "datasets")),
        help="parent of the per-env datasets, used to set DATASET_DIR per probe run")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--envs", default=",".join(DEFAULT_ENVS))
    ap.add_argument("--recipes", default="", help="comma list (default: all found)")
    ap.add_argument("--only", default="", help="comma list of env:recipe to restrict to, "
                                               "e.g. point_maze_medium:aggmlp")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma list of encoded,rollout (default both)")
    ap.add_argument("--py", default=sys.executable, help="interpreter for the probes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="probe split seed")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--tmax", type=int, default=0, help="rollout horizon (0 = shortest)")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="cap episodes per split; the ROLLOUT probe only (smoke runs)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--skip-report", action="store_true",
                    help="do not run analysis/probe_report.py afterwards")
    args = ap.parse_args()

    envs = [e for e in args.envs.split(",") if e]
    recipes = [r for r in args.recipes.split(",") if r] or None
    methods = [m for m in args.methods.split(",") if m]
    for m in methods:
        if m not in METHODS:
            raise SystemExit("--methods entries must be in %s" % ", ".join(METHODS))
    only = set(tuple(p.split(":", 1)) for p in args.only.split(",") if p) or None

    groups, skipped, _ = probe_ckpts.discover(args.ckpt_root, envs, recipes)
    print(probe_ckpts.format_table(groups, skipped, args.ckpt_root, verbose=True))
    os.makedirs(args.outdir, exist_ok=True)

    plan = []
    for key in sorted(groups):
        env, recipe = key
        if only and key not in only:
            continue
        arms = groups[key]
        missing = [a for a in probe_ckpts.ARM_ORDER if a not in arms]
        if missing:
            print("note: %s/%s has only %d arm(s) (missing %s); its figures will show "
                  "just those arms" % (env, recipe, len(arms), ",".join(missing)))
        plan.append((env, recipe, arms))
    if not plan:
        raise SystemExit("no group selected (--only/--envs/--recipes filtered everything)")

    print("\n%d group(s) x %d method(s) = %d probe runs\n"
          % (len(plan), len(methods), len(plan) * len(methods)))
    failures = []
    for (env, recipe, arms) in plan:
        for method in methods:
            cmd = build_cmd(env, recipe, arms, args) + ["--feature-source", method]
            label = "%s/%s" % (env, recipe)
            if args.dry_run:
                print("[dry-run] DATASET_DIR=%s %s"
                      % (env_dataset_dir(env, args.data_root), " ".join(cmd)))
                continue
            print("=== %s [%s]" % (label, method), flush=True)
            env_vars = os.environ.copy()
            env_vars["DATASET_DIR"] = env_dataset_dir(env, args.data_root)
            rc = subprocess.run(cmd, cwd=REPO, env=env_vars).returncode
            if rc:
                print("[FAIL] %s [%s] rc=%d -- continuing with the next one"
                      % (label, method, rc))
                failures.append((label, method, rc))
    if args.dry_run:
        return 0

    if not args.skip_report:
        print("\n=== report ===", flush=True)
        rc = subprocess.run([args.py, REPORT, "--outdir", args.outdir,
                             "--ckpt-root", args.ckpt_root,
                             "--envs", ",".join(envs)], cwd=REPO).returncode
        if rc:
            failures.append(("probe_report", "-", rc))

    print("\n=== summary ===")
    if failures:
        print("failed: %s" % ", ".join("%s[%s] rc=%d" % f for f in failures))
        return 1
    print("all %d probe runs finished; figures + markdown under %s"
          % (len(plan) * len(methods), args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
