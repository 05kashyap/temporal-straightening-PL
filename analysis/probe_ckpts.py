#!/usr/bin/env python
"""analysis/probe_ckpts.py
========================
Discover the checkpoint run dirs the probe scripts should cover, grouped by
(env, recipe) -- one group = one plot with its four arms (baseline, straighten,
p_reg, both), exactly the grouping run_mpc.sh uses for planning.

Why discovery instead of the hard-coded tables in analysis/linear_probe.py:
those tables hold one recipe per env, so the global / flatten / aggmlp
checkpoints collide on the same output filenames. Here the recipe is read from
each run dir's OWN hydra.yaml (`encoder.projector` + `encoder.agg_type`), and the
arm from the name tokens, so every checkpoint is labelled by what it is.

Recipe labels: global | flatten | aggmlp | aggmean | noproj_<agg>
Arm tokens (the same rules run_mpc.sh's auto-discovery uses):
    cos AND two-thirds -> both;  cos only -> straighten;  two-thirds only -> p_reg;
    otherwise baseline (the name contains _False_).

Usage:
    python analysis/probe_ckpts.py --ckpt-root checkpoints/test
    python analysis/probe_ckpts.py --ckpt-root "$CKPT_ROOT/test" --json
    python analysis/probe_ckpts.py --ckpt-root checkpoints/test --verbose
"""

import argparse
import json
import os
import sys

from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARM_ORDER = ("baseline", "straighten", "p_reg", "both")
RECIPE_ORDER = ("global", "flatten", "aggmlp", "aggmean")


def arm_of(name):
    """Which of the four arms a run-dir name denotes (None if unrecognisable).

    Token rules mirror run_mpc.sh's auto-discovery: a name carrying both
    regularizers is 'both', only cos is 'straighten', only two-thirds is 'p_reg',
    and the unregularized one is the baseline (its name contains _False_ because
    straighten/twothirds are both false).
    """
    has_cos = "cos" in name
    has_tt = ("wothirds" in name) or ("twothirds" in name)
    if has_cos and has_tt:
        return "both"
    if has_cos:
        return "straighten"
    if has_tt:
        return "p_reg"
    if "_False_" in name or name.endswith("_False"):
        return "baseline"
    return None


def recipe_label(projector, agg_type):
    """Short label for how a run pooled its patch tokens."""
    proj = "none" if projector in (None, "", "None") else str(projector)
    agg = "flatten" if agg_type in (None, "", "None") else str(agg_type)
    if proj == "global":
        return "global"
    if proj == "none":
        return "noproj_%s" % agg
    return {"mlp": "aggmlp", "flatten": "flatten", "mean": "aggmean"}.get(agg, "agg%s" % agg)


def load_run_cfg(model_dir):
    cfg_path = os.path.join(model_dir, "hydra.yaml")
    if not os.path.isfile(cfg_path):
        raise RuntimeError("no hydra.yaml in %s" % model_dir)
    return OmegaConf.load(cfg_path)


def describe(model_dir, cfg=None):
    """(env, recipe, arm) for one run dir, from its own config and name."""
    cfg = cfg if cfg is not None else load_run_cfg(model_dir)
    try:
        env = cfg.get("env", {}).get("name")
    except Exception:                                          # noqa: BLE001
        env = None
    try:
        enc = cfg.get("encoder", {})
        recipe = recipe_label(enc.get("projector"), enc.get("agg_type"))
    except Exception:                                          # noqa: BLE001
        recipe = "unknown"
    return env, recipe, arm_of(os.path.basename(os.path.normpath(model_dir)))


def discover(root, envs=None, recipes=None, require_ckpt=True):
    """-> ({ (env, recipe): {arm: abs_dir} }, [(name, reason), ...], all_dirs)

    A directory counts as a checkpoint run dir only if it has hydra.yaml (and,
    unless require_ckpt is False, checkpoints/model_latest.pth): everything else
    in the tree (smoke_*, validate_*, summaries/, pre-multi-seed leftovers) is
    reported as skipped instead of silently polluting a group.
    """
    groups, skipped, all_dirs = {}, [], {}
    if not os.path.isdir(root):
        raise RuntimeError("checkpoint root does not exist: %s" % root)
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if not os.path.isfile(os.path.join(d, "hydra.yaml")):
            continue
        if require_ckpt and not os.path.isfile(
                os.path.join(d, "checkpoints", "model_latest.pth")):
            skipped.append((name, "no checkpoints/model_latest.pth"))
            continue
        try:
            cfg = load_run_cfg(d)
        except Exception as exc:                               # noqa: BLE001
            skipped.append((name, "unreadable hydra.yaml: %s" % exc))
            continue
        env, recipe, arm = describe(d, cfg)
        if arm is None:
            skipped.append((name, "cannot tell which arm from the name"))
            continue
        all_dirs.setdefault(env, {})[name] = dict(recipe=recipe, arm=arm)
        if envs and env not in envs:
            continue
        if recipes and recipe not in recipes:
            continue
        groups.setdefault((env, recipe), {})[arm] = d
    return groups, skipped, all_dirs


def format_table(groups, skipped, root, verbose=False):
    def sort_key(key):
        env, recipe = key
        return (env, RECIPE_ORDER.index(recipe) if recipe in RECIPE_ORDER else 99)

    lines = ["checkpoint root: %s" % root,
             "%-20s %-10s %5s  %s" % ("env", "recipe", "arms", "arm -> run dir")]
    for (env, recipe) in sorted(groups, key=sort_key):
        arms = groups[(env, recipe)]
        missing = [a for a in ARM_ORDER if a not in arms]
        note = "" if not missing else "   MISSING: %s" % ",".join(missing)
        lines.append("%-20s %-10s %5d  %s%s"
                     % (env, recipe, len(arms),
                        ",".join(a for a in ARM_ORDER if a in arms), note))
        if verbose:
            for a in ARM_ORDER:
                if a in arms:
                    lines.append("    %-11s %s" % (a, os.path.basename(arms[a])))
    if skipped:
        shown = ", ".join(n for n, _ in skipped[:6]) + (" ..." if len(skipped) > 6 else "")
        lines.append("skipped %d entr%s (not a run dir with a checkpoint): %s"
                     % (len(skipped), "y" if len(skipped) == 1 else "ies", shown))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Group checkpoint run dirs by (env, recipe)")
    ap.add_argument("--ckpt-root", default=os.path.join(REPO, "checkpoints", "test"))
    ap.add_argument("--envs", default="", help="comma list to keep (default: all found)")
    ap.add_argument("--recipes", default="", help="comma list to keep (default: all found)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--verbose", action="store_true", help="list every arm -> run dir")
    args = ap.parse_args()

    envs = [e for e in args.envs.split(",") if e] or None
    recipes = [r for r in args.recipes.split(",") if r] or None
    groups, skipped, all_dirs = discover(args.ckpt_root, envs, recipes)

    if args.json:
        payload = {
            "root": args.ckpt_root,
            "groups": [{"env": env, "recipe": recipe,
                        "arms": {a: os.path.basename(d)
                                 for a, d in sorted(groups[(env, recipe)].items())},
                        "missing": [a for a in ARM_ORDER if a not in groups[(env, recipe)]]}
                       for (env, recipe) in sorted(groups)],
            "skipped": [{"dir": n, "reason": r} for n, r in skipped],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(format_table(groups, skipped, args.ckpt_root, args.verbose))
    incomplete = [(env, rec) for (env, rec), arms in groups.items()
                  if len(arms) < len(ARM_ORDER)]
    if incomplete:
        print("\n%d group(s) have fewer than %d arms -- the missing arms are named above"
              % (len(incomplete), len(ARM_ORDER)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
