#!/usr/bin/env python
"""helpers/extract_planner_curves.py
===================================
Recover the GD planner's OWN optimization curves from the offline wandb
datastores of planning runs that have ALREADY been done.

Why this exists
---------------
`planning/gd.py` logs `{logging_prefix}/loss` at every GD step of every planning
call. For the open-loop runs in `plan_outputs_gd_ol/<model>_s<seed>_gH<goal_H>/`
(`plan.py` chunks the 50 eval episodes with chunk_size=1) that is

    50 episodes x opt_steps(100) = 5000 records of `plan_0/loss`

per run, i.e. a full planner-convergence experiment (4 checkpoints x 3 seeds, on
the *same 50 eval episodes* whose success rates are quoted in
results/AGG_RESULTS.MD) that costs nothing to read: no GPU, no planning, no new
checkpoint. It also cross-checks itself, because the same datastore holds the
per-episode `mpc/success_rate` records and the `final_eval/success_rate` summary
the published table was aggregated from.

What one loss value is
----------------------
`gd.py` logs `total_loss = loss.mean() * n_evals`. With chunk_size=1 the 50
episodes are planned one at a time, so n_evals=1 and the logged value IS one
episode's objective at that step. The objective is
`planning.objectives.create_objective_fn(alpha=1, base=2, mode=staged)`, and
which branch of `staged` runs depends on the `step` argument the planner passes:

  `MPCPlanner.plan` calls its sub-planner with `step=self.iter` (planning/mpc.py),
  i.e. the MPC iteration index, and `objective_fn_staged` uses the terminal-frame
  objective while `step < T-1` and the base-2 weighted full-horizon objective from
  `step >= T-1` (T = the rollout's latent horizon = goal_H // frameskip = 5 here).

All four checkpoints' runs behind results/AGG_RESULTS.MD set `planner.max_iter=1`
(verified in each run's .hydra/config.yaml and overrides.yaml), so the planner
only ever calls the objective with step=0 < T-1 = 4: **the objective these curves
are logged under is `objective_fn_last`, the terminal-frame MSE with alpha=1**
(same as `planning/objectives.create_objective_fn(alpha=1, base=2, mode="last")`,
where `base` is unused). An earlier version of this docstring claimed the planner
calls the objective without `step` and therefore gets `objective_fn_all`; that was
wrong -- `planning/mpc.py` passes `step=self.iter`, and a run configured with
`max_iter >= goal_H//frameskip` would switch objectives part-way through its GD
loop, which is a caveat to check before comparing such runs with these.

Only the SHAPE of these curves is comparable across checkpoints (different
checkpoints define different latent spaces, so absolute levels differ); `loss_rel`
in the CSV is therefore the curve divided by its own step-1 value, i.e. the
scale-free "fraction of the initial error left" view.

Outputs (default --outdir analysis_outputs)
-------------------------------------------
  planner_loss_curves.csv   one row per (planner, env, arm, seed, episode, step)
  planner_run_meta.csv      one row per run: settings, step-1/final/min loss and
                            the per-episode success mean from the same datastore
  planner_arm_summary.csv   one row per (planner, env, arm): across-seed means of
                            the per-run numbers + the published success rate

Usage
-----
  python helpers/extract_planner_curves.py                  # gd_ol, every env found
  python helpers/extract_planner_curves.py --planner-dir plan_outputs_gd_mpc
  python helpers/extract_planner_curves.py --envs pusht --outdir analysis_outputs
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Arm vocabulary. This is the ONE place the four arms' run-dir names, labels and
# plot styles are defined; analysis/landscape_paper_figure.py imports them, so a
# figure can never disagree with a CSV about which run is which arm.
# ---------------------------------------------------------------------------
MODELS = {
    "pusht": {
        "baseline": "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "point_maze": {
        "baseline": "umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "straighten": "umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "p_reg": "umaze_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "both": "umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
    },
    "point_maze_medium": {
        "baseline": "medium_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06",
        "straighten": "medium_cos1e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06",
        "p_reg": "medium_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "both": "medium_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
    },
    "wall": {
        "baseline": "wall_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "wall_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "wall_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "wall_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
}

ARM_ORDER = ["baseline", "straighten", "p_reg", "both"]
ARM_LABEL = {
    "baseline": "baseline",
    "straighten": "straightening",
    "p_reg": "p-reg",
    "both": "both",
}
ARM_COLOR = {
    "baseline": "#4d4d4d",
    "straighten": "#1f77b4",
    "p_reg": "#d62728",
    "both": "#2ca02c",
}
ARM_MARKER = {"baseline": "o", "straighten": "s", "p_reg": "^", "both": "D"}

# Names other (older) scripts used for the same arm, so grids/CSVs produced by
# them can be pooled with the four-arm runs. Example: the legacy
# analysis/loss_landscape_comparison.py called the
# aggcos1e-1 + aggtwothirds5e-2 checkpoint "p_reg_straighten", which IS the
# "both" arm of the four-arm vocabulary (straightening + p-reg).
ARM_ALIAS = {
    "p_reg_straighten": "both",
    "straighten_only": "straighten",
    "baseline_ctx": "baseline",
    "no_straightening": "baseline",
}


def canonical_arm(name):
    """Map any historical arm name onto the four canonical arm names."""
    return ARM_ALIAS.get(str(name), str(name))




def classify_model(model_name):
    """(env, arm) for a run-dir model name, or (None, None) if unrecognised.

    Exact match against MODELS only: each directory is a finished run, so a fuzzy
    match is never needed and never wanted. Unknown names are reported and
    skipped rather than silently bucketed into the nearest arm.
    """
    for env, arms in MODELS.items():
        for arm, name in arms.items():
            if model_name == name:
                return env, arm
    return None, None


def parse_run_dir(run_dir):
    """(model_name, seed, goal_H) from a `plan_outputs_*` run dir name."""
    base = os.path.basename(os.path.normpath(run_dir))
    seed = goal_h = None
    for p in base.split("_"):
        if len(p) > 1 and p[0] == "s" and p[1:].isdigit():
            seed = int(p[1:])
        elif p.startswith("gH") and p[2:].isdigit():
            goal_h = int(p[2:])
    if seed is None or goal_h is None:
        return None, None, None
    suffix = f"_s{seed}_gH{goal_h}"
    if not base.endswith(suffix):
        return None, None, None
    return base[: -len(suffix)], seed, goal_h


def wandb_datastore(run_dir):
    """Path of the .wandb datastore inside one run dir, or None."""
    pats = [
        os.path.join(run_dir, "wandb", "offline-run-*", "run-*.wandb"),
        os.path.join(run_dir, "wandb", "offline-run-*", "*.wandb"),
        os.path.join(run_dir, "wandb", "run-*", "run-*.wandb"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return max(hits, key=os.path.getmtime)   # newest, if a dir was re-run
    return None


def load_history(path):
    """All history records of one .wandb datastore, as a list of dicts.

    wandb is imported lazily, and only its internal datastore reader is used:
    this reads a file that is already on disk, starts no run and syncs nothing.
    """
    try:
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2 as pb
    except ImportError as exc:                          # pragma: no cover
        raise RuntimeError(
            "reading .wandb datastores needs the wandb package (used here only "
            f"as an offline file reader): {exc}")

    ds = DataStore()
    ds.open_for_scan(path)
    out = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:                                # end of datastore
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        d = {}
        for item in rec.history.item:
            if item.key:
                key = item.key
            elif item.nested_key:
                key = ".".join(item.nested_key)
            else:
                continue
            d[key] = json.loads(item.value_json) if item.value_json else None
        out.append(d)
    return out


def run_settings(run_dir):
    """The subset of the run's own hydra config that identifies the protocol."""
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    from omegaconf import OmegaConf

    def get(d, k, default=None):
        try:
            return d.get(k, default)
        except Exception:                                    # noqa: BLE001
            return default

    cfg = OmegaConf.load(cfg_path)
    planner = get(cfg, "planner", {}) or {}
    sub = get(planner, "sub_planner", {}) or {}
    obj = get(cfg, "objective", {}) or {}
    return {
        "n_evals": get(cfg, "n_evals"),
        "goal_source": get(cfg, "goal_source"),
        "goal_H": get(cfg, "goal_H"),
        "seed_cfg": get(cfg, "seed"),
        "chunk_size": get(cfg, "chunk_size"),
        "objective_mode": get(obj, "mode"),
        "objective_alpha": get(obj, "alpha"),
        "objective_base": get(obj, "base"),
        "planner_max_iter": get(planner, "max_iter"),
        "lr": get(sub, "lr"),
        "opt_steps": get(sub, "opt_steps"),
        "sample_type": get(sub, "sample_type"),
        "action_noise": get(sub, "action_noise"),
        "planner_optimizer": get(sub, "optimizer"),
        "use_cosine_scheduler": get(sub, "use_cosine_scheduler"),
    }



def segments_from_history(history, loss_key="plan_0/loss"):
    """Split a flat history into per-planning-call curves.

    `gd.py` logs `{prefix}/loss` once per GD step and the shared `step` counter
    restarts at 1 for every planning call (chunk). A new curve therefore starts
    whenever a loss is logged with step == 1, or when the counter goes backwards
    -- which is how the 5000 records of one open-loop run become 50 curves of 100
    steps. Per-curve success flags come from the `mpc/success_rate` records, which
    the evaluator logs after the planning call they belong to.
    """
    segments = []
    final_eval_success = None
    for rec in history:
        if loss_key in rec:
            step = rec.get("step")
            new = (step == 1) or (not segments) or (
                step is not None and step <= segments[-1]["last_step"])
            if new:
                segments.append({"losses": [], "success": [], "last_step": 0})
            segments[-1]["losses"].append(float(rec[loss_key]))
            if step is not None:
                segments[-1]["last_step"] = int(step)
        if "mpc/success_rate" in rec and segments:
            segments[-1]["success"].append(float(rec["mpc/success_rate"]))
        if "final_eval/success_rate" in rec:
            final_eval_success = float(rec["final_eval/success_rate"])
    return segments, final_eval_success


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def steps_to_gain(losses, frac):
    """First step at which `frac` of the curve's total drop (L_1 -> L_min) is reached.

    A step count is the only convergence number that is comparable across
    checkpoints without any normalisation: it needs no assumption about the latent
    scale, only that "the same error reduction" means the same thing in each. 1 is
    returned when the curve starts at its minimum, and the last step when the
    target is never reached (so the mean is a lower bound of the truth, never an
    optimistic number).
    """
    lo = min(losses)
    hi = losses[0]
    if hi <= lo:
        return 1
    target = hi - frac * (hi - lo)
    for i, v in enumerate(losses):
        if v <= target:
            return i + 1
    return len(losses)


def extract_run(run_dir, planner, goal_H=None):
    """(curve_rows, meta_row) for one finished planning run dir."""
    ds_path = wandb_datastore(run_dir)
    if ds_path is None:
        print(f"  [skip] no .wandb datastore under {run_dir}")
        return [], None
    model, seed, gh = parse_run_dir(run_dir)
    if model is None:
        print(f"  [skip] cannot parse run dir name: {run_dir}")
        return [], None
    env, arm = classify_model(model)
    if env is None:
        print(f"  [skip] model {model!r} is in no MODELS arm table (unknown "
              f"checkpoint) -- {run_dir}")
        return [], None
    if goal_H is not None and gh != goal_H:
        return [], None

    history = load_history(ds_path)
    segments, final_eval_success = segments_from_history(history)
    segments = [s for s in segments if s["losses"]]
    if not segments:
        print(f"  [skip] no plan_0/loss records in {ds_path}")
        return [], None

    settings = run_settings(run_dir)
    rows = []
    for ep, seg in enumerate(segments):
        base = seg["losses"][0]
        for i, loss in enumerate(seg["losses"]):
            rows.append({
                "planner": planner, "env": env, "arm": arm, "model": model,
                "seed": seed, "goal_H": gh, "episode": ep, "step": i + 1,
                "loss": loss, "loss_rel": (loss / base) if base else None,
                "episode_success": (seg["success"][0] if seg["success"] else None),
            })

    meta = {
        "planner": planner, "env": env, "arm": arm, "model": model,
        "seed": seed, "goal_H": gh, "n_episodes": len(segments),
        "n_steps": max(len(s["losses"]) for s in segments),
        "loss_step1": _mean([s["losses"][0] for s in segments]),
        "loss_step10": _mean([s["losses"][min(9, len(s["losses"]) - 1)] for s in segments]),
        "loss_step50": _mean([s["losses"][min(49, len(s["losses"]) - 1)] for s in segments]),
        "loss_final": _mean([s["losses"][-1] for s in segments]),
        "loss_min": _mean([min(s["losses"]) for s in segments]),
        "steps_to_90pct": _mean([steps_to_gain(s["losses"], 0.90) for s in segments]),
        "steps_to_95pct": _mean([steps_to_gain(s["losses"], 0.95) for s in segments]),
        "steps_to_99pct": _mean([steps_to_gain(s["losses"], 0.99) for s in segments]),
        "episode_success_mean": _mean([s["success"][0] for s in segments if s["success"]]),
        "final_eval_success_rate": final_eval_success,
        "run_dir": os.path.relpath(run_dir, REPO),
    }
    meta.update({k: settings.get(k) for k in (
        "opt_steps", "lr", "sample_type", "action_noise", "objective_mode",
        "n_evals", "chunk_size", "planner_max_iter", "goal_source")})
    # A run whose protocol differs from the others' would make the curves
    # incomparable with them, so say so loudly instead of quietly averaging it in.
    if settings:
        bad = [f"{k}={settings[k]!r} (expected {v!r})"
               for k, v in (("objective_mode", "staged"), ("sample_type", "zero"),
                            ("opt_steps", 100), ("action_noise", 0))
               if settings.get(k) is not None and settings.get(k) != v]
        if bad:
            print(f"  [warn] {os.path.basename(run_dir)} protocol differs from the "
                  f"reference protocol: {', '.join(bad)}")
    return rows, meta



# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------
CURVE_COLUMNS = ["planner", "env", "arm", "model", "seed", "goal_H", "episode",
                 "step", "loss", "loss_rel", "episode_success"]

META_COLUMNS = ["planner", "env", "arm", "model", "seed", "goal_H",
                "n_episodes", "n_steps", "opt_steps", "lr", "sample_type",
                "action_noise", "objective_mode", "n_evals", "chunk_size",
                "planner_max_iter", "goal_source", "loss_step1", "loss_step10",
                "loss_step50", "loss_final", "loss_min", "steps_to_90pct",
                "steps_to_95pct", "steps_to_99pct", "episode_success_mean",
                "final_eval_success_rate", "run_dir"]

SUMMARY_COLUMNS = ["planner", "env", "arm", "label", "n_seeds", "seeds",
                   "opt_steps", "lr", "loss_step1_mean", "loss_step1_std",
                   "loss_step10_mean", "loss_step50_mean", "loss_final_mean",
                   "loss_final_std", "loss_reduction_mean",
                   "steps_to_95pct_mean", "steps_to_95pct_std",
                   "success_rate_mean", "success_rate_std",
                   "file_success_rate_mean"]


def summarize_arms(meta_rows):
    """One row per (planner, env, arm): across-seed means of the per-run numbers.

    `loss_reduction = loss_final / loss_step1` (lower = the planner got closer to
    its own attainable loss) and `steps_to_95pct` are the two numbers that survive
    the different latent scales of the four checkpoints; the absolute loss levels
    stay in planner_run_meta.csv for completeness only.
    """
    groups = {}
    for m in meta_rows:
        groups.setdefault((m["planner"], m["env"], m["arm"]), []).append(m)
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    out = []
    for (planner, env, arm), rows in sorted(
            groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 9))):
        rows = sorted(rows, key=lambda r: r["seed"])
        red = [(r["loss_final"] / r["loss_step1"]
                if r["loss_final"] is not None and r["loss_step1"] else None)
               for r in rows]
        out.append({
            "planner": planner, "env": env, "arm": arm, "label": ARM_LABEL[arm],
            "n_seeds": len(rows),
            "seeds": " ".join(str(r["seed"]) for r in rows),
            "opt_steps": rows[0].get("opt_steps"), "lr": rows[0].get("lr"),
            "loss_step1_mean": _mean([r["loss_step1"] for r in rows]),
            "loss_step1_std": _std([r["loss_step1"] for r in rows]),
            "loss_step10_mean": _mean([r["loss_step10"] for r in rows]),
            "loss_step50_mean": _mean([r["loss_step50"] for r in rows]),
            "loss_final_mean": _mean([r["loss_final"] for r in rows]),
            "loss_final_std": _std([r["loss_final"] for r in rows]),
            "loss_reduction_mean": _mean(red),
            "steps_to_95pct_mean": _mean([r["steps_to_95pct"] for r in rows]),
            "steps_to_95pct_std": _std([r["steps_to_95pct"] for r in rows]),
            "success_rate_mean": _mean([r["episode_success_mean"] for r in rows]),
            "success_rate_std": _std([r["episode_success_mean"] for r in rows]),
            "file_success_rate_mean": _mean([r["final_eval_success_rate"] for r in rows]),
        })
    return out


def _fmt(value, spec=".6g"):
    """Format a cell for a CSV/print; None and NaN become an empty string."""
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:                                          # NaN
        return ""
    return format(f, spec)


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c), ".10g") if isinstance(r.get(c), (int, float))
                        else ("" if r.get(c) is None else str(r.get(c)))
                        for c in columns])
    return path


def print_summary(summary):
    print(f"\n{'env':18s} {'arm':13s} {'seeds':13s} {'L@1':>10s} {'L@50':>10s} "
          f"{'L@100':>10s} {'L100/L1':>8s} {'t95':>6s} {'success':>8s}")
    for r in summary:
        print(f"{r['env']:18s} {r['label']:13s} {r['seeds']:13s} "
              f"{_fmt(r['loss_step1_mean']):>10s} {_fmt(r['loss_step50_mean']):>10s} "
              f"{_fmt(r['loss_final_mean']):>10s} "
              f"{_fmt(r['loss_reduction_mean'], '.3f'):>8s} "
              f"{_fmt(r['steps_to_95pct_mean'], '.1f'):>6s} "
              f"{_fmt(r['success_rate_mean'], '.3f'):>8s}")
    print("(L@1/L@50/L@100 = across-seed mean loss at that step; L100/L1 = fraction "
          "of the initial error still\n there at step 100, so lower is better; t95 = "
          "mean GD step at which 95% of each episode's own\n total drop is reached; "
          "success = per-episode success mean from the same run's datastore.\n"
          "Absolute L values are NOT comparable across arms -- only L100/L1 and t95 are.)")



def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract the GD planner's own per-step loss curves from the "
                    "offline wandb datastores of finished planning runs")
    ap.add_argument("--planner-dir", default="plan_outputs_gd_ol",
                    help="directory of run dirs (default plan_outputs_gd_ol)")
    ap.add_argument("--planner", default=None,
                    help="label for the planner in the CSVs (default: the dir name)")
    ap.add_argument("--envs", nargs="*", default=None,
                    help="only these envs (default: every env found in the dir)")
    ap.add_argument("--goal-H", type=int, default=25,
                    help="only runs with this planning horizon (default 25)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--no-per-episode", action="store_true",
                    help="skip planner_loss_curves.csv (the ~60k-row per-step file)")
    return ap.parse_args()


def main():
    args = parse_args()
    planner_dir = (args.planner_dir if os.path.isabs(args.planner_dir)
                   else os.path.join(REPO, args.planner_dir))
    planner = args.planner or os.path.basename(os.path.normpath(planner_dir))
    if not os.path.isdir(planner_dir):
        raise SystemExit(f"no such planner dir: {planner_dir}")
    os.makedirs(args.outdir, exist_ok=True)

    run_dirs = sorted(d for d in glob.glob(os.path.join(planner_dir, "*_s*_gH*"))
                      if os.path.isdir(d))
    print(f"{planner}: {len(run_dirs)} run dirs under "
          f"{os.path.relpath(planner_dir, REPO)}")

    # Filter by env/goal_H BEFORE reading a datastore: reading is cheap but not
    # free (36 runs x 5000 records to parse), and an arm this run is not about
    # should not produce a protocol warning either.
    kept = []
    for d in run_dirs:
        model, _seed, gh = parse_run_dir(d)
        if model is None:
            print(f"  [skip] cannot parse run dir name: {os.path.basename(d)}")
            continue
        if args.goal_H is not None and gh != args.goal_H:
            continue
        env, arm = classify_model(model)
        if env is None:
            print(f"  [skip] {os.path.basename(d)}: checkpoint is in no arm table")
            continue
        if args.envs and env not in args.envs:
            continue
        kept.append(d)

    curve_rows, meta_rows = [], []
    for d in kept:
        rows, meta = extract_run(d, planner, goal_H=args.goal_H)
        if meta is None:
            continue
        curve_rows += rows
        meta_rows.append(meta)
        print(f"  [{meta['env']:18s} {meta['arm']:11s} seed {meta['seed']}] "
              f"{meta['n_episodes']} episodes x {meta['n_steps']} steps, "
              f"success={_fmt(meta['episode_success_mean'], '.3f')} "
              f"(final_eval={_fmt(meta['final_eval_success_rate'], '.3f')})")
    if not meta_rows:
        raise SystemExit("no usable runs; check --planner-dir / --envs / --goal-H")

    summary = summarize_arms(meta_rows)
    meta_path = write_csv(os.path.join(args.outdir, "planner_run_meta.csv"),
                          META_COLUMNS, meta_rows)
    sum_path = write_csv(os.path.join(args.outdir, "planner_arm_summary.csv"),
                         SUMMARY_COLUMNS, summary)
    print(f"\nSaved per-run metadata -> {os.path.relpath(meta_path, REPO)}")
    print(f"Saved per-arm summary  -> {os.path.relpath(sum_path, REPO)}")
    if not args.no_per_episode:
        curve_path = write_csv(os.path.join(args.outdir, "planner_loss_curves.csv"),
                               CURVE_COLUMNS, curve_rows)
        print(f"Saved per-step curves  -> {os.path.relpath(curve_path, REPO)} "
              f"({len(curve_rows)} rows)")
    print_summary(summary)


if __name__ == "__main__":
    main()
