#!/usr/bin/env python
"""analysis/landscape_sweep.py
=============================
The 4-arm action-space loss-landscape sweep used for the paper figures, with
three things the legacy sweep did not have:

1. **The eval episodes, not val episodes.** `plan_targets.pkl` in each
   `plan_outputs_gd_ol/<model>_s<seed>_gH<goal_H>/` dir holds the exact 50
   start/goal pairs that produced the open-loop/closed-loop success rates in
   results/AGG_RESULTS.MD (obs_0, obs_g, state_0/state_g, gt_actions, goal_H=25).
   Those tensors are byte-identical across the four checkpoints of a seed
   (verified in helpers/extract_planner_curves.py's notes), so a grid per arm on
   the same episode index is a *paired* measurement on the planner's own eval
   distribution -- and the same episode indices carry per-episode success flags.
2. **A box that contains the action range.** The four pusht checkpoints' own
   first actions span about +-3.16 (97.5th percentile ~2.4) in normalized units,
   while every legacy grid was swept over +-2 -- so in all 14 cached grids the
   argmin sits ON the wall (see analysis/landscape_metric_audit.py). This sweep
   defaults to +-3.5 and records the wall/censoring statistics with every grid.
3. **Per-step traces.** For every cell the sweep keeps the running-best loss at
   every GD step, so a grid can be read as "what did GD actually achieve at step
   t" (a convergence figure) instead of only "the best value it ever saw".

It is also cheaper than the legacy sweep, though far less than hoped, and the
measured numbers are the point: the initial observation's DINOv2 encoding is
~11% of a rollout forward pass (7 ms of 46 ms on the RTX 3050 Ti), so memoizing
it saves ~5% of a GD step, and batching cells does not work here at all (one
element's autograd graph peaks at ~2.0 GB, and only 3.6 GB is free, so batch 2
already OOMs). The real lever this script provides is therefore not raw speed but
*evidence*: the per-step traces let the sweep run for the same number of GD steps
the planner itself uses, and let a later stage check whether fewer steps would
have given the same picture, instead of guessing.

Correctness is checked, not assumed: `verify` re-sweeps the cached legacy grids
and prints max|new - cached| for the whole grid (and for one row of a cached
grid-13/80-step grid), together with the measured cost, before any figure is
drawn from a new grid; `anchor` then checks the start cell of a new grid against
the loss the planner itself logged there.

Usage
-----
  # 1. reproduce the cached straighten/ep0 grid (must be <= 1e-5) + report speedup
  python analysis/landscape_sweep.py verify --env pusht --arm straighten \
      --legacy-dir analysis_outputs/loss_landscape/grids
  # 1b. is the start cell the loss the planner logged there? (the real gate)
  python analysis/landscape_sweep.py anchor --env pusht --episode 6

  # 2. small design pilot: 4 arms x 1 episode x opt_steps/action_range choices
  python analysis/landscape_sweep.py pilot --env pusht --episode 6

  # 3. the paper sweep: 4 arms x K eval episodes, grid 13, 100 GD steps
  python analysis/landscape_sweep.py sweep --env pusht --episodes 6

Where each command reads and writes (the two gates read inputs that are NOT under
the sweep's own --outdir, so they name their dirs instead of inheriting it):
  pilot / report / sweep   --outdir + --grids-subdir (default analysis_outputs/paper)
  verify                   --legacy-dir (the frozen reference grids)
  anchor                   --outdir/planner_loss_curves.csv (default analysis_outputs)

  # dry run: parse the arguments and stop, without touching a model or a grid --
  # what `run_landscape_night.sh preflight` replays for every stage
  LANDSCAPE_VALIDATE_ARGS=1 python analysis/landscape_sweep.py sweep --env pusht

Outputs (under --outdir, default analysis_outputs/paper)
  grids/<env>_ep<iii>_<arm>_g13_s100_ar3.5.npz   grid + traces + provenance
  landscape_sweep_metrics.csv                    metric battery per grid
"""

import argparse
import csv
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "analysis"), os.path.join(REPO, "helpers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from curvature_analysis import load_model, load_val_dset  # noqa: E402
from planning.objectives import create_objective_fn  # noqa: E402
from landscape_metrics import metrics_for_grid  # noqa: E402
from extract_planner_curves import ARM_LABEL, ARM_ORDER  # noqa: E402

# Checkpoints per env/arm (same dirs as analysis/linear_probe.py MODELS; the
# p-reg-only dir is the "tttwothirds" / "aggtwothirds" run).
MODEL_DIRS = {
    "pusht": {
        "baseline": "checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "checkpoints/test/pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "point_maze": {
        "baseline": "checkpoints/test/umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/umaze_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "both": "checkpoints/test/umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
    },
}

# Which finished planning run supplies plan_targets.pkl for an env (the eval
# episodes are per (env, seed); seed 100 is the reference every figure uses).
PLAN_RUN_DIRS = {
    "pusht": "plan_outputs_gd_ol/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    "point_maze": "plan_outputs_gd_ol/umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
}
DEFAULT_EVAL_SEED = 100
# Cells swept in parallel. Measured on the RTX 3050 Ti (3.6 GB free): the
# autograd graph of ONE cell peaks at ~2.0 GB, so batch 2 already OOMs and the
# default is 1. The batch knob stays because the limit is a property of the
# device (a bigger GPU can use it), and `verify` reports the measured cost so the
# setting is never a guess.
DEFAULT_BATCH = 1
# Where the frozen reference grids and the planner's own logged curves live. Both
# are inputs to a *gate* rather than outputs of a sweep, so the two stages that
# read them name their dirs explicitly (`verify --legacy-dir`, `anchor --outdir`)
# instead of inheriting the sweep's analysis_outputs/paper default -- which is how
# `verify` ended up being handed a --outdir it never declared.
DEFAULT_LEGACY_DIR = os.path.join(REPO, "analysis_outputs", "loss_landscape", "grids")
DEFAULT_CURVES_DIR = os.path.join(REPO, "analysis_outputs")
# `LANDSCAPE_VALIDATE_ARGS=1` makes any command parse its arguments and stop. The
# night script's `preflight` replays every stage's real argv through it, so an
# argument a stage does not declare fails in seconds with the same parser the real
# run uses, instead of ten minutes into the night. (A `verify --outdir` that the
# parser never declared is what motivated it.) The figures script honours the same
# variable.
VALIDATE_ARGS_ENV = "LANDSCAPE_VALIDATE_ARGS"


# ---------------------------------------------------------------------------
# Eval targets (the exact 50 start/goal pairs behind the reported success rates)
# ---------------------------------------------------------------------------
def targets_path(env, seed):
    run = PLAN_RUN_DIRS.get(env)
    if run is None:
        raise SystemExit(f"no planning run dir known for env {env!r}")
    return os.path.join(REPO, f"{run}_s{seed}_gH25", "plan_targets.pkl")


def load_targets(env, seed):
    """{'visual0','proprio0','visualg','propriog','state0','stateg','gt_actions'} as tensors.

    `visual` is stored 0/1 float or uint8 by plan.py depending on the wrapper; both
    are normalised here to the float [0,1] (N,1,H,W) layout that wm.encode_obs
    expects (its own encoder_transform then maps it to the encoder's input range).
    """
    path = targets_path(env, seed)
    if not os.path.isfile(path):
        raise SystemExit(f"missing {os.path.relpath(path, REPO)}")
    with open(path, "rb") as f:
        d = pickle.load(f)

    def img(x):
        v = x["visual"] if isinstance(x, dict) else x
        v = torch.as_tensor(np.asarray(v))
        if v.ndim == 5:                          # (N, 1, H, W, 3) -> (N, 1, 3, H, W)
            v = v.permute(0, 1, 4, 2, 3)
        v = v.float()
        if float(v.max()) > 1.5:
            v = v / 255.0
        return v.contiguous()

    return {
        "visual0": img(d["obs_0"]), "proprio0": torch.as_tensor(d["obs_0"]["proprio"]).float(),
        "visualg": img(d["obs_g"]), "propriog": torch.as_tensor(d["obs_g"]["proprio"]).float(),
        "state0": np.asarray(d["state_0"], dtype=np.float64),
        "stateg": np.asarray(d["state_g"], dtype=np.float64),
        "gt_actions": torch.as_tensor(d["gt_actions"]).float(),
        "goal_H": int(d["goal_H"]),
    }


def select_episodes(n, k, offset=0):
    """k indices spread over n episodes by the legacy spacing rule (no cherry-picking).

    stride = max(1, n // k): for n=50, k=6 that is 0, 8, 16, 24, 32, 40 -- every
    stride-th episode of the eval set, exactly the rule the val-episode landscape
    comparison used, so "which episodes" cannot be tuned after seeing a result.
    """
    k = min(k, n)
    stride = max(1, n // k)
    idx = [min(n - 1, offset + i * stride) for i in range(k)]
    return sorted(set(idx))


def travel_percentile(targets, ep):
    """Where this episode's start->goal distance sits in the eval set (0-100)."""
    d = np.linalg.norm(targets["stateg"] - targets["state0"], axis=1)
    return float(100.0 * np.mean(d <= d[ep]))




# ---------------------------------------------------------------------------
# The sweep itself
# ---------------------------------------------------------------------------
class CachedObsEncoder:
    """`wm.encode_obs` wrapper that answers the repeated initial-frame call from a cache.

    The initial observation of a grid chunk is a constant, yet `wm.rollout`
    re-encodes it (a full DINOv2 ViT pass) at every GD step. Caching it removes
    opt_steps-1 encoder passes per chunk with no numerical change: the observation
    has no gradient, the encoder is frozen in eval mode, and the cached value is
    detached exactly like the constant it is. The cache is cleared per chunk and
    the original method is restored when the chunk is done, so nothing else in the
    process can see a patched model.
    """

    def __init__(self, wm):
        self.wm = wm
        self.orig = wm.encode_obs
        self.cache = {}

    def __enter__(self):
        self.wm.encode_obs = self
        return self

    def __exit__(self, *exc):
        self.wm.encode_obs = self.orig
        self.cache.clear()
        return False

    def clear(self):
        self.cache.clear()

    def __call__(self, obs):
        v = obs["visual"]
        key = (v.data_ptr(), tuple(v.shape), str(v.dtype))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        with torch.no_grad():
            out = self.orig(obs)
        self.cache[key] = out
        return out


def sweep_grid(wm, obs_0, z_obs_g, norm_zero, axs, opt_steps=100, lr=0.1,
               H_latent=5, batch=None, device="cuda", objective_fn=None,
               traces=True, memoize=True, only_rows=None, on_step=None):
    """The grid sweep: fix (a1, a2) of the first latent action, GD-optimize the rest.

    Per cell this reproduces curvature_analysis.loss_landscape exactly, including
    its "best value seen at an evaluation step" convention (the loss after the last
    update is never evaluated). The differences are documented and checked by
    `verify`: cells are swept in batches along a row (Adam is element-wise and the
    cells do not interact, so batching changes only the loss scaling, which is kept
    at sum-over-batch = the single-cell value), and the constant initial-frame
    encoding is memoized.

    Returns a dict:
      grid_loss  (G, G)     the min attainable loss per fixed action
      step0      (G, G)     the loss before any update (the planner's start loss)
      best_trace (G, G, S)  running-best loss per cell at each GD step
      wall_s     float
    """
    objective_fn = objective_fn or create_objective_fn(alpha=1, base=2, mode="last")
    grid = int(len(axs))
    if norm_zero.ndim == 2:
        norm_zero = norm_zero[0]
    act_dim = int(norm_zero.numel())
    H_latent = int(H_latent or 1)
    z_obs_g = {k: (v[:1].expand(obs_0["visual"].shape[0], *v.shape[1:]).contiguous()
                   if v.shape[0] != obs_0["visual"].shape[0] else v)
               for k, v in z_obs_g.items()}

    cells = [(i, j) for i in range(grid) for j in range(grid)
             if only_rows is None or i in set(only_rows)]
    batch = int(batch or len(cells))
    B_all = obs_0["visual"].shape[0]
    grid_loss = np.full((grid, grid), np.nan)
    step0 = np.full((grid, grid), np.nan)
    best_trace = np.full((grid, grid, opt_steps), np.nan) if traces else None
    t0 = time.time()
    for start in range(0, len(cells), batch):
        chunk = cells[start:start + batch]
        B = len(chunk)
        rows = [i for i, _ in chunk]
        cols = [j for _, j in chunk]
        obs_c = {k: v[:1].expand(B, *v.shape[1:]).contiguous() for k, v in obs_0.items()}
        fixed = torch.tensor([[axs[i], axs[j]] for i, j in chunk], dtype=torch.float32,
                             device=device)
        acts = norm_zero.unsqueeze(0).expand(B, H_latent, act_dim).clone()
        acts[:, 0, :2] = fixed
        acts.requires_grad_(True)
        opt = torch.optim.Adam([acts], lr=lr)
        best = torch.full((B,), float("inf"), device=device)
        ctx = CachedObsEncoder(wm) if memoize else None
        if ctx is not None:
            ctx.__enter__()
        try:
            for step in range(opt_steps):
                opt.zero_grad()
                i_z_obses, _ = wm.rollout(obs_0=obs_c, act=acts)
                loss = objective_fn(i_z_obses, z_obs_g)        # (B,)
                with torch.no_grad():
                    vals = loss.detach().cpu().numpy()
                    if step == 0:
                        step0[rows, cols] = vals
                    best = torch.minimum(best, loss.detach())
                    if traces:
                        best_trace[rows, cols, step] = best.cpu().numpy()
                total = loss.mean() * B          # = sum over cells = the B=1 value
                total.backward()
                opt.step()
                with torch.no_grad():
                    acts.data[:, 0, :2] = fixed
                if on_step is not None:
                    on_step(step, float(total.detach()))
        finally:
            if ctx is not None:
                ctx.__exit__()
        grid_loss[rows, cols] = best.cpu().numpy()
    return {"grid_loss": grid_loss, "step0": step0, "best_trace": best_trace,
            "wall_s": time.time() - t0, "n_cells": len(cells), "batch": min(batch, len(cells)),
            "batch_size_used": min(batch, len(cells)), "obs_batch": B_all}


def episode_inputs(wm, train_cfg, val, targets, ep, device):
    """(obs_0, z_obs_g, norm_zero, H_latent, gt_actions) for one eval episode.

    The action normalisation (`norm_zero = -action_mean / action_std`, i.e. the
    normalized real zero action) and the latent horizon come from the checkpoint's
    own train config, exactly as curvature_analysis.loss_landscape derives them.

    Proprio IS normalised here, and that is a deliberate difference from the legacy
    landscape code (which this module otherwise reproduces bit-exactly, see
    `verify`): `plan_targets.pkl` stores the visual already preprocessed (CHW 224,
    the dataset's transform) but proprio in raw physical units, the world model's
    `encode_obs` passes proprio straight into the proprio encoder without
    normalising it, and the planner always feeds `preprocessor.transform_obs`,
    which does `(proprio - proprio_mean) / proprio_std`. Feeding the raw values
    therefore evaluates a *different* objective than the planner optimises:
    measured on pusht seed 100 episode 6, the objective at the planner's own start
    action is

      2.0750 (baseline) / 0.3154 (straightening)  with raw proprio,
      0.3668 (baseline) / 0.7746 (straightening)  with the planner's normalisation,

    against the planner's own logged step-1 losses of 0.3681 / 0.7734. The
    normalised form agrees to 0.2-0.4% (the gap is the two pinned grid cells plus
    float32 logging); the raw form is off by 2.5-5.7x, in opposite directions for
    the two arms. `landscape_sweep.py anchor` re-checks exactly this.
    """
    frameskip = int(train_cfg.frameskip)
    goal_H = int(targets["goal_H"])
    am = val.action_mean.to(device)
    ast = val.action_std.to(device)
    norm_zero = torch.cat([-am / ast] * frameskip)
    H_latent = goal_H // frameskip
    pm = val.proprio_mean.to(device)
    pstd = val.proprio_std.to(device)
    obs_0 = {"visual": targets["visual0"][ep:ep + 1].to(device),
             "proprio": (targets["proprio0"][ep:ep + 1].to(device) - pm) / pstd}
    obs_g = {"visual": targets["visualg"][ep:ep + 1].to(device),
             "proprio": (targets["propriog"][ep:ep + 1].to(device) - pm) / pstd}
    with torch.no_grad():
        z_obs_g = wm.encode_obs(obs_g)
    gt = targets["gt_actions"][ep:ep + 1].to(device)
    return obs_0, z_obs_g, norm_zero, H_latent, gt


def gt_loss_of(wm, obs_0, z_obs_g, gt, objective_fn):
    """The objective at the episode's ground-truth actions: the achievable anchor."""
    with torch.no_grad():
        z_obses, _ = wm.rollout(obs_0=obs_0, act=gt)
        return float(objective_fn(z_obses, z_obs_g).mean())


def grid_filename(env, ep, arm, grid, opt_steps, action_range, tag=""):
    t = f"_{tag}" if tag else ""
    return f"{env}_ep{ep:03d}_{arm}_g{grid}_s{opt_steps}_ar{action_range:g}{t}.npz"


def run_plan(plan, args, model_cache=None, val_cache=None):
    """Execute a list of sweep jobs, saving one .npz per grid (existing = reused).

    `plan` is a list of dicts: arm, episode, opt_steps, action_range, tag.
    Reuse is by filename, so re-running with the same settings costs nothing and a
    name carries every setting that changes the numbers (grid, steps, box).
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    targets = load_targets(args.env, args.eval_seed)
    grids_dir = os.path.join(args.outdir, args.grids_subdir)
    os.makedirs(grids_dir, exist_ok=True)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    model_cache = model_cache if model_cache is not None else {}
    val_cache = val_cache if val_cache is not None else {}
    done, reused = 0, 0
    for job in plan:
        arm = job["arm"]
        ep = job["episode"]
        opt_steps = job.get("opt_steps", args.opt_steps)
        action_range = job.get("action_range", args.action_range)
        tag = job.get("tag", "")
        path = os.path.join(grids_dir, grid_filename(args.env, ep, arm, args.grid,
                                                     opt_steps, action_range, tag))
        if os.path.isfile(path) and not args.force:
            print(f"  [reuse] {os.path.basename(path)}")
            reused += 1
            continue
        if arm not in model_cache:
            model_dir = os.path.join(REPO, MODEL_DIRS[args.env][arm])
            if not os.path.isfile(os.path.join(model_dir, "hydra.yaml")):
                print(f"  [skip] {arm}: no checkpoint at {os.path.relpath(model_dir, REPO)}")
                model_cache[arm] = (None, None)
            else:
                wm, train_cfg = load_model(model_dir, device)
                # the checkpoint stores the training machine's dataset path; point it
                # at the local copy (same fallback the legacy comparison uses)
                from loss_landscape_comparison import redirect_dset_paths
                redirect_dset_paths(train_cfg, args.env, arm)
                val_cache[arm] = load_val_dset(train_cfg)
                model_cache[arm] = (wm, train_cfg)
        wm, train_cfg = model_cache.get(arm, (None, None))
        if wm is None:
            continue
        val = val_cache[arm]
        axs = np.linspace(-action_range, action_range, args.grid)
        obs_0, z_obs_g, norm_zero, H_latent, gt = episode_inputs(
            wm, train_cfg, val, targets, ep, device)
        gl = gt_loss_of(wm, obs_0, z_obs_g, gt, objective_fn)
        out = sweep_grid(wm, obs_0, z_obs_g, norm_zero, axs, opt_steps=opt_steps,
                         lr=args.lr, H_latent=H_latent, batch=args.batch,
                         device=device, objective_fn=objective_fn,
                         memoize=not args.no_memoize)
        np.savez(path, grid_loss=out["grid_loss"].astype(np.float32),
                 step0=out["step0"].astype(np.float32),
                 best_trace=out["best_trace"].astype(np.float32)
                 if out["best_trace"] is not None else np.zeros((1,)),
                 axs=axs.astype(np.float32), gt_loss=np.float32(gl),
                 # the action the planner actually starts from (the normalized real
                 # zero action; the grid axes index its first two dims). It is NEAR
                 # but not AT the origin (~0.04, -0.03 on pusht), so a figure has to
                 # mark it instead of calling the origin "the planner's start".
                 start_action=norm_zero[:2].detach().cpu().numpy().astype(np.float32),
                 # the episode's own first action (the two dims the grid sweeps):
                 # lets a figure show where the true action sits relative to the box
                 gt_first_action=gt[0, 0, :2].detach().cpu().numpy().astype(np.float32),
                 env=args.env, variant=arm, episode_idx=ep, grid=args.grid,
                 opt_steps=opt_steps, lr=args.lr, action_range=action_range,
                 goal_H=int(targets["goal_H"]), batch=out["batch_size_used"],
                 memoize=int(not args.no_memoize), eval_seed=args.eval_seed,
                 frameskip=int(train_cfg.frameskip),
                 model_dir=MODEL_DIRS[args.env][arm],
                 travel_pct=travel_percentile(targets, ep),
                 wall_s=out["wall_s"], tag=tag)
        done += 1
        print(f"  [sweep] {os.path.basename(path)}  {out['wall_s']:.1f}s  "
              f"(batch {out['batch_size_used']}, {args.grid ** 2} cells, "
              f"L_min={np.nanmin(out['grid_loss']):.5g}, L_gt={gl:.5g}, "
              f"argmin_on_edge={int(metrics_for_grid(out['grid_loss'], axs)['argmin_is_edge'])})")
    print(f"\n{done} grid(s) swept, {reused} reused -> "
          f"{os.path.relpath(grids_dir, REPO)}")
    return grids_dir



def write_metrics_csv(grids_dir, out_path):
    """Recompute the metric battery for every grid in a dir (idempotent).

    The CSV is rebuilt from the .npz files rather than accumulated, so adding,
    re-sweeping or dropping a grid can never leave a stale row behind.
    """
    import glob
    from landscape_metrics import METRIC_GROUPS
    files = sorted(glob.glob(os.path.join(grids_dir, "*.npz")))
    if not files:
        print(f"  [note] no grids in {os.path.relpath(grids_dir, REPO)}")
        return None
    rows = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            grid = np.asarray(z["grid_loss"], dtype=np.float64)
            axs = np.asarray(z["axs"], dtype=np.float64) if "axs" in keys else None
            meta = {k: (z[k].item() if getattr(z[k], "shape", ()) == () else z[k])
                    for k in keys if k not in ("grid_loss", "axs", "best_trace")}
            trace = np.asarray(z["best_trace"]) if "best_trace" in keys else None
            step0 = np.asarray(z["step0"], dtype=np.float64) if "step0" in keys else None
        row = {"file": os.path.relpath(p, REPO)}
        row.update(meta)
        row.update(metrics_for_grid(grid, axs))
        if trace is not None and trace.ndim == 3 and trace.shape[2] > 0 and step0 is not None:
            best = np.nanmin(trace, axis=2)
            with np.errstate(invalid="ignore", divide="ignore"):
                gain = (step0 - best) / np.where(step0 > 0, step0, np.nan)
            row["gain_from_init"] = float(np.nanmean(gain))
            row["step0_over_min"] = float(np.nanmean(
                step0 / np.where(np.nanmin(grid) > 0, np.nanmin(grid), np.nan)))
        rows.append(row)
    order = ["env", "variant", "episode_idx", "episode", "tag", "grid",
             "opt_steps", "lr",
             "action_range", "goal_H", "eval_seed", "batch", "memoize",
             "travel_pct", "wall_s", "gt_loss", "gain_from_init", "step0_over_min",
             "file"]
    order += [k for grp in ("legacy_smoothness", "censoring", "init_geometry",
                            "gradient_signal", "descent", "contrast", "basin")
              for k in METRIC_GROUPS[grp]]
    seen, cols = set(), []
    for c in order:
        if c not in seen and any(c in r for r in rows):
            seen.add(c)
            cols.append(c)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else
                        (f"{r[c]:.10g}" if isinstance(r.get(c), (int, float))
                         else str(r.get(c))) for c in cols])
    print(f"Saved {os.path.relpath(out_path, REPO)} ({len(rows)} grids)")
    return out_path



def val_episode_obs(val, ep, goal_H, wm, device):
    """(obs_0, z_obs_g) exactly as curvature_analysis.loss_landscape builds them.

    Used by `verify` only: the cached legacy grids were swept on val episodes, so
    reproducing one requires the val episode's own frames (the eval-episode path
    used everywhere else goes through plan_targets.pkl instead).
    """
    if val.get_seq_length(ep) < goal_H + 1:
        raise SystemExit(f"val episode {ep} is too short for goal_H={goal_H}")
    obs, _, _, _ = val[ep]
    obs_0 = {k: v[0:1].unsqueeze(0).to(device) for k, v in obs.items()}
    obs_g = {k: v[goal_H:goal_H + 1].unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        z_obs_g = wm.encode_obs(obs_g)
    return obs_0, z_obs_g


def _cached_grid_steps(path):
    """The step count a cached grid was swept at, from its own file name.

    80 from `pusht_g13_s80_ep000_straighten.npz`. The cached file IS the reference
    a re-sweep is compared against, so the comparison must use that count and not a
    command-line default that happens to agree today.
    """
    for part in os.path.basename(path).split("_"):
        if len(part) > 1 and part[0] == "s" and part[1:].isdigit():
            return int(part[1:])
    return None


def cached_grid(path):
    """(grid_loss, axs, wall_time_s) from a legacy cached grid file."""
    with np.load(path, allow_pickle=False) as z:
        grid = np.asarray(z["grid_loss"], dtype=np.float64)
        axs = (np.asarray(z["axs"], dtype=np.float64) if "axs" in z.files
               else np.linspace(-2.0, 2.0, grid.shape[0]))
        wall = float(z["wall_time_s"]) if "wall_time_s" in z.files else float("nan")
    return grid, axs, wall


def cmd_verify(args):
    """Re-sweep two cached legacy grids: agreement with them + the real speedup.

    (1) exactness on the cached smoke grid (grid 5, 10 steps): batch 1 without the
        encoding cache IS the legacy algorithm, so it must agree with the cache to
        float32 storage precision; batch 1 + cache and batched + cache then isolate
        the two optimizations one at a time -- "same numbers" is checked, not
        assumed.
    (2) speed on the real cached grid (grid 13, 80 steps): one row of it re-swept,
        compared against that row of the cache, timed against the cached
        wall_time_s (measured for all 169 cells). That per-cell time is what
        decides the budget of the full sweep.
    """
    import glob
    legacy = args.legacy_dir
    if not os.path.isdir(legacy):
        raise SystemExit(f"no cached-grid dir {legacy} -- point --legacy-dir at it")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(REPO, MODEL_DIRS[args.env][args.arm])
    print(f"loading {os.path.relpath(model_dir, REPO)} on {device}")
    wm, train_cfg = load_model(model_dir, device)
    from loss_landscape_comparison import redirect_dset_paths
    redirect_dset_paths(train_cfg, args.env, args.arm)
    val = load_val_dset(train_cfg)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    frameskip = int(train_cfg.frameskip)
    H_latent = int(args.goal_H) // frameskip

    smoke = sorted(glob.glob(os.path.join(legacy,
                                          f"pusht_g5_s10_ep000_{args.arm}.npz")))
    if smoke:
        print("\n== (1) exactness on a cached smoke grid (g5, s10, ar +-2, ep000) ==")
        ref, axs, _ = cached_grid(smoke[0])
        obs_0, z_obs_g = val_episode_obs(val, args.smoke_episode, args.goal_H, wm, device)
        norm_zero = torch.cat([-val.action_mean.to(device) / val.action_std.to(device)]
                              * frameskip)
        for label, memo in (("legacy path (no obs cache)", False),
                            ("with obs cache", True)):
            torch.cuda.reset_peak_memory_stats()
            out = sweep_grid(wm, obs_0, z_obs_g, norm_zero, axs,
                             opt_steps=10, lr=args.lr, H_latent=H_latent, batch=1,
                             device=device, objective_fn=objective_fn, memoize=memo)
            d = float(np.nanmax(np.abs(out["grid_loss"] - ref)))
            print(f"  {label:28s} max|new-cached| = {d:.3e}   ({out['wall_s']:.2f}s for "
                  f"{ref.size} cells, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB)")
        # the batching knob: reported as a measurement, not as an aspiration
        try:
            big = {k: v.expand(2, *v.shape[1:]).contiguous()
                   for k, v in obs_0.items()}
            out = sweep_grid(wm, big, z_obs_g, norm_zero, axs, opt_steps=2, lr=args.lr,
                             H_latent=H_latent, batch=2, device=device,
                             objective_fn=objective_fn)
            print(f"  batch 2 + cache            OK ({out['wall_s']:.2f}s)")
        except Exception as exc:                             # noqa: BLE001
            print(f"  batch 2 + cache            not usable on this device: "
                  f"{type(exc).__name__} {str(exc)[:70]}")
    else:
        print("  [note] no cached g5 smoke grid found; skipping the exactness check")

    # The cached grid IS the reference, so the step count to re-sweep at comes from
    # its own file name and --steps only selects which cached grid to check.
    # (Sweeping at one count while comparing against another would be a silent
    # false failure -- the one thing this gate must never produce.)
    cand = {}
    for cp in sorted(glob.glob(os.path.join(
            legacy, f"pusht_g{args.grid}_s*_ep000_{args.arm}.npz"))):
        cs = _cached_grid_steps(cp)
        if cs is not None:
            cand[cs] = cp
    real = cand.get(args.steps)
    if real is None:
        have = ", ".join(str(s) for s in sorted(cand)) or "none"
        print(f"\n== (2) speed check skipped: no cached g{args.grid} grid for arm "
              f"{args.arm} at {args.steps} steps (cache has: {have}) ==")
        return
    print("\n== (2) speed + row agreement on a cached grid ==")
    print(f"  reference: {os.path.relpath(real, REPO)}")
    refg, axs13, wall13 = cached_grid(real)
    obs_0, z_obs_g = val_episode_obs(val, args.smoke_episode, args.goal_H, wm, device)
    norm_zero = torch.cat([-val.action_mean.to(device) / val.action_std.to(device)]
                          * frameskip)
    row = args.grid // 2
    torch.cuda.reset_peak_memory_stats()
    out = sweep_grid(wm, obs_0, z_obs_g, norm_zero, axs13, opt_steps=args.steps,
                     lr=args.lr, H_latent=H_latent, batch=1, device=device,
                     objective_fn=objective_fn, only_rows=[row], memoize=True)
    d = float(np.nanmax(np.abs(out["grid_loss"][row] - refg[row])))
    cells = refg.shape[0]
    per_cell = out["wall_s"] / cells
    legacy_per_cell = wall13 / (refg.shape[0] ** 2)
    print(f"  row {row} ({cells} cells, {args.steps} GD steps): {out['wall_s']:.1f}s, "
          f"max|new-cached| = {d:.3e}, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    print(f"  {per_cell:.3f} s/cell (new) vs {legacy_per_cell:.3f} s/cell (cached run)"
          f"  ->  {legacy_per_cell / max(per_cell, 1e-9):.2f}x")
    print(f"  whole-grid projection ({cells}x{cells} cells): "
          f"{per_cell * cells ** 2 / 60:.1f} min/grid at {args.steps} steps")
    for s in (20, 30, 100):
        hours_24 = per_cell * (s / max(args.steps, 1)) * cells ** 2 * 24 / 3600
        print(f"    24 grids (4 arms x 6 episodes) at {s:3d} steps: {hours_24:.1f} h")



def _load_grid_npz(path):
    """{meta..., 'grid_loss', 'axs', 'best_trace'} from a sweep grid file."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: (z[k].item() if getattr(z[k], "shape", ()) == () else np.asarray(z[k]))
             for k in z.files}
    d["grid_loss"] = np.asarray(d["grid_loss"], dtype=np.float64)
    if "axs" in d:
        d["axs"] = np.asarray(d["axs"], dtype=np.float64)
    return d


def pilot_report(grids_dir, args, eps):
    """The two design gates, printed as tables from the pilot grids themselves.

    Gate A (the box): for every arm, does +-3.5 put the argmin inside the box
    where +-2 pinned it to the wall, and what does that do to the surface's own
    numbers (wall_ratio, censored share, gain the planner's start still has)?
    Gate B (the step count): using the saturation arm's 100-step grid, how far is
    the surface at 5/20/30 steps from the 100-step surface -- per cell and for the
    argmin position -- which is what decides the sweep's `--opt-steps`.
    """
    def path(arm, ep, steps, ar, tag):
        return os.path.join(grids_dir, grid_filename(args.env, ep, arm, args.grid,
                                                    steps, ar, tag))

    print("\n-- Gate A: box half-range (opt_steps = the pilot step count) --")
    print(f"  {'arm':13s} {'box':>5s} {'argmin_on_edge':>14s} {'wall_ratio':>10s} "
          f"{'censored':>8s} {'center_over_min':>15s} {'L_min/L_gt':>10s}")
    for arm in args.arms:
        for ar in args.pilot_ranges:
            p = path(arm, eps[0], args.pilot_steps[0], ar, "box")
            if not os.path.isfile(p):
                continue
            d = _load_grid_npz(p)
            m = metrics_for_grid(d["grid_loss"], d.get("axs"))
            gt = float(d.get("gt_loss", np.nan))
            print(f"  {arm:13s} {ar:5.2f} {m['argmin_is_edge']:14d} "
                  f"{m['wall_ratio']:10.3f} {m['censored_interior']:8d} "
                  f"{m['center_over_min']:15.3f} "
                  f"{(m['min_loss'] / gt if gt else float('nan')):10.3f}")
    print("  (argmin_on_edge/censored 0 = the optimum is inside the box; "
          "wall_ratio ~1 = the wall is as good as it)")

    print("\n-- Gate B: step saturation on the saturation arm --")
    arm = args.saturation_arm
    p = path(arm, eps[0], max(args.pilot_steps + [args.saturation_steps]),
             args.pilot_ranges[-1], "steps")
    if not os.path.isfile(p):
        print(f"  [skip] no {os.path.basename(p)}")
        return
    d = _load_grid_npz(p)
    tr = d.get("best_trace")
    if tr is None or tr.ndim != 3:
        print("  [skip] that grid has no traces")
        return
    T = tr.shape[2]
    # per cell, that cell's own value after the longest run: the honest reference
    # for "how much has this cell still to fall". Comparing against the scalar
    # global min instead would inflate every short run by the surface's own spread.
    ref_cell = np.asarray(tr[:, :, -1], dtype=np.float64)
    print(f"  {os.path.basename(p)}: {T} steps, L_min={np.nanmin(ref_cell):.6g}, "
          f"own spread (max/min-1)={_rel_spread(ref_cell):.3g}")
    print(f"  {'steps':>6s} {'mean rel gap':>13s} {'max rel gap':>12s} "
          f"{'argmin moves':>13s} {'argmin_on_edge':>15s} {'center_over_min':>16s}")
    a_full = _argmin_of(ref_cell)
    for t in [s for s in args.pilot_steps if s > 0] + [args.saturation_steps]:
        if t > T:
            continue
        g = tr[:, :, t - 1]
        ref = (ref_cell if t < T
               else np.full_like(ref_cell, np.nanmin(ref_cell)) + 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = (g - ref) / np.where(ref > 0, ref, np.nan)
        m = metrics_for_grid(np.where(np.isnan(g), np.inf, g), d.get("axs"))
        print(f"  {t:6d} {np.nanmean(gap):13.3g} {np.nanmax(gap):12.3g} "
              f"{int(_argmin_of(g) != a_full):13d} {m['argmin_is_edge']:15d} "
              f"{m['center_over_min']:16.3f}")
    print("  (rel gap = this cell's value now vs that same cell at the longest run; "
          "the longest run's\n   own row is therefore not a gap at all but the "
          "surface's spread -- it says how much\n   better the best cell is than the "
          "worst, and sets the scale the gaps are read on.\n   'argmin moves' 0 = the "
          "map's minimum is already where it ends up)")
    if args.pilot_steps and max(args.pilot_steps) < T:
        short = _saturation_verdict(tr, max(args.pilot_steps), args.saturation_steps)
        print(f"  verdict: at {max(args.pilot_steps)} steps the cells still sit "
              f"{short['gap'] * 100:.1f}% above where {T} steps leaves them, and the "
              f"argmin {'moves' if short['moves'] else 'is already final'}; "
              f"own spread at {T} is {short['spread'] * 100:.1f}%")


def _rel_spread(g):
    """(max - min) / min over the finite cells of one surface."""
    g = np.asarray(g, dtype=np.float64)
    g = g[np.isfinite(g)]
    if g.size == 0 or np.nanmin(g) <= 0:
        return float("nan")
    return float((np.nanmax(g) - np.nanmin(g)) / np.nanmin(g))


def _saturation_verdict(tr, t_short, t_long):
    """One-line summary of the step-saturation check, for the console log."""
    a = tr[:, :, t_short - 1]
    b = np.asarray(tr[:, :, t_long - 1], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = float(np.nanmean((a - b) / np.where(b > 0, b, np.nan)))
    return {"gap": gap, "spread": _rel_spread(b),
            "moves": _argmin_of(a) != _argmin_of(b)}


def _argmin_of(grid):
    g = np.asarray(grid, dtype=np.float64)
    return tuple(int(i) for i in np.unravel_index(int(np.nanargmin(g)), g.shape))


def episode_list(args):
    """The eval episodes a run sweeps (an explicit --episode-list wins)."""
    n = len(load_targets(args.env, args.eval_seed)["gt_actions"])
    eps = ([int(e) for e in args.episode_list] if args.episode_list
           else select_episodes(n, args.episodes, args.offset))
    for e in eps:
        if not 0 <= e < n:
            raise SystemExit(f"episode {e} out of range for {n} eval episodes")
    return eps


def cmd_pilot(args):
    """One episode x every arm x a grid of (opt_steps, box) choices.

    Answers the two design questions a paper figure cannot dodge:

      * the box, for every arm: the legacy +-2 pinned the argmin to the wall in
        14/14 cached grids, while the four pusht checkpoints' own first actions
        span about +-3.2, so a wider box is a correctness fix, not a tuning knob;
      * the step count: the planner's own convergence curves (Fig. 5) say it
        reaches ~95% of its attainable drop in ~20 steps, but that is per episode,
        not per grid cell -- so the saturation arm is swept at 5/20/30/100 steps
        and the per-cell difference between the short and the long surface is what
        decides the sweep's --opt-steps.
    """
    targets = load_targets(args.env, args.eval_seed)
    eps = episode_list(args)
    print(f"{args.env}: eval seed {args.eval_seed}, {len(eps)} episode(s) {eps} "
          f"(travel percentiles {[round(travel_percentile(targets, e)) for e in eps]})")
    pilot_steps = list(args.pilot_steps)
    box_steps = args.pilot_steps[0] if args.pilot_steps else 30
    plan = [{"arm": arm, "episode": ep, "opt_steps": box_steps, "action_range": ar,
             "tag": "box"}
            for ep in eps for arm in args.arms for ar in args.pilot_ranges]
    sat_steps = sorted({s for s in pilot_steps if s > 0} | {args.saturation_steps})
    if args.saturation_arm:
        plan += [{"arm": args.saturation_arm, "episode": ep, "opt_steps": s,
                  "action_range": args.pilot_ranges[-1], "tag": "steps"}
                 for ep in eps for s in sat_steps]
    grids_dir = run_plan(plan, args)
    write_metrics_csv(grids_dir, os.path.join(args.outdir, "landscape_pilot_metrics.csv"))
    pilot_report(grids_dir, args, eps)


def cmd_sweep(args):
    """The figure sweep: every arm x K eval episodes, one (opt_steps, box)."""
    targets = load_targets(args.env, args.eval_seed)
    eps = episode_list(args)
    print(f"{args.env}: eval seed {args.eval_seed}, {len(eps)} episodes {eps} "
          f"(travel percentiles {[round(travel_percentile(targets, e)) for e in eps]}), "
          f"grid {args.grid}, {args.opt_steps} GD steps, box +-{args.action_range:g}")
    plan = [{"arm": arm, "episode": ep, "opt_steps": args.opt_steps,
             "action_range": args.action_range, "tag": args.tag}
            for ep in eps for arm in args.arms]
    grids_dir = run_plan(plan, args)
    write_metrics_csv(grids_dir, os.path.join(args.outdir, "landscape_sweep_metrics.csv"))



def cmd_anchor(args):
    """Is a grid's start cell the loss the planner itself logged there?

    The strongest correctness check available for a landscape figure: the planner
    logs its objective *before* its first update (`plan_0/loss`, one value per eval
    episode) on the action sequence it starts from, and `episode_inputs` builds the
    same observation and the same `norm_zero` sequence. Evaluating the objective on
    that sequence must reproduce the logged value; if it does not, the surface is
    being computed on a different objective than the planner optimises -- which is
    exactly the bug the unnormalised proprio channel caused (a 2.5-5.7x error in
    inconsistent directions, see `episode_inputs`), and a metric claim like
    "L_min/L_gt" or "the landscape got flatter" cannot be read off such a surface.

    Prints one row per arm: the planner's logged step-1 loss, the value computed
    here, and the relative difference. Run it before trusting a sweep.
    """
    import csv as _csv

    curves = os.path.join(args.outdir, "planner_loss_curves.csv")
    if not os.path.isfile(curves):
        raise SystemExit(f"missing {os.path.relpath(curves, REPO)} -- either run:\n"
                         f"  python helpers/extract_planner_curves.py --envs {args.env}\n"
                         f"or point --outdir at the dir that holds it (it lives in "
                         f"{os.path.relpath(DEFAULT_CURVES_DIR, REPO)}, not in the "
                         f"figures' dir)")
    ref = {}
    with open(curves, newline="") as f:
        for r in _csv.DictReader(f):
            if (r["env"] == args.env and r["arm"] in ARM_ORDER
                    and int(r["seed"]) == args.eval_seed
                    and int(r["episode"]) == args.episode
                    and int(r["step"]) == 1):
                ref[r["arm"]] = float(r["loss"])
    if not ref:
        raise SystemExit(f"no rows for {args.env} seed {args.eval_seed} "
                         f"episode {args.episode} in planner_loss_curves.csv")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    targets = load_targets(args.env, args.eval_seed)
    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    print(f"{args.env}, eval seed {args.eval_seed}, episode {args.episode}: the "
          "planner's own logged step-1 loss vs the same start action here")
    print(f"  {'arm':13s} {'planner L@1':>12s} {'here':>12s} {'rel diff':>10s}")
    worst = 0.0
    for arm in (args.arms or ARM_ORDER):
        if arm not in ref:
            continue
        model_dir = os.path.join(REPO, MODEL_DIRS[args.env][arm])
        if not os.path.isfile(os.path.join(model_dir, "hydra.yaml")):
            print(f"  {arm:13s} [skip] no checkpoint")
            continue
        wm, train_cfg = load_model(model_dir, device)
        from loss_landscape_comparison import redirect_dset_paths
        redirect_dset_paths(train_cfg, args.env, arm)
        val = load_val_dset(train_cfg)
        obs_0, z_obs_g, norm_zero, H_latent, _ = episode_inputs(
            wm, train_cfg, val, targets, args.episode, device)
        act = norm_zero.unsqueeze(0).expand(1, H_latent, norm_zero.numel()).clone()
        with torch.no_grad():
            z_obses, _ = wm.rollout(obs_0=obs_0, act=act)
            here = float(objective_fn(z_obses, z_obs_g).mean())
        rel = (here - ref[arm]) / ref[arm] * 100.0
        worst = max(worst, abs(rel))
        print(f"  {arm:13s} {ref[arm]:12.4f} {here:12.4f} {rel:9.2f}%")
        del wm
        torch.cuda.empty_cache()
    print(f"\n  worst |rel diff| = {worst:.2f}%  (the two pinned grid dims move the "
          "action by ~0.04, so a few tenths of a percent is the expected floor)")
    if worst > 2.0:
        print("  [WARN] the grid is not evaluating the planner's objective -- do not "
              "read metrics off these surfaces")
        sys.exit(1)


def cmd_report(args):
    """Re-print the pilot's two design gates from grids already on disk.

    No model, no GPU, no sweep: the gates live at the end of `pilot`, and this
    re-derives them from the npz files alone so a decision (the box, the step
    count) can be re-read -- or re-read after the gate maths is corrected -- with
    the grids that were already paid for.
    """
    grids_dir = os.path.join(args.outdir, args.grids_subdir)
    if not os.path.isdir(grids_dir):
        raise SystemExit(f"no grid dir {os.path.relpath(grids_dir, REPO)}")
    n = len(sorted(f for f in os.listdir(grids_dir) if f.endswith(".npz")))
    print(f"report from {os.path.relpath(grids_dir, REPO)} ({n} grids), "
          f"box range(s) {args.pilot_ranges}, steps {args.pilot_steps} "
          f"+ {args.saturation_steps}")
    pilot_report(grids_dir, args, episode_list(args))


def add_sweep_args(ap, default_subdir):
    ap.add_argument("--env", default="pusht", choices=sorted(MODEL_DIRS))
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER,
                    help=f"arms to sweep (default: {' '.join(ARM_ORDER)})")
    ap.add_argument("--grid", type=int, default=13, help="cells per axis (default 13)")
    ap.add_argument("--opt-steps", type=int, default=100,
                    help="GD steps per cell (default 100 = the planner's own budget)")
    ap.add_argument("--action-range", type=float, default=3.5,
                    help="box half-range in normalized action units (default 3.5, "
                         "which contains the checkpoints' own first actions +-3.16; "
                         "the legacy default 2 pins the argmin to the wall)")
    ap.add_argument("--lr", type=float, default=0.1, help="Adam lr (default 0.1)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"cells swept in parallel (default {DEFAULT_BATCH})")
    ap.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED,
                    help=f"which planning seed's eval episodes to sweep "
                         f"(default {DEFAULT_EVAL_SEED})")
    ap.add_argument("--episodes", type=int, default=6,
                    help="how many eval episodes, evenly spaced (default 6)")
    ap.add_argument("--offset", type=int, default=0,
                    help="shift the spacing rule (default 0)")
    ap.add_argument("--episode-list", nargs="+", default=None,
                    help="explicit eval episode indices (overrides --episodes)")
    ap.add_argument("--tag", default="", help="suffix for the grid filenames")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs", "paper"))
    ap.add_argument("--grids-subdir", default=default_subdir,
                    help=f"grid dir under --outdir (default {default_subdir})")
    ap.add_argument("--force", action="store_true",
                    help="re-sweep grids whose file already exists")
    ap.add_argument("--no-memoize", action="store_true",
                    help="disable the initial-frame encoding cache (slower; the "
                         "verify command uses it as the reference path)")
    ap.add_argument("--device", default="cuda")


def add_pilot_args(ap):
    ap.add_argument("--pilot-steps", nargs="+", type=int, default=[30],
                    help="opt_steps for the box grids (default 30; the first value "
                         "is used for the 4-arm box comparison)")
    ap.add_argument("--pilot-ranges", nargs="+", type=float, default=[2.0, 3.5],
                    help="action_range values to compare (default 2.0 3.5)")
    ap.add_argument("--saturation-arm", default="straighten", choices=ARM_ORDER + [""],
                    help="arm swept at several step counts to decide --opt-steps "
                         "(default straighten; empty disables the check)")
    ap.add_argument("--saturation-steps", type=int, default=100,
                    help="longest step count of the saturation check (default 100)")


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="4-arm action-space loss-landscape sweep on the eval episodes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="re-sweep cached grids: agreement + speedup")
    p.add_argument("--env", default="pusht", choices=sorted(MODEL_DIRS))
    p.add_argument("--arm", default="straighten", choices=ARM_ORDER)
    p.add_argument("--grid", type=int, default=13)
    p.add_argument("--steps", type=int, default=80,
                   help="which cached grid to check: the file name carries the "
                        "step count it was swept at (default 80 = the cached "
                        "pusht_g13_s80 grid), and the re-sweep uses that count")
    p.add_argument("--legacy-dir", default=DEFAULT_LEGACY_DIR,
                   help="directory holding the cached reference grids (default "
                        f"{os.path.relpath(DEFAULT_LEGACY_DIR, REPO)}); this is the "
                        "one stage whose --outdir is NOT analysis_outputs/paper, "
                        "which is why it names its own dir")
    p.add_argument("--smoke-episode", type=int, default=0,
                   help="val episode of the cached smoke grid (default 0)")
    p.add_argument("--goal_H", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("pilot", help="one episode x arms x (opt_steps, box) choices")
    add_sweep_args(p, "grids_pilot")
    add_pilot_args(p)
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("report", help="re-print the pilot's two design gates from "
                                      "grids already on disk (no GPU)")
    add_sweep_args(p, "grids_pilot_fixed")
    add_pilot_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sweep", help="4 arms x K eval episodes (the paper sweep)")
    add_sweep_args(p, "grids")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("anchor", help="check the start cell against the planner's "
                                      "own logged first loss (run before a sweep)")
    p.add_argument("--env", default="pusht", choices=sorted(MODEL_DIRS))
    p.add_argument("--arms", nargs="+", default=None, choices=ARM_ORDER)
    p.add_argument("--episode", type=int, default=6,
                   help="eval episode index (default 6, the pilot's episode)")
    p.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    p.add_argument("--outdir", default=DEFAULT_CURVES_DIR,
                   help="dir holding planner_loss_curves.csv (default "
                        f"{os.path.relpath(DEFAULT_CURVES_DIR, REPO)}; the figures' "
                        "--outdir is analysis_outputs/paper, this stage reads the "
                        "curves that analysis_outputs/ holds)")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_anchor)
    args = ap.parse_args(argv)
    if os.environ.get(VALIDATE_ARGS_ENV) == "1":
        print(f"[validate-args] {args.cmd}: argv OK ({len(argv)} arguments, "
              f"no work performed)")
        raise SystemExit(0)
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

