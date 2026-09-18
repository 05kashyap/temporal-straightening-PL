#!/usr/bin/env python
"""analysis/loss_landscape_comparison.py
======================================
Does adding p-reg (the two-thirds power-law regularizer) on top of straightening
make the terminal-objective loss landscape *smoother* / *closer to convex*?
Paper Fig. 4 shows the landscape for four single checkpoints; this script turns
that figure into a quantitative, paired comparison of exactly two variants:

    straighten          -> "straighten"
    straighten + p-reg  -> "p_reg_straighten"

Design (what makes it a fair test):
  * The SAME held-out val episode index is used for both variants (same
    start/goal pair), so the comparison is paired and not confounded by which
    episode happened to be picked. The two variants' val sets are additionally
    checked to be identical (same length + same qualifying-episode set).
  * Both variants get byte-identical optimizer settings: grid / opt_steps / lr /
    action_range / goal_H all come from one single resolved config, and each
    returned grid is asserted against it.
  * K episodes per env are chosen by a deterministic spacing rule (every
    len(qualifying)//K-th qualifying episode), i.e. spread over the whole val
    set instead of the first K consecutive ones. Nothing is cherry-picked, and
    the identical rule applies to both variants.

Metrics per (env, episode, variant) grid_loss, computed exactly like
curvature_analysis.loss_landscape (fix the first action, GD-optimize the rest of
the horizon, keep the minimum attainable terminal loss):
  1. roughness = mean(laplacian(grid_loss)^2) / var(grid_loss)
     A discrete 2D Laplacian of the grid (scipy.ndimage.laplace, reflect
     padding), squared, averaged, then normalized by the grid variance so that
     checkpoints with different absolute loss levels are comparable.
     Lower = smoother. `roughness_interior` repeats this on the interior
     (g-2)^2 cells only (dropping the boundary row/column, where the Laplacian
     uses reflected padding) as a secondary robustness number.
  2. local_minima = number of cells strictly lower than every one of their (up
     to 8) in-bounds neighbours. 1 = a single basin (no spurious local optima
     of the min-attainable-loss surface), >1 = extra local optima.
  3. min_loss / loss_range = the attained minimum and the max-min contrast of the
     same grid. Not smoothness numbers, but they say whether the swept action box
     contains an attained optimum at all: `roughness` is contrast-free only for a
     *uniform* rescaling, so a grid that is nearly constant (no gradient signal)
     scores as very smooth. Read them together with the two above.
  4. grid geometry. A Laplacian cannot tell "smooth" from "flat because uniformly
     bad", nor "rough" from "has one steep basin", so three cheap structural
     numbers are recorded next to it:
       basin_area_frac   fraction of cells within 1.1x of the grid minimum (how
                         much of the swept box is a usable start); quantised to
                         1/grid^2, so at grid 13 it moves in steps of 0.6%.
       median_over_min   plateau height relative to the attained optimum (>1
                         means the box has structure to descend).
       border_share      1 - (interior Laplacian energy)/(total energy): how much
                         of `roughness` is the reflected boundary padding rather
                         than the surface. >0.9 means the number is the edge.
       argmin_i / argmin_j / argmin_ax / argmin_ay / argmin_on_edge
                         where the optimum sits. argmin_on_edge=1 means the grid
                         minimum lies ON the boundary of the swept box, i.e. the
                         box does not contain the optimum: min_loss is then a
                         censored lower bound and any "basin width" measured at
                         that point describes the wall, not a basin.

Usage:
    python analysis/loss_landscape_comparison.py                    # full run
    python analysis/loss_landscape_comparison.py --smoke            # plumbing
    python analysis/loss_landscape_comparison.py --envs pusht --episodes 2
Run it through run_loss_landscape_comparison.sh (sources setup.sh), or with
DATASET_DIR exported and the `ts` conda env active.

Options:
    --envs E [E ...]   envs to compare (default: pusht only). Every key of
                       MODEL_DIRS is accepted; point_maze (umaze) is still
                       supported but is not touched unless it is named here.
    --smoke            plumbing preset: --episodes 1 --grid 5 --opt-steps 10,
                       with `_smoke` output names so it cannot be mistaken for
                       evidence. Explicit --episodes/--grid/--opt-steps win.
    --episodes K       held-out val episodes per env (default 6)
    --grid N           landscape grid size per axis (default 13, matching the
                       original Fig. 4 panel; 9 is 2.1x cheaper but coarser)
    --opt-steps N      GD steps per grid point (default 80)
    --lr F             Adam lr for the action optimization (default 0.1)
    --action-range R   grid half-range in normalized action units (default 2.0)
    --goal-H N         horizon in raw frames, must divide by frameskip
                       (default 25, matching the existing Fig. 4 panel)
    --n-viz N          illustrative episodes to plot per env (default 2)
    --no-viz           skip the figures
    --diagnostics      also write the diagnostic figure set: per-episode
                       shared / autoscaled / (L - L_min) heatmaps, per-episode
                       1-D profiles through each variant's argmin plus the
                       basin-area curve, and one paired summary figure per env.
                       Off by default so the default run's figures are unchanged.
    --viz-only         never run a GD sweep: rebuild every metric, CSV and figure
                       from the cached grids, failing if any grid is missing.
                       Implies --reuse-grids and --diagnostics; add --no-viz to
                       refresh only the CSVs. (Any --reuse-grids pass carries the
                       recorded wall times over from the existing per-episode CSV,
                       so a refresh cannot erase how long the run that computed the
                       grids actually took; `reused` still reports whether THIS pass
                       recomputed each grid.)
    --reuse-grids      reuse cached .npz grids when the recorded optimizer
                       settings match (instant metrics/figures re-run)
    --heartbeat S      seconds between "still running" lines while a single grid
                       is being optimized (default 60; 0 disables). One grid-13
                       grid is ~13.5k GD steps, i.e. ~30 min of intended silence
                       between per-grid result lines without this.
    --outdir DIR       output root (default <repo>/analysis_outputs)
    --device DEV       torch device (default cuda)
    --seed INT         torch/numpy seed (default 0)

Outputs (under --outdir; `_smoke` suffix in smoke mode):
    loss_landscape_comparison.csv   aggregate table, one row per (env, variant):
                                    mean/std metrics + paired win count
                                    ("5/6 episodes")
    loss_landscape/per_episode.csv  one row per (env, episode, variant): raw
                                    metrics + full provenance
    loss_landscape/grids/*.npz      cached landscape grids
    loss_landscape/landscape_<env>_ep<idx>.png
                                    paired two-panel heatmap of the illustrative
                                    episodes (chosen by the trend rule below,
                                    not by how they look)
    loss_landscape/diagnostics_<env>_ep<idx>.png
                                    (--diagnostics) 2x3 panels per episode:
                                    shared scale / per-panel autoscale /
                                    (L - L_min) with fixed depth contours
    loss_landscape/profiles_<env>_ep<idx>.png
                                    (--diagnostics) L/L_min vs offset from each
                                    variant's own argmin along both axes + the
                                    basin-area curve P(L <= f*L_min) vs f
    loss_landscape/diagnostics_<env>_summary.png
                                    (--diagnostics) per-env paired summary:
                                    min_loss scatter, basin area, plateau ratio,
                                    the roughness-vs-contrast confound, border
                                    share, and the per-episode roughness ratio

Illustrative-episode rule (fixed before looking at any heatmap, and keyed to the
attained-optimum metric rather than to `roughness`, which is null at K=6 and
partly tracks grid contrast): with r_i = min_loss(straighten) / min_loss(p-reg)
on episode i, plot the episode with the largest r_i (p-reg attains the lowest
optimum relative to straighten) and the episode with the smallest r_i -- the
worst case for that claim, which an honest example figure must show. Remaining
--n-viz slots are back-filled by |log r_i|. The rule, the chosen indices and both
r_i values are printed, and the captions say that the aggregate evidence is the
CSV, not the figure.

Interpretation aids: after each per-env table the run prints a paired-diagnostics
block -- how often each variant is lower, an exact sign test, a Wilcoxon
signed-rank test, the P/S ratio as a median and as a geometric mean with a
bootstrap CI -- plus the 80%-power minimum detectable ratio at this K (at K=6 the
sign test needs 6/6, so a 3/3 split establishes nothing) and a check that
`roughness` is not just tracking the grid's max-min contrast. All of it is
printed only; no metric is changed and nothing is re-run.

Runtime: cost = envs x variants x K x grid^2 x opt_steps rollouts, i.e. O(hours):
~6 h for the default PushT-only, 6-episode, grid-13 run (the paper's Fig. 4
sizing; ~3 h at --grid 9, ~12 h with `--envs pusht point_maze`). Each env prints
its plan (landscapes x GD steps x expected wall time) before the first grid, and
a heartbeat line every --heartbeat seconds while a grid is optimizing -- a single
grid-13 grid is ~30 min, so without it the run looks hung. The heartbeat's first
estimate uses an assumed 0.13 s/step (this laptop's GPU) and every later grid is
estimated from the rate actually measured, so a slower/faster machine self-corrects
after grid 1. Progress, per-call wall time and an ETA are printed per grid, every
grid is cached as soon as it is computed, the per-episode CSV is rewritten after
every episode and the aggregate CSV after every env, so a long run can be
interrupted and resumed (--reuse-grids skips cached grids). `--smoke` is the fast
correctness check.

Known gap: the grid cache is validated on experiment settings (env, variant,
episode_idx, grid / opt_steps / lr / action_range / goal_H, and the filename now
carries every one of those knobs) but it still carries no fingerprint of the val
dataset itself. If DATASET_DIR (or the data behind it) changes, --reuse-grids will
mix grids from two different dataset snapshots while reporting the episode lengths
of whichever set is loaded now, because the `seq_length` column is read at run time
and can therefore disagree with the cached grid it sits next to. After any dataset
change, re-run an env from scratch instead of reusing its grids. The paired
assertion only guarantees that BOTH variants of one run saw the same val set (it
compares val_len + seq_length per episode), which is what makes the comparison
itself valid. A cache written by an older version of this script (filename without
lr/action_range/goal_H) is still READ, but only when every recorded setting matches
the current run, and it is never written to again -- new grids always go to the
fully-keyed filename, so a non-default run cannot overwrite grids computed with the
default settings.

Checkpoint paths live in MODEL_DIRS below; empty strings are placeholders that
are reported and skipped. An env is compared only when BOTH of its variants
resolve (otherwise the comparison would be one-sided). The default run covers
DEFAULT_ENVS only (PushT); point_maze is opt-in.

Context variants (--baseline, off by default) are a separate, deliberately
non-paired read: the no-straightening checkpoint is swept on the same episodes with
the same settings and the same val set (asserted, not assumed) and is drawn as the
first panel of the paper-style figure (--fig-style paper, which also writes the
Fig.-4-style rendering of the pair itself), but its rows go to
per_episode_context.csv and its numbers enter no paired statistic, win count or
headline. Note what this baseline is NOT: the paper's "(a) DINOv2" panel is a raw
encoder panel, whereas MODEL_DIRS["pusht"]["baseline"] is the repo's
trained-without-straightening checkpoint (curvature_analysis.MODELS["pusht"][0]),
i.e. the training ablation, so it differs from the pair in more than just the
regularizer (it also never saw the straightening loss). Only the context CSV and
the fig4 panel depend on it, precisely so that a different reference can be swapped
in (--baseline-ckpt) without touching the paired evidence.
"""
import argparse
import csv
import os
import sys
import threading
import time
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import ndimage, stats

warnings.filterwarnings("ignore")

# A multi-hour run must show progress and the ETA even when its stdout is
# redirected to a file (python block-buffers a non-tty stdout by default); the
# wrapper additionally passes -u, this makes it true regardless of invocation.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:                                   # pragma: no cover (< 3.7)
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from curvature_analysis import (  # noqa: E402
    load_model,
    load_val_dset,
    loss_landscape,
    plot_landscape,
)

# ---------------------------------------------------------------------------
# Variants and their checkpoints (relative to REPO; empty = placeholder).
# The two variants differ ONLY by the added two-thirds p-reg term, so any
# difference in landscape smoothness is attributable to it.
# ---------------------------------------------------------------------------
VARIANT_ORDER = ["straighten", "p_reg_straighten"]
VARIANT_LABEL = {
    "straighten": "straighten",
    "p_reg_straighten": "straighten + p-reg",
    # context-only label (the paper-style figure's reference panel); no paired
    # metric ever carries this variant, so the aggregate table cannot see it.
    "baseline": "no straightening (baseline)",
}

# Context variants: extra checkpoints that are drawn NEXT TO the pair -- the
# paper's Fig. 4 shows a "(a) DINOv2" panel before its straighten panels -- but
# never compared against it. They have their own row file
# (per_episode_context.csv) and their own figure (fig4_*.png), both produced only
# with --baseline, so adding or dropping one cannot move a number in
# loss_landscape_comparison.csv / per_episode.csv.
CONTEXT_VARIANTS = ["baseline"]

MODEL_DIRS = {
    "pusht": {
        "straighten": "checkpoints/test/pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg_straighten": "checkpoints/test/pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        # training.straighten=false and twothirds=false: the repo's own "baseline"
        # dir (curvature_analysis.MODELS["pusht"][0]), i.e. the model without
        # straightening -- the reference the paper's "(a) DINOv2" panel plays.
        "baseline": "checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "point_maze": {
        "straighten": "checkpoints/test/umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "p_reg_straighten": "checkpoints/test/umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        # placeholder: no no-straightening umaze checkpoint with this recipe yet,
        # so --baseline simply skips the context panel for this env
        "baseline": "",
    },
    # PLACEHOLDERS: the medium-maze pair this comparison needs (same recipe as
    # above: dino_channel projector, aggcos1e-1 vs aggcos1e-1 + aggtwothirds5e-2)
    # is not trained yet. Fill both paths in when it is. The existing
    # medium_*_projglobal_* checkpoints are NOT a substitute: they use the
    # dino_global projector AND different encoder lr / cos weights, so a
    # difference between them would not isolate the p-reg term.
    "point_maze_medium": {
        "straighten": "",
        "p_reg_straighten": "",
        "baseline": "",
    },
}

# Presets: the full run follows the spec (K=6, grid=13, opt_steps=80) -- grid 13 is
# the paper's Fig. 4 sizing and what run_loss_landscape_comparison.sh passes;
# --smoke is a fast plumbing check. Only these three knobs differ, and an explicit
# CLI flag always wins over both presets.
SPEC_PRESET = {"episodes": 6, "grid": 13, "opt_steps": 80}
SMOKE_PRESET = {"episodes": 1, "grid": 5, "opt_steps": 10}

# What a bare `--` invocation covers: PushT only. It is the paper's Table 1 /
# Fig. 4 row with the 14x14x8 channel projector, and the only env whose p-reg
# pair (aggcos1e-1 vs aggcos1e-1 + aggtwothirds5e-2) holds everything else fixed.
# point_maze (umaze) still works, it is just not part of the default run: add it
# with `--envs pusht point_maze`. point_maze_medium has no comparable pair yet
# (empty placeholders below) and is skipped rather than silently confounded.
DEFAULT_ENVS = ["pusht"]


def resolve_ckpt_path(path):
    """Return the absolute checkpoint run dir for a path, raising a clear error."""
    model_dir = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.isfile(os.path.join(model_dir, "hydra.yaml")):
        raise RuntimeError(f"Missing hydra.yaml in checkpoint dir: {model_dir}")
    if not os.path.isfile(os.path.join(model_dir, "checkpoints", "model_latest.pth")):
        raise RuntimeError(
            f"Missing checkpoints/model_latest.pth in checkpoint dir: {model_dir}"
        )
    return model_dir


def resolve_ckpt(env, variant):
    """Return the absolute checkpoint run dir, raising a clear error on placeholders."""
    path = MODEL_DIRS[env][variant]
    if not path:
        raise RuntimeError(
            f"MODEL_DIRS[{env!r}][{variant!r}] is an empty placeholder -- "
            "fill in the checkpoint path before running."
        )
    return resolve_ckpt_path(path)


def context_variants(env, override=None):
    """[(variant, model_dir), ...] for this env's context variants that resolve.

    Unlike VARIANT_ORDER these are optional. An empty MODEL_DIRS placeholder means
    "not available for this env" and is reported and skipped, so `--baseline` on an
    env without a no-straightening checkpoint degrades to the paired figures
    instead of killing the run; a non-empty path that does not resolve still
    raises, because that is a typo and not a choice. `override` (--baseline-ckpt)
    replaces the single-dir baseline path, so the reference panel can be pointed at
    another run without editing MODEL_DIRS.
    """
    out = []
    for variant in CONTEXT_VARIANTS:
        path = MODEL_DIRS.get(env, {}).get(variant, "")
        if variant == "baseline" and override:
            path = override
        if not path:
            print(f"  [note] [{env}] no {variant!r} checkpoint in MODEL_DIRS -- "
                  f"no {variant} panel/rows for this env (a context variant is "
                  "optional; the paired comparison is unaffected)")
            continue
        out.append((variant, resolve_ckpt_path(path)))
    return out


def redirect_dset_paths(train_cfg, env, variant):
    """Repoint the checkpoint's saved (old-machine) dataset path at the local copy.

    Same logic as analysis/linear_probe.py:redirect_dset_paths (itself mirroring
    plan.py): if env.dataset.data_path does not exist on this machine, point it at
    $DATASET_DIR/<basename> (default <repo>/data/datasets/<basename>). Kept local
    so this script does not depend on the probe's sklearn stack.
    """
    try:
        stored = train_cfg.env.dataset.get("data_path")
    except Exception:                                    # noqa: BLE001 (mirror probe)
        stored = None
    if not stored:
        return
    stored = os.path.normpath(str(stored))
    if os.path.isdir(stored):
        return
    dname = os.path.basename(stored)
    root = os.environ.get("DATASET_DIR", os.path.join(REPO, "data", "datasets"))
    redirect = os.path.join(root, dname)
    if not os.path.isdir(redirect):
        raise RuntimeError(
            f"[{env}/{variant}] dataset path {stored} is missing and there is no "
            f"local copy at {redirect}; set DATASET_DIR to the folder containing "
            f"'{dname}'.")
    OmegaConf.set_struct(train_cfg, False)
    train_cfg.env.dataset.data_path = redirect
    OmegaConf.set_struct(train_cfg, True)
    print(f"    [{env}/{variant}] dataset data_path {stored} -> {redirect}")


# ---------------------------------------------------------------------------
# Smoothness metrics
# ---------------------------------------------------------------------------
# Thresholds used by the structural (non-Laplacian) metrics below. Kept as named
# constants so a figure/print can quote the same definition as the CSV.
BASIN_FACTOR = 1.1            # "near-optimal" = within 10% of the grid minimum
DEPTH_CONTOURS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)   # fractions of the pair's max depth


def laplacian_roughness(grid_loss, interior=False):
    """mean(laplacian(grid_loss)^2) / var(grid_loss): scale-free roughness.

    The Laplacian is the standard 4-neighbour stencil (scipy.ndimage.laplace,
    reflect padding at the boundary). With interior=True the boundary row/column
    is dropped and the variance of the same interior cells is used as the
    denominator, so the number cannot be driven by the padding choice.
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    lap = ndimage.laplace(g)
    if interior:
        mask = np.zeros(g.shape, dtype=bool)
        mask[1:-1, 1:-1] = True
        lap, g = lap[mask], g[mask]
    mean_sq = float(np.mean(lap ** 2))
    var = float(np.var(g))
    if var == 0.0:
        return 0.0 if mean_sq == 0.0 else float("inf")
    return mean_sq / var


def count_local_minima(grid_loss):
    """Cells strictly lower than all (up to 8) in-bounds neighbours.

    Edges/corners use only the neighbours that exist; non-finite cells are
    skipped. 1 = a single basin (no spurious local optima of the surface).
    A perfectly flat grid gives 0 (no *strict* minimum).
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    n_rows, n_cols = g.shape
    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            v = g[i, j]
            if not np.isfinite(v):
                continue
            is_min = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n_rows and 0 <= jj < n_cols and not v < g[ii, jj]:
                        is_min = False
                        break
                if not is_min:
                    break
            count += int(is_min)
    return count


def border_energy_share(grid_loss):
    """How much of the Laplacian's mean-squared energy sits on the border.

    The boundary row/column has no neighbour outside the box, so
    scipy.ndimage.laplace pads it (reflect) and its values are a property of that
    padding rather than of the surface. ~1.0 means `roughness` is essentially the
    edge (the grid is smooth *inside*), ~0.0 means it is a genuine interior
    feature, and a *negative* value means the border is smoother than the
    interior. Returns nan for a grid with no interior or with zero total energy
    (a perfectly flat gradient field).
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    if min(g.shape) < 3:
        return float("nan")
    lap = ndimage.laplace(g)
    total = float(np.mean(lap ** 2))
    interior = float(np.mean(lap[1:-1, 1:-1] ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - interior / total


def basin_area_frac(grid_loss, factor=BASIN_FACTOR):
    """Fraction of cells within `factor` x the grid minimum (quantised to 1/grid^2).

    Read it together with min_loss: a 100% basin around a bad optimum (ep 12's
    straighten grid attains 0.129 with 100% of its cells within 1.1x of it) is a
    flat, uninformative surface -- it gives an optimizer no gradient signal --
    while a 0.6% basin around the *only* low point in the box (ep 15's p-reg
    grid: 1 cell of 169) is a real, narrow basin. Neither is "better" on its own;
    the pair is what separates them. Returns nan for a non-finite grid minimum.
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    lo = float(np.min(g))
    if not np.isfinite(lo):
        return float("nan")
    return float(np.mean(g <= factor * lo))


def basin_curve(grid_loss, factors):
    """P(L <= f * L_min) for each f in `factors` -- basin area as a function of depth.

    The contrast-free version of basin_area_frac: sweeping f from 1 to 2 shows
    how fast the usable part of the box grows with the depth tolerance, which is
    comparable across variants no matter how their absolute loss levels differ.
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    lo = float(np.min(g))
    if not np.isfinite(lo) or lo <= 0.0:
        return np.full(np.shape(factors), np.nan)
    return np.array([float(np.mean(g <= f * lo)) for f in np.asarray(factors, float)])


def grid_metrics(grid_loss):
    """Smoothness metrics for one landscape grid + its contrast/attained optimum.

    `min_loss` and `loss_range` are not smoothness numbers; they say whether the
    swept action box contains an attained optimum at all and how much contrast it
    has. They are recorded because `roughness` is contrast-free only for a
    *uniform* rescaling of the surface: contrast that comes from one sharper
    feature (a narrower basin, a steeper wall) raises it, so a nearly constant
    grid -- the kind that gives an optimizer no gradient signal -- scores as
    "smooth". Read the two groups together (see print_paired_diagnostics).
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    lo = float(np.min(g))
    hi = float(np.max(g))
    med = float(np.median(g))
    arg_i, arg_j = np.unravel_index(int(np.argmin(g)), g.shape)
    # The box boundary is where the probe is censored (see the module docstring):
    # a minimum ON the edge means the optimum is not inside the swept box.
    on_edge = bool(arg_i in (0, g.shape[0] - 1) or arg_j in (0, g.shape[1] - 1))
    return {
        "roughness": laplacian_roughness(g),
        "roughness_interior": laplacian_roughness(g, interior=True),
        "local_minima": count_local_minima(g),
        "min_loss": lo,
        "loss_range": hi - lo,
        # structural numbers: plateau/plateau-vs-optimum and basin size
        "median_over_min": med / lo if lo > 0.0 else float("nan"),
        "max_over_min": hi / lo if lo > 0.0 else float("nan"),
        "basin_area_frac": basin_area_frac(g),
        "argmin_i": int(arg_i), "argmin_j": int(arg_j),
        "argmin_on_edge": int(on_edge),
        "border_share": border_energy_share(g),
    }


# ---------------------------------------------------------------------------
# Episode selection (identical for every variant)
# ---------------------------------------------------------------------------
def qualifying_episode_indices(val, goal_H):
    """Val indices whose real episode is long enough for a start + goal frame."""
    return [i for i in range(len(val)) if val.get_seq_length(i) >= goal_H + 1]


def select_spaced_episodes(qualifying, k):
    """Pick k episodes spread over the qualifying list (deterministic, no cherry-picking).

    stride = max(1, len(qualifying) // k): e.g. 200 qualifying episodes with k=6
    gives 0, 33, 66, 99, 132, 165 (for the 21-episode PushT val set it gives
    0, 3, 6, 9, 12, 15). If fewer than k qualify, all are returned and the
    caller warns.
    """
    if not qualifying:
        return []
    n = min(k, len(qualifying))
    stride = max(1, len(qualifying) // k)
    return [qualifying[i * stride] for i in range(n)]


# ---------------------------------------------------------------------------
# Grid cache (keeps a multi-hour run restartable; --reuse-grids skips work)
# ---------------------------------------------------------------------------
def _scalar(x):
    """Unwrap a 0-d numpy array (as produced by an .npz) into a python scalar."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def grid_cache_key(run_cfg):
    """Every optimizer setting that changes the numbers, as a filename-safe string.

    The legacy name only carried grid+opt_steps, so `--reuse-grids --action-range
    3` used to correctly reject the cached grid as stale and then WRITE the new
    grid to that same legacy path, silently destroying the default-settings grid.
    New grids therefore always go to this keyed name; the legacy name is only ever
    read (see grid_cache_candidates).
    """
    return (f"g{run_cfg['grid']}_s{run_cfg['opt_steps']}_lr{run_cfg['lr']:g}"
            f"_ar{run_cfg['action_range']:g}_gh{run_cfg['goal_H']}")


def grid_cache_path(grids_dir, env, episode_idx, variant, run_cfg):
    """Write target for one grid: uniquely keyed by every optimizer setting."""
    return os.path.join(
        grids_dir,
        f"{env}_{grid_cache_key(run_cfg)}_ep{episode_idx:03d}_{variant}.npz",
    )


def grid_cache_candidates(grids_dir, env, episode_idx, variant, run_cfg):
    """Paths to try when reading, current naming scheme first.

    A legacy-keyed file (no lr/action_range/goal_H in the name) is still read when
    every setting recorded inside it matches the requested run, so grids from a
    full run done before this key existed remain reusable. It is never a write
    target, which is what makes the reuse safe.
    """
    return [
        grid_cache_path(grids_dir, env, episode_idx, variant, run_cfg),
        os.path.join(
            grids_dir,
            f"{env}_g{run_cfg['grid']}_s{run_cfg['opt_steps']}_"
            f"ep{episode_idx:03d}_{variant}.npz"),
    ]


def cached_grid_matches(payload, run_cfg, env, episode_idx, variant):
    """Check a cached .npz against the requested settings; return (ok, reason)."""
    want = {"env": env, "variant": variant, "episode_idx": episode_idx,
            "grid": run_cfg["grid"], "opt_steps": run_cfg["opt_steps"],
            "lr": run_cfg["lr"], "action_range": run_cfg["action_range"],
            "goal_H": run_cfg["goal_H"]}
    for key, expected in want.items():
        if key not in payload:
            return False, f"missing key {key!r}"
        got = _scalar(payload[key])
        if isinstance(expected, str):
            if str(got) != expected:
                return False, f"{key}: cached {got!r} != requested {expected!r}"
        elif float(got) != float(expected):
            return False, f"{key}: cached {got!r} != requested {expected!r}"
    return True, ""


def cached_grid_or_none(grids_dir, env, episode_idx, variant, run_cfg):
    """The path of a matching cached grid, or None (same matching rules as above).

    Used for the pre-flight check that `--viz-only --baseline` can only pass when
    the context grids already exist: without it, requiring a missing context cache
    would surface as the generic "env skipped" message and look like a broken
    checkpoint pair instead of a missing cache.
    """
    for cand in grid_cache_candidates(grids_dir, env, episode_idx, variant, run_cfg):
        if not os.path.isfile(cand):
            continue
        with np.load(cand, allow_pickle=False) as payload:
            ok, _ = cached_grid_matches(payload, run_cfg, env, episode_idx, variant)
        if ok:
            return cand
    return None


def compute_or_load_grid(env, variant, episode_idx, model_dir, wm, train_cfg, val,
                         device, run_cfg, grids_dir, reuse, on_start=None,
                         require_cached=False):
    """Return (grid_loss, used_idx, reused, wall_time_s) for one landscape.

    `on_start` is called only when the grid really has to be computed (never on a
    cache hit), just before the long GD sweep starts -- used for progress output.
    With `require_cached` (the --viz-only path) a missing matching cache raises
    instead of starting a sweep, so a "refresh the metrics" invocation can never
    silently turn into a multi-hour compute; the write target is always the
    fully-keyed filename, so a non-default run cannot overwrite default grids.
    """
    path = grid_cache_path(grids_dir, env, episode_idx, variant, run_cfg)
    if reuse:
        for cand in grid_cache_candidates(grids_dir, env, episode_idx, variant, run_cfg):
            if not os.path.isfile(cand):
                continue
            with np.load(cand, allow_pickle=False) as payload:
                ok, reason = cached_grid_matches(payload, run_cfg, env, episode_idx, variant)
                if ok:
                    if cand != path:
                        print(f"    [note] reading legacy-keyed cache "
                              f"{os.path.basename(cand)} (read-only; new grids are "
                              f"written as {os.path.basename(path)})")
                    return (np.array(payload["grid_loss"]),
                            int(_scalar(payload["used_idx"])), True, 0.0)
            print(f"    [warn] ignoring stale cache {os.path.basename(cand)}: {reason}")
    if require_cached:
        raise RuntimeError(
            f"[{env}/{variant}] episode {episode_idx}: no cached grid matching these "
            f"settings at {os.path.basename(path)} and --viz-only never computes "
            "one; run once without --viz-only (or without --reuse-grids) first.")
    t0 = time.time()
    if on_start is not None:
        on_start()
    axs, grid_loss, used_idx = loss_landscape(
        wm, train_cfg, val, device,
        goal_H=run_cfg["goal_H"], grid=run_cfg["grid"],
        opt_steps=run_cfg["opt_steps"], lr=run_cfg["lr"],
        action_range=run_cfg["action_range"], episode_idx=episode_idx)
    wall = time.time() - t0
    np.savez(path, grid_loss=grid_loss, axs=axs, used_idx=used_idx,
             env=env, variant=variant, episode_idx=episode_idx,
             grid=run_cfg["grid"], opt_steps=run_cfg["opt_steps"],
             lr=run_cfg["lr"], action_range=run_cfg["action_range"],
             goal_H=run_cfg["goal_H"], model_dir=model_dir,
             # informational only, NOT enforced by cached_grid_matches: requiring a
             # dataset match would invalidate every grid cached before this key
             # existed (see the module docstring's "Known gap").
             dataset_dir=os.environ.get("DATASET_DIR", ""))
    return grid_loss, int(used_idx), False, wall


def assert_paired_provenance(provenance, episodes, env, variants=None):
    """Assert every variant in `variants` ran the exact same experiment per episode.

    `variants` defaults to the compared pair. Passing
    VARIANT_ORDER + context variants additionally pins the baseline panel to the
    same episodes, settings and val set, so "the reference panel is the same
    experiment as the pair" is checked and not assumed.
    """
    variants = list(VARIANT_ORDER if variants is None else variants)
    keys = ("episode_idx", "grid", "opt_steps", "lr", "action_range", "goal_H",
            "val_len", "seq_length")
    ref_v = variants[0]
    for ep in episodes:
        ref = provenance[(ep, ref_v)]
        for variant in variants[1:]:
            other = provenance[(ep, variant)]
            for key in keys:
                if ref[key] != other[key]:
                    raise AssertionError(
                        f"[{env}] episode {ep}: unpaired setting {key!r}: "
                        f"{ref_v}={ref[key]!r} vs {variant}={other[key]!r}")


# ---------------------------------------------------------------------------
# Paired comparison + aggregation
# ---------------------------------------------------------------------------
def _winner(value_a, value_b):
    """Which variant is better (lower = smoother/fewer minima); None = tie."""
    a, b = VARIANT_ORDER
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return None
    if value_a == value_b:
        return None
    return a if value_a < value_b else b


def _winner_higher(value_a, value_b):
    """Which variant is better when *bigger* is better (basin_area_frac)."""
    a, b = VARIANT_ORDER
    if not (np.isfinite(value_a) and np.isfinite(value_b)):
        return None
    if value_a == value_b:
        return None
    return a if value_a > value_b else b


def _std(values):
    """Sample std (ddof=1) over episodes; 0.0 when there is only one."""
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _std_finite(values):
    """`_std` over the finite entries only (structural metrics can be nan)."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return _std(finite) if finite.size else float("nan")


def _mean_finite(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def build_paired_rows(env, episodes, rows):
    """Per-episode paired view: both variants side by side + who wins what."""
    by_key = {(r["episode"], r["variant"]): r for r in rows}
    a, b = VARIANT_ORDER
    out = []
    for ep in episodes:
        ra, rb = by_key[(ep, a)], by_key[(ep, b)]
        for key in ("episode_idx", "grid", "opt_steps", "lr", "action_range",
                    "goal_H", "seq_length"):
            if ra[key] != rb[key]:
                raise AssertionError(
                    f"[{env}] episode {ep}: {key} differs between variants "
                    f"({ra[key]!r} vs {rb[key]!r})")
        out.append({
            "env": env, "episode": ep, "seq_length": ra["seq_length"],
            "roughness": {a: ra["roughness"], b: rb["roughness"]},
            "roughness_interior": {a: ra["roughness_interior"], b: rb["roughness_interior"]},
            "local_minima": {a: ra["local_minima"], b: rb["local_minima"]},
            "min_loss": {a: ra["min_loss"], b: rb["min_loss"]},
            "loss_range": {a: ra["loss_range"], b: rb["loss_range"]},
            # structural numbers: they qualify every smoothness read (see docstring)
            "median_over_min": {a: ra["median_over_min"], b: rb["median_over_min"]},
            "max_over_min": {a: ra["max_over_min"], b: rb["max_over_min"]},
            "basin_area_frac": {a: ra["basin_area_frac"], b: rb["basin_area_frac"]},
            "border_share": {a: ra["border_share"], b: rb["border_share"]},
            "argmin_on_edge": {a: ra["argmin_on_edge"], b: rb["argmin_on_edge"]},
            "winner_roughness": _winner(ra["roughness"], rb["roughness"]),
            "winner_roughness_interior": _winner(ra["roughness_interior"],
                                                 rb["roughness_interior"]),
            "winner_local_minima": _winner(float(ra["local_minima"]),
                                           float(rb["local_minima"])),
            # basin area is the one metric where BIGGER is better, so it cannot go
            # through _winner() (lower-is-better) unless the sign is flipped here.
            "winner_basin_area": _winner_higher(ra["basin_area_frac"],
                                                rb["basin_area_frac"]),
            "winner_min_loss": _winner(ra["min_loss"], rb["min_loss"]),
        })
    return out


def summarize_env(env, paired, rows):
    """Aggregate table rows for one env: mean/std per variant + paired wins.

    paired_win_count_* is "wins/n_episodes" for that variant (lower metric wins;
    ties are counted separately in n_ties_*). n_episodes is the number of paired
    episodes actually compared.
    """
    n = len(paired)
    out = []
    for variant in VARIANT_ORDER:
        rr = [r for r in rows if r["variant"] == variant]
        rough = np.array([r["roughness"] for r in rr], dtype=np.float64)
        rough_i = np.array([r["roughness_interior"] for r in rr], dtype=np.float64)
        minima = np.array([r["local_minima"] for r in rr], dtype=np.float64)
        min_loss = np.array([r["min_loss"] for r in rr], dtype=np.float64)
        loss_range = np.array([r["loss_range"] for r in rr], dtype=np.float64)
        basin = np.array([r["basin_area_frac"] for r in rr], dtype=np.float64)
        border = np.array([r["border_share"] for r in rr], dtype=np.float64)
        on_edge = np.array([r["argmin_on_edge"] for r in rr], dtype=np.float64)
        out.append({
            "env": env, "variant": variant,
            "mean_roughness": float(np.mean(rough)),
            "std_roughness": _std(rough),
            "paired_win_count_roughness": f"{sum(1 for p in paired if p['winner_roughness'] == variant)}/{n}",
            "mean_local_minima": float(np.mean(minima)),
            "std_local_minima": _std(minima),
            "paired_win_count_minima": f"{sum(1 for p in paired if p['winner_local_minima'] == variant)}/{n}",
            "n_episodes": n,
            "n_ties_roughness": sum(1 for p in paired if p["winner_roughness"] is None),
            "n_ties_minima": sum(1 for p in paired if p["winner_local_minima"] is None),
            "mean_roughness_interior": float(np.mean(rough_i)),
            "std_roughness_interior": _std(rough_i),
            "paired_win_count_roughness_interior":
                f"{sum(1 for p in paired if p['winner_roughness_interior'] == variant)}/{n}",
            # contrast / attained optimum (interpretation aids, not smoothness)
            "mean_min_loss": float(np.mean(min_loss)),
            "std_min_loss": _std(min_loss),
            "mean_loss_range": float(np.mean(loss_range)),
            "std_loss_range": _std(loss_range),
            # structural geometry: basin size, padding share, where the optimum is
            "mean_basin_area_frac": _mean_finite(basin),
            "std_basin_area_frac": _std_finite(basin),
            "paired_win_count_basin_area":
                f"{sum(1 for p in paired if p['winner_basin_area'] == variant)}/{n}",
            "paired_win_count_min_loss":
                f"{sum(1 for p in paired if p['winner_min_loss'] == variant)}/{n}",
            "mean_border_share": _mean_finite(border),
            "std_border_share": _std_finite(border),
            "frac_argmin_on_edge": f"{int(np.sum(on_edge))}/{n}",
        })
    return out


# ---------------------------------------------------------------------------
# Progress reporting for the long GD sweeps
# ---------------------------------------------------------------------------
# One landscape = grid^2 x opt_steps GD steps: at grid 13 / 80 steps that is
# 13,520 steps, i.e. ~30 min per grid on the laptop 3050 Ti (~0.13 s/step), so
# consecutive per-grid lines are a long time apart. The ticker below makes that
# visible; --heartbeat 0 silences it. It only prints -- no run state is touched.
DEFAULT_S_PER_STEP = 0.13          # measured on the laptop RTX 3050 Ti
_PRINT_LOCK = threading.Lock()


def fmt_hms(seconds):
    """'2h05m' / '8m12s' -- compact enough for a log line."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Heartbeat:
    """Daemon ticker printing 'still running: X elapsed, ~Y to go' every `period`s.

    The caller passes the measured seconds-per-GD-step (decimal minutes), so the
    estimate tightens as the run goes on; the first grid falls back to
    DEFAULT_S_PER_STEP.
    """

    def __init__(self, label, total_steps, s_per_step, period):
        self.label = label
        self.total_steps = max(int(total_steps), 1)
        self.s_per_step = s_per_step
        self.period = period
        self.t0 = time.time()
        self._stop = threading.Event()
        self._thread = None

    def _tick(self):
        expected = max(self.total_steps * self.s_per_step, 1e-9)
        while not self._stop.wait(self.period):
            elapsed = time.time() - self.t0
            if elapsed < expected:
                left = (1.0 - elapsed / expected) * expected
                msg = (f"{fmt_hms(elapsed)} elapsed, ~{fmt_hms(left)} to go "
                       f"(~{100.0 * elapsed / expected:.0f}% of this grid, "
                       f"{self.s_per_step:.3f}s/step)")
            else:
                # Saying "~0m00s to go" for minutes on end would be a lie; say the
                # assumption was off. The measured rate replaces it on the next grid.
                msg = (f"{fmt_hms(elapsed)} elapsed -- past the {fmt_hms(expected)} "
                       f"estimate at {self.s_per_step:.3f}s/step (slower than "
                       f"assumed; the next grid is recalibrated)")
            with _PRINT_LOCK:
                print(f"      ... {self.label}: {msg}")

    def __enter__(self):
        if self.period > 0:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def announce(self, message):
        """One-off 'this is what is being computed' line, heartbeat-safe."""
        with _PRINT_LOCK:
            print(f"      {message}")


# ---------------------------------------------------------------------------
# Per-env sweep
# ---------------------------------------------------------------------------
def run_env(env, run_cfg, args, device, grids_dir, paths=None, prior_rows=None,
            prior_context_rows=None):
    """Landscape grids + per-episode metrics for one env, both variants paired.

    The same episode indices, grid, optimizer settings and val set are used for
    both variants; anything that would break the pairing raises instead of
    producing a number that looks fine.

    With --baseline the sweep additionally covers the context variant(s)
    (CONTEXT_VARIANTS): same episodes, same settings, same val-set assertion, but
    their rows go to their own CSV and never reach build_paired_rows/summarize_env,
    so the paired evidence is identical with and without them.
    """
    a, b = VARIANT_ORDER
    model_dirs = {v: resolve_ckpt(env, v) for v in VARIANT_ORDER}
    # --baseline-ckpt overrides the single-dir baseline; empty placeholders are
    # reported and skipped (see context_variants)
    ctx_variants = context_variants(env, args.baseline_ckpt) if args.baseline else []
    model_dirs.update(dict(ctx_variants))
    ctx_names = [v for v, _ in ctx_variants]
    # one arm per (variant, role): the role decides which row list a grid feeds
    arms = [(v, "pair") for v in VARIANT_ORDER] + [(v, "context") for v in ctx_names]
    grids, provenance, rows, metrics_by_grid = {}, {}, [], {}
    ctx_rows = []
    ref_qual, ref_len, episodes = None, None, None
    n_calls, n_computed, computed_wall = 0, 0, 0.0
    steps_computed = 0                          # for the measured s/step estimate
    n_planned = None
    # A cache hit does not know how long the grid took, so a reuse pass that just
    # rewrote the CSV would replace real 27-minute wall times with 0.000 and lose
    # the provenance of the run that computed them. Carry the recorded timings
    # over (for this env and these exact settings) and keep `reused` reporting
    # whether *this* pass recomputed the grid, so the CSV is stable across passes.
    prior_wall = (load_prior_provenance(paths["per_episode_csv"], env, run_cfg)
                  if (args.reuse_grids and paths is not None) else {})
    if args.reuse_grids and paths is not None and ctx_names:
        # context rows live in their own file; (variant, episode) keys cannot collide
        prior_wall.update(load_prior_provenance(paths["context_csv"], env, run_cfg))
    axs = landscape_axis(run_cfg)

    for variant, role in arms:
        is_pair = role == "pair"
        if not is_pair and args.viz_only:
            # A missing context grid would otherwise surface as "env skipped", which
            # means "the checkpoint pair did not resolve" -- wrong message and it
            # would hide the pair figures of this env. --viz-only never computes a
            # grid, so name the missing caches and stop instead.
            missing = [os.path.basename(grid_cache_path(grids_dir, env, ep, variant, run_cfg))
                       for ep in episodes
                       if cached_grid_or_none(grids_dir, env, ep, variant, run_cfg) is None]
            if missing:
                raise SystemExit(
                    f"[{env}/{variant}] --viz-only --baseline needs the context grids "
                    f"cached already; {len(missing)} missing: {', '.join(missing)}. Run "
                    "once with --reuse-grids (without --viz-only) to compute them, or "
                    "drop --baseline.")
        wm, train_cfg = load_model(model_dirs[variant], device)
        redirect_dset_paths(train_cfg, env, variant)
        frameskip = int(train_cfg.frameskip)
        if run_cfg["goal_H"] % frameskip:
            raise SystemExit(
                f"[{env}/{variant}] goal_H={run_cfg['goal_H']} is not a multiple of "
                f"frameskip={frameskip}; loss_landscape would silently truncate "
                "the horizon.")
        val = load_val_dset(train_cfg)
        qual = qualifying_episode_indices(val, run_cfg["goal_H"])
        if is_pair and variant == a:
            if not qual:
                raise RuntimeError(
                    f"[{env}] no val episode is long enough for goal_H="
                    f"{run_cfg['goal_H']} (need seq_length >= {run_cfg['goal_H'] + 1})")
            ref_qual, ref_len = qual, len(val)
            episodes = select_spaced_episodes(qual, run_cfg["episodes"])
            if len(episodes) < run_cfg["episodes"]:
                print(f"  [warn] [{env}] only {len(episodes)} qualifying episode(s) "
                      f"(seq_length >= {run_cfg['goal_H'] + 1}) for --episodes "
                      f"{run_cfg['episodes']}; the comparison uses those.")
            n_planned = len(arms) * len(episodes)
            steps_per_grid = run_cfg["grid"] ** 2 * run_cfg["opt_steps"]
            print(f"  [{env}] val episodes={len(val)}  qualifying={len(qual)}  "
                  f"using indices {episodes}")
            extra = (f" + {len(ctx_names)} context variant(s) {ctx_names}"
                     if ctx_names else "")
            print(f"  [{env}] plan: {len(episodes)} episodes x {len(VARIANT_ORDER)} "
                  f"paired variants{extra} = {n_planned} landscapes x "
                  f"{run_cfg['grid']}x{run_cfg['grid']}x{run_cfg['opt_steps']} steps "
                  f"= {n_planned * steps_per_grid:,} GD steps, ~"
                  f"{fmt_hms(n_planned * steps_per_grid * DEFAULT_S_PER_STEP)} at "
                  f"{DEFAULT_S_PER_STEP}s/step; the first grid completes in ~"
                  f"{fmt_hms(steps_per_grid * DEFAULT_S_PER_STEP)}")
        elif len(val) != ref_len or qual != ref_qual:
            raise RuntimeError(
                f"[{env}] {variant!r} does not see the same val set as {a!r} "
                f"({a}: {ref_len} episodes / {len(ref_qual)} qualifying vs "
                f"{variant}: {len(val)} / {len(qual)}) -- the comparison would be "
                "invalid, check the checkpoints' data configs.")
        for ep in episodes:
            s_per_step = (computed_wall / steps_computed if steps_computed
                          else DEFAULT_S_PER_STEP)
            label = f"{VARIANT_LABEL[variant]} ep {ep}"
            with Heartbeat(f"[{env}] {label}", steps_per_grid, s_per_step,
                           args.heartbeat) as beat:
                grid_loss, used_idx, reused, wall = compute_or_load_grid(
                    env, variant, ep, model_dirs[variant], wm, train_cfg, val,
                    device, run_cfg, grids_dir, args.reuse_grids,
                    require_cached=args.viz_only,
                    # only fired on a real computation, never on a cache hit
                    on_start=lambda: beat.announce(
                        f"computing {label} ({run_cfg['grid']}x{run_cfg['grid']} x "
                        f"{run_cfg['opt_steps']} = {steps_per_grid:,} GD steps, ~"
                        f"{fmt_hms(steps_per_grid * s_per_step)} at "
                        f"{s_per_step:.3f}s/step; call {n_calls + 1}/{n_planned})"))
            kept = prior_wall.get((variant, ep))
            if kept is not None:                  # keep the computing run's timing
                wall = kept["wall_time_s"]
            if used_idx != ep:
                raise AssertionError(f"[{env}/{variant}] loss_landscape used episode "
                                     f"{used_idx}, requested {ep}")
            expected_shape = (run_cfg["grid"], run_cfg["grid"])
            if grid_loss.shape != expected_shape:
                raise AssertionError(f"[{env}/{variant}] grid shape "
                                     f"{grid_loss.shape} != {expected_shape}")
            if not np.all(np.isfinite(grid_loss)):
                raise AssertionError(f"[{env}/{variant}] non-finite landscape "
                                     f"on episode {ep}")
            grids[(ep, variant)] = grid_loss
            provenance[(ep, variant)] = {
                "episode_idx": int(used_idx), "grid": run_cfg["grid"],
                "opt_steps": run_cfg["opt_steps"], "lr": run_cfg["lr"],
                "action_range": run_cfg["action_range"], "goal_H": run_cfg["goal_H"],
                "val_len": len(val), "seq_length": int(val.get_seq_length(ep)),
            }
            metrics = grid_metrics(grid_loss)
            metrics_by_grid[(ep, variant)] = metrics
            row = {
                "env": env, "episode": ep, "variant": variant,
                "ckpt": os.path.relpath(model_dirs[variant], REPO),
                "roughness": metrics["roughness"],
                "roughness_interior": metrics["roughness_interior"],
                "local_minima": metrics["local_minima"],
                "min_loss": metrics["min_loss"],
                "loss_range": metrics["loss_range"],
                "median_over_min": metrics["median_over_min"],
                "max_over_min": metrics["max_over_min"],
                "basin_area_frac": metrics["basin_area_frac"],
                "argmin_i": metrics["argmin_i"], "argmin_j": metrics["argmin_j"],
                # action-space coordinates of the argmin (same axis as the figures)
                "argmin_ax": float(axs[metrics["argmin_j"]]),
                "argmin_ay": float(axs[metrics["argmin_i"]]),
                "argmin_on_edge": metrics["argmin_on_edge"],
                "border_share": metrics["border_share"],
                "wall_time_s": wall, "reused": int(reused), "smoke": int(args.smoke),
                **provenance[(ep, variant)],
            }
            if not is_pair:
                # the device that COMPUTED the grid, not the device of a later
                # reuse/viz-only pass (which only read the cache); the row is
                # written from cached grid + carried provenance
                row["device"] = (kept or {}).get("device") or str(device)
            (rows if is_pair else ctx_rows).append(row)
            n_calls += 1
            if not reused:
                n_computed += 1
                computed_wall += wall
                steps_computed += steps_per_grid
            if paths is not None:
                # incremental write: an interrupted multi-hour run still leaves a
                # usable table (the grids themselves are cached too)
                if is_pair:
                    write_per_episode_csv(paths["per_episode_csv"],
                                          (prior_rows or []) + rows)
                else:
                    write_context_csv(paths["context_csv"],
                                      (prior_context_rows or []) + ctx_rows)
            print(f"    [{env:16s} {VARIANT_LABEL[variant]:18s}] ep {ep:3d} "
                  f"(n={provenance[(ep, variant)]['seq_length']:3d}) "
                  f"roughness={metrics['roughness']:.4e} "
                  f"minima={metrics['local_minima']} "
                  f"min_loss={metrics['min_loss']:.4g} "
                  f"range={metrics['loss_range']:.3g} "
                  + ("[cached]" if reused else f"[{wall:.1f}s]"))
            remaining = n_planned - n_calls
            if remaining > 0 and n_computed > 0:
                avg = computed_wall / n_computed
                print(f"      ETA ~{avg * remaining / 60.0:.1f} min for the remaining "
                      f"{remaining} call(s) at {avg:.1f}s each")
        del wm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # the context panels are pinned to the same experiment as the pair, not assumed
    assert_paired_provenance(provenance, episodes, env,
                             variants=VARIANT_ORDER + ctx_names)
    paired = build_paired_rows(env, episodes, rows)
    summary = summarize_env(env, paired, rows)
    if len(episodes) < 2:
        print(f"  [note] [{env}] K={len(episodes)}: no paired win counts beyond "
              f"{len(episodes)}/{len(episodes)} and at most one illustrative figure.")
    return {"env": env, "episodes": episodes, "grids": grids, "rows": rows,
            "paired": paired, "summary": summary, "n_val": ref_len,
            "n_qualifying": len(ref_qual), "model_dirs": model_dirs,
            "metrics": metrics_by_grid, "context_rows": ctx_rows,
            "context_variants": ctx_names}


# ---------------------------------------------------------------------------
# Outputs: paths, CSVs, printed tables (stdlib csv; pandas is not in this env)
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = ["env", "variant", "mean_roughness", "std_roughness",
                   "paired_win_count_roughness", "mean_local_minima",
                   "std_local_minima", "paired_win_count_minima", "n_episodes",
                   "n_ties_roughness", "n_ties_minima", "mean_roughness_interior",
                   "std_roughness_interior", "paired_win_count_roughness_interior",
                   "mean_min_loss", "std_min_loss", "mean_loss_range",
                   "std_loss_range",
                   # structural geometry (see the module docstring, item 4)
                   "mean_basin_area_frac", "std_basin_area_frac",
                   "paired_win_count_basin_area", "paired_win_count_min_loss",
                   "mean_border_share", "std_border_share", "frac_argmin_on_edge"]

PER_EPISODE_COLUMNS = ["env", "variant", "episode", "roughness", "roughness_interior",
                       "local_minima", "seq_length", "grid", "opt_steps", "lr",
                       "action_range", "goal_H", "wall_time_s", "reused", "smoke", "ckpt",
                       "min_loss", "loss_range",
                       "median_over_min", "max_over_min", "basin_area_frac",
                       "argmin_i", "argmin_j", "argmin_ax", "argmin_ay",
                       "argmin_on_edge", "border_share"]

# The context table has exactly one extra column: the device its grids were computed
# on (they may come from a separate, possibly CPU, pass). The paired tables keep
# their column set unchanged, which is what makes the before/after --baseline diff
# a real check that context rows cannot leak into them.
CONTEXT_COLUMNS = PER_EPISODE_COLUMNS + ["device"]


def load_prior_provenance(path, env, run_cfg):
    """{(variant, episode): {"wall_time_s": float, "reused": int}} from an existing CSV.

    A reuse pass rebuilds metrics without recomputing grids, which must not rewrite
    the *provenance* of the run that actually produced them (its wall times). Rows
    are accepted only when their recorded settings match the current run, so a CSV
    from a different configuration cannot donate timings. A missing or unreadable
    CSV simply yields {} (nothing to preserve).
    """
    if not os.path.isfile(path):
        return {}
    want = {"env": env, "grid": str(run_cfg["grid"]), "opt_steps": str(run_cfg["opt_steps"]),
            "goal_H": str(run_cfg["goal_H"])}
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if any(str(row.get(k)) != v for k, v in want.items()):
                    continue
                try:
                    if float(row.get("lr", "nan")) != float(run_cfg["lr"]):
                        continue
                    if float(row.get("action_range", "nan")) != float(run_cfg["action_range"]):
                        continue
                    out[(row["variant"], int(row["episode"]))] = {
                        "wall_time_s": float(row["wall_time_s"]),
                        "reused": int(float(row["reused"])),
                        # present only in the context table; carried so that a later
                        # --viz-only pass cannot relabel a cuda grid as cpu (it only
                        # read the cache, it did not compute the grid)
                        "device": row.get("device", ""),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return out


def _display(path):
    """Path for log lines: relative to REPO when inside it, absolute otherwise."""
    abs_path = os.path.abspath(path)
    if abs_path.startswith(REPO + os.sep):
        return os.path.relpath(abs_path, REPO)
    return abs_path


def _fmt(value, spec=".6g"):
    """Format one metric for a CSV cell; nan/inf stay explicit, never blank."""
    if value is None:
        return ""
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def output_paths(args):
    """Resolve every output path (smoke runs get a `_smoke` suffix)."""
    suffix = "_smoke" if args.smoke else ""
    base = os.path.join(args.outdir, "loss_landscape")
    return {
        "dir": base,
        "figs": base,
        "grids": os.path.join(base, "grids"),
        "summary_csv": os.path.join(args.outdir, f"loss_landscape_comparison{suffix}.csv"),
        "per_episode_csv": os.path.join(base, f"per_episode{suffix}.csv"),
        # only written with --baseline; the paired tables never include these rows
        "context_csv": os.path.join(base, f"per_episode_context{suffix}.csv"),
        "suffix": suffix,
    }


def write_summary_csv(path, summary_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for r in summary_rows:
            w.writerow([
                r["env"], r["variant"],
                f"{r['mean_roughness']:.8g}", f"{r['std_roughness']:.8g}",
                r["paired_win_count_roughness"],
                f"{r['mean_local_minima']:.4g}", f"{r['std_local_minima']:.4g}",
                r["paired_win_count_minima"], r["n_episodes"],
                r["n_ties_roughness"], r["n_ties_minima"],
                f"{r['mean_roughness_interior']:.8g}",
                f"{r['std_roughness_interior']:.8g}",
                r["paired_win_count_roughness_interior"],
                f"{r['mean_min_loss']:.8g}", f"{r['std_min_loss']:.8g}",
                f"{r['mean_loss_range']:.8g}", f"{r['std_loss_range']:.8g}",
                _fmt(r["mean_basin_area_frac"], ".6g"),
                _fmt(r["std_basin_area_frac"], ".6g"),
                r["paired_win_count_basin_area"],
                r["paired_win_count_min_loss"],
                _fmt(r["mean_border_share"], ".6g"), _fmt(r["std_border_share"], ".6g"),
                r["frac_argmin_on_edge"],
            ])


def _per_episode_cells(r):
    """One row as CSV cells; order and formatting must match PER_EPISODE_COLUMNS."""
    return [
        r["env"], r["variant"], r["episode"],
        f"{r['roughness']:.8g}", f"{r['roughness_interior']:.8g}",
        r["local_minima"], r["seq_length"], r["grid"], r["opt_steps"],
        f"{r['lr']:g}", f"{r['action_range']:g}", r["goal_H"],
        f"{r['wall_time_s']:.3f}", r["reused"], r["smoke"], r["ckpt"],
        f"{r['min_loss']:.8g}", f"{r['loss_range']:.8g}",
        f"{r['median_over_min']:.6g}", f"{r['max_over_min']:.6g}",
        f"{r['basin_area_frac']:.6g}", r["argmin_i"], r["argmin_j"],
        f"{r['argmin_ax']:g}", f"{r['argmin_ay']:g}",
        r["argmin_on_edge"], _fmt(r["border_share"], ".6g"),
    ]


def _context_cells(r):
    """PER_EPISODE_COLUMNS + device: a context grid's provenance includes where it ran."""
    return _per_episode_cells(r) + [r.get("device", "")]


def _paired_sort_key(r):
    """env, then the compared variants in VARIANT_ORDER, then episode."""
    return (r["env"], VARIANT_ORDER.index(r["variant"]), r["episode"])


def write_per_episode_csv(path, rows, columns=None, cells=None, sort_key=None):
    """Write a per-(variant, episode) table; the paired table is the default.

    Columns, cell order, number formatting and row order are unchanged from before
    the context feature, which is what lets an old and a new per_episode.csv be
    diffed byte for byte.
    """
    columns = list(PER_EPISODE_COLUMNS if columns is None else columns)
    cells = _per_episode_cells if cells is None else cells
    sort_key = _paired_sort_key if sort_key is None else sort_key
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in sorted(rows, key=sort_key):
            w.writerow(cells(r))


def write_context_csv(path, rows):
    """The --baseline context table: same columns + device, no paired variant in it."""
    write_per_episode_csv(path, rows, columns=CONTEXT_COLUMNS, cells=_context_cells,
                          sort_key=lambda r: (r["env"], r["variant"], r["episode"]))


def print_summary_table(summary_rows):
    print("\n=== Terminal-loss-landscape smoothness (lower is better) ===")
    print(f"{'env':18s} {'variant':18s} {'mean_rough':>11s} {'std_rough':>10s} "
          f"{'win_rough':>9s} {'mean_min':>9s} {'std_min':>8s} {'win_min':>8s} {'K':>3s}")
    for r in summary_rows:
        print(f"{r['env']:18s} {VARIANT_LABEL[r['variant']]:18s} "
              f"{r['mean_roughness']:11.4e} {r['std_roughness']:10.4e} "
              f"{r['paired_win_count_roughness']:>9s} "
              f"{r['mean_local_minima']:9.2f} {r['std_local_minima']:8.2f} "
              f"{r['paired_win_count_minima']:>8s} {r['n_episodes']:3d}")


def print_headlines(summary_rows):
    """Plain-language per-env verdict from the paired counts (p-reg vs straighten)."""
    for env in sorted({r["env"] for r in summary_rows}):
        by_variant = {r["variant"]: r for r in summary_rows if r["env"] == env}
        straight = by_variant[VARIANT_ORDER[0]]
        p_reg = by_variant[VARIANT_ORDER[1]]
        n = straight["n_episodes"]
        direction = ("smoother" if p_reg["mean_roughness"] < straight["mean_roughness"]
                     else "rougher" if p_reg["mean_roughness"] > straight["mean_roughness"]
                     else "equal on average")
        print(f"  [{env}] {VARIANT_LABEL[VARIANT_ORDER[1]]} vs "
              f"{VARIANT_LABEL[VARIANT_ORDER[0]]}: roughness is {direction} "
              f"({p_reg['mean_roughness']:.4e} vs {straight['mean_roughness']:.4e}), "
              f"wins {p_reg['paired_win_count_roughness']} episodes; "
              f"local minima wins {p_reg['paired_win_count_minima']} episodes "
              f"({p_reg['mean_local_minima']:.2f} vs {straight['mean_local_minima']:.2f})")
        print(f"  [{env}] structural read (roughness alone cannot separate these): "
              f"min_loss wins {p_reg['paired_win_count_min_loss']} episodes for p-reg, "
              f"basin-area wins {p_reg['paired_win_count_basin_area']} episodes "
              f"(mean {100 * p_reg['mean_basin_area_frac']:.0f}% of cells vs "
              f"{100 * straight['mean_basin_area_frac']:.0f}%), argmin on the box edge "
              f"{p_reg['frac_argmin_on_edge']} (p-reg) vs "
              f"{straight['frac_argmin_on_edge']} (straighten)")


# ---------------------------------------------------------------------------
# Paired diagnostics: is any of this bigger than episode noise?
# ---------------------------------------------------------------------------
# The table above reports mean/std + win counts, which is easy to over-read at
# small K: with K=6 the two-sided sign test needs *unanimity* to reach p<0.05, so
# a 3/3 split and a 6/0 split look equally "mixed" in a mean/std table. These
# helpers add the exact paired tests, a log-scale effect size with a bootstrap
# CI, the 80%-power minimum detectable ratio, and a guard against the one way the
# primary metric can mislead (a low-contrast grid scores as "smooth").
#
# `lower_is` documents what "won" means for a metric: lower is better for the
# smoothness/optimum metrics, but loss_range is a contrast descriptor -- a
# *smaller* max-min loss is not better, since a nearly constant grid is exactly
# the surface that gives an optimizer nothing to descend.
DIAGNOSTIC_METRICS = [
    ("roughness", "roughness", "better"),
    ("roughness_interior", "roughness_interior", "better"),
    ("local_minima", "local_minima", "better"),
    ("min_loss", "min_loss", "better"),
    ("loss_range", "loss_range", "no"),
    # basin_area_frac is "no" for the same reason as loss_range: it is a
    # structural descriptor, not a quality score. A big basin around a bad
    # optimum is the flat-grid trap, so the ratio below is only meaningful read
    # together with the min_loss row.
    ("basin_area_frac", "basin_area", "no"),
]
DIAGNOSTIC_BOOT = 20000


def paired_metric_stats(paired, key):
    """Exact paired tests + log-scale effect size for one metric.

    Counts which variant is lower on how many episodes, then reports the exact
    sign test, the Wilcoxon signed-rank test and the P/S ratio as a median and as
    a geometric mean with a paired bootstrap CI (episodes resampled with
    replacement, fixed seed, so the number is reproducible).
    """
    a, b = VARIANT_ORDER                          # a = straighten, b = p_reg
    va = np.array([p[key][a] for p in paired], dtype=np.float64)
    vb = np.array([p[key][b] for p in paired], dtype=np.float64)
    d = vb - va                                   # > 0: straighten is lower
    n = len(d)
    n_a_low = int(np.sum(d > 0))
    n_b_low = int(np.sum(d < 0))
    usable = n_a_low + n_b_low
    p_sign = (float(stats.binomtest(min(n_a_low, n_b_low), usable, 0.5).pvalue)
              if usable else float("nan"))
    try:
        p_wilcox = float(stats.wilcoxon(d).pvalue)
    except ValueError:                            # every difference is exactly 0
        p_wilcox = float("nan")
    out = {"n": n, "n_a_low": n_a_low, "n_b_low": n_b_low, "n_ties": n - usable,
           "p_sign": p_sign, "p_wilcoxon": p_wilcox,
           "median_ratio": float("nan"), "geomean_ratio": float("nan"),
           "ci_low": float("nan"), "ci_high": float("nan"),
           "sd_log_ratio": float("nan")}
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(vb / va)
    if np.all(np.isfinite(log_ratio)):            # false if a value is 0 (flat grid)
        draws = np.random.default_rng(0).integers(0, n, size=(DIAGNOSTIC_BOOT, n))
        boot = log_ratio[draws].mean(axis=1)
        out.update(
            median_ratio=float(np.median(np.exp(log_ratio))),
            geomean_ratio=float(np.exp(log_ratio.mean())),
            ci_low=float(np.exp(np.percentile(boot, 2.5))),
            ci_high=float(np.exp(np.percentile(boot, 97.5))),
            sd_log_ratio=float(np.std(log_ratio, ddof=1)) if n > 1 else float("nan"),
        )
    return out


def _mde_ratio(sd_log, n):
    """Smallest P/S ratio an n-episode run has 80% power to detect (two-sided t)."""
    if not np.isfinite(sd_log) or n < 2:
        return float("nan")
    return float(np.exp((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1))
                        * sd_log / np.sqrt(n)))


def print_paired_diagnostics(env, paired):
    """Which variant is lower, how significant, how big, and what not to claim."""
    n = len(paired)
    if n == 0:
        return
    a, b = VARIANT_ORDER
    print(f"\n  --- paired diagnostics [{env}], K={n} "
          f"(the two 'lower' columns are 'lower on k/{n} episodes') ---")
    print(f"    {'metric':18s} {'lower':>5s} {a[:10]:>11s} {b[:10]:>11s} {'ties':>4s} "
          f"{'sign p':>7s} {'wilcox p':>8s} {'med P/S':>8s} "
          f"{'geo P/S [95% CI]':>22s}")
    for key, label, lower_is in DIAGNOSTIC_METRICS:
        st = paired_metric_stats(paired, key)
        a_str, b_str = f"{st['n_a_low']}/{n}", f"{st['n_b_low']}/{n}"
        if np.isfinite(st["geomean_ratio"]):
            ratio = (f"{st['geomean_ratio']:.3g}x [{st['ci_low']:.3g}x, "
                     f"{st['ci_high']:.3g}x]")
            med = f"{st['median_ratio']:.3g}x"
        else:
            ratio, med = "n/a (zero grid)", "n/a"
        print(f"    {label:18s} {lower_is:>5s} {a_str:>11s} {b_str:>11s} "
              f"{st['n_ties']:>4d} {st['p_sign']:>7.3f} {st['p_wilcoxon']:>8.3f} "
              f"{med:>8s} {ratio:>22s}")
    crit = max((t for t in range(n // 2 + 1)
                if stats.binomtest(t, n, 0.5).pvalue <= 0.05), default=None)
    if crit is not None:
        print(f"    [#] K={n}: the two-sided sign test only reaches p<0.05 at "
              f"{n - crit}/{n} (or {crit}/{n}), so at this K a {n // 2}/{n // 2} split is "
              f"uninformative (min achievable p = "
              f"{stats.binomtest(0, n, 0.5).pvalue:.3f}); 'wins k/{n}' alone is not "
              "evidence.")
    rough = paired_metric_stats(paired, "roughness")
    for k in (n, 4 * n):
        mde = _mde_ratio(rough["sd_log_ratio"], k)
        if np.isfinite(mde):
            print(f"    [#] 80%-power minimum detectable P/S roughness ratio: {mde:.3g}x "
                  f"at K={k} (observed spread: sd of log P/S = "
                  f"{rough['sd_log_ratio']:.2f}).")
    if n >= 3:
        rng = np.concatenate([[p["loss_range"][v] for p in paired] for v in (a, b)])
        ro = np.concatenate([[p["roughness"][v] for p in paired] for v in (a, b)])
        rho, rho_p = stats.spearmanr(rng, ro)
        agree = sum(1 for p in paired
                    if (p["loss_range"][a] - p["loss_range"][b])
                    * (p["roughness"][a] - p["roughness"][b]) > 0)
        print(f"    [#] roughness is not contrast-free: Spearman(max-min loss, roughness) "
              f"= {rho:+.2f} (p={rho_p:.3f}) over the {2 * n} grids, and the "
              f"higher-contrast variant is the rougher one on {agree}/{n} episodes. A grid "
              "that is nearly constant -- no attained optimum inside the swept box -- "
              "therefore scores as 'smooth': read the min_loss/loss_range rows next to "
              "the roughness rows.")


def print_geometry_diagnostics(env, env_result, run_cfg):
    """What a Laplacian cannot see: censoring, padding share, basin geometry.

    Printed only; nothing is recomputed. `roughness` is not interpretable alone (a
    flat grid scores as "smooth", one steep basin scores as "rough"), so this
    block forces the structural facts next to it: whether the swept box even
    contains the optimum (argmin on the edge = censored), how much of each
    roughness number is reflected boundary padding, how much of the box is
    near-optimal, and how narrow the optimum is (parabola half-width at 2x loss).
    """
    paired = env_result["paired"]
    n = len(paired)
    if n == 0:
        return
    axs = landscape_axis(run_cfg)
    a, b = VARIANT_ORDER
    print(f"\n  --- grid geometry [{env}], K={n} (box "
          f"[{-run_cfg['action_range']:g}, {run_cfg['action_range']:g}]^2, "
          f"{run_cfg['grid']}x{run_cfg['grid']} cells, "
          f"{100.0 / run_cfg['grid'] ** 2:.1f}% per cell) ---")
    print(f"    {'ep':>4s} {'variant':18s} {'min_loss':>10s} {'med/min':>8s} "
          f"{'basin@1.1x':>10s} {'border':>7s} {'argmin (ax,ay)':>16s} {'edge':>4s} "
          f"{'half-width@2x (ax, ay)':>24s}")
    for p in paired:
        ep = p["episode"]
        for v in VARIANT_ORDER:
            g = env_result["grids"][(ep, v)]
            m = env_result["metrics"][(ep, v)]
            widths = []
            for axis_id in (0, 1):
                line, k = profile_through_argmin(g, m, axis_id)
                hw, ok, one_sided = fit_basin_width(line, k, axs)
                widths.append(f"{hw:.2f}{'!' if one_sided else ''}" if ok else "n/a")
            print(f"    {ep:>4d} {VARIANT_LABEL[v]:18s} {m['min_loss']:10.4g} "
                  f"{m['median_over_min']:8.2f} "
                  f"{100 * m['basin_area_frac']:9.0f}% "
                  f"{_fmt(m['border_share'], '+.2f'):>7s} "
                  f"({axs[m['argmin_j']]:+.2f}, {axs[m['argmin_i']]:+.2f}) "
                  f"{'yes' if m['argmin_on_edge'] else 'no':>4s} "
                  f"{widths[0]:>10s} {widths[1]:>10s}")
    n_edge = {v: sum(1 for p in paired if p["argmin_on_edge"][v]) for v in (a, b)}
    if n_edge[a] or n_edge[b]:
        print(f"    [#] argmin on the box edge: {n_edge[a]}/{n} straighten, "
              f"{n_edge[b]}/{n} p-reg. For those grids the box does not contain the "
              "optimum, so min_loss is a censored lower bound and any width fitted "
              "at that point ('!' above) describes the wall, not a basin. The two "
              "variants censored on the same episode are still comparable; widen "
              "--action-range to test whether the optimum is inside the box at all.")
    for v in (a, b):
        bs = _mean_finite([p["border_share"][v] for p in paired])
        ba = _mean_finite([p["basin_area_frac"][v] for p in paired])
        print(f"    [#] {VARIANT_LABEL[v]:18s} border_share mean {bs:+.2f} "
              f"(>0.9 would mean roughness is the reflected boundary padding; "
              f"cross-check the roughness_interior row above), basin@1.1x mean "
              f"{100 * ba:.0f}% of the box.")
    flat = [(p["episode"], v) for p in paired for v in (a, b)
            if p["basin_area_frac"][v] >= 0.99 and p["median_over_min"][v] < 1.05]
    if flat:
        print(f"    [#] {len(flat)} grid(s) are flat to within 5% over >=99% of the "
              f"box ({', '.join(f'ep{e}/{v}' for e, v in flat)}): those are the grids "
              "where 'smooth' means 'no signal', not 'good landscape'.")


# ---------------------------------------------------------------------------
# Illustrative paired figures (chosen from the metrics, never from the plots)
# ---------------------------------------------------------------------------
def landscape_axis(run_cfg):
    """The action grid axis, built exactly as loss_landscape builds it."""
    return np.linspace(-run_cfg["action_range"], run_cfg["action_range"], run_cfg["grid"])


def profile_through_argmin(grid_loss, metrics, axis_id):
    """The 1-D cut of the grid through its own argmin along one axis.

    axis_id 0 cuts along a_x (one grid row), 1 along a_y (one column). Returns
    (line, k) -- the cut and the index of the argmin along it. Plotting the cut
    against `axis - axis[k]` is what makes two variants with different optima
    directly comparable: each curve reads "moving away from that variant's own
    best start", so the comparison is about basin shape, not about where the
    optimum happened to sit.
    """
    g = np.asarray(grid_loss, dtype=np.float64)
    k = metrics["argmin_i"] if axis_id == 0 else metrics["argmin_j"]
    line = g[k, :] if axis_id == 0 else g[:, k]
    return line, k


def fit_basin_width(line, k, axis, half_window=3):
    """Parabola through the argmin: at what offset would the loss double?

    Fits L = c + a*d^2 to at most `half_window` cells on each side of index k
    (clamped to the grid, so the fit is one-sided when the argmin is on the box
    edge) and returns (half_width_double, ok, one_sided). The number is in action
    units: "moving the first action this far from its optimum at least doubles the
    attainable terminal loss". It is a local shape descriptor, nothing more --
    one_sided=True means there was no surface outside the box to fit, so the value
    is a fit to the wall and must not be read as a basin width.
    """
    line = np.asarray(line, dtype=np.float64)
    k = int(k)
    lo_i, hi_i = max(0, k - half_window), min(line.size - 1, k + half_window)
    one_sided = bool(k in (0, line.size - 1))
    x = np.asarray(axis, dtype=np.float64)[lo_i:hi_i + 1] - float(axis[k])
    y = line[lo_i:hi_i + 1]
    if x.size < 3:
        return float("nan"), False, one_sided
    coef = np.polyfit(x, y, 2)
    a2 = float(coef[0])
    c = float(np.polyval(coef, 0.0))
    if not np.isfinite(a2) or a2 <= 0.0 or c <= 0.0:
        return float("nan"), False, one_sided
    return float(np.sqrt(c / a2)), True, one_sided


def basin_factors(n=64, lo=1.0, hi=2.0):
    """Depth factors for the basin-area curve P(L <= f * L_min)."""
    return np.linspace(lo, hi, n)


def select_illustrative_episodes(paired, n_viz):
    """Pick up to n_viz episodes with a fixed, metric-only rule.

    The rule is keyed to the *decidable* metric, not to `roughness`: at K=6 the
    roughness sign test cannot reach significance (6/6 needed) and roughness
    partly tracks grid contrast, so selecting examples by it selects by noise.
    With r_i = min_loss(straighten) / min_loss(p-reg) on episode i, the rule takes
      * the episode with the largest r_i -- p-reg attains the lowest optimum
        relative to straighten (the claim), and
      * the episode with the smallest r_i -- the worst case for that claim.
    Both tags are **value-conditional**: the second episode is only called a
    counterexample when r_i < 1, i.e. when straighten really does attain the lower
    optimum. On these grids it never does (r_i > 1 on all 6 episodes), so the tag
    says so explicitly rather than labelling the weakest win a counterexample --
    a figure must not claim a direction the numbers do not have. If no r_i is
    usable (>1) at all, the first tag says the comparison found no advantage and
    the figure is just a picture of the cached grids.
    Further slots are back-filled by |log r_i|, largest first, in episode order.
    Returns ([(episode, tag), ...], stats) where stats describes the selection
    metric that was actually used, so the caller can print it truthfully.
    """
    a, b = VARIANT_ORDER
    ratio = {}
    for p in paired:
        lo_a, lo_b = p["min_loss"][a], p["min_loss"][b]
        if lo_a > 0 and lo_b > 0 and np.isfinite(lo_a) and np.isfinite(lo_b):
            ratio[p["episode"]] = float(lo_a / lo_b)
    if not ratio:
        # every grid is flat/degenerate (min_loss == 0): fall back to the episode
        # order, which is all that remains meaningful on such grids.
        chosen = [(p["episode"], "flat grids: no min_loss ratio available")
                  for p in paired[:n_viz]]
        return chosen, {"ratio": {}, "geomean": float("nan"), "n_p_reg_lower": 0,
                        "n": len(paired), "fallback": True}
    stats = {
        "ratio": ratio,
        "geomean": float(np.exp(np.mean([np.log(r) for r in ratio.values()]))),
        "n_p_reg_lower": sum(1 for r in ratio.values() if r > 1.0),
        "n": len(ratio),
        "fallback": False,
    }
    by_ratio = sorted(ratio, key=lambda e: ratio[e])          # ascending ratio
    best, worst = by_ratio[-1], by_ratio[0]
    r_best, r_worst = ratio[best], ratio[worst]
    # The tag has to be decided by the value, not by the slot: "counterexample"
    # is a claim about the direction of r_i, and it is false whenever r_i > 1.
    if r_best > 1.0:
        chosen = [(best, "largest p-reg min-loss advantage "
                   f"(min_loss straighten/p-reg = {r_best:.2f}x)")]
    else:
        chosen = [(best, "no p-reg min-loss advantage on any episode "
                   f"(best straighten/p-reg ratio {r_best:.2f}x)")]
    if len(chosen) < n_viz and worst != best:
        if r_worst < 1.0:
            chosen.append((worst, "worst case for that claim (counterexample): "
                           f"straighten attains the lower optimum "
                           f"({r_worst:.2f}x)"))
        else:
            chosen.append((worst, "weakest p-reg min-loss advantage "
                           f"({r_worst:.2f}x) — no counterexample among K="
                           f"{len(ratio)}: p-reg attains the lower optimum on "
                           "every episode"))
    for ep in sorted(ratio, key=lambda e: -abs(np.log(ratio[e]))):
        if len(chosen) >= n_viz:
            break
        if all(ep != c[0] for c in chosen):
            chosen.append((ep, "back-fill by |log min_loss ratio|"))
    return chosen[:n_viz], stats


def plot_env_examples(env, env_result, args, run_cfg, paths):
    """Paired heatmaps for the illustrative episodes; returns the written paths."""
    chosen, stats = select_illustrative_episodes(env_result["paired"], args.n_viz)
    if stats["fallback"]:
        print(f"  [{env}] illustrative episodes: {[e for e, _ in chosen]} "
              "(no usable min_loss ratio: every grid is degenerate)")
    else:
        chosen_ratio = ", ".join(f"ep{e}: {stats['ratio'][e]:.2f}x" for e, _ in chosen
                                 if e in stats["ratio"])
        print(f"  [{env}] illustrative episodes: {[e for e, _ in chosen]} "
              f"(rule: min_loss(straighten)/min_loss(p-reg), geomean "
              f"{stats['geomean']:.2f}x over K={stats['n']}, p-reg lower on "
              f"{stats['n_p_reg_lower']}/{stats['n']}; picked {chosen_ratio})")
    for _, tag in chosen:
        print(f"    - {tag}")
    written = []
    axs = landscape_axis(run_cfg)
    for ep, tag in chosen:
        seq_len = next(p["seq_length"] for p in env_result["paired"] if p["episode"] == ep)
        suptitle = (f"{env}: episode {ep} (seq_length {seq_len}) — illustrative example, "
                    f"{tag}\naggregate evidence is loss_landscape_comparison"
                    f"{paths['suffix']}.csv, not this figure")
        path = plot_landscape(
            env,
            [{"label": f"{VARIANT_LABEL[v]} (ep {ep})",
              "axs": axs, "grid_loss": env_result["grids"][(ep, v)]}
             for v in VARIANT_ORDER],
            paths["figs"], fname=f"landscape_{env}_ep{ep:03d}{paths['suffix']}.png",
            suptitle=suptitle, figsize=(12, 6),
            # keep the shared colorbar outside the two panels (matplotlib < 3.6
            # otherwise leaves it lying across the right-hand panel)
            cbar_outside=True)
        written.append(path)
        print(f"      figure -> {_display(path)}")
    return written


def plot_env_fig4(env, env_result, args, run_cfg, paths):
    """Paper-style Fig. 4 view: one landscape panel per arm, one row per episode.

    The paper's figure is a row of saturated landscape panels with iso-loss contours;
    this renders the same cached grids the same way, with the context (baseline)
    panel first when --baseline supplied its grids. It is a *view*: every number on
    the panels is computed on the raw grid, and bicubic interpolation only smooths
    the pixels for display (the plain figures and every CSV stay nearest-neighbour
    raw-grid data).

    Three things a paper-style panel tends to hide are labelled instead:
      * the attained minimum is in each title, because a shared 99th-percentile
        colour scale cannot show a level difference (that is what the diagnostics
        figure and the CSVs are for),
      * the 1.1x-L_min boundary is drawn, or reported as off scale when it falls
        outside the clipped range, and
      * an argmin on the box edge is marked: its "basin" is a wall, so its
        min_loss is a censored lower bound.
    """
    chosen, _ = select_illustrative_episodes(env_result["paired"], args.n_viz)
    axs = landscape_axis(run_cfg)
    ext = [axs[0], axs[-1], axs[0], axs[-1]]
    written = []
    for ep, tag in chosen:
        # baseline first (the paper's "(a) DINOv2" slot), then the compared pair, so
        # the row reads left to right as no-straightening -> straighten -> +p-reg
        names = [v for v in list(env_result.get("context_variants", [])) + VARIANT_ORDER
                 if (ep, v) in env_result["grids"]]
        if not names:
            continue
        pair = next(p for p in env_result["paired"] if p["episode"] == ep)
        grids = [np.asarray(env_result["grids"][(ep, v)], dtype=np.float64) for v in names]
        mets = [env_result["metrics"][(ep, v)] for v in names]
        pooled = np.concatenate([g.ravel() for g in grids])
        vmin, vmax = float(np.nanmin(pooled)), float(np.nanpercentile(pooled, 99))
        # 6 iso-loss levels, shared by every panel of this episode (the same levels
        # the shared colourbar is read with); a degenerate all-constant grid has no
        # increasing level set, so the contours are skipped and said to be skipped
        levels = np.unique(np.linspace(vmin, vmax, 8)[1:-1])
        draw_levels = levels.size >= 2
        fig, axes = plt.subplots(1, len(names), squeeze=False,
                                 figsize=(5.2 * len(names) + 1.4, 6.0))
        axes = axes[0]
        im = None
        for ax, name, g, m in zip(axes, names, grids, mets):
            im = ax.imshow(g, origin="lower", aspect="equal", extent=ext, cmap="magma",
                           vmin=vmin, vmax=vmax, interpolation="bicubic")
            l_min = float(m["min_loss"])
            notes = []
            if draw_levels:
                ax.contour(axs, axs, g, levels=levels, colors="w", linewidths=0.6,
                           alpha=0.7)
            else:
                notes.append("flat grid: no iso-loss contours")
            if l_min > 0 and BASIN_FACTOR * l_min <= vmax:
                ax.contour(axs, axs, g, levels=[BASIN_FACTOR * l_min], colors="cyan",
                           linewidths=1.3)
                notes.append(f"cyan: {BASIN_FACTOR:g}$\\times L_{{min}}$")
            elif l_min > 0:
                notes.append(f"{BASIN_FACTOR:g}$\\times L_{{min}}$ off the clipped scale")
            else:
                notes.append(f"$L_{{min}}$=0: no {BASIN_FACTOR:g}$\\times L_{{min}}$ contour")
            ax.plot(axs[m["argmin_j"]], axs[m["argmin_i"]], marker="*", ms=15,
                    mfc="none", mec="cyan", mew=1.3, ls="none")
            if m["argmin_on_edge"]:
                notes.append("argmin ON the box edge (censored)")
            ax.set_title(f"{VARIANT_LABEL[name]}\n"
                         f"min terminal loss {l_min:.3g} · basin@"
                         f"{BASIN_FACTOR:g}$\\times L_{{min}}$ = "
                         f"{100 * m['basin_area_frac']:.0f}% of cells\n"
                         + "  ·  ".join(notes), fontsize=9.5)
            ax.set_xlabel("first action $a_x$ (normalized)")
        axes[0].set_ylabel("first action $a_y$ (normalized)")
        ratio = pair["min_loss"][VARIANT_ORDER[0]] / pair["min_loss"][VARIANT_ORDER[1]]
        ctx_note = (f" + {len(env_result['context_variants'])} reference panel(s)"
                    if env_result.get("context_variants")
                    else " (run with --baseline to add the reference panel)")
        fig.suptitle(
            f"{env} episode {ep} (seq_length {pair['seq_length']}) — paper-style "
            f"Fig. 4 view{ctx_note}\n"
            f"min_loss straighten/p-reg = {ratio:.2f}x (>1: p-reg attains the deeper "
            f"optimum) · {tag}\n"
            f"display: bicubic interpolation of the raw {run_cfg['grid']}x"
            f"{run_cfg['grid']} swept grid; every number is computed on the raw grid, "
            f"and the aggregate evidence is "
            f"loss_landscape_comparison{paths['suffix']}.csv, not this figure",
            fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.colorbar(im, ax=list(axes), fraction=0.022, pad=0.02,
                     label="min terminal loss (shared 99th-pct scale)")
        path = os.path.join(paths["figs"], f"fig4_{env}_ep{ep:03d}{paths['suffix']}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f"      fig4 figure -> {_display(path)}")
    return written


# ---------------------------------------------------------------------------
# Diagnostic figures (--diagnostics / --viz-only): the views a single shared
# colorbar cannot give. Nothing here is a new measurement -- every panel is the
# same cached grid_loss, drawn so the two ways the shared-scale figure can hide
# the comparison stay visible:
#   * one 99th-percentile-clipped scale can saturate one variant and flatten the
#     other into a featureless plate (the ep-3 grids use 18% and 13% of it), and
#   * two nearly identical *levels* drawn in separate axes make the level gap
#     (exactly what a shared bar cannot show) invisible.
# ---------------------------------------------------------------------------
def _variant_styles():
    """Stable plot style per variant (same colours in every diagnostic figure)."""
    return {VARIANT_ORDER[0]: {"color": "#1f77b4", "marker": "o"},
            VARIANT_ORDER[1]: {"color": "#d62728", "marker": "s"}}


def plot_env_diagnostics(env, env_result, args, run_cfg, paths):
    """Per-episode 2x3 heatmaps: shared scale / autoscaled / (L - L_min) + contours.

    Row = variant, column = view. The third column is the one the paper's figure
    cannot show: both variants on ONE absolute depth scale (L - L_min) with fixed
    depth contours, so "how deep is the optimum and how much of the box is within
    x of it" is directly comparable instead of being an artifact of shared colour
    limits.
    """
    axs = landscape_axis(run_cfg)
    ext = [axs[0], axs[-1], axs[0], axs[-1]]
    written = []
    for p in env_result["paired"]:
        ep = p["episode"]
        grids = [np.asarray(env_result["grids"][(ep, v)], dtype=np.float64)
                 for v in VARIANT_ORDER]
        mets = [env_result["metrics"][(ep, v)] for v in VARIANT_ORDER]
        pooled = np.concatenate([g.ravel() for g in grids])
        vmin, vmax = float(np.nanmin(pooled)), float(np.nanpercentile(pooled, 99))
        depth_max = max(float(g.max() - g.min()) for g in grids)
        shrink = {v: ((grids[i].max() - grids[i].min()) / (vmax - vmin)
                      if vmax > vmin else 0.0)
                  for i, v in enumerate(VARIANT_ORDER)}
        levels = [depth_max * f for f in DEPTH_CONTOURS]
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 10))
        for row, (v, m) in enumerate(zip(VARIANT_ORDER, mets)):
            g = grids[row]
            depth = g - g.min()
            edge_note = (" · argmin ON the box edge (censored)"
                         if m["argmin_on_edge"] else "")
            im0 = axes[row, 0].imshow(g, origin="lower", aspect="auto", extent=ext,
                                      cmap="magma", vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(
                f"{VARIANT_LABEL[v]}: shared scale (99th pct clipped)\n"
                f"min {m['min_loss']:.3g}, median/min {m['median_over_min']:.2f}x, "
                f"own span = {100 * shrink[v]:.0f}% of the scale")
            im1 = axes[row, 1].imshow(g, origin="lower", aspect="auto", extent=ext,
                                      cmap="magma")
            axes[row, 1].set_title(
                f"{VARIANT_LABEL[v]}: autoscaled to itself\n"
                f"span {m['loss_range']:.3g}, shape only, no level information")
            im2 = axes[row, 2].imshow(depth, origin="lower", aspect="auto", extent=ext,
                                      cmap="viridis", vmin=0.0, vmax=depth_max)
            cs = axes[row, 2].contour(axs, axs, depth, levels=levels, colors="w",
                                      linewidths=0.7, alpha=0.75)
            axes[row, 2].clabel(cs, fmt="%.2g", fontsize=6)
            axes[row, 2].plot(axs[m["argmin_j"]], axs[m["argmin_i"]], marker="+",
                              color="w", ms=12, mew=2)
            axes[row, 2].set_title(
                f"{VARIANT_LABEL[v]}: L - L_min on one absolute depth scale\n"
                f"basin@1.1x = {100 * m['basin_area_frac']:.0f}% of cells, max depth "
                f"{depth_max:.3g}{edge_note}")
            for col in range(3):
                axes[row, col].set_xlabel("first action $a_x$ (normalized)")
                axes[row, col].set_ylabel("first action $a_y$ (normalized)")
                axes[row, col].set_facecolor("#f0f0f0")
            fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02,
                         label="min terminal loss (shared)")
            fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.02,
                         label="min terminal loss (own scale)")
            fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.02,
                         label="L - L_min (shared depth)")
        ratio = p["min_loss"][VARIANT_ORDER[0]] / p["min_loss"][VARIANT_ORDER[1]]
        fig.suptitle(
            f"{env} episode {ep} (seq_length {p['seq_length']}) — same cached grid "
            f"drawn three ways\nmin_loss straighten/p-reg = {ratio:.2f}x (p-reg deeper "
            f"when >1), basin@1.1x {100 * mets[0]['basin_area_frac']:.0f}% vs "
            f"{100 * mets[1]['basin_area_frac']:.0f}%, roughness "
            f"{mets[0]['roughness']:.3g} vs {mets[1]['roughness']:.3g} — aggregate "
            f"evidence is loss_landscape_comparison{paths['suffix']}.csv", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = os.path.join(paths["figs"],
                            f"diagnostics_{env}_ep{ep:03d}{paths['suffix']}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f"      diagnostics figure -> {_display(path)}")
    return written


def plot_env_profiles(env, env_result, args, run_cfg, paths):
    """Per-episode profiles through each variant's own argmin + the basin curve.

    Left/middle: L/L_min against the offset from that variant's argmin along a_x
    (left) and a_y (middle). Normalising by each variant's own minimum and
    centring on its own argmin removes both the level gap and the "where did the
    optimum sit" difference, so the curves answer exactly one question: how fast
    does the attainable loss grow as the first action moves away from its best
    value? The dotted line is L = 1.1 L_min (the basin threshold) and the legend
    carries the fitted "doubles at" half-width, marked ! when the argmin is on the
    box edge and the fit therefore sees only the wall.
    Right: the basin-area curve P(L <= f L_min) vs f -- the contrast-free version
    of basin_area_frac.
    """
    axs = landscape_axis(run_cfg)
    style = _variant_styles()
    factors = basin_factors()
    written = []
    for p in env_result["paired"]:
        ep = p["episode"]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        for v in VARIANT_ORDER:
            g = np.asarray(env_result["grids"][(ep, v)], dtype=np.float64)
            m = env_result["metrics"][(ep, v)]
            for axis_id, ax in enumerate(axes[:2]):
                line, k = profile_through_argmin(g, m, axis_id)
                hw, ok, one_sided = fit_basin_width(line, k, axs)
                label = (f"{VARIANT_LABEL[v]} (min {m['min_loss']:.3g}; doubles at "
                         f"{'n/a' if not ok else format(hw, '.2f')}"
                         f"{'!' if one_sided else ''} from its argmin)")
                ax.plot(axs - axs[k], line / m["min_loss"], color=style[v]["color"],
                        marker=".", ms=4, lw=1.3, label=label)
                ax.axhline(BASIN_FACTOR, color="grey", ls=":", lw=1)
            axes[2].plot(factors, 100 * basin_curve(g, factors), color=style[v]["color"],
                         lw=1.6, label=f"{VARIANT_LABEL[v]} (at f=1.1: "
                                       f"{100 * m['basin_area_frac']:.0f}% of cells)")
        for axis_id, ax in enumerate(axes[:2]):
            ax.axhline(1.0, color="k", lw=0.8)
            ax.set_yscale("log")
            ax.set_xlabel(f"offset from that variant's own argmin in "
                          f"$a_{'x' if axis_id == 0 else 'y'}$")
            ax.set_ylabel("$L / L_{min}$ (log)")
            ax.set_title(f"profile along $a_{'x' if axis_id == 0 else 'y'}$ through each "
                         f"variant's own argmin\n(dotted: {BASIN_FACTOR:g}x threshold)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[2].axvline(BASIN_FACTOR, color="grey", ls=":")
        axes[2].set_xlabel("depth factor $f$")
        axes[2].set_ylabel("$\\%$ of the box within $f \\cdot L_{min}$")
        axes[2].set_title("basin-area curve\n(contrast-free basin size)")
        axes[2].grid(alpha=0.25)
        axes[2].legend(fontsize=8)
        fig.suptitle(f"{env} episode {ep} — basin shape from the cached grid, "
                     f"aggregate evidence is loss_landscape_comparison"
                     f"{paths['suffix']}.csv", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = os.path.join(paths["figs"],
                            f"profiles_{env}_ep{ep:03d}{paths['suffix']}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f"      profiles figure -> {_display(path)}")
    return written


def plot_env_summary(env, env_result, run_cfg, paths):
    """Six panels: the structural read of this env in one paired figure.

    (1) min_loss straighten vs p-reg, log-log, with the y=x line; (2) the
    roughness-vs-contrast confound that makes the primary metric uninterpretable
    alone; (3) per-episode min_loss ratio, so a single episode dominating the
    aggregate is visible instead of averaged away; (4) basin area; (5) plateau
    height; (6) border share (how much of roughness is the padding). Every panel
    is drawn from the same cached grids as the CSVs.
    """
    paired = env_result["paired"]
    a, b = VARIANT_ORDER
    eps = [p["episode"] for p in paired]
    x = np.arange(len(eps))
    style = _variant_styles()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_scatter, ax_conf, ax_ratio), (ax_basin, ax_plateau, ax_border) = axes

    la = np.array([p["min_loss"][a] for p in paired], dtype=np.float64)
    lb = np.array([p["min_loss"][b] for p in paired], dtype=np.float64)
    ax_scatter.scatter(la, lb, color=style[b]["color"], zorder=3)
    lim = [min(la.min(), lb.min()) * 0.8, max(la.max(), lb.max()) * 1.25]
    ax_scatter.plot(lim, lim, color="k", lw=0.9, ls="--", label="equal min_loss")
    for ep, xa, xb in zip(eps, la, lb):
        ax_scatter.annotate(f"ep{ep}", (xa, xb), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel(f"{VARIANT_LABEL[a]} min_loss (log)")
    ax_scatter.set_ylabel(f"{VARIANT_LABEL[b]} min_loss (log)")
    ratio = la / lb
    ax_scatter.set_title(f"attained minimum per episode: p-reg lower on "
                         f"{int(np.sum(ratio > 1))}/{len(eps)} episodes\ngeometric "
                         f"mean S/P = {float(np.exp(np.mean(np.log(ratio)))):.2f}x "
                         "(>1 = p-reg deeper)")
    ax_scatter.grid(alpha=0.25, which="both")
    ax_scatter.legend(fontsize=8)

    rng = np.concatenate([[p["loss_range"][v] for p in paired] for v in (a, b)])
    ro = np.concatenate([[p["roughness"][v] for p in paired] for v in (a, b)])
    rho, rho_p = stats.spearmanr(rng, ro)
    for v in (a, b):
        ax_conf.scatter([p["loss_range"][v] for p in paired],
                        [p["roughness"][v] for p in paired], label=VARIANT_LABEL[v],
                        color=style[v]["color"], marker=style[v]["marker"], zorder=3)
    ax_conf.set_xscale("log")
    ax_conf.set_yscale("log")
    ax_conf.set_xlabel("max-min loss contrast (log)")
    ax_conf.set_ylabel("roughness (log)")
    ax_conf.set_title(f"why roughness alone is not evidence: Spearman = {rho:+.2f} "
                      f"(p={rho_p:.2f}) over {2 * len(eps)} grids\nthe contrast-free "
                      "reading is basin area / plateau height")
    ax_conf.grid(alpha=0.25, which="both")
    ax_conf.legend(fontsize=8)

    ax_ratio.bar(x, ratio, color=[style[b]["color"] if d > 1 else style[a]["color"]
                                  for d in ratio])
    ax_ratio.axhline(1.0, color="k", lw=0.9)
    ax_ratio.set_yscale("log")
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([str(e) for e in eps])
    ax_ratio.set_xlabel("val episode")
    ax_ratio.set_ylabel("min_loss straighten / p-reg (log)")
    ax_ratio.set_title("per-episode min-loss ratio: is the aggregate one episode?\n"
                       "red = p-reg deeper, blue = straighten deeper")
    ax_ratio.grid(alpha=0.25, axis="y")

    width = 0.38
    for k, v in enumerate((a, b)):
        vals = [100 * p["basin_area_frac"][v] for p in paired]
        ax_basin.bar(x + (k - 0.5) * width, vals, width, label=VARIANT_LABEL[v],
                     color=style[v]["color"])
    ax_basin.set_xticks(x)
    ax_basin.set_xticklabels([str(e) for e in eps])
    ax_basin.set_xlabel("val episode")
    ax_basin.set_ylabel(f"% of box within {BASIN_FACTOR:g}x min")
    ax_basin.set_title("basin area: a 100% bar can be a flat, bad grid\n"
                       "(always read it with panel 1)")
    ax_basin.grid(alpha=0.25, axis="y")
    ax_basin.legend(fontsize=8)

    for k, v in enumerate((a, b)):
        vals = [p["median_over_min"][v] for p in paired]
        ax_plateau.bar(x + (k - 0.5) * width, vals, width, label=VARIANT_LABEL[v],
                       color=style[v]["color"])
    ax_plateau.axhline(1.0, color="k", lw=0.9)
    ax_plateau.set_yscale("log")
    ax_plateau.set_xticks(x)
    ax_plateau.set_xticklabels([str(e) for e in eps])
    ax_plateau.set_xlabel("val episode")
    ax_plateau.set_ylabel("median loss / min loss (log)")
    ax_plateau.set_title("plateau height: 1.0x = the whole box is the optimum\n"
                         "(a flat grid, no gradient signal)")
    ax_plateau.grid(alpha=0.25, axis="y")
    ax_plateau.legend(fontsize=8)

    for k, v in enumerate((a, b)):
        vals = [p["border_share"][v] for p in paired]
        ax_border.bar(x + (k - 0.5) * width, vals, width, label=VARIANT_LABEL[v],
                      color=style[v]["color"])
    ax_border.axhline(0.0, color="k", lw=0.9)
    ax_border.set_xticks(x)
    ax_border.set_xticklabels([str(e) for e in eps])
    ax_border.set_xlabel("val episode")
    ax_border.set_ylabel("1 - interior energy / total energy")
    ax_border.set_title("border share: is roughness the surface or the padding?")
    ax_border.grid(alpha=0.25, axis="y")
    ax_border.legend(fontsize=8)

    fig.suptitle(f"{env}: paired structural summary over K={len(eps)} held-out "
                 f"episodes (all panels are the same cached grids as "
                 f"loss_landscape_comparison{paths['suffix']}.csv)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(paths["figs"],
                        f"diagnostics_{env}_summary{paths['suffix']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"      summary figure -> {_display(path)}")
    return path


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Paired loss-landscape smoothness comparison: straighten vs "
                    "straighten + p-reg (the paper's Fig. 4, made quantitative)")
    ap.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS),
                    choices=sorted(MODEL_DIRS.keys()),
                    help="env(s) to compare (default: pusht; "
                         "e.g. --envs pusht point_maze to add umaze)")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing preset: --episodes 1 --grid 5 --opt-steps 10 "
                         "(an explicit --episodes/--grid/--opt-steps wins)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="held-out val episodes per env (default 6)")
    ap.add_argument("--grid", type=int, default=None,
                    help="landscape grid size per axis (default 13, the paper's "
                         "Fig. 4 sizing; --grid 9 is 2.1x cheaper)")
    ap.add_argument("--opt-steps", type=int, default=None,
                    help="GD steps per landscape grid point (default 80)")
    ap.add_argument("--lr", type=float, default=0.1,
                    help="Adam lr for the action optimization (default 0.1)")
    ap.add_argument("--action-range", type=float, default=2.0,
                    help="grid half-range in normalized action units (default 2.0)")
    ap.add_argument("--goal-H", type=int, default=25,
                    help="horizon in raw frames, must be a multiple of frameskip "
                         "(default 25, matching the existing Fig. 4 panel)")
    ap.add_argument("--n-viz", type=int, default=2,
                    help="illustrative episodes to plot per env (default 2)")
    ap.add_argument("--no-viz", action="store_true", help="skip the figures")
    ap.add_argument("--baseline", action="store_true",
                    help="also sweep the context variant(s) (CONTEXT_VARIANTS: the "
                         "no-straightening 'baseline' checkpoint) on the same "
                         "episodes, writing per_episode_context.csv and, with "
                         "--fig-style paper, the reference panel; context rows never "
                         "enter the paired table, its win counts or its figures")
    ap.add_argument("--baseline-ckpt", default=None, metavar="DIR",
                    help="override the baseline checkpoint dir (default "
                         "MODEL_DIRS[env]['baseline']), so the reference panel can "
                         "be pointed at any run without editing this script "
                         "(needs --baseline)")
    ap.add_argument("--fig-style", choices=("plain", "paper"), default="plain",
                    help="figure rendering: 'plain' = the paired heatmaps as before; "
                         "'paper' = additionally write fig4_<env>_ep###.png, a "
                         "paper-Fig.-4-style row (bicubic display of the swept grid, "
                         "iso-loss contours, argmin marker, one shared 99th-pct "
                         "colourbar, baseline panel first when --baseline is set)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="also write the diagnostic figure set (per-episode "
                         "shared/autoscaled/(L-L_min) heatmaps, profiles through "
                         "each argmin + basin-area curve, and one paired summary "
                         "figure per env)")
    ap.add_argument("--viz-only", action="store_true",
                    help="never compute: rebuild metrics/CSVs/figures from the "
                         "cached grids only (implies --reuse-grids and "
                         "--diagnostics; add --no-viz to refresh CSVs only)")
    ap.add_argument("--reuse-grids", action="store_true",
                    help="reuse cached .npz grids whose recorded settings match "
                         "(wall times already recorded in the CSV are kept)")
    ap.add_argument("--heartbeat", type=float, default=60.0, metavar="S",
                    help="seconds between 'still running' progress lines while a "
                         "grid is being optimized (default 60; 0 disables)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"),
                    help="output root (default <repo>/analysis_outputs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def resolve_run_cfg(args):
    """One config, shared by every loss_landscape call: explicit flag > preset."""
    preset = SMOKE_PRESET if args.smoke else SPEC_PRESET
    run_cfg = {}
    for key, flag_value in (("episodes", args.episodes), ("grid", args.grid),
                            ("opt_steps", args.opt_steps)):
        run_cfg[key] = preset[key] if flag_value is None else flag_value
    run_cfg.update(lr=args.lr, action_range=args.action_range, goal_H=args.goal_H)
    if run_cfg["episodes"] < 1:
        raise SystemExit("--episodes must be >= 1")
    if run_cfg["grid"] < 3:
        raise SystemExit("--grid must be >= 3 (a Laplacian needs neighbours)")
    if run_cfg["opt_steps"] < 1:
        raise SystemExit("--opt-steps must be >= 1")
    return run_cfg


def main():
    args = parse_args()
    if args.viz_only:
        # --viz-only is a cheap refresh pass: it requires the cache to exist and it
        # exists to produce the diagnostic figures, so both are implied rather than
        # asked for twice. --no-viz still wins (CSV-only refresh).
        if not args.reuse_grids:
            args.reuse_grids = True
            print("[note] --viz-only implies --reuse-grids")
        if not args.diagnostics:
            args.diagnostics = True
            print("[note] --viz-only implies --diagnostics (add --no-viz for CSVs only)")
    run_cfg = resolve_run_cfg(args)
    paths = output_paths(args)
    for directory in (paths["figs"], paths["grids"]):
        os.makedirs(directory, exist_ok=True)

    if not os.environ.get("DATASET_DIR"):
        local = os.path.join(REPO, "data", "datasets")
        if os.path.isdir(local):
            os.environ["DATASET_DIR"] = local
            print(f"[note] DATASET_DIR was unset; using the local copy at {local} "
                  "(setup.sh exports this for you)")
        else:
            raise SystemExit("DATASET_DIR is not set and there is no local dataset copy "
                             f"at {local}; source setup.sh first.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"[warn] --device {args.device} requested but CUDA is unavailable; using cpu")

    print(f"device: {device}")
    print(f"envs: {', '.join(args.envs)}"
          + ("" if list(args.envs) == list(DEFAULT_ENVS) else " (non-default selection)"))
    if args.smoke:
        print("SMOKE RUN -- plumbing validation only, not evidence "
              "(every output name carries a _smoke suffix)")
    print(f"run config (identical for every variant and episode): {run_cfg}")
    if args.viz_only:
        print("VIZ-ONLY -- no GD sweep will run: every metric/figure is rebuilt from "
              "the cached grids, and a missing cache raises instead of computing.")
    if args.baseline:
        print("BASELINE PANEL -- the context variant(s) are swept on the same episodes "
              "as the pair, but their rows go to per_episode_context.csv and they are "
              "never part of the paired table, its win counts or its headlines.")
    if args.baseline_ckpt and not args.baseline:
        print("[note] --baseline-ckpt has no effect without --baseline (nothing sweeps "
              "the context variant); add --baseline to use it")
    if run_cfg["grid"] < 5:
        print("[note] --grid < 5: roughness_interior is degenerate (fewer than 3x3 "
              "interior cells); the primary roughness is unaffected")

    summary_rows, all_rows, all_paired, figs, skipped = [], [], [], [], []
    all_context_rows = []
    for env in args.envs:
        print(f"\n=== {env} ===")
        try:
            for variant in VARIANT_ORDER:
                # fails fast, before any model is loaded, on placeholders/typos
                resolve_ckpt(env, variant)
            result = run_env(env, run_cfg, args, device, paths["grids"],
                             paths=paths, prior_rows=all_rows,
                             prior_context_rows=all_context_rows)
        except RuntimeError as e:
            print(f"  [skip] {e}")
            skipped.append(env)
            continue
        summary_rows += result["summary"]
        all_rows += result["rows"]
        all_context_rows += result["context_rows"]
        all_paired.append((env, result["paired"]))
        write_summary_csv(paths["summary_csv"], summary_rows)
        print_summary_table(result["summary"])
        print_headlines(result["summary"])
        print_paired_diagnostics(env, result["paired"])
        print_geometry_diagnostics(env, result, run_cfg)
        if not args.no_viz:
            figs += plot_env_examples(env, result, args, run_cfg, paths)
            if args.fig_style == "paper":
                # the paper-Fig.-4 rendering of the same cached grids; a view, so it
                # is written next to (never instead of) the plain paired heatmaps
                figs += plot_env_fig4(env, result, args, run_cfg, paths)
            if args.diagnostics:
                figs += plot_env_diagnostics(env, result, args, run_cfg, paths)
                figs += plot_env_profiles(env, result, args, run_cfg, paths)
                figs.append(plot_env_summary(env, result, run_cfg, paths))

    if not summary_rows:
        raise SystemExit("No env produced results (placeholders or missing checkpoints); "
                         "nothing to report.")

    write_summary_csv(paths["summary_csv"], summary_rows)
    write_per_episode_csv(paths["per_episode_csv"], all_rows)
    if all_context_rows:
        write_context_csv(paths["context_csv"], all_context_rows)
    print_summary_table(summary_rows)
    print_headlines(summary_rows)
    for env, paired in all_paired:
        print_paired_diagnostics(env, paired)
    if skipped:
        print(f"\n[skipped] {', '.join(skipped)} (variant pair not resolvable -- see the "
              "[skip] messages above; fill in MODEL_DIRS to include them)")
    print(f"\nAggregate table  -> {_display(paths['summary_csv'])}")
    print(f"Per-episode data -> {_display(paths['per_episode_csv'])}")
    if all_context_rows:
        print(f"Context data     -> {_display(paths['context_csv'])} "
              f"({len(all_context_rows)} rows; NOT part of the paired table)")
    print(f"Grid cache       -> {_display(paths['grids'])}/")
    for path in figs:
        print(f"Figure           -> {_display(path)}")


if __name__ == "__main__":
    main()


