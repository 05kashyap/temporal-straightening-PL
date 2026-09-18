#!/usr/bin/env python
"""analysis/landscape_metrics.py
===============================
The metric battery for swept action-space loss-landscape grids: one dict per
grid, computed from the grid alone (plus the axis it was sampled on).

Why a battery and not one number
--------------------------------
The original headline metric, `roughness = mean(laplacian(grid)^2) / var(grid)`,
is a *scale-free* smoothness statistic: it is invariant to a uniform rescaling of
the surface but NOT to how much of the grid's variance sits in one sharp feature,
and (with scipy's reflect padding) it is largely a property of the box boundary.
A nearly constant grid -- exactly the kind that gives an optimizer no gradient
signal -- scores as perfectly smooth. So the audit needs numbers that separate
the three things a landscape can be:

  * smooth but flat/uninformative  -> plateau share, contrast (median/min),
    gradient signal available to GD
  * smooth *and* informative       -> basin area vs depth, descent success
  * rough because it has one deep feature (which is not "bad" for GD)
                                   -> axis non-convexity, trap fraction

and, separately, numbers that say whether the swept box even CONTAINS the
optimum (the censoring family): `argmin_on_edge`, `wall_ratio`,
`censored_interior`. A censored grid can look arbitrarily smooth and its
"minimum" is then just the box wall, not an attainable optimum.

Continuity: `laplacian_roughness`, `count_local_minima`, `border_energy_share`,
`basin_area_frac` and `basin_curve` are imported verbatim from
analysis/loss_landscape_comparison.py, so a number computed here is the same
number the existing audited CSVs report (analysis/landscape_metric_audit.py
asserts that against loss_landscape/per_episode.csv).

Definitions of the new metrics (all computed on the raw min-attainable-loss grid,
no smoothing, no interpolation):
  center_over_min      G[box centre] / min(G), where G is the min-attainable-loss
                       grid. The planner initialises from norm_zero, whose first
                       two dims are ~(0.04, -0.03) -- NOT the origin -- so this is
                       the grid CELL that start falls in, read at the cell's own
                       action (0, 0): the surface exists only on the grid, one cell
                       being 2*ar/(g-1) wide. It is the TERMINAL loss (the best
                       value GD reached from that cell), not the loss before the
                       first update of that cell; a grid file stores the latter as
                       `step0`, so the two are distinguishable.
  center_is_argmin     1 if the planner's start IS the best cell in the box.
  dist_to_argmin_norm  Euclidean distance in normalized action units from the
                       box centre to the argmin; 0 = start is already best.
  margin_to_edge_norm  action_range - max(|argmin|), i.e. how much room the
                       optimum has left before the wall; 0 = on the wall.
  wall_ratio           min over border cells / grid minimum: ~1 means the box
                       wall is as good as the reported optimum, so the optimum is
                       probably OUTSIDE the swept box.
  interior_min_over_min  min over the interior / grid minimum, same reading but
                       with one cell of margin; > 1.05 flags a censored grid.
  grad_abs_median      median |dL/dx| over the grid (per normalized action unit).
  grad_rel             grad_abs_median * dx / median(L): the relative loss change
                       one grid cell away, i.e. how much signal one GD step of
                       that size has. Contrast-free.
  descent_success_frac fraction of cells whose steepest-descent walk (8-neighbour,
                       strict improvement, ties by fixed order) ends at the global
                       argmin. 1 = every start reaches the best cell; low values
                       mean real traps (which is what makes optimizing hard).
  descent_path_median  median number of steps those walks take.
  axis_nonconvex_frac  fraction of 1-D neighbour triples along either axis where
                       the middle cell is above the average of its two
                       neighbours (a bump/valley along a swept axis).
  basin_frac_<f>       P(L <= f * L_min) for f in BASIN_FACTORS (the contrast-free
                       basin profile).
  plateau_share_1.02   P(L <= 1.02 * L_min): how much of the box is at the bottom
                       (a very flat, uninformative surface has ~1.0).
  p90_over_min, iqr_ratio  p90/min and p75/p25: contrast/skew of the surface.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loss_landscape_comparison import (  # noqa: E402
    BASIN_FACTOR,
    basin_area_frac,
    basin_curve,
    border_energy_share,
    count_local_minima,
    grid_metrics,
    laplacian_roughness,
)

BASIN_FACTORS = (1.02, 1.05, 1.1, 1.25, 1.5, 2.0)
CENSORED_FACTOR = 1.05        # interior min this much above the global min = censored

__all__ = [
    "BASIN_FACTORS", "CENSORED_FACTOR", "basin_area_frac", "basin_curve",
    "border_energy_share", "count_local_minima", "grid_metrics",
    "laplacian_roughness", "metrics_for_grid", "center_geometry", "slope_stats",
    "descent_stats", "axis_nonconvex_frac", "censoring_stats", "basin_profile",
]



def _grid_and_axis(grid, axs=None):
    g = np.asarray(grid, dtype=np.float64)
    if axs is None:
        n = g.shape[0]
        axs = np.linspace(-2.0, 2.0, n)
    return g, np.asarray(axs, dtype=np.float64)


def _argmin(g):
    return tuple(int(i) for i in np.unravel_index(int(np.argmin(g)), g.shape))


def center_geometry(grid, axs=None):
    """Where the planner's own start (the box centre) sits on the surface."""
    g, axs = _grid_and_axis(grid, axs)
    ci = int(np.argmin(np.abs(axs)))
    cj = ci
    lo = float(np.min(g))
    ai, aj = _argmin(g)
    center = float(g[ci, cj])
    return {
        "center_value": center,
        "center_over_min": center / lo if lo > 0 else float("nan"),
        "center_is_argmin": int((ai, aj) == (ci, cj)),
        "center_rank_frac": float(np.mean(g < center)),   # 0 = start is the best
        "dist_to_argmin_norm": float(np.hypot(axs[ai] - axs[ci], axs[aj] - axs[cj])),
    }


def slope_stats(grid, axs=None):
    """How much gradient signal one grid cell of displacement carries."""
    g, axs = _grid_and_axis(grid, axs)
    dx = float(axs[1] - axs[0]) if len(axs) > 1 else 1.0
    gy, gx = np.gradient(g, dx, dx)
    mag = np.hypot(gx, gy)
    med_l = float(np.median(g))
    return {
        "grad_abs_median": float(np.median(mag)),
        "grad_abs_mean": float(np.mean(mag)),
        "grad_rel": float(np.median(mag) * dx / med_l) if med_l > 0 else float("nan"),
        "grad_rel_over_min": (float(np.median(mag) * dx / float(np.min(g)))
                              if np.min(g) > 0 else float("nan")),
    }


def descent_stats(grid):
    """Steepest-descent walk from every cell: does it reach the global argmin?

    The discrete version of what a first-order optimizer does: move to the best
    strictly-better 8-neighbour, stop when none is better. A cell whose walk stops
    anywhere but the global argmin is a genuine trap for that start, which is
    exactly the property `roughness` cannot see (a grid can be very rough and
    trap-free, or very smooth with plateaus that give no descent direction).
    """
    g = np.asarray(grid, dtype=np.float64)
    n = g.shape[0]
    tgt = _argmin(g)
    ok, lens = 0, []
    for i in range(n):
        for j in range(n):
            ci, cj, steps = i, j, 0
            while steps <= n * n:
                best, bv = None, g[ci, cj]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ii, jj = ci + di, cj + dj
                        if 0 <= ii < n and 0 <= jj < n and g[ii, jj] < bv:
                            bv, best = g[ii, jj], (ii, jj)
                if best is None:
                    break
                ci, cj = best
                steps += 1
            ok += int((ci, cj) == tgt)
            lens.append(steps)
    return {
        "descent_success_frac": ok / (n * n),
        "descent_path_median": float(np.median(lens)),
        "descent_path_p90": float(np.percentile(lens, 90)),
    }


def axis_nonconvex_frac(grid):
    """Fraction of 1-D neighbour triples that are non-convex along a swept axis."""
    g = np.asarray(grid, dtype=np.float64)
    bad = tot = 0
    for axis in (0, 1):
        glu = g if axis == 0 else g.T
        mid = glu[1:-1]
        bad += int(np.sum(mid > 0.5 * (glu[:-2] + glu[2:])))
        tot += int(mid.size)
    return float(bad / tot) if tot else float("nan")


def censoring_stats(grid, axs=None):
    """Is the reported optimum inside the swept box, or is it the box wall?

    The whole point of a landscape figure is that the optimizer could have reached
    the plot's minimum; if the minimum sits on the border and the border is much
    better than every interior cell, the true optimum is outside the box and the
    panel is a monotone ramp towards the wall, not a basin.
    """
    g, axs = _grid_and_axis(grid, axs)
    lo = float(np.min(g))
    border = np.zeros(g.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    ai, aj = _argmin(g)
    interior = ~border
    return {
        "argmin_is_edge": int(border[ai, aj]),
        "wall_ratio": (float(np.min(g[border])) / lo) if lo > 0 else float("nan"),
        "interior_min_over_min": (float(np.min(g[interior])) / lo
                                  if lo > 0 and interior.any() else float("nan")),
        "censored_interior": int(np.min(g[interior]) > CENSORED_FACTOR * lo
                                 if interior.any() else True),
        "margin_to_edge_norm": float(axs[-1] - max(abs(axs[ai]), abs(axs[aj]))),
        "argmin_ax": float(axs[ai]),
        "argmin_ay": float(axs[aj]),
    }


def basin_profile(grid):
    """Contrast-free basin profile: basin_curve at BASIN_FACTORS + plateau share."""
    g = np.asarray(grid, dtype=np.float64)
    curve = basin_curve(g, BASIN_FACTORS)
    out = {f"basin_frac_{f:g}": float(v) for f, v in zip(BASIN_FACTORS, curve)}
    out["plateau_share_1.02"] = out["basin_frac_1.02"]
    return out



def range_ratios(grid):
    """Scale-free contrast/skew of the surface (all relative to the minimum)."""
    g = np.asarray(grid, dtype=np.float64)
    lo = float(np.min(g))
    if lo <= 0:
        return {k: float("nan") for k in
                ("median_over_min", "p90_over_min", "iqr_ratio", "max_over_min")}
    p25, p75, p90 = (float(np.percentile(g, q)) for q in (25, 75, 90))
    return {
        "median_over_min": float(np.median(g)) / lo,
        "p90_over_min": p90 / lo,
        "iqr_ratio": (p75 / p25) if p25 > 0 else float("nan"),
        "max_over_min": float(np.max(g)) / lo,
    }


def metrics_for_grid(grid, axs=None):
    """Every metric of the battery for one grid: {name: float}.

    `grid` is the min-attainable-loss surface as saved by the sweep (row = first
    swept action coordinate, column = second), `axs` the swept values (default
    +-2 on 13 points, the legacy setting, so a grid loaded without its axis still
    gets the same centre/wall geometry as the old figures).
    """
    g, axs = _grid_and_axis(grid, axs)
    out = dict(grid_metrics(g))          # legacy: roughness, local_minima, ...
    out.update(center_geometry(g, axs))
    out.update(slope_stats(g, axs))
    out.update(descent_stats(g))
    out.update(censoring_stats(g, axs))
    out.update(basin_profile(g))
    out.update(range_ratios(g))
    out["axis_nonconvex_frac"] = axis_nonconvex_frac(g)
    out["grid"] = int(g.shape[0])
    out["action_range"] = float(axs[-1])
    out["min_loss"] = float(np.min(g))
    return out


# Metric groups, used by the audit's printed report and the figures so a table
# and a plot never disagree about what belongs to what.
METRIC_GROUPS = {
    "legacy_smoothness": ["roughness", "roughness_interior", "local_minima"],
    "censoring": ["argmin_is_edge", "wall_ratio", "interior_min_over_min",
                  "censored_interior", "margin_to_edge_norm"],
    "init_geometry": ["center_over_min", "center_is_argmin", "center_rank_frac",
                      "dist_to_argmin_norm"],
    "gradient_signal": ["grad_abs_median", "grad_rel", "grad_rel_over_min"],
    "descent": ["descent_success_frac", "descent_path_median"],
    "contrast": ["min_loss", "loss_range", "median_over_min", "p90_over_min",
                 "iqr_ratio", "max_over_min", "axis_nonconvex_frac"],
    "basin": ["basin_frac_1.02", "basin_frac_1.05", "basin_frac_1.1",
              "basin_frac_1.25", "basin_frac_1.5", "basin_frac_2"],
    "diagnostic": ["border_share", "basin_area_frac", "argmin_i", "argmin_j",
                   "argmin_ax", "argmin_ay", "grid", "action_range"],
}


def group_of(metric):
    for grp, names in METRIC_GROUPS.items():
        if metric in names:
            return grp
    return "other"


# ---------------------------------------------------------------------------
# Declared "better landscape" direction, so the ordering gate cannot be tuned
# after seeing the numbers: -1 = lower is a better landscape, +1 = higher is,
# 0 = no directional claim (diagnostic / absolute-level / ambiguous-by-design).
# A metric with direction 0 is reported but never counted as passing the gate.
# ---------------------------------------------------------------------------
DIRECTION = {
    "roughness": -1, "roughness_interior": -1, "local_minima": -1,
    "center_over_min": -1, "dist_to_argmin_norm": -1, "center_rank_frac": -1,
    "wall_ratio": -1, "interior_min_over_min": -1, "argmin_is_edge": -1,
    "median_over_min": -1, "p90_over_min": -1, "iqr_ratio": -1,
    "max_over_min": -1, "axis_nonconvex_frac": -1, "descent_path_median": -1,
    "descent_path_p90": -1,
    "grad_abs_median": +1, "grad_rel": +1, "grad_rel_over_min": +1,
    "descent_success_frac": +1,
    "basin_frac_1.02": +1, "basin_frac_1.05": +1, "basin_frac_1.1": +1,
    "basin_frac_1.25": +1, "basin_frac_1.5": +1, "basin_frac_2": +1,
    # no directional claim:
    "min_loss": 0, "loss_range": 0, "center_value": 0, "grad_abs_mean": 0,
    "plateau_share_1.02": 0, "basin_area_frac": 0, "border_share": 0,
    "center_is_argmin": 0, "argmin_i": 0, "argmin_j": 0, "argmin_ax": 0,
    "argmin_ay": 0, "margin_to_edge_norm": 0, "grid": 0, "action_range": 0,
    "argmin_on_edge": 0,
}

