#!/usr/bin/env python
"""analysis/landscape_paper_figure.py
====================================
Paper figures for the loss-landscape / planner-convergence story, built ONLY
from the CSVs and grid files the other scripts write (so a figure can never
disagree with a number in a table):

  planner-curves   Fig. 5: the GD planner's OWN convergence, all four arms
                   (helpers/extract_planner_curves.py -> planner_*.csv)
  landscape        Fig. 4 hero (one landscape per arm, one episode) + Fig. 6
                   diagnostics, from the sweep grids (analysis/landscape_sweep.py)
                   + analysis_outputs/paper/landscape_paper_metrics.csv, the
                   curated per-grid table whose numbers the paper text may quote
                   (--paper-style adds the paper's OWN Fig. 4 look: two bare
                   panels, no axes/ticks/colorbar, black = lower loss)

Design rules the figures follow (and that the CSVs support):
  * Four arms, one colour each, in the same order everywhere (baseline /
    straightening / p-reg / both), imported from the extractor.
  * No cross-arm comparison of absolute loss levels: different checkpoints define
    different latent spaces, so only a curve divided by its own step-1 value, a
    step count, and ratios to the SAME grid's own numbers (L_min/L_gt,
    start/min) are plotted.
  * Every figure states its own caveats (censoring, what the ordering gate found)
    instead of leaving a reader to infer a claim that the data does not support.

Usage:
  python analysis/landscape_paper_figure.py planner-curves --env pusht
  python analysis/landscape_paper_figure.py landscape --env pusht \
      --outcomes analysis_outputs/planner_loss_curves.csv
"""

import argparse
import csv
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "helpers"), os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from extract_planner_curves import (  # noqa: E402
    ARM_COLOR,
    ARM_LABEL,
    ARM_MARKER,
    ARM_ORDER,
    steps_to_gain,
)

DPI = 150

# ---------------------------------------------------------------------------
# The paper's OWN Fig. 4 (papers/ts.md L280-290), read off the figure assets in
# papers/2603.12231v3.pdf page 4 rather than guessed:
#   * exactly TWO panels, "(a) DINOv2" and "(b) Straightened", side by side;
#   * bare heatmaps -- no axes, ticks, labels, title, colorbar, legend or markers;
#   * the caption says "darker colors indicating lower loss", so the ramp is
#     black (low) -> red -> orange -> yellow -> white (high) = matplotlib "hot";
#   * 42-51 distinct RGB values per panel and no axis-aligned comb of colour
#     steps -> filled contours at ~42 levels (in panel (b) 41 of its 42 exact
#     colours each cover >=0.5% of the panel: flat bands, almost no blends), not
#     a smooth or a cell-block image;
#   * each panel reaches both ends of the ramp -> one colour range PER PANEL.
# The arm labels below are the honest translation of those panel titles onto the
# four checkpoints this repo trains ("both" adds the repo's two-thirds
# retargeting term to the cosine straightening the paper's Eq. 6 defines, so its
# panel says so instead of silently claiming to be the paper's checkpoint).
# ---------------------------------------------------------------------------
PAPER_PANEL_LETTERS = "abcdefgh"
PAPER_ARM_LABEL = {
    "baseline": "DINOv2",
    "straighten": "Straightened",
    "both": "Straightened (cos + retarget)",
    "p_reg": "p-reg (two-thirds)",
}
# The pair the paper shows, then the cosine-only supplement: the paper's own text
# names the straightening term (Eq. 6) as the method, so the supplement is the
# strictly literal (b) and is written alongside it rather than instead of it.
PAPER_PAIRS = (("baseline", "both"), ("baseline", "straighten"))


def arm_style(arm):
    """One place where an arm becomes a (colour, marker) pair."""
    return {"color": ARM_COLOR[arm], "marker": ARM_MARKER[arm]}


def load_curves(path):
    """{(arm, seed, episode): dict} from planner_loss_curves.csv.

    The CSV is the artifact of record: everything a figure shows is read back
    from it (never recomputed from the run dirs), so re-plotting cannot silently
    change a number a table already quotes.
    """
    series = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["arm"], int(row["seed"]), int(row["episode"]))
            s = series.setdefault(key, {"steps": [], "loss": [], "loss_rel": [],
                                        "success": None})
            s["steps"].append(int(row["step"]))
            s["loss"].append(float(row["loss"]))
            s["loss_rel"].append(float(row["loss_rel"]) if row["loss_rel"] else np.nan)
            s["success"] = (float(row["episode_success"])
                            if row["episode_success"] else None)
    for s in series.values():
        order = np.argsort(s["steps"])
        for k in ("steps", "loss", "loss_rel"):
            s[k] = np.asarray(s[k], dtype=np.float64)[order]
        s["steps_to_95pc"] = steps_to_gain(list(s["loss"]), 0.95)
    return series


def load_meta(path):
    """[row, ...] from planner_run_meta.csv (per run: success + settings)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _number(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else np.nan


def paired_arm_tests(series, metric="rel100"):
    """Wilcoxon signed-rank of every arm pair over the MATCHED eval episodes.

    Every open-loop run of a seed plans the same 50 episodes (plan_targets.pkl is
    byte-identical across the four checkpoints), so (seed, episode) is a genuine
    pairing key and a paired test is the right one: it removes the huge
    between-episode spread instead of averaging it into the noise.
    """
    def per_episode(arm):
        return {(s, e): v for (a, s, e), v in series.items() if a == arm}

    def value(v):
        if metric == "rel100":
            return v["loss_rel"][-1]
        if metric == "t95":
            return v["steps_to_95pc"]
        raise ValueError(metric)

    tabs = {a: per_episode(a) for a in ARM_ORDER if any(a == k[0] for k in series)}
    arms = list(tabs)
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            keys = sorted(set(tabs[a]) & set(tabs[b]))
            if len(keys) < 5:
                continue
            xa = np.array([value(tabs[a][k]) for k in keys])
            xb = np.array([value(tabs[b][k]) for k in keys])
            d = xa - xb
            if np.allclose(d, 0):
                out.append((a, b, len(keys), 0.0, float("nan")))
                continue
            try:
                p = float(stats.wilcoxon(xa, xb).pvalue)
            except Exception:                                # noqa: BLE001
                p = float("nan")
            out.append((a, b, len(keys), float(np.median(d)), p))
    return out


def curves_matrix(series, env, arm, key="loss_rel"):
    """(seeds, n_steps) matrix of that arm's per-seed mean curve.

    Each episode contributes its own normalised curve (`key`), so a seed row is
    the mean over that seed's 50 eval episodes -- the same 50 whose success rate
    the seed's run reported.
    """
    seeds = sorted({s for (a, s, _e) in series if a == arm})
    rows = []
    for seed in seeds:
        eps = [v[key] for (a, s, _e), v in series.items() if a == arm and s == seed]
        if not eps:
            continue
        n = min(len(e) for e in eps)
        rows.append(np.mean(np.stack([e[:n] for e in eps]), axis=0))
    n = min(len(r) for r in rows)
    return seeds, np.stack([r[:n] for r in rows])


def fig_planner_curves(series, meta, env, outdir):
    """Fig. 5: the planner's own convergence, four arms, four views.

    (a) mean normalised loss curve per arm (log y), one thin line per seed;
    (b) ECDF of each episode's step at which 95% of its own total drop is reached
        -- a step count is the only convergence number that needs no cross-arm
        scale assumption at all;
    (c) the per-episode success rate of those same runs (per-seed dots on bars),
        i.e. the outcome the convergence is supposed to explain;
    (d) within each arm, the final normalised loss of the episodes that succeeded
        vs the ones that failed -- whether "converged deeper" is what "succeeded"
        means, or whether the two are independent (which decides how much the
        landscape story may claim).
    """
    arms = [a for a in ARM_ORDER if any(a == k[0] for k in series)]
    if not arms:
        raise SystemExit(f"no rows for any of {ARM_ORDER} in the curves CSV")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_curve, ax_ecdf), (ax_succ, ax_split) = axes

    mean_curves = {}
    for arm in arms:
        seeds, mat = curves_matrix(series, env, arm, "loss_rel")
        steps = np.arange(1, mat.shape[1] + 1)
        for r in mat:
            ax_curve.plot(steps, r, color=ARM_COLOR[arm], lw=0.8, alpha=0.45)
        mean = mat.mean(axis=0)
        mean_curves[arm] = mean
        ax_curve.plot(steps, mean, color=ARM_COLOR[arm], lw=2.2,
                      marker=ARM_MARKER[arm], markevery=max(1, len(steps) // 10),
                      ms=5, label=f"{ARM_LABEL[arm]}  (step100={mean[-1]:.3f}, "
                                 f"{len(seeds)} seeds)")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("GD step of the planner's own optimization")
    ax_curve.set_ylabel("loss / its own step-1 value  (log)")
    ax_curve.set_title("(a) planner convergence, 4 checkpoints, 50 eval episodes\n"
                       "the same protocol that produced the success rates")
    ax_curve.grid(alpha=0.25, which="both")
    ax_curve.legend(fontsize=8, loc="upper right")

    per_arm_t95 = {}
    for arm in arms:
        vals = np.array([v["steps_to_95pc"] for (a, _s, _e), v in series.items()
                         if a == arm], dtype=np.float64)
        per_arm_t95[arm] = vals
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax_ecdf.step(xs, ys, where="post", color=ARM_COLOR[arm], lw=2.0,
                     label=f"{ARM_LABEL[arm]}  median {np.median(xs):.0f}, "
                           f"mean {xs.mean():.1f}")
    ax_ecdf.set_xlabel("step reaching 95% of the episode's own total drop")
    ax_ecdf.set_ylabel("fraction of episodes")
    ax_ecdf.set_title("(b) how fast each episode converges (all episodes per arm)\n"
                      "left = faster; the last step means the target was never "
                      "reached")
    ax_ecdf.grid(alpha=0.25)
    ax_ecdf.legend(fontsize=8, loc="lower right")

    x = np.arange(len(arms))
    succ = [np.array([_number(m, "episode_success_mean") for m in meta
                      if m["env"] == env and m["arm"] == arm]) for arm in arms]
    means = [float(np.nanmean(s)) for s in succ]
    errs = [float(np.nanstd(s)) for s in succ]
    ax_succ.bar(x, means, yerr=errs, capsize=4, color=[ARM_COLOR[a] for a in arms],
                alpha=0.85)
    for i, (s, m) in enumerate(zip(succ, means)):
        ax_succ.scatter(np.full(len(s), i), s, color="k", zorder=3, s=18)
        ax_succ.text(i, m + 0.03, f"{m:.3f}", ha="center", fontsize=9)
    ax_succ.set_xticks(x)
    ax_succ.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax_succ.set_ylim(0, 1.0)
    ax_succ.set_ylabel("open-loop success rate (50 episodes)")
    ax_succ.set_title("(c) outcome of those same runs\nbars = across-seed mean, "
                      "dots = one seed (n=3), error bar = population std")
    ax_succ.grid(alpha=0.25, axis="y")

    width = 0.36
    pooled = {"rel": [], "succ": []}
    for i, arm in enumerate(arms):
        for flag, off in ((1.0, -width / 2), (0.0, width / 2)):
            vals = [v["loss_rel"][-1] for (a, _s, _e), v in series.items()
                    if a == arm and v["success"] is not None and v["success"] == flag]
            if not vals:
                continue
            ax_split.bar(i + off, float(np.mean(vals)), width, color=ARM_COLOR[arm],
                         alpha=0.95 if flag == 1.0 else 0.45,
                         hatch="" if flag == 1.0 else "//",
                         label=("succeeded" if flag == 1.0 else "failed")
                         if i == 0 else None)
            ax_split.text(i + off, float(np.mean(vals)) * 1.05, f"n={len(vals)}",
                          ha="center", fontsize=7)
        pooled["rel"] += [v["loss_rel"][-1] for (a, _s, _e), v in series.items()
                          if a == arm and v["success"] is not None]
        pooled["succ"] += [v["success"] for (a, _s, _e), v in series.items()
                           if a == arm and v["success"] is not None]
    rho, pval = stats.spearmanr(pooled["succ"], pooled["rel"])
    ax_split.set_xticks(np.arange(len(arms)))
    ax_split.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax_split.set_ylabel("loss_rel at step 100 (lower = converged deeper)")
    ax_split.set_title("(d) convergence is not success: final loss of the episodes "
                       f"that succeeded vs failed\nSpearman(success, final loss) = "
                       f"{rho:+.2f} (p={pval:.1e}) over {len(pooled['succ'])} episodes")
    ax_split.grid(alpha=0.25, axis="y")
    ax_split.legend(fontsize=8)

    fig.suptitle(f"GD planner convergence on the {env} eval set: offline wandb "
                 "datastores of the runs behind results/AGG_RESULTS.MD,\nread back "
                 "from analysis_outputs/planner_*.csv (no new planning was run)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    path = os.path.join(outdir, f"fig5_planner_curves_{env}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, mean_curves, per_arm_t95


# ---------------------------------------------------------------------------
# Landscape figures (Fig. 4 hero + Fig. 6 diagnostics) from the sweep grids
# ---------------------------------------------------------------------------
def load_grid(path):
    """{meta..., grid_loss, axs, best_trace, step0} from one sweep grid .npz."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: (z[k].item() if getattr(z[k], "shape", ()) == () else np.asarray(z[k]))
             for k in z.files}
    d["grid_loss"] = np.asarray(d["grid_loss"], dtype=np.float64)
    if "axs" in d:
        d["axs"] = np.asarray(d["axs"], dtype=np.float64)
    d["file"] = os.path.basename(path)
    return d


def sweep_rows(grids_dir, env):
    """[grid dict, ...] for one env, from the sweep output dir."""
    if not os.path.isdir(grids_dir):
        raise SystemExit(f"no grid dir {grids_dir} -- run: "
                         f"python analysis/landscape_sweep.py sweep --env {env}")
    rows = [load_grid(os.path.join(grids_dir, name))
            for name in sorted(os.listdir(grids_dir))
            if name.startswith(f"{env}_") and name.endswith(".npz")]
    if not rows:
        raise SystemExit(f"no {env} grids in {grids_dir}")
    return rows


def _grid_by(rows, arm, episode, tag=""):
    for d in rows:
        if (d.get("variant") == arm and int(d.get("episode_idx", -1)) == int(episode)
                and str(d.get("tag", "")) == tag):
            return d
    return None


def _sel(d, tag="", action_range=None):
    """True if a grid row passes the --tag / --action-range filters.

    Needed because one grid dir can hold several boxes for the same arm (the
    pilot sweeps +-2 and +-3.5 side by side, and the saturation grids share the
    arm/episode), and the figures must draw exactly one grid per arm. `tag` is an
    exact match; `action_range` is a float compare against the grid's provenance.
    """
    if tag and str(d.get("tag", "")) != str(tag):
        return False
    if action_range is None:
        return True
    try:
        return abs(float(d.get("action_range", float("nan")))
                   - float(action_range)) < 1e-6
    except (TypeError, ValueError):
        return False


def fig_landscape_hero(rows, env, episode, outdir, arms=None, tag="", suffix=None,
                       shared_scale=False):
    """Fig. 4: one panel per arm, one episode, one colour scale per panel by default.

    Every panel carries the numbers that make the picture readable instead of
    decorative: the attained minimum relative to the episode's own ground-truth
    action loss (L_min/L_gt), how much better the box's optimum is than the
    planner's actual start cell (start/min), where the argmin is, whether it sits
    on the wall (censoring), and the panel's own colour range. The legacy +-2 box
    is drawn as a dashed square, so the censoring the old figure suffered is
    visible rather than asserted.

    The default is one colour range PER PANEL, because the four checkpoints'
    absolute losses differ by ~10x (p-reg's grid spans 0.17-1.7, straightening's
    0.27-0.29): a single shared scale renders three of the four panels as one flat
    colour. The per-panel range is printed in each title and the suptitle states
    that levels are not comparable across arms; --shared-scale draws the
    conservative one-scale version for an appendix.
    """
    from landscape_metrics import metrics_for_grid

    arms = [a for a in (arms or ARM_ORDER) if _grid_by(rows, a, episode, tag)]
    if not arms:
        raise SystemExit(f"no grids for episode {episode!r} (tag {tag!r}) in this dir")
    ncol = min(2, len(arms))
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 6.6 * nrow),
                             squeeze=False)
    lo = min(float(np.nanmin(_grid_by(rows, a, episode, tag)["grid_loss"]))
             for a in arms)
    hi = max(float(np.nanmax(_grid_by(rows, a, episode, tag)["grid_loss"]))
             for a in arms)
    for k, arm in enumerate(arms):
        ax = axes[k // ncol][k % ncol]
        d = _grid_by(rows, arm, episode, tag)
        g, axs = d["grid_loss"], d["axs"]
        extent = [axs[0], axs[-1], axs[0], axs[-1]]
        vmin, vmax = ((lo, hi) if shared_scale
                      else (float(np.nanmin(g)), float(np.nanmax(g))))
        im = ax.imshow(g, origin="lower", extent=extent, cmap="viridis",
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                           label="terminal objective" if k == 0 else None)
        i, j = np.unravel_index(int(np.nanargmin(g)), g.shape)
        ax.plot(axs[i], axs[j], "*", ms=16, color="red", mec="black", mew=0.6)
        # The planner's start is the normalized real zero action, which is NEAR but
        # not AT the origin on pusht (~0.04, -0.03), so plot where it actually is
        # whenever the grid recorded it (grids written before that was stored fall
        # back to the origin and say so).
        sa = d.get("start_action")
        have_sa = sa is not None and np.all(np.isfinite(np.asarray(sa, dtype=float)))
        sx, sy = (float(np.asarray(sa, dtype=float).reshape(-1)[0]),
                  float(np.asarray(sa, dtype=float).reshape(-1)[1])) if have_sa \
            else (0.0, 0.0)
        ax.plot(sx, sy, "o", ms=8, color="white", mec="black", mew=1.0)
        leg = [plt.Line2D([], [], ls="none", marker="*", ms=13, color="red",
                          mec="black", label="argmin (best fixed action)"),
               plt.Line2D([], [], ls="none", marker="o", ms=7, color="white",
                          mec="black",
                          label="planner's start" if have_sa else
                          "grid origin (this grid stores no start action)")]
        gtx, gty = ((float(d["gt_first_action"][0]), float(d["gt_first_action"][1]))
                    if "gt_first_action" in d else (np.nan, np.nan))
        if np.isfinite(gtx):
            ax.plot(gtx, gty, "x", ms=11, color="magenta", mew=2.2)
            leg.append(plt.Line2D([], [], ls="none", marker="x", ms=10,
                                  color="magenta", mew=2,
                                  label="episode's true first action"))
        if abs(axs[-1] - 2.0) > 1e-9:
            ax.add_patch(plt.Rectangle((-2, -2), 4, 4, fill=False, ec="w",
                                       ls="--", lw=1.0, alpha=0.85))
            leg.append(plt.Line2D([], [], ls="--", color="w",
                                  label="the legacy +-2 box"))
        m = metrics_for_grid(g, axs)
        gtl = float(d.get("gt_loss", np.nan))
        ratio = m["min_loss"] / gtl if gtl else float("nan")
        ax.set_title(f"{ARM_LABEL.get(arm, arm)}\n"
                     f"$L_{{min}}/L_{{gt}}$ = {ratio:.3f}    start/min = "
                     f"{m['center_over_min']:.2f}    argmin = ({axs[i]:.1f}, "
                     f"{axs[j]:.1f})    "
                     f"{'ON WALL' if m['argmin_is_edge'] else 'inside'}\n"
                     f"colour range {vmin:.3g} - {vmax:.3g}", fontsize=9)
        ax.set_xlabel("first action dim 1 (normalized)")
        ax.set_ylabel("first action dim 2 (normalized)")
        ax.legend(handles=leg, fontsize=7, loc="upper right", framealpha=0.85)
    for k in range(len(arms), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    d0 = _grid_by(rows, arms[0], episode, tag)
    scale_txt = ("all panels on ONE colour scale" if shared_scale else
                 "one colour range per panel -- the checkpoints' absolute loss "
                 "levels are NOT comparable")
    fig.suptitle(f"Action-space terminal-loss landscape -- {env} eval episode "
                 f"{int(episode)} (travel p{d0.get('travel_pct', float('nan')):.0f} of "
                 f"the eval set)\nmin attainable loss after {d0.get('opt_steps')} GD "
                 f"steps per cell, grid {d0.get('grid')}, box "
                 f"+-{float(d0.get('action_range', 0)):g}, {scale_txt} "
                 f"(L_min/L_gt < 1 = the box beat the episode's own action; "
                 f"start/min = the terminal loss at the grid cell nearest the "
                 f"planner's start, over this grid's own minimum)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.5, w_pad=1.5)
    os.makedirs(outdir, exist_ok=True)
    sfx = suffix if suffix is not None else (f"_{tag}" if tag else "")
    path = os.path.join(outdir,
                        f"fig4_landscape_{env}_ep{int(episode):03d}{sfx}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fig_landscape_paper_style(rows, env, episode, outdir, arms=("baseline", "both"),
                              tag="", suffix=None, shared_scale=False, levels=42,
                              render="contourf", cmap="hot"):
    """The paper's OWN Fig. 4 rendering: two bare panels and nothing else.

    Built from the same grids and the same numbers as fig_landscape_hero -- this
    is a *view* of them, and it is deliberately stripped down to what the paper
    prints: no axes, ticks, labels, title, colorbar, legend or markers, a
    black-is-low ramp (the caption's "darker colors indicating lower loss"), and
    one colour range per panel (both of the paper's panels reach both ends of
    their ramp). `render="contourf"` with ~42 levels is the default because the
    paper's panels are 41-42 flat, area-dominant fill colours with almost no
    blends -- a filled-contour count, not the 50 dominant plus ~500 total colours
    a bicubic image of the same grid produces.

    Everything a reader of a bare heatmap cannot see -- which checkpoint each
    panel is, the box, the grid, the steps, each panel's colour range and argmin,
    whether that argmin is a wall/corner cell -- goes to a `.txt` sidecar next to
    the PNG and to the log, because a panel with no axes cannot state its own
    caveats.
    """
    from landscape_metrics import metrics_for_grid

    want = [str(a) for a in arms]
    have = [a for a in want if _grid_by(rows, a, episode, tag)]
    missing = [a for a in want if a not in have]
    if missing:
        print(f"  [note] paper-style Fig. 4: no grid for episode {episode!r} "
              f"(tag {tag!r}) for arm(s) {missing} -- this pair cannot be drawn "
              f"from this dir")
    if not have:
        return None
    if len(have) < len(want):
        print(f"  [note] paper-style Fig. 4: drawing only {have} "
              f"(the paper's own figure has two panels)")

    grids = {a: _grid_by(rows, a, episode, tag) for a in have}
    axs = np.asarray(grids[have[0]]["axs"], dtype=np.float64)
    for a, d in grids.items():
        if not np.array_equal(np.asarray(d["axs"], dtype=np.float64), axs):
            print(f"  [note] paper-style Fig. 4: {a}'s grid axis differs from "
                  f"{have[0]}'s -- the panels are then not aligned")
    extent = [axs[0], axs[-1], axs[0], axs[-1]]

    def _range(g):
        vmin, vmax = float(np.nanmin(g)), float(np.nanmax(g))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return vmin, vmin + 1e-12       # a flat grid still has to render
        return vmin, vmax

    rng = {a: _range(np.asarray(d["grid_loss"], dtype=np.float64))
           for a, d in grids.items()}
    lo = min(v[0] for v in rng.values())
    hi = max(v[1] for v in rng.values())

    fig, axes = plt.subplots(1, len(have), figsize=(3.55 * len(have), 3.8),
                             squeeze=False)
    scale_txt = ("ONE colour range for all panels" if shared_scale
                 else "one colour range per panel")
    prov = ["# paper-style Fig. 4 (papers/ts.md L280-290) -- bare panels on purpose",
            "# caption: \"The heatmap represents the minimum attainable loss for "
            "each initial action choice, with darker colors indicating lower loss\"",
            f"# cmap={cmap} (black = lowest loss); render={render}"
            + (f"; levels={int(levels)}" if render == "contourf" else
               "; bicubic display of the raw grid")
            + f"; {scale_txt}; no axes/ticks/labels/colorbar/legend/markers",
            f"env={env} episode={int(episode)} tag={tag!r} "
            f"grid={grids[have[0]].get('grid')} "
            f"opt_steps={grids[have[0]].get('opt_steps')} "
            f"box=+-{float(grids[have[0]].get('action_range', float('nan'))):g} "
            f"travel_pct={grids[have[0]].get('travel_pct')}"]
    for k, arm in enumerate(have):
        ax = axes[0][k]
        d = grids[arm]
        g = np.asarray(d["grid_loss"], dtype=np.float64)
        vmin, vmax = (lo, hi) if shared_scale else rng[arm]
        if render == "contourf":
            ax.contourf(axs, axs, np.ma.masked_invalid(g),
                        levels=np.linspace(vmin, vmax, max(2, int(levels)) + 1),
                        cmap=cmap)
        else:
            ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=extent,
                      cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                      interpolation="bicubic")
        ax.set_xlim(axs[0], axs[-1])
        ax.set_ylim(axs[0], axs[-1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, -0.015,
                f"({PAPER_PANEL_LETTERS[k]}) {PAPER_ARM_LABEL.get(arm, arm)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=11)
        m = metrics_for_grid(g, axs)
        gtl = float(d.get("gt_loss", np.nan))
        prov.append(
            f"panel ({PAPER_PANEL_LETTERS[k]}) arm={arm} "
            f"label={PAPER_ARM_LABEL.get(arm, arm)!r} "
            f"vmin={vmin:.6g} vmax={vmax:.6g} "
            f"L_min={m['min_loss']:.6g} L_gt={gtl:.6g} "
            f"L_min/L_gt={(m['min_loss'] / gtl) if gtl else float('nan'):.3f} "
            f"argmin=({m['argmin_ax']:.2f}, {m['argmin_ay']:.2f}) "
            f"argmin_on_wall={int(m['argmin_is_edge'])} "
            f"basin_frac_1.1={m['basin_frac_1.1']:.3f} "
            f"start_over_min={m['center_over_min']:.3f}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08,
                        wspace=0.03)
    os.makedirs(outdir, exist_ok=True)
    sfx = suffix if suffix is not None else (f"_{tag}" if tag else "")
    stem = (f"fig4_paperstyle_{env}_ep{int(episode):03d}_"
            f"{'_vs_'.join(have)}{sfx}")
    path = os.path.join(outdir, stem + ".png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    with open(os.path.join(outdir, stem + ".txt"), "w") as f:
        f.write("\n".join(prov) + "\n")
    print("  paper-style panels: "
          + " , ".join(f"({PAPER_PANEL_LETTERS[k]}) {a}"
                       for k, a in enumerate(have))
          + "  [how to read: black = lowest terminal goal cost]")
    return path


def fig_landscape_diagnostics(rows, metrics_rows, env, outdir, outcomes_path=None,
                              tag="", suffix=None):
    """Fig. 6: the things a landscape figure has to show to be trustworthy.

    (a) what GD actually did on the grid (per-step traces, mean over cells), not
        just the best value it ever saw;
    (b) censoring: how many grids have their argmin on the box wall (the legacy
        +-2 box did that in 14/14 grids, so this panel checks the new box per arm);
    (c) L_min/L_gt per grid: whether the swept box's best beats the episode's own
        ground-truth action loss at all;
    (d) start/min: the terminal loss at the grid cell nearest the planner's own
        start, over the grid's minimum -- i.e. how much better the best cell is than
        where the planner begins (read at that cell, not at the exact start action,
        and after GD from it, not before the first update: `step0` is the latter);
    (e/f) the ordering gate (metrics x outcome-supported arm pairs) when
        --outcomes is given, plus a text panel with the audit's verdicts, so the
        figure states its own caveats.
    """
    from landscape_metrics import metrics_for_grid

    arms = [a for a in ARM_ORDER if any(d.get("variant") == a for d in rows)]
    episodes = sorted({int(d["episode_idx"]) for d in rows})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (ax_trace, ax_cens, ax_ratio), (ax_center, ax_gate, ax_text) = axes

    for arm in arms:
        curves = []
        for d in rows:
            if d.get("variant") != arm or d.get("best_trace") is None:
                continue
            tr = np.asarray(d["best_trace"], dtype=np.float64)
            s0 = np.asarray(d.get("step0", np.nan), dtype=np.float64)
            if tr.ndim != 3 or tr.shape[2] < 2 or not np.all(np.isfinite(s0)):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                curves.append(np.nanmean(tr / s0[:, :, None], axis=(0, 1)))
        if not curves:
            continue
        mat = np.stack(curves)
        mean = np.nanmean(mat, axis=0)
        steps = np.arange(1, len(mean) + 1)
        for r in mat:
            ax_trace.plot(steps, r, color=ARM_COLOR[arm], lw=0.7, alpha=0.35)
        ax_trace.plot(steps, mean, color=ARM_COLOR[arm], lw=2.2,
                      label=f"{ARM_LABEL[arm]} ({mat.shape[0]} grids)")
    ax_trace.set_yscale("log")
    ax_trace.set_xlabel("GD step inside each grid cell")
    ax_trace.set_ylabel("best loss so far / loss at step 1  (log)")
    ax_trace.set_title("(a) what GD reaches inside one grid cell\n"
                       "(mean over cells, one line per grid, same budget and\n"
                       "objective for every arm, no per-arm rescaling)",
                       fontsize=10)
    ax_trace.grid(alpha=0.25, which="both")
    if ax_trace.get_legend_handles_labels()[0]:
        ax_trace.legend(fontsize=8)
    else:
        ax_trace.text(0.5, 0.5, "no per-step traces in these grids\n"
                                "(legacy grids only store the final surface)",
                     transform=ax_trace.transAxes, ha="center", fontsize=9)

    x = np.arange(len(arms))
    edge = [sum(int(metrics_for_grid(d["grid_loss"], d["axs"])["argmin_is_edge"])
                for d in rows if d.get("variant") == arm) for arm in arms]
    n_arm = [sum(d.get("variant") == arm for d in rows) for arm in arms]
    inside = [n - e for n, e in zip(n_arm, edge)]
    ax_cens.bar(x, inside, color=[ARM_COLOR[a] for a in arms], alpha=0.85,
                label="argmin inside the box")
    ax_cens.bar(x, edge, bottom=inside, color="crimson", alpha=0.85,
                label="argmin ON the wall (censored)")
    for i, (n, e) in enumerate(zip(n_arm, edge)):
        ax_cens.text(i, n + 0.1, f"{e}/{n} on wall", ha="center", fontsize=9)
    ax_cens.set_xticks(x)
    ax_cens.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax_cens.set_ylabel("grids")
    ax_cens.set_title("(b) is the optimum inside the swept box?\n"
                      "(the legacy +-2 box censored 14/14 cached grids)",
                      fontsize=10)
    ax_cens.grid(alpha=0.25, axis="y")
    ax_cens.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                   ncol=2, frameon=False)

    rng = np.random.RandomState(0)
    for i, arm in enumerate(arms):
        vals = [float(np.nanmin(d["grid_loss"])) / float(d["gt_loss"])
                for d in rows if d.get("variant") == arm and d.get("gt_loss")]
        ax_ratio.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals,
                         color=ARM_COLOR[arm], s=40)
        if vals:
            ax_ratio.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color="k", lw=2)
            ax_ratio.text(i, np.median(vals), f" {np.median(vals):.2f}", fontsize=8)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=1)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax_ratio.set_ylabel("$L_{min}$ (grid) / $L_{gt}$ (episode's own action)")
    ax_ratio.set_title("(c) does the box beat the episode's true action?\n"
                       "<1 = yes (GD found something better); dots = one grid,\n"
                       "bar = median", fontsize=10)
    ax_ratio.grid(alpha=0.25, axis="y")

    rng = np.random.RandomState(1)
    for i, arm in enumerate(arms):
        vals = [metrics_for_grid(d["grid_loss"], d["axs"])["center_over_min"]
                for d in rows if d.get("variant") == arm]
        ax_center.scatter(i + rng.uniform(-0.1, 0.1, len(vals)), vals,
                          color=ARM_COLOR[arm], s=40)
        if vals:
            ax_center.plot([i - 0.22, i + 0.22], [np.median(vals)] * 2, color="k", lw=2)
    ax_center.axhline(1.0, color="k", ls="--", lw=1)
    ax_center.set_yscale("log")
    ax_center.set_xticks(x)
    ax_center.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax_center.set_ylabel("L(planner's start) / L(grid minimum)")
    ax_center.set_title("(d) is the planner's start already the box's best?\n"
                        "1.0 = nothing left to descend from the start cell",
                        fontsize=10)
    ax_center.grid(alpha=0.25, axis="y")

    _gate_panel(ax_gate, ax_text, metrics_rows, env, outcomes_path)
    fig.suptitle(f"Landscape diagnostics: {env}, {len(rows)} grids "
                 f"({len(arms)} arms x {len(episodes)} episodes); metrics from "
                 "analysis/landscape_metrics.py", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=1.5)
    os.makedirs(outdir, exist_ok=True)
    sfx = suffix if suffix is not None else (f"_{tag}" if tag else "")
    path = os.path.join(outdir, f"fig6_landscape_diagnostics_{env}{sfx}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _gate_panel(ax_gate, ax_text, metrics_rows, env, outcomes_path):
    """Panel (e): the ordering gate; panel (f): the verdict text.

    The gate compares a metric's declared direction against the arm pairs whose
    success difference is itself significant (exact McNemar on per-episode
    successes). Without --outcomes the panel says so instead of showing a
    pass/fail that has no reference behind it.
    """
    from landscape_metric_audit import load_outcomes, ordering_report
    from landscape_metrics import DIRECTION

    seed = 100
    outcomes = load_outcomes(outcomes_path, env, seed) if outcomes_path else {}
    ax_gate.axis("off")
    ax_text.axis("off")
    lines = []
    if not outcomes:
        ax_gate.set_title("(e) ordering gate SKIPPED")
        ax_text.text(0, 1, "No outcome reference was given.\n\n"
                     "Pass --outcomes analysis_outputs/planner_loss_curves.csv to\n"
                     "compare each metric's ordering against the per-episode\n"
                     "success of the same runs (exact McNemar per arm pair).\n\n"
                     "Without it, a 'the landscape got smoother' claim has no\n"
                     "outcome to be right or wrong about, which is exactly what\n"
                     "the legacy roughness number suffered from.",
                     va="top", fontsize=9, family="monospace")
        return
    pairs, res = ordering_report(metrics_rows, outcomes, env)
    if not res:
        n_ep = len({m.get("episode", m.get("episode_idx")) for m in metrics_rows})
        ax_gate.set_title(f"(e) ordering gate: no supported pair\n"
                          f"({n_ep} swept episode(s): the exact McNemar needs "
                          ">=10 matched)")
        ax_text.text(0, 1, "The success differences between the arms on these "
                     "episodes are\nnot significant (or there are fewer than the "
                     "10 matched episodes the exact\nMcNemar needs), so no metric "
                     "can be scored against them. The per-arm\nmedians are in "
                     "landscape_paper_metrics.csv.", va="top", fontsize=9,
                     family="monospace")
        return
    metrics = [r["metric"] for r in res]
    M = np.zeros((len(metrics), max(1, len(pairs))))
    for i, r in enumerate(res):
        for j, pr in enumerate(pairs):
            M[i, j] = np.nan if not pr["supported"] else (
                1.0 if f"{pr['arm_a']}>{pr['arm_b']}" in r["detail"] else 0.0)
    ax_gate.imshow(np.ma.masked_invalid(M), cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto")
    ax_gate.set_yticks(np.arange(len(metrics)))
    ax_gate.set_yticklabels([f"{m} ({DIRECTION.get(m, 0):+d})" for m in metrics],
                            fontsize=7)
    ax_gate.set_xticks(np.arange(len(pairs)))
    ax_gate.set_xticklabels([f"{pr['arm_a'][:5]} vs {pr['arm_b'][:5]}\np={pr['p']:.1g}"
                             for pr in pairs], fontsize=7)
    ax_gate.set_title("(e) ordering gate: green = the metric agrees with the\n"
                      "measured success ordering on a significant pair")
    lines.append(f"orderings from success (seed {seed}, McNemar p<0.05):")
    for pr in pairs:
        lines.append(f"  {pr['arm_a']:>10s} vs {pr['arm_b']:<10s} "
                     f"{pr['success_a']:.3f} vs {pr['success_b']:.3f} "
                     f"p={pr['p']:.2g} {'SUPPORTED' if pr['supported'] else ''}")
    n_pass = sum(r["passes"] for r in res)
    lines.append(f"\n{n_pass}/{len(res)} directional metrics agree with every "
                 "supported pair:")
    for r in res:
        lines.append(f"  {'PASS' if r['passes'] else 'fail'}  {r['metric']:<22s} "
                     f"{r['agree']}/{r['supported_pairs']}")
    lines.append("\nmetrics whose DIRECTION is 0 make no 'better landscape' claim")
    lines.append("and are never counted as passing.")
    ax_text.text(0, 1, "\n".join(lines), va="top", fontsize=7.5, family="monospace")


def cmd_planner_curves(args):
    """Fig. 5 from planner_loss_curves.csv / planner_run_meta.csv."""
    curves_csv = os.path.join(args.outdir, "planner_loss_curves.csv")
    meta_csv = os.path.join(args.outdir, "planner_run_meta.csv")
    for p in (curves_csv, meta_csv):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {os.path.relpath(p, REPO)} -- run first:\n"
                             "  python helpers/extract_planner_curves.py "
                             f"--envs {args.env}")
    figdir = args.figdir or os.path.join(args.outdir, "paper")
    os.makedirs(figdir, exist_ok=True)

    series = load_curves(curves_csv)
    meta = load_meta(meta_csv)
    envs = sorted({m["env"] for m in meta})
    if args.env not in envs:
        raise SystemExit(f"env {args.env!r} is not in the CSVs (found: {envs})")

    path, mean_curves, per_arm_t95 = fig_planner_curves(series, meta, args.env, figdir)
    print(f"Saved {os.path.relpath(path, REPO)}")
    for arm in ARM_ORDER:
        if arm not in mean_curves:
            continue
        t = per_arm_t95[arm]
        print(f"  {ARM_LABEL[arm]:13s} loss_rel@100={mean_curves[arm][-1]:.4f}  "
              f"t95 mean={t.mean():5.1f} median={np.median(t):5.1f}  "
              f"at-the-cap={int(np.sum(t >= t.max()))}/{len(t)}")

    # Paired tests over the MATCHED eval episodes (the honest form of "arm X
    # converged deeper than arm Y"): the four checkpoints plan the same 50
    # episodes per seed, so (seed, episode) is a real pairing key.
    keys = {"rel100": "loss_rel@100 (lower = converged deeper)",
            "t95": "step reaching 95% of own drop (lower = faster)"}
    for metric, desc in keys.items():
        print(f"\nPaired Wilcoxon over matched (seed, episode) pairs -- {desc}")
        for a, b, n, med, p in paired_arm_tests(series, metric):
            stars = "" if p != p else ("***" if p < 1e-3 else
                                       ("**" if p < 1e-2 else
                                        ("*" if p < 5e-2 else "  ")))
            print(f"  {ARM_LABEL[a]:13s} - {ARM_LABEL[b]:13s} n={n:3d} "
                  f"median diff={med:+.4f} p={p:.2e} {stars}")


def cmd_landscape(args):
    """Fig. 4 (hero) + Fig. 6 (diagnostics) + the curated per-grid metric CSV."""
    from landscape_metrics import metrics_for_grid

    grids_dir = args.grids_dir or os.path.join(args.outdir, "paper", "grids")
    figdir = args.figdir or os.path.join(args.outdir, "paper")
    bits = []
    if args.tag:
        bits.append(str(args.tag))
    if args.action_range is not None:
        bits.append(f"ar{args.action_range:g}")
    sfx = f"_{'_'.join(bits)}" if bits else ""
    rows = sweep_rows(grids_dir, args.env)
    keep = [d for d in rows if _sel(d, args.tag, args.action_range)]
    if len(keep) != len(rows):
        print(f"  filters --tag {args.tag!r} / --action-range "
              f"{args.action_range}: {len(keep)}/{len(rows)} grids kept")
    rows = keep
    if not rows:
        raise SystemExit("no grids survive --tag / --action-range")
    episodes = sorted({int(d["episode_idx"]) for d in rows})
    print(f"{args.env}: {len(rows)} grids, {len(episodes)} episodes {episodes} "
          f"from {os.path.relpath(grids_dir, REPO)}")

    if args.episode is None:
        travels = {int(d["episode_idx"]): float(d.get("travel_pct", np.nan))
                   for d in rows}
        ep = sorted(travels, key=lambda e: abs(travels[e] - 50.0))[0]
        print(f"  hero episode {ep} (travel p{travels[ep]:.0f} -- the median of the "
              "set, so the hero is neither the easiest nor the hardest)")
    else:
        ep = int(args.episode)
    paths = [fig_landscape_hero(rows, args.env, ep, figdir, tag=args.tag,
                                suffix=sfx,
                                shared_scale=bool(getattr(args, "shared_scale",
                                                          False)))]
    if getattr(args, "paper_style", False):
        # The paper's own two-panel look, from the same grids: the pair the paper
        # prints and (unless --paper-arms asked for something else) the
        # cosine-only supplement. Missing arms are reported, never invented.
        pairs = ([tuple(args.paper_arms)] if getattr(args, "paper_arms", None)
                 else list(PAPER_PAIRS))
        for pair in pairs:
            p = fig_landscape_paper_style(
                rows, args.env, ep, figdir, arms=pair, tag=args.tag, suffix=sfx,
                shared_scale=bool(getattr(args, "shared_scale", False)),
                levels=int(getattr(args, "paper_levels", 42)),
                render=str(getattr(args, "paper_render", "contourf")),
                cmap=str(getattr(args, "paper_cmap", "hot")))
            if p:
                paths.append(p)

    metrics_csv = args.metrics_csv or os.path.join(args.outdir, "paper",
                                                  "landscape_sweep_metrics.csv")
    metrics_rows = []
    if os.path.isfile(metrics_csv):
        from landscape_metric_audit import coerce_rows
        with open(metrics_csv, newline="") as f:
            raw = coerce_rows(list(csv.DictReader(f)))
        if raw and ("episode" in raw[0] or "episode_idx" in raw[0]):
            metrics_rows = [m for m in raw if _sel(m, args.tag, args.action_range)]
        else:
            print(f"  [note] {os.path.relpath(metrics_csv, REPO)} carries no "
                  "episode column -- the ordering gate needs the episode, so the "
                  "metric rows are recomputed from the grids")
    if not metrics_rows:
        metrics_rows = [{"env": args.env, "variant": d.get("variant"),
                         "episode": d.get("episode_idx"),
                         "tag": d.get("tag"),
                         "action_range": d.get("action_range"),
                         **metrics_for_grid(d["grid_loss"], d["axs"])} for d in rows]
        print(f"  [note] {len(metrics_rows)} metric rows recomputed from "
              f"{os.path.relpath(grids_dir, REPO)}")
    paths.append(fig_landscape_diagnostics(rows, metrics_rows, args.env, figdir,
                                           outcomes_path=args.outcomes,
                                           tag=args.tag, suffix=sfx))

    cols = [("env", lambda d, m: args.env),
            ("arm", lambda d, m: d.get("variant")),
            ("episode", lambda d, m: d.get("episode_idx")),
            ("travel_pct", lambda d, m: d.get("travel_pct")),
            ("grid", lambda d, m: d.get("grid")),
            ("opt_steps", lambda d, m: d.get("opt_steps")),
            ("action_range", lambda d, m: d.get("action_range")),
            ("L_min", lambda d, m: m["min_loss"]),
            ("L_gt", lambda d, m: d.get("gt_loss")),
            ("L_min_over_L_gt", lambda d, m: (m["min_loss"] / float(d["gt_loss"])
                                               if d.get("gt_loss") else np.nan)),
            ("start_over_min", lambda d, m: m["center_over_min"]),
            ("argmin_ax", lambda d, m: m["argmin_ax"]),
            ("argmin_ay", lambda d, m: m["argmin_ay"]),
            ("argmin_on_wall", lambda d, m: m["argmin_is_edge"]),
            ("wall_ratio", lambda d, m: m["wall_ratio"]),
            ("basin_frac_1.1", lambda d, m: m["basin_frac_1.1"]),
            ("descent_success_frac", lambda d, m: m["descent_success_frac"]),
            ("grad_rel", lambda d, m: m["grad_rel"]),
            ("median_over_min", lambda d, m: m["median_over_min"]),
            ("roughness_legacy", lambda d, m: m["roughness"])]
    out_csv = os.path.join(args.outdir, "paper",
                           f"landscape_paper_metrics{sfx}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in cols])
        for d in rows:
            m = metrics_for_grid(d["grid_loss"], d["axs"])
            cells = []
            for _name, fn in cols:
                v = fn(d, m)
                cells.append("" if v is None or v != v else
                             (f"{v:.6g}" if isinstance(v, (int, float)) else str(v)))
            w.writerow(cells)
    for p in paths:
        print(f"Saved {os.path.relpath(p, REPO)}")
    print(f"Saved {os.path.relpath(out_csv, REPO)} ({len(rows)} grids; the curated "
          "table the paper text may quote -- the legacy roughness is kept only "
          "for contrast)")
    print(f"\n  {'arm':13s} {'L_min/L_gt':>10s} {'start/min':>10s} {'on wall':>9s} "
          f"{'basin1.1':>9s} {'descent':>8s} {'grad_rel':>9s}")
    for arm in ARM_ORDER:
        sub = []
        for d in rows:
            if d.get("variant") != arm:
                continue
            m = metrics_for_grid(d["grid_loss"], d["axs"])
            m["ratio"] = (float(np.nanmin(d["grid_loss"])) / float(d["gt_loss"])
                          if d.get("gt_loss") else np.nan)
            sub.append(m)
        if not sub:
            continue
        med = lambda k: float(np.nanmedian([s[k] for s in sub]))        # noqa: E731
        print(f"  {ARM_LABEL[arm]:13s} {med('ratio'):10.3f} "
              f"{med('center_over_min'):10.3f} "
              f"{sum(int(s['argmin_is_edge']) for s in sub):4d}/{len(sub):<3d} "
              f"{med('basin_frac_1.1'):9.3f} {med('descent_success_frac'):8.3f} "
              f"{med('grad_rel'):9.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_common(ap):
    ap.add_argument("--env", default="pusht", help="env to draw (default pusht)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"),
                    help="where the CSVs live and the figures are written "
                         "(default <repo>/analysis_outputs; figures go to "
                         "<outdir>/paper/)")
    ap.add_argument("--figdir", default=None,
                    help="override the figure directory (default <outdir>/paper)")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Paper figures for the loss-landscape / planner-convergence "
                    "story, drawn from the analysis_outputs CSVs and grid files")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("planner-curves", help="Fig. 5: GD planner convergence, 4 arms")
    add_common(p)
    p.set_defaults(func=cmd_planner_curves)

    p = sub.add_parser("landscape", help="Fig. 4 hero + Fig. 6 diagnostics from the "
                                         "sweep grids")
    add_common(p)
    p.add_argument("--grids-dir", default=None,
                   help="sweep grid dir (default <outdir>/paper/grids)")
    p.add_argument("--metrics-csv", default=None,
                   help="landscape_sweep_metrics.csv (default <outdir>/paper/...)")
    p.add_argument("--episode", type=int, default=None,
                   help="hero episode (default: the median-travel episode present)")
    p.add_argument("--tag", default="",
                   help="only grids whose tag matches (default: untagged sweep grids)")
    p.add_argument("--action-range", type=float, default=None,
                   help="only grids swept over this box half-range (e.g. 3.5) -- "
                        "needed when one dir holds several boxes per arm")
    p.add_argument("--shared-scale", action="store_true",
                   help="Fig. 4: put all arms on ONE colour scale (conservative, "
                        "but the arms' absolute losses differ ~10x so three panels "
                        "go flat; the default is one range per panel, printed in "
                        "each title)")
    p.add_argument("--paper-style", action="store_true",
                   help="additionally write the paper's OWN Fig. 4 look: bare "
                        "two-panel figures (no axes/ticks/labels/colorbar/legend/"
                        "markers), hot colormap with black = lower loss, one range "
                        "per panel, filled contours. Writes "
                        "fig4_paperstyle_<env>_ep###_<a>_vs_<b>.png plus a .txt "
                        "sidecar with the numbers a bare panel cannot show")
    p.add_argument("--paper-arms", nargs=2, default=None, metavar=("(a)", "(b)"),
                   help="the two checkpoints --paper-style draws, in panel order "
                        "(default: both pairs, baseline-vs-both then the "
                        "cosine-only baseline-vs-straighten supplement)")
    p.add_argument("--paper-levels", type=int, default=42,
                   help="--paper-style filled-contour levels (default 42: the "
                        "paper's own panels are ~42 flat, area-dominant fill "
                        "colours with almost no blends)")
    p.add_argument("--paper-render", choices=("contourf", "imshow"),
                   default="contourf",
                   help="--paper-style rendering: filled contours (default, what "
                        "the paper's panels show) or a bicubic image of the raw "
                        "grid")
    p.add_argument("--paper-cmap", default="hot",
                   help="--paper-style colormap (default hot: black = low loss, "
                        "matching the caption's 'darker colors indicating lower "
                        "loss')")
    p.add_argument("--outcomes", default=None,
                   help="planner_loss_curves.csv for the ordering-gate panel")
    p.set_defaults(func=cmd_landscape)
    args = ap.parse_args()
    if os.environ.get("LANDSCAPE_VALIDATE_ARGS") == "1":
        # Parse and stop: `run_landscape_night.sh preflight` replays every stage's
        # real argv against this parser, so an undeclared flag fails in seconds.
        print(f"[validate-args] {args.cmd}: argv OK (no work performed)")
        raise SystemExit(0)
    return args


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
