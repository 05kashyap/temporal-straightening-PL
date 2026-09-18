#!/usr/bin/env python
"""analysis/landscape_metric_audit.py
====================================
Metric audit for swept action-space loss-landscape grids: recompute the whole
battery in analysis/landscape_metrics.py on cached grids, check it against the
numbers the existing audited CSVs already report, and say what the metrics can
and cannot support.

What this answers (the honest version of "which landscape metric is paper-worthy")
----------------------------------------------------------------------------------
1. Do the reused legacy metrics reproduce analysis_outputs/loss_landscape/
   per_episode*.csv exactly? (max abs diff printed; a mismatch is a bug, not a
   difference of opinion.)
2. Is the swept box censoring the optimum? Per grid: argmin-on-edge share, mean
   wall_ratio, censored-interior share. The 16 cached grids all used +-2, while
   the four pusht checkpoints' own eval first-actions reach +-3.2 in normalized
   units, so the box does not contain the action range the planner plans over.
3. Which metrics are the same metric? Spearman redundancy pairs over all grids
   (|rho| >= 0.9 reported): `roughness` vs `roughness_interior` is the pair the
   old headline rested on.
4. Does a metric separate the arms at all? Paired Wilcoxon over the episodes a
   metric has for both arms, per metric, with the sign of the difference.
5. Which metrics predict the OUTCOME? Needs grids on the eval episodes plus their
   per-episode success flags: pass --outcomes analysis_outputs/
   planner_loss_curves.csv. Without it the audit prints "skipped" instead of
   inventing a reference ordering.

Outputs (under --outdir)
  landscape_metric_audit.csv            one row per grid + the whole battery
  paper/figA_metric_audit_<env>.png     6-panel diagnostic of the above

Usage:
  python analysis/landscape_metric_audit.py                      # cached +-2 grids
  python analysis/landscape_metric_audit.py --all                # every cached grid
  python analysis/landscape_metric_audit.py --sweep-dir analysis_outputs/paper/grids \\
      --outcomes analysis_outputs/planner_loss_curves.csv
"""

import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "analysis"), os.path.join(REPO, "helpers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from extract_planner_curves import (  # noqa: E402
    ARM_COLOR,
    ARM_LABEL,
    ARM_ORDER,
    canonical_arm,
)
from landscape_metrics import METRIC_GROUPS, metrics_for_grid  # noqa: E402

DPI = 150
LEGACY_GRIDS = os.path.join(REPO, "analysis_outputs", "loss_landscape", "grids")
LEGACY_PER_EPISODE = os.path.join(REPO, "analysis_outputs", "loss_landscape",
                                  "per_episode.csv")
LEGACY_CONTEXT = os.path.join(REPO, "analysis_outputs", "loss_landscape",
                              "per_episode_context.csv")

# The metrics the old per-episode CSVs already carry, for the reproduction check.
REPRODUCE = ["roughness", "roughness_interior", "local_minima", "min_loss",
             "loss_range", "basin_area_frac", "border_share", "argmin_on_edge"]



def _scalar(x):
    return x.item() if isinstance(x, np.ndarray) and x.shape == () else x


def load_grid_file(path):
    """(meta, grid, axs) from a grid .npz (new sweep layout or legacy layout)."""
    with np.load(path, allow_pickle=False) as p:
        keys = set(p.files)
        grid = np.asarray(p["grid_loss"], dtype=np.float64)
        axs = np.asarray(p["axs"], dtype=np.float64) if "axs" in keys else None
        meta = {k: _scalar(p[k]) for k in keys if k not in ("grid_loss", "axs")}
    meta["file"] = os.path.relpath(path, REPO)
    return meta, grid, axs


def discover(grids_dir, want_grid=None, want_opt_steps=None, want_range=None):
    """Every usable grid under `grids_dir` matching the requested settings."""
    rows = []
    for path in sorted(glob.glob(os.path.join(grids_dir, "*.npz"))):
        try:
            meta, grid, axs = load_grid_file(path)
        except Exception as exc:                             # noqa: BLE001
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] < 3:
            continue
        if want_grid is not None and int(meta.get("grid", grid.shape[0])) != want_grid:
            continue
        if want_opt_steps is not None and meta.get("opt_steps") is not None \
                and int(meta["opt_steps"]) != want_opt_steps:
            continue
        if want_range is not None and meta.get("action_range") is not None \
                and abs(float(meta["action_range"]) - want_range) > 1e-9:
            continue
        rows.append((meta, grid, axs))
    return rows


def audit_grid(meta, grid, axs):
    """One output row: provenance + the whole metric battery."""
    row = {
        "env": meta.get("env", ""),
        "variant": canonical_arm(meta.get("variant", "")),
        "variant_raw": meta.get("variant", ""),
        "episode": meta.get("episode_idx", ""),
        "grid": meta.get("grid", grid.shape[0]),
        "opt_steps": meta.get("opt_steps", ""),
        "action_range": meta.get("action_range", ""),
        "goal_H": meta.get("goal_H", ""),
        "lr": meta.get("lr", ""),
        "file": meta.get("file", ""),
    }
    row.update(metrics_for_grid(grid, axs))
    return row


def reproduction_check(rows):
    """Max |new - old| per metric against the audited per-episode CSVs.

    The legacy CSVs are the reference: if this audit's `roughness` differs from
    the one in loss_landscape/per_episode.csv for the same (env, variant,
    episode), one of the two is wrong and nothing else in this file matters.
    """
    old = {}
    for path in (LEGACY_PER_EPISODE, LEGACY_CONTEXT):
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                old[(r["env"], r["variant"], str(int(float(r["episode"]))))] = r
    diffs = {}
    matched = 0
    for row in rows:
        if row["episode"] == "":
            continue
        ref = old.get((row["env"], row["variant"], str(int(row["episode"]))))
        if ref is None:
            continue
        matched += 1
        for m in REPRODUCE:
            if m not in ref or ref[m] in ("", None):
                continue
            diffs[m] = max(diffs.get(m, 0.0), abs(float(row[m]) - float(ref[m])))
    return matched, diffs


def censoring_report(rows):
    """Per (env, variant): how much of the box is censored by construction."""
    out = []
    for env, variant in sorted({(r["env"], r["variant"]) for r in rows}):
        sub = [r for r in rows if r["env"] == env and r["variant"] == variant]
        out.append({
            "env": env, "variant": variant, "n": len(sub),
            "argmin_is_edge_frac": float(np.mean([r["argmin_is_edge"] for r in sub])),
            "wall_ratio_mean": float(np.mean([r["wall_ratio"] for r in sub])),
            "censored_interior_frac": float(np.mean([r["censored_interior"] for r in sub])),
            "margin_to_edge_mean": float(np.mean([r["margin_to_edge_norm"] for r in sub])),
            "action_range": sub[0]["action_range"],
        })
    return out


def coerce_rows(rows):
    """Numeric-looking CSV strings -> floats, so the metric math sees numbers.

    Rows that come back from a CSV (e.g. landscape_sweep_metrics.csv) are all
    strings; the audit's paired tests and the ordering gate need floats, and
    silently treating every metric as non-numeric would make the gate skip itself
    without saying why.
    """
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if isinstance(v, str):
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            else:
                d[k] = v
        out.append(d)
    return out


def _numeric_metrics(rows):
    """Every battery metric that is numeric in these rows, in group order."""
    keys = []
    for grp in ("legacy_smoothness", "censoring", "init_geometry",
                "gradient_signal", "descent", "contrast", "basin", "diagnostic"):
        keys += list(METRIC_GROUPS[grp])
    seen, out = set(), []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        if any(isinstance(r.get(k), (int, float)) for r in rows):
            out.append(k)
    return out


CORRELATION_CANDIDATES = ("roughness", "roughness_interior", "local_minima",
                          "min_loss", "loss_range", "border_share", "basin_area_frac",
                          "center_over_min", "wall_ratio", "interior_min_over_min",
                          "grad_rel", "descent_success_frac", "axis_nonconvex_frac",
                          "median_over_min", "p90_over_min", "basin_frac_1.1",
                          "basin_frac_2", "plateau_share_1.02")


def redundancy_report(rows, threshold=0.9):
    """Spearman pairs with |rho| >= threshold: metrics that carry one message."""
    metrics = [m for m in CORRELATION_CANDIDATES if any(m in r for r in rows)]
    vals = {m: np.array([float(r.get(m, np.nan)) for r in rows]) for m in metrics}
    pairs = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            xa, xb = vals[a], vals[b]
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() < 4 or np.allclose(xa[ok], xa[ok][0]) \
                    or np.allclose(xb[ok], xb[ok][0]):
                continue
            rho = float(stats.spearmanr(xa[ok], xb[ok]).statistic)
            if abs(rho) >= threshold:
                pairs.append((a, b, rho, int(ok.sum())))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def paired_variant_tests(rows, variant_a, variant_b):
    """Paired Wilcoxon per metric over matched (env, episode) grids of two arms."""
    ta = {(r["env"], r["episode"]): r for r in rows if r["variant"] == variant_a}
    tb = {(r["env"], r["episode"]): r for r in rows if r["variant"] == variant_b}
    keys = sorted(set(ta) & set(tb))
    out = []
    if len(keys) < 5:
        return out
    for metric in _numeric_metrics(rows):
        xa = np.array([float(ta[k].get(metric, np.nan)) for k in keys])
        xb = np.array([float(tb[k].get(metric, np.nan)) for k in keys])
        ok = np.isfinite(xa) & np.isfinite(xb)
        if ok.sum() < 5 or np.allclose(xa[ok] - xb[ok], 0):
            continue
        try:
            p = float(stats.wilcoxon(xa[ok], xb[ok]).pvalue)
        except Exception:                                    # noqa: BLE001
            p = float("nan")
        out.append({"metric": metric, "n": int(ok.sum()),
                    "median_diff": float(np.median(xa[ok] - xb[ok])),
                    "frac_a_lower": float(np.mean(xa[ok] < xb[ok])), "p": p})
    return sorted(out, key=lambda d: d["p"])


def load_outcomes(path, env, seed):
    """{(arm, episode): success 0/1} for one env/seed from planner_loss_curves.csv."""
    out = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["env"] != env or int(r["seed"]) != seed or not r["episode_success"]:
                continue
            key = (r["arm"], int(r["episode"]))
            out.setdefault(key, float(r["episode_success"]))
    return out


def mcnemar(succ_a, succ_b):
    """(b-c)/n over discordant pairs + exact two-sided binomial p, plus (b, c)."""
    b = int(np.sum((succ_a > 0.5) & (succ_b <= 0.5)))
    c = int(np.sum((succ_a <= 0.5) & (succ_b > 0.5)))
    if b + c == 0:
        return 0.0, 1.0, b, c
    if hasattr(stats, "binomtest"):
        p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    else:                                                    # scipy < 1.7
        p = float(stats.binom_test(min(b, c), b + c, 0.5))
    return (b - c) / (b + c), p, b, c


def _ep_of(r):
    """Episode index of a metric row, whichever column its producer wrote.

    The audit's own CSVs carry `episode`; the sweep's landscape_sweep_metrics.csv
    carries `episode_idx`. The ordering gate is fed from whichever CSV exists (the
    figure reads the sweep's), so accept either instead of crashing on the other.
    """
    for key in ("episode", "episode_idx"):
        if r.get(key) not in (None, ""):
            return int(float(r[key]))
    raise KeyError("metric row has neither 'episode' nor 'episode_idx'")


def ordering_report(rows, outcomes, env):
    """The gate: does a metric's arm ordering agree with the measured outcomes?

    The reference ordering is the per-episode success on the SAME episodes (paired
    binary -> exact McNemar), not the aggregate table: only arm pairs whose
    success difference is itself significant (p < 0.05) define a direction a
    landscape metric can be asked to predict. A metric passes for such a pair when
    its DIRECTION-adjusted difference has the same sign as the success
    difference. Metrics with DIRECTION 0 are reported but never counted.
    """
    from landscape_metrics import DIRECTION

    arms = [a for a in ARM_ORDER if any(r["variant"] == a for r in rows)]
    per_arm = {a: {(r["env"], _ep_of(r)): r for r in rows if r["variant"] == a}
               for a in arms}
    pairs = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            eps = sorted({ep for (arm, ep) in outcomes if arm == a} &
                         {ep for (arm, ep) in outcomes if arm == b})
            keys = [(env, ep) for ep in eps
                    if (env, ep) in per_arm[a] and (env, ep) in per_arm[b]]
            if len(keys) < 10:
                continue
            sa = np.array([outcomes[(a, k[1])] for k in keys])
            sb = np.array([outcomes[(b, k[1])] for k in keys])
            delta, p, n_a, n_b = mcnemar(sa, sb)
            pairs.append({"arm_a": a, "arm_b": b, "n": len(keys),
                          "success_a": float(sa.mean()), "success_b": float(sb.mean()),
                          "n_a_wins": n_a, "n_b_wins": n_b, "delta": delta,
                          "p": p, "supported": bool(p < 0.05), "keys": keys})

    res = []
    for metric in _numeric_metrics(rows):
        d = DIRECTION.get(metric, 0)
        if d == 0:
            continue
        agree = total = 0
        detail = []
        for pr in pairs:
            if not pr["supported"]:
                continue
            xa = np.array([float(per_arm[pr["arm_a"]][k].get(metric, np.nan))
                           for k in pr["keys"]])
            xb = np.array([float(per_arm[pr["arm_b"]][k].get(metric, np.nan))
                           for k in pr["keys"]])
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() < 10:
                continue
            md = float(np.median(xa[ok] - xb[ok]))
            total += 1
            if md == 0:
                detail.append(f"{pr['arm_a']}~{pr['arm_b']}(tie)")
                continue
            metric_says_a = d * md > 0          # does the metric prefer arm_a?
            outcome_says_a = pr["delta"] > 0    # does the outcome prefer arm_a?
            agree += int(metric_says_a == outcome_says_a)
            detail.append(f"{pr['arm_a']}>{pr['arm_b']}" if metric_says_a == outcome_says_a
                          else f"{pr['arm_a']}<{pr['arm_b']}")
        if total:
            res.append({"metric": metric, "direction": d, "supported_pairs": total,
                        "agree": agree, "passes": bool(agree == total),
                        "detail": ",".join(detail)})
    return pairs, sorted(res, key=lambda d: (-d["passes"], -d["agree"]))


def plot_audit(rows, env, outdir, expected_range=None):
    """Six panels that show, not assert, why the old metric needed auditing."""
    arms = [a for a in ARM_ORDER if any(r["variant"] == a for r in rows)]
    arms += [a for a in sorted({r["variant"] for r in rows}) if a not in arms]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (ax_conf, ax_pos, ax_wall), (ax_center, ax_trap, ax_edge) = axes

    for arm in arms:
        sub = [r for r in rows if r["variant"] == arm]
        ax_conf.scatter([r["loss_range"] for r in sub], [r["roughness"] for r in sub],
                        color=ARM_COLOR.get(arm, "#888888"),
                        label=ARM_LABEL.get(arm, arm), zorder=3)
    rng = np.array([r["loss_range"] for r in rows], float)
    ro = np.array([r["roughness"] for r in rows], float)
    rho = float(stats.spearmanr(rng, ro).statistic) if len(rows) > 3 else np.nan
    ax_conf.set_xscale("log")
    ax_conf.set_yscale("log")
    ax_conf.set_xlabel("grid contrast  max - min  (log)")
    ax_conf.set_ylabel("roughness = mean(laplacian^2)/var  (log)")
    ax_conf.set_title(f"roughness vs contrast: Spearman = {rho:+.2f} over {len(rows)} "
                      "grids\n'a smoother arm' can be a contrast difference")
    ax_conf.grid(alpha=0.25, which="both")
    ax_conf.legend(fontsize=8)

    ar = float(rows[0]["action_range"]) if rows and rows[0]["action_range"] != "" else 2.0
    ax_pos.add_patch(plt.Rectangle((-ar, -ar), 2 * ar, 2 * ar, fill=False,
                                   ec="k", lw=1.2))
    for arm in arms:
        sub = [r for r in rows if r["variant"] == arm]
        ax_pos.scatter([r["argmin_ax"] for r in sub], [r["argmin_ay"] for r in sub],
                       color=ARM_COLOR.get(arm, "#888888"), s=42, zorder=3,
                       label=f"{ARM_LABEL.get(arm, arm)} "
                             f"({sum(int(r['argmin_is_edge']) for r in sub)}/{len(sub)} "
                             "on wall)")
    if expected_range:
        ax_pos.add_patch(plt.Rectangle((-expected_range, -expected_range),
                                       2 * expected_range, 2 * expected_range,
                                       fill=False, ec="tab:red", ls="--", lw=1.2,
                                       label="range of the checkpoints' own first "
                                             f"actions (~{expected_range:g})"))
    ax_pos.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax_pos.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax_pos.set_aspect("equal")
    ax_pos.set_xlabel("argmin: first fixed action (normalized)")
    ax_pos.set_ylabel("argmin: second fixed action (normalized)")
    ax_pos.set_title("where the reported optimum sits in the swept box\n"
                     "points on the black border = the optimum was censored")
    ax_pos.grid(alpha=0.25)
    ax_pos.legend(fontsize=7, loc="upper left")

    x = np.arange(len(arms))
    wall = [float(np.mean([r["wall_ratio"] for r in rows if r["variant"] == arm]))
            for arm in arms]
    ax_wall.bar(x, wall, color=[ARM_COLOR.get(a, "#888888") for a in arms], alpha=0.85)
    ax_wall.axhline(1.0, color="k", lw=1.0, ls="--")
    ax_wall.set_xticks(x)
    ax_wall.set_xticklabels([ARM_LABEL.get(a, a) for a in arms])
    ax_wall.set_ylabel("min border loss / global min")
    ax_wall.set_title("wall_ratio ~ 1 means the box wall is as good as the "
                      "'optimum',\nso the optimum is not inside the swept box")
    ax_wall.grid(alpha=0.25, axis="y")


    rng_state = np.random.RandomState(0)
    for i, arm in enumerate(arms):
        vals = [r["center_over_min"] for r in rows if r["variant"] == arm]
        ax_center.scatter(i + rng_state.uniform(-0.12, 0.12, len(vals)), vals,
                          color=ARM_COLOR.get(arm, "#888888"), s=38)
    ax_center.axhline(1.0, color="k", lw=1.0, ls="--")
    ax_center.set_xticks(x)
    ax_center.set_xticklabels([ARM_LABEL.get(a, a) for a in arms])
    ax_center.set_yscale("log")
    ax_center.set_ylabel("L(planner's own start) / L(min in box)")
    ax_center.set_title("is the planner's start already the best cell?\n1.0 = no "
                        "room to improve, high = the box has room")
    ax_center.grid(alpha=0.25, axis="y")

    for arm in arms:
        sub = [r for r in rows if r["variant"] == arm]
        ax_trap.scatter([r["basin_frac_1.1"] for r in sub],
                        [r["descent_success_frac"] for r in sub],
                        color=ARM_COLOR.get(arm, "#888888"), s=42, zorder=3,
                        label=ARM_LABEL.get(arm, arm))
    ax_trap.set_xlabel("basin_frac_1.1 (fraction of box within 1.1x min)")
    ax_trap.set_ylabel("descent_success_frac (starts reaching the argmin)")
    ax_trap.set_title("a flat basin is not the same as an easy one: the two "
                      "properties\nthat one roughness number conflates")
    ax_trap.grid(alpha=0.25)
    ax_trap.legend(fontsize=8)

    marg = [float(np.mean([r["margin_to_edge_norm"] for r in rows
                           if r["variant"] == arm])) for arm in arms]
    ax_edge.bar(x - 0.2, marg, 0.4, color=[ARM_COLOR.get(a, "#888888") for a in arms],
                alpha=0.85, label="mean margin of argmin to the wall")
    cens = [float(np.mean([r["censored_interior"] for r in rows
                           if r["variant"] == arm])) for arm in arms]
    ax_edge.bar(x + 0.2, [c * ar for c in cens], 0.4, color="k", alpha=0.35,
                label=f"censored share x {ar:g} (scaled to fit)")
    ax_edge.set_xticks(x)
    ax_edge.set_xticklabels([ARM_LABEL.get(a, a) for a in arms])
    ax_edge.set_ylabel("normalized action units")
    ax_edge.set_title("censoring summary: margin left at the reported optimum\n"
                      "(0 = the optimum is pinned to the wall)")
    ax_edge.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Landscape metric audit: {env}, {len(rows)} grids", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"figA_metric_audit_{env}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Recompute the landscape metric battery on cached grids, "
                    "verify it against the audited CSVs and report what the "
                    "metrics can and cannot support")
    ap.add_argument("--grids-dir", default=LEGACY_GRIDS,
                    help="grid directory to audit (default the legacy "
                         "analysis_outputs/loss_landscape/grids)")
    ap.add_argument("--sweep-dir", default=None,
                    help="an additional grid directory (e.g. the new sweep's)")
    ap.add_argument("--grid", type=int, default=13,
                    help="only grids sampled with this many points per axis "
                         "(default 13; ignored with --all)")
    ap.add_argument("--opt-steps", type=int, default=None,
                    help="only grids with this many GD steps (default: any)")
    ap.add_argument("--action-range", type=float, default=None,
                    help="only grids swept over +-this (default: any)")
    ap.add_argument("--all", action="store_true",
                    help="audit every grid in the directories, no settings filter")
    ap.add_argument("--env", default="pusht", help="env to report (default pusht)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--outcomes", default=None,
                    help="planner_loss_curves.csv: per-episode success flags for "
                         "the ordering gate (only meaningful for grids swept on "
                         "the eval episodes)")
    ap.add_argument("--outcome-seed", type=int, default=100,
                    help="which eval seed's episodes the grids use (default 100)")
    ap.add_argument("--expected-range", type=float, default=3.5,
                    help="the checkpoints' own first actions span about +-this in "
                         "normalized units; drawn on the argmin panel (default 3.5)")
    ap.add_argument("--no-figure", action="store_true")
    return ap.parse_args()


def _print_censoring(rows):
    print("\nCensoring by construction (does the swept box contain the optimum?)")
    print(f"  {'variant':16s} {'n':>3s} {'edge':>6s} {'wall':>7s} {'censored':>9s} "
          f"{'margin':>7s} {'range':>6s}")
    for r in censoring_report(rows):
        print(f"  {r['variant']:16s} {r['n']:3d} {r['argmin_is_edge_frac']:6.2f} "
              f"{r['wall_ratio_mean']:7.3f} {r['censored_interior_frac']:9.2f} "
              f"{r['margin_to_edge_mean']:7.3f} {r['action_range']:6.2f}")
    print("  edge = share of grids whose argmin is on the box border; wall = mean "
          "min-border/min\n  (1.0 = the wall is as good as the reported optimum); "
          "censored = share whose interior min is\n  >5% above the global min.")


def _print_redundancy(rows):
    pairs = redundancy_report(rows)
    print(f"\nMetrics carrying the same message (|Spearman| >= 0.9, "
          f"{len(rows)} grids)")
    if not pairs:
        print("  none")
    for a, b, rho, n in pairs:
        print(f"  {a:24s} vs {b:24s} rho={rho:+.3f} n={n}")


def main():
    args = parse_args()
    dirs = [args.grids_dir] + ([args.sweep_dir] if args.sweep_dir else [])
    rows = []
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"no such grid dir: {d}")
        sel = discover(d, None if args.all else args.grid,
                       args.opt_steps, args.action_range)
        for meta, grid, axs in sel:
            if meta.get("env") and meta["env"] != args.env:
                continue
            rows.append(audit_grid(meta, grid, axs))
        print(f"{os.path.relpath(d, REPO)}: {len(sel)} matching grids")
    if not rows:
        raise SystemExit("no grids to audit; check --grids-dir / --grid / --all")
    os.makedirs(args.outdir, exist_ok=True)

    # 1. does the battery reproduce the audited numbers?
    matched, diffs = reproduction_check(rows)
    print(f"\nReproduction check against loss_landscape/per_episode*.csv "
          f"({matched} grids matched on env/variant/episode)")
    if matched == 0:
        print("  [note] no grid matched an entry in the old CSVs "
              "(different episodes/settings) -- nothing to cross-check here")
    for m, dmax in sorted(diffs.items(), key=lambda kv: -kv[1]):
        flag = "OK " if dmax <= 1e-6 else ("ok ~" if dmax <= 1e-4 else "MISMATCH")
        print(f"  {flag} {m:20s} max|new-old| = {dmax:.3e}")

    _print_censoring(rows)
    _print_redundancy(rows)

    # 2. does a metric separate the arms on matched episodes?
    variants = [v for v in ARM_ORDER if any(r["variant"] == v for r in rows)]
    extra = [v for v in sorted({r["variant"] for r in rows}) if v not in variants]
    combos = []
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            ea = {(r["env"], r["episode"]) for r in rows if r["variant"] == a}
            eb = {(r["env"], r["episode"]) for r in rows if r["variant"] == b}
            if len(ea & eb) >= 5:
                combos.append((a, b, len(ea & eb)))
    for a, b, n in sorted(combos, key=lambda t: -t[2]):
        tests = paired_variant_tests(rows, a, b)
        print(f"\nPaired {ARM_LABEL.get(a, a)} vs {ARM_LABEL.get(b, b)} "
              f"({n} matched episodes, Wilcoxon; descriptive, no outcome reference)")
        if not tests:
            print("  [note] no metric with finite values on >=5 matched episodes")
        for t in tests[:10]:
            print(f"  {t['metric']:24s} n={t['n']:2d} "
                  f"median diff={t['median_diff']:+.4g} "
                  f"(A lower in {100 * t['frac_a_lower']:.0f}% of pairs) "
                  f"p={t['p']:.3f}")
    if extra:
        print(f"  [note] context-only variant(s) present with fewer episodes: "
              f"{', '.join(extra)} (not part of the paired tests)")

    # 3. the gate: ordering vs measured outcome
    outcomes = load_outcomes(args.outcomes, args.env, args.outcome_seed)
    if not outcomes:
        print("\nOrdering gate: SKIPPED -- no outcome data for these grids "
              "(pass --outcomes analysis_outputs/planner_loss_curves.csv and audit "
              "grids swept on the same eval episodes)")
    else:
        pairs, res = ordering_report(rows, outcomes, args.env)
        print(f"\nOrdering gate vs measured success on the same episodes "
              f"(seed {args.outcome_seed}, McNemar p<0.05 defines a direction)")
        for pr in pairs:
            tag = "SUPPORTED" if pr["supported"] else "not sig."
            print(f"  {ARM_LABEL.get(pr['arm_a'], pr['arm_a']):13s} vs "
                  f"{ARM_LABEL.get(pr['arm_b'], pr['arm_b']):13s} n={pr['n']:3d} "
                  f"success {pr['success_a']:.3f} vs {pr['success_b']:.3f} "
                  f"(b-c)/n={pr['delta']:+.2f} McNemar p={pr['p']:.3f}  {tag}")
        if res:
            print(f"\n  {'metric':24s} {'pairs':>5s} {'agree':>5s}  verdict")
            for r in res:
                print(f"  {r['metric']:24s} {r['supported_pairs']:5d} "
                      f"{r['agree']:5d}  {'PASS' if r['passes'] else 'fail'}"
                      f"  [{r['detail']}]")
        else:
            print("  [note] no supported arm pair had grids for both arms")

    # 4. CSV
    columns = ["env", "variant", "episode", "grid", "opt_steps", "action_range",
               "goal_H", "lr", "file"] + _numeric_metrics(rows)
    path = os.path.join(args.outdir, "landscape_metric_audit.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if r.get(c) is None else
                        (f"{r[c]:.10g}" if isinstance(r.get(c), (int, float))
                         else str(r.get(c))) for c in columns])
    print(f"\nSaved {os.path.relpath(path, REPO)} ({len(rows)} grids x "
          f"{len(columns)} columns)")

    if not args.no_figure:
        fig = plot_audit(rows, args.env, os.path.join(args.outdir, "paper"),
                         expected_range=args.expected_range)
        print(f"Saved {os.path.relpath(fig, REPO)}")


if __name__ == "__main__":
    main()

