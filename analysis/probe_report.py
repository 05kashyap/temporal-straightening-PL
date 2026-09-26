#!/usr/bin/env python
"""analysis/probe_report.py
=========================
Turn the probe CSVs into the markdown that documents every figure.

analysis/run_probes.py runs the two probes (grounded `--feature-source encoded`
and ungrounded `--feature-source rollout`) for every (env, recipe) group of
checkpoints. This script reads the CSVs those runs leave in --outdir and writes:

    analysis_outputs/probe_report.md          index: one section per figure
    analysis_outputs/<figure_stem>.md         the same section, beside its PNG
    analysis_outputs/linear_probe_results.csv (merged, one row per arm/probe/target)
    analysis_outputs/rollout_probe_{curves,summary}.csv (merged)

The figures themselves carry one short title only (a long sentence used to
overflow the axes box), so the explanation -- what the axes are, what each metric
means, which checkpoint each line is, the hyperparameters and the exact command
that regenerates the figure -- lives here.

Usage:
    python analysis/probe_report.py --outdir analysis_outputs \
        --ckpt-root checkpoints/test [--envs point_maze,point_maze_medium,pusht]
"""

import argparse
import csv
import glob
import os
import sys

try:                                                     # `python analysis/probe_report.py`
    import probe_ckpts                                   # noqa: E402
except ImportError:                                      # `from analysis import probe_report`
    from analysis import probe_ckpts                     # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENVS = ("point_maze", "point_maze_medium", "pusht")

METHODS = {
    "linear_probe": dict(
        title="grounded probe",
        source="encoded (real val frames)",
        csv="linear_probe_results_%s_%s.csv",
        png="linear_probe_%s_%s.png",
        blurb=(
            "Bars: held-out mean R^2 of a **ridge** probe that reads the agent's true "
            "position (the first two state dimensions) out of the *real* encoder latents "
            "of held-out episodes. Higher means the frozen representation still carries "
            "the agent's location. The same run also fits a small **MLP** probe and a "
            "**full-state** target; those numbers are in the tables below and in the CSV."),
        metrics=(
            "`mean_r2` = mean over output dimensions of the per-dimension R^2 on the "
            "probe-test split (episodes held out from the probe's own fit; the split is "
            "by episode identity, never by frame, so no frame can leak between splits). "
            "`per_dim_r2` lists the individual dimensions, `n_test` the number of test "
            "frames.")),
    "rollout_probe": dict(
        title="ungrounded rollout probe",
        source="rollout (the model's own imagined latents)",
        csv="rollout_probe_%s_%s_summary.csv",
        png="rollout_probe_%s_%s.png",
        blurb=(
            "Lines: held-out R^2 of a **single ridge** probe that predicts the agent's "
            "true position from the latents the model *imagines* for step t of its own "
            "rollout (no real frames are encoded after the history window). R^2(1) is "
            "the first predicted step, R^2(t) the t-th; a falling curve means the "
            "imagined latents drift away from anything linearly decodable, even though "
            "the pictures may still look plausible."),
        metrics=(
            "`r2_t1` = R^2 at the first predicted step; `r2_tmax` = R^2 at the arm's own "
            "horizon; `retention` = r2_tmax / r2_t1 (how much decodability survives); "
            "`r2_all` = the pooled R^2 over every (episode, timestep) row; "
            "`shared_horizon`/`shared_r2`/`shared_retention` are the same quantities at "
            "min(T_max) over the arms of this group, which is the honest comparison when "
            "the arms have different horizons. `n_train_eps`/`n_test_eps` are the "
            "episodes used per split, `n_dropped` those too short for T_max.")),
}


def read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge(outdir, pattern, out_name):
    """Concatenate every per-group CSV matching `pattern` into one file."""
    files = sorted(glob.glob(os.path.join(outdir, pattern)))
    rows, fields = [], []
    for p in files:
        for r in read_rows(p):
            for k in r:
                if k not in fields:
                    fields.append(k)
            rows.append(r)
    if not rows:
        return None, 0
    out_path = os.path.join(outdir, out_name)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return out_path, len(files)


def fnum(v, digits=4):
    try:
        return ("%%.%df" % digits) % float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else "-"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def ckpt_path(row, ckpt_root):
    """Checkpoint run dir for a CSV row (basename -> path, relative to the repo)."""
    ck = row.get("ckpt", "")
    if not ck:
        return "?"
    for root in ("checkpoints/test", ckpt_root):
        cand = os.path.join(root, ck)
        if root and os.path.isdir(cand):
            return os.path.relpath(cand, REPO)
    return ck


def ckpts_arg(ckpt_by_arm):
    return ",".join("%s=%s" % (a, d) for a, d in sorted(ckpt_by_arm.items()))


def render_grounded(env, label, rows, ckpt_root, outdir):
    """Markdown section for one grounded (bar) figure."""
    sel = [r for r in rows if r["env"] == env and r["label"] == label]
    if not sel:
        return None
    by_arm = {}
    for r in sel:
        by_arm.setdefault(r["variant"], {})[(r["target"], r["probe_type"])] = r
    stem = "linear_probe_%s_%s" % (env, label)
    first = sel[0]
    ckpt_by_arm = {a: ckpt_path(by_arm[a][sorted(by_arm[a])[0]], ckpt_root) for a in by_arm}

    def cell(arm, key):
        r = by_arm[arm].get(key)
        return fnum(r["mean_r2"]) if r else "-"

    body = ["# `%s.png`" % stem,
            "",
            "![%s](%s.png)" % (stem, stem),
            "",
            "**Method** %s (`--feature-source encoded`), on the *real* frames of held-out "
            "episodes." % METHODS["linear_probe"]["title"],
            "",
            "**Group** env `%s`, recipe `%s`" % (env, label),
            "",
            METHODS["linear_probe"]["blurb"],
            "",
            "## Checkpoints in this figure",
            "",
            table(["arm", "checkpoint run dir"],
                  [(a, "`%s`" % ckpt_by_arm[a]) for a in ckpt_by_arm]),
            "",
            "## Numbers",
            "",
            table(["arm", "R^2 pos (ridge)", "R^2 pos (MLP)", "R^2 full (ridge)",
                   "R^2 full (MLP)", "n_test"],
                  [(a, cell(a, ("pos", "ridge")), cell(a, ("pos", "mlp")),
                    cell(a, ("full", "ridge")), cell(a, ("full", "mlp")),
                    (by_arm[a][sorted(by_arm[a])[0]].get("n_test", "-"))) for a in by_arm]),
            "",
            "## Metric definitions",
            "",
            METHODS["linear_probe"]["metrics"],
            "",
            "## Run parameters",
            "",
            table(["split seed", "test fraction"],
                  [(first.get("seed", "-"), first.get("test_frac", "-"))]),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python analysis/linear_probe.py --env %s --label %s \\" % (env, label),
            "  --feature-source encoded --outdir %s \\"
            % os.path.relpath(outdir, REPO),
            "  --ckpts \"%s\"" % ckpts_arg(ckpt_by_arm),
            "```",
            "",
            "Raw rows: `%s.csv` (also merged into `linear_probe_results.csv`)." % stem,
            ""]
    return stem, "\n".join(body)


def render_rollout(env, label, summaries, curve_rows, ckpt_root, outdir):
    """Markdown section for one ungrounded (R^2(t) curve) figure."""
    sel = [r for r in summaries if r["env"] == env and r["label"] == label]
    if not sel:
        return None
    by_arm = {r["variant"]: r for r in sel}
    stem = "rollout_probe_%s_%s" % (env, label)
    first = sel[0]
    ckpt_by_arm = {a: ckpt_path(by_arm[a], ckpt_root) for a in by_arm}

    def curve(arm):
        pts = [r for r in curve_rows
               if r["env"] == env and r["label"] == label and r["variant"] == arm
               and not str(r["t"]).strip().lower().startswith("all")]
        return {int(float(r["t"])): r["r2"] for r in pts}

    curves = {a: curve(a) for a in by_arm}
    steps = [t for t in (1, 2, 4, 8) if any(t in curves[a] for a in curves)]

    body = ["# `%s.png`" % stem,
            "",
            "![%s](%s.png)" % (stem, stem),
            "",
            "**Method** %s (`--feature-source rollout`): the probe reads the latents the "
            "model *predicts* for its own rollout, not the real frames."
            % METHODS["rollout_probe"]["title"],
            "",
            "**Group** env `%s`, recipe `%s`" % (env, label),
            "",
            METHODS["rollout_probe"]["blurb"],
            "",
            "## Checkpoints in this figure",
            "",
            table(["arm", "checkpoint run dir", "T_max", "R^2(t=1)", "R^2(T_max)",
                   "retention", "shared horizon", "R^2(shared)", "retention(shared)",
                   "train/test episodes", "dropped"],
                  [(a, "`%s`" % ckpt_by_arm[a], by_arm[a].get("t_max", "-"),
                    fnum(by_arm[a].get("r2_t1")), fnum(by_arm[a].get("r2_tmax")),
                    fnum(by_arm[a].get("retention")),
                    by_arm[a].get("shared_horizon", "-"),
                    fnum(by_arm[a].get("shared_r2")),
                    fnum(by_arm[a].get("shared_retention")),
                    "%s/%s" % (by_arm[a].get("n_train_eps", "-"),
                               by_arm[a].get("n_test_eps", "-")),
                    by_arm[a].get("n_dropped", "-")) for a in by_arm]),
            ""]
    if steps:
        body += ["## Curve digest (R^2 at step t)",
                 "",
                 table(["arm"] + ["t=%d" % t for t in steps] + ["t=T_max"],
                       [(a, *[fnum(curves[a].get(t, "nan")) for t in steps],
                         fnum(by_arm[a].get("r2_tmax"))) for a in by_arm]),
                 "",
                 "The full curve is in `%s_curves.csv`." % stem,
                 ""]
    body += ["## Metric definitions",
             "",
             METHODS["rollout_probe"]["metrics"],
             "",
             "## Run parameters",
             "",
             table(["split seed", "test fraction", "pooling mode", "num_hist",
                    "frameskip", "shared horizon"],
                   [(first.get("seed", "-"), first.get("test_frac", "-"),
                     first.get("mode", "-"), first.get("num_hist", "-"),
                     first.get("frameskip", "-"), first.get("shared_horizon", "-"))]),
             "",
             "T_max is the common horizon of the group: by default the shortest episode "
             "horizon, so nothing is dropped; episodes shorter than it are excluded per "
             "arm and counted in the `dropped` column. The history window (`num_hist` "
             "frames) is fed to the predictor but is not probed, because those frames are "
             "real encodings rather than predictions.",
             "",
             "## Reproduce",
             "",
             "```bash",
             "python analysis/linear_probe.py --env %s --label %s \\" % (env, label),
             "  --feature-source rollout --outdir %s \\"
             % os.path.relpath(outdir, REPO),
             "  --ckpts \"%s\"" % ckpts_arg(ckpt_by_arm),
             "```",
             "",
             "Raw rows: `%s_curves.csv` + `%s_summary.csv` (both merged into "
             "`rollout_probe_curves.csv` / `rollout_probe_summary.csv`)." % (stem, stem),
             ""]
    return stem, "\n".join(body)


def main():
    ap = argparse.ArgumentParser(description="Markdown report for the probe figures")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--ckpt-root", default=os.path.join(REPO, "checkpoints", "test"))
    ap.add_argument("--envs", default=",".join(DEFAULT_ENVS),
                    help="envs to report on (default: the three requested)")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip rewriting the merged single-file CSVs")
    args = ap.parse_args()
    envs = set(e for e in args.envs.split(",") if e)
    os.makedirs(args.outdir, exist_ok=True)

    grounded = []
    for p in sorted(glob.glob(os.path.join(args.outdir, "linear_probe_results_*.csv"))):
        grounded += read_rows(p)
    summaries, curves = [], []
    for p in sorted(glob.glob(os.path.join(args.outdir, "rollout_probe_*_summary.csv"))):
        summaries += read_rows(p)
    for p in sorted(glob.glob(os.path.join(args.outdir, "rollout_probe_*_curves.csv"))):
        curves += read_rows(p)

    merges = []
    if not args.no_merge:
        for pat, name in (("linear_probe_results_*.csv", "linear_probe_results.csv"),
                          ("rollout_probe_*_summary.csv", "rollout_probe_summary.csv"),
                          ("rollout_probe_*_curves.csv", "rollout_probe_curves.csv")):
            path, n = merge(args.outdir, pat, name)
            if path:
                merges.append((os.path.basename(path), n))

    figures = []
    g_keys = sorted({(r["env"], r["label"]) for r in grounded if r.get("env") in envs})
    for (env, label) in g_keys:
        out = render_grounded(env, label, grounded, args.ckpt_root, args.outdir)
        if out:
            figures.append(("linear_probe", env, label, out[0], out[1]))
    r_keys = sorted({(r["env"], r["label"]) for r in summaries if r.get("env") in envs})
    for (env, label) in r_keys:
        out = render_rollout(env, label, summaries, curves, args.ckpt_root, args.outdir)
        if out:
            figures.append(("rollout_probe", env, label, out[0], out[1]))

    if not figures:
        raise SystemExit("no per-group CSVs found in %s -- run analysis/run_probes.py first"
                         % args.outdir)

    index = ["# Probe report",
             "",
             "One section per figure (details, checkpoints, numbers and the command that "
             "regenerates it). Each figure also has its own `<figure>.md` next to it.",
             ""]
    if merges:
        index += ["Merged tables: "
                  + ", ".join("`%s` (%d per-group files)" % (n, c) for n, c in merges),
                  ""]
    for kind, env, label, stem, md in figures:
        with open(os.path.join(args.outdir, stem + ".md"), "w") as fh:
            fh.write(md)
        index += ["## %s" % stem,
                  "",
                  "![%s](%s.png)" % (stem, stem),
                  "",
                  "- method: %s" % METHODS[kind]["title"],
                  "- env / recipe: `%s` / `%s`" % (env, label),
                  "- details: [`%s.md`](%s.md)  ·  curve/summary CSV: `%s*.csv`"
                  % (stem, stem, stem),
                  ""]
    index_path = os.path.join(args.outdir, "probe_report.md")
    with open(index_path, "w") as fh:
        fh.write("\n".join(index))
    print("wrote %s (%d figures) + per-figure .md" % (index_path, len(figures)))
    for kind, env, label, stem, _md in figures:
        print("   %-10s %-20s %-10s %s.png" % (kind, env, label, stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())


