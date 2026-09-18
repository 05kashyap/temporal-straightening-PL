#!/usr/bin/env bash
# =============================================================================
# run_landscape_paper.sh — the 4-arm loss-landscape / planner-convergence figures
#
# Turns the paper's Fig. 4 (a landscape per checkpoint) into a four-arm,
# mechanism-level figure set for PushT (and later umaze), plus the metric audit
# that keeps it honest. It does NOT touch the existing two-arm harness
# (analysis/loss_landscape_comparison.py + run_loss_landscape_comparison.sh),
# which keeps its own grids, CSVs and figures exactly as they are.
#
# Stages (each is also usable on its own):
#   curves   helpers/extract_planner_curves.py -> the GD planner's OWN per-step
#            loss curves, read out of the offline wandb datastores of the runs
#            behind results/AGG_RESULTS.MD (free: no GPU, no planning), plus
#            analysis/landscape_paper_figure.py planner-curves (Fig. 5)
#   audit    analysis/landscape_metric_audit.py -> the whole metric battery on
#            cached grids, verified against the existing audited CSVs, with the
#            censoring/confound report and the ordering gate
#   verify   analysis/landscape_sweep.py verify -> re-sweeps cached legacy grids
#            and prints max|new-cached| and the real speedup (this is the check
#            that licenses sweeping new grids at all)
#   pilot    analysis/landscape_sweep.py pilot -> 4 arms x 1 episode x
#            (opt_steps, box) choices; fixes the two design knobs
#   sweep    analysis/landscape_sweep.py sweep -> 4 arms x K eval episodes
#            (the grids the figures are drawn from)
#   figures  analysis/landscape_paper_figure.py (Fig. 4 hero + Fig. 5 + Fig. 6)
#   all      curves -> audit -> verify -> pilot -> sweep -> figures, each stage
#            logged to analysis_outputs/paper/logs/<stage>.log
#
# Usage:
#   bash run_landscape_paper.sh all                 # everything, PushT
#   bash run_landscape_paper.sh curves              # Fig. 5 only (seconds, no GPU)
#   bash run_landscape_paper.sh audit               # metric audit (CPU only)
#   bash run_landscape_paper.sh verify              # correctness + speedup check
#   ENV=point_maze bash run_landscape_paper.sh sweep --episodes 6
#   bash run_landscape_paper.sh sweep --episodes 6 --action-range 3.5
#   EXTRA_ARGS are passed straight to the stage script, so every stage's own
#   --help is the reference for its options.
#
# Notes:
#   * Stage sizes: one grid is grid^2 x opt_steps rollouts; the runner prints the
#     measured per-grid time in `verify`, so the sweep budget is a measurement and
#     not a guess (the legacy sweep took ~27 min for a 13x13/80 grid).
#   * Every grid file name carries its own settings (episode, arm, grid,
#     opt_steps, action_range), and an existing file is reused instead of
#     re-swept, so a crashed long run resumes by simply being re-run.
#   * The four arms' absolute loss levels are NOT comparable (each checkpoint has
#     its own latent scale); the scripts therefore only ever report shapes,
#     per-arm normalised curves and step counts -- no "arm X has lower loss".
#   * The GPU driver on this machine currently reports an NVML version mismatch
#     ("nvidia-smi: Failed to initialize NVML"), which makes torch's caching
#     allocator abort at batch >1. PYTORCH_CUDA_ALLOC_CONF below works around it;
#     a reboot/reload of the nvidia module also fixes it properly.
# =============================================================================
set -euo pipefail
# Resolve the repo root THIS script lives under, instead of assuming the caller's
# directory: the code paths below (helpers/..., analysis/...) and setup.sh's
# DATASET_DIR default ($PWD/data/datasets) are both relative to the repo root, so
# the script works the same whether it is invoked as
# `bash run_scripts/run_landscape_paper.sh` from the root or from run_scripts/.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/setup.sh"            # exports DATASET_DIR (and MUJOCO_* paths)
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"
PY="${PYTHON:-$HOME/miniconda3/envs/ts/bin/python}"
ENV_NAME="${ENV:-pusht}"
OUTDIR="${OUTDIR:-$PWD/analysis_outputs}"
FIGDIR="$OUTDIR/paper"
LOGDIR="$FIGDIR/logs"

run_stage() {                            # run_stage <name> <cmd...>
    local name="$1"; shift
    mkdir -p "$LOGDIR"
    echo "=== $name ==="
    "$@" 2>&1 | tee "$LOGDIR/$name.log"
}

cmd="${1:-all}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  curves)
    exec "$PY" -u helpers/extract_planner_curves.py --envs "$ENV_NAME" \
         --outdir "$OUTDIR" "$@"
    ;;
  audit)
    exec "$PY" -u analysis/landscape_metric_audit.py --env "$ENV_NAME" \
         --outdir "$OUTDIR" "$@"
    ;;
  verify)
    exec "$PY" -u analysis/landscape_sweep.py verify --env "$ENV_NAME" \
         --outdir "$FIGDIR" "$@"
    ;;
  anchor)
    # the gate that licenses a new grid: the objective at the planner's own start
    # action must reproduce the loss the planner logged there (plan_0/loss). This
    # is what caught the unnormalised proprio channel; `verify` still checks the
    # numerics of sweep_grid against the cached legacy grids.
    exec "$PY" -u analysis/landscape_sweep.py anchor --env "$ENV_NAME" "$@"
    ;;
  pilot)
    exec "$PY" -u analysis/landscape_sweep.py pilot --env "$ENV_NAME" \
         --outdir "$FIGDIR" "$@"
    ;;
  sweep)
    exec "$PY" -u analysis/landscape_sweep.py sweep --env "$ENV_NAME" \
         --outdir "$FIGDIR" "$@"
    ;;
  figures)
    exec "$PY" -u analysis/landscape_paper_figure.py "$@" --outdir "$OUTDIR"
    ;;
  all)
    mkdir -p "$LOGDIR"
    run_stage 1-curves  "$PY" -u helpers/extract_planner_curves.py \
        --envs "$ENV_NAME" --outdir "$OUTDIR"
    run_stage 2-audit   "$PY" -u analysis/landscape_metric_audit.py \
        --env "$ENV_NAME" --outdir "$OUTDIR" "$@"
    run_stage 3-verify  "$PY" -u analysis/landscape_sweep.py verify \
        --env "$ENV_NAME" --outdir "$FIGDIR"
    run_stage 3b-anchor "$PY" -u analysis/landscape_sweep.py anchor \
        --env "$ENV_NAME"
    run_stage 4-pilot   "$PY" -u analysis/landscape_sweep.py pilot \
        --env "$ENV_NAME" --outdir "$FIGDIR" "$@"
    run_stage 5-sweep   "$PY" -u analysis/landscape_sweep.py sweep \
        --env "$ENV_NAME" --outdir "$FIGDIR" "$@"
    run_stage 6-figures "$PY" -u analysis/landscape_paper_figure.py planner-curves \
        --env "$ENV_NAME" --outdir "$OUTDIR"
    echo
    echo "stage logs: $(ls "$LOGDIR"/*.log | tr '\n' ' ')"
    ;;
  *)
    sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
    echo "unknown stage: $cmd" >&2
    exit 2
    ;;
esac
