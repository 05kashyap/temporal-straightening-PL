#!/usr/bin/env bash
# =============================================================================
# run_curvature_analysis.sh — two-thirds power-law + loss-landscape plots
#
# Quick qualitative plots for PointMaze-umaze and PushT (one held-out episode):
#   1. pooled log-log slope of speed vs curvature (expect ~ -1/3 for
#      two-thirds / both models)          -> analysis_outputs/slope_<env>.png
#   2. aligned speed/curvature time series -> analysis_outputs/timeseries_<env>.png
#   3. action-space loss landscape (PushT) -> analysis_outputs/landscape_pusht.png
# plus analysis_outputs/summary.json (slopes, bootstrap CIs, landscape stats).
#
# Usage:
#   bash run_curvature_analysis.sh                 # defaults (single episode)
#   bash run_curvature_analysis.sh --max-episodes 80   # pooled slope w/ CI
#   bash run_curvature_analysis.sh --grid 15 --opt-steps 100  # finer landscape
#   bash run_curvature_analysis.sh --env point_maze      # just one env
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

source setup.sh                          # exports DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-offline}"
PY="${PYTHON:-/home/shanveen-ortho-clinic/miniconda3/envs/ts/bin/python}"

# Single episode by default (fast, per-episode geometry is the point of the
# plots); pass through any extra args the user wants (--max-episodes, --grid, ...).
exec "$PY" curvature_analysis.py \
    --env point_maze --env pusht \
    --max-episodes "${MAX_EPISODES:-1}" \
    --grid "${GRID:-11}" \
    --opt-steps "${OPT_STEPS:-50}" \
    --bootstrap "${BOOTSTRAP:-1000}" \
    "$@"
