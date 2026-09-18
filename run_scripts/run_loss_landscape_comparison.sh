#!/usr/bin/env bash
# =============================================================================
# run_loss_landscape_comparison.sh — paired terminal-loss-landscape smoothness
#
# Makes the paper's Fig. 4 quantitative: for each env it sweeps K held-out val
# episodes in the action-space loss landscape of TWO checkpoints that differ
# only by the added two-thirds p-reg term
#     straighten          (aggcos1e-1)
#     straighten + p-reg  (aggcos1e-1 + aggtwothirds5e-2)
# with the SAME episode index, grid, optimizer settings and val set for both,
# and reports per env/checkpoint
#     roughness = mean(laplacian(grid_loss)^2) / var(grid_loss)   (lower better)
#     local_minima = strict 8-neighbour local minima of the grid  (1 = single basin)
# plus paired win counts ("5/6 episodes") and illustrative two-panel figures.
# Everything lands in analysis_outputs/:
#     loss_landscape_comparison.csv, loss_landscape/per_episode.csv,
#     loss_landscape/grids/*.npz, loss_landscape/landscape_<env>_ep<idx>.png
# plus, only with --baseline:
#     loss_landscape/per_episode_context.csv, loss_landscape/fig4_<env>_ep<idx>.png
#
# Usage:
#   bash run_loss_landscape_comparison.sh                 # full run: PushT, K=6, grid=13
#   bash run_loss_landscape_comparison.sh --smoke         # fast plumbing check
#   bash run_loss_landscape_comparison.sh --reuse-grids   # resume from cached grids
#   bash run_loss_landscape_comparison.sh --reuse-grids --viz-only   # metrics+figures
#                                                         # from the cache, no GPU work
#   bash run_loss_landscape_comparison.sh --reuse-grids --diagnostics  # + the extra
#                                                         # diagnostic figure set
#   bash run_loss_landscape_comparison.sh --reuse-grids --baseline --fig-style paper
#                                                         # + the no-straightening
#                                                         # context panel (6 extra grids)
#                                                         # and the paper-style Fig. 4
#   ... --baseline --baseline-ckpt <dir>                  # point that panel at another run
#   ENVS="pusht point_maze" bash run_loss_landscape_comparison.sh   # add umaze
#   bash run_loss_landscape_comparison.sh --episodes 2 --grid 9 --opt-steps 100
# Notes:
#   * ENVS / EPISODES / GRID / OPT_STEPS (env vars) size the run. ENVS defaults to
#     PushT only: umaze (point_maze) is opt-in, and point_maze_medium has no
#     matching p-reg pair yet (empty MODEL_DIRS placeholders -> skipped).
#     A --envs on the command line comes after and therefore overrides ENVS.
#   * --baseline is a CONTEXT read, not a third arm: the no-straightening
#     checkpoint is swept on the same episodes/settings (that pairing is asserted)
#     so it can be drawn as the reference panel, but its rows go to
#     per_episode_context.csv and it enters no paired statistic, win count or
#     headline -- the paired CSVs are identical with and without it. It costs one
#     more grid per episode (~+50% wall time), so leave it off unless the panel is
#     wanted; --fig-style paper renders fig4_*.png (bicubic display of the raw
#     swept grid: every printed number still comes from the raw grid).
#   * GRID defaults to 13, the paper's Fig. 4 sizing; 9 is 2.1x cheaper.
#   * With --smoke the script's own preset (1 episode, grid 5, 10 steps) is left
#     alone, so the plumbing check stays cheap; pass --smoke --grid 7 to smoke
#     bigger.
#   * the full default run is envs x 2 variants x 6 episodes x grid^2 x opt_steps
#     rollouts, i.e. ~6 h on a laptop GPU for the PushT-only default (~12 h with
#     point_maze added); grids are cached, so a crashed run is resumed with
#     --reuse-grids at almost no extra cost.
#   * --viz-only never runs a sweep and needs no GPU: it rebuilds the metrics,
#     CSVs and figures from the cached grids (and fails loudly if one is missing).
#     Cached grid filenames carry every optimizer setting (grid, opt_steps, lr,
#     action_range, goal_H); a filename without lr/action_range/goal_H is a legacy
#     cache, which is only ever READ, so a non-default run cannot overwrite it.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

source setup.sh                          # exports DATASET_DIR (and MUJOCO_* paths)
export WANDB_MODE="${WANDB_MODE:-offline}"
PY="${PYTHON:-$HOME/miniconda3/envs/ts/bin/python}"

# No injected sizes in smoke mode: an explicit --episodes/--grid/--opt-steps beats
# the script's smoke preset, which would silently turn a plumbing check into the
# full multi-hour run. ENVS is unquoted on purpose so ENVS="pusht point_maze"
# splits into two values; a --envs passed on the command line comes later and so
# overrides it (argparse keeps the last occurrence).
args=(--outdir "${OUTDIR:-$PWD/analysis_outputs}" --envs ${ENVS:-pusht})
if [[ " $* " != *" --smoke "* ]]; then
    args+=(--episodes "${EPISODES:-6}" --grid "${GRID:-13}" --opt-steps "${OPT_STEPS:-80}")
fi

exec "$PY" -u analysis/loss_landscape_comparison.py "${args[@]}" "$@"
