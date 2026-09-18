#!/usr/bin/env bash
# =============================================================================
# run_landscape_paperstyle_full.sh — overnight run, end to end, INCLUDING the
# paper's own Fig. 4 look (the two-bare-panel version `--paper-style` draws)
#
# Stages (all resumable: an existing grid file is reused, so a run that is killed
# or split across two nights resumes by being re-run with the same knobs -- only
# the missing grids cost anything):
#   1 preflight    every stage's real argv through the real parsers. No GPU.
#   2 gates        landscape_sweep.py verify + anchor. A failure here means the
#                  grid is not the planner's own objective: do not sweep.
#   3 probe        300-step saturation grid -> the step-count decision (~48 min)
#   4 sweep        the paper sweep, 4 arms x E episodes x G^2 cells (~9-14 h)
#   5 figures      house figures: Fig. 4 (4-panel), Fig. 5, Fig. 6
#   6 paper-figs   paper-style Fig. 4: baseline-vs-both + the straighten-only
#                  supplement, from the sweep grids, then both pilot boxes.
#                  CPU only, seconds.
# Stages 1-5 are exactly `run_landscape_night.sh`'s own stages, so a night run
# already started there is continued here without redoing the sweep; stage 6 is
# the only thing this script adds.
#
# Usage:
#   nohup bash run_scripts/run_landscape_paperstyle_full.sh > /tmp/paperstyle.log 2>&1 &
#   EPISODES=4 bash run_scripts/run_landscape_paperstyle_full.sh      # ~9.7 h
#   bash run_scripts/run_landscape_paperstyle_full.sh paper-figures   # redraw only
#
# Knobs (env vars, same names as the night script): ENV=pusht GRID=13 EPISODES=6
#   STEPS=100 ACTION_RANGE=3.5 PROBE_STEPS=300 ARMS OUTDIR PYTHON, plus
#   PAPER_LEVELS=42 (filled-contour levels: the paper's own panels are ~42 flat,
#   area-dominant fill colours with almost no blends) and ALLOW_CONCURRENT=1 to
#   skip the one-landscape-run guard below.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/setup.sh"          # exports DATASET_DIR (and MUJOCO_* paths)
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync}"

NIGHT="$SCRIPT_DIR/run_landscape_night.sh"
PY="${PYTHON:-$HOME/miniconda3/envs/ts/bin/python}"
ENV_NAME="${ENV:-pusht}"
OUTDIR="${OUTDIR:-$PWD/analysis_outputs}"
FIGDIR="$OUTDIR/paper"
SWEEP_DIR="$FIGDIR/grids"
PILOT_DIR="$FIGDIR/grids_pilot_fixed"
ACTION_RANGE="${ACTION_RANGE:-3.5}"
PAPER_LEVELS="${PAPER_LEVELS:-42}"
PILOT_EPISODE="${PILOT_EPISODE:-6}"

# the night script reads its knobs from the environment, so forward them
export ENV="$ENV_NAME" OUTDIR
export GRID="${GRID:-13}" EPISODES="${EPISODES:-6}" STEPS="${STEPS:-100}"
export ACTION_RANGE PROBE_STEPS="${PROBE_STEPS:-300}"
if [[ -n "${ARMS:-}" ]]; then export ARMS; fi

log() { echo; echo "=== $*  ($(date '+%H:%M:%S')) ==="; }

# paper_style <grids-dir> <metrics-csv> [extra args...]
paper_style() {
    local grids="$1" csv="$2"; shift 2
    "$PY" -u analysis/landscape_paper_figure.py landscape \
        --env "$ENV_NAME" --outdir "$OUTDIR" \
        --grids-dir "$grids" --metrics-csv "$csv" \
        --paper-style --paper-levels "$PAPER_LEVELS" "$@"
}

cmd="${1:-all}"
case "$cmd" in
    all)           ;;
    paper-figures) SKIP_PIPELINE=1 ;;
    *) echo "usage: $0 [all|paper-figures]" >&2; exit 2 ;;
esac

if [[ "${SKIP_PIPELINE:-0}" != 1 ]]; then
    # Only one landscape run at a time: the sweep owns the GPU (and a second
    # sweep would just re-do the same grids while the first one holds the memory).
    if [[ "${ALLOW_CONCURRENT:-0}" != 1 ]] && pgrep -f 'analysis/landscape_sweep.py' >/dev/null 2>&1; then
        echo "another landscape run is already active:" >&2
        pgrep -af 'analysis/landscape_sweep.py' >&2
        echo "let it finish (it reuses grids anyway), or set ALLOW_CONCURRENT=1" >&2
        exit 1
    fi

    echo "paper-style night run: env=$ENV_NAME grid=$GRID steps=$STEPS episodes=$EPISODES"
    echo "                       box=+-$ACTION_RANGE probe=$PROBE_STEPS arms=${ARMS:-all}"
    echo "                       grids -> $SWEEP_DIR (existing files are reused)"
    log "1/6 preflight";        bash "$NIGHT" preflight
    log "2/6 gates";            bash "$NIGHT" gates
    log "3/6 probe";            bash "$NIGHT" probe
    log "4/6 sweep";            bash "$NIGHT" sweep
    log "5/6 house figures";    bash "$NIGHT" figures
fi

log "6/6 paper-style Fig. 4 (sweep grids)"
if [[ "${LANDSCAPE_VALIDATE_ARGS:-0}" != 1 && "$(ls -A "$SWEEP_DIR" 2>/dev/null | wc -l)" -eq 0 ]]; then
    echo "no grids in $SWEEP_DIR -- skipping (run the sweep stage first)" >&2
else
    paper_style "$SWEEP_DIR" "$FIGDIR/landscape_sweep_metrics.csv" \
        --action-range "$ACTION_RANGE"
fi

log "6/6 paper-style Fig. 4 (pilot grids, both boxes)"
for ar in "$ACTION_RANGE" 2.0; do
    paper_style "$PILOT_DIR" "$FIGDIR/landscape_pilot_metrics.csv" \
        --tag box --action-range "$ar" --episode "$PILOT_EPISODE"
done

log "done at $(date '+%F %H:%M:%S')"
echo "paper-style figures:"
ls -la --time-style=+%H:%M "$FIGDIR"/fig4_paperstyle_*.png 2>/dev/null || true
echo "stage logs: $FIGDIR/logs"
