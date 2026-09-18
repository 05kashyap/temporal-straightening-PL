#!/usr/bin/env bash
# =============================================================================
# run_landscape_night.sh — unattended run: finish the probing, sweep the grids,
# draw the corrected figures
#
# WHY THIS IS A SEPARATE SCRIPT
# The corrected pilot (grids_pilot_fixed/, swept after the unnormalised-proprio
# bug was fixed) changed the step-count answer: at 30 GD steps the cells still
# sit 41.7% above where 100 steps leaves them, so 30 is not an option, and the
# 100-step surface itself still has a 3.0% spread -- nothing measured so far says
# 100 is *enough*. Measuring that needs one more probe (a 300-step grid, ~48 min),
# and the paper sweep itself is ~9-14 h, which is why this is a night script.
#
# WHAT IT DOES (each stage is also usable on its own)
#   gates    landscape_sweep.py verify + anchor -- the two correctness gates.
#            Aborts the run if either fails; nothing below is worth GPU time if
#            the grid is not the planner's own objective.
#   probe    one 300-step saturation grid (straighten, +-3.5, episode 6, g9) into
#            grids_pilot_fixed/, then `report` reads the 30/100/300 table off the
#            grids. The 30- and 100-step grids are already there and are reused,
#            so this stage costs only the new 300-step grid (~48 min).
#   sweep    the paper sweep: 4 arms x EPISODES eval episodes at GRID x GRID cells
#            and STEPS GD steps, into analysis_outputs/paper/grids/.
#   figures  Fig. 4 (hero) + Fig. 6 (diagnostics) from the sweep grids, the
#            curated metrics CSV, Fig. 5 (planner curves) and the pilot's own
#            Fig. 4 set from grids_pilot_fixed/. All CPU-only.
#   preflight (alias: check-args) replays EVERY stage's real argv through the real
#            argparse parsers (LANDSCAPE_VALIDATE_ARGS=1) and does no work: no GPU,
#            no grid, seconds not minutes. This stage exists because `verify` was
#            once handed a --outdir its parser never declared, which cost the whole
#            run; an argument mismatch now fails before anything is spent.
#   all      preflight -> gates -> probe -> sweep -> figures
#
# COST (measured, batch 1: ~0.117 s per cell per GD step)
#   gates         ~9 min   (4 checkpoints loaded; the anchor gate fails loudly)
#   probe         ~48 min  (81 cells x 300 steps; the 30/100 grids are reused)
#   sweep         GRID^2 x STEPS x 0.117 s per grid:
#                   g13/100 steps -> ~33 min/grid -> 24 grids = 13.2 h  (default)
#                   g13/100 steps -> 16 grids (EPISODES=4)       =  8.8 h
#                   g11/100 steps -> 16 grids (GRID=11 EPISODES=4) =  6.3 h
#   figures       ~3 min
# A grid whose file already exists is REUSED, so a run that is killed (or split
# across two nights) resumes by being re-run with the same arguments; only the
# missing grids cost anything.
#
# Usage:
#   bash run_scripts/run_landscape_night.sh all                 # ~14 h, 6 episodes
#   EPISODES=4 bash run_scripts/run_landscape_night.sh all      # ~9.7 h
#   bash run_scripts/run_landscape_night.sh preflight           # argv check, no GPU
#   bash run_scripts/run_landscape_night.sh probe               # the step decision
#   bash run_scripts/run_landscape_night.sh sweep               # the sweep alone
#   bash run_scripts/run_landscape_night.sh figures             # redraw (no GPU)
#   nohup bash run_scripts/run_landscape_night.sh all > /tmp/night.log 2>&1 &
# Knobs (env vars): ENV, GRID=13, EPISODES=6, STEPS=100, ACTION_RANGE=3.5,
#   PROBE_STEPS=300, ARMS, OUTDIR, LEGACY_DIR, CURVES_DIR, VERIFY_GRID=13,
#   VERIFY_STEPS=80, ANCHOR_EPISODE=6, SKIP_VERIFY=1 (skip the legacy-convention
#   gate only; the anchor gate and everything else still run). Stage logs land in
#   analysis_outputs/paper/logs/.
#
# Notes:
#   * STEPS=100 is deliberately the planner's own budget
#     (planner.sub_planner.opt_steps=100 with max_iter=1, objective mode=last), so
#     a 100-step grid is the planner's own procedure started from every fixed
#     action -- the strongest thing the figure can claim. If `probe` reports a gap
#     much above ~2% from 100 to 300 steps, the surfaces are the planner's
#     100-step outcome rather than the converged landscape, and the figure text
#     must say so (`report` prints the number to quote).
#   * Do not mix grids swept with and without the proprio fix, and do not sweep
#     into a directory that holds pre-fix grids: a dropped grid file is silent.
#     Pre-fix grids live in grids_pilot/ (kept only as the record of the bug).
#   * The sweep is 4 arms x K episodes and nothing else runs meanwhile: this
#     machine's GPU has ~3.6 GB free, and batch >1 aborts under the current NVML
#     mismatch (PYTORCH_CUDA_ALLOC_CONF below is the workaround).
# =============================================================================
set -euo pipefail

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

GRID="${GRID:-13}"                       # cells per axis
EPISODES="${EPISODES:-6}"                # eval episodes per arm (24 grids at 6)
STEPS="${STEPS:-100}"                    # GD steps per cell = the planner's budget
ACTION_RANGE="${ACTION_RANGE:-3.5}"      # box half-range (2 pins the argmin to wall)
PROBE_STEPS="${PROBE_STEPS:-300}"        # longest run of the saturation probe
ARMS="${ARMS:-}"                         # empty = all four arms
PILOT_DIR="$FIGDIR/grids_pilot_fixed"    # the corrected pilot (post-proprio-fix)
SWEEP_DIR="$FIGDIR/grids"
# The two gates read inputs that are deliberately NOT under $FIGDIR, so each names
# its own dir (verify: the frozen reference grids; anchor: the planner's curves).
LEGACY_DIR="${LEGACY_DIR:-$OUTDIR/loss_landscape/grids}"
CURVES_DIR="${CURVES_DIR:-$OUTDIR}"
VERIFY_GRID="${VERIFY_GRID:-13}"         # cached reference grid to re-sweep
VERIFY_STEPS="${VERIFY_STEPS:-80}"       # the step count that grid was swept at
ANCHOR_EPISODE="${ANCHOR_EPISODE:-6}"    # eval episode the anchor gate checks

run_stage() {                            # run_stage <name> <cmd...>
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    if [[ "${PREFLIGHT:-0}" == 1 ]]; then
        log="$LOGDIR/preflight-$name.log"      # never clobber a real stage log
        name="preflight:$name"
    fi
    mkdir -p "$LOGDIR"
    local t0; t0=$(date +%s)
    echo "=== $name  ($(date '+%H:%M:%S')) ==="
    "$@" 2>&1 | tee "$log"
    echo "=== $name done in $(( ($(date +%s) - t0) / 60 )) min ==="
}

arm_args=()
if [[ -n "$ARMS" ]]; then
    # shellcheck disable=SC2206
    arm_args=(--arms $ARMS)
fi
# the arm whose surface is probed at several step counts: the first arm listed,
# or the sweep script's own default when no arm filter was given
SAT_ARM="${ARMS%% *}"
SAT_ARM="${SAT_ARM:-straighten}"

gates() {
    # 1-verify: the convention gate -- the new code must reproduce the frozen legacy
    # grids byte-for-byte (see results/LANDSCAPE_RESULTS.md §1). It reads the cached
    # grids, whose --outdir is analysis_outputs/loss_landscape (NOT the figures'
    # analysis_outputs/paper, which is what used to be passed here), and the step
    # count to re-sweep at is the one in the cached file's own name.
    if [[ "${SKIP_VERIFY:-0}" == 1 ]]; then
        echo "1-verify: SKIPPED (SKIP_VERIFY=1). 2-anchor is the correctness gate;"
        echo "          re-run '$0 gates' without SKIP_VERIFY to check the legacy path."
    else
        run_stage 1-verify "$PY" -u analysis/landscape_sweep.py verify \
            --env "$ENV_NAME" --legacy-dir "$LEGACY_DIR" \
            --grid "$VERIFY_GRID" --steps "$VERIFY_STEPS"
    fi
    # 2-anchor: the correctness gate -- the start cell must reproduce the loss the
    # planner itself logged before its first update. Reads the curves that live in
    # analysis_outputs/ (not in the figures' dir), hence --outdir "$CURVES_DIR".
    run_stage 2-anchor "$PY" -u analysis/landscape_sweep.py anchor \
        --env "$ENV_NAME" --outdir "$CURVES_DIR" \
        --episode "$ANCHOR_EPISODE" "${arm_args[@]}"
}

preflight() {
    # Replay every stage's real argv through the real parsers and stop. A flag a
    # stage does not declare fails HERE, in seconds, instead of after the GPU has
    # been busy for ten minutes -- which is exactly how `verify --outdir` behaved.
    echo "preflight: replaying every stage's argv through argparse (no GPU, no work)"
    export LANDSCAPE_VALIDATE_ARGS=1
    export PREFLIGHT=1
    local rc=0
    gates || rc=1
    probe || rc=1
    sweep || rc=1
    figures || rc=1
    unset LANDSCAPE_VALIDATE_ARGS PREFLIGHT
    if [[ $rc -ne 0 ]]; then
        echo "preflight FAILED: a stage cannot parse the arguments this script passes" >&2
        echo "           (see $LOGDIR/preflight-*.log); nothing was run." >&2
        return 1
    fi
    echo "preflight OK: every stage accepts its argv."
}

probe() {
    mkdir -p "$PILOT_DIR"
    echo "saturation probe: $PROBE_STEPS steps on episode 6 (30/100 are reused)"
    run_stage 3-probe "$PY" -u analysis/landscape_sweep.py pilot \
        --env "$ENV_NAME" --outdir "$FIGDIR" --grids-subdir grids_pilot_fixed \
        --grid 9 --episode-list 6 --pilot-steps 30 100 \
        --pilot-ranges 2.0 "$ACTION_RANGE" \
        --saturation-arm "$SAT_ARM" --saturation-steps "$PROBE_STEPS" \
        "${arm_args[@]}"
    # re-read the gates off the grids: no GPU, and this is the table to quote
    run_stage 3b-report "$PY" -u analysis/landscape_sweep.py report \
        --env "$ENV_NAME" --outdir "$FIGDIR" --grids-subdir grids_pilot_fixed \
        --grid 9 --episode-list 6 --pilot-steps 30 100 \
        --pilot-ranges 2.0 "$ACTION_RANGE" \
        --saturation-steps "$PROBE_STEPS"
    echo
    echo "DECISION INPUT -- in 3b-report's Gate B table read the 100-step row:"
    echo "  mean rel gap <= ~1% -> STEPS=$STEPS (default) is both the planner's own"
    echo "                         budget and a converged surface: say both."
    echo "  mean rel gap  > ~2% -> keep STEPS=$STEPS (it is the planner's budget) and"
    echo "                         quote the gap in the figure text: the surfaces are"
    echo "                         then the planner's 100-step outcome, not the"
    echo "                         converged landscape."
}

sweep() {
    mkdir -p "$SWEEP_DIR"
    echo "sweep: grid $GRID x $GRID, $STEPS steps, $EPISODES episodes,"
    echo "       box +-$ACTION_RANGE, arms ${ARMS:-all} -> $SWEEP_DIR"
    run_stage 4-sweep "$PY" -u analysis/landscape_sweep.py sweep \
        --env "$ENV_NAME" --outdir "$FIGDIR" --grid "$GRID" --opt-steps "$STEPS" \
        --action-range "$ACTION_RANGE" --episodes "$EPISODES" "${arm_args[@]}"
}

figures() {
    run_stage 5-fig-sweep "$PY" -u analysis/landscape_paper_figure.py landscape \
        --env "$ENV_NAME" --outdir "$OUTDIR" \
        --grids-dir "$SWEEP_DIR" \
        --metrics-csv "$FIGDIR/landscape_sweep_metrics.csv" \
        --action-range "$ACTION_RANGE"
    run_stage 6-fig-curves "$PY" -u analysis/landscape_paper_figure.py \
        planner-curves --env "$ENV_NAME" --outdir "$OUTDIR"
    # the pilot's own set, redrawn from the corrected pilot grids (CPU only)
    for ar in "$ACTION_RANGE" 2.0; do
        run_stage "7-fig-pilot-ar$ar" "$PY" -u analysis/landscape_paper_figure.py \
            landscape --env "$ENV_NAME" --outdir "$OUTDIR" \
            --grids-dir "$PILOT_DIR" \
            --metrics-csv "$FIGDIR/landscape_pilot_metrics.csv" \
            --tag box --action-range "$ar" --episode 6
    done
    echo
    echo "figures and curated tables in $FIGDIR:"
    ls -la --time-style=+%H:%M "$FIGDIR"/fig4_*.png "$FIGDIR"/fig5_*.png \
        "$FIGDIR"/fig6_*.png "$FIGDIR"/landscape_paper_metrics*.csv 2>/dev/null || true
}

cmd="${1:-all}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  preflight|check-args) preflight ;;
  gates)   gates ;;
  probe)   probe ;;
  sweep)   sweep ;;
  figures) figures ;;
  all)
    echo "night run: env=$ENV_NAME grid=$GRID steps=$STEPS episodes=$EPISODES"
    echo "           box=+-$ACTION_RANGE probe=$PROBE_STEPS arms=${ARMS:-all}"
    echo "           grids -> $SWEEP_DIR (existing files are reused, so a killed"
    echo "           run resumes by being re-run with the same arguments)"
    preflight
    gates
    probe
    sweep
    figures
    echo
    echo "night run complete at $(date '+%F %H:%M:%S')"
    echo "logs: $LOGDIR"
    ;;
  *)
    sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
    echo "unknown stage: $cmd" >&2
    exit 2
    ;;
esac
