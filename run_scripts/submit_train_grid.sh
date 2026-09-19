#!/usr/bin/env bash
# =============================================================================
# run_scripts/submit_train_grid.sh -- submit the full training grid.
#
#   {umaze, medium, pusht} x {channel, global} = 6 jobs, one GPU each.
#   Each job runs the four arms (baseline, straighten, p_reg, both) sequentially
#   inside the project container; see run_scripts/train_server.slurm.
#
# Run from a login node (needs sbatch + $SCRATCH), from anywhere:
#   bash run_scripts/submit_train_grid.sh
#
# Knobs: EPOCHS NUM_WORKERS CKPT_ROOT ART_ROOT FRESH SKIP_FINISHED OVERLAY_MODE
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/scratch/akn7847}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/datasets/worldmodelcheckpoints}"
mkdir -p "$CKPT_ROOT/logs"          # sbatch --output points into $CKPT_ROOT/logs
cd "$REPO"

for e in umaze medium pusht; do
    for d in channel global; do
        echo "==> $e $d"
        sbatch run_scripts/train_server.slurm "$e" "$d"
    done
done

squeue -u "$USER"
