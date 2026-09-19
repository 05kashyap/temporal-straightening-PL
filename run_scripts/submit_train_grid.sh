#!/usr/bin/env bash
# =============================================================================
# run_scripts/submit_train_grid.sh -- submit the full training grid.
#
#   {umaze,medium,pusht} x {channel,global} = 6 configs, 4 arms each.
#
# TWO LAYOUTS (MODE):
#
#   MODE=one-gpu (default) -- ONE sbatch job, --gres=gpu:1, and all six configs
#     run CONCURRENTLY on that single GPU (each config still runs its four arms
#     sequentially). Slurm cannot give one GPU to several ALLOCATIONS -- every
#     --gres=gpu:1 job owns its device -- so the six processes must live inside
#     one allocation: train_server.slurm forwards its two args to
#     train_server.sh, and `all all` makes that script fan out and wait (see the
#     GRID MODE docs in train_server.sh). A run needs ~20 GB of VRAM, so six fit
#     in an H200s 140 GB with room to spare. If the three `global` (batch 32)
#     runs ever OOM, stage them:
#       CONFIGS="umaze:global medium:global pusht:global" bash run_scripts/submit_train_grid.sh
#
#   MODE=per-gpu -- the original six independent jobs, one GPU each (identical to
#     `bash run_scripts/train_server.slurm --grid`). Use it on a node with >= 6
#     free GPUs.
#
# Usage (from a login node, inside the repo):
#   bash run_scripts/submit_train_grid.sh                 # one GPU, 6 configs in parallel
#   MODE=per-gpu bash run_scripts/submit_train_grid.sh    # six jobs, six GPUs
#   DRY_RUN=1 bash run_scripts/submit_train_grid.sh       # print, submit nothing
#
# Knobs (export before calling; forwarded into the job):
#   CPUS=48 MEM=192G TIME=48:00:00        # resources of the single allocation
#   CONFIGS="umaze:global medium:global"  # subset / ramping
#   MAX_PARALLEL=6 STAGGER=30 NUM_WORKERS=4
#   EPOCHS=20 FRESH=0 SKIP_FINISHED=1 CKPT_ROOT=... ART_ROOT=... EXCLUDE=...
#
# Why CPUS/MEM matter: each config runs one trainer plus NUM_WORKERS dataloader
# workers, so six configs want ~MAX_PARALLEL*NUM_WORKERS cores and RAM for all
# those workers. GRID mode also drops NUM_WORKERS from the yaml default of 16 to
# 4, because 6x16 = 96 workers would thrash the node and the /scratch filesystem
# (every sample re-reads a whole episode tensor -- there is no episode cache).
#
# The sbatch flags here OVERRIDE the #SBATCH lines inside train_server.slurm, so
# that (gitignored, server-local) file stays exactly as it is. The exported knobs
# reach the container because apptainer passes the host environment through.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/scratch/akn7847}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/datasets/worldmodelcheckpoints}"
SLURM_TEMPLATE="${SLURM_TEMPLATE:-run_scripts/train_server.slurm}"
mkdir -p "$CKPT_ROOT/logs"          # sbatch --output points into $CKPT_ROOT/logs
cd "$REPO"

MODE="${MODE:-one-gpu}"
CPUS="${CPUS:-48}"                  # 6 configs x NUM_WORKERS(4) + slack
MEM="${MEM:-192G}"                  # 6 trainers + their workers
TIME="${TIME:-48:00:00}"
EXCLUDE="${EXCLUDE:-}"

if [[ ! -f "$SLURM_TEMPLATE" ]]; then
    echo "submit_train_grid.sh: $SLURM_TEMPLATE not found." >&2
    echo "  It is gitignored (*.slurm) and lives on the server -- commit it or set SLURM_TEMPLATE=." >&2
    exit 2
fi

case "$MODE" in
    one-gpu)
        sbatch_args=( --job-name=ts-grid --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" )
        if [[ -n "$EXCLUDE" ]]; then sbatch_args+=( --exclude="$EXCLUDE" ); fi
        echo "==> MODE=one-gpu: all configs in parallel on ONE GPU"
        echo "    sbatch ${sbatch_args[*]} $SLURM_TEMPLATE all all"
        echo "    knobs: CONFIGS=${CONFIGS:-<all six>} MAX_PARALLEL=${MAX_PARALLEL:-6} STAGGER=${STAGGER:-30} NUM_WORKERS=${NUM_WORKERS:-4} EPOCHS=${EPOCHS:-20}"
        if [[ "${DRY_RUN:-0}" = "1" ]]; then
            echo "    (DRY_RUN=1: nothing submitted)"
        else
            sbatch "${sbatch_args[@]}" "$SLURM_TEMPLATE" all all
        fi
        ;;
    per-gpu)
        echo "==> MODE=per-gpu: 6 independent jobs, one GPU each"
        for e in umaze medium pusht; do
            for d in channel global; do
                echo "    sbatch $SLURM_TEMPLATE $e $d"
                if [[ "${DRY_RUN:-0}" != "1" ]]; then
                    sbatch "$SLURM_TEMPLATE" "$e" "$d"
                fi
            done
        done
        ;;
    *)
        echo "MODE must be one-gpu or per-gpu (got '$MODE')" >&2
        exit 2
        ;;
esac

if [[ "${DRY_RUN:-0}" = "1" ]]; then
    echo
    echo "DRY_RUN=1: printed the submissions, submitted nothing."
    exit 0
fi

echo
squeue -u "$USER"
