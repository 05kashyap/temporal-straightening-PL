#!/usr/bin/env bash
# =============================================================================
# dataset_paths.sh -- the DINO-WM data layout on this machine, in ONE place.
#
# Sourced by run_scripts/train_server.sh (training), run_mpc.sh and
# setup_mujoco_server.sh (planning/MPC), so those paths cannot disagree about
# where the data is.
#
# Why this file exists: the datasets are NOT nested uniformly on disk --
#
#   $DATA_ROOT/point_maze/point_maze       (one level deeper than the name says)
#   $DATA_ROOT/point_maze_medium
#   $DATA_ROOT/pusht/pusht_noise
#
# Training does not care, because train_server.sh passes the *exact* directory
# (env.dataset.data_path=$DATA_PATH). Planning does: plan.py asks for
# $DATASET_DIR/<env> (conf/env/*.yaml: data_path: ${oc.env:DATASET_DIR}/point_maze),
# so DATASET_DIR has to be the per-env PARENT of the data directory. That is what
# dataset_dir_for() returns, and it is why a single DATASET_DIR cannot serve all
# three envs.
#
# On a new machine, override SCRATCH / DATA_ROOT / CKPT_ROOT / ART_ROOT (or edit
# the defaults below) -- exactly like the old train_server.sh EDIT-ME block.
# =============================================================================
SCRATCH="${SCRATCH:-/scratch/akn7847}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/datasets/worldmodeldata}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/datasets/worldmodelcheckpoints}"
ART_ROOT="${ART_ROOT:-$SCRATCH/datasets/worldmodelart}"

# $1 = umaze|medium|pusht -> the directory the *_dset.py loader reads
data_dir_for() {
    case "$1" in
        umaze)  echo "$DATA_ROOT/point_maze/point_maze" ;;
        medium) echo "$DATA_ROOT/point_maze_medium" ;;
        pusht)  echo "$DATA_ROOT/pusht/pusht_noise" ;;
        *)      echo "" ;;
    esac
}

# $1 = umaze|medium|pusht -> the value plan.py needs as DATASET_DIR
dataset_dir_for() {
    local d
    d="$(data_dir_for "$1")"
    [ -n "$d" ] && dirname "$d"
}
