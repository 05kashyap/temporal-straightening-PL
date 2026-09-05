#!/usr/bin/env bash
# =============================================================================
# kaggle_train.sh -- train this repo's world models on Kaggle WITHOUT any
# MuJoCo / gym / planning eval. Only train.py is invoked; plan.py is never run.
# (train.py's own per-epoch val() is model-internal latent metrics on the
# loaded trajectories -- no simulator involved.)
#
# Usage:
#     bash kaggle_train.sh <env> <variant> [epochs] [batch_size]
#
#     env:        point_maze | point_maze_medium | pusht | wall
#     variant:    baseline | straighten | twothirds | both     (default: straighten)
#     epochs:     optional (defaults per env: 20, pusht = 2)
#     batch_size: optional (defaults per env: mazes 32, pusht/wall 16)
#
# Environment variables:
#     DATASET_DIR   (required) folder that CONTAINS the env dataset directory,
#                   e.g. $DATASET_DIR/point_maze/{states.pth, actions.pth,
#                   seq_lengths.pth, obses/}. If you uploaded the dataset as a
#                   Kaggle Dataset, use DATASET_DIR=/kaggle/input/<slug>.
#     EPOCHS        optional override for epochs (3rd positional arg wins).
#     NUM_HIST      optional predictor context frames (default 6; conf/train.yaml
#                   default is 3). Same knob as run.sh's NUM_HIST.
#     USE_GRAD_CHECKPOINT optional 'true' to gradient-checkpoint the predictor
#                   Transformer (memory <-> compute; default false = exact prior numerics).
#     REG_WINDOW    optional P-Reg stats window in frames (reg_window; default =
#                   num_hist+num_pred). May EXCEED num_hist+num_pred: the dataloader
#                   then feeds reg_window frames/sample while the predictor context
#                   stays num_hist.
#
# Kaggle notes:
#   - Run kaggle_setup.sh first (pip deps).
#   - mixed_precision is forced to "no": Kaggle T4/P100 do NOT support bf16
#     (conf/train.yaml defaults to bf16).
#   - has_decoder=False => decoder-less training (the paper/OL_RESULTS setup);
#     it also skips the VQVAE, so no distributed_fn/vqvae deps are needed.
#   - Checkpoints land in ./checkpoints/<env>_<variant>/ under /kaggle/working;
#     download them before the session ends (Kaggle /kaggle/working is wiped).
#   - WANDB_MODE=offline is set; results are stored locally, no API key needed.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ENV="${1:-point_maze}"
VARIANT="${2:-straighten}"
EPOCHS_ARG="${3:-}"
BATCH_ARG="${4:-}"
NUM_HIST="${NUM_HIST:-3}"   # predictor context frames (conf/train.yaml default is 3)
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-false}"  # gradient-checkpoint the predictor Transformer (memory <-> compute); default off = exact prior numerics
REG_WINDOW="${REG_WINDOW:-}"  # P-Reg stats window in frames (reg_window); empty = num_hist+num_pred. Larger than that grows the dataloader window (predictor context stays num_hist).

# --- required dataset mount ------------------------------------------------
export DATASET_DIR="${DATASET_DIR:-${KAGGLE_DATASET_MOUNT:-}}"
if [ -z "$DATASET_DIR" ]; then
    echo "ERROR: set DATASET_DIR to the folder containing the env dataset." >&2
    echo "  e.g.  export DATASET_DIR=/kaggle/input/your-dataset-slug" >&2
    exit 1
fi

export WANDB_MODE=offline

# --- per-env paper defaults (encoder, regularizer strings, batch, epochs) --
case "$ENV" in
    point_maze)
        ENCODER=dino_global
        STRAIGHTEN="cos1e-1"
        TWOTHIRDS="twothirds5e-2"
        BATCH=32
        DEF_EPOCHS=20
        LR=1e-5
        LR_BASELINE=1e-5
        ;;
    point_maze_medium)
        ENCODER=dino_global
        STRAIGHTEN="cos1e-2"     # paper Table 1 dagger: lambda=0.01 for Medium
        TWOTHIRDS="twothirds5e-2"
        BATCH=32
        DEF_EPOCHS=20
        LR=1e-5
        LR_BASELINE=1e-6         # Table 3 footnote: baseline medium lr=1e-6
        ;;
    pusht)
        ENCODER=dino_channel
        STRAIGHTEN="aggcos1e-1"
        TWOTHIRDS="aggtwothirds5e-2"
        BATCH=16                 # 14x14 channel attention; 32 OOMs a 16GB GPU
        DEF_EPOCHS=2             # paper Appendix A.3
        LR=1e-5
        LR_BASELINE=1e-5
        ;;
    wall)
        ENCODER=dino_channel
        STRAIGHTEN="aggcos1e-1"
        TWOTHIRDS="aggtwothirds5e-2"
        BATCH=16
        DEF_EPOCHS=20
        LR=1e-5
        LR_BASELINE=1e-5
        ;;
    *)
        echo "Unknown env '$ENV' (choose point_maze | point_maze_medium | pusht | wall)." >&2
        exit 1
        ;;
esac

EPOCHS="${EPOCHS_ARG:-$DEF_EPOCHS}"
# batch_size: positional arg wins, then $BATCH_SIZE env var, then per-env default.
if [ -n "$BATCH_ARG" ]; then
    BATCH="$BATCH_ARG"
elif [ -n "${BATCH_SIZE:-}" ]; then
    BATCH="$BATCH_SIZE"
fi

# --- variant -> straighten/twothirds/lr ------------------------------------
case "$VARIANT" in
    baseline)
        S_VAL=False
        T_VAL=False
        LR_USED=$LR_BASELINE
        ;;
    straighten)
        S_VAL="$STRAIGHTEN"
        T_VAL=False
        LR_USED=$LR
        ;;
    twothirds)
        S_VAL=False
        T_VAL="$TWOTHIRDS"
        LR_USED=$LR
        ;;
    both)
        S_VAL="$STRAIGHTEN"
        T_VAL="$TWOTHIRDS"
        LR_USED=$LR
        ;;
    *)
        echo "Unknown variant '$VARIANT' (choose baseline | straighten | twothirds | both)." >&2
        exit 1
        ;;
esac

echo "================================================================"
echo " env=$ENV variant=$VARIANT encoder=$ENCODER epochs=$EPOCHS"
echo " straighten=$S_VAL twothirds=$T_VAL encoder_lr=$LR_USED batch=$BATCH num_hist=$NUM_HIST use_grad_checkpoint=$USE_GRAD_CHECKPOINT"
echo " has_decoder=False mixed_precision=no  DATASET_DIR=$DATASET_DIR"
echo "================================================================"

reg_arg=()
if [ -n "$REG_WINDOW" ]; then
    reg_arg=(reg_window="$REG_WINDOW")
fi

python train.py --config-name train.yaml \
    env="$ENV" \
    encoder="$ENCODER" \
    training.straighten="$S_VAL" \
    training.twothirds="$T_VAL" \
    training.batch_size="$BATCH" \
    training.epochs="$EPOCHS" \
    num_hist="$NUM_HIST" \
    predictor.use_grad_checkpoint="$USE_GRAD_CHECKPOINT" \
    training.encoder_lr="$LR_USED" \
    training.mixed_precision=no \
    has_decoder=False \
    model.train_decoder=False \
    env.num_workers=4 \
    "${reg_arg[@]}" \
    hydra.run.dir="checkpoints/${ENV}_${VARIANT}"
