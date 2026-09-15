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
#     batch_size: optional (defaults per env: 16 everywhere -- the 14x14 channel
#                 attention makes the paper's batch 32 OOM even on a 16GB Kaggle GPU)
#
# Encoders: every env trains the SPATIAL channel projector (encoder=dino_channel =
#   DINOv2 patch tokens -> a 14x14x8 channel map + the learned aggregation head) with
#   the agg-mode regularizers, i.e. the paper's Table 1 best rows (with straightening,
#   lambda=0.1 for all spatial features): UMaze 94.00% open-loop / 100.00% MPC,
#   Medium 82.67 / 98.67, PushT 77.33 / 85.33, Wall 90.67 / 100.00. The earlier
#   point_maze / point_maze_medium runs used the 1x384 global projector
#   (encoder=dino_global, Table 1 rows: UMaze 38.67 / 96.00, Medium 22.67 / 78.00)
#   and are no longer produced by this script.
#
# Environment variables:
#     DATASET_DIR   (required) folder that CONTAINS the env dataset directory,
#                   e.g. $DATASET_DIR/point_maze/{states.pth, actions.pth,
#                   seq_lengths.pth, obses/}. If you uploaded the dataset as a
#                   Kaggle Dataset, use DATASET_DIR=/kaggle/input/<slug>.
#     EPOCHS        optional override for epochs (3rd positional arg wins).
#     NUM_HIST      optional predictor context frames (default 3 = paper Table 3
#                   "history frames 3"; same as conf/train.yaml and as the pusht/wall
#                   channel checkpoints already on disk). run.sh uses 3 for the mazes
#                   too. Same knob as run.sh's NUM_HIST.
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
        ENCODER=dino_channel     # paper Table 1 best row: DINOv2(patch)+proj 14x14x8
        STRAIGHTEN="aggcos1e-1"  # spatial features: lambda=0.1 (Table 1 caption)
        TWOTHIRDS="aggtwothirds5e-2"
        BATCH=16                 # 14x14 channel attention; 32 OOMs a 16GB GPU
        DEF_EPOCHS=20
        LR=1e-5
        LR_BASELINE=1e-6         # Table 3 footnote: no-straightening lr=1e-6
        ;;
    point_maze_medium)
        ENCODER=dino_channel     # paper Table 1 best row: DINOv2(patch)+proj 14x14x8
        STRAIGHTEN="aggcos1e-1"  # spatial features: lambda=0.1 (Table 1 caption); the
                                 # old cos1e-2 dagger applied to the 1x384 global row
        TWOTHIRDS="aggtwothirds5e-2"
        BATCH=16                 # 14x14 channel attention; 32 OOMs a 16GB GPU
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
echo " straighten=$S_VAL twothirds=$T_VAL encoder_lr=$LR_USED batch=$BATCH num_hist=$NUM_HIST use_grad_checkpoint=$USE_GRAD_CHECKPOINT reg_window=$REG_WINDOW"
echo " has_decoder=False mixed_precision=mixed_precision  DATASET_DIR=$DATASET_DIR"
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
    training.mixed_precision=mixed_precision \
    has_decoder=False \
    model.train_decoder=False \
    env.num_workers=4 \
    "${reg_arg[@]}" \
    # NOTE: this dir name only encodes env+variant, not the encoder/projector, so a
    # point_maze(medium) run now lands in the same checkpoints/<env>_<variant> folder that
    # the legacy 1x384 global-projector run used. On a fresh Kaggle session that is the
    # narrowest change; if you resume an old session's working dir, delete
    # checkpoints/point_maze* there first so the channel runs do not mix with the old ones.
    hydra.run.dir="checkpoints/${ENV}_${VARIANT}"
