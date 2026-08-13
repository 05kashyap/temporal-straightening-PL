#!/usr/bin/env bash
# =============================================================================
# Temporal straightening + two-thirds pipeline (PointMaze umaze | PushT)
#
#   Step 1: train a baseline world model WITHOUT regularizers
#   Step 2: evaluate it via planning (plan.py: GD + CEM)
#   Step 3: train a world model WITH straightening
#   Step 4: evaluate it via planning (plan.py: GD + CEM)
#   Step 5: train a world model WITH the two-thirds regularizer only
#   Step 6: evaluate it via planning (plan.py: GD + CEM)
#   Step 7: train a world model WITH both straightening and two-thirds
#   Step 8: evaluate it via planning (plan.py: GD + CEM)
#
# ENV=point_maze (default): PointMaze (umaze) task.
# ENV=pusht: PushT task, same 4-variant experiment. Uses the original repo task
#   config (env=pusht, dataset, frameskip, predictor=vit) but with
#   encoder=dino_global -- a trainable projector is REQUIRED for the regularizers
#   to have an effect (encoder=dino is all-frozen, making straighten/twothirds
#   inert). PushT planning adds objective.alpha=1 per the README.
#
# Results are saved under:
#   checkpoints/test/<env>_<straighten>_tt<twothirds>_agg32_.../  training ckpts + config
#   plan_outputs_gd/...  plan_outputs_cem/...   planning logs.json + videos
#   results/                                   aggregated copies of logs.json
#
# Usage:
#   bash run.sh                  # run the full pipeline (resumes existing runs)
#   ENV=pusht bash run.sh        # same experiment on PushT (original repo task config + alpha=1)
#   FRESH=1 bash run.sh          # delete the training run dirs for the selected ENV first
#   BATCH_SIZE=8 bash run.sh      # smaller batch if memory is ever tight
#   N_EVALS=10 bash run.sh       # fewer eval episodes for faster planning
#   TRAIN_DECODER=True bash run.sh  # also train the VQVAE decoder (required for planner videos)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- environment ------------------------------------------------------------
source setup.sh                          # exports DATASET_DIR
export DATASET_DIR="${DATASET_DIR:-$PWD/data/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # no wandb api key on this machine

# MuJoCo 2.1.2 is required by the point_maze simulator during planning eval.
MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco-2.1.2}"
export MUJOCO_PY_MUJOCO_PATH
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export PATH="${PATH:+$PATH:}/home/shanveen-ortho-clinic/miniconda3/envs/ts/bin"  # patchelf (mujoco_py one-time cymj build)

# ---- knobs ------------------------------------------------------------------
PY="${PYTHON:-/home/shanveen-ortho-clinic/miniconda3/envs/ts/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"   # config default; fits this GPU with decoder off + dino_global (3-token attention)
STRAIGHTEN="${STRAIGHTEN:-cos1e-1}"  # README: cos1e-1 = patch-wise curvature regularization
TWOTHIRDS="${TWOTHIRDS:-twothirds5e-2}"  # two-thirds: twothirds5e-2 (cos) or aggtwothirds5e-2 (aggcos)
# Encoder must have a trainable projector (dino_global / dino_channel) for the regularizers to
# have a training effect; encoder=dino (no projector) makes straighten/twothirds inert.
# The RUN_* dir names below assume encoder=dino_global (projglobal / hw1).
EPOCHS="${EPOCHS:-7}"
PLANNERS="${PLANNERS:-gd cem}"   # planners with configs in conf/plan_*.yaml
GOAL_H="${GOAL_H:-25}"           # keep divisible by frameskip (5)
N_EVALS="${N_EVALS:-50}"         # eval episodes (config default); fits: dino_global predictor attends over 3 tokens
NUM_SAMPLES="${NUM_SAMPLES:-200}" # CEM candidates per traj (config default); fits with dino_global
TRAIN_DECODER="${TRAIN_DECODER:-False}"  # also train the VQVAE decoder (required for planner videos)
FRESH="${FRESH:-0}"
ENV="${ENV:-pusht}"     # task: point_maze (default) | pusht

CKBPT="./checkpoints"

# ---- per-task setup ----------------------------------------------------------
#  - point_maze: the original experiment (env=point_maze, encoder=dino_global).
#  - pusht:      original repo task config (env=pusht, dataset, predictor=vit) +
#                encoder=dino_global so the regularizers train the projector;
#                planning adds objective.alpha=1 (README's PushT note).
case "$ENV" in
    point_maze)
        TRAIN_TASK_OVERRIDES="env=point_maze encoder=dino_global"
        PLAN_TASK_OVERRIDES=""
        RUN_FALSE="test/umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_TRUE="test/umaze_${STRAIGHTEN}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/umaze_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_BOTH="test/umaze_${STRAIGHTEN}_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        ;;
    pusht)
        TRAIN_TASK_OVERRIDES="env=pusht encoder=dino_global"
        PLAN_TASK_OVERRIDES="objective.alpha=1"
        RUN_FALSE="test/pusht_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_TRUE="test/pusht_${STRAIGHTEN}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/pusht_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_BOTH="test/pusht_${STRAIGHTEN}_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        ;;
    *)
        echo "Unknown ENV='$ENV' (choose point_maze or pusht)" >&2
        exit 1
        ;;
esac

# ---- helpers ----------------------------------------------------------------
train() {  # $1 = straighten value, $2 = twothirds value, $3 = run dir name
    echo "==== TRAIN env=$ENV straighten=$1 twothirds=$2 decoder=$TRAIN_DECODER ===="
    # shellcheck disable=SC2086  # TRAIN_TASK_OVERRIDES is meant to be word-split
    "$PY" train.py --config-name train.yaml $TRAIN_TASK_OVERRIDES \
        training.straighten="$1" training.twothirds="$2" \
        training.batch_size="$BATCH_SIZE" training.epochs="$EPOCHS" \
        model.train_decoder="$TRAIN_DECODER" has_decoder="$TRAIN_DECODER" \
        hydra.run.dir="$CKBPT/$3"
}

plan_model() {  # $1 = model_name (relative to ckpt_base_path)
    local model_name="$1"
    for planner in $PLANNERS; do
        echo "==== PLAN env=$ENV planner=$planner model=$model_name ===="
        # ckpt_base_path must be absolute: hydra chdirs to the run dir, so a
        # relative path would resolve against the wrong directory. The run dir
        # is also overridden to a clean name for readable output folders.
        local num_samples_arg=()
        if [ "$planner" = "cem" ]; then
            # 200 CEM samples OOM the 12 GB RTX 4070; keep it configurable.
            num_samples_arg=(planner.sub_planner.num_samples="$NUM_SAMPLES")
        fi
        # shellcheck disable=SC2086  # PLAN_TASK_OVERRIDES is meant to be word-split
        "$PY" plan.py --config-name "plan_${planner}.yaml" \
            ckpt_base_path="$PWD/$CKBPT/$model_name" model_name="$model_name" \
            hydra.run.dir="plan_outputs_${planner}/$(echo "$model_name" | tr '/' '_')_gH${GOAL_H}" \
            goal_H="$GOAL_H" n_evals="$N_EVALS" $PLAN_TASK_OVERRIDES "${num_samples_arg[@]}"
    done
}

# ---- optional clean start ---------------------------------------------------
if [ "$FRESH" = "1" ]; then
    echo ">> FRESH=1: deleting existing run dirs for env=$ENV"
    rm -rf "$CKBPT/$RUN_FALSE" "$CKBPT/$RUN_TRUE" "$CKBPT/$RUN_TWOTHIRDS" "$CKBPT/$RUN_BOTH"
fi

# # ---- step 1 & 2: baseline (no regularizers) ---------------------------------
# echo "===================== 1) TRAIN baseline (straighten=False) ============="
# train False False "$RUN_FALSE"
# echo "===================== 2) EVAL baseline model ==========================="
# plan_model "$RUN_FALSE"

# ---- step 3 & 4: straightening ----------------------------------------------
echo "===================== 3) TRAIN (straighten=$STRAIGHTEN) ==============="
train "$STRAIGHTEN" False "$RUN_TRUE"
echo "===================== 4) EVAL straightening model ======================"
plan_model "$RUN_TRUE"

# # ---- step 5 & 6: two-thirds regularizer only ---------------------------------
# echo "===================== 5) TRAIN (twothirds=$TWOTHIRDS) =================="
# train False "$TWOTHIRDS" "$RUN_TWOTHIRDS"
# echo "===================== 6) EVAL two-thirds model ========================="
# plan_model "$RUN_TWOTHIRDS"

# ---- step 7 & 8: straightening + two-thirds ----------------------------------
# echo "===================== 7) TRAIN (straighten=$STRAIGHTEN, twothirds=$TWOTHIRDS) ===="
# train "$STRAIGHTEN" "$TWOTHIRDS" "$RUN_BOTH"
# echo "===================== 8) EVAL both model ==============================="
# plan_model "$RUN_BOTH"

# ---- aggregate results -------------------------------------------------------
echo "===================== Collecting results =============================="
mkdir -p results
if [ -d plan_outputs_gd ] || [ -d plan_outputs_cem ]; then
    find plan_outputs_gd plan_outputs_cem -name logs.json -print0 2>/dev/null |
        while IFS= read -r -d '' f; do
            cp "$f" "results/$(echo "$f" | tr '/' '_')"
            echo "  copied $f"
        done
else
    echo "  (no plan_outputs_* dirs found)"
fi

echo
echo "Done (env=$ENV)."
echo "  Training checkpoints : $CKBPT/test/${ENV}_*"
echo "  Planning logs/videos : plan_outputs_gd/ , plan_outputs_cem/"
echo "  Aggregated logs.json : results/"
