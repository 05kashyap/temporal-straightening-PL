#!/usr/bin/env bash
# =============================================================================
# Temporal straightening + two-thirds pipeline (PointMaze umaze | PushT | Wall)
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
# ENV=point_maze: PointMaze (umaze) task.
# ENV=pusht (default): PushT task, same 4-variant experiment, but with the SPATIAL
#   channel projector (encoder=dino_channel, 14x14x8) -- the paper reports ~2%
#   open-loop GD success on PushT with the 1-token global projector (dino_global)
#   vs ~70% with the channel projector. Regularizers act on the aggregation head
#   there (aggcos1e-1 / aggtwothirds5e-2). The paper trains PushT for only 2 epochs
#   (Appendix A.3); planning adds objective.alpha=1 per the README and uses
#   reduced n_evals / num_samples because the 14x14 attention is heavier.
# ENV=wall: Wall task, same channel-projector setup (paper App. A.1: 1920 trajs x
#   50 steps, 20 epochs). Planning follows the paper's Section 5.3: start/goal
#   states sampled from test trajectories (goal_source='dset') so goals are
#   reachable within 25 steps, and only the target IMAGE is used in the objective
#   (alpha=0) -- both are the plan_gd/plan_cem defaults, so no plan overrides.
#
# Results are saved under:
#   checkpoints/test/<env>_<straighten>_tt<twothirds>_agg32_.../  training ckpts + config
#   plan_outputs_gd/...  plan_outputs_cem/...   planning logs.json + videos
#   results/                                   aggregated copies of logs.json
#
# Usage:
#   bash run.sh                  # run the full pipeline (resumes existing runs)
#   ENV=wall bash run.sh         # same experiment on Wall (dino_channel + aggcos, dset goals + alpha=0 per paper 5.3)
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
ENV="${ENV:-wall}"     # task: point_maze | pusht (default) | wall

# Per-task defaults (BATCH_SIZE/STRAIGHTEN/TWOTHIRDS/EPOCHS/N_EVALS/NUM_SAMPLES
# are set per task below; explicit env overrides always win).
#  - point_maze uses encoder=dino_global (1 global token) + cos-mode regularizers;
#    20 epochs is the paper protocol for the mazes.
#  - pusht and wall use encoder=dino_channel (14x14x8 spatial features): the paper's
#    Table 1 shows only ~2% open-loop GD success on PushT with the 1-token global
#    projector vs ~70% with the channel projector (wall: 80% -> 90.67% with
#    straightening). Regularizers act on the learned aggregation head (aggcos1e-1 /
#    aggtwothirds5e-2). Pusht trains 2 epochs (paper A.3), wall 20 (paper A.1); the
#    14x14 attention OOMs at batch 32 on the 12 GB GPU, and planning runs 50 evals
#    in chunks of 10 to stay within memory. CEM uses the paper's num_samples=200,
#    rolled out in chunks of 50 (CEM_SAMPLE_CHUNK_SIZE) to fit the same GPU.
case "$ENV" in
    pusht)
        BATCH_SIZE="${BATCH_SIZE:-16}"  # paper default 32 OOMs the 12 GB GPU (dino_channel 14x14 attention); 16 fits, 8 is the safe fallback
        STRAIGHTEN="${STRAIGHTEN:-aggcos1e-1}"
        TWOTHIRDS="${TWOTHIRDS:-aggtwothirds5e-2}"
        EPOCHS="${EPOCHS:-2}"      # paper protocol (A.3: "We train for 2 epochs")
        N_EVALS="${N_EVALS:-50}"   # total eval episodes (processed in CHUNK_SIZE chunks)
        NUM_SAMPLES="${NUM_SAMPLES:-200}"  # paper's CEM candidates per traj (was 50)
        CEM_SAMPLE_CHUNK_SIZE="${CEM_SAMPLE_CHUNK_SIZE:-50}"  # roll the 200 candidates out in chunks of 50 (identical result, fits the 12 GB GPU)
        CHUNK_SIZE="${CHUNK_SIZE:-10}"  # plan/eval episodes in chunks of this size to bound memory
        ;;
    wall)
        BATCH_SIZE="${BATCH_SIZE:-16}"  # paper default 32 OOMs the 12 GB GPU (dino_channel 14x14 attention); 16 fits, 8 is the safe fallback
        STRAIGHTEN="${STRAIGHTEN:-aggcos1e-1}"
        TWOTHIRDS="${TWOTHIRDS:-aggtwothirds5e-2}"
        EPOCHS="${EPOCHS:-20}"     # paper protocol (A.1: "We train for 20 epochs")
        N_EVALS="${N_EVALS:-50}"   # total eval episodes (processed in CHUNK_SIZE chunks)
        NUM_SAMPLES="${NUM_SAMPLES:-200}"  # paper's CEM candidates per traj (was 50)
        CEM_SAMPLE_CHUNK_SIZE="${CEM_SAMPLE_CHUNK_SIZE:-50}"  # roll the 200 candidates out in chunks of 50 (identical result, fits the 12 GB GPU)
        CHUNK_SIZE="${CHUNK_SIZE:-10}"  # plan/eval episodes in chunks of this size to bound memory
        ;;
    *)
        # point_maze
        BATCH_SIZE="${BATCH_SIZE:-32}"   # config default; fits this GPU with decoder off + dino_global (3-token attention)
        STRAIGHTEN="${STRAIGHTEN:-cos1e-1}"  # README: cos1e-1 = patch-wise curvature regularization
        TWOTHIRDS="${TWOTHIRDS:-twothirds5e-2}"  # two-thirds: twothirds5e-2 (cos) or aggtwothirds5e-2 (aggcos)
        EPOCHS="${EPOCHS:-20}"
        N_EVALS="${N_EVALS:-50}"         # eval episodes (config default)
        NUM_SAMPLES="${NUM_SAMPLES:-200}" # CEM candidates per traj (config default)
        CHUNK_SIZE="${CHUNK_SIZE:-}"     # empty = evaluate all n_evals at once (fits for dino_global)
        ;;
esac
# Encoder must have a trainable projector (dino_global / dino_channel) for the regularizers to
# have a training effect; encoder=dino (no projector) makes straighten/twothirds inert.
# The RUN_* dir names below assume dino_global (projglobal/hw1) for point_maze and
# dino_channel (projchannel/dim8/hw14) for pusht.
PLANNERS="${PLANNERS:-gd cem}"   # planners with configs in conf/plan_*.yaml
GOAL_H="${GOAL_H:-25}"           # keep divisible by frameskip (5)
TRAIN_DECODER="${TRAIN_DECODER:-False}"  # also train the VQVAE decoder (required for planner videos)
FRESH="${FRESH:-0}" # 1 = fresh

CKBPT="./checkpoints"

# ---- per-task setup ----------------------------------------------------------
#  - point_maze: the original experiment (env=point_maze, encoder=dino_global).
#  - pusht:      env=pusht with the SPATIAL channel projector (encoder=dino_channel,
#                14x14x8) -- see the knobs comment for why. Regularizers use the
#                aggcos modes; planning adds objective.alpha=1 (README's PushT note).
#  - wall:       env=wall with the same channel-projector setup (the paper's main
#                wall config); planning follows paper 5.3 -- dset goals + alpha=0
#                (the plan defaults; see the comment in the wall case below).
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
        TRAIN_TASK_OVERRIDES="env=pusht encoder=dino_channel"
        PLAN_TASK_OVERRIDES="objective.alpha=1"
        RUN_FALSE="test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TRUE="test/pusht_${STRAIGHTEN}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/pusht_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_BOTH="test/pusht_${STRAIGHTEN}_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        ;;
    wall)
        TRAIN_TASK_OVERRIDES="env=wall encoder=dino_channel"
        # Paper 5.3: wall start/goal states are sampled from test trajectories so
        # goals are reachable within 25 steps (goal_source='dset'), and only the
        # target IMAGE is used in the objective ("for other environments, we only
        # use target images" -> alpha=0). Both are the plan_gd/plan_cem defaults,
        # so no plan overrides. (DINO-WM's random_state+alpha=1 wall config only
        # worked with its closed-loop MPC, which run.sh does not use.)
        PLAN_TASK_OVERRIDES=""
        RUN_FALSE="test/wall_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TRUE="test/wall_${STRAIGHTEN}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/wall_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_BOTH="test/wall_${STRAIGHTEN}_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        ;;
    *)
        echo "Unknown ENV='$ENV' (choose point_maze, pusht, or wall)" >&2
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
        local cem_chunk_arg=()
        if [ "$planner" = "cem" ]; then
            # num_samples is the paper's 200; CEM_SAMPLE_CHUNK_SIZE rolls the
            # candidates out in chunks so they fit the 12 GB RTX 4070 without
            # changing the result (losses are concatenated before topk).
            num_samples_arg=(planner.sub_planner.num_samples="$NUM_SAMPLES")
            if [ -n "${CEM_SAMPLE_CHUNK_SIZE:-}" ]; then
                cem_chunk_arg=(planner.sub_planner.sample_chunk_size="$CEM_SAMPLE_CHUNK_SIZE")
            fi
        fi
        local chunk_arg=()
        if [ -n "$CHUNK_SIZE" ]; then
            chunk_arg=(chunk_size="$CHUNK_SIZE")
        fi
        # shellcheck disable=SC2086  # PLAN_TASK_OVERRIDES is meant to be word-split
        "$PY" plan.py --config-name "plan_${planner}.yaml" \
            ckpt_base_path="$PWD/$CKBPT/$model_name" model_name="$model_name" \
            hydra.run.dir="plan_outputs_${planner}/$(echo "$model_name" | tr '/' '_')_gH${GOAL_H}" \
            goal_H="$GOAL_H" n_evals="$N_EVALS" $PLAN_TASK_OVERRIDES "${num_samples_arg[@]}" "${cem_chunk_arg[@]}" "${chunk_arg[@]}"
    done
}

# ---- optional clean start ---------------------------------------------------
if [ "$FRESH" = "0" ]; then
    echo ">> FRESH=1: deleting existing run dirs for env=$ENV"
    rm -rf "$CKBPT/$RUN_FALSE" "$CKBPT/$RUN_TRUE" "$CKBPT/$RUN_TWOTHIRDS" "$CKBPT/$RUN_BOTH"
fi

# ---- step 1 & 2: baseline (no regularizers) ---------------------------------
echo "===================== 1) TRAIN baseline (straighten=False) ============="
train False False "$RUN_FALSE"
echo "===================== 2) EVAL baseline model ==========================="
plan_model "$RUN_FALSE"

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

# # ---- step 7 & 8: straightening + two-thirds ----------------------------------
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
