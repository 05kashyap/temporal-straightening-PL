#!/usr/bin/env bash
# =============================================================================
# run_wall_ablation.sh — Wall two-thirds-lambda / straightening-coefficient ablation
#
# Goal: find out whether the two-thirds regularizer can HELP on wall at all.
# The main "both" run (aggcos1e-1 + aggtwothirds5e-2) sits BELOW straightening
# (0.68/0.60 vs 0.90/1.00). A diagnostic showed the fixed λ_tt=0.05 term is
# ~44x the wall prediction loss (raw var(r_t) ~2.0 vs ~0.25 elsewhere), so this
# script sweeps LOWER two-thirds lambdas and HIGHER straightening coefficients.
#
# Each config: train 8 epochs (batch 16, env=wall, encoder=dino_channel,
# decoder off), then plan GD + CEM (paper-5.3 config: dset goals, alpha=0,
# goal_H=25, n_evals=50; CEM num_samples=200 rolled out in chunks of 50).
#
# Config grid (8 runs):
#   straightening sweep  (two-thirds fixed at λ=0.05):  aggcos{1e-1,3e-1,5e-1,1}
#   two-thirds sweep     (straighten fixed at λ=0.1):   aggtwothirds{1e-2,5e-3,1e-3}
#   reference:           straightening only, no two-thirds (aggcos1e-1 + False)
#
# Usage:
#   bash run_wall_ablation.sh             # full grid, train + plan
#   SKIP_PLAN=1 bash run_wall_ablation.sh # training only (faster)
#   EPOCHS=8 BATCH_SIZE=16 bash run_wall_ablation.sh
#
# Runtime estimate: ~8 runs x (8 epochs ~1.7h + GD ~10min + CEM ~40min) ~= 20h
# on the 12 GB GPU, sequential. Training dirs: checkpoints/test/ablation_wall_*;
# plan dirs: plan_outputs_{gd,cem}/ablation_* (kept separate from the main runs).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

source setup.sh                          # exports DATASET_DIR
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco-2.1.2}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
PY="${PYTHON:-$HOME/miniconda3/envs/ts/bin/python}"

CKBPT="./checkpoints"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GOAL_H="${GOAL_H:-25}"
N_EVALS="${N_EVALS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"                # paper's CEM candidates
CEM_SAMPLE_CHUNK_SIZE="${CEM_SAMPLE_CHUNK_SIZE:-50}"  # roll out in chunks of 50
CHUNK_SIZE="${CHUNK_SIZE:-10}"
SKIP_PLAN="${SKIP_PLAN:-0}"

# <straighten> <twothirds>
CONFIGS=(
    # straightening coefficient sweep (two-thirds fixed at λ_tt = 0.05)
    # "aggcos1e-1 aggtwothirds5e-2"   # λ_st = 0.1  (the original "both" config, 8ep)
    # "aggcos3e-1 aggtwothirds5e-2"   # λ_st = 0.3
    # "aggcos5e-1 aggtwothirds5e-2"   # λ_st = 0.5
    # "aggcos1    aggtwothirds5e-2"   # λ_st = 1.0
    # two-thirds lambda sweep (straightening fixed at λ_st = 0.1)
    "aggcos1e-1 aggtwothirds1e-2"   # λ_tt = 0.01
    "aggcos1e-1 aggtwothirds5e-3"   # λ_tt = 0.005
    "aggcos1e-1 aggtwothirds1e-3"   # λ_tt = 0.001
    # reference: straightening only, no two-thirds
    "aggcos1e-1 False"              # straighten-only reference
)

run_name() {  # $1=straighten, $2=twothirds
    echo "test/ablation_wall_${1}_${2}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
}

train() {  # $1=straighten, $2=twothirds, $3=run dir name
    echo "==== TRAIN wall straighten=$1 twothirds=$2 epochs=$EPOCHS ===="
    "$PY" train.py --config-name train.yaml env=wall encoder=dino_channel \
        training.straighten="$1" training.twothirds="$2" \
        training.batch_size="$BATCH_SIZE" training.epochs="$EPOCHS" \
        model.train_decoder=False has_decoder=False \
        hydra.run.dir="$CKBPT/$3"
}

plan_model() {  # $1 = model_name
    local m="$1"
    for planner in gd cem; do
        echo "==== PLAN planner=$planner model=$m ===="
        local sa=()
        if [ "$planner" = "cem" ]; then
            sa=(planner.sub_planner.num_samples="$NUM_SAMPLES"
                planner.sub_planner.sample_chunk_size="$CEM_SAMPLE_CHUNK_SIZE")
        fi
        # paper-5.3 wall config = plan defaults (dset goals, alpha=0); output dir
        # uses an "ablation_" prefix so the main runs are never overwritten.
        "$PY" plan.py --config-name "plan_${planner}.yaml" \
            ckpt_base_path="$PWD/$CKBPT/$m" model_name="$m" \
            hydra.run.dir="plan_outputs_${planner}/ablation_$(echo "$m" | tr '/' '_')_gH${GOAL_H}" \
            goal_H="$GOAL_H" n_evals="$N_EVALS" chunk_size="$CHUNK_SIZE" "${sa[@]}"
    done
}

for cfg in "${CONFIGS[@]}"; do
    ST=$(echo "$cfg" | awk '{print $1}')
    TT=$(echo "$cfg" | awk '{print $2}')
    RUN="$(run_name "$ST" "$TT")"
    echo "==================== CFG straighten=$ST twothirds=$TT ===================="
    train "$ST" "$TT" "$RUN"
    if [ "$SKIP_PLAN" = "1" ]; then
        echo ">> SKIP_PLAN=1: skipping planning for $RUN"
        continue
    fi
    plan_model "$RUN"
done

echo "==================== Ablation summary ===================="
for cfg in "${CONFIGS[@]}"; do
    ST=$(echo "$cfg" | awk '{print $1}')
    TT=$(echo "$cfg" | awk '{print $2}')
    RUN="$(run_name "$ST" "$TT")"
    for planner in gd cem; do
        log="plan_outputs_${planner}/ablation_$(echo "$RUN" | tr '/' '_')_gH${GOAL_H}/logs.json"
        if [ -f "$log" ]; then
            sr=$(grep 'final_eval/success_rate' "$log" | tail -1 | grep -oE '[0-9.]+' | head -1)
            echo "  $RUN [$planner] success=$sr"
        fi
    done
done
echo "Done."
