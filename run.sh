#!/usr/bin/env bash
# =============================================================================
# Temporal straightening + two-thirds pipeline (PointMaze umaze | PushT | Wall | Granular | Rope)
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
# ENV=granular / ENV=rope: DINO-WM deformable protocol (env=deformable_env, SoftGym).
#   Training: 1000 x 20-step trajectories, frameskip=1 (DINO-WM App. Table 11) and
#   100 epochs (App. Table 12). num_hist stays at 3 (not DINO-WM's H=1) because the
#   straighten/two-thirds losses need >=3 latent frames per window -- H=1 crashes
#   the curvature loss and makes the two-thirds variance vacuous. Eval follows the
#   paper's MPC + Chamfer-Distance protocol on 10 instances (App. Table 8;
#   open-loop CEM/GD are not reported for rope/granular), so PLANNERS defaults to
#   gd_mpc + mpc_cem. Deformable planning steps the SoftGym sim through pyflex,
#   which is installed into the ts conda env (see setup.sh / ~/PyFleX).
#
# Results are saved under:
#   checkpoints/test/<env>_<straighten>_tt<twothirds>_agg32_.../  training ckpts + config
#   plan_outputs_gd/...  plan_outputs_cem/...   planning logs.json + videos
#   results/                                   aggregated copies of logs.json
#
# Usage:
#   bash run.sh                  # run the full pipeline (resumes existing runs)
#   ENV=wall bash run.sh         # same experiment on Wall (dino_channel + aggcos, dset goals + alpha=0 per paper 5.3)
#   ENV=point_maze_medium bash run.sh  # PointMaze-Medium (dino_global, straighten cos1e-2 = lambda 0.01, encoder_lr 1e-6)
#   ENV=granular bash run.sh     # DINO-WM deformable protocol on Granular (fs=1, 100 epochs; MPC planning needs PyFleX)
#   ENV=rope bash run.sh         # same on Rope
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
export PATH="${PATH:+$PATH:}$HOME/miniconda3/envs/ts/bin"  # patchelf/gcc (mujoco_py one-time cymj build)

# ---- knobs ------------------------------------------------------------------
PY="${PYTHON:-$HOME/miniconda3/envs/ts/bin/python}"
ENV="${ENV:-point_maze_medium}"     # task: point_maze | pusht | wall (default) | granular | rope

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
#  - granular/rope use the DINO-WM deformable protocol: env=deformable_env,
#    encoder=dino_channel, frameskip=1, 100 epochs; num_hist=3 is kept so the
#    regularizers are well-defined. Eval is MPC (gd_mpc/mpc_cem) on 10 instances.
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
    granular|rope)
        # DINO-WM deformable protocol (App. Tables 11/12): 1000 x 20-step
        # trajectories, frameskip=1, 100 epochs, MPC eval on 10 instances.
        # num_hist stays 3 (not DINO-WM's H=1): the straighten/two-thirds losses
        # require >=3 latent frames per window (H=1 crashes total_curvature).
        # H=3 also makes the 14x14 predictor as heavy as wall's -> batch 16
        # (the paper's batch 32 would OOM the 12 GB GPU, same as pusht/wall).
        BATCH_SIZE="${BATCH_SIZE:-16}"  # same 14x14 attention as pusht/wall at H=3: 32 OOMs, 16 fits
        STRAIGHTEN="${STRAIGHTEN:-aggcos1e-1}"
        TWOTHIRDS="${TWOTHIRDS:-aggtwothirds5e-2}"
        EPOCHS="${EPOCHS:-50}"   # DINO-WM App. Table 12 (100 epochs, fs=1)
        N_EVALS="${N_EVALS:-3}"   # fast comparison default (paper evaluates 10; raise for more confidence)
        MPC_MAX_ITER="${MPC_MAX_ITER:-4}"  # 4 MPC iters x n_taken_actions=5 covers goal_H=19;
                                           # more just replans past-horizon windows at ~8s/action sim cost
        NUM_SAMPLES="${NUM_SAMPLES:-300}"  # DINO-WM MPC-CEM population (conf/planner/mpc_cem.yaml)
        CEM_SAMPLE_CHUNK_SIZE="${CEM_SAMPLE_CHUNK_SIZE:-50}"
        CHUNK_SIZE="${CHUNK_SIZE:-1}"  # fs=1 => 19-step WM rollout; GD autograd over the
                                       # full horizon needs ~5 GB/eval (batch 2 OOMs the
                                       # 12 GB GPU), so plan/eval one episode per chunk
        PLANNERS="${PLANNERS:-gd_mpc}"  # mpc_cem is the paper's deformable planner but costs ~5-8h
                                        # on this 12 GB GPU (CEM eval_every=1 runs a real-sim eval per
                                        # opt step); re-enable with PLANNERS="gd_mpc mpc_cem"
        GOAL_H="${GOAL_H:-19}"    # 20-frame episodes: dset goal = final frame of a val trajectory
        ;;
    point_maze_medium)
        # PointMaze-Medium (D4RL maze2d medium), paper-accurate settings:
        # dino_global, straighten cos1e-2 (Table 1 dagger: lambda=0.01 for
        # Medium-global; UMaze/Wall global used 0.1/0.001). encoder_lr: the
        # baseline (no straightening) uses 1e-6, all other variants use 1e-5
        # (Table 3 footnote a), 20 epochs.
        BATCH_SIZE="${BATCH_SIZE:-32}"   # config default 32; fits this GPU with decoder off + dino_global (3-token attention)
        STRAIGHTEN="${STRAIGHTEN:-cos1e-1}"  
        TWOTHIRDS="${TWOTHIRDS:-twothirds5e-2}"  # two-thirds: twothirds5e-2 (cos) or aggtwothirds5e-2 (aggcos)
        EPOCHS="${EPOCHS:-20}"
        N_EVALS="${N_EVALS:-50}"         # eval episodes (config default)
        NUM_SAMPLES="${NUM_SAMPLES:-200}" # CEM candidates per traj (config default)
        CHUNK_SIZE="${CHUNK_SIZE:-}"     # empty = evaluate all n_evals at once (fits for dino_global)
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
    point_maze_medium)
        TRAIN_TASK_OVERRIDES="env=point_maze_medium encoder=dino_global"  # encoder_lr: 1e-6 for baseline, 1e-5 for the rest (Table 3 fn a)
        PLAN_TASK_OVERRIDES=""
        RUN_FALSE="test/medium_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06"
        RUN_TRUE="test/medium_${STRAIGHTEN}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/medium_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        RUN_BOTH="test/medium_${STRAIGHTEN}_tt${TWOTHIRDS}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
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
    granular|rope)
        # DINO-WM deformable protocol: env=deformable_env (SoftGym/Flex wrapper),
        # dino_channel projector so the regularizers act on the aggregation head
        # (same rationale as pusht/wall). Both kwargs.object_name and
        # dataset.object_name are pinned so the training run's saved config (which
        # planning reloads) carries the right object. Eval uses the paper's MPC
        # planners (plan_gd_mpc / plan_mpc_cem); mode=last + alpha=0 keep the
        # objective on the final-frame image only (DINO-WM's C = ||z_T - z_g||^2).
        TRAIN_TASK_OVERRIDES="env=deformable_env env.kwargs.object_name=${ENV} env.dataset.object_name=${ENV} encoder=dino_channel num_hist=3 frameskip=1"
        PLAN_TASK_OVERRIDES="objective.alpha=0 objective.mode=last"
        RUN_FALSE="test/${ENV}_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TRUE="test/${ENV}_${STRAIGHTEN}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_TWOTHIRDS="test/${ENV}_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        RUN_BOTH="test/${ENV}_${STRAIGHTEN}_${TWOTHIRDS}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        ;;
    *)
        echo "Unknown ENV='$ENV' (choose point_maze, pusht, wall, granular, or rope)" >&2
        exit 1
        ;;
esac

# ---- helpers ----------------------------------------------------------------
train() {  # $1 = straighten value, $2 = twothirds value, $3 = run dir name, $4 = encoder_lr (optional)
    echo "==== TRAIN env=$ENV straighten=$1 twothirds=$2 decoder=$TRAIN_DECODER ===="
    # shellcheck disable=SC2086  # TRAIN_TASK_OVERRIDES is meant to be word-split
    local lr_arg=()
    if [ -n "$4" ]; then
        lr_arg=(training.encoder_lr="$4")
    fi
    "$PY" train.py --config-name train.yaml $TRAIN_TASK_OVERRIDES \
        training.straighten="$1" training.twothirds="$2" \
        training.batch_size="$BATCH_SIZE" training.epochs="$EPOCHS" \
        model.train_decoder="$TRAIN_DECODER" has_decoder="$TRAIN_DECODER" \
        "${lr_arg[@]}" \
        hydra.run.dir="$CKBPT/$3"
}

plan_model() {  # $1 = model_name (relative to ckpt_base_path)
    local model_name="$1"
    for planner in $PLANNERS; do
        echo "==== PLAN env=$ENV planner=$planner model=$model_name ===="
        # Deformable planning steps the SoftGym/Flex sim, which needs pyflex.
        if [[ "$ENV" == "granular" || "$ENV" == "rope" ]]; then
            if ! "$PY" -c "import pyflex" >/dev/null 2>&1; then
                echo "!! SKIP $planner planning: pyflex (the SoftGym/Flex backend) is not installed."
                echo "   The deformable sim (env/deformable_env) needs it to step the env."
                echo "   It is installed into the 'ts' conda env; see ~/PyFleX/bindings/"
                echo "   (build: cmake .. + make with CUDA 9.2 cudart + pybind11, then copy"
                echo "   pyflex*.so + libSDL2 into \$CONDA_PREFIX/lib/python3.9/site-packages/)."
                echo "   setup.sh also sets PYFLEXROOT, EGL_GPU=0 and the mujoco paths."
                echo "   Training is unaffected; only planning is skipped."
                continue
            fi
        fi
        # ckpt_base_path must be absolute: hydra chdirs to the run dir, so a
        # relative path would resolve against the wrong directory. The run dir
        # is also overridden to a clean name for readable output folders.
        local num_samples_arg=()
        local cem_chunk_arg=()
        if [ "$planner" = "cem" ] || [ "$planner" = "mpc_cem" ]; then
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
        local mpc_iter_arg=()
        if [ "$planner" = "gd_mpc" ] || [ "$planner" = "mpc_cem" ]; then
            # MPC replans the full remaining horizon every iteration and each
            # eval re-executes the whole accumulated action history in the real
            # sim (~8s/action), so cap iterations to what the horizon needs
            # (default 4 covers goal_H=19 with n_taken_actions=5).
            mpc_iter_arg=(planner.max_iter="${MPC_MAX_ITER:-4}")
        fi
        # shellcheck disable=SC2086  # PLAN_TASK_OVERRIDES is meant to be word-split
        "$PY" plan.py --config-name "plan_${planner}.yaml" \
            ckpt_base_path="$PWD/$CKBPT/$model_name" model_name="$model_name" \
            hydra.run.dir="plan_outputs_${planner}/$(echo "$model_name" | tr '/' '_')_gH${GOAL_H}" \
            goal_H="$GOAL_H" n_evals="$N_EVALS" $PLAN_TASK_OVERRIDES "${num_samples_arg[@]}" "${cem_chunk_arg[@]}" "${mpc_iter_arg[@]}" "${chunk_arg[@]}"
    done
}

# ---- optional clean start ---------------------------------------------------
# FRESH=1 deletes the run dirs first (default FRESH=0 resumes existing runs).
if [ "$FRESH" = "1" ]; then
    echo ">> FRESH=1: deleting existing run dirs for env=$ENV"
    rm -rf "$CKBPT/$RUN_FALSE" "$CKBPT/$RUN_TRUE" "$CKBPT/$RUN_TWOTHIRDS" "$CKBPT/$RUN_BOTH"
fi

# # ---- step 1 & 2: baseline (no regularizers) ---------------------------------
# echo "===================== 1) TRAIN baseline (straighten=False) ============="
# train False False "$RUN_FALSE" 1e-6  # paper Table 3 footnote: baseline (no straightening) uses lr 1e-6; the rest use 1e-5
# echo "===================== 2) EVAL baseline model ==========================="
# plan_model "$RUN_FALSE"

# # ---- step 3 & 4: straightening ----------------------------------------------
# echo "===================== 3) TRAIN (straighten=$STRAIGHTEN) ==============="
# train "$STRAIGHTEN" False "$RUN_TRUE"
# echo "===================== 4) EVAL straightening model ======================"
# plan_model "$RUN_TRUE"

# ---- step 5 & 6: two-thirds regularizer only ---------------------------------
echo "===================== 5) TRAIN (twothirds=$TWOTHIRDS) =================="
train False "$TWOTHIRDS" "$RUN_TWOTHIRDS"
echo "===================== 6) EVAL two-thirds model ========================="
plan_model "$RUN_TWOTHIRDS"

# ---- step 7 & 8: straightening + two-thirds ----------------------------------
echo "===================== 7) TRAIN (straighten=$STRAIGHTEN, twothirds=$TWOTHIRDS) ===="
train "$STRAIGHTEN" "$TWOTHIRDS" "$RUN_BOTH"
echo "===================== 8) EVAL both model ==============================="
plan_model "$RUN_BOTH"

# ---- aggregate results -------------------------------------------------------
echo "===================== Collecting results =============================="
mkdir -p results
for pd in plan_outputs_gd plan_outputs_cem plan_outputs_gd_mpc plan_outputs_mpc_cem; do
    if [ -d "$pd" ]; then
        find "$pd" -name logs.json -print0 2>/dev/null |
            while IFS= read -r -d '' f; do
                cp "$f" "results/$(echo "$f" | tr '/' '_')"
                echo "  copied $f"
            done
    fi
done

echo
echo "Done (env=$ENV)."
echo "  Training checkpoints : $CKBPT/test/${ENV}_*"
echo "  Planning logs/videos : plan_outputs_gd/ , plan_outputs_cem/ , plan_outputs_gd_mpc/ , plan_outputs_mpc_cem/"
echo "  Aggregated logs.json : results/"
