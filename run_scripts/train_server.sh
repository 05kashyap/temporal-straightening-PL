#!/usr/bin/env bash
# =============================================================================
# train_server.sh — one job = (environment, DINOv2 version) x 4 arms, decoder ON.
# TRAINING ONLY: no planning, no eval, no result aggregation.
#
# The four arms (the repo's arm vocabulary, helpers/extract_planner_curves.py):
#   baseline    training.straighten=False        training.twothirds=False
#   straighten  training.straighten=<curvature>  training.twothirds=False
#   p_reg       training.straighten=False        training.twothirds=<P-Reg>
#   both        both regularizers
#
# Usage (from anywhere inside the repo):
#   bash run_scripts/train_server.sh <umaze|medium|pusht> <channel|global> [gpu]
#
#   bash run_scripts/train_server.sh umaze  channel 0      # umaze + channel projector, GPU 0
#   bash run_scripts/train_server.sh medium global 3       # medium + global projector, GPU 3
#   DRY_RUN=1 bash run_scripts/train_server.sh pusht global   # print the 4 argvs, run nothing
#   FRESH=1   bash run_scripts/train_server.sh umaze channel  # wipe those 4 run dirs first
#
# One job = 4 sequential runs; the full grid is {umaze,medium,pusht} x
# {channel,global} = 6 jobs. The optional 3rd argument pins
# CUDA_VISIBLE_DEVICES so several jobs can share a node without all using GPU 0.
#
# GRID MODE -- ALL SIX CONFIGS IN PARALLEL ON ONE GPU:
#   bash run_scripts/train_server.sh all            # or: grid
#   CONFIGS="umaze:global medium:global pusht:global" bash run_scripts/train_server.sh all
#   MAX_PARALLEL=4 STAGGER=60 NUM_WORKERS=4 bash run_scripts/train_server.sh all
#
#   Slurm cannot give one GPU to several ALLOCATIONS (each --gres=gpu:1 job owns
#   its device), so the parallelism lives INSIDE one allocation: `all` launches
#   the configs as concurrent children of this script and waits for them. That is
#   what run_scripts/submit_train_grid.sh submits --
#     sbatch --cpus-per-task=48 --mem=192G --time=48:00:00 \
#            run_scripts/train_server.slurm all all
#   Each child is an ordinary single-process run: Hydra is in RUN mode, so there
#   is no DDP rendezvous and no MASTER_PORT collision (train.py only calls
#   dist.init_process_group for RunMode.MULTIRUN, train.py L41). Children keep
#   their own run dirs, logs and wandb dirs.
#
#   Sizing: a run needs ~20 GB of VRAM, so 6 x 20 GB = 120 GB of an H200's 140 GB
#   (fits with ~15% headroom). Grid mode therefore defaults NUM_WORKERS=4 -- not
#   16, since 6 x 16 = 96 dataloader workers would thrash the node and /scratch --
#   and pins OMP_NUM_THREADS=nproc/MAX_PARALLEL, because torch otherwise lets
#   every process use every core. Request --cpus-per-task >= MAX_PARALLEL*NUM_WORKERS.
#
# GPU-AGNOSTIC: nothing here is tuned to a particular card. Batch size, epochs and
# lrs are the paper's numbers and are NOT scaled by device name, so the same
# command runs on a 4090 / L40S / A100 / H100 / ... unchanged; pass BATCH_SIZE=<n>
# only if a smaller card OOMs. The preflight prints the card it actually sees.
#
# WHAT THE PAPER DOES (Temporal Straightening; Table 1 L711-713, Sec. 5 L519-524,
# B.6 L1389-1396, Table 3 L1192-1203) — the two recipes below are exactly that:
#   channel = the paper's MAIN setup: frozen DINOv2 patch features + a trainable
#             CHANNEL projector -> 14x14x8, curvature on a learnable MLP
#             AGGREGATION HEAD (out dim 128) -> aggcos1e-1 / aggtwothirds5e-2,
#             lambda_curv = 0.1. All three envs use that head: Medium used to be
#             pooled with FLATTEN (paper B.6 [flatten], encoder.agg_type=flatten);
#             those older medium_*aggflatten* runs stay on disk as that ablation
#             arm, while new Medium runs are tagged aggmlp.
#   global  = the paper's "DINOv2(patch)+proj 1x384" row: the projector collapses
#             the patch grid to a single 1x384 vector, and for global features
#             (n_v = 1) the paper "compute[s] the cosine similarity directly
#             between vectors" -> NO aggregation head and no pooling variant, with
#             lambda_curv = 0.01 ("0.1 for agg and 0.01 for the rest", B.6).
#             In this repo that is the plain (non-agg) mode: cos1e-2 / ttwothirds5e-2.
#   Paper Table 3: batch 32, history frames 3, frameskip 5, 20 epochs (mazes),
#   projector lr 1e-5 and 1e-6 when training WITHOUT straightening (footnote a).
#   The decoder is trained jointly "solely for interpretability purposes" (L695,
#   detached via stop-gradient), which is why every arm here has it ON
#   (model.train_decoder=True has_decoder=True => decoder weights land in the ckpt).
#
# WHERE THINGS ARE WRITTEN
#   training  : $CKPT_ROOT/test/<run_name>/{checkpoints/model_{n,latest}.pth,
#               hydra.yaml, rollout_plots/e<n>_rollout/*.png, wandb/}
#               train.py writes all of these relative to its run dir (Hydra chdirs
#               there), so the run dir is the unit of "where are my artifacts".
#   logs      : $CKPT_ROOT/logs/<run_name>.log -- each arm's own stdout+stderr, via
#               `tee -a` (append, so a resumed arm keeps writing to the same file).
#               train.py itself only logs to the console; run dirs written by older
#               launchers may additionally contain a train.log of their own.
#   artifacts : $ART_ROOT/<run_name> is a symlink to that run dir (LINK_RUNS=1).
#               Decoded planner VIDEOS come from the PLANNING stage, which should
#               write under $ART_ROOT: nothing in *training* emits videos, only
#               decoded reconstruction PNG grids (rollout_plots/).
#
# RESUMING / EPOCHS
#   train.py auto-resumes from <run_dir>/checkpoints/model_latest.pth (weights,
#   optimizers, epoch, mid-epoch batch), so re-running a job continues it.
#   training.epochs is the TARGET TOTAL (training.epochs_mode=target, the default in
#   conf/train.yaml): an arm saved at epoch 12 with EPOCHS=20 trains 13..20 and stops.
#   It never overshoots; raise EPOCHS to train an arm further (EPOCHS=25 -> 21..25), or set
#   training.epochs_mode=additional for the legacy behaviour of adding EPOCHS per launch.
#   SKIP_FINISHED=1 (default, target mode only) skips an arm that already reached EPOCHS.
#   STATUS=1 lists what each arm still needs and trains nothing; grid mode probes with it
#   to skip fully-finished configs (sugar: bash run_scripts/train_server.sh status).
#
# SERVER REQUIREMENTS: the `ts` conda env (environment.yaml), the three datasets,
#   and the pinned DINOv2 weights (torch.hub pulls facebookresearch/dinov2 @
#   b48308a4 once, or copy ~/.cache/torch/hub over). Preflight checks all three.
#
# Knobs (env vars, all optional): EPOCHS=20 BATCH_SIZE= NUM_HIST=3 NUM_WORKERS=
#   REG_WINDOW= FRESH=0 DRY_RUN=0 SKIP_FINISHED=1 LINK_RUNS=1 CKPT_ROOT= ART_ROOT=
#   PYTHON= GPU= (or the 3rd positional) WANDB_MODE=offline EPOCHS_MODE=target STATUS=0
# Grid-mode only: CONFIGS="<env>:<dino> ..." MAX_PARALLEL= STAGGER=30 OMP_NUM_THREADS=
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../train.py" ]]; then
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/train.py" ]]; then
    REPO="$SCRIPT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/train.py" ]]; then
    REPO="$SLURM_SUBMIT_DIR"     # sbatch runs a spool copy; submit from the repo
elif [[ -f "$PWD/train.py" ]]; then
    REPO="$PWD"
else
    echo "train_server.sh: cannot find train.py -- run this from inside the repo" >&2
    exit 2
fi
cd "$REPO"

# Optional: this repo's env exports (DATASET_DIR, Mujoco paths, ts-env PATH).
# Missing setup.sh on a fresh server checkout is fine -- everything this script
# needs is in the EDIT-ME block below.
for _setup in "$REPO/run_scripts/setup.sh" "$REPO/setup.sh"; do
    if [[ -f "$_setup" ]]; then
        # shellcheck source=/dev/null
        source "$_setup"
        break
    fi
done



# ─── EDIT ME ────────────────────────────────────────────────────────────────
# The only paths you should need to touch on a new machine. (Each can also be
# overridden from the environment, but editing the strings here is enough.)
# The data layout lives in run_scripts/dataset_paths.sh so training, planning and
# the smoke test cannot disagree about it (it also provides data_dir_for /
# dataset_dir_for, which PLANNING needs because the three datasets are nested
# differently: $DATA_ROOT/point_maze/point_maze vs $DATA_ROOT/point_maze_medium).
# Override SCRATCH / DATA_ROOT / CKPT_ROOT / ART_ROOT from the environment.
source "$SCRIPT_DIR/dataset_paths.sh"

DATA_DIR_umaze="$(data_dir_for umaze)"      # states.pth, actions.pth, seq_lengths.pth, obses/
DATA_DIR_medium="$(data_dir_for medium)"    # same layout as umaze
DATA_DIR_pusht="$(data_dir_for pusht)"      # train/ and val/
# CKPT_ROOT (training run dirs) and ART_ROOT (plan outputs) come from the helper.
# CKPT_ROOT keeps the literal "checkpoints/" substring on purpose: train.py derives
# the wandb run name as saved_folder.split("checkpoints/")[-1] (train.py L38). The
# nesting also means `checkpoints/` (gitignored) keeps server runs out of git, and
# $REPO/checkpoints/test/... (laptop runs) can never be resumed by accident.

# Python for this repo. Priority: explicit $PYTHON > a working $PY (run_scripts/
# setup.sh exports this laptop's env path) > the activated conda env
# (`conda activate ts` -- the usual thing on a server) > the ts env INSIDE the
# server container (/opt/miniconda, see SERVER_CONTEXT.md) > this laptop's
# miniconda > whatever `python` resolves to. The preflight below fails loudly,
# naming the exact path, if the chosen interpreter has no torch.
#
# NOTE for the server: the ts env lives in the container overlay, so a launcher
# started on the HOST (outside apptainer) falls through to /usr/bin/python, which
# has no torch. Submit the grid instead -- run_scripts/submit_train_grid.sh wraps
# everything in the container -- or enter the container and `conda activate ts`
# first. Grid mode fails fast with that message; PYTHON=/path/to/python overrides.
if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"                                           # explicit override wins
elif [[ -z "${PY:-}" || ! -x "${PY:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PY="$CONDA_PREFIX/bin/python"                      # activated env
    elif [[ -x "/opt/miniconda/envs/ts/bin/python" ]]; then
        PY="/opt/miniconda/envs/ts/bin/python"             # ts env in the container overlay
    elif [[ -x "$HOME/miniconda3/envs/ts/bin/python" ]]; then
        PY="$HOME/miniconda3/envs/ts/bin/python"           # dev-laptop default
    else
        PY="$(command -v python || echo python)"           # PATH (on a server host: /usr/bin/python)
    fi
fi
# ────────────────────────────────────────────────────────────────────────────

ENV_SEL="${1:-}"
# `status` sugar: report every arm resume state and exit (same as STATUS=1).
if [[ "$ENV_SEL" == "status" ]]; then STATUS=1; ENV_SEL=all; fi
DINO="${2:-}"
GPU="${3:-${GPU:-}}"
if [[ -z "$ENV_SEL" || ( -z "$DINO" && "$ENV_SEL" != all && "$ENV_SEL" != grid ) ]]; then
    echo "usage: bash run_scripts/train_server.sh <umaze|medium|pusht> <channel|global> [gpu-index]" >&2
    echo "       bash run_scripts/train_server.sh <all|grid>   # every config on ONE GPU, in parallel" >&2
    echo "       bash run_scripts/train_server.sh status       # which arms still need training (no GPU work)" >&2
    exit 2
fi

EPOCHS="${EPOCHS:-20}"                # target total epochs per arm (paper: 20 for the mazes)
# Epoch semantics, read out of the config so the launcher skip logic matches train.py:
#   target     (default) -- EPOCHS is the total number of epochs the arm should reach.
#   additional           -- legacy: every launch trains EPOCHS more epochs.
EPOCHS_MODE="${EPOCHS_MODE:-$(grep -m1 -oE '^[[:space:]]*epochs_mode:[[:space:]]*[A-Za-z]+' "$REPO/conf/train.yaml" 2>/dev/null | awk -F: '{print $2}' | tr -d '[:space:]' || true)}"
EPOCHS_MODE="${EPOCHS_MODE:-target}"
case "$EPOCHS_MODE" in
    target|additional) ;;
    *) echo "EPOCHS_MODE must be target or additional (got '$EPOCHS_MODE')" >&2; exit 2 ;;
esac
STATUS="${STATUS:-0}"                 # 1 = report per-arm resume state and exit
NUM_HIST="${NUM_HIST:-3}"             # paper Table 3: 3 history frames
NUM_WORKERS="${NUM_WORKERS:-}"        # empty = conf/env/*.yaml (16)
REG_WINDOW="${REG_WINDOW:-}"          # empty = num_hist+num_pred; override the P-Reg stats window
FRESH="${FRESH:-0}"                   # 1 = delete the 4 run dirs before training
DRY_RUN="${DRY_RUN:-0}"               # 1 = print everything, train nothing
SKIP_FINISHED="${SKIP_FINISHED:-1}"   # 1 = skip an arm that already reached EPOCHS (target mode)
LINK_RUNS="${LINK_RUNS:-1}"           # 1 = symlink each run dir into $ART_ROOT
BATCH_OVERRIDE="${BATCH_SIZE:-}"      # empty = the per-recipe default below

# ─── grid mode: every config on the ONE visible GPU ─────────────────────────
# `all` / `grid` re-invokes this script once per (env, dino) pair as background
# children (a wave at a time) and waits for them. Slurm hands one GPU to one
# ALLOCATION, so co-scheduling lives here, not in the submission script.
if [[ "$ENV_SEL" == "all" || "$ENV_SEL" == "grid" ]]; then
    SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
    CONFIGS="${CONFIGS:-umaze:channel umaze:global medium:channel medium:global pusht:channel pusht:global}"
    read -r -a _cfgs <<< "$CONFIGS"
    if [[ ${#_cfgs[@]} -eq 0 ]]; then
        echo "CONFIGS is empty -- nothing to launch" >&2; exit 2
    fi
    MAX_PARALLEL="${MAX_PARALLEL:-${#_cfgs[@]}}"
    STAGGER="${STAGGER:-30}"
    if [[ "$DRY_RUN" = "1" ]]; then STAGGER=0; fi       # a dry run should not sleep
    NUM_WORKERS="${NUM_WORKERS:-4}"                     # 6 x 16 workers would thrash node + /scratch
    if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
        OMP_NUM_THREADS=$(( ($(nproc) + MAX_PARALLEL - 1) / MAX_PARALLEL ))
    fi
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
    export OMP_NUM_THREADS MKL_NUM_THREADS

    echo "=========================================================================="
    echo "train_server.sh GRID: ${#_cfgs[@]} configs, waves of $MAX_PARALLEL, ONE GPU"
    echo "  configs : $CONFIGS"
    echo "  device  : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all visible>}   nproc=$(nproc)   cores/child=$OMP_NUM_THREADS"
    echo "  per job : NUM_WORKERS=$NUM_WORKERS EPOCHS=$EPOCHS DRY_RUN=$DRY_RUN staggered ${STAGGER}s"
    echo "  logs    : $CKPT_ROOT/logs/<env>_<dino>.grid.log  (+ per-arm <run_name>.log)"
    echo "=========================================================================="

    # Fail fast on the classic server mistake: running the launcher on the HOST,
    # outside the container, where `python` is /usr/bin/python and has no torch.
    # Without this, six children start and die one by one with the same message.
    if ! _torch_err="$("$PY" -c 'import torch' 2>&1)"; then
        echo "FATAL: '$PY' cannot import torch:" >&2
        printf '%s\n' "$_torch_err" | tail -6 >&2
        if [[ "$DRY_RUN" = "1" ]]; then
            echo "  WARNING : DRY_RUN still prints argvs, a real run will fail." >&2
        else
            echo "       On the server the ts env lives INSIDE the project container. Either submit the" >&2
            echo "       grid (it wraps everything in apptainer):" >&2
            echo "         bash run_scripts/submit_train_grid.sh" >&2
            echo "         sbatch run_scripts/train_server.slurm all all" >&2
            echo "       or enter the container and 'conda activate ts' first (SERVER_CONTEXT.md 4/8)." >&2
            echo "       If you sourced an env file that prepends PYTHONPATH or LD_LIBRARY_PATH (MuJoCo /" >&2
            echo "       planning setup), retry without it: a shadowed or mismatched library breaks torch." >&2
            echo "       Override the interpreter with PYTHON=/path/to/ts/env/bin/python." >&2
            exit 1
        fi
    fi

    # Children are new processes and this script exports only a few vars, so pass
    # every knob explicitly: an un-exported EPOCHS/NUM_WORKERS would be invisible.
    child_env=("TRAIN_SERVER_GRID=1")
    for _v in PY PYTHON CKPT_ROOT ART_ROOT EPOCHS NUM_HIST NUM_WORKERS REG_WINDOW FRESH \
              DRY_RUN SKIP_FINISHED LINK_RUNS WANDB_MODE DATASET_DIR TORCH_HOME \
              PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS MKL_NUM_THREADS GPU EPOCHS_MODE; do
        if [[ -n "${!_v:-}" ]]; then child_env+=("$_v=${!_v}"); fi
    done
    if [[ -n "$BATCH_OVERRIDE" ]]; then child_env+=("BATCH_SIZE=$BATCH_OVERRIDE"); fi

    # STATUS mode (grid): report every config, then STOP -- never launch a child.
    if [[ "$STATUS" = "1" ]]; then
        for cfg in "${_cfgs[@]}"; do
            _e="${cfg%%:*}"; _d="${cfg##*:}"
            echo "== ${_e}/${_d}"
            env "${child_env[@]}" STATUS=1 bash "$SELF" "$_e" "$_d" || true
        done
        exit 0
    fi

    mkdir -p "$CKPT_ROOT/logs"
    rc_total=0
    _w=0
    while (( _w < ${#_cfgs[@]} )); do
        _wave=( "${_cfgs[@]:_w:MAX_PARALLEL}" )
        pids=(); labels=(); logs=()
        for cfg in "${_wave[@]}"; do
            _e="${cfg%%:*}"; _d="${cfg##*:}"
            _log="$CKPT_ROOT/logs/${_e}_${_d}.grid.log"
            # Probe this config first: the STATUS report is cheap (it only reads
            # checkpoints) and lets us skip configs whose four arms are all at target.
            _st="$(env "${child_env[@]}" STATUS=1 bash "$SELF" "$_e" "$_d" 2>&1 || true)"
            if [[ "$_st" == *STATUS_ALL_DONE=1* ]]; then
                echo "  [skip]   ${_e}/${_d}  -- all 4 arms already at target $EPOCHS"
                continue
            fi
            printf '%s\n' "$_st" | sed 's/^/      /'
            echo "  start    ${_e}/${_d}  ->  $_log"
            env "${child_env[@]}" STATUS=0 bash "$SELF" "$_e" "$_d" "$GPU" >"$_log" 2>&1 &
            pids+=("$!"); labels+=("$_e/$_d"); logs+=("$_log")
            if [[ "$STAGGER" -gt 0 ]]; then sleep "$STAGGER"; fi
        done
        _i=0
        for _pid in "${pids[@]}"; do
            if wait "$_pid"; then
                echo "  [done]   ${labels[$_i]}"
            else
                _rc=$?
                echo "  [FAILED] ${labels[$_i]} (exit $_rc) -- see ${logs[$_i]}" >&2
                rc_total=1
            fi
            _i=$(( _i + 1 ))
        done
        _w=$(( _w + MAX_PARALLEL ))
    done

    echo
    echo "=========================================================================="
    echo "grid done at $(date '+%F %H:%M:%S')  configs=${#_cfgs[@]}  waves=$(( (${#_cfgs[@]} + MAX_PARALLEL - 1) / MAX_PARALLEL ))"
    echo "  grid logs : $CKPT_ROOT/logs/<env>_<dino>.grid.log"
    echo "  arm logs  : $CKPT_ROOT/logs/<run_name>.log   (4 arms per config, sequential)"
    if [[ "$rc_total" != "0" ]]; then
        echo "at least one config failed -- read its .grid.log for the failing arm" >&2
        exit 1
    fi
    exit 0
fi

case "$ENV_SEL" in
    umaze)  TRAIN_ENV=point_maze;        DATA_PATH="$DATA_DIR_umaze";  CHECK_DATA=(states.pth actions.pth obses) ;;
    medium) TRAIN_ENV=point_maze_medium; DATA_PATH="$DATA_DIR_medium"; CHECK_DATA=(states.pth actions.pth obses) ;;
    pusht)  TRAIN_ENV=pusht;             DATA_PATH="$DATA_DIR_pusht";  CHECK_DATA=(train val) ;;
    *) echo "unknown env '$ENV_SEL' (choose umaze, medium or pusht)" >&2; exit 2 ;;
esac
case "$DINO" in
    channel|global) ENCODER="dino_$DINO" ;;
    *) echo "unknown dino '$DINO' (choose channel or global)" >&2; exit 2 ;;
esac

# ─── recipe + run-dir names per (env, dino) ─────────────────────────────────
# Loss strings are the repo's own: aggcos/aggtwothirds = regularizers on the
# aggregation head (channel projector), cos/twothirds = directly on the feature
# vector (global projector). Names follow run.sh / run_mpc.sh so the later
# planning runs find them; each name's lr suffix must match the lr passed for
# that arm. "aggmlp" in a Medium name tags the learned aggregation head (the
# yaml default), and "aggflatten" tags the older flatten-pooled runs (paper B.6
# [flatten], kept as that ablation arm) -- so neither recipe can ever resume the
# other head's run dir.
case "$ENV_SEL:$DINO" in
    umaze:channel)
        AGG_OVERRIDE="encoder.agg_type=mlp"
        STRAIGHTEN=aggcos1e-1; TWOTHIRDS=aggtwothirds5e-2; DEF_BATCH=16
        LR_BASE=1e-6; LR_REG=1e-5
        NAME_BASELINE=umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
        NAME_STRAIGHTEN=umaze_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_PREG=umaze_ttaggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_BOTH=umaze_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        ;;
    medium:channel)
        # The same learned aggregation head as umaze/pusht (the encoder yaml default):
        # the 196x8 channel tokens give agg_mlp_in_dim = 196*8 = 1568, exactly as in
        # umaze. Medium's older runs pooled with FLATTEN instead (paper B.6 [flatten]);
        # they stay on disk under the aggflatten tag, and the new head-MLP runs are
        # tagged aggmlp so the two recipes can never resume each other's ckpt. The
        # baseline has no loss string to tag, hence its explicit aggmlp marker.
        AGG_OVERRIDE="encoder.agg_type=mlp"
        STRAIGHTEN=aggcos1e-1; TWOTHIRDS=aggtwothirds5e-2; DEF_BATCH=16
        LR_BASE=1e-6; LR_REG=1e-5
        NAME_BASELINE=medium_False_aggmlp_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
        NAME_STRAIGHTEN=medium_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_PREG=medium_ttaggmlpwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_BOTH=medium_aggmlpcos1e-1_aggmlpwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        ;;
    pusht:channel)
        # run.sh trains pusht's baseline at 1e-5 (kept here so an existing
        # pusht_False_..._lr1e-05 run is resumed instead of forked).
        AGG_OVERRIDE="encoder.agg_type=mlp"
        STRAIGHTEN=aggcos1e-1; TWOTHIRDS=aggtwothirds5e-2; DEF_BATCH=16
        LR_BASE=1e-5; LR_REG=1e-5
        NAME_BASELINE=pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_STRAIGHTEN=pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_PREG=pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_BOTH=pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        ;;
    umaze:global|medium:global|pusht:global)
        # Paper Table 1 "1x384" row: the cosine is computed directly between the
        # global vectors, so there is no aggregation head to configure here. Do NOT
        # add encoder.agg_type=mlp: that head is built for 196x384 inputs, and it is
        # called on saved checkpoints by curvature_analysis.py:169,
        # analysis/curvature_distributions.py:49 and analysis/linear_probe.py:144 --
        # it would crash on the 1x384 global token. The yaml default (flatten) is
        # both the paper's behaviour and what those analysis scripts expect.
        AGG_OVERRIDE=""
        STRAIGHTEN=cos1e-2; TWOTHIRDS=ttwothirds5e-2; DEF_BATCH=32
        LR_BASE=1e-6; LR_REG=1e-5
        NAME_BASELINE=${ENV_SEL}_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06
        NAME_STRAIGHTEN=${ENV_SEL}_cos1e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05
        NAME_PREG=${ENV_SEL}_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05
        NAME_BOTH=${ENV_SEL}_cos1e-2_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05
        ;;
    *)
        echo "unsupported combination '$ENV_SEL:$DINO'" >&2
        exit 2
        ;;
esac
# ─── STATUS: what each arm still needs (no training, no dataset/GPU preflight) ─
# Reads only <run_dir>/checkpoints/model_latest.pth, so grid mode can probe every
# config cheaply before deciding what to launch.
if [[ "$STATUS" = "1" ]]; then
    echo "=== $ENV_SEL:$DINO   target EPOCHS=$EPOCHS   epochs_mode=$EPOCHS_MODE"
    printf '    %-68s %6s %8s  %s\n' "arm (run_name)" "saved" "target" "state"
    _done=0; _res=0; _fresh=0
    for _n in "$NAME_BASELINE" "$NAME_STRAIGHTEN" "$NAME_PREG" "$NAME_BOTH"; do
        _ck="$CKPT_ROOT/test/$_n/checkpoints/model_latest.pth"
        if [[ ! -f "$_ck" ]]; then
            printf '    %-68s %6s %8s  %s\n' "$_n" "-" "$EPOCHS" "fresh (no ckpt)"
            _fresh=$((_fresh + 1)); continue
        fi
        _ss="$("$PY" -c 'import sys,torch;ck=torch.load(sys.argv[1],map_location="cpu");print(int(ck.get("epoch") or 0), int(ck.get("current_iter") or 0))' "$_ck" 2>/dev/null || true)"
        read -r _se _si <<< "$_ss"
        if [[ ! "${_se:-}" =~ ^[0-9]+$ ]]; then
            printf '    %-68s %6s %8s  %s\n' "$_n" "?" "$EPOCHS" "unreadable ckpt"
            _res=$((_res + 1)); continue
        fi
        _si="${_si:-0}"
        if [[ "$EPOCHS_MODE" = "target" && "$_se" -ge "$EPOCHS" && "$_si" -eq 0 ]]; then
            printf '    %-68s %6s %8s  %s\n' "$_n" "$_se" "$EPOCHS" "done"
            _done=$((_done + 1))
        else
            _first="$(( _si > 0 ? _se : _se + 1 ))"; if [[ "$_first" -lt 1 ]]; then _first=1; fi
            if [[ "$EPOCHS_MODE" = "target" ]]; then _last="$EPOCHS"; else _last=$(( _first + EPOCHS - 1 )); fi
            printf '    %-68s %6s %8s  %s\n' "$_n" "$_se" "$EPOCHS" "resume -> trains $_first..$_last"
            _res=$((_res + 1))
        fi
    done
    echo "    summary: $_done done, $_res to resume, $_fresh fresh (of 4 arms)"
    if [[ "$_done" -eq 4 ]]; then echo "STATUS_ALL_DONE=1"; fi
    exit 0
fi

BATCH_SIZE="${BATCH_OVERRIDE:-$DEF_BATCH}"

# ─── preflight ──────────────────────────────────────────────────────────────
problems=()
notes=()

if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    notes+=("CUDA_VISIBLE_DEVICES=$GPU")
fi

if [[ ! -x "$PY" ]]; then
    problems+=("python not executable: $PY   (edit PY, or export PYTHON=/path/to/python)")
else
    if ! py_info="$("$PY" -c 'import torch,sys;print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))' 2>&1)"; then
        problems+=("$PY cannot import torch: $py_info")
        problems+=("  -> most likely you are running OUTSIDE the project container: the ts env lives in"
                    "     the overlay at /opt/miniconda. Use run_scripts/submit_train_grid.sh (one job,"
                    "     all configs on one GPU) or enter the container and 'conda activate ts' first."
                    "     SERVER_CONTEXT.md sections 4 and 8; PYTHON=/path/to/python overrides this.")
    else
        notes+=("python/torch: $py_info")
    fi
fi

if [[ ! -d "$DATA_PATH" ]]; then
    problems+=("dataset dir missing: $DATA_PATH   (edit DATA_DIR_${ENV_SEL} in the EDIT-ME block)")
else
    for _f in "${CHECK_DATA[@]}"; do
        if [[ ! -e "$DATA_PATH/$_f" ]]; then
            problems+=("dataset incomplete: $DATA_PATH/$_f is missing")
        fi
    done
    notes+=("dataset: $DATA_PATH")
fi

TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
if ! compgen -G "$TORCH_HUB_DIR/facebookresearch_dinov2_*" >/dev/null; then
    notes+=("DINOv2 hub checkout NOT cached under $TORCH_HUB_DIR -- the first run needs internet (torch.hub) or a copied cache")
fi
if [[ ! -f "$TORCH_HUB_DIR/checkpoints/dinov2_vits14_pretrain.pth" ]]; then
    notes+=("DINOv2 weights NOT cached at $TORCH_HUB_DIR/checkpoints/dinov2_vits14_pretrain.pth -- same as above")
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET_DIR="${DATASET_DIR:-$(dirname "$DATA_PATH")}"
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

if [[ ${#notes[@]} -gt 0 ]]; then
    printf '  %s\n' "${notes[@]}"
fi
if [[ ${#problems[@]} -gt 0 ]]; then
    echo "PREFLIGHT FAILURES:" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    if [[ "$DRY_RUN" = "1" ]]; then
        echo "  (DRY_RUN=1: continuing anyway -- these abort a real run)" >&2
    else
        exit 1
    fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$CKPT_ROOT" "$ART_ROOT"
fi

# ─── job header ─────────────────────────────────────────────────────────────
ARM_TOTAL=4
ARM_I=0
echo "=========================================================================="
echo "train_server.sh  env=$ENV_SEL ($TRAIN_ENV)  dino=$DINO ($ENCODER)  decoder=on"
echo "  arms      : baseline, straighten, p_reg, both"
if [[ -n "$AGG_OVERRIDE" ]]; then
    echo "  agg head  : $AGG_OVERRIDE"
else
    echo "  agg head  : none (losses are global-vector mode; yaml agg_type is unused)"
fi
echo "  regularize: straighten=$STRAIGHTEN  twothirds=$TWOTHIRDS"
echo "  optim     : batch=$BATCH_SIZE epochs=$EPOCHS (per launch) num_hist=$NUM_HIST lr baseline=$LR_BASE regularized=$LR_REG"
echo "  data      : $DATA_PATH"
echo "  ckpts     : $CKPT_ROOT/test/<run_name>"
echo "  artifacts : $ART_ROOT (symlinks; decoded planner videos belong here later)"
echo "  run dirs  :"
for _n in "$NAME_BASELINE" "$NAME_STRAIGHTEN" "$NAME_PREG" "$NAME_BOTH"; do
    if [[ -f "$CKPT_ROOT/test/$_n/checkpoints/model_latest.pth" ]]; then
        printf '    %s  [resume]\n' "$_n"
    else
        printf '    %s  [fresh]\n' "$_n"
    fi
done

# ─── the four arms ──────────────────────────────────────────────────────────
train_arm() {  # $1 label, $2 straighten, $3 twothirds, $4 run name, $5 encoder_lr
    local label="$1" st="$2" tt="$3" name="$4" lr="$5"
    ARM_I=$((ARM_I + 1))
    local run_dir="$CKPT_ROOT/test/$name"
    local latest="$run_dir/checkpoints/model_latest.pth"

    echo
    echo "=== [$ARM_I/$ARM_TOTAL] $label -> $run_dir"

    if [[ "$FRESH" = "1" && -d "$run_dir" ]]; then
        if [[ "$DRY_RUN" = "1" ]]; then
            echo "    [dry-run] would rm -rf $run_dir"
        else
            echo "    FRESH=1: removing $run_dir"
            rm -rf "$run_dir"
        fi
    fi

    if [[ "$SKIP_FINISHED" = "1" && "$FRESH" != "1" && -f "$latest" ]]; then
        local saved_state="" saved_epoch="" saved_iter="0"
        saved_state="$("$PY" -c 'import sys,torch;ck=torch.load(sys.argv[1],map_location="cpu");print(int(ck.get("epoch") or 0), int(ck.get("current_iter") or 0))' "$latest" 2>/dev/null || true)"
        read -r saved_epoch saved_iter <<< "$saved_state"
        saved_iter="${saved_iter:-0}"
        # target mode: finished only when the target epoch is complete (current_iter == 0).
        # additional mode never skips -- each launch is meant to add EPOCHS more.
        if [[ "$EPOCHS_MODE" = "target" && "${saved_epoch:-}" =~ ^[0-9]+$ && "$saved_epoch" -ge "$EPOCHS" && "$saved_iter" -eq 0 ]]; then
            echo "    already at target: saved epoch $saved_epoch >= EPOCHS=$EPOCHS -- skipping"
            echo "    (raise EPOCHS to train it further, or FRESH=1 to restart from scratch)"
            return 0
        fi
        if [[ "${saved_epoch:-}" =~ ^[0-9]+$ ]]; then
            echo "    resuming: saved epoch $saved_epoch (batch $saved_iter into it) -> target $EPOCHS [$EPOCHS_MODE mode]"
        fi
    fi

    local cmd=( "$PY" -u train.py --config-name train.yaml
                "env=$TRAIN_ENV" "encoder=$ENCODER" )
    if [[ -n "$AGG_OVERRIDE" ]]; then
        cmd+=( "$AGG_OVERRIDE" )
    fi
    cmd+=( "training.straighten=$st"
           "training.twothirds=$tt"
           "training.batch_size=$BATCH_SIZE"
           "training.epochs=$EPOCHS"
           "training.encoder_lr=$lr"
           "num_hist=$NUM_HIST"
           "model.train_decoder=True"
           "has_decoder=True"
           "env.dataset.data_path=$DATA_PATH"
           "hydra.run.dir=$run_dir" )
    if [[ -n "$NUM_WORKERS" ]]; then
        cmd+=( "env.num_workers=$NUM_WORKERS" )
    fi
    if [[ -n "$REG_WINDOW" ]]; then
        cmd+=( "reg_window=$REG_WINDOW" )
    fi

    printf '    $'; printf ' %q' "${cmd[@]}"; printf '\n'
    if [[ "$DRY_RUN" = "1" ]]; then
        return 0
    fi
    mkdir -p "$CKPT_ROOT/logs"
    "${cmd[@]}" 2>&1 | tee -a "$CKPT_ROOT/logs/$name.log"
    if [[ "$LINK_RUNS" = "1" && -d "$run_dir" ]]; then
        ln -sfn "$run_dir" "$ART_ROOT/$name"
    fi
}

train_arm "baseline"   False         False        "$NAME_BASELINE"   "$LR_BASE"
train_arm "straighten" "$STRAIGHTEN" False        "$NAME_STRAIGHTEN" "$LR_REG"
train_arm "p_reg"      False         "$TWOTHIRDS" "$NAME_PREG"       "$LR_REG"
train_arm "both"       "$STRAIGHTEN" "$TWOTHIRDS" "$NAME_BOTH"       "$LR_REG"

if [[ "$DRY_RUN" = "1" ]]; then
    echo
    echo "DRY_RUN=1: printed $ARM_TOTAL argvs, trained nothing."
    exit 0
fi

# ─── summary ────────────────────────────────────────────────────────────────
echo
echo "=========================================================================="
echo "done at $(date '+%F %H:%M:%S')  env=$ENV_SEL dino=$DINO"
_ok=0
for _n in "$NAME_BASELINE" "$NAME_STRAIGHTEN" "$NAME_PREG" "$NAME_BOTH"; do
    if [[ -f "$CKPT_ROOT/test/$_n/checkpoints/model_latest.pth" ]]; then
        echo "  [ok]      $_n"
        _ok=$((_ok + 1))
    else
        echo "  [MISSING] $_n  ($CKPT_ROOT/test/$_n/checkpoints/model_latest.pth)"
    fi
done
echo "  $_ok/$ARM_TOTAL runs have a ckpt; decoded recon PNGs: <run_dir>/rollout_plots/"
if [[ "$LINK_RUNS" = "1" ]]; then
    echo "  artifact symlinks: $ART_ROOT/<run_name>"
fi
