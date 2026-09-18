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
#             lambda_curv = 0.1. Medium is the one env the paper pools with
#             FLATTEN instead of the aggregation head (encoder.agg_type=flatten).
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
#               train.log, hydra.yaml, rollout_plots/e<n>_rollout/*.png, wandb/}
#               train.py writes all of these relative to its run dir (Hydra chdirs
#               there), so the run dir is the unit of "where are my artifacts".
#   artifacts : $ART_ROOT/<run_name> is a symlink to that run dir (LINK_RUNS=1).
#               Decoded planner VIDEOS come from the PLANNING stage, which should
#               write under $ART_ROOT: nothing in *training* emits videos, only
#               decoded reconstruction PNG grids (rollout_plots/).
#
# RESUMING / EPOCHS
#   train.py auto-resumes from <run_dir>/checkpoints/model_latest.pth (weights,
#   optimizers, epoch, mid-epoch batch), so re-running a job continues it.
#   training.epochs is PER LAUNCH: a re-launch of a finished run would train
#   EPOCHS *more* epochs, hence the default SKIP_FINISHED=1 guard, which skips an
#   arm whose saved epoch >= EPOCHS.
#
# SERVER REQUIREMENTS: the `ts` conda env (environment.yaml), the three datasets,
#   and the pinned DINOv2 weights (torch.hub pulls facebookresearch/dinov2 @
#   b48308a4 once, or copy ~/.cache/torch/hub over). Preflight checks all three.
#
# Knobs (env vars, all optional): EPOCHS=20 BATCH_SIZE= NUM_HIST=3 NUM_WORKERS=
#   REG_WINDOW= FRESH=0 DRY_RUN=0 SKIP_FINISHED=1 LINK_RUNS=1 CKPT_ROOT= ART_ROOT=
#   PYTHON= GPU= (or the 3rd positional) WANDB_MODE=offline
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../train.py" ]]; then
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [[ -f "$SCRIPT_DIR/train.py" ]]; then
    REPO="$SCRIPT_DIR"
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
DATA_DIR_umaze="$REPO/data/datasets/point_maze"          # states.pth, actions.pth, seq_lengths.pth, obses/
DATA_DIR_medium="$REPO/data/datasets/point_maze_medium"  # same layout as umaze
DATA_DIR_pusht="$REPO/data/datasets/pusht_noise"         # train/ and val/

CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints_server}"    # training run dirs (ckpts + logs + recon PNGs)
ART_ROOT="${ART_ROOT:-$REPO/analysis_outputs/server}"  # decoded videos / plan outputs (planning stage)

PY="${PYTHON:-${PY:-$HOME/miniconda3/envs/ts/bin/python}}"  # the ts env python
# ────────────────────────────────────────────────────────────────────────────

ENV_SEL="${1:-}"
DINO="${2:-}"
GPU="${3:-${GPU:-}}"
if [[ -z "$ENV_SEL" || -z "$DINO" ]]; then
    echo "usage: bash run_scripts/train_server.sh <umaze|medium|pusht> <channel|global> [gpu-index]" >&2
    exit 2
fi

EPOCHS="${EPOCHS:-20}"                # per launch (paper: 20 epochs for the mazes)
NUM_HIST="${NUM_HIST:-3}"             # paper Table 3: 3 history frames
NUM_WORKERS="${NUM_WORKERS:-}"        # empty = conf/env/*.yaml (16)
REG_WINDOW="${REG_WINDOW:-}"          # empty = num_hist+num_pred; override the P-Reg stats window
FRESH="${FRESH:-0}"                   # 1 = delete the 4 run dirs before training
DRY_RUN="${DRY_RUN:-0}"               # 1 = print everything, train nothing
SKIP_FINISHED="${SKIP_FINISHED:-1}"   # 1 = skip an arm whose saved epoch >= EPOCHS
LINK_RUNS="${LINK_RUNS:-1}"           # 1 = symlink each run dir into $ART_ROOT
BATCH_OVERRIDE="${BATCH_SIZE:-}"      # empty = the per-recipe default below

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
# vector (global projector; also Medium's flatten pooling). Names follow
# run.sh / run_mpc.sh so the later planning runs find them; each name's lr
# suffix must match the lr passed for that arm. "aggflatten" in a Medium name is
# run.sh's marker for the flatten head (the default head is MLP, so the two can
# never collide).
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
        # Paper B.6 / run.sh: Medium is the flatten-pooled env.
        AGG_OVERRIDE="encoder.agg_type=flatten"
        STRAIGHTEN=aggcos1e-1; TWOTHIRDS=aggtwothirds5e-2; DEF_BATCH=16
        LR_BASE=1e-6; LR_REG=1e-5
        NAME_BASELINE=medium_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
        NAME_STRAIGHTEN=medium_aggflattencos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_PREG=medium_ttaggflattenwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
        NAME_BOTH=medium_aggflattencos1e-1_aggflattenwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
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
        local saved_epoch=""
        saved_epoch="$("$PY" -c 'import sys,torch;ck=torch.load(sys.argv[1],map_location="cpu");print(int(ck.get("epoch") or 0))' "$latest" 2>/dev/null || echo "")"
        if [[ -n "$saved_epoch" && "$saved_epoch" -ge "$EPOCHS" ]]; then
            echo "    already trained: saved epoch $saved_epoch >= EPOCHS=$EPOCHS -- skipping"
            echo "    (EPOCHS=$((EPOCHS + saved_epoch)) to extend it, or FRESH=1 to restart)"
            return 0
        fi
        if [[ -n "$saved_epoch" ]]; then
            echo "    resuming from epoch $saved_epoch (this launch trains $EPOCHS more)"
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
    "${cmd[@]}"
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
