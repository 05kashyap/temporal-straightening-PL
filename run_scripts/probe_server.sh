#!/bin/bash
# =============================================================================
# probe_server.sh -- sbatch-able wrapper that runs BOTH linear probes for every
#                    checkpoint group on one GPU node.
#
#   sbatch run_scripts/probe_server.sh                         # all 3 envs, both methods
#   sbatch run_scripts/probe_server.sh --dry-run               # print the commands only
#   ENVS=point_maze_medium RECIPES=aggmlp sbatch ...           # restrict the groups
#   ONLY=point_maze_medium:aggmlp sbatch ...                   # a single group
#   METHODS=rollout sbatch ...                                 # one method
#   LIMIT_EPISODES=2 TMAX=8 sbatch ...                         # quick smoke (rollout only)
#   OVERLAY_RW=1 sbatch ...                                    # writable overlay (see below)
#
# Inside the project container (so the ts env, the DINOv2 cache and DATA_ROOT match
# training/planning) this runs:
#
#   python analysis/run_probes.py --ckpt-root $CKPT_ROOT/test --data-root $DATA_ROOT \
#        --outdir $OUTDIR --envs $ENVS [--recipes ...] [--only ...] [--methods ...]
#
# run_probes.py discovers every (env, recipe) group (recipe from each run dir's own
# hydra.yaml: encoder.projector + encoder.agg_type -> global / flatten / aggmlp), runs
# the grounded (--feature-source encoded) and ungrounded (--feature-source rollout)
# probe on the group's four arms, then calls analysis/probe_report.py, which writes
# analysis_outputs/probe_report.md plus one <figure>.md beside every PNG.
#
# The overlay stays READ-ONLY by default: unlike planning, the probes never import
# mujoco_py, so they do not take the build lock in mujoco_py/generated/ that forces
# planning jobs to mount it read-write -- several probe jobs can share one overlay.
# OVERLAY_RW=1 exists in case something ever needs to write inside the image.
# =============================================================================
#SBATCH --job-name=ts-probe
#SBATCH --account=torch_pr_718_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/akn7847/datasets/worldmodelcheckpoints/logs/slurm-probe-%j.out
#SBATCH --error=/scratch/akn7847/datasets/worldmodelcheckpoints/logs/slurm-probe-%j.err
set -euo pipefail

REPO_HOST="${REPO_HOST:-}"
if [ -z "$REPO_HOST" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "$PWD"; do
        if [ -n "$_cand" ] && [ -f "$_cand/analysis/run_probes.py" ]; then REPO_HOST="$_cand"; break; fi
    done
    unset _cand
fi
if [ -z "$REPO_HOST" ] || [ ! -f "$REPO_HOST/analysis/run_probes.py" ]; then
    echo "FATAL: cannot locate the repo (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, \$PWD=$PWD)." >&2
    echo "       Submit from the repo root: cd ~/wm/temporal-straightening-PL && sbatch run_scripts/probe_server.sh" >&2
    exit 1
fi

SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/akn7847/containers/temporal-straightening/overlay-50G-10M.ext3}"
CONTAINER_CONDA="${CONTAINER_CONDA:-/opt/miniconda}"
CONDA_ENV="${CONDA_ENV:-ts}"
ENV_FILE="${ENV_FILE:-$HOME/mujoco_env.sh}"
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer 2>/dev/null || true)}"
if [ -z "${REPO_IN_CONTAINER:-}" ]; then
    case "$REPO_HOST" in
        "$HOME"/*) REPO_IN_CONTAINER="$REPO_HOST" ;;
        *)         REPO_IN_CONTAINER=/home/akn7847/wm/temporal-straightening-PL ;;
    esac
fi

# knobs handed to run_probes.py (plain env vars; apptainer passes them through)
ENVS="${ENVS:-point_maze,point_maze_medium,pusht}"
RECIPES="${RECIPES:-}"
ONLY="${ONLY:-}"
METHODS="${METHODS:-encoded,rollout}"
LIMIT_EPISODES="${LIMIT_EPISODES:-0}"
TMAX="${TMAX:-0}"
SEED="${SEED:-0}"
TEST_FRAC="${TEST_FRAC:-0.3}"

source "$REPO_HOST/run_scripts/dataset_paths.sh"
OUTDIR="${OUTDIR:-$REPO_IN_CONTAINER/analysis_outputs}"
MOUNT="$OVERLAY:ro"; [ "${OVERLAY_RW:-0}" = "1" ] && MOUNT="$OVERLAY"
BIND_ARGS=(--bind "$HOME:$HOME")
[ -n "${EXTRA_BIND:-}" ] && BIND_ARGS+=(--bind "$EXTRA_BIND")

EXTRA=()
[ "${1:-}" = "--dry-run" ] && EXTRA+=(--dry-run)      # pass through, if given
PROBE_ARGS=(analysis/run_probes.py --ckpt-root "$CKPT_ROOT/test" --data-root "$DATA_ROOT"
            --outdir "$OUTDIR" --envs "$ENVS" --methods "$METHODS"
            --seed "$SEED" --test-frac "$TEST_FRAC" "${EXTRA[@]}")
[ -n "$RECIPES" ] && PROBE_ARGS+=(--recipes "$RECIPES")
[ -n "$ONLY" ] && PROBE_ARGS+=(--only "$ONLY")
[ "$LIMIT_EPISODES" != "0" ] && PROBE_ARGS+=(--limit-episodes "$LIMIT_EPISODES")
[ "$TMAX" != "0" ] && PROBE_ARGS+=(--tmax "$TMAX")

echo "==================== $(date '+%F %H:%M:%S') ===================="
echo "host    : $(hostname)  job=${SLURM_JOB_ID:-local}  gpu=$(nvidia-smi -L 2>/dev/null | head -1 || true)"
echo "repo    : $REPO_HOST  ->  $REPO_IN_CONTAINER"
echo "ckpt    : $CKPT_ROOT/test     data: DATA_ROOT=$DATA_ROOT"
echo "groups  : envs=$ENVS recipes=${RECIPES:-<all>} only=${ONLY:-<all>} methods=$METHODS"
echo "overlay : $MOUNT   outdir=$OUTDIR"
echo "tool    : apptainer=$APPTAINER_BIN"
[ -f "$SIF" ] || { echo "FATAL: container image not found: $SIF" >&2; exit 1; }
[ -f "$OVERLAY" ] || { echo "FATAL: overlay not found: $OVERLAY" >&2; exit 1; }
[ -d "$CKPT_ROOT/test" ] || { echo "FATAL: checkpoint dir not found: $CKPT_ROOT/test" >&2; exit 1; }
[ -z "$APPTAINER_BIN" ] && { echo "FATAL: no apptainer on PATH (set APPTAINER_BIN)" >&2; exit 1; }

"$APPTAINER_BIN" exec --fakeroot --nv "${BIND_ARGS[@]}" --overlay "$MOUNT" "$SIF" \
    bash -lc "
        set -e
        export PATH='$CONTAINER_CONDA/bin:\$PATH'
        source '$CONTAINER_CONDA/etc/profile.d/conda.sh'
        conda activate '$CONDA_ENV'
        [ -r '$ENV_FILE' ] && source '$ENV_FILE'
        cd '$REPO_IN_CONTAINER'
        export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
        export DATA_ROOT='$DATA_ROOT'
        echo \"[container] python=\$(command -v python)  DATA_ROOT=\$DATA_ROOT\"
        python ${PROBE_ARGS[*]}
    "
echo "==================== done $(date '+%F %H:%M:%S') ===================="