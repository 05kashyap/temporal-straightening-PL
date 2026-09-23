#!/bin/bash
# =============================================================================
# mpc_server.sh -- sbatch-able wrapper that runs run_mpc.sh (planning / MPC) inside
#                  the project container on one GPU node.
#
#   sbatch run_scripts/mpc_server.sh                       # default JOBS (3 envs x gd_mpc, all arms)
#   sbatch run_scripts/mpc_server.sh umaze all both        # one job: env variant planner
#   FULL=0 sbatch run_scripts/mpc_server.sh                # validation: OOM check + runtime estimate
#   OL=1   sbatch run_scripts/mpc_server.sh                # open loop (one plan per episode)
#   SEEDS="100" PREFLIGHT=0 OVERLAY_RW=1 sbatch ...        # knobs (see below)
#   JOBS="umaze:all:gd_mpc medium:both:both" sbatch ...    # explicit grid
#
# Inside one allocation the jobs run SEQUENTIALLY -- planning wants the whole GPU --
# each into $CKPT_ROOT/logs/mpc_<env>_<variant>_<planner>.log, then a pass/fail summary.
#
# Arm names: run_mpc.sh's built-in MODELS are the dev machine's names. The checkpoints
# trained here use different recipes (projchannel, ttaggtwothirds, aggflatten), so
# run_mpc.sh DISCOVERS the four arms under $CKPT_ROOT/test by token (baseline=_False_,
# straighten=cos without two-thirds, p_reg=two-thirds without cos, both=cos+two-thirds)
# whenever the built-ins are absent, and aborts with the candidate list if an arm is
# ambiguous. ARM_NAMES="..." still wins, and is then used for every job in the grid.
#
# Chunking: the evaluation is NOT chunked by default -- CHUNK=null / OL_CHUNK=null /
# CEM_CHUNK=null put all n_evals (50) episodes in one batch, which also starts one
# env process per episode. Those workers are CPU-bound and a job here is capped at 16
# CPUs (#SBATCH below), so 50 simulators share 16 cores: fine, but if the run turns
# out simulator-bound, CHUNK=16 OL_CHUNK=16 matches the allocation (3 batches instead
# of 50). The script prints that hint at startup. CHUNK=1 restores the 12 GB-laptop
# behaviour; CEM_CHUNK=50 bounds CEM candidate memory.
#
# Preflight (PREFLIGHT=1, the default) runs run_scripts/mujoco_smoke.py in the same
# container first, so a broken simulator/dataset/EGL fails in seconds instead of after
# plan.py's startup. The first run of a *new* machine may also want OVERLAY_RW=1 (the
# first import of mujoco_py compiles cymj into the overlay); afterwards read-only is
# enough and can share the node with a training job.
#
# Edit the #SBATCH lines for your account/partition if they differ.
# =============================================================================
#SBATCH --job-name=ts-mpc
#SBATCH --account=torch_pr_718_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # unchunked planning: one env process per episode (n_evals=50)
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/akn7847/datasets/worldmodelcheckpoints/logs/slurm-mpc-%j.out
#SBATCH --error=/scratch/akn7847/datasets/worldmodelcheckpoints/logs/slurm-mpc-%j.err
set -euo pipefail

# Locate the checkout. When sbatch invokes this file it runs a *spool copy*
# (/opt/slurm/data/slurmd/job<N>/slurm_script), so $BASH_SOURCE cannot be used to find the
# repo: prefer SLURM_SUBMIT_DIR (where sbatch was called from), then the script's own path
# for a direct `bash run_scripts/mpc_server.sh`, then $PWD. Override with REPO_HOST=...
REPO_HOST="${REPO_HOST:-}"
if [ -z "$REPO_HOST" ]; then
    for _cand in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "$PWD"; do
        if [ -n "$_cand" ] && [ -f "$_cand/run_scripts/dataset_paths.sh" ]; then REPO_HOST="$_cand"; break; fi
    done
    unset _cand
fi
if [ -z "$REPO_HOST" ] || [ ! -f "$REPO_HOST/run_scripts/dataset_paths.sh" ]; then
    echo "FATAL: cannot locate the repo (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, script dir, \$PWD=$PWD)." >&2
    echo "       Submit from the repo root: cd ~/wm/temporal-straightening-PL && sbatch run_scripts/mpc_server.sh" >&2
    echo "       (or export REPO_HOST=/path/to/temporal-straightening-PL)." >&2
    exit 1
fi
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-/scratch/akn7847/containers/temporal-straightening/overlay-50G-10M.ext3}"
# Where the checkout is visible *inside* the container. --bind "$HOME:$HOME" maps $HOME to
# the identical path, so a checkout under $HOME is visible at that same path and needs no
# extra mount. (This used to be a bare hardcoded default, which silently mismatched a
# checkout that lived anywhere else.)
if [ -z "${REPO_IN_CONTAINER:-}" ]; then
    case "$REPO_HOST" in
        "$HOME"/*) REPO_IN_CONTAINER="$REPO_HOST" ;;
        *)         REPO_IN_CONTAINER=/home/akn7847/wm/temporal-straightening-PL ;;
    esac
else
    [ "$REPO_IN_CONTAINER" = "$REPO_HOST" ] || echo "note    : REPO_IN_CONTAINER=$REPO_IN_CONTAINER differs from REPO_HOST=$REPO_HOST"
fi
# Container-side conda prefix (the ts env has moved before: /opt/conda-envs/ts), the env
# name inside it, and the planning env file written by setup_mujoco_server.sh.
CONTAINER_CONDA="${CONTAINER_CONDA:-/opt/miniconda}"
CONDA_ENV="${CONDA_ENV:-ts}"
ENV_FILE="${ENV_FILE:-$HOME/mujoco_env.sh}"
BODY="${BODY:-$HOME/.ts_mpc_body.sh}"
# shared data/checkpoint layout, same file train_server.sh uses
source "$REPO_HOST/run_scripts/dataset_paths.sh"

FULL="${FULL:-1}"
OL="${OL:-0}"
SEEDS="${SEEDS:-100 101 102}"
PREFLIGHT="${PREFLIGHT:-1}"
CHUNK="${CHUNK:-null}"            # closed-loop plan/eval chunking (null = one batch for all n_evals)
OL_CHUNK="${OL_CHUNK:-null}"      # open-loop ditto
CEM_CHUNK="${CEM_CHUNK:-null}"    # CEM sample_chunk_size (null = all candidates at once)

# Unchunked planning starts one env process per episode; say so rather than silently
# chunking (or silently oversubscribing the allocation).
_cpus="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 1)}"
if { [ "$CHUNK" = "null" ] || [ "$OL_CHUNK" = "null" ]; } && [ "$_cpus" -lt 50 ]; then
    echo "note    : unchunked planning starts one env process per episode (50) on $_cpus CPUs;"
    echo "          if the run is simulator-bound, resubmit with CHUNK=$_cpus OL_CHUNK=$_cpus"
fi
unset _cpus
CKBPT_PATH="${CKBPT:-$CKPT_ROOT/test}"
MOUNT="$OVERLAY:ro"; [ "${OVERLAY_RW:-0}" = "1" ] && MOUNT="$OVERLAY"
if [ "$#" -ge 2 ]; then
    JOBS="$1:$2:${3:-both}"
else
    JOBS="${JOBS:-umaze:all:gd_mpc medium:all:gd_mpc pusht:all:gd_mpc}"
fi

for arg in "$@"; do case "$arg" in -h|--help) sed -n '2,32p' "$0"; exit 0 ;; esac; done

if ! mkdir -p "$CKPT_ROOT/logs" 2>/dev/null; then
    echo "FATAL: cannot create $CKPT_ROOT/logs -- is CKPT_ROOT (=$CKPT_ROOT) correct for this machine?" >&2
    exit 1
fi
echo "==================== $(date '+%F %H:%M:%S') ===================="
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$(nvidia-smi -L 2>/dev/null | head -1 || true)"
echo "repo    : $REPO_HOST"
echo "cpus    : ${SLURM_CPUS_PER_TASK:-<not under slurm>} requested (nproc=$(nproc 2>/dev/null || echo ?))"
echo "body    : $BODY  (in container: repo=$REPO_IN_CONTAINER, conda=$CONTAINER_CONDA:$CONDA_ENV, env file=$ENV_FILE)"
echo "jobs    : $JOBS"
echo "mode    : FULL=$FULL OL=$OL seeds='$SEEDS' preflight=$PREFLIGHT overlay=$MOUNT"
echo "chunk   : closed=$CHUNK open=$OL_CHUNK cem=$CEM_CHUNK   (null = no chunking)"
echo "ckpt    : $CKBPT_PATH"
echo "data    : DATA_ROOT=$DATA_ROOT"
[ -f "$SIF" ] || { echo "FATAL: container image not found: $SIF" >&2; exit 1; }
[ -f "$OVERLAY" ] || { echo "FATAL: overlay not found: $OVERLAY" >&2; exit 1; }
[ -d "$CKBPT_PATH" ] || { echo "FATAL: checkpoint dir not found: $CKBPT_PATH (CKPT_ROOT/test)" >&2; exit 1; }

# ---- the in-container body: PREAMBLE FIRST, then the statements ----------------------
# The preamble has to come *before* the body. The body runs under `set -u`, so a value it
# uses before its own `export` line kills the job -- which is exactly what happened once:
#
#   /home/akn7847/.ts_mpc_body.sh: line 7: ENV_FILE: unbound variable
#
# because the export block was appended with `>>` *after* the body's `exit "$rc"`, so it
# never executed at all. PREAMBLE_VARS is the single source of truth here: it drives both
# the export block and the ordering check further down, so the two cannot drift apart.
PREAMBLE_VARS=(REPO_IN_CONTAINER CONTAINER_CONDA CONDA_ENV FULL OL SEEDS PREFLIGHT JOBS
               CHUNK OL_CHUNK CEM_CHUNK CKBPT_PATH CKPT_ROOT ENV_FILE OL_SUFFIX)
OL_SUFFIX=""
if [ "$OL" = "1" ]; then OL_SUFFIX=".ol"; fi

{
echo '#!/usr/bin/env bash'
echo '# Generated by run_scripts/mpc_server.sh -- preamble here, body below (see its note).'
echo 'set -uo pipefail'
for _v in "${PREAMBLE_VARS[@]}"; do echo "export $_v=$(printf '%q' "${!_v}")"; done
unset _v
cat <<'BODYEOF'
export PATH="$CONTAINER_CONDA/bin:$PATH"
source "$CONTAINER_CONDA/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
[ -r "$ENV_FILE" ] || { echo "FATAL: $ENV_FILE not visible in the container -- add --bind \$HOME:\$HOME" >&2; exit 9; }
source "$ENV_FILE"
cd "$REPO_IN_CONTAINER" || { echo "FATAL: $REPO_IN_CONTAINER is not visible in the container" >&2; exit 1; }
source run_scripts/dataset_paths.sh || { echo "FATAL: run_scripts/dataset_paths.sh missing under $REPO_IN_CONTAINER -- is REPO_IN_CONTAINER right?" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
log_dir="$CKPT_ROOT/logs"
mkdir -p "$log_dir"
echo "[container] python=$(command -v python)  DATA_ROOT=$DATA_ROOT  log_dir=$log_dir"

if [ "$PREFLIGHT" = "1" ]; then
    echo "---- preflight: mujoco / gym / EGL / datasets ----"
    python run_scripts/mujoco_smoke.py || {
        echo "FATAL: preflight failed -- not starting plan.py (fix the stage above)." >&2
        exit 1
    }
fi

rc=0
fail=""
for job in $JOBS; do
    env_sel="${job%%:*}"; rest="${job#*:}"
    variant="${rest%%:*}"; planner="${rest#*:}"
    log="$log_dir/mpc_${env_sel}_${variant}_${planner}${OL_SUFFIX}.log"
    echo
    echo "---- $(date '+%F %H:%M:%S') $env_sel / $variant / $planner -> $log ----"
    if bash run_scripts/run_mpc.sh "$env_sel" "$variant" "$planner" --ckpt "$CKBPT_PATH" 2>&1 | tee -a "$log"; then
        echo "  [ok]   $env_sel $variant $planner"
    else
        echo "  [FAIL] $env_sel $variant $planner  (see $log)" >&2
        fail="$fail $env_sel/$variant/$planner"
        rc=1
    fi
done

echo
echo "==================== summary ===================="
for job in $JOBS; do
    env_sel="${job%%:*}"; rest="${job#*:}"
    variant="${rest%%:*}"; planner="${rest#*:}"
    log="$log_dir/mpc_${env_sel}_${variant}_${planner}${OL_SUFFIX}.log"
    sr=$(grep -oE 'success_rate[ =:]+[0-9.]+' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || true)
    printf '  %-28s %-9s %-8s success_rate=%s\n' "$env_sel" "$variant" "$planner" "${sr:-<n/a>}"
    echo "      log: $log"
done
[ -n "$fail" ] && echo "failed:$fail"
echo "summaries: $REPO_IN_CONTAINER/plan_outputs_*/summaries/"
exit "$rc"
BODYEOF
} > "$BODY"
chmod +x "$BODY"

# ---- the guard that was missing when this died in the container -----------------------
# Lint the generated file *here*, where a mistake costs seconds instead of a queue slot:
# every name in PREAMBLE_VARS must be exported before the first line that uses it. This is
# why DRY_RUN=1 is worth running -- it lints the file, it does not just print it.
bash -n "$BODY" || { echo "FATAL: the generated body has a syntax error: $BODY" >&2; exit 1; }
_bad=""
for _v in "${PREAMBLE_VARS[@]}"; do
    # No pipe to head here on purpose: under set -o pipefail an early exit gives the
    # upstream grep SIGPIPE, the pipeline fails, and set -e kills the script -- silently.
    # Strip the line number with parameter expansion instead.
    _def="$(grep -nE "^export ${_v}=" "$BODY" || true)"; _def="${_def%%:*}"
    _use="$(grep -nE -e "[$]${_v}[^A-Za-z0-9_]" -e "[$][{]${_v}[^A-Za-z0-9_]" "$BODY" || true)"; _use="${_use%%:*}"
    if [ -z "$_def" ]; then
        _bad="$_bad ${_v}(never exported)"
    elif [ -n "$_use" ] && [ "$_use" -lt "$_def" ]; then
        _bad="$_bad ${_v}(line $_use uses it, line $_def defines it)"
    fi
done
unset _v _def _use
if [ -n "$_bad" ]; then
    echo "FATAL: the generated body uses values before they are defined:$_bad" >&2
    echo "       The preamble must be written before the body -- see the note above." >&2
    exit 1
fi
unset _bad

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--dry-run: 'bash -n' and the preamble-order check passed."
    echo "--dry-run: apptainer exec --fakeroot --nv --bind \$HOME:\$HOME --overlay $MOUNT $SIF bash -lc 'bash $BODY'"
    echo "---- generated body ----"
    sed -n '1,200p' "$BODY"
    exit 0
fi

if [ ! -r "$ENV_FILE" ]; then
    echo "FATAL: ENV_FILE=$ENV_FILE is not readable here -- run run_scripts/setup_mujoco_server.sh" >&2
    echo "       first (SERVER_CONTEXT 11.2), or set ENV_FILE=/path/to/mujoco_env.sh." >&2
    exit 1
fi

apptainer exec --fakeroot --nv --bind "$HOME:$HOME" --overlay "$MOUNT" "$SIF" \
    bash -lc "bash $BODY"
