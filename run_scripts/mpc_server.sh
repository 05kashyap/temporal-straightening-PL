#!/bin/bash
# =============================================================================
# mpc_server.sh -- sbatch-able wrapper that runs run_mpc.sh (planning / MPC) inside
#                  the project container on one GPU node.
#
#   sbatch run_scripts/mpc_server.sh                       # default JOBS (3 envs x gd_mpc, all arms)
#   sbatch run_scripts/mpc_server.sh umaze all both        # one job: env variant planner
#   FULL=0 sbatch run_scripts/mpc_server.sh                # validation: OOM check + runtime estimate
#   OL=1   sbatch run_scripts/mpc_server.sh                # open loop (one plan per episode)
#   TS_ENV_START_METHOD=spawn FULL=0 sbatch ...             # spawned env workers (the EGL/fork fix, 11.4)
#   ARM_NAMES="baseline straighten p_reg both" FULL=0 sbatch ...   # which four run dirs (else auto-discovered)
#   SEEDS="100" PREFLIGHT=0 OVERLAY_RW=1 sbatch ...        # knobs (see below)
#   JOBS="umaze:all:gd_mpc medium:both:both" sbatch ...    # explicit grid
#   LOG_TAG=flatten ARM_NAMES="..." sbatch ...             # parallel recipe jobs (see below)
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
# ARM_NAMES and TS_ENV_START_METHOD are in PREAMBLE_VARS: the container body exports them
# explicitly (no dependence on apptainer's environment pass-through) and the header prints
# both, so "was the knob actually set?" is never a question. TS_ENV_START_METHOD=spawn is
# the EGL+fork fix -- the env workers are forked, and a child that forks after GL has been
# initialised fails to initialise OpenGL (11.4).
#
# Sibling jobs and the log file: the log name is mpc_<env>_<variant>_<planner>[.ol].log, so
# two recipes of one env (medium flatten vs medium aggmlp) submitted in parallel append to
# the SAME file. The summary is scoped by the last banner in that file, so with two writers
# it can attribute the other job's !! / [estimate] lines to this one. LOG_TAG=<recipe> puts
# the recipe in the file name instead: LOG_TAG=aggmlp -> mpc_medium_all_mpc_cem.aggmlp.log.
# (Unset = the old name, so nothing changes for single-recipe runs.)
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
# plan.py's startup. OVERLAY_RW=1 is needed for every planning run, not just the first:
# mujoco_py takes a write lock (fasteners.InterProcessLock) next to cymj*.so on *every*
# `import mujoco_py`, before it checks whether cymj is already built, so the default ':ro'
# mount fails there (OSError .../mujocopy-buildlock). Corollary: do not run this while
# another job holds the same overlay read-write.
#
# Edit the #SBATCH lines for your account/partition if they differ.
# =============================================================================
#SBATCH --job-name=ts-mpc
#SBATCH --account=torch_pr_718_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16   # unchunked planning: one env process per episode (n_evals=50)
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
# Which apptainer to run. PATH alone is not trustworthy here: site startup files can
# prepend their own apptainer inside every bash process (that is how the self-test's own
# stub got skipped on torch-login-b-2), so an explicit APPTAINER_BIN wins and the header
# prints what will actually be used.
APPTAINER_BIN="${APPTAINER_BIN:-$(command -v apptainer 2>/dev/null || true)}"
# The generated body is PER JOB (SLURM_JOB_ID when under sbatch, else the shell's PID). It used
# to be one fixed path -- $HOME/.ts_mpc_body.sh -- which was fine for a single job but is not
# safe for the intended parallel arm jobs: each job writes its own values (JOBS/ARM_NAMES/
# SEEDS/CHUNK, ...) into the body's preamble, so two jobs starting together could overwrite the
# file while the other bash was still reading it -- one job would silently run the other's
# configuration, or die on a partly written file. The per-job file is kept after the run for
# post-mortem; `rm -f ~/.ts_mpc_body.*.sh` cleans them up.
BODY="${BODY:-$HOME/.ts_mpc_body.${SLURM_JOB_ID:-$$}.sh}"
# shared data/checkpoint layout, same file train_server.sh uses
source "$REPO_HOST/run_scripts/dataset_paths.sh"

FULL="${FULL:-1}"
OL="${OL:-0}"
SEEDS="${SEEDS:-100 101 102}"
PREFLIGHT="${PREFLIGHT:-1}"
PROBE="${PROBE:-0}"              # 1 = run run_scripts/gl_backend_probe.py first
CHUNK="${CHUNK:-null}"            # closed-loop plan/eval chunking (null = one batch for all n_evals)
OL_CHUNK="${OL_CHUNK:-null}"      # open-loop ditto
CEM_CHUNK="${CEM_CHUNK:-null}"    # CEM sample_chunk_size (null = all candidates at once)
# Both are handed straight to run_mpc.sh / plan.py. They are in PREAMBLE_VARS, so the
# container body exports (and the header prints) exactly the value the stages will see,
# instead of relying on apptainer's host-environment pass-through.
ARM_NAMES="${ARM_NAMES:-}"                       # 4 run-dir names: baseline straighten p_reg both
TS_ENV_START_METHOD="${TS_ENV_START_METHOD:-}"   # spawn = fresh interpreter per env worker (EGL/fork fix)
LOG_TAG="${LOG_TAG:-}"                           # optional suffix on the per-job log (parallel recipes)
# Planning budgets. run_mpc.sh reads every one of these with ${X:-default}, so exporting them
# (empty included) only pins what the container sees; before this they were pass-through only,
# which is exactly the sort of thing that can differ silently between two sibling jobs. The
# CEM ones end up in the run dir name (ns/opt), so the values stay verifiable afterwards.
FULL_N_EVALS="${FULL_N_EVALS:-}"       # full run: episodes to evaluate   (run_mpc.sh: 50)
FULL_MAX_ITER="${FULL_MAX_ITER:-}"     # full run: MPC iteration cap      (20)
GD_OPT="${GD_OPT:-}"                   # GD sub-planner opt steps         (100)
CEM_SAMPLES="${CEM_SAMPLES:-}"         # closed-loop CEM candidates       (200)
CEM_OPT="${CEM_OPT:-}"                 # closed-loop CEM opt steps        (10)
OL_N_EVALS="${OL_N_EVALS:-}"           # open-loop episodes               (= FULL_N_EVALS)
OL_GD_OPT="${OL_GD_OPT:-}"             # open-loop GD opt steps           (100)
OL_CEM_SAMPLES="${OL_CEM_SAMPLES:-}"   # open-loop CEM candidates         (300)
OL_CEM_OPT="${OL_CEM_OPT:-}"           # open-loop CEM opt steps          (30)

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

# --- binds ---------------------------------------------------------------------------
# $HOME is always bound (the env file and the checkout live there). EXTRA_BIND adds one more
# host:container pair. LOCK_BIND=1 moves mujoco_py's build lock out of the overlay: planning
# needs the overlay writable only because mujoco_py takes a write lock at
#   <ts env>/lib/python3.9/site-packages/mujoco_py/generated/mujocopy-buildlock
# on every import (SERVER_CONTEXT 11.1). A read-write ext3 overlay is single-mount, which is
# why a second planning job cannot start while one is running; bound to a file in $HOME, the
# overlay needs no writes, so it can stay ':ro' (the default) and any number of jobs -- e.g.
# the closed-loop and the open-loop run -- can share it concurrently.
CONTAINER_ENV_PREFIX="${CONTAINER_ENV_PREFIX:-/opt/conda-envs/ts}"
CONTAINER_LOCK="${CONTAINER_LOCK:-$CONTAINER_ENV_PREFIX/lib/python3.9/site-packages/mujoco_py/generated/mujocopy-buildlock}"
BIND_ARGS=(--bind "$HOME:$HOME")
if [ -n "${EXTRA_BIND:-}" ]; then BIND_ARGS+=(--bind "$EXTRA_BIND"); fi
if [ "${LOCK_BIND:-0}" = "1" ]; then
    LOCK_FILE="$HOME/.mujoco/mujocopy-buildlock"
    mkdir -p "$(dirname "$LOCK_FILE")" 2>/dev/null || true
    [ -f "$LOCK_FILE" ] || : > "$LOCK_FILE" 2>/dev/null || true
    if [ -f "$LOCK_FILE" ]; then
        BIND_ARGS+=(--bind "$LOCK_FILE:$CONTAINER_LOCK")
    else
        echo "FATAL: LOCK_BIND=1 but $LOCK_FILE cannot be created" >&2
        exit 1
    fi
fi
# The body reports a read-only mount when the preflight fails: that alone is fatal for
# `import mujoco_py` (see the note above), so it is passed along with the values.
OVERLAY_RO=0; case "$MOUNT" in *:ro) OVERLAY_RO=1 ;; esac
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
echo "tool    : apptainer=${APPTAINER_BIN:-NOT FOUND (PATH and APPTAINER_BIN are empty)}"
echo "binds   : ${BIND_ARGS[*]}   (LOCK_BIND=${LOCK_BIND:-0}, EXTRA_BIND=${EXTRA_BIND:-<none>})"
echo "probe   : PROBE=$PROBE   (1 = run the fork/spawn EGL probe before the preflight)"
echo "jobs    : $JOBS"
echo "mode    : FULL=$FULL OL=$OL seeds='$SEEDS' preflight=$PREFLIGHT sanity=${SANITY:-1} overlay=$MOUNT"
echo "chunk   : closed=$CHUNK open=$OL_CHUNK cem=$CEM_CHUNK   (null = no chunking)"
echo "ckpt    : $CKBPT_PATH"
echo "data    : DATA_ROOT=$DATA_ROOT"

# Only $HOME is mounted into the image (see the apptainer line at the bottom), so a checkout
# elsewhere is invisible inside it. Say so *here*, where a fix costs seconds, instead of
# letting the body fail on `cd` after a queue slot. (This is how a /tmp clone behaves.)
case "$REPO_IN_CONTAINER" in
    "$HOME"/*) ;;
    *)  if [ "${ALLOW_OUTSIDE_HOME:-0}" = "1" ]; then
            echo "warn    : REPO_IN_CONTAINER=$REPO_IN_CONTAINER is outside \$HOME; only \$HOME is bound,"
            echo "          so this works only if you mounted that path yourself (ALLOW_OUTSIDE_HOME=1)."
        else
            echo "FATAL: the checkout is at $REPO_HOST (outside \$HOME), but only \$HOME is bound into the" >&2
            echo "       container (--bind \"\$HOME:\$HOME\"), so the body cannot cd into" >&2
            echo "       REPO_IN_CONTAINER=$REPO_IN_CONTAINER." >&2
            echo "       Fix: submit from a checkout under \$HOME, or set REPO_IN_CONTAINER=<path visible in" >&2
            echo "       the image> and add --bind <host>:<container> to the apptainer line, then rerun with" >&2
            echo "       ALLOW_OUTSIDE_HOME=1 to confirm you did that." >&2
            exit 1
        fi
        ;;
esac
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
               CHUNK OL_CHUNK CEM_CHUNK CKBPT_PATH CKPT_ROOT ENV_FILE OL_SUFFIX OVERLAY_RO PROBE
               ARM_NAMES TS_ENV_START_METHOD LOG_TAG
               FULL_N_EVALS FULL_MAX_ITER GD_OPT CEM_SAMPLES CEM_OPT
               OL_N_EVALS OL_GD_OPT OL_CEM_SAMPLES OL_CEM_OPT)
OL_SUFFIX=""
if [ "$OL" = "1" ]; then OL_SUFFIX=".ol"; fi

{
echo '#!/usr/bin/env bash'
echo '# Generated by run_scripts/mpc_server.sh -- preamble here, body below (see its note).'
echo 'set -uo pipefail'
for _v in "${PREAMBLE_VARS[@]}"; do echo "export $_v=$(printf '%q' "${!_v}")"; done
unset _v
cat <<'BODYEOF'
[ -r "$CONTAINER_CONDA/etc/profile.d/conda.sh" ] || { echo "FATAL: $CONTAINER_CONDA/etc/profile.d/conda.sh is not visible in the image -- is CONTAINER_CONDA right?" >&2; exit 1; }
export PATH="$CONTAINER_CONDA/bin:$PATH"
source "$CONTAINER_CONDA/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "FATAL: conda activate $CONDA_ENV failed (CONTAINER_CONDA=$CONTAINER_CONDA)" >&2; exit 1; }
command -v python >/dev/null 2>&1 || { echo "FATAL: no python on PATH after activating CONTAINER_CONDA=$CONTAINER_CONDA, CONDA_ENV=$CONDA_ENV -- are those right?" >&2; exit 1; }
[ -r "$ENV_FILE" ] || { echo "FATAL: $ENV_FILE not visible in the container -- add --bind \$HOME:\$HOME" >&2; exit 9; }
source "$ENV_FILE"
cd "$REPO_IN_CONTAINER" || { echo "FATAL: $REPO_IN_CONTAINER is not visible in the container" >&2; exit 1; }
source run_scripts/dataset_paths.sh || { echo "FATAL: run_scripts/dataset_paths.sh missing under $REPO_IN_CONTAINER -- is REPO_IN_CONTAINER right?" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
log_dir="$CKPT_ROOT/logs"
mkdir -p "$log_dir"
echo "[container] python=$(command -v python)  DATA_ROOT=$DATA_ROOT  log_dir=$log_dir"
echo "[container] knobs  : TS_ENV_START_METHOD=${TS_ENV_START_METHOD:-<unset>}  ARM_NAMES=${ARM_NAMES:-<auto-discovered>}  LOG_TAG=${LOG_TAG:-<none>}"
# Budgets as run_mpc.sh will read them, defaults included: an OL/CL comparison cannot then
# silently run different CEM budgets (the run dir name carries ns/opt as well).
echo "[container] budgets: FULL_N_EVALS=${FULL_N_EVALS:-50(def)} FULL_MAX_ITER=${FULL_MAX_ITER:-20(def)} GD_OPT=${GD_OPT:-100(def)} CEM_SAMPLES=${CEM_SAMPLES:-200(def)} CEM_OPT=${CEM_OPT:-10(def)}"
echo "[container] budgets: OL_N_EVALS=${OL_N_EVALS:-<FULL>} OL_GD_OPT=${OL_GD_OPT:-100(def)} OL_CEM_SAMPLES=${OL_CEM_SAMPLES:-300(def)} OL_CEM_OPT=${OL_CEM_OPT:-30(def)}"

if [ "$PROBE" = "1" ]; then
    echo "---- GL probe: does a forked env worker initialise EGL here? ----"
    python run_scripts/gl_backend_probe.py || echo "[probe] exited $? -- continuing anyway"
fi

if [ "$PREFLIGHT" = "1" ]; then
    echo "---- preflight: mujoco / gym / EGL / datasets ----"
    python run_scripts/mujoco_smoke.py || {
        echo "FATAL: preflight failed -- not starting plan.py (fix the stage above)." >&2
        if [ "$OVERLAY_RO" = "1" ]; then
            echo "       NOTE: the overlay is mounted read-only for this run, and mujoco_py takes a" >&2
            echo "       write lock next to cymj*.so on EVERY import -- so ':ro' fails there even with" >&2
            echo "       cymj already built. If the failing stage mentions mujocopy-buildlock:" >&2
            echo "       OVERLAY_RW=1 sbatch run_scripts/mpc_server.sh" >&2
        fi
        exit 1
    }
fi

rc=0
fail=""
for job in $JOBS; do
    env_sel="${job%%:*}"; rest="${job#*:}"
    variant="${rest%%:*}"; planner="${rest#*:}"
    log="$log_dir/mpc_${env_sel}_${variant}_${planner}${OL_SUFFIX}${LOG_TAG:+.${LOG_TAG}}.log"
    echo
    # This banner is tee'd into $log on purpose: the log is append-only across jobs, so the
    # summary below needs a marker to tell THIS job's output from an earlier run's (it used
    # to grep the whole file and therefore report the previous run's numbers).
    echo "---- $(date '+%F %H:%M:%S') $env_sel / $variant / $planner -> $log ----" | tee -a "$log"
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
    log="$log_dir/mpc_${env_sel}_${variant}_${planner}${OL_SUFFIX}${LOG_TAG:+.${LOG_TAG}}.log"
    # $log is append-only across jobs, so every read is scoped to THIS job's section: the
    # banner above the run_mpc.sh call marks where it starts. Reading the whole file (the old
    # `tail -1` on success_rate, or the first four [estimate] lines anywhere in it) reported an
    # *earlier* run's numbers -- e.g. a success_rate for a run that never got past [1/4].
    _from=$(grep -nE '^---- [0-9]{4}-[0-9]{2}-[0-9]{2} .* -> ' "$log" 2>/dev/null | tail -1 | cut -d: -f1 || true)
    _sec() { if [ -n "$_from" ]; then tail -n "+$_from" "$log"; else cat "$log"; fi; }
    sr=$(_sec | grep -oE 'success_rate[ =:]+[0-9.]+' | tail -1 | grep -oE '[0-9.]+' || true)
    printf '  %-28s %-9s %-8s success_rate=%s\n' "$env_sel" "$variant" "$planner" "${sr:-<n/a>}"
    # Name the failing stages of this run: success_rate=<n/a> alone does not say where it died,
    # and run_mpc.sh's own "!!" lines are the only place that knows. sed, not head: under
    # pipefail an early exit would SIGPIPE the upstream grep.
    _nbad=$(_sec | grep -c '^  !!' || true)
    if [ "${_nbad:-0}" -gt 0 ]; then _sec | grep '^  !!' | sed -n '1,4p' | sed 's/^/      /'; fi
    if [ "$FULL" = "0" ]; then
        # A FULL=0 run has no success_rate by design: its [estimate] lines are the result,
        # and they are what decides whether the full run fits the allocation.
        _nest=$(_sec | grep -c '\[estimate\]' || true)
        if [ "${_nest:-0}" -gt 0 ]; then
            _sec | grep '\[estimate\]' | sed -n '1,4p' | sed 's/^/      /'
        else
            echo "      no [estimate] lines in this job's section -- it did not reach [3/4]"
        fi
    fi
    echo "      log: $log"
done
unset _from _nbad _nest
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
    echo "--dry-run: apptainer exec --fakeroot --nv ${BIND_ARGS[*]} --overlay $MOUNT $SIF bash -lc 'bash $BODY'"
    echo "---- generated body ----"
    sed -n '1,200p' "$BODY"
    exit 0
fi

if [ ! -r "$ENV_FILE" ]; then
    echo "FATAL: ENV_FILE=$ENV_FILE is not readable here -- run run_scripts/setup_mujoco_server.sh" >&2
    echo "       first (SERVER_CONTEXT 11.2), or set ENV_FILE=/path/to/mujoco_env.sh." >&2
    exit 1
fi

if [ -z "$APPTAINER_BIN" ]; then
    echo "FATAL: no apptainer found (not on PATH, and APPTAINER_BIN is unset)${SLURM_JOB_ID:+ -- job $SLURM_JOB_ID}." >&2
    echo "       Load the module you normally use, or point at it yourself, e.g." >&2
    echo "       APPTAINER_BIN=/share/apps/apptainer/bin/apptainer sbatch run_scripts/mpc_server.sh" >&2
    exit 1
fi

"$APPTAINER_BIN" exec --fakeroot --nv "${BIND_ARGS[@]}" --overlay "$MOUNT" "$SIF" \
    bash -lc "bash $BODY"
