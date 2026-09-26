#!/usr/bin/env bash
# =============================================================================
# probe_container.sh -- the in-container half of run_scripts/probe_server.sh.
#
# A real file rather than a `bash -lc "..."` string, because that string is what
# job 18573156 died in:
#
#   /opt/miniconda/etc/profile.d/conda.sh: line 71: dirname: command not found
#   /opt/miniconda/etc/profile.d/conda.sh: line 71: dirname: command not found
#
# i.e. the job's PATH carried no coreutils directory, so conda.sh could not work,
# `conda activate` then failed, and the unguarded `set -e` in that string killed
# the run with nothing in the log to say why. Here:
#
#   * PATH is *built* (absolute) instead of inherited, so dirname/sed/python exist
#     even when the submitting shell or the image profile dropped /usr/bin;
#   * conda activation is best-effort -- the interpreter is resolved by absolute
#     path, so a broken conda.sh can no longer take the job down;
#   * every stage prints, and a missing python is a named FATAL;
#   * the argv is an array, so no word-splitting guesses about paths or lists.
#
# Usage (the wrapper calls this with the container's bash):
#   bash probe_container.sh CKPT_ROOT DATA_ROOT OUTDIR ENVS METHODS SEED \
#        TEST_FRAC RECIPES ONLY LIMIT_EPISODES TMAX
# Environment: CONTAINER_CONDA, CONDA_ENV, REPO_IN_CONTAINER, ENV_FILE, DRY_RUN
# =============================================================================
set -uo pipefail          # deliberately NOT -e: each step is checked and reported

CKPT_ROOT="${1:?checkpoint root}"; DATA_ROOT="${2:?data root}"; OUTDIR="${3:?outdir}"
ENVS="${4:?envs}"; METHODS="${5:-encoded,rollout}"; SEED="${6:-0}"; TEST_FRAC="${7:-0.3}"
RECIPES="${8:-}"; ONLY="${9:-}"; LIMIT_EPISODES="${10:-0}"; TMAX="${11:-0}"
CONTAINER_CONDA="${CONTAINER_CONDA:-/opt/conda-envs/ts}"
CONDA_ENV="${CONDA_ENV:-ts}"
ENV_FILE="${ENV_FILE:-}"
REPO="${REPO_IN_CONTAINER:?repo path inside the container}"

# 1) Build PATH from scratch. conda.sh, sed, dirname, nvidia-smi ... all live in the
#    system dirs, and inheriting a PATH without them is exactly what broke.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
for d in "$CONTAINER_CONDA/bin" "$CONTAINER_CONDA/envs/$CONDA_ENV/bin"; do
    [ -d "$d" ] && export PATH="$d:$PATH"
done
echo "[container] PATH=$PATH"

# 2) conda is a convenience only. Sourced with -u disabled, because conda.sh is not
#    written for `set -u`, and its own errors must not become fatal here.
set +u
if [ -r "$CONTAINER_CONDA/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONTAINER_CONDA/etc/profile.d/conda.sh" 2>/dev/null \
        || echo "[container] warning: conda.sh reported errors (PATH above is still usable)"
    conda activate "$CONDA_ENV" 2>/dev/null \
        || echo "[container] warning: 'conda activate $CONDA_ENV' failed -- using an explicit python"
else
    echo "[container] note: no conda.sh under $CONTAINER_CONDA -- using an explicit python"
fi
set -u

# 3) Resolve the interpreter explicitly (several plausible layouts), then prove it.
PY=""
for root in "$CONTAINER_CONDA" /opt/conda-envs/ts /opt/miniconda; do
    [ -n "$root" ] || continue
    for sub in "envs/$CONDA_ENV/bin" "bin"; do
        cand="$root/$sub/python"
        if [ -x "$cand" ]; then PY="$cand"; break 2; fi
    done
done
[ -n "$PY" ] || PY="$(command -v python || true)"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "FATAL: no python found -- tried \$CONTAINER_CONDA/envs/\$CONDA_ENV/bin, \$CONTAINER_CONDA/bin," \
         "/opt/conda-envs/ts, /opt/miniconda, then PATH=$PATH" >&2
    exit 1
fi
echo "[container] python=$PY ($("$PY" -V 2>&1))"
echo "[container] repo=$REPO  ckpt=$CKPT_ROOT  DATA_ROOT=$DATA_ROOT  OUTDIR=$OUTDIR"

cd "$REPO" || { echo "FATAL: $REPO is not visible in the container" >&2; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT
if [ -r "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE" && echo "[container] sourced ENV_FILE=$ENV_FILE"
else
    echo "[container] ENV_FILE='$ENV_FILE' not readable -- skipping (the probes do not need mujoco)"
fi

# 4) argv as an array: no word-splitting guesses about paths or comma lists.
ARGS=(analysis/run_probes.py --ckpt-root "$CKPT_ROOT" --data-root "$DATA_ROOT"
      --outdir "$OUTDIR" --envs "$ENVS" --methods "$METHODS"
      --seed "$SEED" --test-frac "$TEST_FRAC")
[ -n "$RECIPES" ] && ARGS+=(--recipes "$RECIPES")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
[ "$LIMIT_EPISODES" != "0" ] && ARGS+=(--limit-episodes "$LIMIT_EPISODES")
[ "$TMAX" != "0" ] && ARGS+=(--tmax "$TMAX")
[ "${DRY_RUN:-0}" = "1" ] && ARGS+=(--dry-run)
echo "[container] running: $PY ${ARGS[*]}"
"$PY" "${ARGS[@]}"
rc=$?
echo "[container] run_probes.py rc=$rc"
exit "$rc"
