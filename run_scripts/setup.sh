#!/usr/bin/env bash
# Laptop environment exports for this repo (user kashyap, miniconda at $HOME/miniconda3).
# Sourced by run.sh / run_mpc.sh / run_wall_ablation.sh / run_curvature_analysis.sh
# / run_loss_landscape_comparison.sh from the repo root (they `cd` to their own
# directory first).

# Datasets (DINO-WM download; layout: <dir>/{point_maze,point_maze_medium,pusht_noise,wall_single,...}).
export DATASET_DIR="${DATASET_DIR:-$PWD/data/datasets}"

# ts conda env python + bin (patchelf/gcc needed for mujoco_py's one-time cymj build).
# The prefix is probed so this same file works on the laptop ($HOME/miniconda3) and in
# the project container (/opt/miniconda): an activated env (CONDA_PREFIX) wins, then the
# two known prefixes; PYTHON=/path/to/python overrides everything.
# `ts` (an activated env whose directory is named ts) wins over the two known prefixes,
# so activating another env by accident cannot silently pick the wrong interpreter.
TS_ENV_PREFIX=""
for cand in "$HOME/miniconda3/envs/ts" "/opt/conda-envs/ts" "/opt/miniconda/envs/ts"; do
    if [ -x "$cand/bin/python" ]; then TS_ENV_PREFIX="$cand"; break; fi
done
if [ -n "${CONDA_PREFIX:-}" ] && [ "$(basename "${CONDA_PREFIX:-}")" = "ts" ]         && [ -x "$CONDA_PREFIX/bin/python" ]; then
    TS_ENV_PREFIX="$CONDA_PREFIX"
elif [ -z "$TS_ENV_PREFIX" ] && [ -n "${CONDA_PREFIX:-}" ]         && [ -x "$CONDA_PREFIX/bin/python" ]; then
    TS_ENV_PREFIX="$CONDA_PREFIX"      # last resort: whatever env is activated
fi
if [ -n "$TS_ENV_PREFIX" ]; then
    TS_ENV_BIN="$TS_ENV_PREFIX/bin"
    export PY="${PYTHON:-$TS_ENV_BIN/python}"
    export PATH="$TS_ENV_BIN${PATH:+:$PATH}"
else
    export PY="${PYTHON:-python3}"
fi

# MuJoCo 2.1.2 (mujoco210 binary dist). run.sh uses the mujoco-2.1.2 name; both are
# provided (mujoco-2.1.2 -> mujoco210 symlink is created during setup).
MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export MUJOCO_PY_MUJOCO_PATH
# MUJOCO_LD_MODE=append puts the MuJoCo dirs LAST instead of first: prepending them in
# front of the conda libs is the suspected cause of "cannot import torch" on the server
# (mujoco_py's cymj is built with an rpath to $MUJOCO_PY_MUJOCO_PATH/bin anyway). Dirs we
# manage are de-duplicated, so sourcing this twice (run_mpc.sh after ~/mujoco_env.sh)
# does not grow LD_LIBRARY_PATH.
_ts_ld=""; _ts_ifs="$IFS"; IFS=':'
for _ts_d in ${LD_LIBRARY_PATH:-}; do
    case "$_ts_d" in
        ""|"$MUJOCO_PY_MUJOCO_PATH/bin"|"/usr/lib/nvidia") ;;
        *) _ts_ld="${_ts_ld:+$_ts_ld:}$_ts_d" ;;
    esac
done
IFS="$_ts_ifs"
if [ "${MUJOCO_LD_MODE:-prepend}" = "append" ]; then
    export LD_LIBRARY_PATH="${_ts_ld:+$_ts_ld:}$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia"
else
    export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia${_ts_ld:+:$_ts_ld}"
fi
unset _ts_ld _ts_d _ts_ifs

# A headless GPU node has no DISPLAY and nothing in the repo sets MUJOCO_GL for planning,
# so mujoco_py would default to GLFW and fail to open a window: use EGL instead. Left
# alone when DISPLAY exists (the laptop) or MUJOCO_GL is already set.
if [ -z "${DISPLAY:-}" ] && [ -z "${MUJOCO_GL:-}" ]; then
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

# Headless EGL (deformable/PyFleX planning only; harmless otherwise).
export EGL_GPU=0

# d4rl tries to import optional env stacks (mjrl/flow/carla) it doesn't find;
# suppress those import warnings.
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"

# PyFleX (deformable env, ENV=granular|rope) is NOT installed on this laptop.
if [ -d "$HOME/PyFleX" ]; then
    export PYFLEXROOT="$HOME/PyFleX"
fi

