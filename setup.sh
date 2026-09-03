#!/usr/bin/env bash
# Laptop environment exports for this repo (user kashyap, miniconda at $HOME/miniconda3).
# Sourced by run.sh / run_mpc.sh / run_wall_ablation.sh / run_curvature_analysis.sh
# from the repo root (they `cd` to their own directory first).

# Datasets (DINO-WM download; layout: <dir>/{point_maze,point_maze_medium,pusht_noise,wall_single,...}).
export DATASET_DIR="${DATASET_DIR:-$PWD/data/datasets}"

# ts conda env python + bin (patchelf/gcc needed for mujoco_py's one-time cymj build).
TS_ENV_BIN="$HOME/miniconda3/envs/ts/bin"
export PY="${PYTHON:-$TS_ENV_BIN/python}"
if [ -d "$TS_ENV_BIN" ]; then
    export PATH="$TS_ENV_BIN${PATH:+:$PATH}"
fi

# MuJoCo 2.1.2 (mujoco210 binary dist). run.sh uses the mujoco-2.1.2 name; both are
# provided (mujoco-2.1.2 -> mujoco210 symlink is created during setup).
MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export MUJOCO_PY_MUJOCO_PATH
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

# Headless EGL (deformable/PyFleX planning only; harmless otherwise).
export EGL_GPU=0

# d4rl tries to import optional env stacks (mjrl/flow/carla) it doesn't find;
# suppress those import warnings.
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"

# PyFleX (deformable env, ENV=granular|rope) is NOT installed on this laptop.
if [ -d "$HOME/PyFleX" ]; then
    export PYFLEXROOT="$HOME/PyFleX"
fi

