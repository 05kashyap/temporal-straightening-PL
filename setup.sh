export DATASET_DIR=/home/shanveen-ortho-clinic/Documents/Projects/temporal-straightening-PL/data/datasets

# --- PyFleX / deformable planning (env=deformable_env, ENV=granular|rope) ----
# pyflex (the SoftGym/Flex backend) is a prebuilt module in the 'ts' env; it
# needs PYFLEXROOT (scene asset paths) and the EGL device platform to render
# headlessly. mujoco210 is required just to import the 'env' package (pointmaze
# imports mujoco_py at package import time, even for deformable planning).
export PYFLEXROOT=/home/shanveen-ortho-clinic/PyFleX
export EGL_GPU=0  # headless EGL: use the first NVIDIA device directly (no X server)
if [ -d "$HOME/.mujoco/mujoco210" ]; then
    export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
fi

