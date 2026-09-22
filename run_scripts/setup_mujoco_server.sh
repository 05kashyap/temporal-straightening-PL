#!/usr/bin/env bash
# =============================================================================
# setup_mujoco_server.sh -- bring the planning / MPC stage up inside the project
# container: MuJoCo 2.1.2 + mujoco_py (cymj) + headless EGL, then prove it with
# run_scripts/mujoco_smoke.py.
#
# Training (`train.py`) needs NONE of this; it is for `plan.py` / `run_mpc.sh`
# and the landscape/figure stage. The script writes ~/mujoco_env.sh (planning-only
# exports, never ~/.bashrc) and then runs the smoke test in the container.
#
# Usage (login node, or inside an srun allocation):
#   bash run_scripts/setup_mujoco_server.sh                 # write env + build cymj + smoke (WRITABLE overlay)
#   bash run_scripts/setup_mujoco_server.sh --check         # read-only overlay (use while training holds it rw)
#   bash run_scripts/setup_mujoco_server.sh --no-smoke      # only (re)write ~/mujoco_env.sh
#   MUJOCO_LD_MODE=prepend bash run_scripts/setup_mujoco_server.sh   # if the loader cannot find libmujoco210.so
#
# Why writable: mujoco_py compiles its `cymj` Cython extension on first import
# into site-packages INSIDE the overlay, so the first import needs write access.
# A running training job holds that overlay read-write, which is what --check is
# for (it works once cymj exists). Anything that imports the sim also needs
# MUJOCO_GL=egl on a headless node -- nothing in the repo sets it for planning.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
SCRATCH="${SCRATCH:-/scratch/akn7847}"
OVERLAY="${OVERLAY:-$SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3}"
REPO_IN_CONTAINER="${REPO_IN_CONTAINER:-/home/akn7847/wm/temporal-straightening-PL}"
MUJOCO_DIR="${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}"
ENV_FILE="${ENV_FILE:-$HOME/mujoco_env.sh}"
CONDA_ENV="${CONDA_ENV:-/opt/miniconda/envs/ts}"
MUJOCO_LD_MODE="${MUJOCO_LD_MODE:-append}"
CKPT_ROOT="${CKPT_ROOT:-$SCRATCH/datasets/worldmodelcheckpoints}"
DATASET_DIR="${DATASET_DIR:-$SCRATCH/datasets}"

CHECK_ONLY=0
RUN_SMOKE=1
for arg in "$@"; do
    case "$arg" in
        --check)    CHECK_ONLY=1 ;;
        --no-smoke) RUN_SMOKE=0 ;;
        -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# --- host-side prerequisites -------------------------------------------------
[ -f "$SIF" ] || { echo "FATAL: container image not found: $SIF" >&2; exit 1; }
[ -f "$OVERLAY" ] || { echo "FATAL: overlay not found: $OVERLAY" >&2; exit 1; }
if [ ! -f "$MUJOCO_DIR/bin/libmujoco210.so" ]; then
    cat >&2 <<EOF
FATAL: MuJoCo 2.1.2 not found under $MUJOCO_DIR
  mkdir -p "\$HOME/.mujoco"
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "\$HOME/.mujoco"
  tar -xzf "\$HOME/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "\$HOME/.mujoco"
  ls "\$HOME/.mujoco/mujoco210/bin"      # want libmujoco210.so + libglew*.so
(An exported MUJOCO_PY_MUJOCO_PATH in one interactive shell is not enough --
 this script and ~/mujoco_env.sh set it for every planning process.)
EOF
    exit 1
fi
[ -d "$REPO/run_scripts" ] || { echo "FATAL: run this from inside the repo ($REPO)" >&2; exit 1; }

# --- ~/mujoco_env.sh: planning-only exports ---------------------------------
cat > "$ENV_FILE" <<EOF
#!/usr/bin/env bash
# Written by run_scripts/setup_mujoco_server.sh -- source it in PLANNING jobs only
# (plan.py / run_mpc.sh / landscape). Do NOT source it before train.py: it puts the
# MuJoCo libs in LD_LIBRARY_PATH, which is the suspected cause of "cannot import
# torch" when they shadow the conda ones (MUJOCO_LD_MODE=append keeps them last).
export MUJOCO_PY_MUJOCO_PATH="\${MUJOCO_PY_MUJOCO_PATH:-$MUJOCO_DIR}"
export MUJOCO_GL="\${MUJOCO_GL:-egl}"                 # headless EGL; unset = glfw/X11
export PYOPENGL_PLATFORM="\${PYOPENGL_PLATFORM:-egl}"
export EGL_GPU="\${EGL_GPU:-0}"                       # PyFleX only; harmless here
export D4RL_SUPPRESS_IMPORT_ERROR="\${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTHON="\${PYTHON:-$CONDA_ENV/bin/python}"     # run_mpc.sh's default is the laptop path
export MUJOCO_LD_MODE="\${MUJOCO_LD_MODE:-$MUJOCO_LD_MODE}"

_mj="\$MUJOCO_PY_MUJOCO_PATH/bin"
# idempotent: drop any earlier copies of the two dirs we manage, then add once
_ld=""; _old_ifs="\$IFS"; IFS=':'
for _d in \${LD_LIBRARY_PATH:-}; do
    case "\$_d" in
        ""|"\$_mj"|"/usr/lib/nvidia") ;;
        *) _ld="\${_ld:+\$_ld:}\$_d" ;;
    esac
done
IFS="\$_old_ifs"
if [ "\$MUJOCO_LD_MODE" = "prepend" ]; then
    export LD_LIBRARY_PATH="\$_mj:/usr/lib/nvidia\${_ld:+:\$_ld}"
else
    export LD_LIBRARY_PATH="\${_ld:+\$_ld:}\$_mj:/usr/lib/nvidia"
fi
unset _mj _ld _d _old_ifs
export DATASET_DIR="\${DATASET_DIR:-$DATASET_DIR}"
EOF
chmod +x "$ENV_FILE"
echo "wrote $ENV_FILE   (MUJOCO_LD_MODE=$MUJOCO_LD_MODE, MUJOCO_GL=egl, PYTHON=$CONDA_ENV/bin/python)"

# --- run the smoke test inside the container ---------------------------------
MOUNT=(--fakeroot --nv --overlay "$OVERLAY")
if [ "$CHECK_ONLY" = "1" ]; then
    MOUNT=(--fakeroot --nv --overlay "$OVERLAY:ro")
    echo "  (--check: read-only overlay, so no cymj build -- works once it is built)"
fi

echo
echo "=== container : $SIF"
echo "=== overlay   : $OVERLAY$([ "$CHECK_ONLY" = "1" ] && echo ' (:ro)')"
echo "=== env file  : $ENV_FILE"
echo "=== repo      : $REPO_IN_CONTAINER"
if [ "$RUN_SMOKE" = "1" ]; then
    set +e
    apptainer exec "${MOUNT[@]}" "$SIF" bash -lc "
        set -euo pipefail
        export PATH=/opt/miniconda/bin:\$PATH
        source /opt/miniconda/etc/profile.d/conda.sh
        conda activate ts
        source $ENV_FILE
        cd $REPO_IN_CONTAINER
        export PYTHONPATH=\$PWD\${PYTHONPATH:+:\$PYTHONPATH}
        python -c 'import sys; print(\"[container] python\", sys.executable)'
        python run_scripts/mujoco_smoke.py
    "
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: the in-container smoke test failed (exit $rc)." >&2
        echo "       Fix the FAIL stages printed above, then re-run this script." >&2
        exit "$rc"
    fi
else
    echo "  (--no-smoke: skipping the container run)"
fi

cat <<EOF

next: the planning / MPC stage can run here. Validation pass, then the real MPC:

  source $ENV_FILE                 # planning-only env (MuJoCo + EGL)
  cd $REPO_IN_CONTAINER
  export PYTHONPATH=\$PWD
  FULL=0 bash run_scripts/run_mpc.sh umaze False gd_mpc --ckpt $CKPT_ROOT/test --seeds 100
  FULL=1 bash run_scripts/run_mpc.sh umaze False gd_mpc --ckpt $CKPT_ROOT/test --seeds 100

  (FULL=0 = run_mpc.sh's validation mode: OOM check + runtime estimate. Add
   'all' as the variant argument to cover all four arms, and OL=1 for open loop.)

results : plan_outputs_gd_mpc/<model>_s100_gH25/  (logs.json, plan.log, plan_targets.pkl)
summary : plan_outputs_gd_mpc/summaries/<model>_gH25.json
tables  : python analysis/div_emb_tables.py   (needs those summaries)
See SERVER_CONTEXT.md, section 'MPC / planning stage', for the failure / fix table.
EOF
