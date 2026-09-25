#!/usr/bin/env python
"""run_scripts/mujoco_smoke.py -- in-container acceptance test for the *planning*
stage (MuJoCo + gym + EGL). Run it inside the project container:

    source ~/mujoco_env.sh
    python run_scripts/mujoco_smoke.py

Why this exists
---------------
`train.py` (world-model training) needs torch and nothing else. `plan.py` /
`run_mpc.sh` additionally need the simulator, which is pulled in on a completely
different path:

    plan.py -> gym.make("point_maze")            (registered in env/__init__.py)
            -> env.pointmaze.PointMazeWrapper
            -> env.pointmaze.maze_model.MazeEnv  (gym.envs.mujoco.mujoco_env)
            -> mujoco_py -> the MuJoCo 2.1.2 binaries in $MUJOCO_PY_MUJOCO_PATH

Each stage fails in its own way on a fresh machine, so they are checked one at a
time and the failure is reported with the fix that applies:

  1. environment     the variables the other stages need (reported, never fatal)
  2. torch           import AFTER the MuJoCo paths are in LD_LIBRARY_PATH -- the
                     failure mode where mujoco's libs shadow conda's, which is
                     why run_scripts/setup.sh supports MUJOCO_LD_MODE=append
  3. mujoco_py       the first import compiles the `cymj` Cython extension, which
                     writes into site-packages -> needs a WRITABLE overlay
                     (apptainer ... --overlay "$OVERLAY", not ":ro") plus a
                     compiler and patchelf (both come from environment.yaml)
  4. libmujoco210    the dynamic loader can actually find the MuJoCo .so
  5. gym envs        `import env` registers point_maze / point_maze_medium /
                     pusht / wall
  6. sim step        gym.make + reset + step really runs the physics
  7. offscreen render  EGL (headless) or GLFW (X11) render to an array; the step
                     that fails on a GPU node without MUJOCO_GL=egl
  8. planning inputs  DATASET_DIR / checkpoint dirs for a real run (warn only)

Exit code 0 only if every required stage passed.
"""

import os
import pathlib
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

STAGES = []      # (name, ok, required, detail)
STATE = {}       # values shared between stages


def check(name, fn, required=True, hints=()):
    """Run one stage, print its output and record the result."""
    print("=" * 78)
    print(">> %s" % name)
    try:
        detail = fn()
    except BaseException as exc:  # noqa: BLE001 - a smoke test reports everything
        tb = traceback.format_exc(limit=8)
        lines = tb.splitlines()
        # a failed cymj compile dumps hundreds of lines: keep the tail
        print(tb if len(lines) <= 30 else "\n".join(lines[-22:]))
        last = tb.strip().splitlines()[-1]
        blob = (tb + str(exc)).lower()
        extra = []
        if "read-only file system" in blob or "permission denied" in blob:
            extra.append("mujoco_py needs a WRITABLE overlay on every import, not only the "
                         "first: it takes a write lock next to cymj*.so "
                         "(mujoco_py/generated/mujocopy-buildlock) before it even looks at "
                         "the cache, so ':ro' fails here even with cymj already built")
            extra.append("with run_scripts/mpc_server.sh that is OVERLAY_RW=1; otherwise "
                         "drop ':ro' from the --overlay flag -- and do not run two jobs "
                         "against the same writable overlay at once")
        if ("mujoco210" in blob or "libmujoco210" in blob
                or "cannot open shared object" in blob):
            extra.append("check MUJOCO_PY_MUJOCO_PATH and that LD_LIBRARY_PATH "
                         "contains $MUJOCO_PY_MUJOCO_PATH/bin "
                         "(try MUJOCO_LD_MODE=prepend)")
        if "glew" in blob or "gl.h" in blob or "egl" in blob:
            extra.append("GL/EGL headers or libs missing: environment.yaml installs "
                         "glew / xorg-libx11 / xorg-xorgproto for this; the EGL "
                         "runtime needs /usr/lib/nvidia on LD_LIBRARY_PATH and "
                         "MUJOCO_GL=egl")
        print("   FAIL: %s" % name)
        for h in list(hints) + extra:
            print("   hint: %s" % h)
        STAGES.append((name, False, required, last))
        return None
    print("   OK  : %s" % name)
    STAGES.append((name, True, required, detail if isinstance(detail, str) else ""))
    return detail


def stage_environment():
    """Report (never fail) the variables the rest of the test depends on."""
    keys = ["MUJOCO_PY_MUJOCO_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_GPU",
            "LD_LIBRARY_PATH", "DISPLAY", "CUDA_VISIBLE_DEVICES", "DATASET_DIR",
            "MUJOCO_LD_MODE", "PYTHON", "MUJOCO_PY_FORCE_CPU", "TS_ENV_START_METHOD",
            "DATA_ROOT", "CKPT_ROOT"]
    for k in keys:
        print("   %-22s %s" % (k, os.environ.get(k, "<unset>")))
    mj = os.environ.get("MUJOCO_PY_MUJOCO_PATH",
                        os.path.expanduser("~/.mujoco/mujoco210"))
    print("   %-22s %s" % ("MUJOCO_PY_MUJOCO_PATH/bin",
                           "present" if os.path.isdir(os.path.join(mj, "bin"))
                           else "MISSING"))
    print("   %-22s %s" % ("python", sys.executable))
    print("   %-22s %s / %s" % ("sys.prefix / CONDA_PREFIX", sys.prefix,
                                   os.environ.get("CONDA_PREFIX", "<unset>")))
    return "mujoco=%s MUJOCO_GL=%s" % (mj, os.environ.get("MUJOCO_GL", "<unset>"))


def stage_torch():
    import torch
    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "<none>"
    return "torch %s cuda=%s (%s)" % (torch.__version__, cuda, name)


def stage_mujoco_py():
    import mujoco_py
    print("   mujoco_py %s from %s" % (mujoco_py.__version__,
                                       os.path.dirname(mujoco_py.__file__)))
    try:
        import mujoco_py.cymj as cymj
        print("   cymj      %s" % getattr(cymj, "__file__", "<builtin>"))
    except Exception as exc:  # diagnostics only
        print("   cymj      not importable: %s" % exc)
    return "mujoco_py %s" % mujoco_py.__version__


def stage_load_lib():
    """Informational: how the loader sees libmujoco210.so.

    A plain ctypes.CDLL("libmujoco210.so") fails on this stack with an undefined
    symbol for glewBindBuffer: the MuJoCo library references glew symbols that live
    in the sibling libglewegl.so / libglewosmesa.so, and mujoco_py's cymj extension
    is what links them. So this is NOT a failure by itself; it is reported together
    with the LD_PRELOAD that makes it work, because that is the standard fix if the
    cymj import ever raises the same symbol.
    """
    import ctypes
    mj = os.environ.get("MUJOCO_PY_MUJOCO_PATH",
                        os.path.expanduser("~/.mujoco/mujoco210"))
    try:
        ctypes.CDLL("libmujoco210.so")
        return "libmujoco210.so loads directly"
    except OSError as exc:
        print("   plain load failed: %s" % str(exc).strip().splitlines()[-1])
    for name in ("libglewegl.so", "libglewosmesa.so", "libglew.so"):
        cand = os.path.join(mj, "bin", name)
        if not os.path.isfile(cand):
            continue
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL("libmujoco210.so")
            print("   preloading %s makes it load -> if the cymj import fails with"
                  " an undefined glew symbol, use LD_PRELOAD=%s" % (cand, cand))
            return "loads with LD_PRELOAD=%s" % cand
        except OSError:
            continue
    raise RuntimeError("libmujoco210.so did not load even with the sibling glew "
                       "libraries present")


def stage_toolchain():
    """Informational: what the cymj build will use.

    cymj is compiled on first import, so these decide whether that succeeds: Cython
    must be 0.29.x (mujoco-py 2.1.2.14 predates Cython 3 -- environment.yaml pins
    0.29.37), gcc >= 14 turns -Wincompatible-pointer-types into a hard error (hence the
    patch below), and the builder that gets selected depends on the driver-lib
    directory, NOT on MUJOCO_GL.
    """
    import re
    import subprocess
    try:
        cy = subprocess.check_output(
            [sys.executable, "-c", "import Cython; print(Cython.__version__)"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        cy = "MISSING"
    gcc = subprocess.run("gcc -dumpversion", shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout.decode().strip() or "?"
    on_path = subprocess.call("type nvidia-smi", shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    print("   python         %s" % sys.executable)
    print("   Cython         %s" % cy)
    print("   gcc            %s" % gcc)
    print("   nvidia-smi     %s" % ("on PATH" if on_path else
                                     "absent (fine with the builder patch below)"))
    for c in ("/usr/local/nvidia/lib64", "/usr/lib/nvidia", "/.singularity.d/libs"):
        print("   %-14s %s" % (c, "exists" if os.path.isdir(c) else "missing"))
    if cy != "MISSING" and int(cy.split(".")[0]) >= 3:
        raise RuntimeError("Cython %s is too new for mujoco-py 2.1.2.14: "
                           "pip install cython==0.29.37, then remove cymj.c + "
                           "generated/_pyxbld_* and import again" % cy)
    m = re.match(r"(\d+)", gcc)
    if m and int(m.group(1)) >= 14:
        return "Cython %s, gcc %s (needs the -Wno-incompatible-pointer-types patch)" % (cy, gcc)
    return "Cython %s, gcc %s" % (cy, gcc)


def stage_patches():
    """Are run_scripts/patch_mujoco_py.py's edits present in site-packages?

    Required on GCC >= 14 and inside the container; where mujoco_py already imports
    (nvidia-smi present, gcc <= 13) a missing patch is only a warning.
    """
    import mujoco_py
    b = pathlib.Path(mujoco_py.__file__).parent / "builder.py"
    src = b.read_text()
    have = {
        "compile flags": "-DGLEW_NO_GLU" in src,
        "gpu builder": ("patched: GPU/EGL" in src) or ("patched: always use the GPU" in src),
        "driver lib dir": "patched: container driver libs" in src,
    }
    for k, v in have.items():
        print("   %-16s %s" % (k, "present" if v else "MISSING"))
    if not any(have.values()):
        raise RuntimeError("mujoco_py has none of the local patches -- run "
                           "'python run_scripts/patch_mujoco_py.py' (see its --help)")
    missing = [k for k, v in have.items() if not v]
    if missing:
        print("   note: %s not applied -- run run_scripts/patch_mujoco_py.py if the "
              "import below picks the CPU builder" % ", ".join(missing))
    return "patches: %s" % ", ".join("%s=%s" % (k, "yes" if v else "no")
                                      for k, v in have.items())


def stage_gym_envs():
    """`import env` registers the ids plan.py / conf/env/*.yaml ask for."""
    import gym
    import env  # noqa: F401  - registers point_maze / point_maze_medium / ...
    wanted = ["point_maze", "point_maze_medium", "pusht", "wall"]
    have = []
    for env_id in wanted:
        try:
            gym.spec(env_id)
            have.append(env_id)
        except Exception:
            pass
    print("   registered: %s" % ", ".join(have))
    if "point_maze" not in have:
        raise RuntimeError("gym.make('point_maze') will not resolve -- is `import "
                           "env` working and PYTHONPATH set to the repo root?")
    return "point_maze registered"


def stage_make_env():
    """gym.make + reset + step: the physics really runs."""
    import gym
    env = gym.make("point_maze")
    env.reset()
    action = env.action_space.sample() * 0.0     # zero action: no divergence
    obs, rew, done, info = env.step(action)
    qpos = [round(float(v), 4) for v in env.sim.data.qpos[:2]]
    print("   action_space %s" % env.action_space)
    print("   sim qpos     %s" % qpos)
    print("   step         rew=%.4f done=%s" % (float(rew), bool(done)))
    STATE["env"] = env
    return "point_maze steps (qpos=%s)" % qpos


def stage_render():
    """Offscreen render: EGL on a headless node, GLFW when DISPLAY is set."""
    env = STATE.get("env")
    if env is None:
        raise RuntimeError("no env from the previous stage")
    import numpy as np
    img = np.asarray(env.sim.render(width=84, height=84))
    var = float(img.var())
    mode = os.environ.get("MUJOCO_GL") or ("glfw (DISPLAY)"
                                           if os.environ.get("DISPLAY") else "<unset>")
    print("   image        shape=%s dtype=%s var=%.2f" % (img.shape, img.dtype, var))
    if var <= 0.0:
        raise RuntimeError("render came back blank (all pixels equal)")
    return "offscreen render %s via %s" % (img.shape, mode)


# Per-env data layout, mirroring run_scripts/dataset_paths.sh -- the single source of
# truth shared with train_server.sh / run_mpc.sh. The three datasets are NOT nested the
# same way, and plan.py asks for $DATASET_DIR/<env>, so DATASET_DIR differs per env
# (which is why one DATASET_DIR cannot serve all of them).
LAYOUT = {
    "point_maze": ("point_maze/point_maze",
                   ["states.pth", "actions.pth", "seq_lengths.pth"], ["obses"], []),
    "point_maze_medium": ("point_maze_medium",
                          ["states.pth", "actions.pth", "seq_lengths.pth"], ["obses"], []),
    "pusht_noise": ("pusht/pusht_noise", ["states.pth", "seq_lengths.pkl"], ["obses"],
                    ["rel_actions.pth", "abs_actions.pth"]),
}


def _dataset_ok(path, required, dirs, any_of):
    """True when `path` is the directory the dataset loader actually reads."""
    if not all(os.path.isfile(os.path.join(path, f)) for f in required):
        return False
    if any_of and not any(os.path.isfile(os.path.join(path, f)) for f in any_of):
        return False
    return all(os.path.isdir(os.path.join(path, d)) for d in dirs)


def stage_planning_inputs():
    """Warn-only: the datasets a real plan.py run needs, per env, plus checkpoints.

    plan.py reads `$DATASET_DIR/<env>` via datasets/*_dset.py, which loads states.pth
    etc. plus an obses/ directory (pusht additionally rel_actions.pth|abs_actions.pth
    and seq_lengths.pkl). DATASET_DIR is the *parent* of the data directory and is
    therefore per env -- the value to export is printed next to each env below.
    """
    missing = []
    root = os.environ.get("DATA_ROOT")
    if not root:
        missing.append("DATA_ROOT is unset (dataset_paths.sh default: "
                       "$SCRATCH/datasets/worldmodeldata)")
    elif not os.path.isdir(root):
        missing.append("DATA_ROOT=%s does not exist" % root)
    else:
        print("   DATA_ROOT  %s" % root)
        for env_name, (rel, required, dirs, any_of) in LAYOUT.items():
            data = os.path.join(root, rel)
            if _dataset_ok(data, required, dirs, any_of):
                print("   %-18s ok      DATASET_DIR=%s" % (env_name, os.path.dirname(data)))
            elif os.path.isdir(data):
                have = ", ".join(sorted(os.listdir(data))[:5])
                print("   %-18s incomplete" % env_name)
                missing.append("%s lacks %s (has: %s)"
                               % (data, ", ".join(required + dirs), have))
            else:
                print("   %-18s MISSING %s" % (env_name, data))
                missing.append("%s not found -- check DATA_ROOT" % data)
    # Resolve the DIRECTORY THAT HOLDS THE RUN DIRS. mpc_server.sh exports CKBPT_PATH (what it
    # passes to run_mpc.sh as --ckpt, i.e. $CKPT_ROOT/test), so that wins; then CKBPT; then
    # $CKPT_ROOT/test when it exists. Probing bare $CKPT_ROOT counts test/ and logs/ as "arm
    # dirs" and reports "0 with model_latest.pth" on a perfectly healthy layout.
    ckpt = os.environ.get("CKBPT_PATH") or os.environ.get("CKBPT") or ""
    if not ckpt:
        _root_ck = os.environ.get("CKPT_ROOT")
        _test_ck = os.path.join(_root_ck, "test") if _root_ck else ""
        ckpt = _test_ck if (_test_ck and os.path.isdir(_test_ck)) else (_root_ck or "checkpoints/test")
    if not os.path.isdir(ckpt):
        missing.append("no checkpoint dir at %s (run_mpc.sh needs --ckpt $CKPT_ROOT/test)" % ckpt)
    else:
        arms = sorted(d for d in os.listdir(ckpt)
                      if os.path.isdir(os.path.join(ckpt, d)))
        ready = [d for d in arms
                 if os.path.isfile(os.path.join(ckpt, d, "checkpoints", "model_latest.pth"))]
        print("   checkpoints  %s: %d arm dirs, %d with model_latest.pth"
              % (ckpt, len(arms), len(ready)))
        if not ready and arms:
            missing.append("no <run dir>/checkpoints/model_latest.pth under %s (arm dirs: %s)"
                           % (ckpt, ", ".join(arms[:3]) + (", ..." if len(arms) > 3 else "")))
        elif not arms:
            missing.append("no run (arm) dirs under %s" % ckpt)
    for m in missing:
        print("   WARN  %s" % m)
    return ("warnings: %d" % len(missing)) if missing else "datasets and checkpoints look right"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MuJoCo/gym/EGL acceptance test for the planning stage.")
    parser.add_argument("--skip-render", action="store_true",
                        help="skip the offscreen render stage (e.g. on a GPU-less "
                             "login node)")
    args = parser.parse_args()

    print("mujoco smoke test  repo=%s  python=%s" % (REPO, sys.executable))
    check("environment (reported)", stage_environment)
    check("toolchain (Cython / gcc / driver dirs)", stage_toolchain)
    check("mujoco_py patches applied", stage_patches, required=False)
    check("torch import, with the MuJoCo paths set", stage_torch,
          hints=["if this fails with a stdc++/GL symbol error, the MuJoCo bin dir "
                 "is shadowing conda's libs -> export MUJOCO_LD_MODE=append"])
    check("mujoco_py import (builds cymj on first use)", stage_mujoco_py,
          hints=["a compile error means no compiler/patchelf/GL headers in the env, "
                 "or the overlay is read-only: mujoco_py takes a write lock in "
                 "mujoco_py/generated/ on EVERY import, so ':ro' always fails here "
                 "(use OVERLAY_RW=1)",
                 "an undefined-symbol error for glewBindBuffer needs "
                 "LD_PRELOAD=<glew .so> (the loader stage prints which one)"])
    check("libmujoco210.so loader (informational)", stage_load_lib,
          required=False)
    check("gym envs registered (import env)", stage_gym_envs)
    check("point_maze make/reset/step", stage_make_env)
    if args.skip_render:
        print(">> offscreen render skipped (--skip-render)")
    else:
        check("offscreen render (EGL or GLFW)", stage_render,
              hints=["export MUJOCO_GL=egl + PYOPENGL_PLATFORM=egl on a headless "
                     "node and keep /usr/lib/nvidia on LD_LIBRARY_PATH"])
    check("planning inputs (warn only)", stage_planning_inputs, required=False)

    print("=" * 78)
    print("=== mujoco smoke summary ===")
    failed = 0
    for name, ok, required, detail in STAGES:
        tag = "PASS" if ok else ("FAIL" if required else "warn")
        print(("  %-5s %-44s %s" % (tag, name, detail)).rstrip())
        if not ok and required:
            failed += 1
    if failed:
        print("\n%d required stage(s) failed -- fix those before run_mpc.sh." % failed)
        return 1
    print("\nall required stages passed: the planning/MPC stage can run here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
