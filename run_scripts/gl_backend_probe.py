#!/usr/bin/env python
"""run_scripts/gl_backend_probe.py -- why does the *env worker* fail to initialise OpenGL?

`mujoco_smoke.py` proves the offscreen render works in a single process, yet the planning
job dies inside an env worker with

    RuntimeError: Failed to initialize OpenGL
      ... mujoco_py.cymj.OffscreenOpenGLContext.__init__

The workers are forked (`env/venv.py` uses `multiprocessing.Process`, and nothing sets a
start method), so this probe renders the same way in several places and reports which work:

  1. main process                        (what the smoke test does -- the baseline)
  2. a forked child                      (what the job's env workers are today)
  3. a spawned child                     (a fresh interpreter: the candidate fix)
  4/5. the same two with bare mujoco_py  (isolates EGL+fork from the repo's env)

A child that *hangs* counts as a failure too: a process that forks after any GL use can
inherit a live X11/EGL connection and block, which is the same problem as the error.

Run it inside the container, with the same environment the job uses:

    source ~/mujoco_env.sh
    python run_scripts/gl_backend_probe.py

Exit code 0 when the recommended configuration renders everywhere it must.
"""

import ctypes.util
import glob
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BARE_XML = """
<mujoco>
  <worldbody><light pos="0 0 3"/><geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody>
</mujoco>
"""


def _render(use_repo_env: bool):
    """Build an env (repo or bare) and render once -- the exact call that fails."""
    import numpy as np
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    if use_repo_env:
        import gym
        import env  # noqa: F401  - registers point_maze
        env_obj = gym.make("point_maze")
        env_obj.reset()
        img = np.asarray(env_obj.sim.render(width=84, height=84))
    else:
        import mujoco_py
        sim = mujoco_py.MjSim(mujoco_py.MjModel.from_xml_string(BARE_XML))
        img = np.asarray(sim.render(width=64, height=64))
    return "shape=%s var=%.2f" % (img.shape, float(img.var()))


def _child(use_repo_env: bool, conn):
    try:
        conn.send(("ok", _render(use_repo_env)))
    except BaseException as exc:  # noqa: BLE001 - the probe reports everything
        conn.send(("fail", "%s: %s" % (type(exc).__name__, exc)))
    finally:
        conn.close()


def in_child(use_repo_env: bool, method: str, timeout: float = 60.0):
    """Run _render in a 'fork' or 'spawn' child; return (ok, detail)."""
    ctx = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(use_repo_env, child_conn))
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout):
            ok, detail = parent_conn.recv()
        else:
            ok, detail = False, "timed out after %.0fs" % timeout
    except EOFError:
        ok, detail = False, "child died without reporting (exit %s)" % proc.exitcode
    finally:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
    return ok, detail

def environment_report():
    print("== environment ==")
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_PY_MUJOCO_PATH", "DISPLAY",
                "MUJOCO_LD_MODE", "CUDA_VISIBLE_DEVICES"):
        print("   %-20s %s" % (key, os.environ.get(key, "<unset>")))
    print("   %-20s %s" % ("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "<unset>")))
    print("   %-20s %s" % ("libEGL", ctypes.util.find_library("EGL") or "<not found>"))
    print("   %-20s %s" % ("libGL", ctypes.util.find_library("GL") or "<not found>"))
    vendors = sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json"))
    print("   %-20s %s" % ("EGL vendors", ", ".join(os.path.basename(v) for v in vendors)
                           or "<none: the NVIDIA EGL vendor is not registered>"))
    for path in vendors:
        try:
            with open(path) as handle:
                print("      %s: %s" % (os.path.basename(path), handle.read().strip()))
        except OSError:
            pass


def main() -> int:
    environment_report()

    print("\n== 1. main process (the smoke test's path) ==")
    try:
        print("   OK   %s" % _render(use_repo_env=True))
        main_ok = True
    except BaseException as exc:  # noqa: BLE001
        print("   FAIL %s: %s" % (type(exc).__name__, exc))
        main_ok = False

    results = {}
    for method in ("fork", "spawn"):
        for use_repo in (True, False):
            label = "%s child, %s" % (method, "repo env" if use_repo else "bare mujoco_py")
            print("\n== %s ==" % label)
            ok, detail = in_child(use_repo, method)
            print("   %s %s" % ("OK  " if ok else "FAIL", detail))
            results[(method, use_repo)] = ok

    print("\n== verdict ==")
    if main_ok and results[("fork", True)] and results[("spawn", True)]:
        print("   every configuration renders: no start-method change is needed.")
        print("   (If the job still fails, send this whole output with the job log.)")
        return 0
    if results[("fork", True)] and not results[("spawn", True)]:
        print("   fork works, spawn does not: keep the default (no TS_ENV_START_METHOD).")
        return 0
    if not results[("fork", True)] and results[("spawn", True)]:
        print("   fork fails, spawn works -- this is the EGL/fork problem. Run the job with:")
        print("       TS_ENV_START_METHOD=spawn OVERLAY_RW=1 FULL=0 sbatch run_scripts/mpc_server.sh")
        if not results[("fork", False)]:
            print("   (bare mujoco_py fails in a forked child too: it is EGL + fork, not the")
            print("    repo's env.)")
        return 0
    print("   neither fork nor spawn can initialise OpenGL in a child, so the EGL stack")
    print("   itself is the problem. The alternative is the CPU/OSMesa backend, which needs")
    print("   cymj rebuilt with MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=osmesa (SERVER_CONTEXT 11.4).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
