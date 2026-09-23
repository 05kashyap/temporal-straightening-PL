#!/usr/bin/env python
"""run_scripts/patch_mujoco_py.py -- the local edits mujoco-py 2.1.2.14 needs on
this stack, applied idempotently to site-packages/mujoco_py/builder.py.

Why these exist
---------------
mujoco-py 2.1.2.14 predates GCC 14 and containerised GPU setups. Without them the
first `import mujoco_py` fails in one of three ways, all seen on NYU Torch:

  1. compile flags -- GCC >= 14 promotes `-Wincompatible-pointer-types` to a hard
     error and the generated cymj.c trips it ("passing argument 1 of '__pyx_f_...
     WrapMjVisual_global_' from incompatible pointer type"); the dev machine has
     this patch baked into its install. `-DGLEW_NO_GLU` is needed too, so bundled
     glew.h does not pull in <GL/glu.h> -> <GL/gl.h>, which this conda env lacks.
  2. builder choice -- force LinuxGPUExtensionBuilder (EGL). Upstream decides via
     `MUJOCO_PY_FORCE_CPU is None and get_nvidia_lib_dir() is not None`, and inside
     apptainer there is no `nvidia-smi`, so it silently picks the CPU builder, which
     then needs GL/osmesa.h -- and apt cannot write /var/lib/apt in this image.
  3. driver lib dir -- get_nvidia_lib_dir() must return a directory that EXISTS:
     `apptainer --nv` creates /.singularity.d/libs, and /usr/local/nvidia/lib64 can
     be created (it persists in the overlay). Otherwise the GPU branch raises
     "Missing path to your environment variable ... :None".

Idempotent and reportable: every patch is classified present/applied, --check never
writes anything, and --builder-path lets you patch a *copy* (that is how this script
is tested, so the working local env is never touched).

Usage
-----
    python run_scripts/patch_mujoco_py.py --check
    python run_scripts/patch_mujoco_py.py
    python run_scripts/patch_mujoco_py.py --builder-path /tmp/builder_copy.py --check
"""

import argparse
import os
import pathlib
import site
import sys

MARK_FLAGS = "-DGLEW_NO_GLU"
MARK_GPU = "patched: GPU/EGL"
MARK_GPU_ALT = "patched: always use the GPU"
MARK_LIB = "patched: container driver libs"

FLAG_BLOCK = '''        # GCC >= 14 promotes these to hard errors; mujoco_py 2.1.2.14's\n        # generated cymj.c trips them (built originally with GCC <= 11).\n        # GLEW_NO_GLU: bundled glew.h is self-contained for GL; without it\n        # glew.h #includes <GL/glu.h> -> <GL/gl.h>, and no gl.h exists in the\n        # conda env (no libgl1-mesa-dev equivalent).\n        try:\n            for flag in (\n                "-Wno-incompatible-pointer-types",\n                "-Wno-error=incompatible-pointer-types",\n                "-Wno-error=int-conversion",\n                "-Wno-error=implicit-function-declaration",\n                "-DGLEW_NO_GLU",\n            ):\n                if flag not in self.compiler.compiler_so:\n                    self.compiler.compiler_so.append(flag)\n        except AttributeError:\n            pass\n'''

FLAG_ANCHOR = ('        except (AttributeError, ValueError):\n'
               '            pass\n'
               '        build_ext.build_extensions(self)\n')

LIB_BLOCK = ('    # patched: container driver libs -- there is no nvidia-smi inside apptainer and\n'
             '    # no /usr/lib/nvidia; --nv provides /.singularity.d/libs and\n'
             '    # /usr/local/nvidia/lib64 can be created. Without this the GPU branch\n'
             '    # raises "Missing path to your environment variable ... :None".\n'
             '    for _cand in ("{first}", "/usr/lib/nvidia", "/.singularity.d/libs"):\n'
             '        if exists(_cand):\n'
             '            return _cand\n')

DRIVER_CANDIDATES = ("/usr/local/nvidia/lib64", "/usr/lib/nvidia", "/.singularity.d/libs")


def builder_path(explicit=None):
    if explicit:
        return pathlib.Path(explicit)
    for cand in list(site.getsitepackages()) + [site.getusersitepackages()]:
        p = pathlib.Path(cand) / "mujoco_py" / "builder.py"
        if p.is_file():
            return p
    raise SystemExit("FATAL: could not find mujoco_py/builder.py -- pass --builder-path")


def patch_flags(src):
    """GCC >= 14 / glew compile flags."""
    if MARK_FLAGS in src:
        return src, "already", "compile flags present (-Wno-incompatible-pointer-types ... -DGLEW_NO_GLU)"
    if src.count(FLAG_ANCHOR) != 1:
        return src, "failed", "anchor for the build_extensions block not found (count=%d)" % src.count(FLAG_ANCHOR)
    return src.replace(FLAG_ANCHOR, FLAG_ANCHOR.replace('        build_ext', FLAG_BLOCK + '        build_ext'), 1), "applied", "added 5 compile flags"


def patch_builder_choice(src):
    """Force the GPU/EGL builder unless MUJOCO_PY_FORCE_CPU asks for the CPU one."""
    if MARK_GPU in src or MARK_GPU_ALT in src:
        return src, "already", "GPU/EGL builder is forced"
    old = "        if os.getenv('MUJOCO_PY_FORCE_CPU') is None and get_nvidia_lib_dir() is not None:\n"
    new = ("        if os.getenv('MUJOCO_PY_FORCE_CPU') in (None, '', '0', 'false', 'False'):"
           "  # patched: GPU/EGL by default\n")
    if src.count(old) != 1:
        return src, "failed", "condition line not found (count=%d)" % src.count(old)
    return src.replace(old, new, 1), "applied", "GPU/EGL builder forced (MUJOCO_PY_FORCE_CPU=1 still selects CPU)"


def patch_lib_dir(src, first_candidate=DRIVER_CANDIDATES[0]):
    """get_nvidia_lib_dir() must return a directory that exists."""
    if MARK_LIB in src:
        return src, "already", "candidate list present"
    anchor = "def get_nvidia_lib_dir():\n"
    if src.count(anchor) != 1:
        return src, "failed", "def get_nvidia_lib_dir() not found (count=%d)" % src.count(anchor)
    return (src.replace(anchor, anchor + LIB_BLOCK.format(first=first_candidate), 1),
            "applied", "added candidate list (%s, /usr/lib/nvidia, /.singularity.d/libs)"
            % first_candidate)


def ensure_driver_dir(dry_run, candidates=DRIVER_CANDIDATES):
    """Make at least one candidate exist (best effort) and report which one."""
    for c in candidates:
        if os.path.isdir(c):
            return c, "exists: %s" % c
    target = candidates[0]
    if dry_run:
        return None, "no candidate exists yet (would create %s)" % target
    try:
        os.makedirs(target, exist_ok=True)
        return target, "created %s (persists in the container overlay)" % target
    except OSError as exc:
        return None, "could not create %s (%s)" % (target, exc)


def main():
    ap = argparse.ArgumentParser(
        description="Patch mujoco_py 2.1.2.14 for GCC >= 14 and GPU/EGL containers.")
    ap.add_argument("--builder-path", default=None,
                    help="path to builder.py (default: the installed mujoco_py)")
    ap.add_argument("--check", action="store_true",
                    help="report what is present/applied; never write")
    ap.add_argument("--no-mkdir", action="store_true",
                    help="do not create the driver-lib directory")
    args = ap.parse_args()

    p = builder_path(args.builder_path)
    src0 = p.read_text()
    print("builder.py   : %s" % p)

    src, report = src0, []
    for name, fn in (("compile flags", patch_flags),
                     ("builder choice", patch_builder_choice),
                     ("driver lib dir", patch_lib_dir)):
        src, status, detail = fn(src)
        report.append((name, status, detail))
        print("  %-15s %-8s %s" % (name, status, detail))

    driver, note = ensure_driver_dir(args.check or args.no_mkdir)
    print("  %-15s %-8s %s" % ("driver dir", "ok" if driver else "warn", note))

    if src != src0:
        if args.check:
            print("--check: would patch %s (nothing written)" % p)
        else:
            p.write_text(src)
            print("wrote %s" % p)
    else:
        print("no changes needed")

    failed = [n for n, s, _ in report if s == "failed"]
    if failed:
        print("FATAL: could not patch: %s" % ", ".join(failed))
        return 1
    print("next: rm -f <site-packages>/mujoco_py/cymj.c; "
          "rm -rf <site-packages>/mujoco_py/generated/_pyxbld_*; python -c 'import mujoco_py'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
