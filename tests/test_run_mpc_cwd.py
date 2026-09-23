"""Regression tests for run_scripts/run_mpc.sh's working directory and exit status.

Two bugs from the first real `FULL=0` run on the cluster (slurm-mpc-18336008):

  * `run_plan ... > plan_outputs_gd_mpc/validate_<model>_setup.log` opened the log before
    run_plan's own mkdir ran, so on a fresh checkout *every* env failed with
    "No such file or directory", was skipped, and the wrapper still printed `[ok]`.
  * the script `cd`'d into run_scripts/ and never came back, so `"$PY" plan.py` looked for
    plan.py in the wrong directory and the run dirs were created under run_scripts/.

Both are checked here with a stub interpreter, on a throwaway copy of the repo so nothing
is written into the checkout. No GPU, no datasets, no plan.py execution.

Run with either:
    python tests/test_run_mpc_cwd.py
    pytest tests/test_run_mpc_cwd.py        # from the repo root (ts env)
"""

import os
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def make_repo_copy(tmp_path, archive_from="HEAD"):
    """A pristine copy of the repo (git archive) with run_scripts/ from the working tree."""
    dest = tmp_path / "repo"
    dest.mkdir()
    tar = subprocess.run(["git", "archive", archive_from], cwd=str(REPO),
                         capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
    # the scripts under test come from the working tree, so this works before a commit too
    shutil.rmtree(dest / "run_scripts")
    shutil.copytree(REPO / "run_scripts", dest / "run_scripts")
    return dest


def stub_interpreter(path, fail_plan_py):
    """A fake ts interpreter: the real run_mpc.sh only needs *something* executable."""
    body = "#!/bin/bash\n"
    if fail_plan_py:
        body += 'case "$*" in *plan.py*) echo "[stub] plan.py fails"; exit 1 ;; esac\n'
    body += "exit 0\n"
    path.write_text(body)
    path.chmod(0o755)
    return path


def run_case(tmp_path, fail_plan_py):
    repo = make_repo_copy(tmp_path)
    stub = stub_interpreter(tmp_path / "python_stub", fail_plan_py)
    (tmp_path / "ck" / "a").mkdir(parents=True)
    (tmp_path / "data" / "point_maze" / "point_maze").mkdir(parents=True)
    env = dict(os.environ, DATA_ROOT=str(tmp_path / "data"), PYTHON=str(stub),
               ARM_NAMES="a b c d", FULL="0")
    proc = subprocess.run(["bash", "run_scripts/run_mpc.sh", "umaze", "False", "gd_mpc",
                           "--ckpt", str(tmp_path / "ck")],
                          cwd=str(repo), env=env, capture_output=True, text=True, timeout=300)
    return repo, proc.stdout + proc.stderr, proc.returncode


def test_output_dirs_are_created_in_the_repo_root(tmp_path):
    repo, out, rc = run_case(tmp_path, fail_plan_py=False)
    assert "No such file or directory" not in out, out
    assert (repo / "plan_outputs_gd_mpc" / "validate_a_setup.log").exists(), out
    assert not (repo / "run_scripts" / "plan_outputs_gd_mpc").exists(), \
        "outputs must not be created under run_scripts/"
    assert rc == 0, out


def test_the_driver_runs_plan_py_from_the_repo_root(tmp_path):
    """The stub logs its own cwd, which is how the wrong-directory bug was found."""
    repo = make_repo_copy(tmp_path)
    stub = tmp_path / "python_stub"
    stub.write_text('#!/bin/bash\necho "STUB cwd=$PWD args=$*"\nexit 0\n')
    stub.chmod(0o755)
    (tmp_path / "ck" / "a").mkdir(parents=True)
    (tmp_path / "data" / "point_maze" / "point_maze").mkdir(parents=True)
    env = dict(os.environ, DATA_ROOT=str(tmp_path / "data"), PYTHON=str(stub),
               ARM_NAMES="a b c d", FULL="0")
    subprocess.run(["bash", "run_scripts/run_mpc.sh", "umaze", "False", "gd_mpc",
                    "--ckpt", str(tmp_path / "ck")],
                   cwd=str(repo), env=env, capture_output=True, text=True, timeout=300)
    log = (repo / "plan_outputs_gd_mpc" / "validate_a_setup.log").read_text()
    cwd = log.split("STUB cwd=", 1)[1].split(" ", 1)[0]
    assert pathlib.Path(cwd).resolve() == repo.resolve(), \
        "plan.py was started from %s, not the repo root" % cwd
    assert (pathlib.Path(cwd) / "plan.py").exists(), "plan.py must be reachable from there"


def test_a_failed_run_makes_the_script_exit_non_zero(tmp_path):
    repo, out, rc = run_case(tmp_path, fail_plan_py=True)
    assert "No such file or directory" not in out, out
    assert "!! setup run rc=" in out, out          # the failure is named ...
    assert "at least one run failed" in out, out   # ... and not swallowed
    assert rc != 0, "a failed validate run must not exit 0 (the wrapper would print [ok])"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
