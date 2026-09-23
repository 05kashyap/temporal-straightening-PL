#!/usr/bin/env bash
# =============================================================================
# selftest_mpc_server.sh -- verify run_scripts/mpc_server.sh WITHOUT a GPU, a
#                           container or the datasets. Safe on the login node.
#
# Why this exists: the wrapper generates the in-container script in two pieces -- a
# preamble of `export VALUE=...` lines, then the body that uses those values. If the
# preamble ends up after the body's first use of them, the job dies *inside* the
# container, e.g.
#
#     /home/akn7847/.ts_mpc_body.sh: line 7: ENV_FILE: unbound variable
#
# which costs a queue slot instead of seconds. This script reproduces that class by
# running the real wrapper with apptainer, the container's conda and run_mpc.sh
# replaced by stubs, and checks:
#
#   1. the wrapper's own guard accepts the generated body (and the preamble is first)
#   2. a full invocation runs end to end: conda -> env file -> repo -> log dir ->
#      per-job loop -> summary line (rc=0, success_rate parsed)
#   3. the *historical* layout (preamble appended after the body) still fails, so
#      this test would notice if the bug came back
#   4. the preflight gate stops the run when the smoke test fails
#   5. a missing ENV_FILE is reported before apptainer starts
#   6. a checkout outside $HOME is refused, and ALLOW_OUTSIDE_HOME=1 overrides
#   7. LIVE=1 additionally starts the real image (fakeroot + overlay + the ts env)
#   2b. a wrong CONTAINER_CONDA is named, not a cascade of "python not found"
#
# Check 0 is a gate: it proves the stubs are the binaries that will be used, and 0b
# proves the wrapper takes the stub from APPTAINER_BIN even when PATH prefers another
# apptainer -- which is exactly what happens on a login node whose bash startup files
# rewrite PATH. On a
# node where the temp dir cannot be exec (noexec /tmp) bash would silently skip
# them and run the cluster's apptainer, so this script would report failures that
# have nothing to do with the wrapper. It probes for an exec-capable directory
# (falling back to $HOME) and stops if the stubs are not what runs.
#
#   bash run_scripts/selftest_mpc_server.sh        # one PASS/FAIL per check, exit code
#   LIVE=1 bash run_scripts/selftest_mpc_server.sh # + start the real container

#   KEEP=1 bash run_scripts/selftest_mpc_server.sh # keep the temp dir (prints it)
#
# No pipe into `head`/`grep -q` anywhere: this file runs under `set -o pipefail`,
# where an early exit upstream gives SIGPIPE and fails the pipeline at random.
# =============================================================================
set -uo pipefail

REPO_HOST_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REAL_APPTAINER="$(command -v apptainer 2>/dev/null || true)"   # before any stub exists

# The stubs must be executables that really run. Where /tmp is mounted noexec, bash skips a
# stub it cannot execute and uses the next apptainer on PATH -- the cluster's own, which
# then rejects the empty stub image and makes this script report failures that have nothing
# to do with the wrapper. So probe for a directory that allows execution, and fall back to
# $HOME if the temp dir cannot do it.
_make_temp() {
    local base="$1" d
    d="$(mktemp -d "$base/ts_selftest.XXXXXX" 2>/dev/null)" || return 1
    printf '#!/bin/bash\nexit 0\n' > "$d/.exectest" 2>/dev/null || { rm -rf "$d"; return 1; }
    chmod +x "$d/.exectest" 2>/dev/null
    if "$d/.exectest" 2>/dev/null; then printf '%s\n' "$d"; return 0; fi
    rm -rf "$d"
    return 1
}
TEMP_NOTE="in ${TMPDIR:-/tmp}"
T="$(_make_temp "${TMPDIR:-/tmp}")" || {
    T="$(_make_temp "$HOME")" || {
        echo "FATAL: nowhere to put an executable stub (tried ${TMPDIR:-/tmp} and $HOME)." >&2
        echo "       If the stubs cannot run, the cluster's own apptainer is used and this test" >&2
        echo "       measures the wrong thing. Run it from a filesystem that allows exec." >&2
        exit 1
    }
    TEMP_NOTE="under \$HOME (/tmp cannot execute a script here)"
}
KEEP="${KEEP:-0}"
trap '[ "$KEEP" = "1" ] || rm -rf "$T"' EXIT

pass=0; fail=0
ok() { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
no() { printf '  FAIL  %s\n' "$1"; printf '        | %s\n' "${2:-<no output>}" | tail -10; fail=$((fail + 1)); }

# --- stubs: apptainer, the container conda, run_mpc.sh, python -------------------------
mkdir -p "$T/bin" "$T/ck/test" "$T/miniconda/etc/profile.d" "$T/spool" "$T/miniconda_badpython/etc/profile.d" "$T/miniconda_badpython/bin"
: > "$T/fake.sif"
: > "$T/fake.overlay"
printf 'conda() { return 0; }\nPATH=%s/bin:$PATH\n' "$T/miniconda" > "$T/miniconda/etc/profile.d/conda.sh"
mkdir -p "$T/miniconda/bin"
printf '#!/bin/bash\necho 3.9.23\n' > "$T/miniconda/bin/python"
chmod +x "$T/miniconda/bin/python"
printf '#!/bin/bash\nexport MUJOCO_PATH=%s\n' "$T/miniconda" > "$T/mujoco_env.sh"
cat > "$T/bin/apptainer" <<'STUB'
#!/bin/bash
# stands in for: apptainer exec --fakeroot --nv --bind X --overlay Y <image> <cmd...>
if [ "${1:-}" = "--ts-selftest" ]; then echo STUB_APPTAINER_OK; exit 0; fi
args=("$@"); i=0
while [ "$i" -lt "${#args[@]}" ]; do
    case "${args[$i]}" in
        exec|run) i=$((i + 1)); continue ;;
        --bind|--overlay|-B|--env) i=$((i + 2)); continue ;;
        --*) i=$((i + 1)); continue ;;
        *) break ;;
    esac
done
i=$((i + 1))
exec "${args[@]:$i}"
STUB
cat > "$T/bin/bash" <<'STUB'
#!/bin/bash
if [ "${1:-}" = "--ts-selftest" ]; then echo STUB_BASH_OK; exit 0; fi
case "$*" in
    *run_mpc.sh*) echo "[stub run_mpc.sh] $*"; echo "success_rate=0.99"; exit 0 ;;
esac
exec /bin/bash "$@"
STUB
# a second container conda whose python fails, for the preflight check (the body prepends
# $CONTAINER_CONDA/bin to PATH, so the stub has to live in there)
printf 'conda() { return 0; }\nPATH=%s/bin:$PATH\n' "$T/miniconda_badpython" > "$T/miniconda_badpython/etc/profile.d/conda.sh"
printf '#!/bin/bash\necho "[stub python] mujoco is broken" >&2\nexit 1\n' > "$T/miniconda_badpython/bin/python"
chmod +x "$T/bin/apptainer" "$T/bin/bash" "$T/miniconda_badpython/bin/python"
# tell the caller where the temp dir is (and why), before the first check
TEMP_NOTE="${TEMP_NOTE:-}"

# --- the wrapper exactly as sbatch sees it: a spool copy, not the checkout file --------
cp "$REPO_HOST_SELF/run_scripts/mpc_server.sh" "$T/spool/slurm_script"
WRAP="$T/spool/slurm_script"
export SLURM_SUBMIT_DIR="$REPO_HOST_SELF"
export BODY="$T/body.sh" CKPT_ROOT="$T/ck" CKBPT="$T/ck/test" SIF="$T/fake.sif" \
       OVERLAY="$T/fake.overlay" CONTAINER_CONDA="$T/miniconda" ENV_FILE="$T/mujoco_env.sh"
# the simulated container sees the whole filesystem, so point the body at this
# checkout wherever it lives and acknowledge the (stubbed) mount
export REPO_IN_CONTAINER="$REPO_HOST_SELF" ALLOW_OUTSIDE_HOME=1
# PATH order is not trustworthy (site startup files can prepend their own apptainer
# inside any bash process, the stub included), so hand the wrapper the stub directly
export APPTAINER_BIN="$T/bin/apptainer"

echo "repo : $REPO_HOST_SELF"
echo "temp : $T  $TEMP_NOTE"
echo "real : apptainer=${REAL_APPTAINER:-<none on PATH>}  # what LIVE=1 would start"


echo
echo "0. the stubs are the executables that will run (gate)"
_stub_apt="$(PATH="$T/bin:$PATH" "$T/bin/apptainer" --ts-selftest 2>&1 || true)"
_stub_sh="$(PATH="$T/bin:$PATH" "$T/bin/bash" --ts-selftest 2>&1 || true)"
_res_apt="$(PATH="$T/bin:$PATH" command -v apptainer 2>/dev/null || true)"
_res_sh="$(PATH="$T/bin:$PATH" command -v bash 2>/dev/null || true)"
if [ "$_stub_apt" = "STUB_APPTAINER_OK" ] && [ "$_stub_sh" = "STUB_BASH_OK" ] \
   && [ "$_res_apt" = "$T/bin/apptainer" ] && [ "$_res_sh" = "$T/bin/bash" ]; then
    ok "apptainer -> $_res_apt, bash -> $_res_sh (both stubs execute)"
else
    printf '  FAIL  the stubs would NOT be used: apptainer=%s (%s) bash=%s (%s)\n' \
           "$_res_apt" "$_stub_apt" "$_res_sh" "$_stub_sh"
    printf '        Without them this test would start the real apptainer and the real\n'
    printf '        run_mpc.sh, so its results would mean nothing. Stopping here.\n'
    printf '        temp dir: %s -- noexec filesystem? KEEP=1 keeps it for inspection.\n' "$T"
    exit 1
fi

echo
echo
echo "0b. a decoy apptainer earlier in PATH must not win (this is what the cluster did)"
mkdir -p "$T/decoy"
cat > "$T/decoy/apptainer" <<'STUB'
#!/bin/bash
echo DECOY_APPTAINER_RAN >&2
exit 251
STUB
chmod +x "$T/decoy/apptainer"
_decoy_res="$(PATH="$T/decoy:$T/bin:/usr/bin:/bin" command -v apptainer 2>/dev/null || true)"
out="$(PATH="$T/decoy:$T/bin:/usr/bin:/bin" APPTAINER_BIN="$T/bin/apptainer" PREFLIGHT=0 \
       bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$_decoy_res" = "$T/decoy/apptainer" ] && [ "$rc" -eq 0 ] && ! grep -q DECOY_APPTAINER_RAN <<<"$out"; then
    ok "PATH preferred the decoy ($_decoy_res) but the explicit APPTAINER_BIN ran"
else
    no "the decoy in PATH won instead of APPTAINER_BIN (rc=$rc, resolved=$_decoy_res)" "$out"
fi

echo
echo "1. the wrapper's preamble-order guard"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 DRY_RUN=1 bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && grep -q 'preamble-order check passed' <<<"$out"; then
    ok "guard accepted the generated body (rc=0)"
else
    no "guard rejected the generated body (rc=$rc)" "$out"
fi
head3="$(sed -n '1,3p' "$BODY")"
if grep -q '^set -uo pipefail' <<<"$head3" && grep -q '^export REPO_IN_CONTAINER=' "$BODY"; then
    ok "the preamble is written before the body"
else
    no "the generated body does not start with the preamble" "$head3"
fi

echo
echo "2. a full invocation, end to end"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && grep -q '\[ok\]   umaze all gd_mpc' <<<"$out" && grep -q 'success_rate=0.99' <<<"$out"; then
    ok "wrapper ran the job loop and summarised it (rc=0)"
else
    no "wrapper did not complete (rc=$rc)" "$out"
fi
if grep -q '\[container\] python=' <<<"$out"; then
    ok "the body activated conda, sourced the env file and cd'd to the repo"
else
    no "the body did not reach its own echo (conda / env file / repo)" "$out"
fi

echo
echo
echo "2b. a wrong CONTAINER_CONDA must be named, not cascade into 'python: not found'"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 CONTAINER_CONDA="$T/definitely_not_there" CONDA_ENV=ts \
       bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && grep -q 'CONTAINER_CONDA' <<<"$out" && grep -q 'conda.sh is not visible' <<<"$out"; then
    ok "reported as a labelled FATAL naming CONTAINER_CONDA"
else
    no "a wrong CONTAINER_CONDA was not reported clearly (rc=$rc)" "$out"
fi

echo
echo "3. the historical layout still fails (so this test has teeth)"
if command -v python3 >/dev/null 2>&1; then
    python3 - "$BODY" "$T/body_broken.sh" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text().splitlines(True)
exports = [l for l in src if l.startswith('export ') and '$(printf' not in l]
rest = [l for l in src if l not in exports]
pathlib.Path(sys.argv[2]).write_text(''.join(rest + exports))   # the buggy layout
PY
    out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 bash "$T/body_broken.sh" 2>&1)"; rc=$?
    if [ "$rc" -ne 0 ] && grep -q 'unbound variable' <<<"$out"; then
        ok "preamble-after-body dies with 'unbound variable' (as it did on the server)"
    else
        no "the historical layout did NOT fail -- the reproduction is wrong" "$out"
    fi
else
    printf '  SKIP  python3 not found, cannot build the historical layout\n'
fi

echo
echo "4. the preflight gate"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=1 CONTAINER_CONDA="$T/miniconda_badpython" \
       bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && grep -q 'preflight failed' <<<"$out"; then
    ok "a failing mujoco_smoke.py stops the run before plan.py"
else
    no "the preflight gate did not fire (rc=$rc)" "$out"
fi

echo
echo "5. a missing ENV_FILE"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 ENV_FILE="$T/does_not_exist.sh" bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && grep -q 'setup_mujoco_server.sh' <<<"$out"; then
    ok "reported before apptainer starts, with the fix to run"
else
    no "a missing env file was not reported usefully (rc=$rc)" "$out"
fi

echo
echo "6. a checkout outside \$HOME (only --bind \$HOME:\$HOME is passed)"
# "Outside $HOME" is forced by pointing HOME at a path that cannot contain the checkout:
# that is exactly what the wrapper's check tests, and -- unlike placing the fake repo in a
# /tmp-based directory -- it also works when $T itself fell back to $HOME (noexec /tmp).
FAKE_HOME=/nonexistent-ts-selftest-home
mkdir -p "$T/fake_repo/run_scripts"
cp "$REPO_HOST_SELF/run_scripts/mpc_server.sh" "$REPO_HOST_SELF/run_scripts/dataset_paths.sh" "$T/fake_repo/run_scripts/"
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 HOME="$FAKE_HOME" REPO_HOST="$T/fake_repo" REPO_IN_CONTAINER="$T/fake_repo" \
       ALLOW_OUTSIDE_HOME=0 bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -ne 0 ] && grep -q 'only \$HOME is bound' <<<"$out"; then
    ok "refused before apptainer, with the fix spelled out"
else
    no "an outside-\$HOME checkout was not refused (rc=$rc)" "$out"
fi
out="$(PATH="$T/bin:$PATH" PREFLIGHT=0 HOME="$FAKE_HOME" REPO_HOST="$T/fake_repo" ALLOW_OUTSIDE_HOME=1 \
       REPO_IN_CONTAINER="$T/fake_repo" bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
if [ "$rc" -eq 0 ] && grep -q 'outside \$HOME' <<<"$out"; then
    ok "ALLOW_OUTSIDE_HOME=1 proceeds with a warning"
else
    no "the escape hatch did not work (rc=$rc)" "$out"
fi

echo
echo "7. the real container (LIVE=1)"
if [ "${LIVE:-0}" != "1" ]; then
    printf '  SKIP  LIVE=1 starts the real image with the real apptainer (needs apptainer + SIF;\n'
    printf '        LIVE_CONTAINER_CONDA / LIVE_CONDA_ENV override the conda env it checks)\n'
else
    live_sif="${LIVE_SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
    live_ovl="${LIVE_OVERLAY:-/scratch/akn7847/containers/temporal-straightening/overlay-50G-10M.ext3}"
    if [ -z "$REAL_APPTAINER" ]; then
        printf '  SKIP  no apptainer on PATH on this machine\n'
    elif [ ! -f "$live_sif" ]; then
        printf '  SKIP  image not found: %s (set LIVE_SIF=...)\n' "$live_sif"
    elif [ ! -f "$live_ovl" ]; then
        printf '  SKIP  overlay not found: %s (set LIVE_OVERLAY=...)\n' "$live_ovl"
    else
        # Mirror the body's first steps: the image has no python until its conda env is
        # activated (asking a bare image for python is what this check got wrong at first).
        live_conda="${LIVE_CONTAINER_CONDA:-/opt/miniconda}"
        live_env="${LIVE_CONDA_ENV:-ts}"
        out="$("$REAL_APPTAINER" exec --fakeroot --nv --bind "$HOME:$HOME" --overlay "$live_ovl:ro" \
               "$live_sif" bash -lc \
               "echo CONTAINER_OK; export PATH=$live_conda/bin:\$PATH; source $live_conda/etc/profile.d/conda.sh; conda activate $live_env; echo PY=\$(command -v python); python -c 'import sys; print(sys.version.split()[0])'" 2>&1)"; rc=$?
        pyver="$(grep -oE '^[0-9]+\.[0-9]+\.[0-9]+$' <<<"$out" | tail -1)"
        if [ "$rc" -eq 0 ] && grep -q CONTAINER_OK <<<"$out"; then
            ok "the image starts: fakeroot + overlay + $live_conda:$live_env + python ${pyver:-?}"
        else
            no "the real container did not start (rc=$rc; CONTAINER_OK present = apptainer/fakeroot/overlay are fine)" "$out"
        fi
    fi
fi

echo
printf 'selftest: %d passed, %d failed\n' "$pass" "$fail"
[ "$KEEP" = "1" ] && echo "kept: $T (body: $T/body.sh)"
[ "$fail" -eq 0 ] || exit 1
exit 0
