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
#
#   bash run_scripts/selftest_mpc_server.sh        # 7 PASS/FAIL lines, exit code
#   KEEP=1 bash run_scripts/selftest_mpc_server.sh # keep the temp dir (prints it)
#
# No pipe into `head`/`grep -q` anywhere: this file runs under `set -o pipefail`,
# where an early exit upstream gives SIGPIPE and fails the pipeline at random.
# =============================================================================
set -uo pipefail

REPO_HOST_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(mktemp -d "${TMPDIR:-/tmp}/ts_selftest.XXXXXX")"
KEEP="${KEEP:-0}"
trap '[ "$KEEP" = "1" ] || rm -rf "$T"' EXIT

pass=0; fail=0
ok() { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
no() { printf '  FAIL  %s\n' "$1"; printf '        | %s\n' "${2:-<no output>}" | sed -n '1,4p'; fail=$((fail + 1)); }

# --- stubs: apptainer, the container conda, run_mpc.sh, python -------------------------
mkdir -p "$T/bin" "$T/ck/test" "$T/miniconda/etc/profile.d" "$T/spool" "$T/py_ok" "$T/py_bad"
: > "$T/fake.sif"
: > "$T/fake.overlay"
printf 'conda() { return 0; }\n' > "$T/miniconda/etc/profile.d/conda.sh"
printf '#!/bin/bash\nexport MUJOCO_PATH=%s\n' "$T/miniconda" > "$T/mujoco_env.sh"
cat > "$T/bin/apptainer" <<'STUB'
#!/bin/bash
# stands in for: apptainer exec --fakeroot --nv --bind X --overlay Y <image> <cmd...>
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
case "$*" in
    *run_mpc.sh*) echo "[stub run_mpc.sh] $*"; echo "success_rate=0.99"; exit 0 ;;
esac
exec /bin/bash "$@"
STUB
printf '#!/bin/bash\necho "[stub python] %s"\nexit 0\n' 'ok' > "$T/py_ok/python"
printf '#!/bin/bash\necho "[stub python] mujoco is broken"\nexit 1\n' > "$T/py_bad/python"
chmod +x "$T/bin/apptainer" "$T/bin/bash" "$T/py_ok/python" "$T/py_bad/python"

# --- the wrapper exactly as sbatch sees it: a spool copy, not the checkout file --------
cp "$REPO_HOST_SELF/run_scripts/mpc_server.sh" "$T/spool/slurm_script"
WRAP="$T/spool/slurm_script"
export SLURM_SUBMIT_DIR="$REPO_HOST_SELF"
export BODY="$T/body.sh" CKPT_ROOT="$T/ck" CKBPT="$T/ck/test" SIF="$T/fake.sif" \
       OVERLAY="$T/fake.overlay" CONTAINER_CONDA="$T/miniconda" ENV_FILE="$T/mujoco_env.sh"

echo "repo : $REPO_HOST_SELF"
echo "temp : $T"


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
out="$(PATH="$T/bin:$T/py_bad:$PATH" PREFLIGHT=1 bash "$WRAP" umaze all gd_mpc 2>&1)"; rc=$?
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
printf 'selftest: %d passed, %d failed\n' "$pass" "$fail"
[ "$KEEP" = "1" ] && echo "kept: $T (body: $T/body.sh)"
[ "$fail" -eq 0 ] || exit 1
exit 0
