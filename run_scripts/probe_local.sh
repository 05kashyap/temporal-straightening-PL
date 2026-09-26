#!/bin/bash
# =============================================================================
# probe_local.sh -- run both linear probes on THIS machine (no sbatch, no container).
#
#   bash run_scripts/probe_local.sh                        # all 3 envs, full val set
#   ONLY=point_maze_medium:aggmlp bash run_scripts/probe_local.sh
#   LIMIT_EPISODES=300 bash run_scripts/probe_local.sh     # subsampled quick pass
#   METHODS=rollout bash run_scripts/probe_local.sh        # one method only
#
#   # per-group loop, one report at the end (what an overnight local sweep looks like)
#   # (the knob is PROBE_GROUPS, not GROUPS: $GROUPS is a read-only bash builtin)
#   PROBE_GROUPS="point_maze_medium:aggmlp point_maze_medium:flatten point_maze_medium:global \
#                 point_maze:aggmlp point_maze:global pusht:aggmlp pusht:global" \
#     bash run_scripts/probe_local.sh
#
#   PYTHON=/path/to/python  DATA_ROOT=/media/.../WorldModelDatasets  DEVICE=cpu  ...
#
# Everything else is analysis/run_probes.py's CLI (see its --help); this script only
# picks the interpreter, DATA_ROOT and the optional per-group loop.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in "$HOME/miniconda3/envs/ts/bin/python" /opt/conda-envs/ts/bin/python; do
        [ -x "$cand" ] && PY="$cand" && break
    done
fi
[ -n "$PY" ] || { echo "FATAL: no python found -- set PYTHON=/path/to/ts/python" >&2; exit 1; }

# data/datasets/* are symlinks to the dataset drive on this machine; the probe
# redirects a checkpoint's stored path to $DATASET_DIR/<name>, and run_probes.py
# derives the per-env DATASET_DIR from --data-root by probing both layouts.
export DATA_ROOT="${DATA_ROOT:-$REPO/data/datasets}"
CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints}"
OUTDIR="${OUTDIR:-$REPO/analysis_outputs}"
ENVS="${ENVS:-point_maze,point_maze_medium,pusht}"

ARGS=(--ckpt-root "$CKPT_ROOT/test" --data-root "$DATA_ROOT" --outdir "$OUTDIR"
      --envs "$ENVS" --device "${DEVICE:-cuda}")
[ -n "${RECIPES:-}" ] && ARGS+=(--recipes "$RECIPES")
[ -n "${ONLY:-}" ] && ARGS+=(--only "$ONLY")
[ -n "${METHODS:-}" ] && ARGS+=(--methods "$METHODS")
[ -n "${SEED:-}" ] && ARGS+=(--seed "$SEED")
[ -n "${TEST_FRAC:-}" ] && ARGS+=(--test-frac "$TEST_FRAC")
[ -n "${LIMIT_EPISODES:-}" ] && ARGS+=(--limit-episodes "$LIMIT_EPISODES")
[ -n "${TMAX:-}" ] && ARGS+=(--tmax "$TMAX")

echo "repo     : $REPO"
echo "python   : $PY"
echo "ckpt root: $CKPT_ROOT/test   DATA_ROOT: $DATA_ROOT   outdir: $OUTDIR"
echo "envs     : $ENVS  recipes=${RECIPES:-<all>}  only=${ONLY:-<all>}  methods=${METHODS:-encoded,rollout}"
[ -n "${LIMIT_EPISODES:-}" ] && echo "NOTE     : --limit-episodes=$LIMIT_EPISODES subsamples the val set; \
the CSVs record n_train_eps/n_test_eps, so a subsampled figure is self-documenting."

rc=0
if [ -n "${PROBE_GROUPS:-}" ]; then
    for g in $PROBE_GROUPS; do
        echo
        echo "===== group $g ====="
        "$PY" analysis/run_probes.py "${ARGS[@]}" --only "$g" --skip-report || { echo "[FAIL] $g" >&2; rc=1; }
    done
    echo
    echo "===== report ====="
    "$PY" analysis/probe_report.py --outdir "$OUTDIR" --ckpt-root "$CKPT_ROOT/test" \
        --envs "$ENVS" || rc=1
else
    "$PY" analysis/run_probes.py "${ARGS[@]}" || rc=1
fi
exit "$rc"
