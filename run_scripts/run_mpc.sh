#!/usr/bin/env bash
# =============================================================================
# run_mpc.sh -- faithful closed-loop MPC on the trained models. Default: run
#              the complete faithful MPC; FULL=0 switches to a quick validation
#              mode (OOM check + full-run time estimate); OL=1 switches to the
#              true OPEN-LOOP plan (one plan for the whole goal horizon).
#
# Faithful MPC = the paper's closed-loop MPC (temporal-straightening, Table 4/5):
#   n_taken_actions=5, GD opt_steps=100 (Adam, lr 0.1, zero init); CEM
#   num_samples=300, opt_steps=30 (plan_mpc_cem.yaml defaults; the paper's CEM
#   is open-loop only). max_iter is capped at 20 -- the MPC loop exits early
#   on success, so the cap is only a safety bound.
#
# Open loop (OL=1) = the paper's open-loop planning row: conf/plan_gd.yaml /
#   conf/plan_cem.yaml, i.e. max_iter=1 with n_taken_actions=goal_H (25), so the
#   sub-planner looks 5 model steps (= goal_H/frameskip) ahead, plans once, and
#   the whole action sequence is executed in the real sim with no feedback and
#   no re-planning. Results go to plan_outputs_gd_ol/ and plan_outputs_cem_ol/
#   so the closed-loop logs in plan_outputs_gd_mpc/ + plan_outputs_mpc_cem/ (and
#   the OL_RESULTS.md tables derived from them) are never overwritten.
#
# Usage:
#   bash run_mpc.sh <env> [variant] [planner]
#     env:     umaze | medium | pusht | wall
#     variant: all (default) | False | straighten | twothirds | both
#     planner: gd_mpc | mpc_cem | both (default)
#   Flags (optional; the positional and env-var forms above keep working, and the
#   flags win over the CKBPT/SEEDS env vars; they may appear anywhere):
#     --ckpt PATH    checkpoint .pth | single run dir | dir of model run dirs
#                    (else CKBPT, else checkpoints/test)
#     --seeds LIST   eval seeds, space- or comma-separated (else SEEDS)
#     -h | --help    usage
#   bash run_mpc.sh medium both gd_mpc --ckpt /path/to/ckpt_or_dir --seeds "100 101 102"
#   bash run_mpc.sh <env>             # full faithful MPC (default)
#   FULL=0 bash run_mpc.sh <env>      # validation: OOM check + time estimate + batch sanity
#   OL=1 bash run_mpc.sh <env> all both   # OPEN LOOP: one plan per episode for the whole goal_H
#   SEEDS="0 1 2" bash run_mpc.sh <env>  # FULL runs: one plan.py run per seed, report mean +/- std (default: 100 101 102)
#   For every (model, planner), the multi-seed mean +/- std (and per-seed values for
#   all final_eval metrics) is persisted to plan_outputs_<planner>/summaries/<model>_gH<goal_H>.json.
#
# OL=1 open-loop mode (per env/model/seed): plan once with max_iter=1 and
#   n_taken_actions=goal_H, save to plan_outputs_gd_ol/<model>_s<seed>_gH<goal_H>
#   (GD) or plan_outputs_cem_ol/... (CEM), then aggregate with
#   aggregate_mpc_summary.py gd_ol|cem_ol <model> <goal_H> <seeds...>.
#   Budgets default to the same sub-planner budget as the closed-loop runs
#   (GD opt_steps=100; CEM num_samples=300, opt_steps=30) so the OPEN vs CLOSED
#   gap is purely "feedback vs no feedback"; override with OL_GD_OPT /
#   OL_CEM_SAMPLES / OL_CEM_OPT / OL_CEM_CHUNK / OL_N_EVALS / OL_CHUNK.
#
# By default CKBPT is a DIRECTORY of model run dirs and each run dir's checkpoint is
# auto-discovered (checkpoints/<run_dir>/model_latest.pth). To run on one exact
# checkpoint instead, point CKBPT at the checkpoint FILE (or pass --ckpt <path>; the
# flag wins over the env var, and a relative path resolves against the repo root):
#   CKBPT=checkpoints/win7/pointmaze/model_latest.pth bash run_mpc.sh <env> [variant] [planner]
#
# FULL=0 validation mode, per selected (model, planner):
#   1) setup measurement : max_iter=1, opt_steps=0 (model/dset/workspace + evals)
#   2) planner smoke      : max_iter=1, tiny budget (GD opt_steps=3; CEM
#                           num_samples=8, opt_steps=2, sample_chunk_size=8)
#   -> reports OOM if any; extrapolates the full faithful-MPC runtime from
#      plan.py's own [timing] perform_planning_s line.
# =============================================================================

# Work from the repo ROOT: plan.py, conf/ and plan_outputs_* all live one level up (the
# wrapper's summary and analysis/div_emb_tables.py read the outputs there). The previous
# `cd "$(dirname "$0")"` left the shell inside run_scripts/, so `"$PY" plan.py` could not be
# found and every run dir was created under run_scripts/ instead.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
source "$HERE/setup.sh"
# The shared DINO-WM layout (DATA_ROOT per env, CKPT_ROOT/ART_ROOT); the smoke test
# and train_server.sh read the same file.
source "$HERE/dataset_paths.sh"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_MODE
# Interpreter: PYTHON=... wins, then an activated env, then the two known prefixes
# (laptop $HOME/miniconda3, project container /opt/miniconda) -- same probe as setup.sh.
if [ -z "${PYTHON:-}" ]; then
    PY=""
    for cand in "$HOME/miniconda3/envs/ts" "/opt/conda-envs/ts" "/opt/miniconda/envs/ts"; do
        if [ -x "$cand/bin/python" ]; then PY="$cand/bin/python"; break; fi
    done
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]             && { [ "$(basename "$CONDA_PREFIX")" = "ts" ] || [ -z "$PY" ]; }; then
        PY="$CONDA_PREFIX/bin/python"
    fi
else
    PY="$PYTHON"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "FATAL: no ts interpreter found -- set PYTHON=/path/to/envs/ts/bin/python" >&2
    exit 1
fi
# Planning needs BOTH torch and the simulator, so check torch here: a broken environment
# then fails immediately with the real traceback instead of deep inside plan.py.
if ! _torch_err="$("$PY" -c 'import torch' 2>&1)"; then
    echo "FATAL: '$PY' cannot import torch:" >&2
    printf '%s\n' "$_torch_err" | tail -6 >&2
    echo "       If you sourced a MuJoCo env file that PREPENDS LD_LIBRARY_PATH, retry" >&2
    echo "       with MUJOCO_LD_MODE=append (see run_scripts/setup.sh)." >&2
    exit 1
fi

usage() {
    echo "usage: bash run_mpc.sh <env> [variant] [planner] [--ckpt PATH] [--seeds \"100 101 102\"]"
    echo "  env:     umaze | medium | pusht | wall"
    echo "  variant: all | False | straighten | twothirds | both   (default all)"
    echo "  planner: gd_mpc | mpc_cem | both                        (default both)"
    echo "  --ckpt:  checkpoint .pth | single run dir | dir of run dirs   (else CKBPT, else checkpoints/test)"
    echo "  --seeds: eval seeds, space- or comma-separated               (else SEEDS, else 100 101 102)"
    echo "  FULL=0 runs validation instead (OOM check + time estimate; default is the full faithful MPC)"
    echo "  OL=1   runs OPEN LOOP instead (max_iter=1, n_taken_actions=goal_H; outputs in plan_outputs_{gd,cem}_ol/)"
}

# --- CLI flags (optional) ------------------------------------------------------
# The <env> [variant] [planner] positionals and the CKBPT/SEEDS env vars keep
# working; the flags win over the env vars and may appear anywhere in the command.
CLI_CKPT=""
CLI_SEEDS=""
POS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt)    [ -n "${2:-}" ] || { echo "--ckpt needs a path" >&2; exit 1; }
                   CLI_CKPT="$2"; shift 2 ;;
        --ckpt=*)  CLI_CKPT="${1#*=}"; shift ;;
        --seeds)   [ -n "${2:-}" ] || { echo "--seeds needs a seeds list" >&2; exit 1; }
                   CLI_SEEDS="$2"; shift 2 ;;
        --seeds=*) CLI_SEEDS="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; while [ $# -gt 0 ]; do POS+=("$1"); shift; done ;;
        -*)        echo "unknown flag '$1'" >&2; usage >&2; exit 1 ;;
        *)         POS+=("$1"); shift ;;
    esac
done
ENV_SEL="${POS[0]:-}"
VARIANT="${POS[1]:-all}"
PLANNER_SEL="${POS[2]:-both}"
# Extra positionals were always ignored (only <env> [variant] [planner] are read);
# flag them so e.g. a trailing "objective.mode=all" is not mistaken for an override
# (the objective mode is set per env via OBJ_OVERRIDES in the case block below).
if [ "${#POS[@]}" -gt 3 ]; then
    echo "note: ignoring extra positional args: ${POS[*]:3}" >&2
fi
FULL="${FULL:-1}"   # full faithful MPC by default; FULL=0 = validation (OOM + estimate)
OL="${OL:-0}"       # OL=1 = true open loop (one plan for the whole goal_H); wins over FULL

CKBPT="${CLI_CKPT:-${CKBPT:-checkpoints/test}}"   # run.sh stores trained models under checkpoints/test/
# A bad --ckpt would otherwise be swallowed by the per-model "[skip] model not
# available" branch below, so fail loudly on an explicitly requested path.
if [ -n "$CLI_CKPT" ]; then
    if [ ! -e "$CKBPT" ]; then
        echo "--ckpt: '$CKBPT' does not exist" >&2
        exit 1
    elif [ -f "$CKBPT" ] && [[ "$CKBPT" != *.pth ]]; then
        echo "--ckpt: '$CKBPT' is a file but not a .pth checkpoint" >&2
        exit 1
    elif [ ! -f "$CKBPT" ] && [ ! -d "$CKBPT" ]; then
        echo "--ckpt: '$CKBPT' is neither a .pth file nor a directory" >&2
        exit 1
    fi
fi

# Direct-checkpoint mode: if CKBPT points at an existing checkpoint .pth file, run MPC
# on that exact file instead of auto-discovering model run dirs inside a directory.
if [ -f "$CKBPT" ] && [[ "$CKBPT" == *.pth ]]; then
    DIRECT_CKBPT=1
else
    DIRECT_CKBPT=0
fi

# Run-dir fallback: if CKBPT is a directory that itself holds a checkpoint .pth file
# (i.e. it IS a single run dir, not a parent of model run dirs), use that checkpoint.
if [ "$DIRECT_CKBPT" = "0" ] && [ -d "$CKBPT" ]; then
    _pths=()
    for _f in "$CKBPT"/*.pth; do
        [ -f "$_f" ] && _pths+=("$_f")
    done
    if [ "${#_pths[@]}" -eq 1 ]; then
        CKBPT="${_pths[0]}"
        DIRECT_CKBPT=1
        echo "run-dir checkpoint detected: using $CKBPT"
    elif [ "${#_pths[@]}" -gt 1 ] && [ -f "$CKBPT/model_latest.pth" ]; then
        CKBPT="$CKBPT/model_latest.pth"
        DIRECT_CKBPT=1
        echo "run-dir checkpoint detected: using $CKBPT"
    fi
fi

# (env/variant/planner positionals were already parsed above, together with the flags)
SEEDS="${CLI_SEEDS:-${SEEDS:-100 101 102}}"  # eval seeds for FULL runs: one plan.py run per seed, then mean +/- std
SEEDS="${SEEDS//,/ }"   # accept "100,101,102" as well as "100 101 102"
for _s in $SEEDS; do
    if [[ ! "$_s" =~ ^[0-9]+$ ]]; then
        echo "bad seed '$_s' (from --seeds/SEEDS): integers only" >&2
        exit 1
    fi
done
unset _s

if [ -z "$ENV_SEL" ]; then
    usage >&2
    exit 1
fi

# --- per-env map --------------------------------------------------------------
case "$ENV_SEL" in
    umaze)
        ENV_NAME=point_maze
        GOAL_H=25
        # mazes use the weighted intermediate-state objective (paper Sec 5.3: mode=all)
        OBJ_OVERRIDES="objective.alpha=0 objective.mode=last"
        MODELS=(
            "umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
            "umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
            "umaze_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
            "umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        )
        ;;
    medium)
        ENV_NAME=point_maze_medium
        GOAL_H=25
        # mazes use the weighted intermediate-state objective (paper Sec 5.3: mode=all)
        OBJ_OVERRIDES="objective.alpha=0 objective.mode=last"
        MODELS=(
            "medium_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06"
            "medium_cos1e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06"
            "medium_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
            "medium_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"
        )
        ;;
    pusht)
        ENV_NAME=pusht
        GOAL_H=25
        OBJ_OVERRIDES="objective.alpha=1"
        MODELS=(
            "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
            "pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
            "pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
            "pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        )
        ;;
    wall)
        ENV_NAME=wall
        GOAL_H=25
        # mazes use the weighted intermediate-state objective (paper Sec 5.3: mode=all)
        OBJ_OVERRIDES="objective.alpha=0 objective.mode=last"
        MODELS=(
            "wall_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
            "wall_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"
        )
        # The wall models were trained with data_path hardcoded to
        # data/datasets/wall_single. On this laptop the datasets live under
        # DATASET_DIR (setup.sh); only auto-symlink when the external copy that
        # exists on the original dev machine is present.
        if [ ! -e "$PWD/data/datasets/wall_single" ]; then
            if [ -d /run/media/shanveen-ortho-clinic/datadrv/WorldModelDatasets/wall_single ]; then
                ln -s /run/media/shanveen-ortho-clinic/datadrv/WorldModelDatasets/wall_single "$PWD/data/datasets/wall_single"
                echo "  [wall] created symlink: data/datasets/wall_single -> disk mount"
            else
                echo "  [wall] data/datasets/wall_single not found; place wall_single under DATASET_DIR"
            fi
        fi
        ;;
    *)
        echo "unknown env '$ENV_SEL' (choose umaze | medium | pusht | wall)" >&2
        exit 1
        ;;
esac
# The four MODELS above are the dev machine's run-dir names. On another machine, or
# after retraining with a different recipe, override them -- index-aligned with the
# variants all|False|straighten|twothirds|both:
#   ARM_NAMES="baseline straighten p_reg both" bash run_mpc.sh <env> all both ...
if [ -n "${ARM_NAMES:-}" ]; then
    # The list is WORD-SPLIT, so commas are not separators: "a, b, c, d" parses as the names
    # "a," "b," "c," -- the count check still passes and the failure only shows up much later as
    # "model not available"/"have no run dir". A comma list is the natural thing to type, so
    # accept it instead of failing on it.
    ARM_NAMES="${ARM_NAMES//,/ }"
    read -r -a _arm_names <<< "$ARM_NAMES"
    if [ "${#_arm_names[@]}" -ne 4 ]; then
        echo "ARM_NAMES needs 4 space-separated run-dir names (baseline straighten p_reg both), got ${#_arm_names[@]}" >&2
        exit 1
    fi
    MODELS=("${_arm_names[@]}")
    _arms_explicit=1
    unset _arm_names
fi
# Datasets: plan.py reads $DATASET_DIR/<env> and the three datasets are nested
# differently, so DATASET_DIR is per-env (dataset_paths.sh mirrors the exact path
# training gets via env.dataset.data_path). This replaces setup.sh's laptop
# default ($PWD/data/datasets) when the server layout is present; override by
# exporting DATA_ROOT=/path/to/worldmodeldata.
_mpr_data="$(data_dir_for "$ENV_SEL")"
if [ -n "$_mpr_data" ] && [ -d "$_mpr_data" ]; then
    export DATASET_DIR="$(dataset_dir_for "$ENV_SEL")"
fi
unset _mpr_data

# --- variant / planner selection ---------------------------------------------
case "$VARIANT" in
    all)        IDX=(0 1 2 3) ;;
    False)      IDX=(0) ;;
    straighten) IDX=(1) ;;
    twothirds)  IDX=(2) ;;
    both)       IDX=(3) ;;
    *) echo "unknown variant '$VARIANT' (all|False|straighten|twothirds|both)" >&2; exit 1 ;;
esac

case "$PLANNER_SEL" in
    gd_mpc)  PLANNERS=(gd_mpc) ;;
    mpc_cem) PLANNERS=(mpc_cem) ;;
    both)    PLANNERS=(gd_mpc mpc_cem) ;;
    *) echo "unknown planner '$PLANNER_SEL' (gd_mpc|mpc_cem|both)" >&2; exit 1 ;;
esac

# --- faithful (full) and smoke budgets ---------------------------------------
# Overridable so a paper-parity run needs no edit: FULL_N_EVALS=50 GD_OPT=100 come from the
# paper's Table 4, and conf/plan_gd_mpc.yaml's own default is max_iter=4 (sized for a
# goal_H=19 horizon). max_iter is a CAP: the loop exits as soon as the episode succeeds,
# which takes ~5 iters for goal_H=25 (25 / n_taken_actions=5).
FULL_N_EVALS="${FULL_N_EVALS:-50}"
FULL_MAX_ITER="${FULL_MAX_ITER:-20}"
GD_OPT="${GD_OPT:-100}"              # paper Table 4
CEM_SAMPLES="${CEM_SAMPLES:-200}"   # plan_mpc_cem.yaml default (DINO-WM MPC CEM budget)
CEM_OPT="${CEM_OPT:-10}"            # plan_mpc_cem.yaml default
CEM_CHUNK="${CEM_CHUNK:-50}"   # CEM sample_chunk_size; "null" rolls every candidate at once

S_N_EVALS=1
S_MAX_ITER=1
S_CHUNK=1               # batch-1 everywhere: the safest memory config
# Episodes per plan/eval chunk for the FULL runs (closed loop). 1 is the default and
# what a 12 GB dev GPU needs: batch-1 rollouts and ONE env process. "null" puts all
# n_evals in a single batch, which also starts one env process per episode -- so
# request at least n_evals CPUs (run_scripts/mpc_server.sh does) to use them.
CHUNK="${CHUNK:-$S_CHUNK}"
S_GD_OPT=3
S_CEM_SAMPLES=8
S_CEM_OPT=2
S_CEM_CHUNK=8
# Batch sanity stage for FULL=0 (on by default; SANITY=0 skips it). n_evals is deliberately
# larger than n_plot_samples (10) so that batch-dependent code really runs: the other stages
# use n_evals=1 and chunk_size=1, which is exactly what let a chunk_size > n_plot_samples
# IndexError in planning/evaluator.py survive validation and kill the full run.
S_SANITY_N_EVALS="${S_SANITY_N_EVALS:-12}"
SANITY="${SANITY:-1}"

# --- open-loop budgets (OL=1) ------------------------------------------------
# Same sub-planner budget as the closed-loop runs so OPEN vs CLOSED differs only
# in whether the plan is re-planned from env feedback; override via the env vars.
OL_N_EVALS="${OL_N_EVALS:-50}"
OL_CHUNK="${OL_CHUNK:-1}"                 # episodes per plan() call (1 = one episode per plan)
OL_GD_OPT="${OL_GD_OPT:-$GD_OPT}"
OL_CEM_SAMPLES="${OL_CEM_SAMPLES:-300}"   # plan_mpc_cem.yaml default
OL_CEM_OPT="${OL_CEM_OPT:-30}"            # plan_mpc_cem.yaml default
OL_CEM_CHUNK="${OL_CEM_CHUNK:-$CEM_CHUNK}"

# 5 env actions per MPC iteration for these fs=5 envs (n_taken_actions=5 -> /frameskip)
ITERS_HORIZON=$(( GOAL_H / 5 ))
[ "$ITERS_HORIZON" -lt 1 ] && ITERS_HORIZON=1

# --- helpers ------------------------------------------------------------------
obj_for() {  # $1 = planner
    if [ "$ENV_SEL" = "pusht" ] && [ "$1" = "gd_mpc" ]; then
        echo "objective.alpha=1 objective.mode=staged"
    else
        echo "$OBJ_OVERRIDES"
    fi
}

get_sr() {  # $1 = logs.json
    grep -oE '"final_eval/success_rate": ?[0-9.]+' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9.]+'
}

report_mean_std() {  # $1 planner, $2 model, $3 goal_H, $4... seeds
    local planner="$1" model="$2" goal_h="$3"; shift 3
    if [ "$#" -eq 0 ]; then
        echo "  === $planner / $model: no seeds selected -> no mean/std ==="
        return
    fi
    # Aggregate the per-seed logs.json files and persist mean +/- std (for every
    # final_eval metric) to plan_outputs_<planner>/summaries/<model>_gH<goal_h>.json.
    # It also prints the legacy console summary line. rc=1 => no seed had results.
    # The aggregator lives in helpers/ (it moved there with the other analysis helpers); the old
    # bare-root path made this a SILENT no-op -- the fallback line below then blamed the seeds
    # ("no seed runs with final_eval") even when they had run fine, and
    # summaries/<model>_gH<goal_h>.json was never written. It resolves the run dirs relative to
    # the CWD, so keep this call from the repo root (run_mpc.sh cd's there at the top).
    local _agg="$PWD/helpers/aggregate_mpc_summary.py"
    [ -f "$_agg" ] || _agg="$PWD/aggregate_mpc_summary.py"
    if ! "$PY" "$_agg" "$planner" "$model" "$goal_h" "$@"; then
        echo "  === $planner / $model: no mean/std (aggregator error above, or no seed had a"
        echo "      final_eval); per-seed values: plan_outputs_${planner}/${model}_s<seed>_gH${goal_h}/logs.json"
    fi
}


ckpt_full() {  # $1 = model run dir name (unused in direct-checkpoint mode)
    if [ "$DIRECT_CKBPT" = "1" ]; then
        if [[ "$CKBPT" = /* ]]; then echo "$CKBPT"; else echo "$PWD/$CKBPT"; fi
    else
        if [[ "$CKBPT" = /* ]]; then echo "$CKBPT/$1"; else echo "$PWD/$CKBPT/$1"; fi
    fi
}
# Preflight (run-dir mode): resolve the four arm names. The built-in MODELS above are the
# dev machine's run-dir names; on a machine retrained with different recipes
# (projchannel instead of projglobal, ttaggtwothirds, aggflatten, ...) they are absent, so
# the four arms are DISCOVERED among ${ENV_SEL}_* by token:
#   baseline   _False_
#   straighten cos, without two-thirds
#   p_reg      two-thirds (twothirds / wothirds), without cos
#   both       cos AND two-thirds
# An arm with zero or several candidates aborts and prints them -- a wrong mapping must
# never be silently averaged into a table. ARM_NAMES=... (above) skips all of this.
_count_words() { set -- $1; echo "$#"; }
if [ "$DIRECT_CKBPT" != "1" ]; then
    _root="$CKBPT"; case "$_root" in /*) ;; *) _root="$PWD/$_root" ;; esac
    _missing=()
    for _i in "${IDX[@]}"; do
        [ -d "$_root/${MODELS[$_i]}" ] || _missing+=("${MODELS[$_i]}")
    done
    if [ "${#_missing[@]}" -gt 0 ] && [ -z "${_arms_explicit:-}" ]; then
        _base=""; _str=""; _preg=""; _both=""
        for _d in "$_root"/${ENV_SEL}_*; do
            [ -d "$_d" ] || continue
            _c="$(basename "$_d")"
            _cos=0; _tt=0
            case "$_c" in *cos*) _cos=1 ;; esac
            case "$_c" in *wothirds*|*twothirds*) _tt=1 ;; esac
            if [ "$_cos" = "1" ] && [ "$_tt" = "1" ]; then _both="${_both:+$_both }$_c"
            elif [ "$_cos" = "1" ]; then _str="${_str:+$_str }$_c"
            elif [ "$_tt" = "1" ]; then _preg="${_preg:+$_preg }$_c"
            else case "$_c" in *_False_*) _base="${_base:+$_base }$_c" ;; esac
            fi
        done
        _ok=1
        for _pair in "baseline=$_base" "straighten=$_str" "p_reg=$_preg" "both=$_both"; do
            _nm="${_pair%%=*}"; _val="${_pair#*=}"
            if [ "$(_count_words "$_val")" -ne 1 ]; then
                echo "FATAL: cannot resolve arm '$_nm' under $_root (candidates: ${_val:-<none>})" >&2
                _ok=0
            fi
        done
        if [ "$_ok" != "1" ]; then
            echo "  arm dirs present for env $ENV_SEL:" >&2
            for _d in "$_root"/${ENV_SEL}_*; do [ -d "$_d" ] && echo "    $(basename "$_d")" >&2; done
            echo "  fix: give the four names explicitly, in variant order (baseline straighten p_reg both):" >&2
            echo "    ARM_NAMES='<0> <1> <2> <3>' bash $0 $ENV_SEL all ..." >&2
            exit 1
        fi
        MODELS=("$_base" "$_str" "$_preg" "$_both")
        _arms_discovered=1
        echo "=== arms for $ENV_SEL (discovered under $CKBPT):"
        echo "      baseline=$_base"
        echo "      straighten=$_str"
        echo "      p_reg=$_preg"
        echo "      both=$_both"
    elif [ "${#_missing[@]}" -gt 0 ]; then
        echo "FATAL: ${#_missing[@]} selected arm(s) have no run dir under CKBPT=$CKBPT:" >&2
        for _mm in "${_missing[@]}"; do echo "  missing: $_mm" >&2; done
        _n=0
        for _d in "$_root"/${ENV_SEL}_*; do
            [ -d "$_d" ] || continue
            _n=$(( _n + 1 )); echo "  present: $(basename "$_d")" >&2
        done
        [ "$_n" -eq 0 ] && echo "  (no ${ENV_SEL}_* dirs -- is --ckpt pointing at the right place?)" >&2
        echo "  fix: four names in variant order (baseline straighten p_reg both), e.g." >&2
        echo "    ARM_NAMES='<0> <1> <2> <3>' bash $0 $ENV_SEL all ..." >&2
        exit 1
    fi
    unset _missing _i _root _d _n _mm _c _cos _tt _ok _pair _nm _val _base _str _preg _both || true
fi

run_plan() {  # $1 planner, $2 model, $3 run.dir, $4 n_evals, $5 max_iter, $6... extra args
    local planner="$1" model="$2" rundir="$3" n_evals="$4" max_iter="$5"; shift 5
    # RUN_CFG lets OL=1 swap plan_gd_mpc.yaml/plan_mpc_cem.yaml for the open-loop
    # configs (plan_gd.yaml/plan_cem.yaml); chunk_size is RUN_CHUNK (default S_CHUNK).
    local cfg="${RUN_CFG:-plan_${planner}.yaml}"
    local chunk="${RUN_CHUNK:-$S_CHUNK}"
    mkdir -p "$(dirname "$rundir")"
    "$PY" plan.py --config-name "$cfg" \
        ckpt_base_path="$(ckpt_full "$model")" model_name="$model" \
        hydra.run.dir="$rundir" \
        goal_H="$GOAL_H" n_evals="$n_evals" \
        planner.max_iter="$max_iter" chunk_size="$chunk" $(obj_for "$planner") "$@"
}

print_full_cmd() {  # $1 planner, $2 model
    local planner="$1" model="$2" cfg="plan_${planner}.yaml"
    local extra
    if [ "$planner" = "mpc_cem" ]; then
        extra="planner.sub_planner.sample_chunk_size=$CEM_CHUNK"
    else
        extra="planner.sub_planner.opt_steps=$GD_OPT"
    fi
    echo "  FULL: $PY plan.py --config-name $cfg \\"
    echo "        ckpt_base_path=$(ckpt_full "$model") model_name=$model \\"
    echo "        hydra.run.dir=plan_outputs_${planner}/${model}_gH${GOAL_H} \\"
    echo "        goal_H=$GOAL_H n_evals=$FULL_N_EVALS chunk_size=$CHUNK \\"
    echo "        planner.max_iter=$FULL_MAX_ITER $(obj_for "$planner") $extra"
}

estimate() {  # $1 planner, $2 model, $3 t_setup, $4 smoke_log
    local planner="$1" model="$2" t_setup="$3" log="$4"
    local pp
    pp=$(grep -o 'perform_planning_s=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
    if [ -z "$pp" ]; then
        echo "  [estimate] no [timing] line in smoke log; cannot extrapolate."
        return 1
    fi
    if [ "$planner" = "gd_mpc" ]; then
        awk -v pp="$pp" -v steps="$S_GD_OPT" -v n="$FULL_N_EVALS" -v iters="$ITERS_HORIZON" \
            -v budget="$GD_OPT" -v setup="$t_setup" -v m="$model" -v cap="$FULL_MAX_ITER" '
            BEGIN {
                per = pp / steps
                full = n * iters * budget * per + setup
                print "  [estimate] full faithful gd_mpc on " m " (per-episode, batch-1 estimate):"
                print "      ~" int(full/60) " min if the goal is reached in ~" iters " MPC iters (typical)"
                print "      up to ~" int(full * cap / iters / 60) " min if all " cap " iters run the full budget"
                print "      (" per " s per GD opt step at batch 1; raising chunk_size batches evals"
                print "      through the GPU and cuts the optimization time roughly linearly)"
            }' /dev/null
    else
        awk -v pp="$pp" -v samp="$S_CEM_SAMPLES" -v steps="$S_CEM_OPT" -v n="$FULL_N_EVALS" \
            -v iters="$ITERS_HORIZON" -v ns="$CEM_SAMPLES" -v budget="$CEM_OPT" -v setup="$t_setup" \
            -v m="$model" -v cap="$FULL_MAX_ITER" -v cchunk="$CEM_CHUNK" '
            BEGIN {
                per = pp / (samp * steps)
                full = n * iters * ns * budget * per + setup
                print "  [estimate] full faithful mpc_cem on " m " (per-episode, batch-1 estimate):"
                print "      ~" int(full/60) " min if the goal is reached in ~" iters " MPC iters (typical)"
                print "      up to ~" int(full * cap / iters / 60) " min if all " cap " iters run the full budget"
                print "      (" per " s per CEM sample-step at batch 1; sample_chunk_size=" cchunk " bounds"
                print "      GPU memory while keeping the full " ns "-sample budget)"
            }' /dev/null
    fi
}
# Any run that fails, or that produces no result, sets RC_FAILED; the script exits with it.
# Without this, mpc_server.sh reports "[ok]" for envs that failed (its [FAIL] comes from
# this exit status) and the summary lists success_rate=<n/a> with no failing env named.
RC_FAILED=0

# --- failure hints ------------------------------------------------------------
# A forked env worker that dies while initialising GL prints the OpenGL traceback in the
# worker and leaves the parent with EOFError from the pipe, so the stage only reports
# "rc=1" and neither the [estimate] lines nor success_rate say why. Recognise the known
# messages and name the fix (SERVER_CONTEXT 11.4; run_scripts/gl_backend_probe.py decides
# whether spawn is the right answer on this machine).
gl_failure_hint() {  # $1 log file, $2 stage label -- 0 when the log shows a GL init failure
    grep -qiE 'Failed to initialize OpenGL|OffscreenOpenGLContext|GLEW init' "$1" 2>/dev/null || return 1
    echo "  !! $2: an env worker could not initialise GL."
    echo "     The workers are forked, and a child that forks after GL has been initialised"
    echo "     cannot initialise EGL -- the parent's own renders and the [timing] lines above"
    echo "     it are fine. Re-run with  TS_ENV_START_METHOD=spawn  (fresh interpreter per"
    echo "     worker). Confirm the diagnosis first with:"
    echo "         PROBE=1 OVERLAY_RW=1 FULL=0 sbatch run_scripts/mpc_server.sh"
    return 0
}

# --- run ----------------------------------------------------------------------
echo "=== run_mpc.sh: ckpt=$CKBPT seeds='$SEEDS' env=$ENV_SEL ($ENV_NAME) variants=$VARIANT planners=${PLANNERS[*]} FULL=$FULL OL=$OL ==="
echo "=== data: DATASET_DIR=$DATASET_DIR  (DATA_ROOT=${DATA_ROOT:-<default>}) ==="
# Print the two knobs whose absence is invisible in the stage logs: an unset
# TS_ENV_START_METHOD is exactly what makes a forked env worker fail to initialise OpenGL
# (SERVER_CONTEXT 11.4), and ARM_NAMES is what decides which four run dirs MODELS points at.
echo "=== knobs: TS_ENV_START_METHOD=${TS_ENV_START_METHOD:-<unset>} MUJOCO_GL=${MUJOCO_GL:-<unset>} ARM_NAMES=${ARM_NAMES:-<auto-discovered>} ==="
# Index -> variant -> run dir, with the regularizer tokens the name carries. The four names are
# INDEX-ALIGNED with the variants (baseline straighten p_reg both) and nothing in a name enforces
# that, so a swapped pair runs the wrong arm with no error at all: `variant=twothirds` would
# measure "both" and p_reg would never be measured. The tokens here are the same ones the
# auto-discovery uses, so a mismatch is called out now rather than in a mislabelled table later.
if [ -n "${_arms_explicit:-}${_arms_discovered:-}" ] && [ "$DIRECT_CKBPT" != "1" ]; then
    _ck_root="$CKBPT"; case "$_ck_root" in /*) ;; *) _ck_root="$PWD/$_ck_root" ;; esac
    _lab=(False straighten twothirds both)
    _want=(none cos twothirds cos+twothirds)
    _arm_bad=""
    echo "=== arms (index -> variant -> run dir; the name's tokens must match the variant):"
    for _i in 0 1 2 3; do
        _n="${MODELS[$_i]}"
        _cos=0; _tt=0
        case "$_n" in *cos*) _cos=1 ;; esac
        case "$_n" in *wothirds*|*twothirds*) _tt=1 ;; esac
        _tok="none"
        if [ "$_cos" = "1" ] && [ "$_tt" = "1" ]; then _tok="cos+twothirds"
        elif [ "$_cos" = "1" ]; then _tok="cos"
        elif [ "$_tt" = "1" ]; then _tok="twothirds"
        fi
        _where="ok"; [ -d "$_ck_root/$_n" ] || _where="NO-DIR"
        printf '      [%d] %-10s %-8s %-7s %s\n' "$_i" "${_lab[$_i]}" "$_where" "$_tok" "$_n"
        [ "$_tok" = "${_want[$_i]}" ] || _arm_bad="$_arm_bad [$_i]=${_lab[$_i]}(has $_tok)"
    done
    if [ -n "$_arm_bad" ]; then
        echo "!! ARM_NAMES is index-aligned (baseline straighten p_reg both) but these slots do not" >&2
        echo "   match the variant they will be used for:$_arm_bad" >&2
        echo "   A swap runs the wrong arm with no other symptom -- e.g. variant=twothirds measuring" >&2
        echo "   'both', so p_reg is never measured. Reorder the names to match the tokens above." >&2
    fi
    unset _ck_root _lab _want _arm_bad _i _n _cos _tt _tok _where
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--- DRY_RUN: resolved configuration (nothing is run) ---"
    for _i in "${IDX[@]}"; do
        for _pl in "${PLANNERS[@]}"; do
            echo "  env=$ENV_SEL variant=$VARIANT arm=${MODELS[$_i]} planner=$_pl"
            echo "      seeds=$SEEDS FULL=$FULL OL=$OL n_evals=$FULL_N_EVALS max_iter=$FULL_MAX_ITER"
            echo "      chunk(closed)=$CHUNK  OL_CHUNK=$OL_CHUNK  cem_sample_chunk=$CEM_CHUNK  (null = no chunking)"
            echo "      ckpt=$CKBPT DATASET_DIR=$DATASET_DIR"
        done
    done
    echo "DRY_RUN=1: printed the plan, ran nothing."
    unset _i _pl
    exit 0
fi
if [ "$DIRECT_CKBPT" = "1" ]; then
    echo "direct-checkpoint mode: running MPC on $CKBPT"
    model="$(basename "$CKBPT")"
    model="${model%.pth}"
    MODELS_SEL=( "$model" )
else
    MODELS_SEL=()
    for i in "${IDX[@]}"; do
        MODELS_SEL+=( "${MODELS[$i]}" )
    done
fi
for model in "${MODELS_SEL[@]}"; do
    if [ "$DIRECT_CKBPT" != "1" ] && { [ -z "$model" ] || [ ! -d "$CKBPT/$model" ]; }; then
        echo "[skip] model not available for this env: $CKBPT/$model"
        RC_FAILED=1
        continue
    fi
    for planner in "${PLANNERS[@]}"; do
        if [ "$OL" = "1" ]; then
            # --- OPEN LOOP: one plan for the whole goal horizon, executed once ---
            # conf/plan_gd.yaml / conf/plan_cem.yaml (max_iter=1, n_taken_actions
            # = goal_H -> the sub-planner looks goal_H/frameskip model steps ahead
            # and the full action sequence is executed without re-planning).
            # Outputs go to plan_outputs_{gd,cem}_ol/ so the closed-loop logs in
            # plan_outputs_gd_mpc/ + plan_outputs_mpc_cem/ are left untouched.
            case "$planner" in
                gd_mpc)  ol_tag=gd_ol ;  RUN_CFG=plan_gd.yaml ;;
                *)       ol_tag=cem_ol ; RUN_CFG=plan_cem.yaml ;;
            esac
            RUN_CHUNK="$OL_CHUNK"
            echo "===== OPEN LOOP: $planner / $model (seeds: $SEEDS, n_evals=$OL_N_EVALS, chunk=$OL_CHUNK) ====="
            for s in $SEEDS; do
                rundir="plan_outputs_${ol_tag}/${model}_s${s}_gH${GOAL_H}"
                # fresh run: logs.json is append-mode, so a crashed/partial dir
                # would otherwise corrupt the new run's results.
                rm -rf "$rundir"
                if [ "$planner" = "gd_mpc" ]; then
                    run_plan "$planner" "$model" "$rundir" "$OL_N_EVALS" 1 \
                        seed="$s" planner.n_taken_actions="$GOAL_H" \
                        planner.sub_planner.opt_steps="$OL_GD_OPT"
                else
                    run_plan "$planner" "$model" "$rundir" "$OL_N_EVALS" 1 \
                        seed="$s" planner.n_taken_actions="$GOAL_H" \
                        planner.sub_planner.num_samples="$OL_CEM_SAMPLES" \
                        planner.sub_planner.opt_steps="$OL_CEM_OPT" \
                        planner.sub_planner.sample_chunk_size="$OL_CEM_CHUNK"
                fi
                rc=$?
                sr=$(get_sr "$rundir/logs.json")
                if [ -z "$sr" ]; then
                    echo "  seed $s rc=$rc (results in $rundir/logs.json); no final_eval/success_rate found"
                    echo "     (rc=$rc: if the job log shows 'Failed to initialize OpenGL' in an env worker,"
                    echo "      that is the forked-worker EGL problem -- re-run with TS_ENV_START_METHOD=spawn)"
                    RC_FAILED=1
                    continue
                fi
                echo "  seed $s success_rate=$sr"
            done
            report_mean_std "$ol_tag" "$model" "$GOAL_H" $SEEDS
            unset RUN_CFG RUN_CHUNK
            continue
        fi
        if [ "$FULL" = "1" ]; then
            echo "===== FULL MPC: $planner / $model (seeds: $SEEDS) ====="
            RUN_CHUNK="$CHUNK"
            for s in $SEEDS; do
                rundir="plan_outputs_${planner}/${model}_s${s}_gH${GOAL_H}"
                # fresh run: logs.json is append-mode, so a crashed/partial dir
                # would otherwise corrupt the new run's results.
                rm -rf "$rundir"
                if [ "$planner" = "gd_mpc" ]; then
                    run_plan "$planner" "$model" "$rundir" "$FULL_N_EVALS" "$FULL_MAX_ITER" \
                        seed="$s" planner.sub_planner.opt_steps="$GD_OPT"
                else
                    # num_samples/opt_steps come from the plan_mpc_cem.yaml defaults
                    # (300 / 30); only the GPU-memory chunk size is overridden here.
                    run_plan "$planner" "$model" "$rundir" "$FULL_N_EVALS" "$FULL_MAX_ITER" \
                        seed="$s" planner.sub_planner.sample_chunk_size="$CEM_CHUNK"
                fi
                rc=$?
                sr=$(get_sr "$rundir/logs.json")
                if [ -z "$sr" ]; then
                    echo "  seed $s rc=$rc (results in $rundir/logs.json); no final_eval/success_rate found"
                    echo "     (rc=$rc: if the job log shows 'Failed to initialize OpenGL' in an env worker,"
                    echo "      that is the forked-worker EGL problem -- re-run with TS_ENV_START_METHOD=spawn)"
                    RC_FAILED=1
                    continue
                fi
                echo "  seed $s success_rate=$sr"
            done
            report_mean_std "$planner" "$model" "$GOAL_H" $SEEDS
            unset RUN_CHUNK
            continue
        fi
        echo "===== validate: $planner / $model ====="
        # The `> "$log1"` redirects below open their file *before* run_plan runs, so the
        # `mkdir -p "$(dirname "$rundir")"` inside run_plan comes too late: on a fresh
        # checkout every validate run died with
        #   plan_outputs_gd_mpc/validate_<model>_setup.log: No such file or directory
        # (it only ever worked where an earlier full run had created the directory).
        mkdir -p "plan_outputs_${planner}"
        log1="plan_outputs_${planner}/validate_${model}_setup.log"
        if [ "$planner" = "gd_mpc" ]; then
            setup_extra=(planner.sub_planner.opt_steps=0)
        else
            setup_extra=(planner.sub_planner.opt_steps=0 \
                         planner.sub_planner.num_samples=1 \
                         planner.sub_planner.sample_chunk_size=1)
        fi
        echo "  [1/4] setup measurement..."
        t0=$(date +%s)
        run_plan "$planner" "$model" "plan_outputs_${planner}/validate_${model}_setup_gH${GOAL_H}" \
            "$S_N_EVALS" "$S_MAX_ITER" "${setup_extra[@]}" > "$log1" 2>&1
        rc1=$?
        t1=$(date +%s)
        t_setup=$(( t1 - t0 ))
        if grep -qi 'OutOfMemoryError\|out of memory' "$log1"; then
            echo "  !! OOM during setup. The model does not fit at batch 1; nothing more to chunk."
            RC_FAILED=1
            continue
        fi
        if [ $rc1 -ne 0 ]; then
            echo "  !! setup run rc=$rc1 (see $log1); skipping."
            gl_failure_hint "$log1" "setup" || true
            RC_FAILED=1
            continue
        fi
        echo "  setup ok (${t_setup}s)"

        log2="plan_outputs_${planner}/validate_${model}_smoke.log"
        if [ "$planner" = "gd_mpc" ]; then
            extra=(planner.sub_planner.opt_steps="$S_GD_OPT")
        else
            extra=(planner.sub_planner.num_samples="$S_CEM_SAMPLES" \
                   planner.sub_planner.opt_steps="$S_CEM_OPT" \
                   planner.sub_planner.sample_chunk_size="$S_CEM_CHUNK")
        fi
        echo "  [2/4] ${planner} smoke..."
        t0=$(date +%s)
        run_plan "$planner" "$model" "plan_outputs_${planner}/validate_${model}_smoke_gH${GOAL_H}" \
            "$S_N_EVALS" "$S_MAX_ITER" "${extra[@]}" > "$log2" 2>&1
        rc2=$?
        t1=$(date +%s)
        t_smoke=$(( t1 - t0 ))
        if grep -qi 'OutOfMemoryError\|out of memory' "$log2"; then
            echo "  !! OOM in ${planner} smoke (${t_smoke}s). Keep chunk_size=1 / reduce"
            echo "     sample_chunk_size or num_samples for the full run."
            RC_FAILED=1
            continue
        fi
        if [ $rc2 -ne 0 ]; then
            echo "  !! ${planner} smoke rc=$rc2 (see $log2); skipping estimate."
            gl_failure_hint "$log2" "smoke" || true
            RC_FAILED=1
            continue
        fi
        echo "  smoke ok (${t_smoke}s)"

        echo "  [3/4] estimate:"
        estimate "$planner" "$model" "$t_setup" "$log2"
        print_full_cmd "$planner" "$model"

        if [ "$SANITY" = "1" ]; then
            echo "  [4/4] batch sanity: $S_SANITY_N_EVALS episodes at chunk_size=${CHUNK:-$S_CHUNK}, 1 iteration"
            log3="plan_outputs_${planner}/validate_${model}_sanity.log"
            if [ "$planner" = "gd_mpc" ]; then
                sanity_extra=(planner.sub_planner.opt_steps=1)
            else
                sanity_extra=(planner.sub_planner.num_samples=8 planner.sub_planner.opt_steps=1 planner.sub_planner.sample_chunk_size=8)
            fi
            RUN_CHUNK="$CHUNK"       # the full run's chunking, not S_CHUNK=1
            t0=$(date +%s)
            run_plan "$planner" "$model" "plan_outputs_${planner}/validate_${model}_sanity_gH${GOAL_H}" "$S_SANITY_N_EVALS" "1" "${sanity_extra[@]}" > "$log3" 2>&1
            rc3=$?
            t1=$(date +%s)
            unset RUN_CHUNK
            if grep -qi 'OutOfMemoryError\|out of memory' "$log3"; then
                echo "  !! OOM at n_evals=$S_SANITY_N_EVALS with chunk_size=$CHUNK: the full run"
                echo "     would not fit either -- lower CHUNK (e.g. 8) and re-run this."
                RC_FAILED=1
                continue
            fi
            if [ $rc3 -ne 0 ]; then
                echo "  !! batch sanity rc=$rc3 (see $log3) -- the full run would fail here too."
                gl_failure_hint "$log3" "batch sanity" || true
                RC_FAILED=1
                continue
            fi
            echo "  sanity ok ($((t1 - t0))s)"
        fi
        echo
    done
done
if [ "$RC_FAILED" -ne 0 ]; then
    echo "=== run_mpc.sh: at least one run failed -- see the !! lines above ==="
else
    echo "=== run_mpc.sh done ==="
fi
exit "$RC_FAILED"
