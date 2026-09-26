#!/usr/bin/env python
"""analysis/linear_probe.py
==========================
Two probes of what a frozen world model's latents carry about the agent's TRUE
state, per checkpoint arm (baseline / straighten / p_reg / both):

  --feature-source encoded   "grounded": probe REAL val frames. Encode every frame
                             with the frozen encoder, pool the latents the way the
                             training regularizer did, fit a ridge + a small MLP and
                             report held-out R^2 for position (first 2 state dims)
                             and for the full state vector.
  --feature-source rollout   "ungrounded": probe the model's OWN imagined rollout.
                             VWorldModel.rollout() imagines each episode, the
                             predicted latents are pooled, ONE ridge is fitted on all
                             (episode, timestep) rows and read out per timestep ->
                             R^2(t), retention R^2(T_max)/R^2(1), and a shared-horizon
                             R^2 so arms with different horizons stay comparable.

Both take the four arms of one (env, recipe) group, either from the legacy tables
below or from --ckpts, which is what analysis/run_probes.py drives: that discovers
every group under a checkpoint root (recipe read from each run dir's own hydra.yaml:
encoder.projector + encoder.agg_type -> global / flatten / aggmlp), runs both methods
per group, and hands the CSVs to analysis/probe_report.py, which writes the markdown
that explains every figure -- which is why the figures carry one short title only.

Usage:
    # one group with explicit checkpoints (what the batch driver does)
    python analysis/linear_probe.py --env point_maze_medium --label aggmlp \
        --ckpts "baseline=checkpoints/test/medium_False_aggmlp_...,straighten=..."
    # the legacy tables (one recipe per env)
    python analysis/linear_probe.py --env point_maze_medium --models channel
    # what is available, and the exact mapping the driver would use?
    python analysis/probe_ckpts.py --ckpt-root checkpoints/test --verbose

Outputs (all under --outdir). They are namespaced by env + recipe label, because the
global / flatten / aggmlp groups of one env used to overwrite each other's figure and
CSV rows:
    linear_probe_<env>_<label>.csv / .png              grounded: rows + bar plot
    rollout_probe_<env>_<label>_curves.csv             ungrounded: R^2(t) rows
    rollout_probe_<env>_<label>_summary.csv            ungrounded: per-arm summary
    rollout_probe_<env>_<label>.png                    ungrounded: the R^2(t) figure

Checkpoint paths come from --ckpts, else the MODELS / CHANNEL_MODELS tables below.
Entries left as empty strings raise a clear RuntimeError and skip that arm, so a
partially-downloaded env still yields results for the arms that ARE present.
"""

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curvature_analysis import (  # noqa: E402
    load_model,
    load_val_dset,
    encode_episode,
    get_mode,
)

try:                                                     # `python analysis/linear_probe.py`
    import probe_ckpts                                   # noqa: E402
except ImportError:                                      # `from analysis import linear_probe`
    from analysis import probe_ckpts                     # noqa: E402

# ---------------------------------------------------------------------------
# Checkpoint run dirs (relative to REPO). The four variants mirror the four
# run.sh / kaggle_train.sh variants. Empty strings are placeholders that will be
# reported (and skipped) at runtime.
# ---------------------------------------------------------------------------
MODELS = {
    "point_maze": {
        "baseline": "checkpoints/test/umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/umaze_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "both": "checkpoints/test/umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
    },
    "point_maze_medium": {
        "baseline": "checkpoints/test/medium_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06",
        "straighten": "checkpoints/test/medium_cos1e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06",
        "p_reg": "checkpoints/test/medium_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
        "both": "checkpoints/test/medium_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05",
    },
    "pusht": {
        "baseline": "checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "checkpoints/test/pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "wall": {
        "baseline": "checkpoints/test/wall_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/wall_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        # Two-thirds / both wall checkpoints are not present locally; fill these
        # in when they become available (empty => clear error + skip).
        "p_reg": "",
        "both": "",
    },
}

VARIANT_ORDER = ["baseline", "straighten", "p_reg", "both"]


# An explicit {arm: run dir} mapping from --ckpts for the group being probed (filled by
# main() below). It takes precedence over the tables so a batch runner can point this
# script at any four checkpoints -- e.g. the aggmlp medium set -- without editing this file.
CLI_ARM_DIRS = {}


def resolve_ckpt(env, variant, table=None):
    """Absolute checkpoint run dir for one arm; table picks projglobal vs channel."""
    if CLI_ARM_DIRS:
        path = CLI_ARM_DIRS.get(variant)
        if not path:
            raise RuntimeError(
                f"--ckpts has no entry for arm {variant!r} (given: {sorted(CLI_ARM_DIRS)})")
    else:
        table = MODELS if table is None else table
        path = table[env][variant]
    if not path:
        raise RuntimeError(
            f"MODELS[{env!r}][{variant!r}] is an empty placeholder -- "
            "fill in the checkpoint path before running."
        )
    model_dir = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.isfile(os.path.join(model_dir, "hydra.yaml")):
        raise RuntimeError(f"Missing hydra.yaml in checkpoint dir: {model_dir}")
    if not os.path.isfile(os.path.join(model_dir, "checkpoints", "model_latest.pth")):
        raise RuntimeError(
            f"Missing checkpoints/model_latest.pth in checkpoint dir: {model_dir}"
        )
    return model_dir


def to_numpy(x):
    """Coerce a torch tensor / ndarray to a float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def pool_features(wm, feats, mode):
    """Pool encoder latents (1, T, p, d) -> (1, T, D) exactly like the regularizer.

    aggcos: aggregate the patch dim with wm.encoder.agg (dino_channel).
    cos   : the latents are already per-frame; squeeze a singleton patch dim
            (dino_global, p=1) or flatten patches as a fallback.
    """
    b, t, p, d = feats.shape
    if mode == "aggcos":
        return wm.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
    if p == 1:
        return feats[:, :, 0, :]          # (b, t, d) for dino_global
    return feats.reshape(b, t, -1)        # (b, t, p*d) fallback


def collect_split(wm, val, indices, device):
    """Encode episodes at `indices`; return (X, Y_pos, Y_full) numpy arrays."""
    mode = get_mode(wm)
    Xs, Yp, Yf = [], [], []
    for i in indices:
        obs, _act, state, _info = val[i]
        # Mirror collect_episodes' length guard, relaxed: the probe only needs
        # >= 1 frame (no velocity triplets), so only empty episodes are dropped
        # (this never triggers for real trajectories).
        if obs["visual"].shape[0] < 1:
            continue
        feats = encode_episode(wm, obs, device)  # (1, T, p, d), under no_grad
        assert not feats.requires_grad, "encode_episode must run under torch.no_grad()"
        with torch.no_grad():
            pooled = pool_features(wm, feats, mode)
        Xs.append(to_numpy(pooled[0]))           # (T, D)
        state_np = to_numpy(state)               # (T, state_dim)
        Yp.append(state_np[:, :2])               # position (first 2 dims)
        Yf.append(state_np)                      # full state vector
    if not Xs:
        raise RuntimeError("No usable episodes in this split (all filtered out).")
    return np.concatenate(Xs, 0), np.concatenate(Yp, 0), np.concatenate(Yf, 0)


def split_episode_indices(n_episodes, test_frac=0.3, seed=0):
    """Shuffle episode indices with a fixed seed and split off a test fraction."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_episodes).tolist()
    n_test = max(1, int(round(n_episodes * test_frac)))
    return idx[n_test:], idx[:n_test]


def train_mlp(X_train, Y_train, X_test, out_dim, epochs=200, lr=1e-3, seed=0):
    """Small MLP probe: Linear(d,128) -> ReLU -> Linear(128,out_dim); full-batch Adam."""
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, out_dim),
    )
    Xtr, Ytr = torch.from_numpy(X_train), torch.from_numpy(Y_train)
    Xte = torch.from_numpy(X_test)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(Xtr), Ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).numpy()


def per_dim_r2(y_true, y_pred):
    r2 = np.asarray(r2_score(y_true, y_pred, multioutput="raw_values"), dtype=np.float64)
    return r2, float(np.nanmean(r2))


def fmt_dims(arr):
    return "[" + ", ".join((f"{v:.4f}" if np.isfinite(v) else "nan") for v in arr) + "]"



def redirect_dset_paths(train_cfg):
    """Repoint the checkpoint's saved (old-machine) dataset path at the local copy.

    Mirrors plan.py: if env.dataset.data_path does not exist on this machine,
    redirect it to $DATASET_DIR/<basename> (default <repo>/data/datasets/<basename>).
    """
    try:
        stored = train_cfg.env.dataset.get("data_path")
    except Exception:
        stored = None
    if not stored:
        return
    stored = os.path.normpath(stored)
    if os.path.isdir(stored):
        return
    dname = os.path.basename(stored)
    root = os.environ.get("DATASET_DIR", os.path.join(REPO, "data", "datasets"))
    redirect = os.path.join(root, dname)
    if not os.path.isdir(redirect):
        raise RuntimeError(
            f"dataset path {stored} is missing and no local copy at {redirect}; "
            f"set DATASET_DIR to the folder containing '{dname}'."
        )
    OmegaConf.set_struct(train_cfg, False)
    train_cfg.env.dataset.data_path = redirect
    OmegaConf.set_struct(train_cfg, True)
    print(f"[linear_probe] dataset data_path {stored} -> {redirect}")


def run_variant(env, variant, device, test_frac, seed, mlp_epochs=200, table=None,
                limit_episodes=0):
    """Fit ridge + MLP probes for one (env, variant); return result rows."""
    model_dir = resolve_ckpt(env, variant, table)
    wm, train_cfg = load_model(model_dir, device)
    wm.eval()
    redirect_dset_paths(train_cfg)
    val = load_val_dset(train_cfg)

    train_idx, test_idx = split_episode_indices(len(val), test_frac=test_frac, seed=seed)
    assert set(train_idx).isdisjoint(set(test_idx)), "probe train/test splits overlap"
    # Hard, on episode IDENTITIES (val.indices) rather than positions: two positions can
    # point at the same trajectory, which is exactly the leak that would inflate
    # every R^2 silently.
    assert_disjoint_episodes(val, train_idx, test_idx)
    if limit_episodes:                 # cheap iteration (smoke runs); the rollout probe
        train_idx = train_idx[:limit_episodes]      # has the same knob, so one command
        test_idx = test_idx[:limit_episodes]        # can limit both methods

    X_tr, Yp_tr, Yf_tr = collect_split(wm, val, train_idx, device)
    X_te, Yp_te, Yf_te = collect_split(wm, val, test_idx, device)
    print(f"  [{env}/{variant}] episodes: train={len(train_idx)} test={len(test_idx)} "
          f"| samples (frames): N_train={X_tr.shape[0]} N_test={X_te.shape[0]} "
          f"| feature_dim={X_tr.shape[1]} | mode={get_mode(wm)}")

    # Standardize X: fit on probe_train only, apply to both. (Y is left raw;
    # R^2 is scale-invariant.)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    rows = []
    for target_name, Y_tr, Y_te in (("pos", Yp_tr, Yp_te), ("full", Yf_tr, Yf_te)):
        out_dim = Y_tr.shape[1]

        # Ridge (multi-output natively), alpha chosen by internal CV.
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
        ridge.fit(X_tr, Y_tr)
        r2_r, mean_r = per_dim_r2(Y_te, ridge.predict(X_te))
        rows.append(dict(env=env, variant=variant, target=target_name,
                         probe_type="ridge", mean_r2=mean_r,
                         per_dim_r2=r2_r, n_test=int(X_te.shape[0]),
                         ckpt=os.path.basename(os.path.normpath(model_dir))))

        # Small MLP probe.
        mlp_pred = train_mlp(X_tr, Y_tr, X_te, out_dim, epochs=mlp_epochs, seed=seed)
        r2_m, mean_m = per_dim_r2(Y_te, mlp_pred)
        rows.append(dict(env=env, variant=variant, target=target_name,
                         probe_type="mlp", mean_r2=mean_m,
                         per_dim_r2=r2_m, n_test=int(X_te.shape[0]),
                         ckpt=os.path.basename(os.path.normpath(model_dir))))
    return rows


# ===========================================================================
# Rollout (ungrounded) probe
# ===========================================================================
# Question: does the model's OWN imagined rollout still linearly decode true
# state, and at which predicted timestep does that degrade?
#
# The encoded probe above encodes real frames. This one asks the world model to
# imagine the episode with VWorldModel.rollout() and probes the PREDICTED
# latents: one RidgeCV is fitted on every (episode, timestep) row of probe_train
# pooled together -- per-timestep fits would have one sample per episode per
# timestep, far too few to fit reliably -- and then evaluated per timestep on
# probe_test, giving an R^2(t) curve plus the pooled R^2 for comparability.
#
# The channel-projector checkpoints (trained on the server, loaded here) use the
# same naming convention as the projglobal ones above: four arms per env.

CHANNEL_MODELS = {
    "point_maze": {
        "baseline": "checkpoints/test/umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06",
        "straighten": "checkpoints/test/umaze_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/umaze_ttaggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "checkpoints/test/umaze_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "point_maze_medium": {
        "baseline": "checkpoints/test/medium_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06",
        "straighten": "checkpoints/test/medium_aggflattencos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/medium_ttaggflattenwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "checkpoints/test/medium_aggflattencos1e-1_aggflattenwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
    "pusht": {
        "baseline": "checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "straighten": "checkpoints/test/pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "p_reg": "checkpoints/test/pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
        "both": "checkpoints/test/pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
    },
}

MODEL_SOURCES = {
    "projglobal": MODELS,
    "channel": CHANNEL_MODELS,
}


def models_for(source):
    """The checkpoint table for --models {projglobal,channel}."""
    if source not in MODEL_SOURCES:
        raise RuntimeError("unknown --models %r (have %s)"
                           % (source, ", ".join(sorted(MODEL_SOURCES))))
    return MODEL_SOURCES[source]


def resolve_ckpt_in(table, env, variant):
    """resolve_ckpt against a chosen table (channel vs projglobal)."""
    if CLI_ARM_DIRS:
        return resolve_ckpt(env, variant)      # explicit --ckpts mapping wins
    if env not in table:
        raise RuntimeError("no %s checkpoints configured for env %r" % (variant, env))
    path = table[env][variant]
    if not path:
        raise RuntimeError(
            "%s/%s is an empty placeholder -- fill in the checkpoint path" % (env, variant))
    model_dir = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.isfile(os.path.join(model_dir, "hydra.yaml")):
        raise RuntimeError("Missing hydra.yaml in checkpoint dir: %s" % model_dir)
    if not os.path.isfile(os.path.join(model_dir, "checkpoints", "model_latest.pth")):
        raise RuntimeError("Missing checkpoints/model_latest.pth in checkpoint dir: %s" % model_dir)
    return model_dir


def episode_ids(val, indices):
    """Underlying trajectory ids for positional indices into the val subset.

    val is a TrajSubset over trajectories, so val.indices[i] is the real episode
    identity while a positional index is not: two positions can share it. The
    leakage guard below needs the identity, not the slot.
    """
    underlying = getattr(val, "indices", None)
    if underlying is None:
        return [int(i) for i in indices]
    return [int(underlying[i]) for i in indices]


def assert_disjoint_episodes(val, train_idx, test_idx):
    """Hard guard: no episode may contribute rows to both probe splits.

    Splitting pooled rows instead of episode identities would inflate every R^2
    silently, with nothing but "good" numbers to notice it by.
    """
    train_eps = episode_ids(val, train_idx)
    test_eps = episode_ids(val, test_idx)
    assert len(set(train_eps)) == len(train_eps), "probe_train uses an episode twice"
    assert len(set(test_eps)) == len(test_eps), "probe_test uses an episode twice"
    overlap = sorted(set(train_eps) & set(test_eps))
    assert not overlap, (
        "probe train/test episode ids overlap (%d shared, e.g. %s): splitting rows "
        "instead of episode identities inflates every R^2" % (len(overlap), overlap[:5]))
    return train_eps, test_eps


def rollout_horizon(act_steps, num_hist):
    """Predicted frames that still have a ground-truth state.

    rollout() returns act_steps + 1 frames: the first num_hist are real encodings of the
    history window, and the last one predicts a step PAST the final observed state, so it
    has no ground truth in the episode. What is left -- act_steps - num_hist frames -- is
    aligned one-to-one with the episode states.
    """
    return int(act_steps) - int(num_hist)


def choose_tmax(horizons, tmax=None):
    """Common horizon plus which episodes are dropped for being shorter.

    horizons: {episode_id: predicted frames available}. Default is min(horizons),
    so nothing is dropped unless --tmax asks for longer than an episode offers.
    """
    if not horizons:
        raise RuntimeError("no episodes available for the rollout probe")
    limit = int(tmax) if tmax else min(horizons.values())
    kept = sorted(eid for eid, h in horizons.items() if h >= limit)
    dropped = sorted(eid for eid, h in horizons.items() if h < limit)
    if not kept:
        raise RuntimeError("T_max=%d exceeds every episode horizon (max %d)"
                           % (limit, max(horizons.values())))
    return limit, kept, dropped


def retention(r2_first, r2_last):
    """R2(T_max) / R2(1) -- the reportable scalar, nan-safe."""
    if r2_first is None or r2_last is None:
        return float("nan")
    if r2_first != r2_first or abs(r2_first) < 1e-9:
        return float("nan")
    return float(r2_last) / float(r2_first)


def model_step_inputs(obs, act, state, frameskip):
    """One val trajectory -> the model's own time base.

    Val items are raw trajectories. The training slice builder
    (datasets/traj_dset.TrajSlicerDataset) strides frames by `frameskip` and
    concatenates that many raw actions into each step, so mirror that here:
    rollout() and encode_act expect `frameskip * action_dim` per step, and obs
    and act must describe the SAME number of steps (not act = obs - 1).
    """
    visual, proprio = obs["visual"], obs["proprio"]
    idx = list(range(0, visual.shape[0], frameskip))
    n_steps = len(idx)
    if act.shape[0] // frameskip < n_steps:
        n_steps = act.shape[0] // frameskip
        idx = idx[:n_steps]
    obs_ms = {"visual": visual[idx], "proprio": proprio[idx]}
    state_ms = state[idx]
    n_raw = n_steps * frameskip
    act_ms = act[:n_raw]
    act_ms = act_ms.reshape(n_steps, frameskip, -1).reshape(n_steps, -1)
    return obs_ms, act_ms, state_ms


def rollout_predictions(wm, obs_ms, act_ms, state_ms, device, mode, num_hist, chunk=None):
    """Pooled latents for the PREDICTED steps only, plus the aligned true states.

    obs_0 is the first num_hist frames (the predictor needs its full history
    window -- one frame would be malformed), act is the episode's real recorded
    action sequence, and everything before num_hist in the output is a real
    encoding rather than a prediction, so it is dropped.
    """
    n_obs = obs_ms["visual"].shape[0]
    assert n_obs == act_ms.shape[0], (
        "episode time bases disagree: %d obs frames vs %d action steps"
        % (n_obs, act_ms.shape[0]))
    assert n_obs > num_hist, "episode has %d steps, need more than num_hist=%d" % (n_obs, num_hist)
    obs_0 = {k: v[:num_hist].unsqueeze(0).to(device) for k, v in obs_ms.items()}
    with torch.no_grad():
        z_obses, _z_full = wm.rollout(obs_0, act_ms.unsqueeze(0).to(device))
    feats = pool_features(wm, z_obses["visual"], mode)   # (1, n_steps + 1, D)
    # [num_hist:] drops the real encodings; [:-1] drops the rollout frame that has no
    # ground-truth state (it predicts one step beyond the end of the episode).
    pooled = to_numpy(feats[0])[num_hist:-1]
    gt = to_numpy(state_ms)[num_hist:]
    assert pooled.shape[0] == gt.shape[0] == n_obs - num_hist, (
        "rollout gave %d usable frames but there are %d ground-truth states "
        "(n_obs=%d, num_hist=%d)" % (pooled.shape[0], gt.shape[0], n_obs, num_hist))
    return pooled, gt


def episode_horizons(val, indices, num_hist, frameskip):
    """Cheap pre-pass: predicted frames per episode, without loading frames."""
    out = {}
    for i in indices:
        try:
            n_raw = int(val.get_seq_length(i))
        except Exception:
            n_raw = int(val[i][1].shape[0]) * frameskip
        steps = len(range(0, max(n_raw, 0), frameskip))
        out[episode_ids(val, [i])[0]] = rollout_horizon(steps, num_hist)
    return out


def collect_rollout_split(wm, val, indices, device, mode, num_hist, frameskip, tmax,
                          verbose=False):
    """(n_ep, tmax, D) predicted features and (n_ep, tmax, 2) true positions.

    Every episode contributes exactly tmax rows for the same aligned t, so the
    per-timestep R^2 is computed on a consistent slice rather than a mix of
    different-length episodes.
    """
    Xs, Ys, eps = [], [], []
    for i in indices:
        item = val[i]
        obs, act, state = item[0], item[1], item[2]
        obs_ms, act_ms, state_ms = model_step_inputs(obs, act, state, frameskip)
        pooled, gt = rollout_predictions(wm, obs_ms, act_ms, state_ms, device, mode, num_hist)
        eid = episode_ids(val, [i])[0]
        if pooled.shape[0] < tmax:
            if verbose:
                print("    ep %d: %d predicted frames < T_max=%d, skipped"
                      % (eid, pooled.shape[0], tmax))
            continue
        Xs.append(pooled[:tmax])
        Ys.append(gt[:tmax, :2])   # position = first 2 state dims, as the encoded probe
        eps.append(eid)
        if verbose:
            print("    ep %d: %d model steps -> %d predicted frames (using %d)"
                  % (eid, act_ms.shape[0], pooled.shape[0], tmax))
    if not Xs:
        raise RuntimeError("no episode reached T_max=%d" % tmax)
    return (np.stack(Xs, 0).astype(np.float32),
            np.stack(Ys, 0).astype(np.float32), eps)


def fit_rollout_probe(X_tr, Y_tr, X_te, Y_te, tmax):
    """ONE ridge fit on all pooled train rows, read out per timestep.

    Deliberately not one fit per timestep: one sample per episode per timestep is
    far too few to fit reliably. A single fit calibrated on every timestep at
    once, evaluated slice by slice, is what turns R^2(t) into a signal about
    degradation rather than a curve of noise. Standardization is fitted on the
    train rows only.
    """
    D = X_tr.shape[-1]
    scaler = StandardScaler().fit(X_tr.reshape(-1, D))
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    ridge.fit(scaler.transform(X_tr.reshape(-1, D)), Y_tr.reshape(-1, Y_tr.shape[-1]))

    def predict(X2d):
        return ridge.predict(scaler.transform(X2d))

    def r2_or_nan(y_true, y_pred):
        if y_true.shape[0] < 2:
            return float("nan")
        return float(np.nanmean(r2_score(y_true, y_pred, multioutput="raw_values")))

    pooled = r2_or_nan(Y_te.reshape(-1, Y_te.shape[-1]), predict(X_te.reshape(-1, X_te.shape[-1])))
    per_t = [r2_or_nan(Y_te[:, t, :], predict(X_te[:, t, :])) for t in range(tmax)]
    return pooled, per_t, scaler, ridge


def run_variant_rollout(env, variant, device, table, test_frac, seed, tmax=None,
                        limit_episodes=0, verbose=False):
    """Rollout-probe rows + a summary dict for one (env, variant)."""
    model_dir = resolve_ckpt_in(table, env, variant)
    wm, train_cfg = load_model(model_dir, device)
    wm.eval()
    redirect_dset_paths(train_cfg)
    val = load_val_dset(train_cfg)
    mode = get_mode(wm)
    num_hist = int(getattr(wm, "num_hist", getattr(train_cfg, "num_hist", 1)))
    frameskip = int(getattr(wm, "frameskip", getattr(train_cfg, "frameskip", 1)))

    train_idx, test_idx = split_episode_indices(len(val), test_frac=test_frac, seed=seed)
    assert_disjoint_episodes(val, train_idx, test_idx)
    if limit_episodes:
        train_idx, test_idx = train_idx[:limit_episodes], test_idx[:limit_episodes]

    horizons = episode_horizons(val, list(train_idx) + list(test_idx), num_hist, frameskip)
    t_max, kept, dropped = choose_tmax(horizons, tmax)
    kept_set = set(kept)
    train_idx = [i for i in train_idx if episode_ids(val, [i])[0] in kept_set]
    test_idx = [i for i in test_idx if episode_ids(val, [i])[0] in kept_set]
    print("  [%s/%s] mode=%s num_hist=%d frameskip=%d | episodes train=%d test=%d"
          % (env, variant, mode, num_hist, frameskip, len(train_idx), len(test_idx)))
    print("              horizons min=%d max=%d | T_max=%d | dropped (shorter than T_max)=%d%s"
          % (min(horizons.values()), max(horizons.values()), t_max, len(dropped),
             (" e.g. %s" % dropped[:5]) if dropped else ""))

    t0 = time.time()
    X_tr, Y_tr, eps_tr = collect_rollout_split(wm, val, train_idx, device, mode, num_hist,
                                               frameskip, t_max, verbose)
    X_te, Y_te, eps_te = collect_rollout_split(wm, val, test_idx, device, mode, num_hist,
                                               frameskip, t_max, verbose)
    pooled, per_t, _scaler, _ridge = fit_rollout_probe(X_tr, Y_tr, X_te, Y_te, t_max)
    dt = time.time() - t0
    # Rollouts run the predictor once per step, so this is the slow part of the probe:
    # print the cost so the full run can be planned (seconds per episode, both splits).
    print("              rollout+fit: %.1fs for %d episodes (%.2fs/episode), %d rows"
          % (dt, len(eps_tr) + len(eps_te), dt / max(1, len(eps_tr) + len(eps_te)),
             X_tr.shape[0] * t_max + X_te.shape[0] * t_max))

    ckpt = os.path.basename(os.path.normpath(model_dir))
    rows = [dict(env=env, variant=variant, feature_source="rollout", t=t + 1, r2=per_t[t],
                 n_test=int(X_te.shape[0]), ckpt=ckpt) for t in range(t_max)]
    rows.append(dict(env=env, variant=variant, feature_source="rollout", t="all", r2=pooled,
                     n_test=int(X_te.shape[0] * t_max), ckpt=ckpt))
    summary = dict(env=env, variant=variant, feature_source="rollout", t_max=t_max,
                   r2_t1=per_t[0], r2_tmax=per_t[-1],
                   retention=retention(per_t[0], per_t[-1]), r2_all=pooled,
                   n_train_eps=len(eps_tr), n_test_eps=len(eps_te), n_dropped=len(dropped),
                   mode=mode, num_hist=num_hist, frameskip=frameskip, ckpt=ckpt)
    return rows, summary


def add_shared_horizon(summaries, curve_rows):
    """Retention at the horizon every arm reaches, for an apples-to-apples table.

    Episode lengths (and therefore T_max) can differ per arm, in which case
    comparing R2(T_max)/R2(1) across arms compares different horizons; the shared
    column is the honest one.
    """
    shared = min(s["t_max"] for s in summaries)
    for s in summaries:
        by_t = {r["t"]: r["r2"] for r in curve_rows
                if r["variant"] == s["variant"] and not isinstance(r["t"], str)}
        s["shared_horizon"] = shared
        s["shared_r2"] = by_t.get(shared, float("nan"))
        s["shared_retention"] = retention(by_t.get(1), by_t.get(shared))
    return shared


def print_rollout_table(summaries, shared):
    print()
    print("=== Rollout probe: R^2 of true position from PREDICTED latents ===")
    print("%-18s %-11s %6s %8s %9s %9s %10s %9s %6s %6s %6s"
          % ("env", "variant", "T_max", "R2(t=1)", "R2(T_max)", "retention",
             "R2(shared)", "ret@sh", "n_tr", "n_te", "drop"))
    for s in summaries:
        print("%-18s %-11s %6d %8.4f %9.4f %9.4f %10.4f %9.4f %6d %6d %6d"
              % (s["env"], s["variant"], s["t_max"], s["r2_t1"], s["r2_tmax"],
                 s["retention"], s["shared_r2"], s["shared_retention"],
                 s["n_train_eps"], s["n_test_eps"], s["n_dropped"]))
    print("shared horizon = min T_max across arms = %d; pooled R^2 (t=all) is in the CSV" % shared)


def write_rollout_csvs(env, curve_rows, summaries, outdir, label="", seed=0, test_frac=0.0):
    """Long-format curve table + one summary row per arm.

    Namespaced by (env, label) so a second recipe cannot overwrite the first -- the
    global / flatten / aggmlp groups of one env used to write the same file. The
    label/ckpt/seed/test_frac columns are what analysis/probe_report.py reads to
    describe each figure without re-running anything.
    """
    if label:
        base = "rollout_probe_%s_%s" % (env, label)
        curve_path = os.path.join(outdir, base + "_curves.csv")
        summary_path = os.path.join(outdir, base + "_summary.csv")
    else:
        curve_path = os.path.join(outdir, "rollout_probe_curves.csv")
        summary_path = os.path.join(outdir, "rollout_probe_summary.csv")
    fields = ["env", "label", "ckpt", "variant", "feature_source", "t", "r2", "n_test"]
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in curve_rows:
            w.writerow([r["env"], label, r.get("ckpt", ""), r["variant"],
                        r["feature_source"], r["t"],
                        "%.6f" % r["r2"] if r["r2"] == r["r2"] else "nan", r["n_test"]])
    sfields = ["env", "label", "ckpt", "variant", "feature_source", "seed", "test_frac",
               "t_max", "r2_t1", "r2_tmax", "retention",
               "r2_all", "shared_horizon", "shared_r2", "shared_retention",
               "n_train_eps", "n_test_eps", "n_dropped", "mode", "num_hist", "frameskip"]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sfields)
        for s in summaries:
            w.writerow([s["env"], label, s.get("ckpt", ""), s["variant"],
                        s["feature_source"], seed, "%.4f" % test_frac, s["t_max"],
                        "%.6f" % s["r2_t1"], "%.6f" % s["r2_tmax"], "%.6f" % s["retention"],
                        "%.6f" % s["r2_all"], s["shared_horizon"], "%.6f" % s["shared_r2"],
                        "%.6f" % s["shared_retention"], s["n_train_eps"], s["n_test_eps"],
                        s["n_dropped"], s["mode"], s["num_hist"], s["frameskip"]])
    return curve_path, summary_path


def plot_rollout_curves(env, curve_rows, summaries, outdir, label=""):
    """One line per arm: R^2(t) over the predicted rollout, shared axes."""
    colors = {"baseline": "tab:gray", "straighten": "tab:blue",
              "p_reg": "tab:orange", "both": "tab:green"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ys_all = []
    for s in summaries:
        pts = [r for r in curve_rows if r["variant"] == s["variant"]
               and not isinstance(r["t"], str)]
        if not pts:
            continue
        xs = [r["t"] for r in pts]
        ys = [r["r2"] for r in pts]
        ys_all += [v for v in ys if v == v]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color=colors.get(s["variant"]),
                label=s["variant"])
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
    # Short axis text only: the arm wording, the history length, the per-arm T_max and
    # what "ungrounded" means live in the .md report beside every figure
    # (analysis/probe_report.py), which is why the old sentence-length title is gone.
    ax.set_xlabel("predicted rollout step t", fontsize=10)
    ax.set_ylabel("R$^2$ (ridge, position)", fontsize=10)
    ax.set_title("ungrounded rollout - %s" % ("%s / %s" % (env, label) if label else env),
                 fontsize=11)
    ax.tick_params(labelsize=9)
    if ys_all:
        ax.set_ylim(min(0.0, min(ys_all) - 0.05), 1.0)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, title="arm", title_fontsize=9)
    path = os.path.join(outdir, plot_stem("rollout_probe", env, label) + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main_rollout(args, device):
    """--feature-source rollout: the ungrounded-prediction probe for one env+recipe."""
    table = models_for(args.models)
    env = args.env
    label = args.label or ""
    curve_rows, summaries, skipped = [], [], []
    for variant in VARIANT_ORDER:
        try:
            rows, summary = run_variant_rollout(
                env, variant, device, table, args.test_frac, args.seed,
                tmax=args.tmax, limit_episodes=args.limit_episodes, verbose=args.verbose)
        except RuntimeError as exc:
            print("[error] %s" % exc)
            skipped.append(variant)
            continue
        curve_rows += rows
        summaries.append(summary)
    if not summaries:
        raise SystemExit("No checkpoint produced rollout results; nothing to report.")
    shared = add_shared_horizon(summaries, curve_rows)
    print_rollout_table(summaries, shared)
    os.makedirs(args.outdir, exist_ok=True)
    curve_path, summary_path = write_rollout_csvs(env, curve_rows, summaries, args.outdir,
                                                  label=label, seed=args.seed,
                                                  test_frac=args.test_frac)
    print("\nSaved curve table    -> %s" % curve_path)
    print("Saved summary table  -> %s" % summary_path)
    png = plot_rollout_curves(env, curve_rows, summaries, args.outdir, label=label)
    if png:
        print("Saved curve figure   -> %s" % png)
    if skipped:
        print("skipped (no usable checkpoint): %s" % ", ".join(skipped))


def plot_stem(method, env, label):
    """Output stem for one (method, env, recipe) figure.

    The recipe is part of the name on purpose: the global, flatten and aggmlp
    checkpoints of one env used to overwrite each other's FIGURE (both plots were
    keyed by env only) and their CSV rows were indistinguishable.
    """
    return "_".join([method, env] + ([label] if label and label != "custom" else []))


def plot_bar(env, rows, outdir, label=""):
    """Bar plot of mean R^2 (ridge, position) grouped by variant."""
    ridge_pos = [r for r in rows if r["probe_type"] == "ridge" and r["target"] == "pos"]
    if not ridge_pos:
        return None
    variants = [r["variant"] for r in ridge_pos]
    means = [r["mean_r2"] for r in ridge_pos]
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.bar(variants, means, color="tab:blue", alpha=0.85)
    ax.set_ylabel("mean R$^2$ (ridge, position)")
    # One short line only. The long explanation belongs in the .md report written
    # next to every figure (analysis/probe_report.py), where it cannot overflow.
    ax.set_title("grounded probe - %s" % ("%s / %s" % (env, label) if label else env),
                 fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=9)
    for i, m in enumerate(means):
        ax.text(i, min(m + 0.02, 0.98), f"{m:.3f}", ha="center", fontsize=9)
    path = os.path.join(outdir, plot_stem("linear_probe", env, label) + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Linear probes on frozen world-model latents")
    ap.add_argument("--env", default="pusht", choices=sorted(MODELS.keys()),
                    help="env to probe (default pusht)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"),
                    help="output dir (default <repo>/analysis_outputs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="probe split seed (default 0)")
    ap.add_argument("--test-frac", type=float, default=0.3,
                    help="fraction of val episodes held out for the probe (default 0.3)")
    ap.add_argument("--mlp-epochs", type=int, default=200,
                    help="full-batch Adam epochs for the MLP probe (default 200)")
    ap.add_argument("--feature-source", choices=["encoded", "rollout"], default="encoded",
                    help="encoded probes real-frame encodings (default); rollout probes the ungrounded predicted rollout with a per-timestep R2(t) curve")
    ap.add_argument("--models", choices=["projglobal", "channel"], default="projglobal",
                    help="legacy checkpoint set, used only when --ckpts is not given")
    ap.add_argument("--tmax", type=int, default=0,
                    help="rollout horizon; 0 = shortest episode horizon (drop nothing)")
    ap.add_argument("--limit-episodes", type=int, default=0,
                    help="cap episodes per split (0 = all; for smoke runs -- applies to "
                         "both the grounded and the rollout probe)")
    ap.add_argument("--verbose", action="store_true", help="per-episode horizons")
    # ---- explicit group (what analysis/run_probes.py drives) -----------------
    ap.add_argument("--ckpts", default="",
                    help="explicit arm=DIR mapping for the group being probed, comma-separated "
                         "in any order, e.g. 'baseline=checkpoints/test/a,straighten=...'. "
                         "Wins over --models; the label defaults to the checkpoint's recipe.")
    ap.add_argument("--label", default="",
                    help="recipe label used in the output names, e.g. aggmlp "
                         "(default: from the first --ckpts checkpoint's hydra.yaml)")
    ap.add_argument("--list", action="store_true",
                    help="print the checkpoint groups discovered under --ckpt-root and exit")
    ap.add_argument("--ckpt-root", default="",
                    help="root scanned by --list (default <repo>/checkpoints/test)")
    ap.add_argument("--envs", default="", help="with --list: comma list of envs to keep")
    ap.add_argument("--recipes", default="", help="with --list: comma list of recipes to keep")
    args = ap.parse_args()

    if args.list:
        root = args.ckpt_root or os.path.join(REPO, "checkpoints", "test")
        groups, skipped, _ = probe_ckpts.discover(
            root, [e for e in args.envs.split(",") if e] or None,
            [r for r in args.recipes.split(",") if r] or None)
        print(probe_ckpts.format_table(groups, skipped, root, verbose=True))
        return

    if args.ckpts:
        for item in args.ckpts.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit("--ckpts entries must be arm=DIR, got %r" % item)
            arm, path = (part.strip() for part in item.split("=", 1))
            if arm not in VARIANT_ORDER:
                raise SystemExit("--ckpts arm %r not in %s" % (arm, ", ".join(VARIANT_ORDER)))
            path = path if os.path.isabs(path) else os.path.join(REPO, path)
            if not os.path.isdir(path):
                raise SystemExit("--ckpts %s=%s is not a directory" % (arm, path))
            CLI_ARM_DIRS[arm] = path
        if not args.label:
            args.label = probe_ckpts.describe(next(iter(CLI_ARM_DIRS.values())))[1]
        print("explicit checkpoints (%s): %s"
              % (args.label, ", ".join("%s=%s" % (a, os.path.basename(d))
                                       for a, d in sorted(CLI_ARM_DIRS.items()))))

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  env: {args.env}")
    args.tmax = args.tmax or None
    if args.feature_source == "rollout":
        main_rollout(args, device)
        return

    rows = []
    for variant in VARIANT_ORDER:
        try:
            rows += run_variant(args.env, variant, device, args.test_frac, args.seed,
                                args.mlp_epochs, table=models_for(args.models),
                                limit_episodes=args.limit_episodes)
        except RuntimeError as e:
            print(f"[error] {e}")

    if not rows:
        raise SystemExit("No checkpoint produced results; nothing to report.")

    # ---- printed table ------------------------------------------------------
    print("\n=== Linear probe results ===")
    print(f"{'env':18s} {'variant':12s} {'target':6s} {'probe':6s} "
          f"{'mean_r2':>9s} {'per_dim_r2':>30s} {'n_test':>7s}")
    for r in rows:
        print(f"{r['env']:18s} {r['variant']:12s} {r['target']:6s} "
              f"{r['probe_type']:6s} {r['mean_r2']:9.4f} "
              f"{fmt_dims(r['per_dim_r2']):>30s} {r['n_test']:7d}")

    # ---- CSV (stdlib; pandas is not in the analysis env) --------------------
    if args.label:                     # namespaced: one file per (env, recipe)
        csv_path = os.path.join(
            args.outdir, "linear_probe_results_%s_%s.csv" % (args.env, args.label))
    else:
        csv_path = os.path.join(args.outdir, "linear_probe_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "label", "ckpt", "variant", "target", "probe_type",
                         "seed", "test_frac", "mean_r2", "per_dim_r2", "n_test"])
        for r in rows:
            writer.writerow([r["env"], args.label, r.get("ckpt", ""), r["variant"],
                             r["target"], r["probe_type"], args.seed,
                             "%.4f" % args.test_frac, f"{r['mean_r2']:.6f}",
                             fmt_dims(r["per_dim_r2"]), r["n_test"]])
    print(f"\nSaved results table -> {csv_path}")

    png = plot_bar(args.env, rows, args.outdir, label=args.label)
    if png:
        print(f"Saved bar plot      -> {png}")


if __name__ == "__main__":
    main()
