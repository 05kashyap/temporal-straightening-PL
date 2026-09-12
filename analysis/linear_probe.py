#!/usr/bin/env python
"""analysis/linear_probe.py
==========================
Linear probing of frozen world-model latents for true agent state.

For each trained variant (baseline / straighten / p_reg / both) of one selected
environment, this script:
  1. loads the frozen world model + its train config (load_model),
  2. gets the held-out val base episodes (load_val_dset),
  3. splits those episodes 70/30 into probe_train / probe_test (fixed seed=0;
     this is a *further* split of the already-held-out val set, purely for the
     probe's own fitting),
  4. encodes every frame with the frozen encoder (encode_episode) and pools the
     latents exactly the way the training regularizer did (get_mode),
  5. fits a ridge probe and a small MLP probe to predict (a) position = first 2
     state dims, and (b) the FULL state vector, then reports held-out per-dim
     and mean R^2.

Usage:
    python analysis/linear_probe.py --env pusht
    python analysis/linear_probe.py --env point_maze
    python analysis/linear_probe.py --env point_maze_medium
    python analysis/linear_probe.py --env wall
Options:
    --env ENV       env to probe (default pusht)
    --outdir DIR    output dir (default <repo>/analysis_outputs, matching
                    curvature_analysis.py / curvature_distributions.py)
    --device DEV    torch device (default cuda)
    --seed INT      probe train/test split seed (default 0)
    --test-frac F   fraction of val episodes held out for the probe (default 0.3)
    --mlp-epochs N  full-batch Adam epochs for the MLP probe (default 200)

Outputs (all under --outdir):
    linear_probe_results.csv   one row per (variant, target, probe_type)
    linear_probe_<env>.png     bar plot: mean R^2 (ridge, position target) per variant

Checkpoint paths live in the MODELS dict below. Entries left as empty strings are
placeholders; when the script reaches one it raises a clear RuntimeError and skips
that variant (so a partially-downloaded env still yields results for the variants
that ARE present).
"""

import argparse
import csv
import os
import sys
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


def resolve_ckpt(env, variant):
    """Return the absolute checkpoint run dir, raising a clear error on placeholders."""
    path = MODELS[env][variant]
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


def run_variant(env, variant, device, test_frac, seed, mlp_epochs=200):
    """Fit ridge + MLP probes for one (env, variant); return result rows."""
    model_dir = resolve_ckpt(env, variant)
    wm, train_cfg = load_model(model_dir, device)
    wm.eval()
    redirect_dset_paths(train_cfg)
    val = load_val_dset(train_cfg)

    train_idx, test_idx = split_episode_indices(len(val), test_frac=test_frac, seed=seed)
    assert set(train_idx).isdisjoint(set(test_idx)), "probe train/test splits overlap"

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
                         per_dim_r2=r2_r, n_test=int(X_te.shape[0])))

        # Small MLP probe.
        mlp_pred = train_mlp(X_tr, Y_tr, X_te, out_dim, epochs=mlp_epochs, seed=seed)
        r2_m, mean_m = per_dim_r2(Y_te, mlp_pred)
        rows.append(dict(env=env, variant=variant, target=target_name,
                         probe_type="mlp", mean_r2=mean_m,
                         per_dim_r2=r2_m, n_test=int(X_te.shape[0])))
    return rows


def plot_bar(env, rows, outdir):
    """Bar plot of mean R^2 (ridge, position) grouped by variant."""
    ridge_pos = [r for r in rows if r["probe_type"] == "ridge" and r["target"] == "pos"]
    if not ridge_pos:
        return None
    variants = [r["variant"] for r in ridge_pos]
    means = [r["mean_r2"] for r in ridge_pos]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, means, color="tab:blue", alpha=0.85)
    ax.set_ylabel("mean R^2 (ridge, position)")
    ax.set_title(f"Linear probe: {env} (position, ridge)")
    ax.set_ylim(0, 1)
    for i, m in enumerate(means):
        ax.text(i, min(m + 0.02, 0.98), f"{m:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, f"linear_probe_{env}.png")
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
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  env: {args.env}")

    rows = []
    for variant in VARIANT_ORDER:
        try:
            rows += run_variant(args.env, variant, device, args.test_frac, args.seed, args.mlp_epochs)
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
    csv_path = os.path.join(args.outdir, "linear_probe_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", "variant", "target", "probe_type",
                         "mean_r2", "per_dim_r2", "n_test"])
        for r in rows:
            writer.writerow([r["env"], r["variant"], r["target"], r["probe_type"],
                             f"{r['mean_r2']:.6f}", fmt_dims(r["per_dim_r2"]), r["n_test"]])
    print(f"\nSaved results table -> {csv_path}")

    png = plot_bar(args.env, rows, args.outdir)
    if png:
        print(f"Saved bar plot      -> {png}")


if __name__ == "__main__":
    main()
