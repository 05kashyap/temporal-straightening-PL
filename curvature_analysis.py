#!/usr/bin/env python
"""
curvature_analysis.py
=====================
Offline analysis of the two-thirds power-law (kappa ~ s^-1/3) and the
action-space loss landscape, for the PointMaze-umaze and PushT world models.

Plots (one set per env):
  1. pooled log-log scatter of speed s_t vs curvature kappa_t with a fitted
     slope + bootstrap CI (the core two-thirds check: expect ~ -1/3 for
     two-thirds/both models, ~0/unpredictable for baseline/straighten-only).
  2. aligned time series of speed s_t (top) and curvature kappa_t (bottom)
     for one representative episode (curvature peaks should line up with
     speed troughs for a law-satisfying model).
  3. action-space loss-landscape heatmap (PushT only, paper Fig. 4): fix the
     first action (ax, ay), GD-optimize the remaining actions, plot the
     minimum attainable terminal-goal loss per initial action.

Kappa/speed are computed exactly like the training regularizer
(VWorldModel._two_thirds_residual): s=(||v1||+||v2||)/2, theta=acos(cos(v1,v2)),
kappa=theta/(s+eps); pooled fit is np.polyfit(log_kappa, log_s, 1) i.e. slope
of log_s vs log_kappa (the two-thirds law is log_s + (1/3)log_kappa = const).

Usage:
  python curvature_analysis.py --env point_maze --env pusht [options]
Options:
  --max-episodes N     held-out episodes used for the slope/timeseries
                       (default 1 = single episode; plots are meant to be quick
                       qualitative checks, raise N only if you want a pooled slope
                       with more statistical power and a meaningful bootstrap CI)
  --grid N             landscape grid size per axis (default 13)
  --opt-steps N        GD steps per landscape point (default 80)
  --action-range R     landscape grid half-range in normalized action units (default 2.0)
  --bootstrap N        bootstrap iterations for the slope CI (default 1000)
  --outdir DIR         output directory (default analysis_outputs)
  --device DEV         torch device (default cuda)
"""
import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from models.dino import DinoV2Encoder  # noqa: E402
import models.visual_world_model  # noqa: E402,F401  (registers VWorldModel for hydra)
from planning.objectives import create_objective_fn  # noqa: E402

ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]

# model-name suffix -> display label (order = plotting order)
MODELS = {
    "point_maze": [
        ("umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05", "baseline"),
        ("umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05", "straightening"),
        ("umaze_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05", "two-thirds"),
        ("umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05", "both"),
    ],
    "pusht": [
        ("pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05", "baseline"),
        ("pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05", "straightening"),
        ("pusht_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05", "two-thirds"),
        ("pusht_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05", "both"),
    ],
}


def load_ckpt(snapshot_path, device):
    # touch DinoV2Encoder so dinov2 is importable for unpickling
    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_dir, device):
    """Load a world model from a checkpoint dir (same recipe as plan.load_model)."""
    cfg_path = os.path.join(model_dir, "hydra.yaml")
    with open(cfg_path, "r") as f:
        train_cfg = OmegaConf.load(f)
    model_ckpt = Path(model_dir) / "checkpoints" / "model_latest.pth"
    result = load_ckpt(model_ckpt, device)
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError(f"Predictor not found in checkpoint {model_ckpt}")
    result["decoder"] = None
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=train_cfg.num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model, train_cfg


def load_val_dset(train_cfg):
    """Return the held-out (val) base dataset (full episodes, not sliced windows)."""
    _, traj_dset = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    return traj_dset["valid"]


def encode_episode(wm, obs, device, chunk=32):
    """Encode a full real episode end-to-end (encoder only, no predictor).

    obs['visual']: (T, 3, H, W) in [0,1] (already default_transform'd by the dset).
    Returns visual embeddings (1, T, p, d).
    """
    visual = obs["visual"]
    proprio = obs["proprio"]
    T = visual.shape[0]
    feats = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        with torch.no_grad():
            z = wm.encode_obs(
                {
                    "visual": visual[s:e].unsqueeze(0).to(device),
                    "proprio": proprio[s:e].unsqueeze(0).to(device),
                }
            )
        feats.append(z["visual"])
    return torch.cat(feats, dim=1)


def curvature_speed_series(feats, mode="aggcos", wm=None):
    """Per-frame speed and curvature (exactly the regularizer's definition).

    feats: (1, T, p, d). Returns (s, kappa) each (T-2,) after optional agg.
    """
    with torch.no_grad():
        if mode == "aggcos":
            b, t, p, d = feats.shape
            feats = wm.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
        v1 = feats[:, 1:-1] - feats[:, :-2]
        v2 = feats[:, 2:] - feats[:, 1:-1]
        s1, s2 = v1.norm(dim=-1), v2.norm(dim=-1)
        cos = F.cosine_similarity(v1, v2, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        avg_len = (s1 + s2) / 2
        kappa = theta / (avg_len + 1e-6)
        return avg_len.squeeze(0), kappa.squeeze(0)


def get_mode(wm):
    """Same pooling the regularizer used for this model; fall back by encoder type."""
    m = getattr(wm, "twothirds_mode", None) or getattr(wm, "curvature_mode", None)
    if m:
        return m
    return "aggcos" if getattr(wm.encoder, "projector_name", "") == "channel" else "cos"


def fit_slope(logk, logs):
    return np.polyfit(logk, logs, 1)


def r2_of_fit(logk, logs, p):
    pred = np.polyval(p, logk)
    ss_res = np.sum((logs - pred) ** 2)
    ss_tot = np.sum((logs - logs.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def bootstrap_ci(ep_logk, ep_logs, n_iter, seed=0):
    """Bootstrap by resampling EPISODES (all points of each sampled episode)."""
    rng = np.random.RandomState(seed)
    n_ep = len(ep_logk)
    slopes = []
    for _ in range(n_iter):
        idx = rng.randint(0, n_ep, n_ep)
        k = np.concatenate([ep_logk[i] for i in idx])
        s = np.concatenate([ep_logs[i] for i in idx])
        slopes.append(np.polyfit(k, s, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return lo, hi, np.array(slopes)

def collect_episodes(env, wm, val, device, max_episodes, seed=0):
    """Encode up to max_episodes held-out episodes and return per-episode
    (log_s, log_kappa) arrays plus the raw (s, kappa) series of the first one.
    """
    mode = get_mode(wm)
    rng = np.random.RandomState(seed)
    n_ep = len(val)
    idxs = list(range(n_ep))
    rng.shuffle(idxs)
    ep_logk, ep_logs = [], []
    ts_s = ts_k = None
    n_used = 0
    for i in idxs:
        if n_used >= max_episodes:
            break
        obs, _, _, _ = val[i]
        T = obs["visual"].shape[0]
        if T < 4:
            continue
        feats = encode_episode(wm, obs, device)
        s, k = curvature_speed_series(feats, mode=mode, wm=wm)
        s = s.reshape(-1).cpu().numpy()
        k = k.reshape(-1).cpu().numpy()
        logk, logs = np.log(k), np.log(s)
        finite = np.isfinite(logk) & np.isfinite(logs)
        if not finite.any():
            continue
        ep_logk.append(logk[finite])
        ep_logs.append(logs[finite])
        if ts_s is None:
            ts_s, ts_k = s, k
        n_used += 1
    if not ep_logk:
        raise RuntimeError("No usable episodes found.")
    return ep_logk, ep_logs, ts_s, ts_k, mode, n_used


def plot_slope(env, results, outdir):
    """results: list of dicts (label, mode, slope, ci, r2, n, logk, logs)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for ax, r in zip(axes.ravel(), results):
        logk, logs = r["logk"], r["logs"]
        ax.hexbin(logk, logs, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        p = r["poly"]
        xs = np.linspace(logk.min(), logk.max(), 50)
        ax.plot(xs, np.polyval(p, xs), "r-", lw=2,
                label=f"slope = {r['slope']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]\nR² = {r['r2']:.3f}")
        ax.axhline(0, color="k", ls=":", lw=0.7)
        ax.set_title(f"{env} · {r['label']}  (mode={r['mode']}, n={r['n']})")
        ax.set_xlabel(r"$\log \kappa_t$")
        ax.set_ylabel(r"$\log s_t$")
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(f"{env}: pooled log-log slope of speed vs curvature (expect −1/3 for two-thirds)",
                 fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, f"slope_{env}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_timeseries(env, results, outdir):
    """results: list of dicts with ts_s, ts_k per variant (one episode each)."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, r in enumerate(results):
        s, k = r["ts_s"], r["ts_k"]
        t = np.arange(len(s))
        axes[0, j].plot(t, s, color="tab:blue", lw=1.2)
        axes[0, j].set_ylabel("speed $s_t$")
        axes[0, j].set_title(f"{env} · {r['label']}")
        axes[1, j].plot(t, k, color="tab:red", lw=1.2)
        axes[1, j].set_ylabel("curvature $\\kappa_t$")
        axes[1, j].set_xlabel("timestep")
        axes[1, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle(f"{env}: aligned speed/curvature time series (one episode) — "
                 "curvature peaks should line up with speed troughs", fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, f"timeseries_{env}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def loss_landscape(wm, train_cfg, val, device, goal_H=25, grid=13,
                   opt_steps=80, lr=0.1, action_range=2.0):
    """Paper Fig. 4: fix first action (ax, ay), GD-optimize the rest of the
    25-step horizon, return (ax_grid, grid_loss) of the min terminal loss."""
    frameskip = int(train_cfg.frameskip)
    raw_act_dim = int(val.action_dim)
    act_dim = raw_act_dim * frameskip          # latent action dim (5 frames x 2)
    H_latent = goal_H // frameskip             # 5 latent steps
    am = val.action_mean.to(device)
    ast = val.action_std.to(device)
    norm_zero = torch.cat([-am / ast] * frameskip)  # (act_dim,) normalized zero action

    # pick a val episode long enough for start + goal
    idx = next(i for i in range(len(val)) if val.get_seq_length(i) >= goal_H + 1)
    obs, _, _, _ = val[idx]
    obs_0 = {k: v[0:1].unsqueeze(0).to(device) for k, v in obs.items()}   # (1,1,3,H,W)
    obs_g = {k: v[goal_H:goal_H + 1].unsqueeze(0).to(device) for k, v in obs.items()}

    objective_fn = create_objective_fn(alpha=1, base=2, mode="last")
    with torch.no_grad():
        z_obs_g = wm.encode_obs(obs_g)

    axs = np.linspace(-action_range, action_range, grid)
    grid_loss = np.full((grid, grid), np.nan)
    for i in range(grid):
        for j in range(grid):
            actions = torch.zeros(1, H_latent, act_dim, device=device)
            actions[0] = norm_zero.unsqueeze(0).expand(H_latent, act_dim)
            actions[0, 0, :2] = torch.tensor([axs[i], axs[j]], device=device)
            fixed = actions[0, 0, :2].clone()
            actions.requires_grad_(True)
            opt = torch.optim.Adam([actions], lr=lr)
            best = float("inf")
            for _ in range(opt_steps):
                opt.zero_grad()
                i_z_obses, _ = wm.rollout(obs_0=obs_0, act=actions)
                loss = objective_fn(i_z_obses, z_obs_g)
                lv = loss.item()
                if lv < best:
                    best = lv
                loss.mean().backward()
                opt.step()
                with torch.no_grad():
                    actions.data[0, 0, :2] = fixed
            grid_loss[i, j] = best
    return axs, grid_loss, idx

def plot_landscape(env, results, outdir):
    """results: list of dicts (label, axs, grid_loss) for the pusht variants."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    all_loss = np.concatenate([r["grid_loss"].ravel() for r in results])
    vmin, vmax = np.nanmin(all_loss), np.nanpercentile(all_loss, 99)
    for ax, r in zip(axes.ravel(), results):
        im = ax.imshow(r["grid_loss"], origin="lower", aspect="auto",
                       extent=[r["axs"][0], r["axs"][-1], r["axs"][0], r["axs"][-1]],
                       cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"{env} · {r['label']}")
        ax.set_xlabel("first action $a_x$ (normalized)")
        ax.set_ylabel("first action $a_y$ (normalized)")
    fig.suptitle(f"{env}: min attainable terminal loss vs first action (darker = lower) — "
                 "landscape should be smoother/closer to convex after straightening", fontsize=13)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="min terminal loss")
    fig.tight_layout()
    path = os.path.join(outdir, f"landscape_{env}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def analyze_env(env, args, device):
    model_dir_base = os.path.join(REPO, "checkpoints", "test")
    outdir = os.path.join(args.outdir, env)
    os.makedirs(outdir, exist_ok=True)

    first_dir = os.path.join(model_dir_base, MODELS[env][0][0])
    _, train_cfg0 = load_model(first_dir, device)
    val = load_val_dset(train_cfg0)
    print(f"[{env}] val episodes: {len(val)}")

    results = []
    for suffix, label in MODELS[env]:
        model_dir = os.path.join(model_dir_base, suffix)
        wm, _ = load_model(model_dir, device)
        ep_logk, ep_logs, ts_s, ts_k, mode, n_used = collect_episodes(
            env, wm, val, device, args.max_episodes, seed=args.seed)
        logk = np.concatenate(ep_logk)
        logs = np.concatenate(ep_logs)
        poly = fit_slope(logk, logs)
        r2 = r2_of_fit(logk, logs, poly)
        lo, hi, _ = bootstrap_ci(ep_logk, ep_logs, args.bootstrap, seed=args.seed)
        res = {"label": label, "model": suffix, "mode": mode, "n": len(logk),
               "n_episodes": n_used, "slope": float(poly[0]), "intercept": float(poly[1]),
               "ci": [float(lo), float(hi)], "r2": float(r2),
               "poly": poly, "logk": logk, "logs": logs, "ts_s": ts_s, "ts_k": ts_k}
        results.append(res)
        print(f"  [{env}] {label:14s} slope={poly[0]:+.3f} "
              f"CI=[{lo:+.3f}, {hi:+.3f}] R2={r2:.3f} n={len(logk)} mode={mode}")

    slope_path = plot_slope(env, results, outdir)
    ts_path = plot_timeseries(env, results, outdir)

    summary = {r["label"]: {k: v for k, v in r.items()
                            if k not in ("poly", "logk", "logs", "ts_s", "ts_k")}
               for r in results}
    summary["_figures"] = {"slope": os.path.relpath(slope_path, args.outdir),
                           "timeseries": os.path.relpath(ts_path, args.outdir)}

    if env == "pusht":
        for r in results:
            model_dir = os.path.join(model_dir_base, r["model"])
            wm, train_cfg = load_model(model_dir, device)
            axs, grid_loss, ep_idx = loss_landscape(
                wm, train_cfg, val, device, grid=args.grid,
                opt_steps=args.opt_steps, action_range=args.action_range)
            r["axs"] = axs
            r["grid_loss"] = grid_loss
            r["landscape_episode"] = int(ep_idx)
        landscape_path = plot_landscape(env, results, outdir)
        summary["_figures"]["landscape"] = os.path.relpath(landscape_path, args.outdir)
        for r in results:
            gl = r["grid_loss"]
            summary[r["label"]]["landscape"] = {
                "min": float(np.nanmin(gl)), "mean": float(np.nanmean(gl)),
                "std": float(np.nanstd(gl)), "episode": int(r["landscape_episode"]),
            }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Two-thirds power-law + loss-landscape analysis")
    ap.add_argument("--env", action="append", choices=["point_maze", "pusht"],
                    default=[], help="env(s) to analyze (repeatable; default: both)")
    ap.add_argument("--max-episodes", type=int, default=1,
                    help="held-out episodes used for the slope/timeseries "
                         "(default 1 = single episode; raise e.g. to 80 for more "
                         "statistical power, at the cost of runtime)")
    ap.add_argument("--grid", type=int, default=13, help="landscape grid per axis (default 13)")
    ap.add_argument("--opt-steps", type=int, default=80,
                    help="GD steps per landscape grid point (default 80)")
    ap.add_argument("--action-range", type=float, default=2.0,
                    help="landscape grid half-range in normalized action units (default 2.0)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap iterations for slope CI (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.env:
        args.env = ["point_maze", "pusht"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_summaries = {}
    for env in args.env:
        all_summaries[env] = analyze_env(env, args, device)

    combined = os.path.join(args.outdir, "summary.json")
    with open(combined, "w") as f:
        json.dump(all_summaries, f, indent=2, default=float)
    print(f"\nSaved summaries to {combined} and figures under {args.outdir}/<env>/")


if __name__ == "__main__":
    main()

