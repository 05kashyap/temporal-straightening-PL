#!/usr/bin/env python
"""
analysis/curvature_distributions.py
===================================
Phase-1 diagnostic: is Wall's latent curvature concentrated near zero relative to
PointMaze/PushT? Pools (1 - cos theta) over ~50 held-out val episodes of each
straightening-only checkpoint, prints a comparison table and a shared-axis
log-x histogram. Reuses the validated load_model / load_val_dset / encode_episode
machinery from curvature_analysis.py.
"""
import os
import sys
import argparse

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from curvature_analysis import (  # noqa: E402
    load_model,
    load_val_dset,
    encode_episode,
    get_mode,
)

ENVS = [
    ("point_maze",
     "checkpoints/test/umaze_cos1e-1_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05"),
    ("pusht",
     "checkpoints/test/pusht_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"),
    ("wall",
     "checkpoints/test/wall_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05"),
]


def encode_curvature_series(wm, obs, device, mode="aggcos", chunk=32):
    """Encode one full held-out episode; return (theta, kappa, weight=1-cos)."""
    feats = encode_episode(wm, obs, device, chunk=chunk)  # (1, T, p, d)
    with torch.no_grad():
        if mode == "aggcos":
            b, t, p, d = feats.shape
            feats = wm.encoder.agg(feats.reshape(b * t, p, d)).reshape(b, t, -1)
        v1 = feats[:, 1:-1] - feats[:, :-2]
        v2 = feats[:, 2:] - feats[:, 1:-1]
        s1, s2 = v1.norm(dim=-1), v2.norm(dim=-1)
        avg_len = (s1 + s2) / 2
        cos = torch.nn.functional.cosine_similarity(v1, v2, dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        kappa = theta / (avg_len + 1e-6)
        weight = 1.0 - cos
    return (theta.flatten().cpu().numpy(),
            kappa.flatten().cpu().numpy(),
            weight.flatten().cpu().numpy())


def pool_env(wm, val, device, n_episodes=50, mode="aggcos"):
    all_theta, all_kappa, all_weight = [], [], []
    for i in range(min(n_episodes, len(val))):
        obs, _, _, _ = val[i]
        if obs["visual"].shape[0] < 4:
            continue
        theta, kappa, weight = encode_curvature_series(wm, obs, device, mode=mode)
        all_theta.append(theta)
        all_kappa.append(kappa)
        all_weight.append(weight)
    return (np.concatenate(all_theta), np.concatenate(all_kappa), np.concatenate(all_weight))


def summarize(name, theta, kappa, weight):
    return {
        "env": name,
        "n_points": len(weight),
        "median_theta": float(np.median(theta)),
        "median_weight": float(np.median(weight)),
        "frac_weight_lt_0.01": float(np.mean(weight < 0.01)),
        "frac_weight_lt_0.10": float(np.mean(weight < 0.10)),
        "p90_weight": float(np.percentile(weight, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--outdir", default=os.path.join(REPO, "analysis_outputs"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    summaries, pooled = [], {}
    for env, mdir in ENVS:
        wm, cfg = load_model(os.path.join(REPO, mdir), device)
        val = load_val_dset(cfg)
        mode = get_mode(wm)
        theta, kappa, weight = pool_env(wm, val, device,
                                        n_episodes=args.n_episodes, mode=mode)
        summaries.append(summarize(env, theta, kappa, weight))
        pooled[env] = (theta, kappa, weight)
        print(f"[{env}] mode={mode} n={len(weight)}")

    print("\n=== Curvature distribution summary (weight = 1 - cos theta) ===")
    hdr = (f"{'env':12s} {'n':>7s} {'med_theta':>10s} {'med_weight':>10s} "
           f"{'frac<0.01':>9s} {'frac<0.10':>9s} {'p90_w':>8s}")
    print(hdr)
    for s in summaries:
        print(f"{s['env']:12s} {s['n_points']:7d} {s['median_theta']:10.4f} "
              f"{s['median_weight']:10.4f} {s['frac_weight_lt_0.01']:9.4f} "
              f"{s['frac_weight_lt_0.10']:9.4f} {s['p90_weight']:8.4f}")

    # 3-panel shared-axis log-x histogram of weight
    all_w = np.concatenate([pooled[e][2] for e in ("point_maze", "pusht", "wall")])
    logw = np.log10(np.clip(all_w, 1e-6, None))
    lo, hi = float(logw.min()), float(logw.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, env in zip(axes, ("point_maze", "pusht", "wall")):
        w = pooled[env][2]
        ax.hist(np.log10(np.clip(w, 1e-6, None)), bins=60, range=(lo, hi),
                color="tab:blue", alpha=0.8)
        ax.axvline(np.log10(np.median(w)), color="red", ls="--", lw=1, label="median")
        ax.set_title(f"{env} (n={len(w)})")
        ax.set_xlabel(r"$\log_{10}(1-\cos\theta)$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Curvature magnitude 1 − cos θ, straightening-only models, val episodes "
                 "(shared axes)")
    fig.tight_layout()
    path = os.path.join(args.outdir, "curvature_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved histogram -> {path}")


if __name__ == "__main__":
    main()
