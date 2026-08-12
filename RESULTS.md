# Results: Temporal Straightening + Two-Thirds Regularization on PointMaze (umaze)

Compiled from the runs on this machine (single NVIDIA RTX 4070 12 GB). All four world models were
trained with `encoder=dino_global` (frozen DINOv2 vits14 + **trainable** GlobalProjector),
`batch_size=32`, `TRAIN_DECODER=False` (decoder-less checkpoints), and evaluated at identical
planning settings (`goal_H=25`, `n_evals=50`, GD `opt_steps=100`, CEM `num_samples=200`).

## The question

Does adding the **two-thirds power-law regularizer** on top of temporal straightening improve
goal-reaching planning, compared to straightening alone?

## Headline result

**Yes -- two-thirds + straightening beats straightening alone in both planners.**

| Model (dino_global) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline (no regularizers) | 0.18 | 0.56 | 3.54 | 2.73 |
| straightening only (`cos1e-1`) | 0.44 | 0.52 | 2.36 | 2.76 |
| two-thirds only (`twothirds5e-2`) | 0.20 | 0.58 | 3.68 | 2.97 |
| **both** | **0.60** | **0.62** | **2.33** | 2.82 |

- **GD**: both (0.60) vs straightening only (0.44) = **+16 pp** (+36% relative); also the lowest
  mean final-state distance (2.33).
- **CEM**: both (0.62) vs straightening only (0.52) = **+10 pp**.

## What the metrics mean

- `success_rate`: fraction of the 50 eval episodes in which the **executed** MuJoCo rollout reaches
  the goal.
- `mean_state_dist`: mean distance between the final executed state and the goal state.
- Planning is open-loop (`MPC max_iter=1`): one action sequence is planned in latent space
  (5 latent steps x frameskip 5 = 25 env frames) and executed.

## Interpretation

1. **Straightening works** (GD): 0.44 vs baseline 0.18 -- the paper's core effect reproduces.
2. **Two-thirds alone ~ baseline** (GD 0.20 vs 0.18; CEM 0.58 vs 0.56): by itself it does not help.
3. ...but **combined with straightening it gives the best model in both planners**, which is exactly
   what this experiment was testing.
4. **CEM > GD overall** (CEM scores 200 candidates per iteration; GD optimizes a single action
   sequence), so CEM has less headroom -- which is why its delta (+10 pp) is smaller than GD's
   (+16 pp).

### Statistical caveat

`n_evals=50` gives a binomial standard error of ~0.07. The GD gain (+16 pp, ~2.3 SE) is the
strongest evidence; the CEM gain (+10 pp, ~1.4 SE) is suggestive but within noise. Re-running with
more evals (e.g. 200) would tighten this.

---

## Training-time behaviour of the regularizers (from the offline wandb logs)

The regularizers were active: they moved over training, and the keys are absent in the runs where
they were disabled. Values below are the cumulative means recorded in each run's offline wandb file
(early = first logged batch, late = last logged flush).

| Model | curvature loss (early -> late) | two-thirds loss (early -> late) |
|---|---|---|
| baseline | -- | -- |
| straightening only | ~1.37 -> ~0.50 | -- |
| two-thirds only | -- | ~0.004 -> ~0.003 |
| both | ~0.62 -> ~0.49 | ~0.043 -> ~0.079 |

Note: `train_loss` ends higher for the regularized runs (0.05 vs 0.007 baseline) -- expected, since
the regularizers trade a little prediction loss for latent structure.

## Setup recap (see EXPERIMENT.md for the full parameter reference)

- `encoder=dino_global` -- frozen DINOv2 vits14 + trainable GlobalProjector -> 1 token per frame.
- `training.straighten=cos1e-1` (patch-wise curvature, scale 0.1); `training.twothirds=twothirds5e-2`
  (two-thirds power-law residual variance, scale 0.05).
- `batch_size=32`, `frameskip=5`, `num_hist=3`, `num_pred=1`, ~10 epochs per launch (resumed runs).
- `TRAIN_DECODER=False` -- no decoder in the checkpoints, so planning produces metrics but no
  decoded-frame videos.
- Planning: `goal_H=25`, `n_evals=50`; GD (lr 0.1, opt_steps 100, zero init);
  CEM (num_samples 200, topk 30, opt_steps 10). Wandb offline, single 12 GB GPU.

## Raw artifacts

- Planning logs (one per model x planner):
  - `plan_outputs_gd/test_umaze_{False,cos1e-1,tttwothirds5e-2,cos1e-1_tttwothirds5e-2}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05_gH25/logs.json`
  - `plan_outputs_cem/...` (same naming)
- Training runs: `checkpoints/test/umaze_{...}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05/`
  (each contains `checkpoints/model_{1..N}.pth`, `hydra.yaml`, `train.log`, `wandb/`).
- The per-run offline wandb files hold the full loss curves for reproduction.
