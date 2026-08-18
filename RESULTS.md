# Results: Temporal Straightening + Two-Thirds Regularization (PointMaze umaze | PushT | Wall)

Compiled from the runs on this machine (single NVIDIA RTX 4070 12 GB). Three experiments are
documented: **PointMaze-umaze** (`encoder=dino_global`), **PushT** (`encoder=dino_channel`), and
**Wall** (`encoder=dino_channel`).
All three train decoder-less world models (`TRAIN_DECODER=False`) and evaluate open-loop planning at
`goal_H=25`. All runs use the offline-wandb logs and `logs.json` under `plan_outputs_*`.

## The question

Does adding the **two-thirds power-law regularizer** on top of temporal straightening improve
goal-reaching planning, compared to straightening alone?

## PointMaze (umaze) results

### Headline result

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

### What the metrics mean

- `success_rate`: fraction of the 50 eval episodes in which the **executed** MuJoCo rollout reaches
  the goal.
- `mean_state_dist`: mean distance between the final executed state and the goal state.
- Planning is open-loop (`MPC max_iter=1`): one action sequence is planned in latent space
  (5 latent steps x frameskip 5 = 25 env frames) and executed.

### Interpretation

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

## PushT results (encoder = dino_channel)

The same 4-variant experiment on **PushT**, but with the **spatial** channel projector instead of the
1-token global projector: the paper's Table 1 shows only ~2% open-loop GD success on PushT with
`dino_global` vs ~70% with the channel projector, so PushT uses `encoder=dino_channel` (14x14x8
spatial features -- also keeps the regularizers trainable). Regularizers act on the learned
aggregation head (`aggcos` modes): `training.straighten=aggcos1e-1`, `training.twothirds=aggtwothirds5e-2`.
The paper trains PushT for only **2 epochs** (Appendix A.3); planning adds `objective.alpha=1`
(README's PushT note) and runs the full 50 evals in chunks of 10 (`n_evals=50`, `chunk_size=10`)
to fit the 14x14 attention on the 12 GB GPU. All four models: `batch_size=16`, `epochs=2`,
`frameskip=5`, `num_hist=3`, `TRAIN_DECODER=False`.

> Eval-config fix (Aug 2026): the PushT CEM numbers below were re-run at the paper's **`num_samples=200`**
> (the earlier 50-sample runs were a 12-GB-GPU memory workaround and underpowered CEM:
> `topk=30` of 50 samples keeps 60% of candidates, collapsing selection). `planning/cem.py` gained a
> `sample_chunk_size` knob that rolls the 200 candidates out in chunks of 50 — mathematically identical
> to a single 200-sample forward, but memory-bounded. The old 50-sample results
> (CEM 0.24 / 0.24 / 0.10 / 0.30) are preserved under `/tmp/eval_backup_old/`.

| Model (dino_channel) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline (no regularizers) | 0.74 | **0.72** | 50.62 | 50.82 |
| straightening only (`aggcos1e-1`) | 0.80 | **0.78** | 51.41 | 53.06 |
| two-thirds only (`aggtwothirds5e-2`) | 0.54 | **0.50** | 41.66 | 51.78 |
| **both** | **0.80** | **0.80** | **37.73** | 45.41 |

- **GD**: straightening and both tie at **0.80** (baseline 0.74); the both-model reaches the goal far
  more precisely (**state-dist 37.7** vs 51.4) -- the same pattern as PointMaze.
- **CEM (paper protocol, 200 samples)**: baseline **0.72** and straightening **0.78** sit right on the
  paper's Table B.3 dino_channel CEM numbers (71.33 / 80.00); both reaches **0.80**.

Interpretation:

1. Straightening gives the best GD success (0.80 vs baseline 0.74); two-thirds **alone** hurts
   (0.54 GD / 0.50 CEM) -- consistent with PointMaze.
2. On GD the straighten/both tie at the 0.02 resolution (n=50), but state distance clearly favors
   both (37.7 vs 51.4).
3. On CEM, **both is the best model** (0.80), and the corrected 200-sample protocol brings the
   baseline/straightening in line with the paper (72/78 vs 71/80).

Statistical caveat (PushT): `n_evals=50` -> SE ~0.07, so the GD 0.80-vs-0.80 tie is within noise at
this sample size; the state-distance gap (37.7 vs 51.4) is the more informative signal. PushT
open-loop planning is also inherently noisy (contact-rich dynamics), so treat the absolute rates as
ballpark.

Raw artifacts (PushT):

- Planning logs (one per model x planner):
  - `plan_outputs_gd/test_pusht_{False,aggcos1e-1,aggtwothirds5e-2,aggcos1e-1_aggtwothirds5e-2}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_gH25/logs.json`
  - `plan_outputs_cem/...` (same naming); each dir also has `plan_targets.pkl` and `plan.log`.
- Training runs: `checkpoints/test/pusht_{...}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/`
  (2 epochs, batch 16; each contains `checkpoints/model_{1,2}.pth`, `hydra.yaml`, `wandb/`).

---

## Wall results (encoder = dino_channel)

Same 4-variant experiment on **Wall** (`env=wall`), spatial channel projector (14x14x8). Wall uses the
paper's Section 5.3 setup: start/goal states are sampled from **test trajectories** (`goal_source='dset'`,
the plan_gd/plan_cem default) so every goal is reachable within 25 steps, and only the target **image**
drives the objective (`alpha=0`, visual-only — "for other environments, we only use target images").

> Config-fix note: `run.sh` previously passed DINO-WM's wall settings
> (`goal_source=random_state objective.alpha=1`), which only make sense with DINO-WM's **closed-loop MPC**
> (open-loop `run.sh` does not use). Random-state goals were mean 36.8 apart (max 54) — well beyond what
> a single 25-frame open-loop plan can cover (wall max step 1.8 x 25 = 45) — so the planner stalled at
> the wall (success 0.0-0.06). With dset goals (mean 20.3, **max 42.1 — all ≤ 45, all reachable**) the
> corrected results below are reproduced. Old (wrong-config) plan dirs are backed up in
> `/tmp/wall_plan_backup_random_state/{gd,cem}/`.

> Note (Aug 18): all four rows below now share **one consistent protocol** — fresh **20-epoch**
> dino_channel models, `goal_source=dset`, `alpha=0`, and the paper's **`num_samples=200`** CEM
> (chunked ×50 to fit the 12 GB GPU). Baseline and straightening were retrained + replanned on Aug 18;
> two-thirds and both were trained Aug 17 (both 20 epochs, same config).

| Model (dino_channel) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline (no regularizers) | 0.36 | 0.44 | 6.56 | 5.54 |
| straightening only (`aggcos1e-1`) | **0.90** | **1.00** | **1.99** | **1.52** |
| two-thirds only (`aggtwothirds5e-2`) | 0.68 | 0.68 | 5.29 | 4.55 |
| **both** (`aggcos1e-1_aggtwothirds5e-2`) | 0.68 | 0.60 | 3.52 | 3.68 |

- **Straightening now reproduces the paper's wall numbers**: GD 0.90 ≈ paper's 90.67 and CEM **1.00 = the
  paper's 100.00** (Table 23, dino_channel). State distance drops from ~6.5 (baseline) to 1.5-2.0 and
  visual distance to 0.40 — the straightened model reliably reaches the wall goals.
- **Baseline (0.36/0.44) is far below the paper's 80/92** — the fresh single-seed baseline model is
  genuinely weak here (mean state-dist 6.56, well above the 4.5 success threshold). This is a
  model-quality/seed gap, not an eval-config issue (protocol is identical to the paper's).
- **`both` (0.68/0.60) sits *below* straightening (0.90/1.00)** — this persists even on the fully
  consistent protocol and is the notable wall-specific result: unlike umaze/pusht (where `both` was the
  best or tied model), adding two-thirds on top of straightening *hurts* wall planning. Two-thirds alone
  (0.68) beats the baseline but trails straightening.
- `mean_state_dist` ~3.5-5.3 for the two-thirds/both rows sits at/above the threshold (4.5) — those
  models end near the goal but often just outside it.

Raw artifacts (Wall):

- Planning logs:
  - `plan_outputs_gd/test_wall_{False,aggcos1e-1,aggtwothirds5e-2,aggcos1e-1_aggtwothirds5e-2}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_gH25/logs.json`
  - `plan_outputs_cem/...` (same naming); each dir also has `plan_targets.pkl` and `plan.log`.
- Training runs (all four on disk, each 20 epochs):
  `checkpoints/test/wall_{False,aggcos1e-1,aggtwothirds5e-2,aggcos1e-1_aggtwothirds5e-2}_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/`

---

## Training-time behaviour of the regularizers (PointMaze, from the offline wandb logs)

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

## Setup recap for PointMaze (see EXPERIMENT.md for the full parameter reference)

- `encoder=dino_global` -- frozen DINOv2 vits14 + trainable GlobalProjector -> 1 token per frame.
- `training.straighten=cos1e-1` (patch-wise curvature, scale 0.1); `training.twothirds=twothirds5e-2`
  (two-thirds power-law residual variance, scale 0.05).
- `batch_size=32`, `frameskip=5`, `num_hist=3`, `num_pred=1`, ~10 epochs per launch (resumed runs).
- `TRAIN_DECODER=False` -- no decoder in the checkpoints, so planning produces metrics but no
  decoded-frame videos.
- Planning: `goal_H=25`, `n_evals=50`; GD (lr 0.1, opt_steps 100, zero init);
  CEM (num_samples 200, topk 30, opt_steps 10). Wandb offline, single 12 GB GPU.

## Raw artifacts (PointMaze)

- Planning logs (one per model x planner):
  - `plan_outputs_gd/test_umaze_{False,cos1e-1,tttwothirds5e-2,cos1e-1_tttwothirds5e-2}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05_gH25/logs.json`
  - `plan_outputs_cem/...` (same naming)
- Training runs: `checkpoints/test/umaze_{...}_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05/`
  (each contains `checkpoints/model_{1..N}.pth`, `hydra.yaml`, `train.log`, `wandb/`).
- The per-run offline wandb files hold the full loss curves for reproduction.
