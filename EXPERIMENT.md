# Experiment: Temporal Straightening + Two-Thirds Regularization on PointMaze (umaze)

Local configuration used on this machine (single **NVIDIA RTX 4070 12 GB**, Python 3.9 / `ts` conda
env). The defaults in `conf/*.yaml` target H100s; the values below are the ones actually used by
`run.sh`, with the GPU-constrained overrides called out.

---

## How to run

```bash
bash run.sh                      # default: trains the straighten+twothirds model, then plans (GD + CEM)
TRAIN_DECODER=True bash run.sh   # also train the VQVAE decoder (needed for the planner's decoded videos)
N_EVALS=10 bash run.sh           # fewer eval episodes for faster planning (50 is the default)
FRESH=1 bash run.sh              # delete existing run dirs first
```

Exactly what `run.sh` executes for the active variant (step 7/8):

```bash
PY=/home/shanveen-ortho-clinic/miniconda3/envs/ts/bin/python
M=test/umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05

# training (10 epochs per launch -- see the resume caveat below)
"$PY" train.py --config-name train.yaml env=point_maze encoder=dino_global \
  training.straighten=cos1e-1 training.twothirds=twothirds5e-2 \
  training.batch_size=8 training.epochs=10 \
  model.train_decoder=False has_decoder=False \
  hydra.run.dir=./checkpoints/$M

# planning (GD and CEM), 10 eval episodes each
"$PY" plan.py --config-name plan_gd.yaml  ckpt_base_path="$PWD/checkpoints/$M" model_name="$M" goal_H=25 n_evals=10
"$PY" plan.py --config-name plan_cem.yaml ckpt_base_path="$PWD/checkpoints/$M" model_name="$M" goal_H=25 n_evals=10 planner.sub_planner.num_samples=50
```

### Run-dir naming convention

`checkpoints/test/<env>_<straighten>_tt<twothirds>_agg32_proj<projector>_dim<dim>_hw<hw>_sg<stop_grad>_lr<encoder_lr>`

For this setup: `umaze_cos1e-1_tttwothirds5e-2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05`.

---

## Training hyperparameters

| Parameter | Value | What it is / does |
|---|---|---|
| `env` | `point_maze` (umaze) | Task: reach a random goal in a U-shaped maze. 2000 rollouts / 200k frames in `data/datasets/point_maze`. |
| `encoder` | `dino_global` | DINOv2 ViT-S/14 patch features (256 tokens, 16x16 grid, dim 384) + **trainable GlobalProjector** that pools to a single 384-dim token per frame. The backbone stays frozen; the projector is what the regularizers train. |
| `img_size` | 224 | Input resolution. |
| `frameskip` | 5 | World-model timestep = every 5th env frame (so a latent step spans 5 raw frames). |
| `num_hist` | 3 | Context frames the predictor conditions on. |
| `num_pred` | 1 | Frames the predictor is asked to produce (only 1 supported). |
| `batch_size` | **32** (config default) | Images per minibatch (32x4 frames forward+backward). Fits this GPU with the decoder off and `dino_global` (3-token attention). |
| `epochs` | **10** | **Per launch**, not an absolute total (see resume caveat). |
| `seed` | 0 | RNG seed. |
| `encoder_lr` | 1e-5 | LR for trainable encoder modules (none here -- see caveat). |
| `predictor_lr` | 5e-4 | LR for the latent-predictor transformer. |
| `decoder_lr` | 3e-4 | LR for the VQVAE decoder (only used if the decoder is trained). |
| `action_encoder_lr` | 5e-4 | LR for the action (proprio) encoder. |
| `stop_grad` | True | Detaches the target frames in the predictor loss (`z_tgt.detach()`), a standard world-model trick to prevent representation collapse. |
| `mixed_precision` | bf16 | Mixed-precision training. |
| `straighten` | `cos1e-1` | Curvature regularizer, patch-wise (`cos`) mode, scale 0.1. See below. |
| `twothirds` | `twothirds5e-2` | Two-thirds power-law regularizer, `cos` mode, scale 0.05. See below. |
| `TRAIN_DECODER` | **False** | Also train + save the VQVAE decoder (`model.train_decoder` and `has_decoder`). Required for planner videos. |
| `decoder_start_epoch` | 1 | Decoder starts optimizing at this epoch (irrelevant when `TRAIN_DECODER=False`). |
| `save_every_x_iterations` | 1000 | Checkpoint + wandb flush cadence. |
| `save_every_x_epoch` | 1 | Save per-epoch checkpoints (`model_<epoch>.pth`). |

### Model architecture (for `encoder=dino_global`)

- **Encoder**: DINOv2 `dinov2_vits14` backbone (22M params) is **always frozen** in this codebase
  (`train.py` `_configure_encoder_trainability`), producing 256 patch tokens x 384 dim. The
  **GlobalProjector** (3 conv layers, pooling to a 1x1 grid) is trainable and outputs a single
  384-dim token per frame -- the part the straightening/two-thirds regularizers train.
- **Predictor**: `ViTPredictor`, depth 6, heads 16, dim_head 64, mlp_dim 2048, dropout 0.1, pool `mean`.
  It operates on the flattened (frames x 256 patches) token sequence; input dim is 404 =
  384 (visual) + 10 (proprio emb) + 10 (action emb, 2 dims x frameskip 5).
- **Decoder** (only when `TRAIN_DECODER=True`): `VQVAE`, channel 384, n_embed 2048, n_res_block 4,
  n_res_channel 128, `quantize=False`.

---

## Regularizers (the new losses)

### `training.straighten=cos1e-1` -- curvature / temporal straightening

Encourages locally **straight latent trajectories**: consecutive latent velocities should point the
same way. For two consecutive step vectors `v1, v2` of a window it computes
`curvature = 1 - cosine_similarity(v1, v2)`, masks out (near-)zero-length steps, and averages.

- Mode is picked by the string prefix: `cos` = patch-wise (each of the 256 patches scored separately),
  `aggcos` = pooled features via the encoder's `agg` head.
- `1e-1` is the weight `straighten_scale` the curvature term is multiplied by before being added to
  the total loss (`loss = loss + curvature * 0.1`).
- Intuition: straight trajectories are easier for gradient-based planning (Euclidean distance in
  latent space becomes a better proxy for the true planning objective).

### `training.twothirds=twothirds5e-2` -- two-thirds power-law regularizer

Pushes latent trajectories to obey the perceptual **2/3 power law** (speed and curvature co-vary as
`s ~ kappa^(-1/3)`). For each time window it computes the residual
`r_t = log(s_t) + (1/3) log(kappa_t)` (s = average step length, kappa = turn angle / s) and then
minimizes the **variance of r across the time axis** (biased / `unbiased=False` -- with only 2
residual points per window under `num_hist=3`, the unbiased estimator would be much noisier). A
constant `r_t` means the trajectory follows the power law exactly; the loss is that variance.

- `twothirds` prefix = patch-wise, `aggtwothirds` = pooled; `5e-2` is the scale (here `0.05`).
- Unlike the curvature loss, `_two_thirds_residual` uses `clamp_min(step_thresh)` instead of boolean
  masking, so per-(sample, patch) grouping is preserved for the variance term.

### Why the regularizers work here: the projector is the trainable part

The DINOv2 backbone is **always frozen** in this codebase, but `encoder=dino_global` attaches a
**trainable GlobalProjector** on top. The curvature / two-thirds losses are computed on the projector
output, so they **do** produce gradients and train it -- this is what "training the representation"
means in this setup (verified empirically: the projector params receive non-zero gradients, and the
curvature loss decreased over the Phase-5 smoke runs).

If you ever switch back to `encoder=dino` (no projector), the regularizers become inert: they are
computed on the frozen DINO features and add **zero gradient** to any trainable parameter (the
predictor and action/proprio encoders still train, but only via the prediction loss `z_loss`).
That encoder is the paper's no-regularizer baseline.

---

## Planning hyperparameters

Planning = ask the trained world model **which action sequence best reaches a goal latent**, then
roll that sequence out in the real (MuJoCo) point_maze simulator and measure how often it actually
reaches the goal. `run.sh` runs two planners over the same eval episodes.

### Shared parameters (`run.sh` + `conf/plan_{gd,cem}.yaml`)

| Parameter | Value | What it is / does |
|---|---|---|
| `GOAL_H` / `goal_H` | 25 | Goal horizon in **env frames**. The goal latent is the dataset observation 25 frames ahead of the start. plan.py divides this (and `horizon`, `n_taken_actions`) by `frameskip=5`, so the world model only rolls out **5 latent steps**; the executed plan spans 5 x 5 = 25 env frames. |
| `N_EVALS` | **50** (config default) | Number of evaluation episodes (initial states + goals sampled from the dataset) planning is run on. Success rate = fraction of episodes that reach the goal. See the dedicated note below. |
| `NUM_SAMPLES` | **200** (config default) | (CEM only) Random action-sequence candidates sampled per eval per CEM iteration. See the dedicated note below. |
| `goal_source` | `dset` | Goals come from the dataset (random future states), not the planner. |
| `num_start_frames` | 1 | Initial observation given to the model = 1 frame. |
| `n_plot_samples` | 10 | How many episodes get plots/videos. |
| `decode_for_viz` | true | Decode latent rollouts into images for the env-vs-imagined videos (needs a trained decoder). |
| `model_epoch` | `latest` | Which checkpoint to load (`model_latest.pth`, updated every 1000 iters + every epoch). |

### Objective (how "reached the goal" is scored during planning)

`objective: alpha=0, mode=last` -- the planner minimizes the **MSE between the final predicted latent
frame and the goal latent** (`objective_fn_last` in `planning/objectives.py`). `alpha=0` means the
visual latent is the only term (proprio latent weighted by `alpha` is off); `base=2` is only used by
`mode=all`. The `evaluator` computes the reported success by whether the **executed** env trajectory
actually reaches the goal state.

### MPC wrapper (both planners run under this)

`planner = MPCPlanner, max_iter=1, n_taken_actions=25` (`25 / frameskip 5 = 5` latent steps).
`max_iter=1` makes this effectively **open-loop planning**: plan once over the whole horizon and
execute it, with no closed-loop re-planning.

### GD planner (`plan_gd.yaml`) -- gradient descent on the action sequence

Optimizes the action sequence directly by **backpropagating the goal-loss through the world model**
into the actions (`actions.requires_grad = True`, Adam optimizer, cosine LR schedule):

| Parameter | Value | What it does |
|---|---|---|
| `opt_steps` | 100 | Number of gradient steps on the actions. Each step does a full 5-step latent rollout + backward through the predictor. |
| `lr` | 0.1 | Adam learning rate for the action optimization. |
| `sample_type` | `zero` | Actions initialized to zeros (normalized). |
| `action_noise` | 0 | Per-step Gaussian noise added to actions after each gradient step. |
| `horizon` | 25 -> 5 | Rollout length (env frames -> latent steps after /frameskip). |
| `eval_every` | -1 | No env evaluation during optimization; only the final plan is evaluated. |

### CEM planner (`plan_cem.yaml`) -- cross-entropy method

Samples a batch of random action sequences from a Gaussian, scores them in the world model, keeps the
best (`topk`) as "elites", and refits the Gaussian around them; repeat for `opt_steps` iterations.
Forward-only (no backprop through the model), which is why it tolerates large batches:

| Parameter | Value | What it does |
|---|---|---|
| `num_samples` | **200** (config default) | How many random action candidates are drawn and scored **per eval episode, per iteration**. |
| `topk` | 30 | How many of the best candidates are kept as elites to refit the sampling distribution. |
| `opt_steps` | 10 | Number of CEM iterations (sample -> score -> elite-refit cycles). |
| `var_scale` | 1 | Initial standard deviation of the sampling Gaussian. |
| `horizon` | 25 -> 5 | Same /frameskip conversion as GD. |
| `eval_every` | 1 | Re-evaluate the current best plan in the env after **every** CEM iteration. |

### Why the config defaults (`n_evals=50`, `num_samples=200`) fit on this GPU

Both values are the **config defaults** -- no overrides needed. The earlier OOMs that forced
`n_evals=10` / `num_samples=50` were specific to the old `encoder=dino` setup, where the ViT
predictor attends over 3x256 = 768 tokens per window; that attention matrix was what exhausted the
12 GB GPU. With `encoder=dino_global` (1 pooled token per frame), the predictor only attends over
**3 tokens**, so both planners fit comfortably at the defaults:

- **`N_EVALS` = number of evaluation episodes.** GD backprops all `n_evals` rollouts in one batch;
  with 3-token attention and no decoder, 50 evals peak at < 1 GB.
- **`NUM_SAMPLES` = CEM candidates per episode.** 200 candidates through the 3-token predictor (plus
  the frozen-DINO encode of the start frames) also peaks at < 1 GB. `topk=30` stays valid because it
  only needs `num_samples >= topk`.

Measured on this machine (decoder off, `dino_global`): training at `batch_size=32` ~2.3 GB,
GD planning `n_evals=50` ~0.8 GB, CEM planning `num_samples=200` ~0.8 GB -- all against 12.88 GB.

---

## Caveats / gotchas to remember

1. **`EPOCHS=10` is per launch, not an absolute total.** The training loop is
   `for epoch in range(resumed_epoch + 1, resumed_epoch + 1 + total_epochs)`. Because `run.sh`
   resumes existing runs (dir already has a checkpoint), re-running it keeps adding 10 epochs
   (e.g. it showed "epoch 11" when it resumed from epoch 1). Use `FRESH=1` for a clean 10-epoch
   total.
2. **`TRAIN_DECODER=False` removes the planner's decoded videos.** Without a trained decoder the
   checkpoint has no decoder, so the planner runs `has_decoder=False` and the eval skips decoding:
   you still get all numeric metrics (`logs.json`: success rate, state/visual/proprio distances,
   `plan_targets.pkl`) but **no `output_final_*.mp4` env-vs-imagined videos**. Set
   `TRAIN_DECODER=True` for videos.
3. **Regularizers train the projector, not the frozen DINO backbone.** With `encoder=dino_global`
   the curvature / two-thirds losses act on the trainable GlobalProjector output. (`encoder=dino` --
   no projector -- makes them inert; that is the paper's no-regularizer baseline.)
4. **Checkpoint resume**: planning loads `model_latest.pth` (updated every 1000 iters and every
   epoch), so interrupting training at any point still leaves a usable model for planning.

---

## Environment / versions

- Conda env `ts` (Python 3.9), torch 2.3.0+cu121, torchvision 0.18.0, decord 0.6.0, wandb (offline
  mode, `WANDB_MODE=offline`), MuJoCo 2.1.2 at `~/.mujoco/mujoco-2.1.2` (`MUJOCO_PY_MUJOCO_PATH` must
  be set for planning).
- DINOv2 weights: pinned `facebookresearch/dinov2` commit `b48308a...` (see `models/dino.py`).
- Data: `data/datasets/point_maze` (2000 rollouts, 200k frames) and `data/datasets/pusht_noise`.
- Outputs: `checkpoints/test/umaze_*` (training), `plan_outputs_gd/`, `plan_outputs_cem/`
  (planning logs + videos), `results/` (aggregated `logs.json` copies).
