# SERVER_CONTEXT.md — running this repo on a new machine

Hand-off notes for reproducing the Temporal Straightening training run on a server.
Every claim below was checked on the development laptop (conda env `ts`, Python
3.9.23, torch 2.3.0+cu121 — `nvidia-smi` there reports an RTX 3050 Ti Laptop GPU;
`EXPERIMENT.md` in the repo was written on a 4070 12 GB machine) while writing this
file, or is a direct citation of the code / paper. Anything I could not verify is
marked **unverified**.

- **Repo:** https://github.com/05kashyap/temporal-straightening-PL.git
- **Branch:** `train-server` (already pushed; contains `run_scripts/train_server.sh`)
- **GPU:** any single NVIDIA card with enough memory — nothing in the code or the
  launcher is tied to a specific model (no device-conditional logic anywhere).
  The repo's `conf/` defaults were written for H100s; the paper used one card per run.
- **Paper:** *Temporal Straightening for Latent Planning*. The PDF/text is **not** in
  the repo (`papers/` is gitignored); the copy used locally is
  `~/Documents/utilities/markitdown/files/ts.md`. Line references below ("L711")
  point into that text.

---

## 0. TL;DR

```bash
git clone https://github.com/05kashyap/temporal-straightening-PL.git
cd temporal-straightening-PL
git switch train-server
conda env create -f environment.yaml && conda activate ts   # §4
nvidia-smi                                                   # confirm the card
ls data/datasets                                             # 3 datasets, §3
DRY_RUN=1 bash run_scripts/train_server.sh umaze channel     # prints 4 argvs, trains nothing
bash run_scripts/train_server.sh umaze channel 0             # 4 arms, decoder ON, GPU 0
```

Six such jobs are the whole training grid: `{umaze, medium, pusht} x {channel, global}`
(§1, §5). Each job is 4 sequential runs; the 3rd positional argument pins
`CUDA_VISIBLE_DEVICES`, so jobs can run in parallel on different cards.

---

## 1. What is being trained

One job = one environment × one DINOv2 feature recipe × the repo's **four arms** —
the baseline plus the three regularizer combinations (the vocabulary used by
`helpers/extract_planner_curves.py`):

| arm | `training.straighten` | `training.twothirds` |
|---|---|---|
| `baseline` | `False` | `False` |
| `straighten` | curvature (cosine-based) | `False` |
| `p_reg` | `False` | two-thirds (P-Reg) | 
| `both` | both |

The two recipes, both taken from the paper:

| | **channel** | **global** |
|---|---|---|
| encoder config | `encoder=dino_channel` | `encoder=dino_global` |
| projector | 14×14 patch grid → 8 channels | whole grid → single **1×384** vector (paper Table 1, L711-713) |
| aggregation | learnable MLP head (out dim 128), `encoder.agg_type=mlp` | **none** — paper §5 L519-524: for global features (n_v = 1) "compute the cosine similarity directly between vectors" |
| loss strings | `aggcos1e-1`, `aggtwothirds5e-2` (λ_curv = 0.1) | `cos1e-2`, `ttwothirds5e-2` (λ_curv = 0.01, B.6 L1389-1396) |
| exception | **medium** flattens instead of the MLP head (`encoder.agg_type=flatten`, paper B.6 / `run.sh`) | — |
| effective batch | 16 | 32 (paper Table 3 L1192-1203) |

Shared, from paper Table 3 and L695-698: 3 history frames, frameskip 5,
**20 epochs**, projector lr `1e-5` (`1e-6` for the arm trained without straightening,
per footnote a), and the VQVAE **decoder trained jointly "solely for interpretability
purposes"** (stop-gradient detached) — which is why every arm here passes
`model.train_decoder=True has_decoder=True`, so the decoder weights are in every
checkpoint (needed for the decoded-video / reconstruction figures).

Run-dir names encode the recipe, so the four arms can never collide:

```
umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06            <- baseline
umaze_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05       <- straighten
umaze_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 <- p_reg
umaze_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05  <- both
umaze_False_agg32_projglobal_dim384_hw1_sgTrue_lr1e-06            <- global: baseline
...
```

---

## 2. Repo, branch, and what is **not** in git

```bash
git clone https://github.com/05kashyap/temporal-straightening-PL.git
cd temporal-straightening-PL
git switch train-server          # == origin/train-server, the branch that has train_server.sh
```

The model/training code (`train.py`, `models/`, `conf/`, `datasets/`, `planning/`,
`metrics/`, `utils.py`, `custom_resolvers.py`, `plan.py`, `environment.yaml`,
`README.md`, `EXPERIMENT.md`, `run_scripts/train_server.sh`, this file) **is** in git.

**Not** in git — the clone will not have these:

| gitignored (`.gitignore`) | why it matters |
|---|---|
| `data/` | the datasets (§3) |
| `checkpoints/` | training run dirs (§6) |
| `*outputs*` → `analysis_outputs/` | landscape/sweep outputs |
| `results/` | **`results/LANDSCAPE_RESULTS.md`** + Fig. 4/5/6 notes live here |
| `papers/` | the paper text |
| `wandb/`, `figures/`, `Old results/` | logs / old figures |

| untracked (not ignored, so also absent from the clone) | why it matters |
|---|---|
| `analysis/landscape_sweep.py`, `landscape_paper_figure.py`, `landscape_metrics.py`, `landscape_metric_audit.py`, `loss_landscape_comparison.py` | the whole figure pipeline (§8) |
| `run_scripts/*.sh` **except** `train_server.sh` | `setup.sh`, `run.sh`, `run_mpc.sh`, `run_landscape_night.sh`, `run_landscape_paperstyle_full.sh`, … |
| `helpers/` | `extract_planner_curves.py`, `aggregate_mpc_summary.py` |
| `kaggle/` | Kaggle-only helpers (not for the server) |

**Uncommitted working-tree edits on the laptop** (they exist only locally — the
branch has the older version). Two are worth knowing about:

1. `conf/train.yaml` — adds `${replace_substring:${training.twothirds},…}` to the
   Hydra run-dir name. Without it, `p_reg` collides with `baseline` and `both`
   collides with `straighten` (the same folder ⇒ a resumed run silently loads a
   different variant's checkpoint; that is what the comment in the file warns about).
   **`train_server.sh` is immune** because it passes an explicit `hydra.run.dir`,
   but if you run `train.py` directly on the server you want this fix.
2. `models/visual_world_model.py` — log lines only (prints the two-thirds settings).
3. `curvature_analysis.py` — analysis-side changes (not needed for training).

Ready-to-paste sync from the laptop (adjust `SERVER` and paths):

```bash
REPO=~/Documents/Projects/temporal-straightening-PL
rsync -av --exclude __pycache__ $REPO/analysis/     SERVER:~/temporal-straightening-PL/analysis/
rsync -av --exclude __pycache__ $REPO/run_scripts/  SERVER:~/temporal-straightening-PL/run_scripts/
rsync -av --exclude __pycache__ $REPO/helpers/      SERVER:~/temporal-straightening-PL/helpers/
rsync -av $REPO/conf/train.yaml               SERVER:~/temporal-straightening-PL/conf/train.yaml
rsync -av $REPO/models/visual_world_model.py  SERVER:~/temporal-straightening-PL/models/visual_world_model.py
rsync -av $REPO/results/                      SERVER:~/temporal-straightening-PL/results/   # docs + figures
rsync -av ~/Documents/utilities/markitdown/files/ts.md SERVER:~/temporal-straightening-PL/papers/ts.md
```

---

## 3. Datasets

The env configs read the data root from the environment variable `DATASET_DIR`
(`conf/env/*.yaml`: `data_path: ${oc.env:DATASET_DIR}/<name>`), and
`run_scripts/train_server.sh` both exports `DATASET_DIR` (as the parent of its
`DATA_DIR_*` paths) **and** passes `env.dataset.data_path=…` explicitly, so the
launcher works either way.

Required, with the sizes measured here (following the symlinks):

| dir | size | layout |
|---|---|---|
| `point_maze` (umaze) | 29 GB | `states.pth`, `actions.pth`, `seq_lengths.pth`, `obses/` |
| `point_maze_medium` | 57 GB | same as `point_maze` |
| `pusht_noise` | 7.0 GB | `train/` and `val/`, each with `abs_actions.pth`, `rel_actions.pth`, `obses/` |

≈ 93 GB total. The launcher's preflight checks exactly those entries and refuses
to start if one is missing.

Source: the DINO-WM datasets (README "Datasets"):
`https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28` — unzip and point
`DATASET_DIR` at the folder that contains `point_maze/` etc.

On the laptop these three entries are **symlinks** to an external drive
(`/media/kashyap/datadrv/WorldModelDatasets/…`) and `data/` is gitignored, so nothing
about the data arrives through git — either download it on the server or rsync the
target directories. If the server keeps the data outside the repo, edit the three
`DATA_DIR_*` lines at the top of `run_scripts/train_server.sh` (that is the "EDIT ME"
block).

---

## 4. Environment

```bash
conda env create -f environment.yaml      # env name: ts
conda activate ts
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`environment.yaml` is the authoritative spec: **Python 3.9**, **torch 2.3.0** /
torchvision 0.18.0 (the PyPI Linux wheels are the **cu121** builds, so any recent
NVIDIA driver works — no local CUDA toolkit needed), numpy 1.26.4, hydra-core 1.2.0 +
hydra-submitit-launcher 1.2.0, wandb 0.13.1, accelerate 0.26.1, einops 0.4.1, gym
0.23.1, mujoco 3.2.7, dm-control 1.0.27, mujoco-py 2.1.2.14, d4rl 1.1, pymunk 6.8.0,
pygame 2.5.2, pybullet 3.2.7, shapely 2.0.3, trimesh 4.4.0, opencv-python 4.6.0.66,
scikit-image 0.19.3, imageio(+ffmpeg) 2.34.1, moviepy 1.0.3, **matplotlib 3.5.3**,
scipy 1.13.1, scikit-learn 1.5.0, tqdm, requests, pyyaml, pytest 8.2.1, decord 0.6.0.

The conda side also installs `c-compiler`/`cxx-compiler`/`patchelf`/`cmake`/`make`/
`glew`/`xorg-libx11`/`xorg-xorgproto` on purpose: **mujoco_py compiles its `cymj`
extension on first import** (its builder is patched with `-DGLEW_NO_GLU`), which needs
a compiler and `patchelf`. Do not remove those if you plan to run planning/eval.

- **MuJoCo 2.1.2 (`mujoco210`)** — needed only by the **planning / landscape** stage
  (`plan.py`, gym envs), **not** by `train.py`. README → "Mujoco": download
  `mujoco210-linux-x86_64.tar.gz` into `~/.mujoco/` and export
  `MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210` plus
  `LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia`. `run_scripts/setup.sh`
  (untracked — rsync it) already does those exports, incl. `D4RL_SUPPRESS_IMPORT_ERROR=1`
  and `EGL_GPU=0` for headless EGL.
- **DINOv2 weights** — `models/dino.py` loads
  `torch.hub.load("facebookresearch/dinov2:b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8", "dinov2_vits14")`.
  The first training run downloads the hub checkout (~tens of MB) plus the LVD-142M
  `dinov2_vits14` weights (~84 MB) into `~/.cache/torch/hub/`, specifically:

  ```
  ~/.cache/torch/hub/facebookresearch_dinov2_b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8/
  ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth
  ```

  No internet on the server? Copy exactly those two paths from a machine that ran it
  (`rsync -av ~/.cache/torch/hub/ SERVER:~/.cache/torch/hub/`). `models/dino.py` also
  patches `torch.hub._validate_not_a_forked_repo` and re-inserts the hub dir on
  `sys.path`, which is what lets saved checkpoints (they reference `dinov2` classes)
  unpickle.
- **Weights & Biases** — project `temporal_straightening_<env.name>` (e.g.
  `temporal_straightening_point_maze`). `train_server.sh` defaults to
  `WANDB_MODE=offline`, so runs write `wandb/` **inside the run dir** and never block
  on network; sync later with `wandb sync <run_dir>/wandb/*` or export
  `WANDB_MODE=online` (plus `WANDB_API_KEY`) to log live.

---

## 5. Running the training

```bash
cd ~/temporal-straightening-PL                       # any cwd inside the repo works
conda activate ts
bash run_scripts/train_server.sh <umaze|medium|pusht> <channel|global> [gpu-index]
```

- One job = **4 sequential runs** (`baseline`, `straighten`, `p_reg`, `both`), all with
  the decoder on. `train.py` logs to stdout; the launcher tees each arm into
  `$CKPT_ROOT/logs/<run_name>.log` (appended, so resumed arms continue the same file)
  and prints a summary of which of the 4 arms has a checkpoint at the end.
- `[gpu-index]` sets `CUDA_VISIBLE_DEVICES` for that job, so several jobs can share one
  node. Omit it to use the default device.
- **Always dry-run first** — `DRY_RUN=1` prints the four real `train.py` argvs and
  trains nothing:

```bash
DRY_RUN=1 bash run_scripts/train_server.sh umaze channel
```

- **Smoke test** (one arm, 1 epoch, tiny batch, throwaway dir) before committing to the
  full grid — this is also how you measure seconds-per-epoch on the new card:

```bash
EPOCHS=1 BATCH_SIZE=4 CKPT_ROOT=/tmp/ts-smoke bash run_scripts/train_server.sh umaze global 0
```

- The full grid, one card per job (run inside `tmux`/`screen`, or with `nohup`):

```bash
cd ~/temporal-straightening-PL
i=0
for e in umaze medium pusht; do
  for d in channel global; do
    nohup bash run_scripts/train_server.sh "$e" "$d" "$i" > "train_${e}_${d}.log" 2>&1 &
    i=$((i + 1))
  done
done
```

| env var | default | meaning |
|---|---|---|
| `EPOCHS` | `20` | **per launch** (paper), see §6 |
| `BATCH_SIZE` | `32` (global) / `16` (channel) | lower it if the card OOMs |
| `NUM_HIST` | `3` | paper Table 3 |
| `NUM_WORKERS` | env yaml (`16`) | 6 parallel jobs × 16 = 96 workers; lower it on a small node |
| `REG_WINDOW` | untruncated (`num_hist+num_pred`) | P-Reg stats window |
| `FRESH` | `0` | `1` = delete the 4 run dirs before training |
| `DRY_RUN` | `0` | `1` = print argvs, train nothing |
| `SKIP_FINISHED` | `1` | skip an arm whose saved epoch ≥ `EPOCHS` |
| `LINK_RUNS` | `1` | symlink each run dir into `$ART_ROOT` |
| `CKPT_ROOT` | `$REPO/checkpoints/server` | where run dirs go |
| `ART_ROOT` | `$REPO/analysis_outputs/server` | symlink farm for downstream artifacts |
| `PYTHON` | activated conda env → `$HOME/miniconda3/envs/ts/bin/python` → `python` on PATH | interpreter (the preflight reports which one and fails loudly) |
| `WANDB_MODE` | `offline` | `online` to log live |

---

## 6. What a run writes, resume semantics, disk

Hydra chdirs into the run dir, so it is the unit of "where are my artifacts"
(`$CKPT_ROOT/test/<run_name>/`):

```
checkpoints/model_latest.pth        # every epoch, and every 1000 batches
checkpoints/model_<epoch>.pth
hydra.yaml                          # resolved config (+ wandb_run_id for resuming)
rollout_plots/e<n>_rollout/*.png    # decoded reconstructions (reconstruct_every_x_batch=1000)
wandb/                              # offline wandb files (WANDB_MODE=offline)
```

plus the launcher's per-arm console log at **`$CKPT_ROOT/logs/<run_name>.log`**
(`tee -a`, appended across resumes; `train.py` itself only prints to stdout). Run dirs
produced by older launchers have a `train.log` *inside* the run dir instead.

Checkpoint contents (`train.py` `_keys_to_save`): `epoch`, `current_iter` (mid-epoch
resume point), `encoder` + `encoder_optimizer`, `predictor` + `predictor_optimizer`,
`decoder` + `decoder_optimizer` (present because `train_decoder=True`),
`action_encoder`, `proprio_encoder`. Sanity check after a run:

```bash
python -c "import torch; print(sorted(torch.load('checkpoints/model_latest.pth', map_location='cpu').keys()))"
# must contain 'decoder' -- that is what the decoded-video / reconstruction figures need
```

**Resume:** if `checkpoints/model_latest.pth` exists, `train.py` loads it automatically
(weights, optimizers, epoch, mid-epoch batch counter) — there is no flag. Because
`training.epochs` counts **per launch**, re-running a finished run would train `EPOCHS`
*more* epochs; that is why the launcher defaults to `SKIP_FINISHED=1`, which skips an
arm whose saved epoch ≥ `EPOCHS`. Extend a run with `EPOCHS=<saved+N>`, restart it with
`FRESH=1`.

**Disk:** measured checkpoint sizes here are **0.39 GB** without the decoder and
**0.57 GB** with it. One epoch writes `model_latest.pth` + `model_<epoch>.pth`, so
≈ 21 × 0.57 GB ≈ **12 GB per arm ≈ 48 GB per job ≈ 290 GB for all six jobs** — budget
~400 GB. (Arithmetic from the measured per-checkpoint size, not a full-server
measurement.)

---

## 7. Gotchas / failure modes

- **Don't point `CKPT_ROOT` at an existing run tree.** Run-dir names encode the
  regularizers but *not* "decoder on", so a same-named older run would be resumed and
  its decoder would start from scratch. The default (`checkpoints/server/…`) cannot
  collide with the laptop's `checkpoints/test/…`.
- **If you run `train.py` yourself instead of the launcher**, apply the `conf/train.yaml`
  run-dir fix first (§2), otherwise `baseline`/`p_reg` and `straighten`/`both` share a
  directory and a resume can silently load the wrong variant.
- **`DATASET_DIR` is mandatory** when you bypass the launcher: `conf/env/*.yaml` resolve
  `${oc.env:DATASET_DIR}` and Hydra errors out if it is unset.
- **Precision:** `training.mixed_precision: bf16` by default → needs Ampere or newer
  (A100 / L40S / H100 / 40-series are fine). For older cards (V100, T4) try
  `training.mixed_precision=fp16` or `no` — *unverified end-to-end, test with the §5
  smoke command first.*
- **`conf/train.yaml` composes `override hydra/launcher: submitit_slurm`** with
  `gres: gpu:h100:1`. Single runs never use the launcher (it only applies to `-m`
  multiruns on SLURM), so the `h100` string is not a hardware requirement — just don't
  use `-m` unless you actually are on SLURM.
- **`num_workers: 16` per env yaml** → 96 dataloader processes if all six jobs run at
  once. Lower with `NUM_WORKERS=8` on a small node.
- **What the preflight reports** (before anything is trained): the chosen interpreter +
  torch version + whether CUDA is visible + the card name, the dataset dir and whether
  the expected files exist, and whether the DINOv2 hub cache is already present (a
  warning only — it can download on first use). It hard-fails on a missing dataset or a
  torch-less python.
- **`wandb 0.13.1`** is pinned for py3.9; the `kaggle/kaggle_setup.sh` header documents
  which pins must change on newer Pythons. Don't rebuild the env on 3.12 without reading
  it (hydra 1.2.0 also breaks on ≥3.11).
- **No DDP inside a single run** — one process = one GPU. Parallelism only via the six
  jobs (or all 24 arms if you launch arm-level runs yourself).
- **An arm that fails stops the whole job** (`set -euo pipefail`). Fix the cause and
  re-run the *same* command: finished arms are skipped (`SKIP_FINISHED`) and unfinished
  ones resume from their `model_latest.pth`, so nothing is lost. The failed arm's output
  is in `$CKPT_ROOT/logs/<run_name>.log`.

---

## 8. After training: the landscape / figure stage

These scripts are **untracked** (rsync them, §2) and need the checkpoints from §5 *plus*
the planning stack (MuJoCo/gym/d4rl/EGL, §4) because they run the planner itself;
the figure stages are CPU-only.

| script | what it does |
|---|---|
| `run_scripts/run_landscape_night.sh` | staged run: `preflight` (argv check, no GPU) → `gates` (correctness gates, aborts on failure) → `probe` (a 300-step saturation grid, decides whether 100 GD steps is enough) → `sweep` (4 arms × episodes × `GRID`² cells × `STEPS` GD steps) → `figures` (Fig. 4 hero + Fig. 6 diagnostics + Fig. 5 planner curves). Knobs: `ENV GRID=13 EPISODES=6 STEPS=100 ACTION_RANGE=3.5 PROBE_STEPS=300 ARMS OUTDIR LEGACY_DIR CURVES_DIR SKIP_VERIFY`. Measured cost on the laptop (~0.117 s per cell per GD step, batch 1): gates ≈ 9 min, probe ≈ 48 min, sweep ≈ 13.2 h (13²/100 steps/24 grids), figures ≈ 3 min. Grids already on disk are **reused**, so a killed run resumes by re-running it. |
| `run_scripts/run_landscape_paperstyle_full.sh` | end-to-end overnight variant that ends with the **paper-style** Fig. 4 (see below). |
| `analysis/landscape_paper_figure.py` | the renderer, incl. `--paper-style --paper-levels 42 --paper-render contourf --paper-cmap hot` (flat-fill colours, paper layout) → `fig4_paperstyle_<env>_ep###_<a>_vs_<b>[_tag][_ar].png` + a `.txt` sidecar. |
| `analysis/landscape_sweep.py`, `landscape_metrics.py`, `landscape_metric_audit.py`, `loss_landscape_comparison.py`, `curvature_distributions.py`, `linear_probe.py`, `helpers/extract_planner_curves.py` | the rest of the metric/figure pipeline (all untracked, rsync them). |

Outputs land in `analysis_outputs/…` (gitignored), logs in
`analysis_outputs/paper/logs/`. The write-up of what these figures mean, how to redraw
them and the verified-vs-paper panel comparison is `results/LANDSCAPE_RESULTS.md`
(gitignored — copy it over, §2).

Start with the free stage:

```bash
bash run_scripts/run_landscape_night.sh preflight     # replays every stage's argv, no GPU, seconds
```

---

## 9. First-run checklist on the server

1. `git switch train-server` … `git log -1 --oneline` shows the commit that added
   `run_scripts/train_server.sh` and this file; `git status` should be clean apart from
   your local edits.
2. `conda activate ts && python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
   → prints `2.3.0+cu121 True <your card>`.
3. Data present: `ls data/datasets` (or your own `DATA_DIR_*` paths) shows
   `point_maze`, `point_maze_medium`, `pusht_noise`.
4. `DRY_RUN=1 bash run_scripts/train_server.sh <env> <recipe>` for all six combinations →
   each prints 4 argvs and exits 0 (this is what was verified on the laptop).
5. Smoke test on one arm (§5) → 1 epoch completes, and
   `torch.load(.../checkpoints/model_latest.pth).keys()` contains `decoder`.
6. Check the DINOv2 cache was populated (§4) so later runs don't depend on the network.
7. Launch the six real jobs (tmux/`nohup`, §5) and watch the first `train.log` for one
   epoch to get the real seconds/epoch on that card before walking away.

When something looks off: `train.log` + `hydra.yaml` inside the run dir reproduce the
exact configuration and the error, and `DRY_RUN=1` re-prints the argv that produced it.



