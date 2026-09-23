# SERVER_CONTEXT.md — running this repo on a new machine


## # Running Code on NYU Torch

This is my quick-reference workflow for running `temporal-straightening-PL` on NYU's Torch HPC cluster.

## 1. Connect to Torch

From my local machine:

```bash
ssh akn7847@login.torch.hpc.nyu.edu
```

My useful Slurm accounts are:

```text
torch_pr_718_cds
torch_pr_718_cilvr
```

For this project, use:

```text
torch_pr_718_cds
```

Check available accounts with:

```bash
my_slurm_accounts
```

All Torch jobs need a valid `--account`.

## 2. VS Code

### Recommended: Torch code-server

Torch provides VS Code-like `code-server` through Open OnDemand.

Open:

```text
https://ood.torch.hpc.nyu.edu
```

Then:

1. Log in.
2. Open **Interactive Apps**.
3. Start the **code-server / VS Code** app.
4. Click **Connect to VS Code** when the session starts.
5. Open:

```text
/home/akn7847/wm/temporal-straightening-PL
```

The code-server session is mainly for editing/browsing. Do not depend on it for the GPU itself.

### Local VS Code Remote-SSH

This is possible, but Torch's Microsoft device authentication can be awkward inside VS Code. If using it, the working SSH command is:

```bash
ssh akn7847@login.torch.hpc.nyu.edu
```

## 3. Start a GPU session

For interactive development/testing:

```bash
srun \
  --account=torch_pr_718_cds \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=04:00:00 \
  --pty bash
```

A shorter allocation is better when only testing.

Example for a quick test:

```bash
srun \
  --account=torch_pr_718_cds \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=10G \
  --time=00:15:00 \
  --pty bash
```
Full-size interactive allocation (enough for the six-configs-on-one-GPU
run below):

```bash
srun \
  --account=torch_pr_718_cds \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=48:00:00 \
  --pty bash
```

Check the node:

```bash
hostname
nvidia-smi
```

## 4. Enter the project container

The project uses this Torch CUDA image:

```text
/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

The persistent writable overlay is:

```text
$SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3
```

Launch it with:

```bash
apptainer shell --fakeroot --nv \
  --overlay $SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3 \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

`--nv` exposes the NVIDIA GPU to the container.

`--fakeroot` is required for the writable overlay setup.

Do not run heavy container setup/install work on the login node; use a compute allocation.

## 5. Activate the Conda environment

Inside the container:

```bash
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
conda activate ts
```

Project directory:

```bash
cd /home/akn7847/wm/temporal-straightening-PL
```

The Conda environment is stored persistently in the overlay.

## 6. Verify GPU/PyTorch

Run:

```bash
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'
```

Known-good result:

```text
PyTorch: 2.3.0
CUDA: 12.1
CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
```

Check important project packages:

```bash
python -m pip list | grep -E 'torch|torchvision|mujoco|d4rl|dm-control'
```

Expected versions:

```text
d4rl          1.1
mujoco        3.2.7
mujoco-py     2.1.2.14
torch         2.3.0
torchvision   0.18.0
```

## 7. Run the project

Once inside the container with `ts` activated:

```bash
cd /home/akn7847/wm/temporal-straightening-PL
```

Then run the project's normal commands, for example:

```bash
python <project_script>.py
```

or its existing shell scripts:

```bash
bash <script>.sh
```

If the project uses Hydra/Submitit, let the project's existing configuration handle the experiment submission rather than manually changing its environment unless necessary.

## 8. Batch jobs for long experiments

For anything that should run unattended, use `sbatch` instead of keeping an interactive terminal open.

Basic structure:

```bash
#!/bin/bash
#SBATCH --job-name=ts
#SBATCH --account=torch_pr_718_cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

apptainer exec \
  --fakeroot \
  --nv \
  --overlay "$SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -lc '
    set -e

    export PATH=/opt/miniconda/bin:$PATH
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate ts

    cd /home/akn7847/wm/temporal-straightening-PL

    python <project_script>.py
  '
```

Submit:

```bash
sbatch run.slurm
```

Monitor:

```bash
squeue -u akn7847
```

Check a completed job:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

Cancel a job:

```bash
scancel <JOBID>
```

### One GPU, all six configs in parallel

Slurm gives a GPU to an *allocation*, not to a process, so six `--gres=gpu:1`
jobs can never share one card: the parallelism has to live inside a single
allocation. `run_scripts/train_server.sh all` does exactly that -- it launches
the six (environment, recipe) configs as concurrent children of one process and
waits for them, each keeping its own run dir, logs and wandb dir.
```bash
# from a login node, inside the repo (ONE job, six configs, ONE GPU)
bash run_scripts/submit_train_grid.sh                 # MODE=one-gpu (default)
MODE=per-gpu bash run_scripts/submit_train_grid.sh    # the old 6-jobs/6-GPUs layout
DRY_RUN=1 bash run_scripts/submit_train_grid.sh       # print the sbatch line only

# already inside an allocation (srun): run the launcher directly
bash run_scripts/train_server.sh all
CONFIGS="umaze:global medium:global pusht:global" bash run_scripts/train_server.sh all
MAX_PARALLEL=3 STAGGER=30 NUM_WORKERS=4 bash run_scripts/train_server.sh all
DRY_RUN=1 bash run_scripts/train_server.sh all        # prints all 24 arm argvs
```
Sizing that matters:

| | per run | x6 |
|---|---|---|
| VRAM | ~20 GB | ~120 GB of the H200's 140 GB (fits, ~15% headroom) |
| CPU | 1 trainer + `NUM_WORKERS` loaders | `--cpus-per-task >= 6*NUM_WORKERS`, e.g. 48 |
| RAM | model + workers | 192G is comfortable |

`submit_train_grid.sh` submits `sbatch --cpus-per-task=48 --mem=192G
--time=48:00:00 run_scripts/train_server.slurm all all`. Those flags override the
`#SBATCH` header inside the (gitignored) `train_server.slurm`, so that
server-local file needs no edits; the launcher's env knobs reach the container
because apptainer passes the host environment through. A job that hits its time
limit is safe to resubmit: `train.py` resumes from `checkpoints/model_latest.pth`
and checkpoints are written at every epoch end (`save_every_x_iterations: 0`). With
the default `epochs_mode=target` the resumed arms stop at `EPOCHS` (no over-training),
and the launcher skips arms/configs that already finished.

Watch out for:

- `NUM_WORKERS=16` x 6 = 96 dataloader workers. GRID mode defaults it to 4, and
  pins `OMP_NUM_THREADS=nproc/MAX_PARALLEL` (torch otherwise lets every process
  use every core).
- IO, not VRAM, is usually the limit: `PointMazeDataset._load_episode_visual_tensor`
  (`datasets/point_maze_dset.py:132`) `torch.load`s a whole episode for every
  sampled slice and caches nothing, so many workers mean heavy repeated reads
  from /scratch. A per-worker episode cache is the follow-up if this bites.
- `STAGGER=30` spaces the launches; they all call `torch.hub.load` for DINOv2 at
  startup (`models/dino.py:146`).
- If the three `global` (batch 32) runs OOM, stage them with `CONFIGS=...` or
  lower `MAX_PARALLEL`.

## 9. Important filesystem locations

### Project

```text
/home/akn7847/wm/temporal-straightening-PL
```

### Scratch

```text
/scratch/akn7847
```

Use `$SCRATCH` for large datasets, checkpoints, logs, and temporary files.

Current personal scratch quota is approximately:

```text
5 TB / 5 million inodes
```

### Container overlay

```text
$SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3
```

### Miniconda inside overlay

```text
/opt/miniconda
```

### Conda environment

```text
/opt/conda-envs/ts
```

### Conda package cache

```text
/opt/conda-pkgs
```

## 10. Important container rule

The base `.sif` is read-only. The persistent overlay contains the writable software environment.

Therefore:

- Project source code stays under `/home/akn7847/wm/temporal-straightening-PL`.
- Large datasets/checkpoints should generally go under `$SCRATCH`.
- Installed software/Conda packages live in the persistent overlay.
- Always attach the same overlay when using the environment.

Avoid simultaneously opening the writable overlay from multiple processes/jobs. For production jobs, the overlay should preferably be mounted read-only if possible.

## 11. Typical workflow

### Edit code

Use Torch code-server:

```text
Open OnDemand → Interactive Apps → code-server → Connect to VS Code
```

Open:

```text
/home/akn7847/wm/temporal-straightening-PL
```

### Run a quick test

From a terminal:

```bash
srun \
  --account=torch_pr_718_cds \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=10G \
  --time=00:15:00 \
  --pty bash
```

Then:

```bash
apptainer shell --fakeroot --nv \
  --overlay $SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3 \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
conda activate ts

cd /home/akn7847/wm/temporal-straightening-PL
```

### Run a long experiment

Create an `sbatch` script and submit:

```bash
sbatch run.slurm
```

This is preferable to leaving an interactive SSH/code-server terminal running for hours.

## 12. Troubleshooting

### `python` or `conda` not found

Inside the container:

```bash
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
```

Then:

```bash
conda activate ts
```

### CUDA is unavailable

Make sure all three conditions are true:

1. You are on a GPU Slurm allocation.
2. The container was launched with `--nv`.
3. The `ts` environment is activated.

Check:

```bash
nvidia-smi
```

and:

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")'
```

### Overlay permission error

Use:

```bash
apptainer shell --fakeroot --nv \
  --overlay "$SCRATCH/containers/temporal-straightening/overlay-50G-10M.ext3" \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

### Job is queued

Check:

```bash
squeue -u akn7847
```

For a job's details:

```bash
scontrol show job <JOBID>
```

Smaller resource requests can sometimes schedule sooner.

### Check job output

For an `sbatch` job:

```bash
cat <job-name>-<JOBID>.out
cat <job-name>-<JOBID>.err
```

or:

```bash
tail -f <job-name>-<JOBID>.out
```

## 13. Current known-good environment

As of September 19, 2026:

```text
Cluster:        NYU Torch
Account:        torch_pr_718_cds
Project:        temporal-straightening-PL
GPU tested:     NVIDIA A100-SXM4-80GB
Host driver:    610.43.02
Container:      CUDA 12.1.1 / Ubuntu 22.04
nvcc:           CUDA 12.1.105
PyTorch:        2.3.0
Torchvision:    0.18.0
Python:         3.9.23
MuJoCo:         3.2.7
mujoco-py:      2.1.2.14
D4RL:           1.1
Conda env:      ts
```

The environment has already been successfully tested with:

```text
CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
===== COMPLETE =====
```

## Official NYU references

- Torch VS Code / code-server: https://services.rt.nyu.edu/docs/hpc/tools_and_software/vscode_remote_ssh_torch/
- Torch Slurm jobs: https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/
- Torch Apptainer: https://services.rt.nyu.edu/docs/hpc/tutorial_apptainer/running_containers/
- Conda with Singularity/Apptainer: https://services.rt.nyu.edu/docs/hpc/containers/singularity_with_conda/
- Torch PyTorch guide: https://services.rt.nyu.edu/docs/hpc/ml_ai_hpc/pytorch_intro/

## Hand-off notes for reproducing the Temporal Straightening training run on a server.
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

**Not** in git — a clone will not bring these:

| gitignored (`.gitignore`) | why it matters |
|---|---|
| `data/` | the datasets (§3) |
| `checkpoints/` | training run dirs (§6) |
| `*outputs*` → `analysis_outputs/` | landscape/sweep outputs |
| `papers/` | the paper text is not distributed with the repo |
| `results/`, `Old results/` | older write-ups. **As of 2026-09-18 `results/` no longer exists in the laptop working tree**: `results/LANDSCAPE_RESULTS.md` (the Fig. 4-6 write-up) went away with the pre-`train-server` docs and is not in git history, so bring your own copy if you have one — otherwise the figure scripts + their `--help` are the reference |
| `wandb/`, `figures/` | logs / old figures |

Everything that used to be untracked or uncommitted is **now committed on
`train-server`** (tip `f2a53d7`, "*Updated vwm to parse cpkt paths properly*"): the whole
`analysis/landscape_*` + `loss_landscape_comparison.py` figure pipeline, `run_scripts/*.sh`
(incl. `setup.sh`, `run.sh`, `run_mpc.sh`, the landscape runners), `helpers/`, `kaggle/`,
and the previously-uncommitted `conf/train.yaml` run-dir fix plus the log-only
`models/visual_world_model.py` change. The laptop tree is clean and in sync with
`origin/train-server`, so **a clone of this branch has all the code** — in particular
`conf/train.yaml` now encodes the two-thirds component in the run-dir name, so
`baseline`/`p_reg` and `straighten`/`both` can no longer share a folder even if you
invoke `train.py` directly instead of the launcher.

So the only things to move by hand are the datasets (§3), the DINOv2 cache if the
server is offline (§4), and (optionally) the paper text:

```bash
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
| `EPOCHS` | `20` | target total per arm (paper), see §6 |
| `BATCH_SIZE` | `32` (global) / `16` (channel) | lower it if the card OOMs |
| `NUM_HIST` | `3` | paper Table 3 |
| `NUM_WORKERS` | env yaml (`16`) | 6 parallel jobs × 16 = 96 workers; lower it on a small node |
| `REG_WINDOW` | untruncated (`num_hist+num_pred`) | P-Reg stats window |
| `FRESH` | `0` | `1` = delete the 4 run dirs before training |
| `DRY_RUN` | `0` | `1` = print argvs, train nothing |
| `SKIP_FINISHED` | `1` | skip an arm that already reached `EPOCHS` (target mode only) |
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
checkpoints/model_latest.pth        # every epoch
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
(weights, optimizers, epoch, mid-epoch batch counter) -- there is no flag.
`training.epochs` is the **target total** (`training.epochs_mode=target`, the default): an arm
saved at epoch 12 with `EPOCHS=20` trains 13..20 and stops, so re-submitting a job that hit its
wall clock finishes the run instead of adding 20 more epochs. `SKIP_FINISHED=1` (default) skips
an arm that already reached `EPOCHS`, and grid mode skips whole configs whose four arms are done.
Check first with `bash run_scripts/train_server.sh status` (or `STATUS=1`), extend an arm with a
larger `EPOCHS` (e.g. `EPOCHS=25` trains 21..25), and use `FRESH=1` only to restart from scratch.
`training.epochs_mode=additional` restores the legacy per-launch behaviour.

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
`analysis_outputs/paper/logs/`. The old write-up of what these figures mean, how to
redraw them and the verified-vs-paper panel comparison
(`results/LANDSCAPE_RESULTS.md`) is gone — see §2 — but each script's `--help`
documents its stages and knobs.

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
7. Launch the six real jobs (tmux/`nohup`, §5) and watch the first arm's
   `$CKPT_ROOT/logs/<run_name>.log` for one epoch to get the real seconds/epoch on that
   card before walking away.

When something looks off: the run dir's `hydra.yaml` plus its
`$CKPT_ROOT/logs/<run_name>.log` reproduce the exact configuration and the error, and
`DRY_RUN=1` re-prints the argv that produced it.




---

## 11. MPC / planning stage (MuJoCo + EGL)

Training needs none of this. `plan.py` / `run_mpc.sh` do, because the point-maze envs
are MuJoCo gym envs:

```
plan.py -> gym.make("point_maze")            (env/__init__.py)
        -> env.pointmaze.PointMazeWrapper
        -> env/pointmaze/maze_model.py: from gym.envs.mujoco import mujoco_env
        -> mujoco_py -> the MuJoCo 2.1.2 binaries in $MUJOCO_PY_MUJOCO_PATH
```

`train_server.sh` deliberately sources none of this, and **do not source
`~/mujoco_env.sh` before a training run**: it adds the MuJoCo libs to `LD_LIBRARY_PATH`,
and *prepending* them ahead of the conda libs is the suspected cause of the
`cannot import torch` failure seen in the resume job (§7). Use `MUJOCO_LD_MODE=append`
(default in `~/mujoco_env.sh`, supported by `run_scripts/setup.sh`) so the MuJoCo dirs
end up last.

### 11.1 One-time setup (do it in a gap between training jobs)

```bash
# MuJoCo 2.1.2 (mujoco210) under ~/.mujoco -- one-time download:
mkdir -p "$HOME/.mujoco"
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "$HOME/.mujoco"
tar -xzf "$HOME/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "$HOME/.mujoco"
ls "$HOME/.mujoco/mujoco210/bin"          # libmujoco210.so + libglew{egl,osmesa}.so

# writes ~/mujoco_env.sh, builds mujoco_py's cymj, runs the acceptance test
bash run_scripts/setup_mujoco_server.sh
```

- `import mujoco_py` always needs a **writable** overlay. The first time it compiles
  `cymj` into site-packages *inside the overlay*, and -- on **every** import, even once
  `cymj` is built -- it takes a write lock (`fasteners.InterProcessLock`, `builder.py`)
  at `mujoco_py/generated/mujocopy-buildlock` *before* it checks whether the extension is
  up to date. A `:ro` mount therefore fails at that import with
  `OSError: [Errno 30] Read-only file system`, no matter how often cymj has been built.
  So `--overlay "$OVERLAY"` (no `:ro`) is not a first-run detail: it is required for every
  planning/MPC run, and two jobs must not hold that overlay at the same time. Verified on
  torch: slurm-mpc-18333846 failed exactly there with the default `:ro` mount.
- Exporting `MUJOCO_PY_MUJOCO_PATH` in one interactive shell is not enough: the
  generated `~/mujoco_env.sh` sets it, plus `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
  `D4RL_SUPPRESS_IMPORT_ERROR=1`, `PYTHON=/opt/miniconda/envs/ts/bin/python` and an
  idempotent `LD_LIBRARY_PATH` for every planning process.
- **Home directory / mounts**: apptainer binds only the *current working directory* by
  default, so the repo is visible inside the container while a file written next to it
  (`~/mujoco_env.sh`) is **not** -- sourcing it fails with `No such file or directory`.
  The script therefore runs with `--bind "$HOME:$HOME"`; add the same flag to your own
  interactive / slurm container runs. `--dry-run` prints the exact `apptainer` command
  and the in-container script without executing anything.
- **Login node vs GPU allocation**: the cymj build and smoke stages 1-6 are CPU-only, so
  they can run on a login node (`--skip-render` also skips the render stage):
  `bash run_scripts/setup_mujoco_server.sh --skip-render`. The offscreen render (stage 7)
  and the `FULL=0` / `FULL=1` MPC runs need a GPU: use a slurm allocation
  (`srun --gres=gpu:1 ...`, or a job script as in §8) and run the same script there.
- **Interpreter path**: on this cluster `conda activate ts` lands in
  **`/opt/conda-envs/ts`** (not `/opt/miniconda/envs/ts`, which does not exist here), so
  nothing may hardcode a prefix: the driver exports `PYTHON=$(command -v python)` after
  activating, the generated `~/mujoco_env.sh` discovers it at source time, and
  `setup.sh` / `run_mpc.sh` probe `CONDA_PREFIX` then `/opt/conda-envs/ts` then
  `/opt/miniconda/envs/ts` then `~/miniconda3/envs/ts`. `--dry-run` prints what the
  smoke test will actually run.
- **glew / `__glewBindBuffer`**: `libmujoco210.so` on its own needs glew symbols that
  live in `$MUJOCO_PY_MUJOCO_PATH/bin/libglewegl.so`, and mujoco_py's `cymj` links
  them itself -- so the smoke test reports that loader check as *information*, not a
  failure. If the `cymj` import ever fails with an undefined `glewBindBuffer` symbol,
  prefix `LD_PRELOAD=$MUJOCO_PY_MUJOCO_PATH/bin/libglewegl.so` (the loader line names
  the file that works).
- **mujoco_py needs three local patches here, and the repo scripts them now**:
  `run_scripts/patch_mujoco_py.py` adds (1) the GCC >= 14 compile flags
  (`-Wno-incompatible-pointer-types` etc. + `-DGLEW_NO_GLU`), (2) forcing the
  `LinuxGPUExtensionBuilder` (EGL) unless `MUJOCO_PY_FORCE_CPU` is set -- upstream
  decides via `nvidia-smi`, which does not exist inside apptainer, and the CPU
  builder then needs `GL/osmesa.h` that cannot be installed here -- and (3) a
  `get_nvidia_lib_dir()` that returns a directory which really exists
  (`/usr/local/nvidia/lib64`, `/usr/lib/nvidia`, `/.singularity.d/libs`).
  `bash run_scripts/setup_mujoco_server.sh --fix-mujoco-py` does all of it plus the
  Cython pin, the `generated/` check and a clean rebuild (needs a rw overlay).
- **`mujoco_py/generated/` holds shipped SOURCES** (`__init__.py`, `const.py`,
  `wrappers.pxi`, which `cymj.pyx` includes). Cleaning a build must delete only
  `cymj.c`, `generated/*.so` and `generated/_pyxbld_*`; `rm -rf generated` breaks the
  package (restore with `pip install --no-cache-dir --force-reinstall --no-deps
  mujoco-py==2.1.2.14`).
- **`apt` cannot be used in this image** (the SIF root is read-only and apt cannot
  write `/var/lib/apt`): install anything extra with `pip`/`conda` in the overlay.
- **Inside the container `$HOME` is `/root`** (fakeroot), so never build
  container-side paths from `$HOME`; the bound home is an absolute path like
  `/home/akn7847/...`, and `--bind "$HOME:$HOME"` is what makes it visible.

### 11.2 Acceptance test

`python run_scripts/mujoco_smoke.py` (inside the container, planning env sourced)
checks in order and prints PASS/FAIL per stage: environment -> `import torch` *with the
MuJoCo paths set* -> `import mujoco_py` (builds cymj) -> `libmujoco210.so` via the
loader -> `import env` registers point_maze/point_maze_medium/pusht/wall ->
`gym.make("point_maze")` reset+step -> offscreen render (EGL/GLFW) -> `DATASET_DIR` and
checkpoint paths (warn only). The `libmujoco210.so` loader line is informational (see
11.1 on glew); the required stages are torch, cymj, env registration, make/step and the
render. The two stages before the import report what decides the build: the
toolchain (Cython / gcc / nvidia-smi / the three driver-lib dirs; Cython >= 3 is a
hard failure with the pin as the fix) and which of the three mujoco_py patches are
present. The last stage is per env: `DATA_ROOT` comes from
`run_scripts/dataset_paths.sh` and the correct `DATASET_DIR` is printed for every
env (`$DATA_ROOT/point_maze` for umaze, `$DATA_ROOT` for medium, `$DATA_ROOT/pusht`
for pusht). Exit 0 = the planning stage can run here. Verified on the
laptop in both modes: GLFW `var=1443`, EGL `var=1414` (`Found 4 GPUs for rendering.
Using device 0`).

### 11.3 Running MPC

```bash
source ~/mujoco_env.sh
export PYTHONPATH="$PWD"                       # run_mpc.sh does this too

FULL=0 bash run_scripts/run_mpc.sh umaze False gd_mpc --ckpt "$CKPT_ROOT/test" --seeds 100
FULL=1 bash run_scripts/run_mpc.sh umaze False gd_mpc --ckpt "$CKPT_ROOT/test" --seeds 100
FULL=1 bash run_scripts/run_mpc.sh umaze all   both   --ckpt "$CKPT_ROOT/test"           # 4 arms x 2 planners
OL=1   bash run_scripts/run_mpc.sh umaze all   both   --ckpt "$CKPT_ROOT/test"           # open loop
```

- `FULL=0` is `run_mpc.sh`'s validation mode: OOM check + runtime estimate. Run it first.
- `CKBPT` defaults to the laptop path `checkpoints/test`; on the server pass
  `--ckpt $CKPT_ROOT/test` (a directory of arm run dirs, each with
  `checkpoints/model_latest.pth`).
- Outputs: `plan_outputs_gd_mpc/<model>_s<seed>_gH25/` + `summaries/<model>_gH25.json`;
  open loop goes to `plan_outputs_gd_ol/` so the closed-loop logs are never overwritten.
  `python analysis/div_emb_tables.py` renders those summaries into markdown tables.
- Slurm: same skeleton as §8 with `--gres=gpu:1 --cpus-per-task=8`; inside the
  `apptainer exec` body add `source ~/mujoco_env.sh` before `bash run_scripts/run_mpc.sh`.
- **Arm names are per-machine.** The four `MODELS` per env inside `run_mpc.sh` are the
  dev machine's run-dir names. The checkpoints trained *on this cluster* use different
  recipes, so pass `--ckpt <one run dir>` per arm, or all four at once via `ARM_NAMES`
  (index-aligned with `all|False|straighten|twothirds|both`):

  ```bash
  ARM_NAMES="umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06 \
             umaze_aggcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
             umaze_ttaggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
             umaze_aggcos1e-1_aggtwothirds5e-2_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05" \
  bash run_scripts/run_mpc.sh umaze all gd_mpc --ckpt "$CKPT_ROOT/test" --seeds 100
  ```

  A name that does not exist under `CKBPT` now aborts with the list of arm dirs that **do**
  exist for that env, so the fix is one line instead of a guess.
- **Datasets**: `DATASET_DIR` must be the parent of `point_maze/`, `point_maze_medium/`,
  `pusht_noise/`, and each of those must directly contain `states.pth`, `actions.pth`,
  `seq_lengths.pth` (pusht: `seq_lengths.pkl` + `rel_actions.pth`/`abs_actions.pth`) and an
  `obses/` directory. On this cluster the files sit one level deeper than the directory
  name suggests (`/scratch/akn7847/datasets/worldmodeldata/point_maze/point_maze`), so the
  value to use is `export DATASET_DIR=/scratch/akn7847/datasets/worldmodeldata/point_maze`
  -- the smoke test detects that nesting and prints the exact line to run.
- Mount flags for a planning job: `--bind "$HOME:$HOME"` (so `~/mujoco_env.sh` is
  readable) and `--overlay "$OVERLAY"` -- mounted **read-write**. It is tempting to think
  planning only reads the overlay, but `import mujoco_py` takes a write lock in
  `mujoco_py/generated/` on every import (11.1), so `:ro` cannot work; with
  `run_scripts/mpc_server.sh` the knob is `OVERLAY_RW=1` (if the preflight fails while the
  mount is `:ro`, the wrapper's own error message now says so). Consequence: an MPC run
  cannot overlap another job holding the same overlay, since ext3 overlays are single-mount.
- **No data/env exports needed**: the driver sources the shared layout, and
  `run_mpc.sh` derives `DATASET_DIR` per env from `DATA_ROOT` itself (printing
  `data: DATASET_DIR=...` in its header). Override `DATA_ROOT=` on the command line or
  pass `--data-root=DIR` to the driver. `ARM_NAMES` / `--arms-from-ckpt` remains the
  per-cluster input for MPC can be omitted entirely (these checkpoints use `projchannel`/`ttagg…`/
  `aggflatten` names rather than the built-in dev names).
- Recovery: if `import mujoco_py` ever regresses, run
  `bash run_scripts/setup_mujoco_server.sh --fix-mujoco-py` (Cython pin + patch +
  clean rebuild in one command).
- **Slurm wrapper**: `run_scripts/mpc_server.sh` is sbatch-able (`sbatch
  run_scripts/mpc_server.sh`; the `.slurm` extension is gitignored, hence the `.sh` name).
  It runs the jobs sequentially inside one GPU allocation, tees each into
  `$CKPT_ROOT/logs/mpc_<env>_<variant>_<planner>.log`, runs `mujoco_smoke.py` as a
  preflight (`PREFLIGHT=1`), prints a success-rate summary, and needs no exports:

  ```bash
  sbatch run_scripts/mpc_server.sh                        # default: 3 envs x gd_mpc, all arms
  sbatch run_scripts/mpc_server.sh umaze all both         # env variant planner
  FULL=0 sbatch run_scripts/mpc_server.sh                 # validation (OOM + estimate)
  OL=1   sbatch run_scripts/mpc_server.sh                 # open loop
  JOBS="umaze:all:gd_mpc medium:both:both" sbatch run_scripts/mpc_server.sh
  ```

  `DRY_RUN=1 bash run_scripts/mpc_server.sh` prints the apptainer command and the
  generated in-container body without running anything. `OVERLAY_RW=1` only if cymj
  still has to be compiled. Arm names are **discovered per env** by `run_mpc.sh`
  (token match: `_False_`, `cos`, two-thirds, both) with an abort-and-list on
  ambiguity, so the MPC grid no longer needs `ARM_NAMES` at all.
- **Chunking on the server**: `run_scripts/mpc_server.sh` plans *unchunked* by default
  (`CHUNK=null`, `OL_CHUNK=null`, `CEM_CHUNK=null`) -- all `n_evals=50` episodes in one
  batch, which also starts one env process per episode. That is the point of the bigger
  GPU, but it needs CPUs to match -- and a job here is capped at **16 CPUs**
  (`#SBATCH --cpus-per-task=16`, `--mem=64G`), so 50 simulators share 16 cores: the
  script prints a startup hint saying so, and `CHUNK=16 OL_CHUNK=16` matches the
  allocation (3 batches instead of 50) if the run turns out simulator-bound. Other
  fallbacks: `CHUNK=8 OL_CHUNK=8`, `CHUNK=1` for the 12 GB-laptop behaviour,
  `CEM_CHUNK=50` to bound the CEM
  candidate-rollout memory. Chunk size never changes the numbers (the divergence
  metrics are exact Frobenius norms under chunking); `n_evals` does, so keep it at 50.
  `DRY_RUN=1 bash run_scripts/run_mpc.sh ...` prints the resolved chunks and budgets.
- **Submit it from the repo root** (`cd ~/wm/temporal-straightening-PL && sbatch
  run_scripts/mpc_server.sh`). `sbatch` runs a *spool copy* of the script, so
  `$BASH_SOURCE` points at `/opt/slurm/data/slurmd/job<N>/slurm_script` and cannot be
  used to find the checkout; the script therefore looks for the repo via
  `SLURM_SUBMIT_DIR` (where `sbatch` was invoked), then its own directory, then `$PWD`,
  and fails with that advice if none of them contains `run_scripts/` (it also prints
  the resolved `repo :` in its header). Clear `REPO_HOST=/path/to/repo` to override.
  `train_server.sh` and `setup_mujoco_server.sh` have the same fallback.
- **If a job dies instantly, run the self-test before resubmitting**:
  `bash run_scripts/selftest_mpc_server.sh` (login node, ~5 s, no GPU, no container: it
  stubs apptainer/conda/`run_mpc.sh`). The wrapper writes the in-container script as
  *preamble of `export VALUE=...` lines, then the body*; the body runs under `set -u`, so
  a preamble that ends up after the body's first use of a value kills the job inside the
  container. That is exactly how `.ts_mpc_body.sh: line 7: ENV_FILE: unbound variable`
  happened once: the exports were appended with `>>` *after* the body's `exit "$rc"`, so
  they never executed. The wrapper now emits the preamble first **and** lints the
  generated file before starting apptainer (`PREAMBLE_VARS` drives both, so they cannot
  drift): every name must be exported before the first line that uses it. The check runs
  under `DRY_RUN=1` as well, so `DRY_RUN=1 bash run_scripts/mpc_server.sh` is a real
  preflight -- it lints *and* prints. `tests/test_mpc_body.py` pins all of this down,
  including the historical layout, which must still fail. The self-test *injects* the stub
  as `APPTAINER_BIN` rather than relying on PATH order, because bash processes on this
  cluster rewrite `PATH` (site startup file / `BASH_ENV`) -- the wrapper takes
  `APPTAINER_BIN` if it is set and prints the binary it will run on its `tool :` line, so
  which apptainer runs is never a guess.
- The self-test has a **container mode**: `LIVE=1 bash run_scripts/selftest_mpc_server.sh`
  starts the *real* image (`apptainer exec --fakeroot --nv --bind $HOME:$HOME --overlay
  <overlay>:ro`), activates the container's conda env exactly like the job does
  (`CONTAINER_CONDA` / `LIVE_CONTAINER_CONDA`, default `/opt/miniconda`, and `ts`), and
  reports `CONTAINER_OK` + the path and version of the `python` it found. So fakeroot, the
  bind, the overlay, the conda prefix and the env are all verified without importing MuJoCo.
  `LIVE_SIF` / `LIVE_OVERLAY` override the paths, and it *skips* itself (instead of failing)
  when apptainer or the image is missing.
- Check 0 of that script is a gate that proves the **stubs are the binaries that run**. On a
  node whose `/tmp` is `noexec`, bash silently skips a stub it cannot execute and runs the
  next `apptainer` on PATH -- the cluster's real one -- which rejects the empty stub image
  and makes the test report failures that have nothing to do with the wrapper. The script
  now probes for an exec-capable temp directory, falls back to `$HOME`, refuses to continue
  if the stubs are not what runs, and prints which apptainer `LIVE=1` would start.
- Two container-side overrides worth knowing: `CONTAINER_CONDA` (default
  `/opt/miniconda`) and `CONDA_ENV` (default `ts`) select the conda to source and the env
  to activate inside the image, and `REPO_IN_CONTAINER` is now *derived* from the
  resolved `REPO_HOST` when the checkout is under `$HOME` (which is bound 1:1 into the
  container) -- the hardcoded `/home/akn7847/wm/...` remains only as the fallback for a
  checkout outside `$HOME`. The header prints all of it on the `body :` line. Because
  `--bind "$HOME:$HOME"` is the only mount, a checkout *outside* `$HOME` cannot be reached
  inside the image at all: the wrapper now refuses to submit in that case (with the fix
  spelled out) instead of letting the body fail on `cd` after a queue slot -- and
  `ALLOW_OUTSIDE_HOME=1` says "I mounted it myself" if you added a `--bind`.
- The wrapper deliberately avoids `... | head -1` inside command substitutions: under
  `set -o pipefail` an early exit gives the upstream process SIGPIPE, the pipeline is
  non-zero, and `set -e` turns that into a silent exit-1 in the middle of the script.
- `PROBE=1` runs `run_scripts/gl_backend_probe.py` inside the job before the preflight: it
  renders in the main process, in a **forked** child and in a **spawned** child (twice more
  with bare mujoco_py), because `env/venv.py`'s workers are forked and a child that forks
  *after* any GL use can fail to initialise GL (`Failed to initialize OpenGL`) or hang, while
  the single-process smoke-test render passes. When the probe says fork is the problem,
  `TS_ENV_START_METHOD=spawn` gives every env worker a fresh interpreter -- `plan.py` honours
  it and `env/venv.py` already hands the env factory over as a `CloudpickleWrapper`, so spawn
  works without touching the paper's code.
- Run dirs land in the **repo root**: `plan_outputs_<planner>/<model>_s<seed>_gH<H>/`, the
  `plan_outputs_<planner>/validate_*_{setup,smoke}.log` files of a `FULL=0` run, and
  `plan_outputs_<planner>/summaries/*.json`. `run_mpc.sh` therefore works from the repo root
  itself (it used to `cd` into `run_scripts/`, so `plan.py` was not found and the dirs were
  created one level too deep; `run.sh` and `run_wall_ablation.sh` had the same flaw and are
  fixed too). A `FULL=0` run prints its `[estimate]` lines in the summary for this reason.
- The tables find those names too: `analysis/div_emb_tables.py` falls back to the
  same token-based discovery when the built-in names are absent, so
  `python3 analysis/div_emb_tables.py` (host python is enough -- it only reads
  the run dirs) renders the closed-loop section from `plan_outputs_gd_mpc/` and
  the open-loop one from `plan_outputs_gd_ol/`, naming the run dirs it used in
  the provenance table.

### 11.4 Failure -> fix

| symptom | cause / fix |
|---|---|
| `cannot import torch` after sourcing a MuJoCo env file | MuJoCo libs ahead of conda's in `LD_LIBRARY_PATH` -> `MUJOCO_LD_MODE=append` |
| `Read-only file system: .../mujoco_py/generated/mujocopy-buildlock` (or any cymj write) | the overlay was mounted `:ro`. mujoco_py takes that write lock on **every** import, built or not, so re-run writable: `OVERLAY_RW=1 sbatch run_scripts/mpc_server.sh` (the smoke test and `setup_mujoco_server.sh --check` mount it read-write too, for the same reason) |
| cymj build: `gl.h` / `glew` missing | the conda env lost `glew` / `xorg-libx11` / `xorg-xorgproto` (environment.yaml) |
| `libmujoco210.so: cannot open shared object file` | `MUJOCO_PY_MUJOCO_PATH` wrong, or its `bin` is not on `LD_LIBRARY_PATH` -> `MUJOCO_LD_MODE=prepend` |
| two jobs fail on the overlay at once | ext3 overlays are single-mount, and a planning run needs it read-write (row above): it cannot share the overlay with a training job, so serialize those (training keeps the default `:ro`, planning does not) |
| `undefined symbol: __glewBindBuffer` | informational for the standalone loader test (cymj links glew itself). If the *cymj import* raises it: `LD_PRELOAD=$MUJOCO_PY_MUJOCO_PATH/bin/libglewegl.so` |
| `gym.error.NameNotFound: Environment point_maze does not exist` | cascade from `import env` failing -- fix the `mujoco_py`/`gym envs` stage above it (usually the same overlay or cymj problem) |
| `.ts_mpc_body.sh: line N: ENV_FILE: unbound variable` (any `unbound variable`) | the generated body used a value before the preamble exported it (the exports were once appended after the body's `exit "$rc"`). Fixed by writing the preamble first; the wrapper now refuses to start apptainer if that inverts. Verify: `bash run_scripts/selftest_mpc_server.sh` |
| header prints, then the job exits 1 with **no** message | a pipeline under `set -o pipefail` whose first stage dies of SIGPIPE (`... \| head -1`): `set -e` aborts silently. The wrapper avoids that pattern (see 11.3) |
| self-test checks 2/4/6 fail with apptainer's `image format not recognized` on an empty stub image, **while check 0 passes** | the wrapper resolved a *different* apptainer than the gate did: bash processes on that login node rewrite `PATH` (site startup file / `BASH_ENV`), and the stub is executed *by bash* too, so PATH order is unreliable for picking a specific binary. Fixed by passing the binary explicitly: the wrapper uses `APPTAINER_BIN`, the self-test injects its stub that way, and check 0b reproduces a decoy earlier in PATH |
| a temp directory that cannot execute a script (`noexec` `/tmp`) | bash then skips a stub it cannot run and silently uses the next one on PATH. The self-test probes for an exec-capable temp dir and falls back to `$HOME` (its header says "under $HOME") |
| the wrapper ran an apptainer you did not expect | `tool : apptainer=...` in the job header says which one it used; set `APPTAINER_BIN=/path/to/apptainer` to pin it |
| `FATAL: apptainer is not on PATH here` | the wrapper needs the same apptainer you use interactively: load the module/alias, or check the wrapper logic with `run_scripts/selftest_mpc_server.sh` |
| `python: command not found` inside the container (as the LIVE check first reported) | the container's conda env was not activated: the bare image has no `python`. Run `LIVE=1 bash run_scripts/selftest_mpc_server.sh` (it activates `CONTAINER_CONDA`:`CONDA_ENV`, default `/opt/miniconda`:ts, and prints the path it found); a wrong prefix is now a labelled `FATAL: .../etc/profile.d/conda.sh is not visible` / `no python on PATH after activating ...` instead of a cascade |
| `plan_outputs_gd_mpc/validate_<model>_setup.log: No such file or directory`, then every env "skipped", `success_rate=<n/a>`, and a `failed:` list naming only one env | the validate redirects opened their log before anything created that directory, and every failure ended in `continue` so the exit status was incidental. Fixed: `run_mpc.sh` creates the directory first and exits non-zero when a run produced no result (so `mpc_server.sh` prints `[FAIL]`) |
| `can't open file 'plan.py'`, or `plan_outputs_*` appearing under `run_scripts/` | the driver was running from `run_scripts/` instead of the repo root (`cd "$(dirname "$0")"` with no way back). Fixed in `run_mpc.sh` / `run.sh` / `run_wall_ablation.sh` |
| `RuntimeError: Failed to initialize OpenGL` in an `env/venv.py` worker (the smoke test's own render passes, `[timing] setup_model_s=` is already printed) | the env workers are **forked**, and a child that forks after GL has been initialised can fail (EGL) or hang (X11/GLFW) while the parent renders fine. Confirm with `PROBE=1 OVERLAY_RW=1 FULL=0 sbatch run_scripts/mpc_server.sh`, then re-run with `TS_ENV_START_METHOD=spawn` |
| ... and if the probe fails for **both** fork and spawn | the EGL stack itself is unusable here: fall back to the CPU/OSMesa backend (`MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=osmesa`, cymj rebuilt -- see 11.1) and send the probe output to whoever set the container up |
| `FATAL: the checkout is at ... (outside $HOME), but only $HOME is bound` | the repo lives outside `$HOME`, the only path mounted into the image. Submit from a checkout under `$HOME`, or set `REPO_IN_CONTAINER` + add your own `--bind`, then `ALLOW_OUTSIDE_HOME=1` |
| `FATAL: ENV_FILE=... is not readable here` | `~/mujoco_env.sh` is not on this machine: run `run_scripts/setup_mujoco_server.sh` (11.1), or point `ENV_FILE` at your own copy |
| `error: passing argument 1 of ... from incompatible pointer type` | GCC >= 14 promotes it: run `python run_scripts/patch_mujoco_py.py` (adds `-Wno-incompatible-pointer-types` + `-DGLEW_NO_GLU`) |
| `Missing path to your environment variable … :None` | `get_nvidia_lib_dir()` returned None: `patch_mujoco_py.py` adds `/.singularity.d/libs` and creates `/usr/local/nvidia/lib64` |
| `fatal error: GL/osmesa.h` / `linuxcpuextensionbuilder` | the CPU builder was selected: same patch (GPU/EGL), or `MUJOCO_PY_FORCE_CPU=1 MUJOCO_GL=osmesa` with OSMesa from conda |
| `'generated/wrappers.pxi' not found` | `mujoco_py/generated/` was deleted: restore with `pip install --no-cache-dir --force-reinstall --no-deps mujoco-py==2.1.2.14` |
| `Permission denied` from `apt-get` | apt cannot write `/var/lib/apt` in this image; use pip/conda |
| a `data/datasets`-style path in an error | `DATASET_DIR` is per env -- `run_mpc.sh` sets it from `DATA_ROOT`; check the `data: DATASET_DIR=...` line in its header |
| `glfw`/window error, no DISPLAY | `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` (auto-set when `DISPLAY` is empty) |
| `d4rl` warnings about mjrl/flow/carla | harmless; `D4RL_SUPPRESS_IMPORT_ERROR=1` silences them. **No d4rl hdf5 is downloaded** -- the `dataset_url` kwargs are never used (no `get_dataset()` call in the repo) |
| `plan.py` cannot find data | `DATASET_DIR` must contain `point_maze`, `point_maze_medium`, `pusht_noise` (the generated env file points it at `$SCRATCH/datasets`; override if yours differs) |
| `no ts interpreter found` | export `PYTHON=/opt/miniconda/envs/ts/bin/python`; `setup.sh` / `run_mpc.sh` now probe the container prefix too |
