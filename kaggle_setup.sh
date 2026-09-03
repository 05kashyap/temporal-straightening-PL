#!/usr/bin/env bash
# =============================================================================
# kaggle_setup.sh -- install ONLY the training dependencies for this repo.
#
# Kaggle notebooks do not need (and generally cannot easily get) MuJoCo / gym /
# d4rl / mujoco_py: those are used exclusively by planning (plan.py), never by
# train.py. This script installs the minimal training stack.
#
# Run inside a Kaggle notebook cell:
#     !bash kaggle_setup.sh
#
# Requirements:
#   - Kaggle notebook with GPU accelerator (T4 x2 / P100). Internet access ON
#     (Settings -> Internet), needed for torch.hub to download DINOv2 weights
#     and code on the first training run (~84 MB, cached under ~/.cache/torch).
#   - Python 3.10/3.11 (current Kaggle images) is fine for every pin below.
#
# Note on the pip pins: torch 2.3.0 + torchvision 0.18.0 are the versions the
# repo was developed against (the PyPI Linux wheels are the cu121 CUDA builds).
# wandb is NOT pinned to the repo's 0.13.1 on purpose: 0.13.1 depends on the
# ancient sdist-only 'pathtools' package, which fails to build on Kaggle's
# Python. wandb 0.19.1 (same init/log/watch API the repo uses) no longer pulls
# pathtools.
# hydra-submitit-launcher is required because conf/train.yaml composes
# `override hydra/launcher: submitit_slurm` even for a single-process run.
# =============================================================================
set -euo pipefail

pip install -q --no-input --upgrade pip setuptools wheel

pip install -q --no-input \
    "torch==2.3.0" \
    "torchvision==0.18.0" \
    "numpy<2" \
    "einops==0.4.1" \
    "omegaconf==2.3.0" \
    "hydra-core==1.2.0" \
    "hydra-submitit-launcher==1.2.0" \
    "accelerate==0.26.1" \
    "wandb==0.19.1" \
    "decord==0.6.0" \
    "tqdm" \
    "psutil" \
    "Pillow" \
    "PyYAML"

echo
echo "Kaggle training dependencies installed."
python - <<'PY'
import torch, torchvision, decord, hydra, omegaconf, accelerate, psutil
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("torchvision", torchvision.__version__, "| decord OK | hydra", hydra.__version__)
PY
