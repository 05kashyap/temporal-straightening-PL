# `old-harness` - the pre-`600c36f` harness that produced the good results

Based on commit **c05b143** ("added mpc results", 2026-08-22) - the code + environment
from when `twothirds + straighten` beat `straighten` on umaze/pusht
(MPC GD success: both 0.90/0.92 vs straighten 0.88/0.86).

## Why this branch exists
- It carries the **original `environment.yaml`** (279-line conda export, exact pinned
  builds). `600c36f` later rewrote it to 75 lines with different pins - the most likely
  cause of the drift between the documented-good numbers and later re-runs.
- The MPC harness settings here match the documented good run:
  `gd_mpc`, `n_evals=50`, `goal_H=25`, `seed=100`, `max_iter=20`, `chunk_size=1`,
  `objective.alpha=0 mode=all`, `opt_steps=100`.

## Only change vs c05b143: system-agnostic paths
So it runs after cloning on any machine:
- `setup.sh`: `DATASET_DIR` -> `${DATASET_DIR:-$PWD/data/datasets}`; `PYFLEXROOT` -> `$HOME/PyFleX` (if present).
- `run*.sh`: Python path `/home/shanveen-ortho-clinic/miniconda3/...` -> `$HOME/miniconda3/...`.

## Setup (new machine)
```bash
conda env create -f environment.yaml && conda activate ts
mkdir -p ~/.mujoco && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/ && tar -xzf ~/.mujoco/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
export DATASET_DIR=/path/to/datasets   # contains point_maze/, point_maze_medium/, pusht_noise/, wall_single/
source setup.sh
```

## Train (umaze, 4 variants)
```bash
M=umaze
for V in "False False baseline" "cos1e-1 False straighten" "False twothirds5e-2 twothirds" "cos1e-1 twothirds5e-2 both"; do
  set -- $V
  python train.py --config-name train.yaml env=point_maze encoder=dino_global \
    training.straighten=$1 training.twothirds=$2 \
    training.batch_size=32 training.epochs=20 model.train_decoder=False has_decoder=False \
    hydra.run.dir=./checkpoints/test/${M}_$1_tt$2_agg32_projglobal_dim384_hw1_sgTrue_lr1e-05
done
```

## Eval (MPC, matches the documented protocol)
```bash
CKBPT=./checkpoints/test bash run_mpc.sh umaze all gd_mpc
```
