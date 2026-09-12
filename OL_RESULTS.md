Experiments were run on the following environments: **PointMaze-umaze** and **PointMaze-medium**, **PushT**, and **Wall**.
All train decoder-less world models and evaluate open and closed loop planning. All evals were single seed. 

## Power Law Regularizer (P-Reg)

The idea is to reduce the speed of trajectories in places with high curvature. If the spacing of latent trajectories that the planner needs to reach is proportional to the amount of curvature it could help with planning. 

This comes naturally from the two-thirds power law which states that the instantaneous speed of a biological movement and the curvature of its path are coupled by an empirical invariant. I use this as motivation behind my regularizer (P-Reg)

### Implementation
- from z_t-1, z_t and z_t+1 we get v1 and v2.
- Speed s = (||v1|| + ||v2||) / 2   
- theta = angle bw (v1, v2)  
- k = theta / s                     --> # curvature

It re-uses quantities already calculated by the straightening regularizer.

- r = log s + 1/3log k 

The term r couples speed and curvature. We want this term to remain constant so that an increase in curvature results in lower speeds along the trajectory window.

Then I regularize its variance along the trajectory rather than its value (this only constrains the coupling exponent, not the absolute speed or curvature, so it does not compete with L_pred for absolute scale).

- L_PReg = Var(r) over t

In my experiments num_hist=3 and num_pred=1. So there are only two residuals calculated (Very local). This is done for compute reasons. Could change window size later.

- (L_PReg is just squared distance between r_1 and r_0 currently.) 
- Temporal straightening takes mean of 2 curvatures in my experiments

# Closed-Loop MPC Results

Each eval episode is executed in the real sim and re-planned until success or the iteration cap. Per env, the four variants are **baseline**, **straighten**,
**P-Reg**, and **P-Reg + straighten** (the "both" model). 

## MPC settings

- `n_evals=50` - 50 executed episodes per eval; `SEEDS="100 101 102"` runs one plan.py eval per seed and
  reports mean +/- std (per-seed `logs.json` lives in `plan_outputs_<planner>/`, aggregated in
  `plan_outputs_<planner>/summaries/`).
- `goal_H=25` - the **goal horizon**. Each eval's start and goal states are sampled from the same held-out
  trajectory segment (`goal_source='dset'`), with the goal state taken 25 env steps after the start state.
  The goal is therefore guaranteed to be reachable from its start within 25 steps. Because the world model
  steps at `frameskip=5`, 25 env frames = 5 model steps; plan.py divides `goal_H` by `frameskip` and the
  sub-planner's lookahead horizon is set to exactly this.
- MPC loop: `n_taken_actions=5` env actions (= 1 world-model step, frameskip 5) executed per iteration,
  `max_iter=20` (safety cap; the loop exits early on success). Every iteration re-plans the full remaining
  horizon in latent space and executes its first `n_taken_actions` actions in the real sim.
- **Sub-planners** -- both score action sequences with the same latent-space objective, but they differ in
  *how* and *how many* candidates they evaluate, which is why the reported planning budgets differ:
  - **GD-MPC** (`planning/gd.py`): gradient-based. One action sequence per episode is a *differentiable*
    tensor optimized by Adam (`lr=0.1`, zero init, cosine schedule) for `opt_steps=100` iterations (all
    episodes batched through the same rollouts). Each iteration is a full world-model rollout **with
    backprop**, so the per-episode planning budget is just **`opt_steps`** (= 100 gradient steps per
    replan); there is no sample count and the cost does not scale with one.
  - **CEM-MPC** (`planning/cem.py`): derivative-free stochastic search. Each of `opt_steps=30` iterations
    independently draws `num_samples=300` candidate action sequences per episode from a Gaussian
    `N(mu, sigma)`, rolls them out in parallel under `no_grad`, keeps the `topk=30` lowest-objective
    elites, and refits `mu`/`sigma` from them (the mean is executed). Every candidate costs only a
    forward pass, so the per-episode planning budget is **`num_samples x opt_steps`**
    (300 x 30 = 9,000 forward rollouts per replan) -- two orders of magnitude more objective evaluations
    than GD's 100, at the price of having no gradient signal to guide the search.
- Objectives: umaze/medium `objective.alpha=0 mode=all` (weighted intermediate-state loss); pusht GD-MPC
  `alpha=1 mode=staged`; pusht CEM `alpha=1 mode=last`.

---
## PushT

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline | 0.7134 +/- 0.0573 | 117.6407 +/- 3.4748 | 3.2984 +/- 0.2459 | 40.6807 +/- 2.0840 | — | — |
| straighten | 0.8267 +/- 0.0249 | 77.6528 +/- 5.4283 | 2.6689 +/- 0.0680 | 28.4118 +/- 1.9050 | — | — |
| p-reg | 0.8467 +/- 0.0094| 66.2865 +/- 4.5531 | 2.8093 +/- 0.1436 | 20.88 | — | — |
| **p-reg+straighten** | **0.9134 +/- 0.0094** | 58.8011 +/- 3.576| 2.6964 +/- 0.0586 | 21.6437 +/- 1.6288 | — | — |

- Planning cost (total MPC iters / 50 eps): GD 520/413/432/378 (baseline / straighten / p-reg  /
  both).
  
## PointMaze-umaze

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline | 0.700 | 3.220 | 0.490 | 1.191 | — | — |
| straighten | 0.880 | 3.892 | 0.460 | 1.325 | — | — |
| p-reg | 0.660 | 3.867 | 0.471 | 1.427 | — | — |
| **p-reg +straighten** | **0.900** | 3.756 | 0.460 | 1.268 | — | — |

- Planning cost (total MPC iters / 50 eps): GD 541/366/549/311 (baseline / straighten / p-reg  /
  both) — fewer iterations means earlier success on average.

### Tests with three seeds

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|

| straighten |  |  |  |  | — | — |
| p-reg +straighten (Win3) | 0.7400 +/- 0.0589 |  |  |  | — | — |
| p-reg +straighten (Win7) | **0.7867 +/- 0.0525** |  |  |  | — | — |
## PointMaze-medium

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline | 0.620 | 4.132 | 0.301 | 1.550 | — | — |
| straighten | 0.700 | 4.814 | 0.290 | 1.699 | — | — |
| p-reg | 0.600 | 4.385 | 0.306 | 1.736 | — | — |
| **p-reg +straighten** | **0.760** | 4.309 | 0.290 | 1.513 | — | — |

- Planning cost (total MPC iters / 50 eps): GD 559/491/542/528 (baseline / straighten / p-reg /
  both).

---

# Open-Loop MPC Results
(`max_iter=1`).

## PushT results (encoder = dino_channel)

All four models trained at: `batch_size=16`, `epochs=2`,
`frameskip=5`, `num_hist=3`, `TRAIN_DECODER=False`.


| Model (dino_channel) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline | 0.74 | 0.72 | 50.62 | 50.82 |
| straightening only | 0.80 | 0.78 | 51.41 | 53.06 |
| p-reg only | 0.54 | 0.50 | 41.66 | 51.78 |
| **both** | **0.80** | **0.80** | 37.73 | 45.41 |


---

## PointMaze (umaze) results

| Model (dino_global) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline | 0.18 | 0.56 | 3.54 | 2.73 |
| straightening only | 0.44 | 0.52 | 2.36 | 2.76 |
| p-reg only | 0.20 | 0.58 | 3.68 | 2.97 |
| **both** | **0.60** | **0.62** | 2.33 | 2.82 |


- Planning is open-loop (`MPC max_iter=1`): one action sequence is planned in latent space
(5 latent steps x 5 frameskip = 25 env frames) and executed.

---

## PointMaze (medium) results

Same protocol as umaze (`dino_global`, decoder-less, 20 epochs, open-loop GD/CEM at `goal_H=25`)
but on the D4RL maze2d-**medium** layout (4000 x 100-step episodes).

| Model (dino_global) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline | 0.08 | 0.38 | 4.31 | 3.26 |
| straightening only | 0.12 | 0.46 | 3.69 | 3.36 |
| p-reg only | 0.22 | 0.24 | 3.85 | 3.70 |
| **both** | **0.16** | **0.46** | 4.03 | 3.19 |

---

## Wall results (encoder = dino_channel)

Wall uses the paper's setup: start/goal states are sampled from **test trajectories** (`goal_source='dset'`,
the plan_gd/plan_cem default) so every goal is reachable within 25 steps.

| Model (dino_channel) | GD success | CEM success | GD state-dist | CEM state-dist |
|---|---|---|---|---|
| baseline | 0.36 | 0.44 | 6.56 | 5.54 |
| **straightening only** | **0.90** | **1.00** | 1.99 | 1.52 |
| p-reg only | 0.68 | 0.68 | 5.29 | 4.55 |
| both | 0.68 | 0.60 | 3.52 | 3.68 |

---
