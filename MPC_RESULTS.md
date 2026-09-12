# Closed-Loop MPC Results (temporal-straightening + two-thirds, faithful MPC)

Compiled from this machine (single NVIDIA RTX 4070 12 GB). **Closed-loop** MPC: each eval episode is
executed in the real sim and re-planned until success or the iteration cap, unlike the open-loop
(`max_iter=1`) results in `RESULTS.md`. Per env, the four variants are **baseline**, **straighten**,
**twothirds**, and **twothirds + straighten** (the "both" model).

## Faithful MPC settings (identical for every row)

- `n_evals=50`, `goal_H=25`, `chunk_size=1`, `seed=100`
- MPC loop: `n_taken_actions=5`, `max_iter=20` (safety cap; exits early on success)
- **GD-MPC**: sub-planner GD, `opt_steps=100`, Adam `lr=0.1`, zero init
- **CEM-MPC** (placeholder — not yet re-run with the corrected config): `num_samples=300`,
  `opt_steps=30`, `sample_chunk_size=50` (`conf/plan_mpc_cem.yaml` defaults)
- Objectives: umaze/medium `objective.alpha=0 mode=all` (weighted intermediate-state loss, paper
  Sec 5.3); pusht GD-MPC `alpha=1 mode=staged`; pusht CEM `alpha=1 mode=last`
- Logs: `plan_outputs_gd_mpc/<model>_gH25/logs.json` (last line = `final_eval/...`)

### Metric meanings

- `success_rate`: fraction of the 50 executed episodes that reach the goal.
- `mean_state_dist` / `mean_visual_dist` / `mean_proprio_dist`: mean distance between the final
  executed state and the goal state, in state / visual (DINO feature) / proprio space.
- `mean_div_visual_emb` / `mean_div_proprio_emb`: divergence between imagined (latent) and executed
  (real) embeddings at the final step.

---

## PointMaze-umaze

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline (`False`) | **0.700** | 3.220 | 0.490 | 1.191 | — | — |
| straighten (`cos1e-1`) | **0.880** | 3.892 | 0.460 | 1.325 | — | — |
| twothirds (`tttwothirds5e-2`) | **0.660** | 3.867 | 0.471 | 1.427 | — | — |
| **twothirds+straighten** | **0.900** | 3.756 | 0.460 | 1.268 | — | — |

- GD headline: **both (0.90) > straighten (0.88) > baseline (0.70) > twothirds (0.66)**; both is best
  and straighten is a close second. Note state-dist is lowest for baseline (3.22) — here success
  tracks the classifier/proximity of the reached region more than raw final distance.
- Planning cost (total MPC iters / 50 eps): GD 541/366/549/311 (baseline / straighten / twothirds /
  both) — fewer iterations means earlier success on average.

## PointMaze-medium

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline (`False`) | **0.620** | 4.132 | 0.301 | 1.550 | — | — |
| straighten (`cos1e-2`) | **0.760** | 4.309 | 0.290 | 1.513 | — | — |
| twothirds (`tttwothirds5e-2`) | **0.600** | 4.385 | 0.306 | 1.736 | — | — |
| **twothirds+straighten** | **0.700** | 4.814 | 0.290 | 1.699 | — | — |

- GD headline: **straighten (0.76) > both (0.70) > baseline (0.62) > twothirds (0.60)** — straightening
  alone is best here; adding two-thirds does not help beyond it.
- Planning cost (total MPC iters / 50 eps): GD 559/491/542/528 (baseline / straighten / twothirds /
  both).
- **Caveats (config inconsistencies in the medium checkpoint set):** baseline/straighten were trained
  with `lr=1e-06`, twothirds/both with `lr=1e-05`; and the straighten model uses `cos1e-2`
  (λ=0.01) while the "both" model's straighten term is `cos1e-1` (λ=0.1) — so cross-variant
  comparisons are not perfectly apples-to-apples.

## PushT

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline (`False`) | **0.720** | 118.1 | 3.164 | 42.42 | — | — |
| straighten (`aggcos1e-1`) | **0.860** | 70.3 | 2.681 | 24.76 | — | — |
| twothirds (`aggtwothirds5e-2`) | **0.840** | 57.2 | 3.007 | 20.88 | — | — |
| **twothirds+straighten** | **0.920** | 53.8 | 2.707 | 19.22 | — | — |

- GD headline: **both (0.92) > straighten (0.86) > twothirds (0.84) > baseline (0.72)**; both is best
  and also reaches the lowest final state-dist (53.8) and proprio-dist (19.2).
- Planning cost (total MPC iters / 50 eps): GD 520/413/432/378 (baseline / straighten / twothirds /
  both).

---

## Appendix A — full `final_eval` metrics (GD-MPC)

| Env | Model | success_rate | mean_state_dist | mean_visual_dist | mean_proprio_dist | mean_div_visual_emb | mean_div_proprio_emb | MPC iters |
|---|---|---|---|---|---|---|---|---|
| umaze | False | 0.700 | 3.220 | 0.490 | 1.191 | 40.271 | 23.888 | 541 |
| umaze | cos1e-1 | 0.880 | 3.892 | 0.460 | 1.325 | 35.446 | 11.041 | 366 |
| umaze | tttwothirds5e-2 | 0.660 | 3.867 | 0.471 | 1.427 | 35.437 | 9.942 | 549 |
| umaze | cos1e-1_tttwothirds5e-2 | 0.900 | 3.756 | 0.460 | 1.268 | 38.269 | 11.677 | 311 |
| medium | False | 0.620 | 4.132 | 0.301 | 1.550 | 84.811 | 13.898 | 559 |
| medium | cos1e-2 | 0.760 | 4.309 | 0.290 | 1.513 | 73.531 | 10.729 | 491 |
| medium | tttwothirds5e-2 | 0.600 | 4.385 | 0.306 | 1.736 | 45.690 | 9.799 | 542 |
| medium | cos1e-1_tttwothirds5e-2 | 0.700 | 4.814 | 0.290 | 1.699 | 70.136 | 10.850 | 528 |
| pusht | False | 0.720 | 118.124 | 3.164 | 42.422 | 58.478 | 1.857 | 520 |
| pusht | aggcos1e-1 | 0.860 | 70.323 | 2.681 | 24.757 | 62.764 | 0.985 | 413 |
| pusht | aggtwothirds5e-2 | 0.840 | 57.184 | 3.007 | 20.875 | 49.607 | 1.849 | 432 |
| pusht | aggcos1e-1_aggtwothirds5e-2 | 0.920 | 53.814 | 2.707 | 19.216 | 40.063 | 0.855 | 378 |

`MPC iters` = total MPC iterations executed across the 50 episodes (sum of per-episode
success-steps; a useful proxy for planning cost / closed-loop effort).

## Appendix B — CEM-MPC: placeholders (to be filled)

The earlier closed-loop CEM runs were invalid and their results removed: the CEM-MPC sub-planner ran
with an effective **1-model-step horizon** (`conf/plan_mpc_cem.yaml` `horizon: 5` ÷ frameskip 5), the
**open-loop CEM budget** (`num_samples=200, opt_steps=10`), and `objective.mode=last` for mazes —
none matching the paper's MPC protocol. Configs are now fixed (`horizon: 25` → 5 model steps,
`num_samples=300`, `opt_steps=30`, `mode=all` for mazes). Re-run with:

```bash
bash run_mpc.sh umaze  all mpc_cem
bash run_mpc.sh medium all mpc_cem
bash run_mpc.sh pusht  all mpc_cem
```

Results land in `plan_outputs_mpc_cem/<model>_gH25/logs.json` (last `final_eval/...` line). Note the
paper reports CEM **open-loop only** (Table 5, `num_samples=200`, `opt_steps=10`); closed-loop CEM is
not a paper-reported metric.

## Statistical caveat

`n_evals=50` → binomial SE ≈ 0.07 on each success_rate. Differences of ~0.1-0.2 (1.4-2.9 SE) are
suggestive but not ironclad; the cleanest signal is pusht/umaze "both ≥ straighten" under GD-MPC,
consistent with the open-loop findings in `RESULTS.md`.


Linear Probe
Simple MLP on the encoder 
How much info about the state of the agent can we recover?
Fit MLP on train set to predict the state of the agent

Visu
-Latent space curv vs env traj (in temo str paper)

Do these exp first, 

Simulate a dataset with more noisy trajectories 
- See if Preg helps there.