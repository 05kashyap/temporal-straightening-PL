# Closed-Loop MPC Results (temporal-straightening + two-thirds, faithful MPC)

Compiled from this machine (single NVIDIA RTX 4070 12 GB). **Closed-loop** MPC: each eval episode is
executed in the real sim and re-planned until success or the iteration cap, unlike the open-loop
(`max_iter=1`) results in `RESULTS.md`. Per env, the four variants are **baseline**, **straighten**,
**twothirds**, and **twothirds + straighten** (the "both" model).

## Faithful MPC settings (identical for every row)

- `n_evals=50`, `goal_H=25`, `chunk_size=1`, `seed=100`
- MPC loop: `n_taken_actions=5`, `max_iter=20` (safety cap; exits early on success)
- **GD-MPC**: sub-planner GD, `opt_steps=100`, Adam `lr=0.1`, zero init
- **CEM-MPC** (placeholder below): `num_samples=200`, `opt_steps=10`, `sample_chunk_size=50`
- Objectives: umaze/medium `objective.alpha=0 mode=last`; pusht GD-MPC `alpha=1 mode=staged`
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
| baseline (`False`) | **0.700** | 3.220 | 0.490 | 1.191 | 0.540 | 3.187 |
| straighten (`cos1e-1`) | **0.880** | 3.892 | 0.460 | 1.325 | **1.000** | 3.317 |
| twothirds (`tttwothirds5e-2`) | **0.660** | 3.867 | 0.471 | 1.427 | 0.640 | 3.634 |
| **twothirds+straighten** | **0.900** | 3.756 | 0.460 | 1.268 | 0.900 | 2.874 |

- GD headline: **both (0.90) > straighten (0.88) > baseline (0.70) > twothirds (0.66)**; both is best
  and straighten is a close second. Note state-dist is lowest for baseline (3.22) — here success
  tracks the classifier/proximity of the reached region more than raw final distance.
- CEM headline: **straighten (1.00) > both (0.90) > twothirds (0.64) > baseline (0.54)** — straighten
  alone is perfect under CEM; both ties GD's best (0.90) and reaches the lowest state-dist (2.87).
- Planning cost (total MPC iters / 50 eps): GD 541/366/549/311; CEM 663/199/517/272 (baseline /
  straighten / twothirds / both) — fewer iterations means earlier success on average.

## PointMaze-medium

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline (`False`) | **0.620** | 4.132 | 0.301 | 1.550 | 0.260 | 3.921 |
| straighten (`cos1e-2`) | **0.760** | 4.309 | 0.290 | 1.513 | 0.380 | 4.032 |
| twothirds (`tttwothirds5e-2`) | **0.600** | 4.385 | 0.306 | 1.736 | 0.320 | 3.920 |
| **twothirds+straighten** | **0.700** | 4.814 | 0.290 | 1.699 | **0.400** | 3.993 |

- GD headline: **straighten (0.76) > both (0.70) > baseline (0.62) > twothirds (0.60)** — straightening
  alone is best here; adding two-thirds does not help beyond it.
- CEM headline: **both (0.40) > straighten (0.38) > twothirds (0.32) > baseline (0.26)** — same winner
  ordering as GD but all CEM scores are well below GD; closed-loop feedback helps GD far more than CEM
  on this env.
- Planning cost (total MPC iters / 50 eps): GD 559/491/542/528; CEM 766/702/731/647 (baseline /
  straighten / twothirds / both).
- **Caveats (config inconsistencies in the medium checkpoint set):** baseline/straighten were trained
  with `lr=1e-06`, twothirds/both with `lr=1e-05`; and the straighten model uses `cos1e-2`
  (λ=0.01) while the "both" model's straighten term is `cos1e-1` (λ=0.1) — so cross-variant
  comparisons are not perfectly apples-to-apples.

## PushT

| Variant | GD success_rate | GD state-dist | GD visual-dist | GD proprio-dist | CEM success_rate | CEM state-dist |
|---|---|---|---|---|---|---|
| baseline (`False`) | **0.720** | 118.1 | 3.164 | 42.42 | 0.460 | 85.3 |
| straighten (`aggcos1e-1`) | **0.860** | 70.3 | 2.681 | 24.76 | 0.460 | 74.6 |
| twothirds (`aggtwothirds5e-2`) | **0.840** | 57.2 | 3.007 | 20.88 | 0.420 | 78.0 |
| **twothirds+straighten** | **0.920** | 53.8 | 2.707 | 19.22 | 0.400 | 70.6 |

- GD headline: **both (0.92) > straighten (0.86) > twothirds (0.84) > baseline (0.72)**; both is best
  and also reaches the lowest final state-dist (53.8) and proprio-dist (19.2).
- CEM headline: **baseline = straighten (0.46) > twothirds (0.42) > both (0.40)** — ordering nearly
  reversed vs GD, and all well below GD; closed-loop re-planning (which CEM does less effectively with
  its batch of 200 candidates) is the differentiator on PushT.
- Planning cost (total MPC iters / 50 eps): GD 520/413/432/378; CEM 577/586/620/632 (baseline /
  straighten / twothirds / both).

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

### Appendix A2 — full `final_eval` metrics (CEM-MPC)

| Env | Model | success_rate | mean_state_dist | mean_visual_dist | mean_proprio_dist | mean_div_visual_emb | mean_div_proprio_emb | MPC iters |
|---|---|---|---|---|---|---|---|---|
| umaze | False | 0.540 | 3.187 | 0.490 | 1.228 | 41.540 | 21.821 | 663 |
| umaze | cos1e-1 | 1.000 | 3.317 | 0.476 | 1.102 | 39.162 | 12.957 | 199 |
| umaze | tttwothirds5e-2 | 0.640 | 3.634 | 0.489 | 1.357 | 38.717 | 14.647 | 517 |
| umaze | cos1e-1_tttwothirds5e-2 | 0.900 | 2.874 | 0.483 | 1.030 | 37.402 | 13.693 | 272 |
| medium | False | 0.260 | 3.921 | 0.302 | 1.553 | 72.608 | 8.707 | 766 |
| medium | cos1e-2 | 0.380 | 4.032 | 0.314 | 1.627 | 83.405 | 9.355 | 702 |
| medium | tttwothirds5e-2 | 0.320 | 3.920 | 0.311 | 1.603 | 44.753 | 7.075 | 731 |
| medium | cos1e-1_tttwothirds5e-2 | 0.400 | 3.993 | 0.317 | 1.613 | 69.278 | 11.208 | 647 |
| pusht | False | 0.460 | 85.294 | 4.652 | 26.389 | 108.543 | 1.479 | 577 |
| pusht | aggcos1e-1 | 0.460 | 74.589 | 4.577 | 22.656 | 87.837 | 1.793 | 586 |
| pusht | aggtwothirds5e-2 | 0.420 | 78.019 | 4.682 | 23.625 | 88.566 | 2.414 | 620 |
| pusht | aggcos1e-1_aggtwothirds5e-2 | 0.400 | 70.565 | 4.738 | 18.324 | 64.902 | 1.662 | 632 |

## Appendix B — CEM-MPC: how these results were produced

All CEM-MPC rows above were run with the full faithful budget (`num_samples=200`, `opt_steps=10`,
`sample_chunk_size=50`, `max_iter=20`, `n_evals=50`, `chunk_size=1`):

```bash
bash run_mpc.sh umaze  all mpc_cem
bash run_mpc.sh medium all mpc_cem
bash run_mpc.sh pusht  all mpc_cem
```

Each run's raw numbers live in `plan_outputs_mpc_cem/<model>_gH25/logs.json` — the last
(`final_eval/...`) line of each file is the source of the table entries above.

## Cross-planner summary

- **GD-MPC is the stronger closed-loop planner here**: it beats CEM on medium (0.60-0.76 vs 0.26-0.40)
  and pusht (0.72-0.92 vs 0.40-0.46), and ties/beats it on umaze (both 0.90; straighten 0.88 GD vs 1.00
  CEM). Closed-loop re-planning from real env feedback rewards GD's single-trajectory refinement more
  than CEM's wide-but-once-per-iteration candidate batch.
- **Best model by planner:** umaze — both (GD) / straighten (CEM, 1.00); medium — straighten (GD) /
  both (CEM); pusht — both (GD) / baseline or straighten (CEM).
- **The two-thirds+straighten ("both") model is the most consistent top performer under GD-MPC** (best
  on umaze and pusht, second on medium), matching the open-loop conclusion in `RESULTS.md`. Under CEM
  the story is weaker (umaze straighten is perfect, medium both wins, pusht both is worst).

## Statistical caveat

`n_evals=50` → binomial SE ≈ 0.07 on each success_rate. Differences of ~0.1-0.2 (1.4-2.9 SE) are
suggestive but not ironclad; the cleanest signals are pusht/umaze "both ≥ straighten" under GD, and
GD ≥ CEM overall on medium/pusht.

