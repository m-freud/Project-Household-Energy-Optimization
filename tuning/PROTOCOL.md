# Hybrid MA Tuning Protocol

This document is a working guideline for tuning the Hybrid MA predictor. It is meant to give the process structure without locking us into decisions too early. We can adjust the order, defaults, and stage boundaries as concrete results come in.

Latest consolidated results:
- [../ma_hypa_tuning_results.md](../ma_hypa_tuning_results.md)

## Goals

1. Find the simplest forecast configuration that clearly beats the current baseline.
2. Keep all scenarios on by default so the result is robust across operating conditions.
3. Tune forecast structure before safety layers.
4. Promote a configuration only after it wins on cost without harming target hit rates.

## Working Defaults

Unless we decide otherwise for a specific test:
- run all scenarios
- use a small tuning household subset first, then re-run the best candidate on all 250 households
- keep the simulation horizon at 96 timesteps
- save results with a unique `run-tag`
- write the exact parameter set into the stage notes before running

Recommended tuning subset:
- `1-48` households for quick tuning
- `1-250` households for final validation

Why `48`:
- it divides evenly across 6 workers
- it reduces the chance of idle workers at the end of a batch
- it stays close to a round-number subset without adding much runtime overhead

## Stage Gates

As a default decision rule, try not to advance to the next stage unless the current stage satisfies both:
- mean total cost improves over the previous stage baseline
- target hit rates stay at or above the baseline, or any drop is explicitly accepted and documented

If two configurations are effectively tied, prefer the simpler one.

## Stage 0: Baseline Short-Only

Purpose: establish a clean anchor with all non-short-window effects turned off.

Parameters:
- `short_window_size = 7`
- `long_window_size = short_window_size`
- `short_weight = 1.0`
- `conf_interval_frct = 0.0`
- `persistence_mode = none`
- `persistence_range = 1`
- `persistence_constant_alpha = 0.0`
- `trend_weight = 0.0`
- `trend_window = 2`
- `trend_range = 1`
- `source_average_beta = 0.0`

Run shape:
- all scenarios
- 1-48 households
- one run-tag, for example `stage0_short_only_sw7`

Suggested exit condition:
- record the baseline total cost, net cost, and all target hit rates
- keep this configuration as the anchor for Stage 1

## Stage 1: Short-Window Search

Purpose: find the best isolated short-window length.

Parameters:
- vary only `short_window_size`
- keep every other parameter exactly as Stage 0

Recommended values:
- `5, 7, 9, 12`
- if the best value is near a boundary, do a second tighter pass around that value

Run shape:
- all scenarios
- 1-48 households
- one run-tag per sweep, for example `stage1_short_window`

Suggested exit condition:
- choose the best short window by mean total cost, with target hit rates as a tie-breaker
- store the winning short window as `short_window_anchor`

## Stage 2: Local Short/Long Refinement

Purpose: check whether a long window adds useful smoothing around the best short window.

Parameters:
- vary `short_window_size` and `long_window_size` together
- keep all other parameters at Stage 0 values

Recommended neighborhood:
- center on `short_window_anchor`
- test a small local set first, for example:
  - short: `anchor-2`, `anchor`, `anchor+2`
  - long: `anchor+24`, `anchor+32`, `anchor+40`
- always enforce `long_window_size >= short_window_size`

Run shape:
- all scenarios
- 1-48 households
- one run-tag per sweep, for example `stage2_local_windows`

Suggested exit condition:
- choose the best local pair as the new structural baseline
- document the selected pair and why it was chosen

## Stage 3: Blend Weight Search

Purpose: tune the short/long blending ratio once the window structure is stable.

Parameters:
- vary `short_weight`
- keep the best window pair from Stage 2 fixed
- keep persistence, trend, source-average, and interval settings off

Recommended values:
- `0.3, 0.5, 0.7, 0.9`

Run shape:
- all scenarios
- 1-48 households
- one run-tag per sweep, for example `stage3_short_weight`

Suggested exit condition:
- keep the best weight only if it improves cost without degrading targets

## Stage 4: Persistence Search

Purpose: add temporal memory after the basic forecast shape is settled.

Parameters:
- vary `persistence_mode`
- then, if needed, vary `persistence_range` and `persistence_constant_alpha`
- keep windows and short weight fixed at the current best values

Recommended order:
1. `none` vs `constant` vs `exponential`
2. if `constant`, tune `persistence_constant_alpha`
3. tune `persistence_range` only after the mode is chosen

Run shape:
- all scenarios
- 1-48 households
- one run-tag per sweep, for example `stage4_persistence`

Suggested exit condition:
- keep persistence only if it gives a measurable gain over the no-persistence baseline

## Stage 5: Secondary Forecast Features

Purpose: add lower-priority forecast refinements only after the core forecast is stable.

Parameters:
- `source_average_beta`
- `trend_weight`
- `trend_window`
- `trend_range`
- `conf_interval_frct`

Recommended order:
1. `source_average_beta`
2. `trend_weight`
3. `trend_window` and `trend_range`
4. `conf_interval_frct`

Run shape:
- all scenarios
- 1-48 households
- one run-tag per sweep, for example `stage5_secondary_features`

Suggested exit condition:
- keep only the additions that improve cost and preserve stability

## Stage 6: Full-Household Validation

Purpose: confirm the best candidate on the full population.

Parameters:
- use the exact best configuration from earlier stages
- do not change any settings except the household set

Run shape:
- all scenarios
- all 250 households
- one run-tag that clearly marks validation, for example `validate_best_candidate`

Exit condition:
- approve the candidate only if it holds up at full scale

## Stage 7: Safety / Controller Hardening

Purpose: tune the last-line protection measures after the forecast is already strong.

Parameters:
- controller buffer measures
- any additional safety margins in the MPC layer
- confidence-related controls that primarily reduce misses rather than improve the forecast itself

Working rule:
- tune these last because they can hide weaknesses in the predictor

Run shape:
- all scenarios
- the current validation household set
- one run-tag per safety pass

Suggested exit condition:
- adopt the smallest safety setting that preserves target hit rates

## Documentation Rule

After each concrete test or stage, record:
- stage name
- exact parameter values
- household count
- scenario set
- run-tag
- winning metric
- why the winner was selected

Keep the notes short and factual so the next stage can reference them directly. If we change the process midstream, update this file after the change is accepted.

## Naming Rule

Use run-tags that encode the stage and the tuned variable, for example:
- `stage0_short_only_sw7`
- `stage1_short_window`
- `stage2_local_windows`
- `stage3_short_weight`
- `stage4_persistence`
- `stage6_validate_best_candidate`

## Decision Rule

Prefer the simplest configuration that is clearly better than the baseline on total cost and does not weaken target hit rates.

## Latest Recorded Results

### Stage 1: Short-Window Search

Run metadata:
- run-tag: stage1_short_window
- households: 1-48
- scenarios: all 6
- preset: short_only
- tested short windows: 5, 7, 9, 12

Summary by average net cost (lower is better):
- short_window_size 7: avg_net_cost 7.886490
- short_window_size 9: avg_net_cost 7.888594
- short_window_size 5: avg_net_cost 7.890380
- short_window_size 12: avg_net_cost 7.897867

Target rates:
- bess_target_rate: 1.0 for all tested windows
- ev1_target_rate: 1.0 for all tested windows
- ev2_target_rate: 0.993056 for all tested windows

Interpretation:
- all tested windows are close in performance
- short_window_size 7 is currently the best anchor on avg_net_cost and stays simple
- proceed to local short/long refinement around 7

### Stage 1 Follow-up: Single Check at Short Window 24

Run metadata:
- run-tag: stage1_short_window_24
- households: 1-48
- scenarios: all 6
- preset: short_only
- tested short windows: 24

Result:
- short_window_size 24: avg_net_cost 7.904424, avg_total_cost 8.653572
- target rates: bess 1.0, ev1 1.0, ev2 0.993056

Comparison to current anchor (short_window_size 7):
- avg_net_cost delta: +0.017934 (24 is worse)
- avg_total_cost delta: +0.017934 (24 is worse)
- target rates: unchanged

Decision:
- keep short_window_size 7 as anchor
- do not include 24 in the local refinement neighborhood for Stage 2

---

## Revised Strategy: Long-Window-Only (Post Stage-1 Reanalysis)

Date: 2026-07-15

Finding:
- Extensive sweeps across all hybrid-MA hyperparameters (short window, long window, blend
  weight, persistence mode/range/alpha, trend, source-average) produced only microscopic
  changes in avg_net_cost — differences were irrelevant in practice.
- The cleanest and most robust configuration is a pure long-window MA with all other
  effects turned off.

Adopted final configuration (Punkt 1 decision):
- short_window_size = 0 (= long-only; short window disabled)
- long_window_size = 96
- short_weight = 0.0
- conf_interval_frct = 0.0
- persistence_mode = constant (used for sweep below; none otherwise)
- persistence_range = 0 (disabled)
- persistence_constant_alpha = 0.0
- trend_weight = 0.0
- source_average_beta = 0.0

Dashboard defaults updated to reflect this configuration.

---

### Stage 10: Constant-Persistence Range Sweep

Purpose: confirm that persistence does not help even when the rest is zeroed out.
Run purely "spaßeshalber" — no plan to adopt persistence.

Run metadata:
- run-tag: stage10_const_persistence_sweep_pr{N}  (one run per range value)
- households: 1-48 (default)
- scenarios: all 6
- preset: manual
- fixed parameters: sw=1, lw=96, weight=0.0, alpha=0.0, trend=0.0, beta=0.0
- varied: persistence_range ∈ {2, 4, 8, 16, 32, 64, 96}

Results (sorted by avg_net_cost, lower is better):

| persistence_range | pairs | avg_net_cost |
|:-----------------:|------:|:------------:|
|  2 | 288 | 7.8718 |
|  4 | 288 | 7.8724 |
|  8 | 288 | 7.8827 |
| 16 | 288 | 7.8953 |
| 64 | 288 | 7.9024 |
| 96 | 288 | 7.9026 |
| 32 | 288 | 7.9042 |

Interpretation:
- Clear monotonic trend: shorter persistence range → lower cost.
- pr=2 is best but only 0.0006 cheaper than pr=4 — practically identical.
- From pr=8 onwards the cost rises meaningfully (+0.011 vs. pr=2).
- At pr=32+ the predictor is so persistence-dominated that it barely reacts to the MA.
- Persistence hurts rather than helps at every tested range.

Decision:
- Persistence is not adopted.
- Final configuration remains: long-window-only (lw=96, all other params = 0 / none).
- Stage 10 is closed.
