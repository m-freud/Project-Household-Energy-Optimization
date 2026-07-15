# HYPA / Hybrid MA Tuning Results

This document records the tuning experiments from `/tuning` in chronological order and closes with the practical decision: the extra model complexity did not buy enough to justify itself, so the simple global moving average remains the right default.

## Context

The tuning work was done on the `1-48` household subset with all 6 scenarios enabled unless stated otherwise. The main objective was to beat the current baseline with a simpler forecast structure only if the gain was real and stable, not microscopic.

## Chronology

### Stage 0: Baseline short-only anchor

- Setup: `short_window_size=7`, `long_window_size=7`, `short_weight=1.0`, `persistence=none`, `trend=0`, `source_average_beta=0`
- Scope: 48 households, 6 scenarios
- Outcome: baseline anchor for the later search
- Takeaway: this was just the starting point, not the final answer

### Stage 1: Short-window search

- Tested short windows: `5`, `7`, `9`, `12`
- Best result: `short_window_size=7` with `avg_net_cost=7.886490`
- Worst among the tested set: `short_window_size=12` with `avg_net_cost=7.897867`
- Takeaway: `7` was the best isolated short-window length, but the spread was small

### Stage 2: Local short/long refinement

- Tested paired short/long windows around the stage 1 anchor
- Representative winning region: `short_window_size=5`, `long_window_size=48`, `short_weight=0.5`
- Best early pair in this stage stayed near `short=5` and `long=48`
- Takeaway: the long window helped more than the short-only anchor, but the improvement was still modest

### Stage 3: Blend weight search

- Tested short/long blending weights around the stage 2 pair
- Relevant tested weights: `0.3`, `0.5`, `0.7`
- Outcome: the `0.5` region was the best of the early blend tests, but the differences were still small
- Takeaway: blend weight mattered, but only at a tiny scale

### Stage 4: Persistence search

- Tested persistence modes and ranges around the best stage 2/3 structure
- Best recorded line in the summary set: `persistence=linear`, `range=1`, `short_window_size=5`, `long_window_size=48`, `short_weight=0.5`
- Example result: `avg_net_cost=7.855176` for the linear `r1` run
- Takeaway: persistence helped a little, but not enough to change the overall picture

### Stage 5: Secondary forecast features

- Tested trend weights and related settings after persistence
- Tested values included `trend_weight=0.1`, `0.2`, `0.4`, `0.8`, `1.0`
- Best recorded trend run among the checked set was essentially tied with the no-trend baseline
- Takeaway: trend was not worth carrying forward as a meaningful improvement

### Stage 6: Source-average split tests

- Tested base-load and PV source-average handling separately
- `base_load` beta test: worse than the no-beta baseline
- `pv_gen` beta test: better than `base_load`, but still only a small gain
- Takeaway: the split showed that `pv_gen` was the only potentially useful side, but the gains were too small to care about

### Stage 7: Short-window-free sanity check

- Tested a very short structural variant: `short_window_size=3`, `long_window_size=3`, `short_weight=1.0`, `pv_beta=0.5`, `persistence=linear r1`
- Outcome: worse than the better stage 6 / stage 8 candidates
- Takeaway: removing the useful long-window structure did not help

### Stage 8: Long-window grid with fixed short window

- Tested a real grid over `long_window_size` and `short_weight`
- Fixed: `short_window_size=5`, `persistence=linear`, `trend=0`, `source_average_beta=0`
- Winning point from the grid: `short_weight=0.2`, `long_window_size=96`
- Best recorded result: `avg_net_cost=7.841434774528772`
- Takeaway: this was the best zero-beta structural result, but the gain over nearby settings was microscopic

### Stage 9: Short weight zero test

- Tested the fully short-free blend at the strongest long window: `short_weight=0.0`, `long_window_size=96`
- Result: `avg_net_cost=7.841717969031795`
- Comparison to stage 8 best: worse by about `0.000283` in `avg_net_cost`
- Takeaway: even this explicit simplification did not beat the best weighted version, but the difference was tiny enough to be operationally irrelevant

## Overall conclusion

The tuning campaign confirmed a simple pattern: the more complicated Hybrid MA variants can move the needle, but only by microscopic amounts. The best improvements were on the order of fractions of a thousandth in cost, and none of the extra knobs produced a change that is large enough to justify the added complexity.

The practical decision is therefore to stop chasing marginal gains and keep the simple global moving average as the default forecast. That choice is easier to reason about, easier to maintain, and effectively optimal for the observed gains in this tuning set.

## Final recommendation

- Keep the global moving average as the baseline model
- Do not invest more effort into beta, trend, or persistence tuning unless a materially larger dataset changes the picture
- Treat the Hybrid MA variants as exploratory only, not as the production default
