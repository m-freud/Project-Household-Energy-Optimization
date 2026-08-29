# Evaluation Strategy

## Decision

We evaluate controllers on all 250 households.

The earlier 20-household runtime subset was useful as a strict no-equivalence test set, but it is not required for the final controller benchmark. Reusing the full 250-household population is acceptable because repeated household variants do not create a fundamental evaluation problem by themselves.

## Predictor Leakage Constraint

Using all 250 households for controller evaluation does not remove the leakage constraint for learned predictors.

For each evaluated household, each predictor must be loaded from a model that was trained on outside profiles for the target being predicted.

This constraint is enforced individually per prediction target:
- base load
- PV generation
- EV1 status
- EV2 status

The key rule is:
- a household may appear in the 250-household controller test set
- but its prediction for a given target must come from a model trained without that household's equivalent source-profile group for that same target

In other words, leakage control is target-specific, not just household-specific.

## Why We Keep All 250 Households

Using all 250 households gives a population-weighted controller evaluation.

This is not perfectly archetype-balanced, because some base-load families occur more often than others. However, the observed base-load equality groups are broad enough that the full-population benchmark remains useful and representative.

So the final approach is:
- keep the 250-household average as the main empirical population result
- add a base-load-group-weighted average as a robustness check against repeated archetypes dominating the score

## Aggregation Metrics

### 1. Raw 250-household average

This is the standard average over all households.

Interpretation:
- performance on the empirical household population

### 2. Base-load-group-weighted average

For each base-load equality group:
- compute the mean controller metric across households in that group

Then:
- average those group means with equal weight across groups

Interpretation:
- performance across distinct base-load archetypes

If the raw average and the base-load-group-weighted average are similar, then repeated base-load families are not materially distorting the controller comparison.

## Minimal Reporting Table

For each controller, report:

| controller | raw mean | base-load-balanced mean | raw rank | balanced rank |
| --- | ---: | ---: | ---: | ---: |

This is the minimum table needed to check both:
- absolute performance level
- whether controller ranking changes under archetype balancing

## Practical Interpretation

If the two averages are close and controller ranking stays stable, the 250-household evaluation is sufficient.

If they differ noticeably, or if controller ranking changes, then repeated base-load archetypes are influencing the headline result and both views should be reported explicitly.

## Sanity Check Result

The new fold setup was checked against the old strict setup using `run_id = 11` as the baseline and the latest default-scenario runs as the comparison.

Default-scenario mean `total_cost` over the 20 test households:

| controller | run 11 mean | latest mean | delta abs | delta pct |
| --- | ---: | ---: | ---: | ---: |
| mpc_rf | 7.503762 | 7.528608 | +0.024847 | +0.331% |
| mpc_ridge | 9.426591 | 9.450052 | +0.023461 | +0.249% |
| mpc_xgb | 7.557479 | 7.517684 | -0.039795 | -0.527% |

Takeaway:
- the new fold setup gives effectively equivalent default-scenario results
- the old 20-household setup was too strict and too small to keep as the main evaluation path
- the old setup can remain only as historical context in this folder
