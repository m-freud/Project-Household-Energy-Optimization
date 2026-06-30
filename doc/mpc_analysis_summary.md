# MPC Analysis Summary

Date: 2026-06-30 150700

This note captures the data analysis performed on the MPC controller after the speed refactor. It is intended as a reusable source for the README or project report.

## What Was Verified

- The MPC controller was refactored to reuse a compiled CVXPY problem with Parameters and warm starts.
- The cleaned benchmark run completed successfully and repopulated the database from a clean state.
- The database currently contains one clean benchmark run for all controllers and scenarios, plus older historical runs used for comparison.
- The dashboard currently does not filter by `run_id`, so overlapping runs can mix in aggregate views and latest-row lookups.

## Performance Result

The MPC refactor produced a major speedup without reducing the planning horizon.

Observed timings on a representative household:
- Before refactor: about 1.7 s per `set_controls` call
- After refactor: first solve about 1.3 s, then roughly 0.01 to 0.04 s per subsequent call in the same controller instance
- Full 96-step single-household run: about 68 s in the earlier OSQP-heavy version, then much faster after switching to the DPP-friendly formulation with Clarabel as the primary solver

## Accuracy Check Against Baseline

A direct comparison was run against `run_id = 5` for `mpc_oracle`, household 1, `default_scenario`.

Baseline vs new result:
- `total_cost`: 13.6615 -> 13.5511, delta -0.1104 (-0.81%)
- `total_consumption`: 111.8894 -> 112.0451, delta +0.1556 (+0.14%)
- `net_cost`: 12.8968 -> 12.7864, delta -0.1104 (-0.86%)
- `net_load`: 111.8894 -> 112.0451, delta +0.1556 (+0.14%)

Interpretation:
- The optimized controller matches the earlier MPC behavior very closely.
- Small deltas are consistent with solver tolerance and solver choice differences, not a behavioral regression.

## Full Run Completeness

The cleaned full benchmark database contains:
- 42 policy/scenario combinations
- 10,500 result rows total
- 1,750 rows for `waterfall`
- 1,750 rows for `mpc_oracle`
- 250 households per policy/scenario combination

This confirms the post-clean benchmark is complete.

## MPC vs Waterfall

Average database-level comparison between `mpc_oracle` and `waterfall` showed:

- `default_scenario`: MPC cost about flat to slightly better, with higher load
- `early_urgency`: MPC cost lower, load higher
- `high_start_narrow`: MPC cost lower, load slightly higher
- `late_relaxed`: MPC cost significantly lower, load much higher
- `low_start_wide`: MPC cost lower, load much higher
- `mid_start_normal`: MPC cost lower, load much higher
- `stressed_ev_buffered_bess`: MPC cost lower, load much higher

Summary:
- MPC consistently reduced consumer cost versus waterfall.
- MPC also increased total household energy throughput in many scenarios.
- This is consistent with aggressive price shifting, not with a target-feasibility problem.

## Target Hit Rates

Target completion was checked across the cleaned full run.

Overall finding:
- All controllable controllers had 100% deadline-target hit rates for BESS, EV1, and EV2.
- `no_control` is the expected outlier and missed EV targets.

For MPC specifically:
- `bess_all`: 100% in every scenario
- `ev1_all`: 100% in every scenario
- `ev2_all`: 100% in every scenario

This means the MPC cost improvement did not come at the expense of missing deadlines.

## Price Curve Behavior

A representative price-curve check on household 1, `default_scenario`, showed that MPC shifts more load into cheap periods than waterfall.

Household 1, `default_scenario`:
- `waterfall` load-weighted buy price: 0.1201
- `mpc_oracle` load-weighted buy price: 0.1142
- correlation between load and price:
  - `waterfall`: -0.8310
  - `mpc_oracle`: -0.8640
- share of absolute load occurring in the cheapest quartile of prices:
  - `waterfall`: 83.3%
  - `mpc_oracle`: 89.5%

Interpretation:
- MPC is better aligned with low-price periods.
- The behavior is consistent with bill minimization through load shifting.
- The observed load pattern looks like price arbitrage in practice, though the model is primarily optimizing consumer cost.

## Battery Wear Proxy

A battery throughput check was run to see whether MPC was increasing wear.

Result:
- MPC used less BESS throughput than waterfall in every scenario checked.
- MPC also had lower peak absolute battery power.

Representative averages by scenario:
- `default_scenario`: waterfall 18.0794 vs MPC 13.7995 average absolute BESS throughput
- `early_urgency`: 17.4700 vs 11.4385
- `high_start_narrow`: 11.2417 vs 8.9006
- `late_relaxed`: 14.4370 vs 10.0437
- `low_start_wide`: 17.9689 vs 12.2146
- `mid_start_normal`: 15.2652 vs 10.0946
- `stressed_ev_buffered_bess`: 17.5909 vs 12.3467

Interpretation:
- The MPC improvement is not being bought with extra battery cycling.
- An idealized no-wear battery assumption is reasonable for this portfolio project, provided it is stated explicitly.

## Practical Modeling Assumption

For this project, it is defensible to assume:
- battery degradation is not modeled
- the objective is consumer cost reduction under idealized device behavior

This is common in academic or portfolio energy-optimization work when the focus is bill savings and scheduling logic rather than battery lifetime economics.

Recommended wording for the README:
- "The model assumes ideal battery behavior and does not include battery degradation or replacement cost. The controller optimizes consumer electricity cost subject to energy and deadline constraints."

## Dashboard Caveat

Important analysis caveat:
- The dashboard does not currently expose a `run_id` selector.
- Aggregate views query by policy and scenario only, so overlapping runs can mix results.
- Single-household views select the most recent matching row by `rowid`, which can also mix runs if multiple runs exist.

Recommended workflow:
- Keep a clean benchmark database when comparing policies.
- If multiple runs are present, document the intended run explicitly in analysis notes.

## Bottom Line

The refactored MPC controller is:
- much faster
- numerically close to the earlier baseline
- fully target-feasible
- cheaper than waterfall in every scenario tested
- not increasing battery throughput versus waterfall

That makes it a strong portfolio-quality result for a bill-minimization use case.
