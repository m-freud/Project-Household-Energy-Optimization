# Tuning

This folder is the working area for Hybrid MA tuning experiments.

Use [PROTOCOL.md](PROTOCOL.md) as the stricter runbook for tuning runs.

## Location

Keep it at the repository root so it stays separate from code, tests, and final reports:
- `src/` for implementation
- `tests/` for runnable sweep scripts
- `reports/` for CSV outputs
- `doc/` for permanent documentation
- `tuning/` for the live tuning protocol, notes, and stage-by-stage experiment plans

## Proposed protocol

### Stage 0: Baseline
- Run one short-window-only configuration.
- Turn off persistence, trend, source-average blending, and interval effects.
- Use the same household set and all scenarios for the whole stage.

### Stage 1: Short-window search
- Sweep only `short_window_size` around the baseline.
- Keep every other parameter fixed.
- Pick the best short window as the anchor.

### Stage 2: Local short/long refinement
- Tune `short_window_size` and `long_window_size` together around the anchor.
- Use small neighborhoods first.
- Do not expand to a wide grid until local behavior is understood.

### Stage 3: Add one model knob at a time
- Tune `short_weight`.
- Then persistence settings.
- Then source-average blending.
- Then trend settings.
- Keep the current best config as the baseline between stages.

### Stage 4: Safety / controller hardening
- Tune confidence intervals.
- Tune controller buffer measures last.
- Treat these as robustness controls rather than forecast quality improvements.

## Run discipline

- Keep all scenarios on by default.
- Use a smaller household subset for tuning runs.
- Re-run the best candidate on the full household set before promoting it.
- Save every stage result with a unique `run-tag`.
- Record the exact parameter set, household count, scenario set, and winning metric.

## Decision rule

Use the simplest configuration that is clearly better than the current baseline on total cost, while preserving target hit rates.
