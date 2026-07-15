# Scenario Grid Design (Current Benchmark Setup)

## Why this redesign

We reduced the scenario space to a focused 3x2 grid so controller comparisons stay interpretable and affordable to run.

Primary benchmark goal:
- Compare controller behavior and ranking across a compact but meaningful set of stress conditions.
- Current target controllers: MPC oracle, MPC MA3, and waterfall.

## Core modeling assumptions

Main scenario axes are:
- Start SoC (low, mid, high)
- SoC target urgency profile (relaxed, stressed)

What is fixed on purpose:
- One shared SoC allowed range across scenarios
- One shared BESS end-of-day target of 0.5

Reasoning:
- In practice, users usually do not set explicit BESS trajectory targets.
- We keep a terminal BESS target to avoid end-of-horizon battery depletion artifacts.
- A neutral 0.5 terminal target keeps the battery at a reasonable level for the next day without over-constraining operations.

## Scenario matrix (3x2)

Relaxed targets:
- relaxed_low_start
- default_scenario (relaxed_mid_start)
- relaxed_high_start

Stressed targets:
- stressed_low_start
- stressed_mid_start
- stressed_high_start

Scenario table:

| Scenario | Urgency | Start level | Start SoC | SoC allowed range | BESS end target | EV targets |
|---|---|---|---:|---|---:|---|
| default_scenario | relaxed | mid | 0.5 | 0.2-0.8 | 0.5 | {32: 0.35, 64: 0.5, 96: 0.8} |
| relaxed_low_start | relaxed | low | 0.2 | 0.2-0.8 | 0.5 | {32: 0.35, 64: 0.5, 96: 0.8} |
| relaxed_high_start | relaxed | high | 0.8 | 0.2-0.8 | 0.5 | {32: 0.35, 64: 0.5, 96: 0.8} |
| stressed_low_start | stressed | low | 0.2 | 0.2-0.8 | 0.5 | {32: 0.6, 64: 0.75, 96: 0.9} |
| stressed_mid_start | stressed | mid | 0.5 | 0.2-0.8 | 0.5 | {32: 0.6, 64: 0.75, 96: 0.9} |
| stressed_high_start | stressed | high | 0.8 | 0.2-0.8 | 0.5 | {32: 0.6, 64: 0.75, 96: 0.9} |

Notes:
- default_scenario currently maps to the mid-start relaxed profile.
## Future extension: SoC-range sensitivity sweep

We intentionally removed SoC-range as a primary benchmark axis for now.

Planned follow-up study:
- Run a dedicated sweep over allowed SoC ranges (for example tight, medium, wide).
- Evaluate the impact on controller ranking, cost, target-hit behavior, and robustness.

This keeps the main benchmark clean while still allowing explicit analysis of flexibility constraints when needed.
