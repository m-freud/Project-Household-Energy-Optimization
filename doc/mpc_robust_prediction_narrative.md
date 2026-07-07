# Robust Prediction-Based MPC Narrative

## 1. Context and Problem Statement

This project addresses household energy optimization with multiple flexible assets (BESS and EV charging) under uncertainty in demand, generation, and availability. The operational objective is not only to reduce energy cost, but to do so while reliably meeting deadline-based state-of-charge targets.

A key practical requirement in this work is strict target compliance: when comparing controllers, missing targets is not acceptable, because any apparent cost advantage from under-serving targets is not a valid operational win.

## 2. Core Approach

We implemented a prediction-based Model Predictive Control (MPC) architecture with two priorities:

1. Feasibility and target satisfaction first.
2. Cost optimization second.

The current controller setup uses a receding-horizon optimization with explicit device constraints and deadline target constraints. At each step, predictions are generated and injected into a compiled MPC problem. The controller then computes controls for BESS and EV charging under a full-day planning horizon.

## 3. What We Learned from Benchmarking

### 3.1 Oracle baseline behavior

The oracle MPC baseline in our stored benchmark data achieves full target compliance across all rows currently evaluated. This makes it a valid reference point for target-safe operation.

### 3.2 Moving Average 2 (MA2) behavior

MA2 can produce lower costs in a subset of runs, but those apparent wins are explained by missed targets.

In the analyzed MA2-vs-oracle wins:

- Every MA2 cost win coincided with at least one missed target.
- There were no MA2 wins that preserved full target compliance.

This is a critical result: lower cost without meeting targets is not an acceptable controller improvement under our evaluation criteria.

### 3.3 Implication for fair controller comparison

Controller comparisons must be conditioned on feasibility and target compliance. A controller that is cheaper but misses deadlines should not be ranked above one that is slightly more expensive but fully compliant.

## 4. Why Robustness Measures Are Necessary

Point forecasts (single best estimates) are often insufficient in real operations because forecast errors are unavoidable. If we optimize only against mean forecasts, the controller can become brittle near deadlines and violate constraints under plausible deviations.

To address this, we introduce safety measures at two levels:

### 4.1 Predictor-side safety

Use uncertainty-aware forecasts instead of only point predictions, for example:

- Lower/median/upper forecast bands.
- Conservative profiles for critical signals.

Typical conservative substitutions in robust mode are:

- Demand/load: upper bound.
- PV generation: lower bound.
- Availability indicators: pessimistic assumption.

### 4.2 Controller-side safety

Add explicit robustness buffers in control constraints, for example:

- Energy buffer: require SOC at deadlines above nominal target by a margin.
- Time buffer: pull effective deadlines earlier to absorb late uncertainty.

These mechanisms reduce the chance of last-minute infeasibility and improve deadline reliability.

## 5. Industry-Aligned Framing

The architecture direction is aligned with common industrial control practice:

- Hard constraints for non-negotiable targets.
- Robust handling of forecast uncertainty.
- Feasibility-first or lexicographic optimization priorities.
- Cost optimization only within the compliant feasible set.

This is a defensible and professional control design narrative.

## 6. Final Narrative for This Phase

A strong summary of the current project phase is:

We implemented a robust prediction-based MPC with safety mechanisms at both predictor and controller levels. With strict target compliance as a hard requirement, we can now compare controllers fairly and avoid false cost wins caused by under-serving targets. The current system already outperforms naive baselines while maintaining compliance. Therefore, after robustness and feasibility are established, the dominant remaining lever for further cost improvement is forecast quality.

## 7. Why Machine Learning Is the Natural Next Step

At this stage, additional gains increasingly depend on better prediction quality rather than ad hoc controller rule tweaks.

Machine learning forecasting is a natural next step because it can:

- Capture nonlinear and context-dependent patterns not easily encoded manually.
- Learn behavior that partially overlaps with moving-average heuristics.
- Discover additional predictive structure beyond handcrafted averages.

Importantly, ML forecasting should be integrated into the same robust MPC framework (bands/scenarios plus buffers), not used as unconstrained point prediction alone.

## 8. Evaluation Principle Going Forward

To preserve scientific and engineering rigor, future benchmark reporting should follow this order:

1. Report target compliance first (must be 100% for accepted runs).
2. Compare costs only among compliant runs.
3. Quantify uncertainty and robustness settings used.
4. Report trade-offs transparently (cost vs conservatism).

This ensures improvements are operationally meaningful, not artifacts of violated constraints.

## 9. Suggested Next Implementation Step

A practical next milestone is to extend the predictor interface to optionally emit forecast bands and add robust-mode switches in MPC:

- point mode: current behavior
- robust mode: conservative band selection
- optional scenario mode: multi-scenario constraints

This keeps backward compatibility while enabling uncertainty-aware optimization and more credible comparisons for ML-based predictors later.
