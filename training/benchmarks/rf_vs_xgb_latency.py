"""Runtime comparison: RandomForest vs XGBoost, comparable hyperparams (100 trees).

Measures:
  1. Batch prediction throughput (fair, apples-to-apples).
  2. Single-row prediction latency (the actual recursive-MPC access pattern).
Then extrapolates to a full household day-run under recursive prediction.
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

RNG = np.random.default_rng(42)

N_FEATURES = 18          # matches training/xgboost base_load feature count
N_TRAIN = 20_000
N_BATCH_TEST = 2_000
N_SINGLE_CALLS = 5_000   # repeated single-row predict() calls

# comparable hyperparams: same tree count / depth, single-threaded
# (MPC replanning is single-threaded per household, so n_jobs=-1 would
#  just hide RF's per-call overhead behind thread-pool dispatch)
RF_PARAMS = dict(n_estimators=100, max_depth=6, n_jobs=1, random_state=42)
XGB_PARAMS = dict(n_estimators=100, max_depth=6, n_jobs=1, random_state=42, tree_method="hist")


def make_data(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    X = RNG.normal(size=(n_rows, N_FEATURES)).astype(np.float32)
    y = X[:, 0] * 2.0 - X[:, 1] + RNG.normal(scale=0.1, size=n_rows)
    return X, y


def bench_batch(model, X: np.ndarray, reps: int = 20) -> float:
    # warmup
    model.predict(X[:1])
    start = time.perf_counter()
    for _ in range(reps):
        model.predict(X)
    elapsed = time.perf_counter() - start
    return elapsed / reps / len(X)  # seconds per row (batched)


def bench_single_row(model, X: np.ndarray, n_calls: int) -> float:
    row = X[:1]
    model.predict(row)  # warmup
    start = time.perf_counter()
    for i in range(n_calls):
        model.predict(X[i % len(X): i % len(X) + 1])
    elapsed = time.perf_counter() - start
    return elapsed / n_calls  # seconds per single-row call


def main() -> None:
    X_train, y_train = make_data(N_TRAIN)
    X_test, _ = make_data(N_BATCH_TEST)

    models = {
        "RandomForest (100 trees)": RandomForestRegressor(**RF_PARAMS),
        "XGBoost (100 trees)": XGBRegressor(**XGB_PARAMS),
    }

    results = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        per_row_batch = bench_batch(model, X_test)
        per_row_single = bench_single_row(model, X_test, N_SINGLE_CALLS)

        results[name] = dict(fit_time=fit_time, batch=per_row_batch, single=per_row_single)

        print(f"\n{name}")
        print(f"  fit time:                {fit_time * 1000:8.2f} ms")
        print(f"  batch predict (per row):  {per_row_batch * 1e6:8.2f} \u00b5s")
        print(f"  single-row predict:       {per_row_single * 1e6:8.2f} \u00b5s")

    rf, xgb = results["RandomForest (100 trees)"], results["XGBoost (100 trees)"]
    speedup = rf["single"] / xgb["single"]
    print(f"\n--- Single-row speedup: XGBoost is {speedup:.1f}x faster than RandomForest ---")

    # --- Extrapolation to a full day-run, two predictor architectures ---
    horizon = 96
    replanning_steps = 96
    n_targets = 4  # base_load, pv_gen, ev1_status, ev2_status

    # recursive predictor: one model called (horizon - 1) times per target
    recursive_calls_per_household = replanning_steps * (horizon - 1) * n_targets

    # composite predictor: a small model_bank of direct-horizon models per
    # target, no recursion (rest of the horizon is interpolated, not predicted)
    bank_size = 8
    composite_calls_per_household = replanning_steps * bank_size * n_targets

    household_counts = [1, 50, 200]

    for label, calls_per_household in [
        ("recursive (1 model, 95 recursive steps/target)", recursive_calls_per_household),
        (f"composite ({bank_size} direct-horizon models/target)", composite_calls_per_household),
    ]:
        print(f"\n=== {label}: {calls_per_household:,} predict() calls / household / day ===")
        for name, r in results.items():
            per_household_s = r["single"] * calls_per_household
            row = "  ".join(
                f"{n}hh: {per_household_s * n:8.1f}s" for n in household_counts
            )
            print(f"  {name:<28s}: {row}")


if __name__ == "__main__":
    main()
