from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from training.evaluation.rollout_errors import _step_error, get_rollout_errors


class ConstantZeroPredictor(BasePredictor):
    def predict(self, household, horizon: int):
        return {"base_load": [0.0] * horizon}

    def predict_base_load(self, household, horizon: int):
        return {"base_load": [0.0] * horizon}

    def predict_pv_gen(self, household, horizon: int):
        return {"pv_gen": [0.0] * horizon}

    def predict_ev_status(self, household, horizon: int, ev_key: str | None = None):
        ev_key = ev_key or "ev1"
        return {
            f"{ev_key}_at_home": [1] * horizon,
            f"{ev_key}_at_charging_station": [0] * horizon,
        }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def manual_rollout_expected(day_profile, target):
    horizon_buckets = {}
    timestep_buckets = {}
    pooled = []

    for start in range(len(day_profile)):
        for horizon in range(1, len(day_profile) - start + 1):
            actual = day_profile[start:start + horizon]
            if target in {"base_load", "pv_gen"}:
                pred = [0.0] * horizon
            else:
                pred = [0] * horizon
            err = _step_error(pred, actual, target)
            pooled.append(err)
            horizon_buckets.setdefault(horizon, []).append(err)
            timestep_buckets.setdefault(start, []).append(err)

    return {
        "horizon_buckets": {k: v for k, v in horizon_buckets.items() if v},
        "timestep_buckets": {k: v for k, v in timestep_buckets.items() if v},
        "horizon_pooled_avg": mean(pooled),
        "horizon_bucket_avg": mean([mean(v) for v in horizon_buckets.values()]),
        "timestep_pooled_avg": mean([x for values in timestep_buckets.values() for x in values]),
        "timestep_bucket_avg": mean([mean(v) for v in timestep_buckets.values()]),
    }


def check_target(day_profile, target):
    predictor = ConstantZeroPredictor()
    result = get_rollout_errors(predictor, day_profile, target=target)
    expected = manual_rollout_expected(day_profile, target)

    assert result["horizon_buckets"] == expected["horizon_buckets"], (target, result["horizon_buckets"], expected["horizon_buckets"])
    assert result["timestep_buckets"] == expected["timestep_buckets"], (target, result["timestep_buckets"], expected["timestep_buckets"])
    assert abs(result["horizon_pooled_avg"] - expected["horizon_pooled_avg"]) < 1e-12, (target, result["horizon_pooled_avg"], expected["horizon_pooled_avg"])
    assert abs(result["horizon_bucket_avg"] - expected["horizon_bucket_avg"]) < 1e-12, (target, result["horizon_bucket_avg"], expected["horizon_bucket_avg"])
    assert abs(result["timestep_pooled_avg"] - expected["timestep_pooled_avg"]) < 1e-12, (target, result["timestep_pooled_avg"], expected["timestep_pooled_avg"])
    assert abs(result["timestep_bucket_avg"] - expected["timestep_bucket_avg"]) < 1e-12, (target, result["timestep_bucket_avg"], expected["timestep_bucket_avg"])

    print(f"{target}: OK")


if __name__ == "__main__":
    base_day_profile = [1.0, 2.0, 3.0, 4.0]
    ev_day_profile = [0.0, 1.0, 2.0, 1.0, 0.0]

    for target in ["base_load", "pv_gen"]:
        check_target(base_day_profile, target)
    for target in ["ev1_status", "ev2_status"]:
        check_target(ev_day_profile, target)

    print("All rollout error checks passed for the constant-zero benchmark.")
