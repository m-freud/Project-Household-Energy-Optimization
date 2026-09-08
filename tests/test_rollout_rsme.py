import math

import pytest

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from training.evaluation.rollout_errors import get_rollout_errors


class ConstantZeroPredictor(BasePredictor):
    def predict(self, household, horizon: int):
        return {"base_load": [0.0] * horizon}

    def predict_ev_status(self, household, horizon: int, ev_key: str | None = None):
        ev_key = ev_key or "ev1"
        return {
            f"{ev_key}_at_home": [1] * horizon,
            f"{ev_key}_at_charging_station": [0] * horizon,
        }

    def predict_base_load(self, household, horizon: int):
        return {"base_load": [0.0] * horizon}

    def predict_pv_gen(self, household, horizon: int):
        return {"pv_gen": [0.0] * horizon}


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _manual_expected_for_constant_zero(day_profile, target):
    horizon_buckets = {}
    timestep_buckets = {}
    pooled_errors = []

    for start in range(len(day_profile)):
        for horizon in range(1, len(day_profile) - start + 1):
            actual_window = day_profile[start:start + horizon]
            if target in {"base_load", "pv_gen"}:
                prediction = [0.0] * horizon
                error = math.sqrt(
                    sum((p - a) ** 2 for p, a in zip(prediction, actual_window)) / len(actual_window)
                )
            else:
                prediction = [0] * horizon
                error = sum(int(p != a) for p, a in zip(prediction, actual_window)) / len(actual_window)

            pooled_errors.append(error)
            horizon_buckets.setdefault(horizon, []).append(error)
            timestep_buckets.setdefault(start, []).append(error)

    return {
        "horizon_buckets": {k: v for k, v in horizon_buckets.items() if v},
        "timestep_buckets": {k: v for k, v in timestep_buckets.items() if v},
        "horizon_pooled_avg": _mean(pooled_errors),
        "horizon_bucket_avg": _mean([_mean(v) for v in horizon_buckets.values()]),
        "timestep_pooled_avg": _mean([err for vals in timestep_buckets.values() for err in vals]),
        "timestep_bucket_avg": _mean([_mean(v) for v in timestep_buckets.values()]),
    }


@pytest.mark.parametrize(
    "target,day_profile",
    [
        ("base_load", [1.0, 2.0, 3.0, 4.0]),
        ("pv_gen", [1.0, 2.0, 3.0, 4.0]),
        ("ev1_status", [0.0, 1.0, 2.0, 1.0, 0.0]),
        ("ev2_status", [2.0, 1.0, 0.0, 1.0, 2.0]),
    ],
)
def test_get_rollout_errors_matches_manual_constant_zero_expected(target, day_profile):
    result = get_rollout_errors(ConstantZeroPredictor(), day_profile, target=target)
    expected = _manual_expected_for_constant_zero(day_profile, target)

    assert result["horizon_buckets"] == expected["horizon_buckets"]
    assert result["timestep_buckets"] == expected["timestep_buckets"]
    assert result["horizon_pooled_avg"] == pytest.approx(expected["horizon_pooled_avg"])
    assert result["horizon_bucket_avg"] == pytest.approx(expected["horizon_bucket_avg"])
    assert result["timestep_pooled_avg"] == pytest.approx(expected["timestep_pooled_avg"])
    assert result["timestep_bucket_avg"] == pytest.approx(expected["timestep_bucket_avg"])


@pytest.mark.parametrize(
    "target,day_profile",
    [
        ("base_load", []),
        ("pv_gen", []),
        ("ev1_status", []),
    ],
)
def test_get_rollout_errors_handles_empty_day_profile(target, day_profile):
    result = get_rollout_errors(ConstantZeroPredictor(), day_profile, target=target)
    assert result == {
        "horizon_buckets": {},
        "timestep_buckets": {},
        "horizon_pooled_avg": 0.0,
        "horizon_bucket_avg": 0.0,
        "timestep_pooled_avg": 0.0,
        "timestep_bucket_avg": 0.0,
    }


def test_get_rollout_error_alias_matches_get_rollout_errors():
    day_profile = [1.0, 2.0, 3.0]
    expected = get_rollout_errors(ConstantZeroPredictor(), day_profile, target="base_load")
    actual = get_rollout_errors(ConstantZeroPredictor(), day_profile, target="base_load")
    assert actual == expected


def test_get_rollout_errors_structure_is_present_for_single_step_profile():
    result = get_rollout_errors(ConstantZeroPredictor(), [5.0], target="base_load")
    assert set(result) == {
        "horizon_buckets",
        "timestep_buckets",
        "horizon_pooled_avg",
        "horizon_bucket_avg",
        "timestep_pooled_avg",
        "timestep_bucket_avg",
    }
    assert result["horizon_buckets"] == {1: [5.0]}
    assert result["timestep_buckets"] == {0: [5.0]}
    assert result["horizon_pooled_avg"] == pytest.approx(5.0)
    assert result["timestep_pooled_avg"] == pytest.approx(5.0)


class LinearPredictor(BasePredictor):
    def predict(self, household, horizon: int):
        return {"base_load": [10.0] * horizon}

    def predict_base_load(self, household, horizon: int):
        return {"base_load": [10.0] * horizon}

    def predict_pv_gen(self, household, horizon: int):
        return {"pv_gen": [2.0] * horizon}

    def predict_ev_status(self, household, horizon: int, ev_key: str | None = None):
        ev_key = ev_key or "ev1"
        return {
            f"{ev_key}_at_home": [0] * horizon,
            f"{ev_key}_at_charging_station": [1] * horizon,
        }


def _manual_linear_predictor_expected(day_profile, target):
    horizon_buckets = {}
    timestep_buckets = {}
    pooled = []
    profile = [float(v) for v in day_profile]

    for start in range(len(profile)):
        for horizon in range(1, len(profile) - start + 1):
            actual_window = profile[start:start + horizon]
            if target in {"base_load", "pv_gen"}:
                prediction = [10.0 if target == "base_load" else 2.0] * horizon
                error = math.sqrt(
                    sum((p - a) ** 2 for p, a in zip(prediction, actual_window)) / len(actual_window)
                )
            else:
                prediction = [2] * horizon
                error = sum(int(p != a) for p, a in zip(prediction, actual_window)) / len(actual_window)

            pooled.append(error)
            horizon_buckets.setdefault(horizon, []).append(error)
            timestep_buckets.setdefault(start, []).append(error)

    return {
        "horizon_buckets": {k: v for k, v in horizon_buckets.items() if v},
        "timestep_buckets": {k: v for k, v in timestep_buckets.items() if v},
        "horizon_pooled_avg": _mean(pooled),
        "horizon_bucket_avg": _mean([_mean(v) for v in horizon_buckets.values()]),
        "timestep_pooled_avg": _mean([x for vals in timestep_buckets.values() for x in vals]),
        "timestep_bucket_avg": _mean([_mean(v) for v in timestep_buckets.values()]),
    }


@pytest.mark.parametrize(
    "target,day_profile",
    [
        ("base_load", [1.0, 2.0, 3.0, 4.0]),
        ("pv_gen", [4.0, 5.0, 2.0, 1.0]),
        ("ev1_status", [0.0, 1.0, 2.0, 1.0, 2.0]),
    ],
)
def test_get_rollout_errors_matches_manual_for_nontrivial_predictor(target, day_profile):
    result = get_rollout_errors(LinearPredictor(), day_profile, target=target)
    expected = _manual_linear_predictor_expected(day_profile, target)

    assert result["horizon_buckets"] == expected["horizon_buckets"]
    assert result["timestep_buckets"] == expected["timestep_buckets"]
    assert result["horizon_pooled_avg"] == pytest.approx(expected["horizon_pooled_avg"])
    assert result["horizon_bucket_avg"] == pytest.approx(expected["horizon_bucket_avg"])
    assert result["timestep_pooled_avg"] == pytest.approx(expected["timestep_pooled_avg"])
    assert result["timestep_bucket_avg"] == pytest.approx(expected["timestep_bucket_avg"])


class ContextAwarePredictor(BasePredictor):
    def predict(self, household, horizon: int):
        return self.predict_base_load(household, horizon)

    def predict_base_load(self, household, horizon: int):
        step = int(getattr(household, "current_timestep", 1))
        return {"base_load": [float(step + 2.5)] * horizon}

    def predict_pv_gen(self, household, horizon: int):
        step = int(getattr(household, "current_timestep", 1))
        return {"pv_gen": [float((step % 4) + 0.5)] * horizon}

    def predict_ev_status(self, household, horizon: int, ev_key: str | None = None):
        ev_key = ev_key or "ev1"
        step = int(getattr(household, "current_timestep", 1))
        remainder = step % 3
        if remainder == 0:
            at_home = [1] * horizon
            at_station = [0] * horizon
        elif remainder == 1:
            at_home = [0] * horizon
            at_station = [0] * horizon
        else:
            at_home = [0] * horizon
            at_station = [1] * horizon
        return {
            f"{ev_key}_at_home": at_home,
            f"{ev_key}_at_charging_station": at_station,
        }


def _manual_context_aware_expected(day_profile, target):
    horizon_buckets = {}
    timestep_buckets = {}
    pooled = []

    for start in range(len(day_profile)):
        for horizon in range(1, len(day_profile) - start + 1):
            actual_window = day_profile[start:start + horizon]
            step = start + 1
            if target == "base_load":
                prediction = [float(step + 2.5)] * horizon
                error = math.sqrt(sum((p - a) ** 2 for p, a in zip(prediction, actual_window)) / len(actual_window))
            elif target == "pv_gen":
                prediction = [float((step % 4) + 0.5)] * horizon
                error = math.sqrt(sum((p - a) ** 2 for p, a in zip(prediction, actual_window)) / len(actual_window))
            else:
                remainder = step % 3
                if remainder == 0:
                    prediction = [0] * horizon
                elif remainder == 1:
                    prediction = [1] * horizon
                else:
                    prediction = [2] * horizon
                error = sum(int(p != a) for p, a in zip(prediction, actual_window)) / len(actual_window)

            pooled.append(error)
            horizon_buckets.setdefault(horizon, []).append(error)
            timestep_buckets.setdefault(start, []).append(error)

    return {
        "horizon_buckets": {k: v for k, v in horizon_buckets.items() if v},
        "timestep_buckets": {k: v for k, v in timestep_buckets.items() if v},
        "horizon_pooled_avg": _mean(pooled),
        "horizon_bucket_avg": _mean([_mean(v) for v in horizon_buckets.values()]),
        "timestep_pooled_avg": _mean([x for vals in timestep_buckets.values() for x in vals]),
        "timestep_bucket_avg": _mean([_mean(v) for v in timestep_buckets.values()]),
    }


@pytest.mark.parametrize(
    "target,day_profile",
    [
        ("base_load", [0.5, 1.5, 2.5, 3.5, 4.5]),
        ("pv_gen", [2.0, 4.0, 3.0, 5.0, 1.0]),
        ("ev1_status", [0.0, 1.0, 2.0, 1.0, 0.0, 2.0]),
        ("ev2_status", [2.0, 0.0, 1.0, 2.0, 1.0, 0.0]),
    ],
)
def test_get_rollout_errors_matches_manual_for_context_aware_predictor(target, day_profile):
    result = get_rollout_errors(ContextAwarePredictor(), day_profile, target=target)
    expected = _manual_context_aware_expected(day_profile, target)

    assert result["horizon_buckets"] == expected["horizon_buckets"]
    assert result["timestep_buckets"] == expected["timestep_buckets"]
    assert result["horizon_pooled_avg"] == pytest.approx(expected["horizon_pooled_avg"])
    assert result["horizon_bucket_avg"] == pytest.approx(expected["horizon_bucket_avg"])
    assert result["timestep_pooled_avg"] == pytest.approx(expected["timestep_pooled_avg"])
    assert result["timestep_bucket_avg"] == pytest.approx(expected["timestep_bucket_avg"])
