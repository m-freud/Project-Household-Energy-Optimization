from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _prediction_key_for_target(target: str) -> str:
    if target in {"base_load", "pv_gen"}:
        return target
    if target in {"ev1_status", "ev2_status"}:
        return f"{target.replace('_status', '')}_status"
    raise ValueError(f"Unsupported target for rollout metric: {target}")


def _step_error(predicted: Sequence[float], actual: Sequence[float], target: str) -> float:
    if not actual:
        return 0.0

    max_len = max(len(predicted), len(actual))
    padded_pred = [float(v) for v in predicted]
    if len(padded_pred) < max_len:
        fill_value = float(padded_pred[-1]) if padded_pred else 0.0
        padded_pred.extend([fill_value] * (max_len - len(padded_pred)))

    padded_actual = [float(v) for v in actual]
    if len(padded_actual) < max_len:
        fill_value = float(padded_actual[-1]) if padded_actual else 0.0
        padded_actual.extend([fill_value] * (max_len - len(padded_actual)))

    if target in {"base_load", "pv_gen"}:
        sq_err = [(p - a) ** 2 for p, a in zip(padded_pred[: len(actual)], padded_actual[: len(actual)])]
        return math.sqrt(sum(sq_err) / len(sq_err)) if sq_err else 0.0

    if target in {"ev1_status", "ev2_status"}:
        mismatches = [int(float(p) != float(a)) for p, a in zip(padded_pred[: len(actual)], padded_actual[: len(actual)])]
        return sum(mismatches) / len(mismatches) if mismatches else 0.0

    raise ValueError(f"Unsupported target for rollout metric: {target}")


def _status_from_home_station(at_home: int, at_charging_station: int) -> int:
    return int(1 - at_home + at_charging_station)


def _home_station_from_status(status: int) -> tuple[int, int]:
    if status == 0:
        return 1, 0
    if status == 1:
        return 0, 0
    if status == 2:
        return 0, 1
    raise ValueError(f"Unsupported EV status value: {status}. Expected one of 0, 1, 2.")


def _build_profile_context(day_profile: Sequence[float], start: int, target: str):
    profile = [float(v) for v in day_profile]
    start_idx = max(0, min(start, len(profile) - 1)) if profile else 0

    if target == "base_load":
        history = {"base_load": {i + 1: float(v) for i, v in enumerate(profile[: start_idx + 1])}}
        oracle_profiles = {"base_load": profile}
        base_load = float(profile[start_idx]) if profile else 0.0
        return SimpleNamespace(
            current_timestep=start + 1,
            base_load=base_load,
            history=history,
            oracle_profiles=oracle_profiles,
            has_pv=False,
        )

    if target == "pv_gen":
        history = {"pv_gen": {i + 1: float(v) for i, v in enumerate(profile[: start_idx + 1])}}
        oracle_profiles = {"pv_gen": profile}
        pv_gen = float(profile[start_idx]) if profile else 0.0
        return SimpleNamespace(
            current_timestep=start + 1,
            pv_gen=pv_gen,
            history=history,
            oracle_profiles=oracle_profiles,
            has_pv=True,
        )

    if target in {"ev1_status", "ev2_status"}:
        ev_key = "ev1" if target == "ev1_status" else "ev2"
        status_key = f"{ev_key}_status"
        status_profile = [int(float(v)) for v in profile]

        history = {
            f"{ev_key}_at_home": {},
            f"{ev_key}_at_charging_station": {},
            status_key: {},
        }
        for idx, status in enumerate(status_profile[: start_idx + 1], start=1):
            at_home, at_charging_station = _home_station_from_status(status)
            history[f"{ev_key}_at_home"][idx] = at_home
            history[f"{ev_key}_at_charging_station"][idx] = at_charging_station
            history[status_key][idx] = status

        oracle_profiles = {
            f"{ev_key}_at_home": [
                _home_station_from_status(status)[0] for status in status_profile
            ],
            f"{ev_key}_at_charging_station": [
                _home_station_from_status(status)[1] for status in status_profile
            ],
            status_key: status_profile,
        }

        return SimpleNamespace(
            current_timestep=start + 1,
            history=history,
            oracle_profiles=oracle_profiles,
            has_pv=False,
        )

    raise ValueError(f"Unsupported target for rollout metric: {target}")


def _predict_for_target(predictor: BasePredictor, context, horizon: int, target: str) -> list[float]:
    if target == "base_load":
        prediction = predictor.predict_base_load(context, horizon)
        return [float(v) for v in prediction.get("base_load", [])]

    if target == "pv_gen":
        prediction = predictor.predict_pv_gen(context, horizon)
        return [float(v) for v in prediction.get("pv_gen", [])]

    if target == "ev1_status":
        prediction = predictor.predict_ev_status(context, horizon, ev_key="ev1")
        at_home = prediction.get("ev1_at_home", [])
        at_station = prediction.get("ev1_at_charging_station", [])
        return [
            int(1 - int(home) + int(station))
            for home, station in zip(at_home, at_station)
        ]

    if target == "ev2_status":
        prediction = predictor.predict_ev_status(context, horizon, ev_key="ev2")
        at_home = prediction.get("ev2_at_home", [])
        at_station = prediction.get("ev2_at_charging_station", [])
        return [
            int(1 - int(home) + int(station))
            for home, station in zip(at_home, at_station)
        ]

    raise ValueError(f"Unsupported target for rollout metric: {target}")


def get_rollout_errors(
    predictor: BasePredictor,
    day_profile: Sequence[float],
    target: str = "base_load",
    reuse_forecast_prefix: bool = False,
) -> dict[str, Any]:
    """Compute rollout error buckets for a predictor over one day profile.

    The day profile acts as ground truth. For every start timestep and horizon,
    the predictor is asked for the remaining forecast and the resulting rollout
    error is stored by (a) horizon length and (b) start timestep. The function
    returns pooled and bucket-averaged metrics for each grouping.
    """
    day_profile = [float(v) for v in day_profile]
    day_len = len(day_profile)
    if day_len == 0:
        return {
            "horizon_buckets": {},
            "timestep_buckets": {},
            "horizon_pooled_avg": 0.0,
            "horizon_bucket_avg": 0.0,
            "timestep_pooled_avg": 0.0,
            "timestep_bucket_avg": 0.0,
        }

    horizon_buckets: dict[int, list[float]] = {h: [] for h in range(1, day_len + 1)}
    timestep_buckets: dict[int, list[float]] = {t: [] for t in range(day_len)}
    pooled_errors: list[float] = []

    for start in range(day_len):
        context = _build_profile_context(day_profile, start=start, target=target)
        max_horizon = day_len - start
        prediction_values = None
        if reuse_forecast_prefix:
            prediction_values = _predict_for_target(
                predictor, context, horizon=max_horizon, target=target
            )
            if not prediction_values:
                prediction_values = [0.0] * max_horizon

        for horizon in range(1, max_horizon + 1):
            actual_window = day_profile[start:start + horizon]
            if not reuse_forecast_prefix:
                prediction_values = _predict_for_target(
                    predictor, context, horizon=horizon, target=target
                )

            if not prediction_values:
                prediction_values = [0.0] * len(actual_window)

            error = _step_error(prediction_values, actual_window, target)
            pooled_errors.append(error)
            horizon_buckets.setdefault(horizon, []).append(error)
            timestep_buckets.setdefault(start, []).append(error)

    horizon_buckets = {k: v for k, v in horizon_buckets.items() if v}
    timestep_buckets = {k: v for k, v in timestep_buckets.items() if v}

    horizon_pooled_avg = _mean(pooled_errors)
    horizon_bucket_avg = _mean([_mean(v) for v in horizon_buckets.values()])
    timestep_pooled_avg = _mean([err for errors in timestep_buckets.values() for err in errors])
    timestep_bucket_avg = _mean([_mean(v) for v in timestep_buckets.values()])

    return {
        "horizon_buckets": horizon_buckets,
        "timestep_buckets": timestep_buckets,
        "horizon_pooled_avg": horizon_pooled_avg,
        "horizon_bucket_avg": horizon_bucket_avg,
        "timestep_pooled_avg": timestep_pooled_avg,
        "timestep_bucket_avg": timestep_bucket_avg,
    }


def get_rollout_error(
    predictor: BasePredictor,
    day_profile: Sequence[float],
    target: str = "base_load",
) -> dict[str, Any]:
    return get_rollout_errors(predictor, day_profile, target=target)
