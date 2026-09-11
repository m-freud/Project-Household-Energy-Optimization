from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TClassifier
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._ev_status import _build_ev_status_features
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import predict_composite
from src.simulation.household import Household


def _status_from_flags(at_home: int, at_station: int) -> int:
    return 1 - int(at_home) + int(at_station)


def _status_to_flags(status: int) -> tuple[int, int]:
    if status == 0:
        return 1, 0
    if status == 1:
        return 0, 0
    if status == 2:
        return 0, 1
    raise ValueError(f"Unexpected EV status class: {status}")


def _predict_single_ev_status(
    model_bank: dict[int, TClassifier],
    household: Household,
    ev_key: str,
    horizon: int,
    interpolation: str,
) -> tuple[list[int], list[int]]:
    if not model_bank:
        return [0] * horizon, [0] * horizon

    home_key = f"{ev_key}_at_home"
    station_key = f"{ev_key}_at_charging_station"
    current_home = int(getattr(household, home_key, 0))
    current_station = int(getattr(household, station_key, 0))
    current_status = _status_from_flags(current_home, current_station)

    home_history = household.history.get(home_key, {})
    station_history = household.history.get(station_key, {})
    status_history = [
        _status_from_flags(home_history[step], station_history[step])
        for step in sorted(home_history)
        if step in station_history
    ]

    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["ev_status"]

    def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
        window = {
            "earliest_start": windows[0]["earliest_start"],
            "latest_end": windows[0]["latest_end"],
        }
        return _build_ev_status_features({
            "timestep": timestep,
            "status": int(round(value)),
            "status_history": status_history + [int(round(item)) for item in feature_history[len(status_history):]],
            "start1_earliest": windows[0]["earliest_start"],
            "end1_latest": windows[0]["latest_end"],
            "start2_earliest": windows[1]["earliest_start"],
            "end2_latest": windows[1]["latest_end"],
            "max_commute_steps_1": windows[0]["max_unavailable_steps"],
            "max_commute_steps_2": windows[1]["max_unavailable_steps"],
        })

    statuses = predict_composite(
        model_bank=model_bank,
        current_timestep=household.current_timestep,
        current_value=current_status,
        history=status_history,
        horizon=horizon,
        model_features=model_features,
        feature_builder=build_features,
        interpolation=interpolation,
        task="classification",
    )
    flags = [_status_to_flags(int(status)) for status in statuses]
    return [home for home, _ in flags], [station for _, station in flags]


def predict_ev_status(
    model_bank_ev1: dict[int, TClassifier] | None,
    model_bank_ev2: dict[int, TClassifier] | None,
    household: Household,
    horizon: int,
    ev_key: str | None = None,
    ev1_interpolation: str = "recursive",
    ev2_interpolation: str = "recursive",
) -> dict[str, list[int]]:
    if ev_key not in {None, "ev1", "ev2"}:
        raise ValueError("wrong ev_key, expected ev1 or ev2")

    results: dict[str, list[int]] = {}
    for key, bank, mode in (
        ("ev1", model_bank_ev1, ev1_interpolation),
        ("ev2", model_bank_ev2, ev2_interpolation),
    ):
        if ev_key is not None and key != ev_key:
            continue
        home, station = _predict_single_ev_status(bank or {}, household, key, horizon, mode)
        results[f"{key}_at_home"] = home
        results[f"{key}_at_charging_station"] = station

    if ev_key is not None:
        return results
    return {
        "ev1_at_home": results.get("ev1_at_home", [0] * horizon),
        "ev1_at_charging_station": results.get("ev1_at_charging_station", [0] * horizon),
        "ev2_at_home": results.get("ev2_at_home", [0] * horizon),
        "ev2_at_charging_station": results.get("ev2_at_charging_station", [0] * horizon),
    }
