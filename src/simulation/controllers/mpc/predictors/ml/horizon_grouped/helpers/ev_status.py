from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers.ev_status import (
    _status_from_flags,
    _status_to_flags,
)
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.grouped_prediction import (
    get_group_model_features,
    predict_horizon_grouped,
)
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._ev_status import (
    _build_ev_status_features,
    _try_bypass,
)
from src.simulation.household import Household


def _predict_single_ev_status(
    model_bank: dict[str, ModelLike],
    household: Household,
    ev_key: str,
    horizon: int,
) -> tuple[list[int], list[int]]:
    home_key = f"{ev_key}_at_home"
    station_key = f"{ev_key}_at_charging_station"
    current_status = _status_from_flags(
        int(getattr(household, home_key, 0)),
        int(getattr(household, station_key, 0)),
    )
    home_history = household.history.get(home_key, {})
    station_history = household.history.get(station_key, {})
    status_history = [
        _status_from_flags(home_history[step], station_history[step])
        for step in sorted(home_history)
        if step in station_history
    ]
    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    model_features = get_group_model_features(
        model_bank,
        ModelConfig.MODEL_FEATURES_BY_FAMILY["global_xgboost"]["ev_status"],
    )
    required_features = model_features[next(iter(model_features))]

    def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
        return _build_ev_status_features(
            {
                "timestep": timestep,
                "status": int(round(value)),
                "status_history": status_history
                + [
                    int(round(item))
                    for item in feature_history[len(status_history) :]
                ],
                "start1_earliest": windows[0]["earliest_start"],
                "end1_latest": windows[0]["latest_end"],
                "start2_earliest": windows[1]["earliest_start"],
                "end2_latest": windows[1]["latest_end"],
                "max_commute_steps_1": windows[0]["max_unavailable_steps"],
                "max_commute_steps_2": windows[1]["max_unavailable_steps"],
            },
            required_features=required_features,
        )

    def try_bypass(
        timestep: int,
        preceding_statuses: list[float],
        current_value: float,
    ) -> float | None:
        return _try_bypass(
            current_time=timestep,
            ev_key=ev_key,
            status_history=[int(round(status)) for status in preceding_statuses],
            current_status=int(round(current_value)),
        )

    statuses = predict_horizon_grouped(
        models_by_group=model_bank,
        model_features_by_group=model_features,
        current_timestep=household.current_timestep,
        current_value=current_status,
        history=[float(status) for status in status_history],
        horizon=horizon,
        feature_builder=build_features,
        task="classification",
        bypass=try_bypass,
    )
    flags = [_status_to_flags(int(status)) for status in statuses]
    return [home for home, _ in flags], [station for _, station in flags]


def predict_ev_status(
    model_bank_ev1: dict[str, ModelLike],
    model_bank_ev2: dict[str, ModelLike],
    household: Household,
    horizon: int,
    ev_key: str | None = None,
) -> dict[str, list[int]]:
    if ev_key not in {None, "ev1", "ev2"}:
        raise ValueError("wrong ev_key, expected ev1 or ev2")

    result: dict[str, list[int]] = {}
    for key, model_bank in (("ev1", model_bank_ev1), ("ev2", model_bank_ev2)):
        if ev_key is not None and key != ev_key:
            continue
        home, station = _predict_single_ev_status(model_bank, household, key, horizon)
        result[f"{key}_at_home"] = home
        result[f"{key}_at_charging_station"] = station

    return result
