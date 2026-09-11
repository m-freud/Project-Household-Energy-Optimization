from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features


def _linear_interpolate(
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int
        ) -> list[float]:
    known_horizons = sorted(pred_dict.keys())

    for h, next_h in zip(known_horizons, known_horizons[1:]):
        gap = next_h - h
        for missing_h in range(h + 1, next_h):
            pred_dict[missing_h] = pred_dict[h] + (pred_dict[next_h] - pred_dict[h]) * (missing_h - h) / gap

    # fill in any missing horizons at the end
    last_h = known_horizons[-1]
    for missing_h in range(last_h + 1, horizon - len(history)):
        pred_dict[missing_h] = pred_dict[last_h]

    pred_list = [pred_dict[h] for h in range(horizon - len(history))]

    return pred_list


def _get_first_h_gap(known_horizons, horizon):
    for i in range(1, horizon + 1):
        if i not in known_horizons:
            prev_h = max(h for h in known_horizons if h < i)
            next_h = min(h for h in known_horizons if h > i)
            gap_len = next_h - prev_h
            return i, gap_len
    return None, None


def _get_longest_model_horizon(model_bank: dict[int, TRegressor], gap_len: int) -> int | None:
    suitable_horizons = [h for h in model_bank.keys() if h <= gap_len]
    if not suitable_horizons:
        return None
    return max(suitable_horizons)


def _recursive_interpolate(
        model_bank: dict[int, TRegressor],
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int
        ) -> list[float]:
    known_horizons = sorted(pred_dict.keys())

    
    while len(known_horizons) < horizon - len(history):
        first_missing_h, gap_len = _get_first_h_gap(known_horizons, horizon)
        if first_missing_h is None or gap_len is None:
            break

        history_for_model = history + [pred_dict[h] for h in sorted(pred_dict.keys()) if h > 0 and h < first_missing_h]
        model_input_features = _build_base_load_features(
            current_timestep=first_missing_h - 1,
            current_base_load=pred_dict[first_missing_h - 1],
            base_load_history=history_for_model,
        )

        model_horizon = _get_longest_model_horizon(model_bank, gap_len)

        if model_horizon is None:
            break

        pred_dict[first_missing_h] = float(model_bank[model_horizon].predict([model_input_features])[0])  # type: ignore
        known_horizons = sorted(pred_dict.keys())

    # end of while loop

    pred_list = [pred_dict[h] for h in range(horizon - len(history))]

    return pred_list


def _interpolate_prediction(
        model_bank,
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int,
        current_timestep: int,
        interpolation: str = "linear"
) -> list[float]:

    if interpolation == "linear":
        interpolated_pred = _linear_interpolate(history, pred_dict, horizon)
        return interpolated_pred + [0.0] * (horizon - len(history) - len(interpolated_pred)) # pad to full horizon

    if interpolation == "recursive":
        interpolated_pred = _recursive_interpolate(model_bank, history, pred_dict, horizon)
        return interpolated_pred + [0.0] * (horizon - len(history) - len(interpolated_pred)) # pad to full horizon

    raise ValueError(f"Unknown interpolation method: {interpolation}")


def _predict_base_load(
        model_bank:dict[int, TRegressor],
        household:Household,
        horizon:int,
        interpolation: str = "linear"
    ) -> list[float]:
    # current values
    current_timestep = household.current_timestep
    current_base_load = household.base_load

    # init sim history
    # we use a list here for convenience
    base_load_history = list(household.history["base_load"].values())

    # anchor at horizon 0 = current (observed) value, so interpolation has a start point
    base_load_pred_dict: dict[int, float] = {0: current_base_load}

    # features are built once from the current state; direct multi-horizon
    # models all predict from the same "now", just at different target offsets
    all_features = _build_base_load_features(
        current_timestep=current_timestep,
        current_base_load=current_base_load,
        base_load_history=base_load_history,
    )

    model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]

    for h, model in model_bank.items():
        # ensure completeness of features for the model (redundant but whatever)
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features (as in model_features)
        model_input = [all_features[f] for f in model_features]

        base_load_pred_dict[h] = float(model.predict([model_input])[0])  # type: ignore

    # now interpolate
    base_load_pred_list = _interpolate_prediction(
        model_bank=model_bank,
        history=base_load_history,
        pred_dict=base_load_pred_dict,
        horizon=horizon,
        current_timestep=current_timestep,
        interpolation=interpolation
    )

    return base_load_pred_list


def predict_base_load(
    model_bank: dict[int, TRegressor],
    household: Household,
    horizon: int=96,
    interpolation: str = "linear",
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    base_load = _predict_base_load(
        model_bank=model_bank,
        household=household,
        horizon=horizon,
        interpolation=interpolation,
    )
    base_load_lb, base_load_ub = make_band(base_load, interval_width_frct)

    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }