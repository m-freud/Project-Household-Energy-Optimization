from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features


def _linear_interpolate(pred_dict: dict[int, float], horizon: int) -> dict[int, float]:
    known_horizons = sorted(pred_dict.keys())

    for h, next_h in zip(known_horizons, known_horizons[1:]):
        gap = next_h - h
        for missing_h in range(h + 1, next_h):
            pred_dict[missing_h] = pred_dict[h] + (pred_dict[next_h] - pred_dict[h]) * (missing_h - h) / gap

    # fill in any missing horizons at the end
    last_h = known_horizons[-1]
    for missing_h in range(last_h + 1, horizon):
        pred_dict[missing_h] = pred_dict[last_h]

    return pred_dict


def _recursive_interpolate(model_bank: dict[int, TRegressor], history: list[float], pred_dict: dict[int, float], horizon: int) -> dict[int, float]:
    known_horizons = sorted(pred_dict.keys())

    return pred_dict


def _interpolate_prediction(
        model_bank,
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int,
        interpolation: str = "linear"
) -> dict[int, float]:

    if interpolation == "linear":
        return _linear_interpolate(pred_dict, horizon)

    if interpolation == "recursive":
        return _recursive_interpolate(model_bank, history, pred_dict, horizon)

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
    base_load_pred_dict = _interpolate_prediction(
        model_bank=model_bank,
        history=base_load_history,
        pred_dict=base_load_pred_dict,
        horizon=horizon,
        interpolation=interpolation
    )

    return [base_load_pred_dict[h] for h in range(horizon)]


def predict_base_load(
    model_bank: dict[int, TRegressor],
    household: Household,
    horizon: int,
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