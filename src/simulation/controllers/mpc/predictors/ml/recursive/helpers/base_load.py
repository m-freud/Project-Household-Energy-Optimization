from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features, _try_bypass
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household


def _predict_base_load(
    model: TRegressor,
    household: Household,
    horizon: int,
) -> list[float]:
    # current values
    current_timestep = household.current_timestep
    current_base_load = household.base_load

    # init sim history
    # we use a list here for convenience
    sim_base_load_history = list(household.history["base_load"].values())

    # init pred
    base_load_pred: list[float] = [current_base_load]

    for prediction_index in range(horizon - 1):
        bypass = _try_bypass(current_timestep)
        if bypass is not None:
            current_base_load = bypass
            base_load_pred.append(current_base_load)
            current_timestep += 1
            continue

        all_features = _build_base_load_features(
            current_timestep=current_timestep,
            current_base_load=current_base_load,
            base_load_history=sim_base_load_history,
        )

        model_family_name = ModelConfig.get_model_family_name(model)
        model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[model_family_name]["base_load"]

        # ensure completness of features for the model
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features (as in model_features)
        model_input = [all_features[f] for f in model_features]

        # update sim hist before next prediction
        sim_base_load_history.append(current_base_load)

        # get prediction and append
        current_base_load = model.predict([model_input])[0] # type: ignore
        base_load_pred.append(current_base_load)

        # incr time
        current_timestep += 1

    return base_load_pred


def predict_base_load(
    model: TRegressor,
    household: Household,
    horizon: int,
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    base_load = _predict_base_load(
        model=model,
        household=household,
        horizon=horizon,
    )
    base_load_lb, base_load_ub = make_band(base_load, interval_width_frct)

    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }