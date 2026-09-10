
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import _build_pv_gen_features, _try_bypass


def _predict_pv_gen(
        model: TRegressor,
        household: Household,
        horizon: int = 96
        ) -> list[float]:
    
    if not household.has_pv:
        return [0.0] * horizon

    daylight_start = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["earliest_start"]
    daylight_end = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["latest_end"]

    # current values
    current_timestep = household.current_timestep
    current_pv_gen = household.pv_gen

    # init sim history
    sim_pv_history = list(household.history["pv_gen"].values())

    # init prediciton
    pv_pred: list[float] = [current_pv_gen]

    for _ in range(horizon - 1):
        bypass = _try_bypass(current_timestep)
        if bypass is not None:
            current_pv_gen = float(bypass)
            pv_pred.append(current_pv_gen)
            current_timestep += 1
            continue


        all_features = _build_pv_gen_features(
            current_timestep=current_timestep,
            current_pv_gen=current_pv_gen,
            pv_history=sim_pv_history,
            daylight_start=daylight_start,
            daylight_end=daylight_end,
        )

        model_family_name = ModelConfig.get_model_family_name(model)
        model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[model_family_name]["pv_gen"]

        # ensure completness of features for the model
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features (as in model_features)
        model_input = [all_features[f] for f in model_features]

        # update sim hist before next prediction
        sim_pv_history.append(float(current_pv_gen))

        current_pv_gen = model.predict([model_input])[0] # type: ignore
        pv_pred.append(current_pv_gen)

        # incr time
        current_timestep += 1

    return pv_pred


def predict_pv_gen(model: TRegressor, household: Household, horizon: int, interval_width_frct: float = 0.0) -> dict[str, list[float]]:
    pv_gen = _predict_pv_gen(model=model, household=household, horizon=horizon)
    pv_gen_lb, pv_gen_ub = make_band(pv_gen, interval_width_frct)

    return {
        "pv_gen": pv_gen,
        "pv_gen_lb": pv_gen_lb,
        "pv_gen_ub": pv_gen_ub,
    }