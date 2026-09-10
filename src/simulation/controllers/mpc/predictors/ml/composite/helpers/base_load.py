


from simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from simulation.controllers.mpc.predictors.shared.make_band import make_band
from simulation.household import Household



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

    # init pred
    base_load_pred: list[float] = [current_base_load]

    base_load_pred_dict: dict[int, float] = {}

    # now for the interesting part 
    # first get predictions for horizons where we have a model
    for h, model in model_bank.items():





    return [0.0] * horizon


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