from src.simulation.household import Household
from xgboost import XGBRegressor


def predict_base_load(household: Household, horizon: int) -> dict[str, list[float]]:
    """
    Predicts the base load for a given household over a specified horizon.

    Args:
        household (Household): The household for which to predict base load.
        horizon (int): The number of time steps to predict.

    Returns:
        dict[str, list[float]]: A dictionary containing the predicted base load.
    """
    # Placeholder implementation; replace with actual prediction logic
    base_load = [0.0] * horizon  # Replace with actual prediction values
    return {"base_load": base_load}