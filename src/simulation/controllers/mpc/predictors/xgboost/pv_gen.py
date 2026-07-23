
from src.simulation.household import Household
def predict_pv_gen(household: Household, horizon: int) -> dict[str, list[float]]:
    """
    Predicts the PV generation for a given household over a specified horizon.

    Args:
        household (Household): The household for which to predict PV generation.
        horizon (int): The number of time steps to predict.

    Returns:
        dict[str, list[float]]: A dictionary containing the predicted PV generation.
    """
    # Placeholder implementation; replace with actual prediction logic
    pv_generation = [0.0] * horizon  # Replace with actual prediction values
    return {"pv_generation": pv_generation}