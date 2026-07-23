from src.simulation.household import Household
from xgboost import XGBRegressor


def _make_band(series: list[float], width_fraction: float) -> tuple[list[float], list[float]]:
    width_fraction = max(0.0, float(width_fraction))
    lower = [max(0.0, value * (1.0 - width_fraction)) for value in series]
    upper = [max(0.0, value * (1.0 + width_fraction)) for value in series]
    # The first forecast point corresponds to a known current-state value,
    # so its interval should have zero width.
    if series:
        lower[0] = float(series[0])
        upper[0] = float(series[0])
    return lower, upper


def _get_features_for_base_load(household: Household, horizon: int) -> list[list[float]]:
    """
    PLACEHOLDER
    Generates features for base load prediction based on household data and prediction horizon.

    Args:
        household (Household): The household for which to generate features.
        horizon (int): The number of time steps to predict.

    Returns:
        list[list[float]]: A list of feature vectors for each time step in the horizon.
    """
    # Placeholder implementation; replace with actual feature generation logic

    features = [[0.0] * 10 for _ in range(horizon)]  # Example: 10 features, all zeros
    return features


def predict_base_load(model: XGBRegressor, household: Household, horizon: int, interval_width_frct: float=0.0) -> dict[str, list[float]]:
    """
    Predicts the base load for a given household over a specified horizon.

    Args:
        household (Household): The household for which to predict base load.
        horizon (int): The number of time steps to predict.

    Returns:
        dict[str, list[float]]: A dictionary containing the predicted base load.
    """
    # Placeholder implementation; replace with actual prediction logic
    base_load_features = _get_features_for_base_load(household, horizon)
    base_load = model.predict(base_load_features).tolist()  # Convert to list for consistency
    
    base_load_lb, base_load_ub = _make_band(base_load, interval_width_frct)

    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }
    