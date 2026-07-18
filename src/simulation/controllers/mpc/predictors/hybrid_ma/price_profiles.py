from __future__ import annotations

from src.simulation.household import Household


# Predictors for buy_price, sell_price, ev_buy_price, grid_prices, ev_station_prices
# (mostly oracle pass-through, with some composition logic for ev_buy_price)

def _lookup_slice(household: Household, key: str, horizon: int) -> list[float]:
    # current_timestep is 1-based while Python lists are 0-based.
    start_time_idx = household.current_timestep - 1
    profile = household.oracle_profiles.get(key, [])
    raw = [float(value) for value in profile[start_time_idx : start_time_idx + horizon]]
    # Pad to the requested horizon so callers always get exactly `horizon` values.
    if len(raw) < horizon:
        fill = raw[-1] if raw else 0.0
        raw.extend([fill] * (horizon - len(raw)))
    return raw


def predict_buy_price(household: Household, horizon: int) -> dict[str, list[float]]:
    return {
        "buy_price": _lookup_slice(household, "buy_price", horizon),
    }


def predict_sell_price(household: Household, horizon: int) -> dict[str, list[float]]:
    return {
        "sell_price": _lookup_slice(household, "sell_price", horizon),
    }


def predict_grid_prices(household: Household, horizon: int) -> dict[str, list[float]]:
    """Known-ahead market prices: oracle pass-through."""

    payload: dict[str, list[float]] = {}
    payload.update(predict_buy_price(household, horizon))
    payload.update(predict_sell_price(household, horizon))
    return payload


def predict_ev_station_prices(household: Household, horizon: int) -> dict[str, list[float]]:
    """Known-ahead EV station tariffs: oracle pass-through."""

    ev1_station_buy_price = _lookup_slice(
        household,
        "ev1_buy_price",
        horizon,
    )
    ev2_station_buy_price = _lookup_slice(
        household,
        "ev2_buy_price",
        horizon,
    )

    return {
        "ev1_station_buy_price": ev1_station_buy_price,
        "ev2_station_buy_price": ev2_station_buy_price,
    }


def compose_ev_buy_prices(
    horizon: int,
    ev_status: dict[str, list[float]],
    grid_prices: dict[str, list[float]],
    station_prices: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Compose effective EV buy prices from EV status and known tariffs.

    Placeholder rule:
    - if at charging station -> station tariff
    - otherwise -> household grid buy price
    """

    ev1_buy: list[float] = []
    ev2_buy: list[float] = []

    for i in range(horizon):
        ev1_station = 1.0 if ev_status["ev1_at_charging_station"][i] > 0 else 0.0
        ev2_station = 1.0 if ev_status["ev2_at_charging_station"][i] > 0 else 0.0

        ev1_home = 1.0 if ev_status["ev1_at_home"][i] > 0 else 0.0
        ev2_home = 1.0 if ev_status["ev2_at_home"][i] > 0 else 0.0

        if not (ev1_station or ev1_home):
            ev1_price = 0.0  # EV1 is not available, price is irrelevant
        else:
            ev1_price = station_prices["ev1_station_buy_price"][i] if ev1_station > 0 else grid_prices["buy_price"][i]

        if not (ev2_station or ev2_home):
            ev2_price = 0.0  # EV2 is not available, price is irrelevant
        else:
            ev2_price = station_prices["ev2_station_buy_price"][i] if ev2_station > 0 else grid_prices["buy_price"][i]

        ev1_buy.append(float(ev1_price))
        ev2_buy.append(float(ev2_price))

    return {
        "ev1_buy_price": ev1_buy,
        "ev2_buy_price": ev2_buy,
    }


def predict_ev_buy_price(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
) -> dict[str, list[float]]:
    grid_prices = predict_grid_prices(household, horizon)
    station_prices = predict_ev_station_prices(household, horizon)
    return compose_ev_buy_prices(horizon, ev_status, grid_prices, station_prices)
