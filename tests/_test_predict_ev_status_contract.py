from __future__ import annotations

from dataclasses import dataclass, field

from src.simulation.controllers.mpc.predictors.hybrid_ap.ev_status import (
    _worst_case_two_commute_starts,
    predict_ev_status,
)


@dataclass
class DummyHousehold:
    current_timestep: int
    ev1_at_home: bool
    ev1_at_charging_station: bool
    ev2_at_home: bool
    ev2_at_charging_station: bool
    buy_price: float = 0.30
    ev1_station_buy_price: float = 0.55
    ev2_station_buy_price: float = 0.55
    buy_price_day_profile: list[float] = field(default_factory=lambda: [0.30 + 0.001 * i for i in range(96)])
    history: dict = field(default_factory=dict)
    past_state: str = "home"

    def __post_init__(self):
        if not self.history:
            self.history = {
                "ev1_at_home": {},
                "ev1_at_charging_station": {},
                "ev2_at_home": {},
                "ev2_at_charging_station": {},
            }
        # Populate observed past timesteps 1..current_timestep-1 so predictor logic
        # can infer historical states from the same structure used in simulation.
        for period in range(1, max(1, int(self.current_timestep))):
            if self.past_state == "home":
                ev_home, ev_station = 1.0, 0.0
            elif self.past_state == "station":
                ev_home, ev_station = 0.0, 1.0
            else:
                ev_home, ev_station = 0.0, 0.0

            self.history["ev1_at_home"][period] = ev_home
            self.history["ev1_at_charging_station"][period] = ev_station
            self.history["ev2_at_home"][period] = ev_home
            self.history["ev2_at_charging_station"][period] = ev_station


def _state_at(output: dict[str, list[float]], ev: str, idx: int) -> str:
    home = output[f"{ev}_at_home"][idx] > 0.5
    station = output[f"{ev}_at_charging_station"][idx] > 0.5
    if home:
        return "home"
    if station:
        return "station"
    return "driving"


def _assert_binary_and_lengths(output: dict[str, list[float]], horizon: int):
    expected_keys = {
        "ev1_at_home",
        "ev2_at_home",
        "ev1_at_charging_station",
        "ev2_at_charging_station",
    }
    assert set(output.keys()) == expected_keys, f"unexpected output keys: {set(output.keys())}"

    for key in [
        "ev1_at_home",
        "ev1_at_charging_station",
        "ev2_at_home",
        "ev2_at_charging_station",
    ]:
        values = output[key]
        assert len(values) == horizon, f"{key} length mismatch: {len(values)} != {horizon}"
        for v in values:
            assert v in (0.0, 1.0), f"{key} must be binary, got {v}"


def test_still_home_keeps_base_profile_now_semantics():
    horizon = 24
    hh = DummyHousehold(
        current_timestep=20,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    first_start, second_start = _worst_case_two_commute_starts(hh, "ev1", commute_steps=5, day_end_period=96)
    first_end = first_start + 5 - 1
    second_end = second_start + 5 - 1

    for i in range(horizon):
        period = hh.current_timestep + i
        if period < first_start:
            expected = "home"
        elif first_start <= period <= first_end:
            expected = "driving"
        elif period < second_start:
            expected = "station"
        elif second_start <= period <= second_end:
            expected = "driving"
        else:
            expected = "home"
        assert _state_at(output, "ev1", i) == expected


def test_driving_now_forces_first_start_to_now():
    horizon = 12
    hh = DummyHousehold(
        current_timestep=35,
        ev1_at_home=False,
        ev1_at_charging_station=False,
        ev2_at_home=False,
        ev2_at_charging_station=False,
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    # If currently driving in first half, first predicted state must be driving now.
    assert _state_at(output, "ev1", 0) == "driving"
    # Commute duration is 5 -> first five predicted states should be driving.
    for i in range(5):
        assert _state_at(output, "ev1", i) == "driving"


def test_predicted_start_now_but_still_home_shifts_right_one_step():
    horizon = 8
    base_hh = DummyHousehold(
        current_timestep=1,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
    )

    first_start, _ = _worst_case_two_commute_starts(base_hh, "ev1", commute_steps=5, day_end_period=96)

    hh = DummyHousehold(
        current_timestep=first_start,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
    )

    output = predict_ev_status(hh, horizon)

    # Intended behavior from contract: if start is predicted for now but car is still home,
    # shift start one period to the right -> first period should remain home.
    assert _state_at(output, "ev1", 0) == "home"


def test_station_state_keeps_second_commute_base_timing():
    horizon = 8
    base_hh = DummyHousehold(
        current_timestep=1,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
    )
    _, second_start = _worst_case_two_commute_starts(base_hh, "ev1", commute_steps=5, day_end_period=96)

    hh = DummyHousehold(
        current_timestep=second_start - 2,
        ev1_at_home=False,
        ev1_at_charging_station=True,
        ev2_at_home=False,
        ev2_at_charging_station=True,
        past_state="station",
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    # While at station before second commute starts, remain station.
    assert _state_at(output, "ev1", 0) == "station"
    assert _state_at(output, "ev1", 1) == "station"
    # At second_start, second commute begins.
    assert _state_at(output, "ev1", 2) == "driving"


def test_driving_second_commute_forces_second_start_to_now():
    horizon = 10
    hh = DummyHousehold(
        current_timestep=80,
        ev1_at_home=False,
        ev1_at_charging_station=False,
        ev2_at_home=False,
        ev2_at_charging_station=False,
        past_state="station",
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    # Analog to first commute: if currently driving in second half, second commute starts now.
    for i in range(5):
        assert _state_at(output, "ev1", i) == "driving"


def test_back_home_stays_home_for_rest_of_day():
    horizon = 2
    hh = DummyHousehold(
        current_timestep=95,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
        past_state="home",
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    assert _state_at(output, "ev1", 0) == "home"
    assert _state_at(output, "ev1", 1) == "home"


def test_predicted_arrival_passed_but_still_driving_shifts_arrival_right():
    horizon = 6
    base_hh = DummyHousehold(
        current_timestep=1,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        ev2_at_home=True,
        ev2_at_charging_station=False,
    )
    _, second_start = _worst_case_two_commute_starts(base_hh, "ev1", commute_steps=5, day_end_period=96)
    second_end = second_start + 5 - 1

    hh = DummyHousehold(
        current_timestep=second_end + 1,
        ev1_at_home=False,
        ev1_at_charging_station=False,
        ev2_at_home=False,
        ev2_at_charging_station=False,
        past_state="station",
    )

    output = predict_ev_status(hh, horizon)
    _assert_binary_and_lengths(output, horizon)

    # If arrival was predicted already but EV is still driving now, shift arrival to the right.
    assert _state_at(output, "ev1", 0) == "driving"


def main():
    tests = [
        test_still_home_keeps_base_profile_now_semantics,
        test_driving_now_forces_first_start_to_now,
        test_predicted_start_now_but_still_home_shifts_right_one_step,
        test_station_state_keeps_second_commute_base_timing,
        test_driving_second_commute_forces_second_start_to_now,
        test_back_home_stays_home_for_rest_of_day,
        test_predicted_arrival_passed_but_still_driving_shifts_arrival_right,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    main()
