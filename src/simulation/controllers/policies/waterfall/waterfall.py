from pathlib import Path
import sys

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.config import Config


ARBITRAGE_MARGIN = 0.03


def _next_target(timestep: int, soc_targets: dict) -> tuple[float, int]:
    deadline = min((t for t in soc_targets if t >= timestep), default=96)
    return soc_targets.get(deadline, 0.0), deadline


def _to_kwh(value: float, capacity: float) -> float:
    return value * capacity if value <= 1.0 else value


def _price_quartiles(profile: list[float]) -> tuple[float | None, float | None]:
    if not profile:
        return None, None
    sorted_profile = sorted(profile)
    n = len(sorted_profile)
    q25 = sorted_profile[max(0, int(n * 0.25) - 1)]
    q75 = sorted_profile[min(int(n * 0.75), n - 1)]
    return q25, q75


def _is_trajectory_urgent(
    current_soc: float,
    target_soc: float,
    timestep: int,
    deadline: int,
    max_power: float,
    efficiency: float,
    buffer_hours: float = 1.0,
) -> bool:
    if target_soc <= current_soc:
        return False

    if max_power <= 0 or efficiency <= 0:
        return True

    remaining_hours = max(deadline - timestep, 0) * Config.DURATION_TIMESTEP
    required_hours = (target_soc - current_soc) / (max_power * efficiency)
    return remaining_hours <= (required_hours + buffer_hours)


def _trajectory_power(
    deficit_kwh: float,
    remaining_steps: int,
    max_charge: float,
    efficiency: float,
) -> float:
    if deficit_kwh <= 0:
        return 0.0

    max_by_deficit = deficit_kwh / (Config.DURATION_TIMESTEP * efficiency)
    even_power = (deficit_kwh / (max(remaining_steps, 1) * Config.DURATION_TIMESTEP)) / efficiency
    return min(max_charge, max_by_deficit, even_power)


def _windows_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def _estimated_unavailable_steps(ev_name: str, timestep: int, deadline: int) -> int:
    windows = Config.EV_UNAVAILABLE_WINDOWS.get(ev_name, [])
    if not windows:
        return 0

    horizon_start = timestep
    horizon_end = max(deadline, timestep + 1)
    blocked_steps = 0

    for window in windows:
        start = int(window.get("earliest_start", 0))
        end = int(window.get("latest_end", 0))
        max_unavailable = int(window.get("max_unavailable_steps", 0))
        if max_unavailable <= 0:
            continue
        if _windows_overlap(horizon_start, horizon_end, start, end):
            blocked_steps += max_unavailable

    return blocked_steps


def _required_soc_floor(
    target_soc: float,
    timestep: int,
    deadline: int,
    max_charge: float,
    efficiency: float,
) -> float:
    remaining_hours = max(deadline - timestep, 0) * Config.DURATION_TIMESTEP
    max_addable = remaining_hours * max_charge * efficiency
    return max(0.0, target_soc - max_addable)


def _is_profitable_discharge(
    current_buy_price: float,
    reference_charge_price: float,
    round_trip_efficiency: float,
    arbitrage_margin: float = ARBITRAGE_MARGIN,
) -> bool:
    if round_trip_efficiency <= 0:
        return False
    effective_cost = (reference_charge_price / round_trip_efficiency) + arbitrage_margin
    return current_buy_price > effective_cost


def waterfall_policy(household: Household, scenario: Scenario) -> dict:
    """
    Waterfall V1 policy.

    Rule stack:
    1) urgent trajectory guarantee,
    2) absorb PV surplus,
    3) cheap-price charge,
    4) expensive-price hold / discharge if profitable,
    5) even-linear fallback to target.
    """
    controls = {"ev1_power": 0.0, "ev2_power": 0.0, "bess_power": 0.0}

    t = household.current_timestep
    buy_price = household.buy_price

    q25, q75 = _price_quartiles(household.buy_price_day_profile)
    price_cheap = q25 is not None and buy_price <= q25
    price_expensive = q75 is not None and buy_price >= q75

    pv_gen = household.pv.generation if household.pv else 0.0
    base_load = household.base_load or 0.0
    pv_surplus = max(0.0, pv_gen - base_load)

    # EVs: charge-only in this simulator model.
    # Only at-home EV charging can consume household PV surplus.
    for power_key, ev, ev_scenario in [
        ("ev1_power", household.ev1, scenario.ev1),
        ("ev2_power", household.ev2, scenario.ev2),
    ]:
        if ev is None or not (ev.at_home or ev.at_charging_station):
            continue

        target_kwh, deadline = _next_target(t, ev_scenario.soc_targets)
        target_kwh = _to_kwh(target_kwh, ev.capacity)
        deficit = target_kwh - ev.soc
        raw_remaining_steps = max(deadline - t, 1)
        blocked_steps = _estimated_unavailable_steps(ev.name, t, deadline)
        remaining_steps = max(1, raw_remaining_steps - blocked_steps)

        ev_profile = getattr(household, f"{ev.name}_buy_price_day_profile", household.buy_price_day_profile)
        ev_q25, ev_q75 = _price_quartiles(ev_profile)
        ev_price_cheap = ev_q25 is not None and ev.buy_price <= ev_q25
        ev_price_expensive = ev_q75 is not None and ev.buy_price >= ev_q75

        planning_max_charge = getattr(ev, "charge_slowest", ev.max_charge)
        if planning_max_charge is None or planning_max_charge <= 0:
            planning_max_charge = ev.max_charge

        effective_deadline = t + remaining_steps
        urgent = _is_trajectory_urgent(
            ev.soc,
            target_kwh,
            t,
            effective_deadline,
            planning_max_charge,
            ev.efficiency,
            buffer_hours=2.0,
        )

        if urgent and deficit > 0:
            controls[power_key] = _trajectory_power(deficit, remaining_steps, ev.max_charge, ev.efficiency)
        elif pv_surplus > 0 and deficit > 0 and ev.at_home:
            max_by_deficit = deficit / (Config.DURATION_TIMESTEP * ev.efficiency)
            power = min(pv_surplus, ev.max_charge, max_by_deficit)
            controls[power_key] = power
            pv_surplus -= power
        elif ev_price_cheap and deficit > 0:
            max_by_deficit = deficit / (Config.DURATION_TIMESTEP * ev.efficiency)
            controls[power_key] = min(ev.max_charge, max_by_deficit)
        elif ev_price_expensive and not urgent:
            pass
        elif deficit > 0:
            controls[power_key] = _trajectory_power(deficit, remaining_steps, ev.max_charge, ev.efficiency)

    if household.bess is None:
        return controls

    bess = household.bess
    target_kwh, deadline = _next_target(t, scenario.bess.soc_targets)
    target_kwh = _to_kwh(target_kwh, bess.capacity)
    deficit = target_kwh - bess.soc
    remaining_steps = max(deadline - t, 1)

    urgent = _is_trajectory_urgent(
        bess.soc,
        target_kwh,
        t,
        deadline,
        bess.max_charge,
        bess.efficiency,
        buffer_hours=0.0,
    )

    required_floor_soc = _required_soc_floor(
        target_soc=target_kwh,
        timestep=t,
        deadline=deadline,
        max_charge=bess.max_charge,
        efficiency=bess.efficiency,
    )
    free_energy_kwh = max(0.0, bess.soc - required_floor_soc)

    ev1_home_power = controls["ev1_power"] if household.ev1 and household.ev1.at_home else 0.0
    ev2_home_power = controls["ev2_power"] if household.ev2 and household.ev2.at_home else 0.0
    net_load_excl_bess = base_load + ev1_home_power + ev2_home_power - pv_gen

    if Config.DURATION_TIMESTEP > 0:
        max_discharge_by_free = (free_energy_kwh * bess.efficiency) / Config.DURATION_TIMESTEP
    else:
        max_discharge_by_free = 0.0
    discharge_to_load = min(
        max(0.0, net_load_excl_bess),
        bess.max_discharge,
        max_discharge_by_free,
    )

    reference_charge_price = q25 if q25 is not None else buy_price
    round_trip_efficiency = bess.efficiency * bess.efficiency
    discharge_profitable = _is_profitable_discharge(
        current_buy_price=buy_price,
        reference_charge_price=reference_charge_price,
        round_trip_efficiency=round_trip_efficiency,
    )

    if urgent and deficit > 0:
        controls["bess_power"] = _trajectory_power(deficit, remaining_steps, bess.max_charge, bess.efficiency)
    elif pv_surplus > 0 and deficit > 0:
        max_by_deficit = deficit / (Config.DURATION_TIMESTEP * bess.efficiency)
        controls["bess_power"] = min(pv_surplus, bess.max_charge, max_by_deficit)
    elif price_cheap and deficit > 0:
        max_by_deficit = deficit / (Config.DURATION_TIMESTEP * bess.efficiency)
        controls["bess_power"] = min(bess.max_charge, max_by_deficit)
    elif price_expensive and not urgent:
        if discharge_to_load > 0 and discharge_profitable:
            controls["bess_power"] = -discharge_to_load
    elif deficit > 0:
        base_charge = _trajectory_power(deficit, remaining_steps, bess.max_charge, bess.efficiency)
        max_by_deficit = deficit / (Config.DURATION_TIMESTEP * bess.efficiency)
        controls["bess_power"] = min(base_charge + pv_surplus, bess.max_charge, max_by_deficit)
    elif discharge_to_load > 0 and discharge_profitable:
        controls["bess_power"] = -discharge_to_load

    return controls