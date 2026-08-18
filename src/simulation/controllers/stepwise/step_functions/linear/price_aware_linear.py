from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.stepwise.step_functions.linear.linear import even_linear_policy, fast_charge_policy
from src.simulation.controllers.stepwise.step_functions.basic_examples import no_control as no_control_policy
from runtime_config import RuntimeConfig


def _q25_threshold(profile: list[float]) -> float | None:
    if not profile:
        return None
    sorted_profile = sorted(profile)
    idx = max(0, int(len(sorted_profile) * 0.25) - 1)
    return sorted_profile[idx]


def _q75_threshold(profile: list[float]) -> float | None:
    if not profile:
        return None
    sorted_profile = sorted(profile)
    idx = min(int(len(sorted_profile) * 0.75), len(sorted_profile) - 1)
    return sorted_profile[idx]


def _ev_quantile_profile(household: Household, ev_name: str) -> list[float]:
    # Use oracle EV buy-price profile directly; this is equivalent to a
    # "last day profile" (same quartiles)
    profile = household.oracle_profiles.get(f"{ev_name}_buy_price", [])
    if profile:
        return [float(value) for value in profile]
    return household.buy_price_day_profile


def _next_target(timestep: int, soc_targets: dict[int, float]) -> tuple[float, int]:
    deadline = min((t for t in soc_targets if t >= timestep), default=96)
    return soc_targets.get(deadline, 0.0), deadline


def _is_trajectory_urgent(
    current_soc: float,
    target_soc: float,
    current_timestep: int,
    deadline: int,
    max_power: float,
    efficiency: float,
    buffer_hours: float = 1.0,
) -> bool:
    if target_soc <= current_soc:
        return False

    if max_power <= 0 or efficiency <= 0:
        return True

    remaining_hours = max(deadline - current_timestep, 0) * RuntimeConfig.DURATION_TIMESTEP
    required_hours = (target_soc - current_soc) / (max_power * efficiency)
    return remaining_hours <= (required_hours + buffer_hours)


def _even_bess_power_to_target(
    current_soc: float,
    target_soc: float,
    current_timestep: int,
    deadline: int,
    max_charge: float,
    max_discharge: float,
    efficiency: float,
) -> float:
    remaining_steps = max(deadline - current_timestep, 1)
    horizon_hours = remaining_steps * RuntimeConfig.DURATION_TIMESTEP

    if target_soc > current_soc:
        required_power = (target_soc - current_soc) / (horizon_hours * efficiency)
        max_power_by_deficit = (target_soc - current_soc) / (RuntimeConfig.DURATION_TIMESTEP * efficiency)
        return min(required_power, max_charge, max_power_by_deficit)

    if current_soc > target_soc:
        required_power = ((current_soc - target_soc) * efficiency) / horizon_hours
        max_power_by_surplus = ((current_soc - target_soc) * efficiency) / RuntimeConfig.DURATION_TIMESTEP
        return -min(required_power, max_discharge, max_power_by_surplus)

    return 0.0


def price_aware_linear(
    household: Household,
    scenario: Scenario|None = None,
    base_behaviour: str = "no_control",
) -> dict:
    """
    Price-aware linear charging for EVs and conservative BESS control.
    Naive target fulfillment like other linear policies, but with some price-awareness.

    - EVs: if urgent, use even-linear; otherwise charge only when cheap, else even-linear.
    - BESS: if urgent, move evenly to target; otherwise reverse-price logic:
      discharge surplus on expensive price, charge deficit on cheap price.
    """

    if scenario is None:
        scenario = household.scenario

    if base_behaviour == "even_linear":
        controls = even_linear_policy(household, scenario)
    elif base_behaviour == "no_control":
        controls = no_control_policy(household, scenario)
    else:
        raise ValueError("base_behaviour must be 'no_control' or 'even_linear'")

    even_controls = even_linear_policy(household, scenario)
    fast_controls = fast_charge_policy(household, scenario)
    timestep = household.current_timestep

    for power_key, ev, ev_scenario in [
        ("ev1_power", household.ev1, scenario.ev1),
        ("ev2_power", household.ev2, scenario.ev2),
    ]:
        if ev is None or not (ev.at_home or ev.at_charging_station):
            continue

        target_soc, deadline = _next_target(timestep, ev_scenario.soc_targets)
        target_soc *= ev.capacity

        if ev.soc >= target_soc:
            continue

        if _is_trajectory_urgent(ev.soc, target_soc, timestep, deadline, ev.max_charge, ev.efficiency):
            controls[power_key] = even_controls[power_key]
            continue

        ev_profile = _ev_quantile_profile(household, ev.name)
        ev_q25 = _q25_threshold(ev_profile)

        if ev_q25 is not None and ev.buy_price <= ev_q25:
            controls[power_key] = fast_controls[power_key]

    if household.bess is None:
        return controls

    bess = household.bess
    pv_generation = household.pv.generation if household.pv else 0.0
    net_load = household.base_load + controls["ev1_power"] + controls["ev2_power"] - pv_generation
    max_discharge_by_load = max(0.0, net_load * bess.efficiency)
    target_soc, deadline = _next_target(timestep, scenario.bess.soc_targets)
    target_soc *= bess.capacity
    soc_deficit = target_soc - bess.soc

    needs_charge_urgently = soc_deficit > 0 and _is_trajectory_urgent(
        bess.soc,
        target_soc,
        timestep,
        deadline,
        bess.max_charge,
        bess.efficiency,
    )
    needs_discharge_urgently = soc_deficit < 0 and _is_trajectory_urgent(
        target_soc,
        bess.soc, # swap target and current for discharge urgency check
        timestep,
        deadline,
        bess.max_discharge,
        bess.efficiency,
        buffer_hours=0.0,
    )

    if needs_charge_urgently or needs_discharge_urgently:
        controls["bess_power"] = _even_bess_power_to_target(
            current_soc=bess.soc,
            target_soc=target_soc,
            current_timestep=timestep,
            deadline=deadline,
            max_charge=bess.max_charge,
            max_discharge=bess.max_discharge,
            efficiency=bess.efficiency,
        )
        return controls

    q25 = _q25_threshold(household.buy_price_day_profile)
    q75 = _q75_threshold(household.buy_price_day_profile)
    price_cheap = q25 is not None and household.buy_price <= q25
    price_expensive = q75 is not None and household.buy_price >= q75

    if soc_deficit < 0 and price_expensive:
        soc_surplus = -soc_deficit
        max_discharge_by_surplus = (soc_surplus * bess.efficiency) / RuntimeConfig.DURATION_TIMESTEP
        controls["bess_power"] = -min(bess.max_discharge, max_discharge_by_surplus, max_discharge_by_load)
    elif soc_deficit > 0 and price_cheap:
        max_charge_by_deficit = soc_deficit / (RuntimeConfig.DURATION_TIMESTEP * bess.efficiency)
        controls["bess_power"] = min(bess.max_charge, max_charge_by_deficit)

    return controls
