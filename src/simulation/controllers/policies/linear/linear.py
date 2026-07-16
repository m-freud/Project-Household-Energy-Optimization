from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.config import Config


def get_next_target(current_timestep: int, target_soc_dict: dict[int, float]) -> tuple[float, int]:
    deadline = min((t for t in target_soc_dict.keys() if t >= current_timestep), default=96)
    target_soc = target_soc_dict.get(deadline, 0.0)
    return target_soc, deadline


def even_linear_policy(household: Household, scenario: Scenario) -> dict:
    # Reach target by charging evenly. Naive baseline. 
    controls = {
        "ev1_power": 0.0,
        "ev2_power": 0.0,
        "bess_power": 0.0,
    }

    for ev, ev_scenario in [(household.ev1, scenario.ev1), (household.ev2, scenario.ev2)]:
        if ev and (ev.at_home or ev.at_charging_station):
            target_soc, deadline = get_next_target(household.current_timestep, ev_scenario.soc_targets)
            target_soc = target_soc * ev.capacity
            soc_deficit = target_soc - ev.soc
            remaining_steps = max(deadline - household.current_timestep, 0)

            if soc_deficit > 0:
                if remaining_steps <= 0:
                    controls[f"{ev.name}_power"] = ev.max_charge
                else:
                    controls[f"{ev.name}_power"] = min(
                        (soc_deficit / (remaining_steps * Config.DURATION_TIMESTEP)) / ev.efficiency,
                        ev.max_charge,
                    )

    if household.bess:
        bess = household.bess
        target_soc, deadline = get_next_target(household.current_timestep, scenario.bess.soc_targets)
        target_soc = target_soc * bess.capacity
        soc_delta = target_soc - bess.soc

        if soc_delta > 0:
            remaining_steps = max(deadline - household.current_timestep, 0)
            max_charge_by_deficit = soc_delta / (Config.DURATION_TIMESTEP * bess.efficiency)

            if remaining_steps <= 0:
                charge_power = min(bess.max_charge, max_charge_by_deficit)
            else:
                charge_power = min(
                    (soc_delta / (remaining_steps * Config.DURATION_TIMESTEP)) / bess.efficiency,
                    bess.max_charge,
                    max_charge_by_deficit,
                )

            controls["bess_power"] = charge_power
        elif soc_delta < 0:
            soc_surplus = -soc_delta
            remaining_steps = max(deadline - household.current_timestep, 0)
            max_discharge_by_surplus = (soc_surplus * bess.efficiency) / Config.DURATION_TIMESTEP

            if remaining_steps <= 0:
                discharge_power = min(bess.max_discharge, max_discharge_by_surplus)
            else:
                discharge_power = min(
                    (soc_surplus * bess.efficiency) / (remaining_steps * Config.DURATION_TIMESTEP),
                    bess.max_discharge,
                    max_discharge_by_surplus,
                )

            controls["bess_power"] = -discharge_power

    return controls


def fast_charge_policy(household: Household, scenario: Scenario) -> dict:
    # Just charge to target as fast as possible. Naive baseline.
    controls = {
        "ev1_power": 0.0,
        "ev2_power": 0.0,
        "bess_power": 0.0,
    }

    if household.ev1 and (household.ev1.at_home or household.ev1.at_charging_station):
        ev1_target_soc, _ = get_next_target(household.current_timestep, scenario.ev1.soc_targets)
        ev1_target_soc = ev1_target_soc * household.ev1.capacity
        if household.ev1.soc < ev1_target_soc:
            required_power = (ev1_target_soc - household.ev1.soc) / (Config.DURATION_TIMESTEP * household.ev1.efficiency)
            controls["ev1_power"] = min(household.ev1.max_charge, required_power)

    if household.ev2 and (household.ev2.at_home or household.ev2.at_charging_station):
        ev2_target_soc, _ = get_next_target(household.current_timestep, scenario.ev2.soc_targets)
        ev2_target_soc = ev2_target_soc * household.ev2.capacity
        if household.ev2.soc < ev2_target_soc:
            required_power = (ev2_target_soc - household.ev2.soc) / (Config.DURATION_TIMESTEP * household.ev2.efficiency)
            controls["ev2_power"] = min(household.ev2.max_charge, required_power)

    if household.bess:
        bess_target_soc, _ = get_next_target(household.current_timestep, scenario.bess.soc_targets)
        bess_target_soc = bess_target_soc * household.bess.capacity

        soc_delta = bess_target_soc - household.bess.soc
        if soc_delta > 0:
            max_charge_by_deficit = soc_delta / (Config.DURATION_TIMESTEP * household.bess.efficiency)
            controls["bess_power"] = min(household.bess.max_charge, max_charge_by_deficit)
        elif soc_delta < 0:
            soc_surplus = -soc_delta
            max_discharge_by_surplus = (soc_surplus * household.bess.efficiency) / Config.DURATION_TIMESTEP
            controls["bess_power"] = -min(
                household.bess.max_discharge,
                max_discharge_by_surplus,
            )

    return controls
