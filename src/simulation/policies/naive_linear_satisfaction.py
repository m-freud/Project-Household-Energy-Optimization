'''
Docstring for src.simulation.policies.rule_based
'''
# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
import math


def naive_linear_satisfaction(household:Household, scenario:Scenario):
    '''
    greedy policy with targets.
    calculates average needed power to meet target soc by deadline,
    then sets charge power to exactly that amount (capped by max charge rate).
    uses pv surplus to charge bess if available.
    '''

    # ev1 control
    if household.ev1:
        current_soc = household.ev1.soc
        required_soc = scenario.ev1.target_soc * household.ev1.capacity
        soc_deficit = required_soc - current_soc
        if soc_deficit > 0 and (household.ev1.at_home or household.ev1.at_charging_station):
            # charge to meet soc requirement
            # calc needed avg power to meet deadline
            deadline = scenario.ev1.deadline
            remaining_timesteps = max(deadline - household.current_timestep, 1)  # avoid division by zero
            avg_needed_power = soc_deficit / (remaining_timesteps/4) * (1/household.ev1.efficiency)

            # set charge power
            charge_power = min(avg_needed_power, household.ev1.max_charge)
            household.controls["ev1_power"] = charge_power
    

    # ev2 control
    if household.ev2:
        current_soc = household.ev2.soc
        required_soc = scenario.ev2.target_soc * household.ev2.capacity
        soc_deficit = required_soc - current_soc
        if soc_deficit > 0 and (household.ev2.at_home or household.ev2.at_charging_station):
            # charge to meet soc requirement
            # calc needed avg power to meet deadline
            deadline = scenario.ev2.deadline
            remaining_timesteps = max(deadline - household.current_timestep, 1)  # avoid division by zero
            avg_needed_power = soc_deficit / (remaining_timesteps/4) * (1/household.ev2.efficiency)

            # set charge power
            charge_power = min(avg_needed_power, household.ev2.max_charge)
            household.controls["ev2_power"] = charge_power


    # bess control - we do this last to 
    if household.bess:
        current_soc = household.bess.soc
        required_soc = scenario.bess.target_soc * household.bess.capacity
        soc_deficit = required_soc - current_soc
        if soc_deficit > 0:
            # charge to meet soc requirement
            # calc needed avg power to meet deadline
            deadline = scenario.bess.deadline
            remaining_timesteps = max(deadline - household.current_timestep, 1)  # avoid division by zero
            avg_needed_power = soc_deficit / (remaining_timesteps/4) * (1/household.bess.efficiency)

            # set charge power based on net load
            charge_power = min(avg_needed_power, household.bess.max_charge)
            household.controls["bess_power"] = charge_power

            # use up pv surplus to charge bess
            if household.net_load < 0:
                surplus_power = -household.net_load
                charge_power = min(surplus_power + charge_power, household.bess.max_charge)
                household.controls["bess_power"] = charge_power
        elif soc_deficit <= 0 and household.net_load > 0:
            # discharge to cover load
            discharge_power = min(household.net_load, household.bess.max_discharge, current_soc * 4)
            household.controls["bess_power"] = -discharge_power

    controls = household.controls

    return household.controls


def last_minute_satisfaction(household:Household, scenario:Scenario):
    """
    Last-minute strategy: do nothing until the last possible timestep when charging
    at full power is required to meet the target by the deadline. When the obligation
    window opens, set charge to the device's `max_charge`. PV surplus is allocated
    first to the BESS, then to EVs (ev1 then ev2).
    """
    # initialize explicit zero-controls for clarity
    household.controls.setdefault("ev1_power", 0.0)
    household.controls.setdefault("ev2_power", 0.0)
    household.controls.setdefault("bess_power", 0.0)

    # helper to compute last start timestep
    def compute_last_start(current_soc, required_soc, max_charge, efficiency, deadline):
        deficit = required_soc - current_soc
        if deficit <= 0:
            return None  # already satisfied
        # energy provided per timestep at full power (kWh)
        per_step_kwh = max_charge * 0.25 * efficiency
        if per_step_kwh <= 0:
            return None
        needed_steps = math.ceil(deficit / per_step_kwh)
        last_start = deadline - needed_steps
        return last_start

    # EV1
    if household.ev1:
        current_soc = household.ev1.soc
        required_soc = scenario.ev1.target_soc * household.ev1.capacity
        last_start = compute_last_start(current_soc, required_soc, household.ev1.max_charge, household.ev1.efficiency, scenario.ev1.deadline)
        if last_start is not None and (household.ev1.at_home or household.ev1.at_charging_station):
            if household.current_timestep >= last_start:
                household.controls["ev1_power"] = household.ev1.max_charge
            else:
                household.controls["ev1_power"] = 0.0

    # EV2
    if household.ev2:
        current_soc = household.ev2.soc
        required_soc = scenario.ev2.target_soc * household.ev2.capacity
        last_start = compute_last_start(current_soc, required_soc, household.ev2.max_charge, household.ev2.efficiency, scenario.ev2.deadline)
        if last_start is not None and (household.ev2.at_home or household.ev2.at_charging_station):
            if household.current_timestep >= last_start:
                household.controls["ev2_power"] = household.ev2.max_charge
            else:
                household.controls["ev2_power"] = 0.0

    # BESS
    if household.bess:
        current_soc = household.bess.soc
        required_soc = scenario.bess.target_soc * household.bess.capacity
        last_start = compute_last_start(current_soc, required_soc, household.bess.max_charge, household.bess.efficiency, scenario.bess.deadline)
        if last_start is not None:
            if household.current_timestep >= last_start:
                # charge at full power
                household.controls["bess_power"] = household.bess.max_charge
            else:
                household.controls["bess_power"] = 0.0

    # PV surplus distribution: if net_load < 0 (surplus), allocate to BESS first, then EVs
    if getattr(household, 'net_load', None) is not None and household.net_load < 0:
        surplus = -household.net_load
        # allocate to BESS
        if household.bess:
            # available headroom for bess (kW)
            current_bess_ctrl = household.controls.get("bess_power", 0.0)
            # headroom = max_charge - current_control
            headroom = max(0, household.bess.max_charge - max(0, current_bess_ctrl))
            allocate = min(surplus, headroom)
            if allocate > 0:
                household.controls["bess_power"] = current_bess_ctrl + allocate
                surplus -= allocate

        # allocate remaining surplus to EVs (ev1 then ev2)
        for ev_key, ev in (("ev1_power", household.ev1), ("ev2_power", household.ev2)):
            if surplus <= 0:
                break
            if ev is None:
                continue
            # only allocate if at home
            if not (ev.at_home):
                continue
            current_ctrl = household.controls.get(ev_key, 0.0)
            headroom = max(0, ev.max_charge - max(0, current_ctrl))
            allocate = min(surplus, headroom)
            if allocate > 0:
                household.controls[ev_key] = current_ctrl + allocate
                surplus -= allocate

    return household.controls


if __name__ == "__main__":
    print("This is the naive_linear_satisfaction policy module.")

    # Example usage
    from src.simulation.scenarios.scenario import default_scenario
    from src.simulation.main import init_simulation
    simulation = init_simulation()

    simulation.run_household(1, policy=naive_linear_satisfaction, start_time=0)
    simulation.run_household(1, policy=last_minute_satisfaction, start_time=0)

    print("Finished running example households.")

    