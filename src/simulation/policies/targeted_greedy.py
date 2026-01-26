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
from src.simulation.scenarios.example_scenarios import Scenario


def targeted_greedy(household:Household, scenario:Scenario):
    '''
    greedy policy with targets.
    calculates average needed power to meet target soc by deadline,
    then sets charge power to exactly that amount (capped by max charge rate).
    uses pv surplus to charge bess if available.
    '''

    # ev1 control
    if household.ev1:
        current_soc = household.ev1.soc
        required_soc = scenario.ev1.target_soc
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
        required_soc = scenario.ev2.target_soc
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
        required_soc = scenario.bess.target_soc
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


# todo add price awareness