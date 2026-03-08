# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.config import Config


def even_linear_policy(
        household: Household,
):
    scenario = household.scenario

    controls = {
        "ev1_power": 0.0,
        "ev2_power": 0.0,
        "bess_power": 0.0,
    }
    
    # EV1, EV2
    for ev in [household.ev1, household.ev2]:
        if ev and (ev.at_home or ev.at_charging_station):
            current_soc = ev.soc

            if ev == household.ev1:
                target_soc = scenario.ev1.target_soc
                deadline = scenario.ev1.deadline
            else:
                target_soc = scenario.ev2.target_soc
                deadline = scenario.ev2.deadline
            
            target_soc = scenario.ev1.target_soc if ev == household.ev1 else scenario.ev2.target_soc
            
            target_soc = target_soc * ev.capacity if target_soc <= 1.0 else target_soc

            soc_deficit = target_soc - current_soc
            max_charge = ev.max_charge
            efficiency = ev.efficiency
            current_timestep = household.current_timestep
            deadline = scenario.ev1.deadline if ev == household.ev1 else scenario.ev2.deadline
            remaining_time = max(deadline - current_timestep, 0)

            if soc_deficit > 0:
                controls[f"{ev.name}_power"] = min((soc_deficit / (remaining_time / Config.DURATION_TIMESTEP)) / efficiency, max_charge)


    # BESS
    if household.bess:
        current_soc = household.bess.soc
        target_soc = scenario.bess.target_soc * household.bess.capacity if scenario.bess.target_soc <= 1.0 else scenario.bess.target_soc
        soc_deficit = target_soc - current_soc
        max_charge = household.bess.max_charge
        efficiency = household.bess.efficiency
        current_timestep = household.current_timestep
        deadline = scenario.bess.deadline
        remaining_time = max(deadline - current_timestep, 0)

        if soc_deficit > 0: # 
            base_charge_power = min((soc_deficit / (remaining_time / Config.DURATION_TIMESTEP)) / efficiency, max_charge)
            surplus_power = -household.net_load
            charge_power = max(base_charge_power, min(base_charge_power + surplus_power, max_charge))
            controls["bess_power"] = charge_power

        elif household.net_load > 0:
            discharge_power = min(
                household.net_load,
                household.bess.max_discharge,
                current_soc * 4 * household.bess.capacity, # max discharge based on current soc
            )
            controls["bess_power"] = -discharge_power

    return controls



def fast_charge_policy(
        household: Household,
):
    controls = {
        "ev1_power": 0.0,
        "ev2_power": 0.0,
        "bess_power": 0.0,
    }

    scenario = household.scenario

    if household.ev1 and (household.ev1.at_home or household.ev1.at_charging_station):
        ev1_target_soc = scenario.ev1.target_soc
        ev1_target_soc = ev1_target_soc * household.ev1.capacity if ev1_target_soc <= 1.0 else ev1_target_soc
        if household.ev1.soc < ev1_target_soc:
            controls["ev1_power"] = household.ev1.max_charge

    if household.ev2 and (household.ev2.at_home or household.ev2.at_charging_station):
        ev2_target_soc = scenario.ev2.target_soc
        ev2_target_soc = ev2_target_soc * household.ev2.capacity if ev2_target_soc <= 1.0 else ev2_target_soc
        if household.ev2.soc < ev2_target_soc:
            controls["ev2_power"] = household.ev2.max_charge

    if household.bess:
        bess_target_soc = scenario.bess.target_soc
        bess_target_soc = bess_target_soc * household.bess.capacity if bess_target_soc <= 1.0 else bess_target_soc
        if household.bess.soc < bess_target_soc:
            controls["bess_power"] = household.bess.max_charge

    return controls
