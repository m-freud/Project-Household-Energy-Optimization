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
from src.config import Config

def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _latest_possible_start_time(current_soc: float,
                                target_soc: float,
                                max_charge: float,
                                efficiency: float,
                                deadline: int) -> int:
    soc_deficit = target_soc - current_soc
    if soc_deficit <= 0:
        return 0

    per_step_kwh = max_charge * Config.DURATION_TIMESTEP * max(efficiency, 1e-9)
    if per_step_kwh <= 0:
        return 0

    needed_steps = int((soc_deficit / per_step_kwh) + 0.999999)
    return max(int(deadline) - needed_steps, 0)


def _required_charge_power(current_soc: float,
                           target_soc: float,
                           max_charge: float,
                           efficiency: float,
                           current_timestep: int,
                           deadline: int,
                           urgency: float,
                           delay: float) -> float:
    soc_deficit = target_soc - current_soc
    if soc_deficit <= 0:
        return 0.0

    latest_possible_start_time = _latest_possible_start_time(
        current_soc=current_soc,
        target_soc=target_soc,
        max_charge=max_charge,
        efficiency=efficiency,
        deadline=deadline,
    )

    control_start_time = int(round(latest_possible_start_time * delay))
    if current_timestep < control_start_time:
        return 0.0

    effective_deadline = int(round((1.0 - urgency) * deadline))
    if current_timestep >= effective_deadline:
        return max_charge

    remaining_timesteps = max(effective_deadline - current_timestep, 1)
    safe_efficiency = max(efficiency, 1e-9)

    avg_needed_power = soc_deficit / (remaining_timesteps * Config.DURATION_TIMESTEP) * (1.0 / safe_efficiency)
    return min(max(avg_needed_power, 0.0), max_charge) # atleast 0 but at most max_charge


def naive_linear_policy(household: Household,
                        urgency: float = 0.5,
                        delay: float = 0.5):
    '''
    Urgency-weighted linear charging policy.

    urgency in [0, 1]: moves deadline to (1 - urgency) * actual_deadline.
    delay in [0, 1]: moves control start to latest_possible_start_time * delay.

    Each device uses its own deadline from scenario, while sharing the same urgency.
    '''
    urgency = _clamp_01(urgency)
    delay = _clamp_01(delay)

    controls = {
        "ev1_power": 0.0,
        "ev2_power": 0.0,
        "bess_power": 0.0,
    }

    scenario = household.scenario

    if household.ev1 and (household.ev1.at_home or household.ev1.at_charging_station):
        controls["ev1_power"] = _required_charge_power(
            current_soc=household.ev1.soc,
            target_soc=scenario.ev1.target_soc * household.ev1.capacity if scenario.ev1.target_soc <= 1.0 else scenario.ev1.target_soc,
            max_charge=household.ev1.max_charge,
            efficiency=household.ev1.efficiency,
            current_timestep=household.current_timestep,
            deadline=scenario.ev1.deadline,
            urgency=urgency,
            delay=delay,
        )

    if household.ev2 and (household.ev2.at_home or household.ev2.at_charging_station):
        controls["ev2_power"] = _required_charge_power(
            current_soc=household.ev2.soc,
            target_soc=scenario.ev2.target_soc * household.ev2.capacity if scenario.ev2.target_soc <= 1.0 else scenario.ev2.target_soc,
            max_charge=household.ev2.max_charge,
            efficiency=household.ev2.efficiency,
            current_timestep=household.current_timestep,
            deadline=scenario.ev2.deadline,
            urgency=urgency,
            delay=delay,
        )

    if household.bess:
        current_soc = household.bess.soc
        target_soc = scenario.bess.target_soc * household.bess.capacity if scenario.bess.target_soc <= 1.0 else scenario.bess.target_soc
        soc_deficit = target_soc - current_soc

        if soc_deficit > 0:
            base_charge = _required_charge_power(
                current_soc=current_soc,
                target_soc=target_soc,
                max_charge=household.bess.max_charge,
                efficiency=household.bess.efficiency,
                current_timestep=household.current_timestep,
                deadline=scenario.bess.deadline,
                urgency=urgency,
                delay=delay,
            )

            if household.net_load < 0:
                surplus_power = -household.net_load
                base_charge = min(base_charge + surplus_power, household.bess.max_charge)

            controls["bess_power"] = base_charge

        elif household.net_load > 0:
            discharge_power = min(
                household.net_load,
                household.bess.max_discharge,
                current_soc * 4 * household.bess.capacity, # max discharge based on current soc
            )
            controls["bess_power"] = -discharge_power

    return controls


def make_linear_policy(urgency: float = 0.5, delay: float = 0.5):
    '''
    Factory that returns a policy callable with fixed urgency and delay.

    Useful for simulation APIs that expect policy(household, scenario),
    while still encoding urgency and delay in the policy name.
    '''
    urgency = _clamp_01(urgency)
    delay = _clamp_01(delay)

    def policy(household: Household, scenario: Scenario):
        return naive_linear_policy(household, scenario, urgency=urgency, delay=delay)

    policy.__name__ = f"linear_u{urgency:.2f}_d{delay:.2f}".replace(".", "_")
    return policy


def latest_possible_charge_policy(household: Household, scenario: Scenario):
    '''
    Special case of naive_linear_policy with urgency=1.0 and delay=1.0.
    '''
    def policy(household: Household, scenario: Scenario):
        return naive_linear_policy(household, scenario, urgency=1.0, delay=1.0)

    policy.__name__ = f"latest_possible_charge"
    return policy


def earliest_possible_charge_policy(household: Household, scenario: Scenario):
    '''
    Special case of naive_linear_policy with urgency=0.0 and delay=0.0.
    '''

    def policy(household: Household, scenario: Scenario):
        return naive_linear_policy(household, scenario, urgency=1.0, delay=0.0)

    policy.__name__ = f"earliest_possible_charge"
    return policy
