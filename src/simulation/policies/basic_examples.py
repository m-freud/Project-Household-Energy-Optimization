'''
Docstring for src.simulation.policies.rule_based
'''

from src.simulation.household import Household


def basic_bess(household:Household):
    '''
    if there is excess PV generation, charge the BESS
    if there is a deficit, discharge the BESS
    '''
    pv_generation = household.pv.generation if household.pv else 0.0
    base_load = household.base_load
    adjusted_base_load = base_load - pv_generation

    if adjusted_base_load < 0 and household.bess:
        excess = -adjusted_base_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        household.controls["bess_power"] = charge_power  # positive for charging
    elif adjusted_base_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(adjusted_base_load, household.bess.max_discharge, household.bess.soc * 4)
        household.controls["bess_power"] = -discharge_power  # negative for discharging
    return household.controls


def basic_ev(household:Household):
    '''
    Simple EV charging policy:
    at charging station: charge if cheaper than at home so far
    at home: charge if cheaper than at charging station, or if pv excess generation
    '''
    
    # EV1
    if household.ev1:
        if household.ev1.at_charging_station:
            # charge if buy price at charging station is lower than at home
            if household.ev1.buy_price < household.buy_price:
                household.controls["ev1_power"] = min(
                    household.ev1.max_charge,
                    (household.ev1.capacity - household.ev1.soc)*4 / household.ev1.efficiency)
        elif household.ev1.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev1.buy_price < household.ev1.buy_price:
                household.controls["ev1_power"] = min(household.ev1.max_charge, (household.ev1.capacity - household.ev1.soc)*4 / household.ev1.efficiency )
    # EV2
    if household.ev2:
        if household.ev2.at_charging_station:
            # charge if buy price at charging station is lower than at home
            if household.ev2.buy_price < household.buy_price:
                household.controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc)*4 / household.ev2.efficiency)
        elif household.ev2.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev2.buy_price < household.ev2.buy_price:
                household.controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc)*4 / household.ev2.efficiency)
                
    return household.controls


def basic_ev_bess(household:Household):
    '''
    Combined policy based on priority:
    charge ev first, then battery
    '''
    controls = basic_ev(household)

    if not household.bess:
        return controls

    # subtract ev charging power from bess power if both are charging
    pv_generation = household.pv.generation if household.pv else 0.0
    base_load = household.base_load

    ev_load = 0
    if household.ev1 and household.ev1.at_home:
        ev_load += controls["ev1_power"]
    if household.ev2 and household.ev2.at_home:
        ev_load += controls["ev2_power"]

    adjusted_base_load = base_load + ev_load - pv_generation

    if adjusted_base_load < 0:
        excess = -adjusted_base_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        controls["bess_power"] = charge_power  # positive for charging
    elif adjusted_base_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(adjusted_base_load, household.bess.max_discharge, household.bess.soc * 4)
        controls["bess_power"] = -discharge_power  # negative for discharging

    return controls


