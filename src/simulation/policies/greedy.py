'''
Docstring for src.simulation.policies.rule_based
'''

from src.simulation.household import Household


def basic_bess(household:Household):
    '''
    if there is excess PV generation, charge the BESS
    if there is a deficit, discharge the BESS
    '''
    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0,
    }

    pv_generation = household.pv.generation if household.pv else 0.0
    load = household.base_load
    net_load = load - pv_generation

    if net_load < 0 and household.bess:
        excess = -net_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        controls["bess_power"] = charge_power  # positive for charging
    elif net_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(net_load, household.bess.max_discharge, household.bess.soc)
        controls["bess_power"] = -discharge_power  # negative for discharging

    return controls


def basic_ev(household:Household):
    '''
    Simple EV charging policy:
    at charging station: charge if cheaper than at home so far
    at home: charge if cheaper than at charging station, or if pv excess generation
    '''

    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0,
    }
    # EV1
    if household.ev1:
        if household.ev1.at_charging_station:
            # charge if buy price at charging station is lower than at home
            if household.ev1.buy_price < household.buy_price:
                controls["ev1_power"] = min(
                    household.ev1.max_charge,
                    (household.ev1.capacity - household.ev1.soc)*4 / household.ev1.efficiency)
        elif household.ev1.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev1.buy_price < household.ev1.buy_price:
                controls["ev1_power"] = min(household.ev1.max_charge, (household.ev1.capacity - household.ev1.soc)*4 / household.ev1.efficiency )
    # EV2
    if household.ev2:
        if household.ev2.at_charging_station:
            # charge if buy price at charging station is lower than at home
            if household.ev2.buy_price < household.buy_price:
                controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc)*4 / household.ev2.efficiency)
        elif household.ev2.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev2.buy_price < household.ev2.buy_price:
                controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc)*4 / household.ev2.efficiency)
                
    return controls


def basic_ev_bess(household:Household):
    '''
    Combined policy based on priority:
    charge ev first, then battery
    '''
    controls = basic_ev(household)

    # subtract ev charging power from bess power if both are charging
    pv_generation = household.pv.generation if household.pv else 0.0
    base_load = household.base_load

    ev_load = 0
    if household.ev1 and household.ev1.at_home:
        ev_load += controls["ev1_power"]
    if household.ev2 and household.ev2.at_home:
        ev_load += controls["ev2_power"]

    net_load = base_load + ev_load - pv_generation

    if net_load < 0 and household.bess:
        excess = -net_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        controls["bess_power"] = charge_power  # positive for charging
    elif net_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(net_load, household.bess.max_discharge, household.bess.soc)
        controls["bess_power"] = -discharge_power  # negative for discharging

    return controls


def advanced_ev_bess(household:Household):
    buy_prices = household.buy_price_day_profile
    sell_prices = household.sell_price_day_profile
    base_load = household.base_load
    pv_generation = household.pv.generation if household.pv else 0.0
    requirements = household.charge_requirements

    net_load = base_load - pv_generation

    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0
    }

    # decision tree time!
    # goal: minimize consumption of the day.
    # requirements: BESS SOC 50% by 96, etc
    # so for any t: do what you have to do for the requirements while minimizing consumption
    # have to? absolutely have to vs just a good time:
    # do stuff when it makes sense. unless u have to. then do it because you have to.
    # decision tree to walk down the priority ladder
    
    # distribute suprlus first
    if net_load < 0:
        surplus = -net_load

        # 1. charge evs first
        evs_at_home = []
        if household.ev1 and household.ev1.at_home:
            evs_at_home.append('ev1')
        if household.ev2 and household.ev2.at_home:
            evs_at_home.append('ev2')

        for ev in evs_at_home:
            if ev.soc < requirements[ev]["soc"] * ev.capacity:
                ev_power_available = surplus / len(evs_at_home)
                ev_max_charge_power = min(
                    ev.max_charge, # max charge rate if you need more than one step
                    (ev.capacity - ev.soc) * 4 / ev.efficiency, # this is what is needed to get to 100% in one step.
                    ev_power_available * 4 # distribute surplus evenly. 2kWh ? you can charge 8 kW for 15min
                )
                controls[f"{ev}_power"] = ev_max_charge_power
                surplus -= ev_max_charge_power

        # 2. charge bess if any surplus left
        if surplus > 0 and household.bess:  
            bess = household.bess
            charge_power = min(surplus, bess.max_charge)
            controls["bess_power"] = charge_power  # positive for charging
    
    return controls
