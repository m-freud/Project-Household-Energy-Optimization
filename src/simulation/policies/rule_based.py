from src.simulation.Household import Household

# here we add increasingly complex policies, depending on more and more inputs.
# once we use all inputs we can start adding learning-based policies, and genetic algorithms if feasible (probably not. but at some point you know)

# so we have these inputs to work with:
'''        
self.env_inputs = [
            "load",
            "pv_gen",
            "ev1_load",
            "ev2_load",
            "buy_price",
            "sell_price",
            "ev1_buy_price",
            "ev2_buy_price",
            "ev1_at_home",
            "ev2_at_home",
            "ev1_at_charging_station",
            "ev2_at_charging_station"
]'''
# including current state and history for each parameter you can think of.

# we can also expand them to get 
'''
 net load,
pv generation forecast,
buy price forecast,
...

'''
# the policy can theoretically use all these inputs and more to decide:
# bess_action, ev1_action, ev2_action
# where each action is a power value (positive for discharge, negative for charge)

# we can start with simple rule-based policies, then move to optimization-based policies (MPC), and finally learning-based policies (RL, GA, etc.)

# Example of a simple rule-based policy
def basic_battery(household:Household, t=0):
    '''
    if there is excess PV generation, charge the BESS
    if there is a deficit, discharge the BESS

    policies are used on updated households.
    any relevant info should be in the household object already.
    including history if needed.
    just access everything in the household.
    '''
    # init controls
    controls = { # controls tell you how much power to put into the device. so + charge, - discharge. the EVs can only charge.
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0,
    }

    pv_generation = household.pv.generation if household.pv else 0
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


def basic_ev_charging(household:Household, t=0):
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
                controls["ev1_power"] = min(household.ev1.max_charge, (household.ev1.capacity - household.ev1.soc) / household.ev1.efficiency)
        elif household.ev1.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev1.buy_price < household.ev1.buy_price:
                controls["ev1_power"] = min(household.ev1.max_charge, (household.ev1.capacity - household.ev1.soc) / household.ev1.efficiency)

    # EV2
    if household.ev2:
        if household.ev2.at_charging_station:
            # charge if buy price at charging station is lower than at home
            if household.ev2.buy_price < household.buy_price:
                controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc) / household.ev2.efficiency)
        elif household.ev2.at_home:
            # charge if buy price at home is lower than at charging station
            if household.ev2.buy_price < household.ev2.buy_price:
                controls["ev2_power"] = min(household.ev2.max_charge, (household.ev2.capacity - household.ev2.soc) / household.ev2.efficiency)

    return controls


def basic_ev_bess(household:Household, t=0):
    '''
    Combined policy based on priority:
    charge ev first, then battery
    '''
    controls = basic_ev_charging(household)

    # subtract ev charging power from bess power if both are charging
    pv_generation = household.pv.generation if household.pv else 0
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
