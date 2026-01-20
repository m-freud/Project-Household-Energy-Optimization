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
    base_load = household.base_load
    adjusted_base_load = base_load - pv_generation

    if adjusted_base_load < 0 and household.bess:
        excess = -adjusted_base_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        controls["bess_power"] = charge_power  # positive for charging
    elif adjusted_base_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(adjusted_base_load, household.bess.max_discharge, household.bess.soc)
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

    adjusted_base_load = base_load + ev_load - pv_generation

    if adjusted_base_load < 0 and household.bess:
        excess = -adjusted_base_load
        # charge BESS with excess PV, up to max charge rate
        charge_power = min(excess, household.bess.max_charge)
        controls["bess_power"] = charge_power  # positive for charging
    elif adjusted_base_load > 0 and household.bess and household.bess.soc > 0:
        # discharge BESS to cover deficit, up to max discharge rate
        discharge_power = min(adjusted_base_load, household.bess.max_discharge, household.bess.soc)
        controls["bess_power"] = -discharge_power  # negative for discharging

    return controls


def advanced_ev_bess(household:Household):
    """
    Advanced greedy policy with constraint-aware decision making.
    
    Hierarchy:
    1. Feasibility Check - verify requirements can be met
    2. Constraint Satisfaction - urgent actions to meet requirements
    3. Opportunistic Optimization - price-aware charging/discharging
    4. Surplus Distribution - utilize excess PV generation
    """
    
    buy_prices = household.buy_price_day_profile
    sell_prices = household.sell_price_day_profile
    base_load = household.base_load
    pv_generation = household.pv.generation if household.pv else 0.0
    requirements = household.charge_requirements
    current_timestep = household.current_timestep

    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0
    }

    # 1. feasibilty check, urgency scoring
    urgency_scores = {}
    
    # Check BESS
    if household.bess and 'bess' in requirements:
        req = requirements['bess']
        deadline = req.get('timestep', 96)
        required_soc = req.get('soc', 0.0) * household.bess.capacity
        current_soc = household.bess.soc
        energy_deficit = max(0, required_soc - current_soc)
        timesteps_remaining = max(1, deadline - current_timestep)
        
        # Calculate minimum charge rate needed per timestep (kWh per 15min)
        min_charge_needed_per_step = energy_deficit / timesteps_remaining
        # Convert to power (kW) - multiply by 4 since timestep is 15min
        min_power_needed = min_charge_needed_per_step * 4
        
        # Urgency: ratio of required power to max power (0 = easy, >1 = urgent/infeasible)
        urgency = min_power_needed / household.bess.max_charge if household.bess.max_charge > 0 else 0
        urgency_scores['bess'] = {
            'urgency': urgency,
            'energy_deficit': energy_deficit,
            'timesteps_remaining': timesteps_remaining,
            'min_power_needed': min_power_needed
        }
    
    # Check EVs
    for ev_name in ['ev1', 'ev2']:
        ev = getattr(household, ev_name, None)
        if ev and ev.at_home and ev_name in requirements:
            req = requirements[ev_name]
            deadline = req.get('timestep', 96)
            required_soc = req.get('soc', 0.0) * ev.capacity
            current_soc = ev.soc
            energy_deficit = max(0, required_soc - current_soc)
            timesteps_remaining = max(1, deadline - current_timestep)
            
            min_charge_needed_per_step = energy_deficit / timesteps_remaining
            min_power_needed = min_charge_needed_per_step * 4 / ev.efficiency
            
            urgency = min_power_needed / ev.max_charge if ev.max_charge > 0 else 0
            urgency_scores[ev_name] = {
                'urgency': urgency,
                'energy_deficit': energy_deficit,
                'timesteps_remaining': timesteps_remaining,
                'min_power_needed': min_power_needed
            }

    
    # constraint satisfaction -> urgent actions
    # Urgency threshold: if urgency > 0.7, we're cutting it close, must charge now
    URGENCY_THRESHOLD = 0.7
    
    for asset_name, scores in urgency_scores.items():
        if scores['urgency'] > URGENCY_THRESHOLD:
            if asset_name == 'bess' and household.bess:
                # Urgent BESS charging needed
                charge_power = min(
                    household.bess.max_charge,
                    scores['min_power_needed']
                )
                controls['bess_power'] = max(controls['bess_power'], charge_power)
                
            elif asset_name in ['ev1', 'ev2']:
                ev = getattr(household, asset_name)
                charge_power = min(
                    ev.max_charge,
                    scores['min_power_needed']
                )
                controls[f'{asset_name}_power'] = max(controls[f'{asset_name}_power'], charge_power)

    # 3. opportunistic optimization - price-aware decisions
    
    # Calculate average future prices for comparison
    remaining_prices = buy_prices[current_timestep:]
    avg_future_price = sum(remaining_prices) / len(remaining_prices) if remaining_prices else buy_prices[current_timestep]
    current_price = buy_prices[current_timestep]
    current_sell_price = sell_prices[current_timestep]
    
    # Price advantage: positive means good time to charge, negative means good time to discharge
    price_advantage = (avg_future_price - current_price) / avg_future_price if avg_future_price > 0 else 0
    
    # If current price is significantly below average and we haven't hit urgent charging
    if price_advantage > 0.15:  # >15% cheaper than average
        # Good time to charge (if not already urgently charging)
        
        # Charge BESS if below requirement and not urgent
        if household.bess and 'bess' in urgency_scores:
            if urgency_scores['bess']['urgency'] < URGENCY_THRESHOLD and urgency_scores['bess']['energy_deficit'] > 0:
                charge_power = household.bess.max_charge
                controls['bess_power'] = max(controls['bess_power'], charge_power)
        
        # Charge EVs if below requirement and not urgent
        for ev_name in ['ev1', 'ev2']:
            if ev_name in urgency_scores:
                if urgency_scores[ev_name]['urgency'] < URGENCY_THRESHOLD and urgency_scores[ev_name]['energy_deficit'] > 0:
                    ev = getattr(household, ev_name)
                    charge_power = ev.max_charge
                    controls[f'{ev_name}_power'] = max(controls[f'{ev_name}_power'], charge_power)
    
    # If current price is significantly above average, consider discharging
    elif price_advantage < -0.15:  # >15% more expensive than average
        # Good time to discharge BESS (if we have slack above requirement)
        if household.bess and 'bess' in requirements:
            req_soc = requirements['bess'].get('soc', 0.0) * household.bess.capacity
            slack = household.bess.soc - req_soc
            
            if slack > 0:
                # Discharge excess to cover base load or sell to grid
                discharge_power = min(
                    household.bess.max_discharge,
                    slack * 4,  # Convert kWh to kW for 15min
                    base_load - pv_generation  # Don't discharge more than needed
                )
                if discharge_power > 0:
                    controls['bess_power'] = -discharge_power

    # 4. surplus distribution - utilize excess PV generation
    # Calculate net load after control decisions
    ev_load = 0
    if household.ev1 and household.ev1.at_home:
        ev_load += controls['ev1_power']
    if household.ev2 and household.ev2.at_home:
        ev_load += controls['ev2_power']
    
    bess_load = controls['bess_power'] if controls['bess_power'] > 0 else 0  # Only count charging
    
    adjusted_base_load = base_load + ev_load + bess_load - pv_generation
    
    # If we have surplus PV after all decisions
    if adjusted_base_load < 0:
        surplus = -adjusted_base_load
        
        # Priority 1: Top up EVs not yet at target
        for ev_name in ['ev1', 'ev2']:
            ev = getattr(household, ev_name, None)
            if ev and ev.at_home and surplus > 0:
                req_soc = requirements.get(ev_name, {}).get('soc', 1.0) * ev.capacity
                if ev.soc < req_soc:
                    additional_charge = min(
                        ev.max_charge - controls[f'{ev_name}_power'],  # Room left in charge rate
                        (req_soc - ev.soc) * 4 / ev.efficiency,  # Energy needed as power
                        surplus * 4  # Available surplus as power
                    )
                    if additional_charge > 0:
                        controls[f'{ev_name}_power'] += additional_charge
                        surplus -= additional_charge / 4
        
        # Priority 2: Top up BESS not yet at target
        if household.bess and surplus > 0:
            req_soc = requirements.get('bess', {}).get('soc', 1.0) * household.bess.capacity
            if household.bess.soc < req_soc:
                additional_charge = min(
                    household.bess.max_charge - controls['bess_power'],
                    (req_soc - household.bess.soc) * 4,
                    surplus
                )
                if additional_charge > 0:
                    controls['bess_power'] += additional_charge
        
        # Any remaining surplus exports to grid (passive, handled by simulation)
    
    return controls
