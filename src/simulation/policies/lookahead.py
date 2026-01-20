"""
Lookahead Greedy Policy

Implements receding horizon optimization with multi-step lookahead.
Unlike myopic greedy (react to current state), this policy:
1. Simulates multiple future timesteps
2. Evaluates action sequences over the lookahead horizon
3. Verifies constraint feasibility before committing
4. Selects actions that minimize total cost while guaranteeing requirements

This represents a professional-grade greedy baseline suitable for comparison
with advanced optimization algorithms (MPC, RL, etc.).
"""

from copy import deepcopy
from src.simulation.household import Household
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from typing import Dict, List, Tuple


class StateProjector:
    """Projects household state forward through time given action sequences."""
    
    @staticmethod
    def project_state(household: Household, actions: List[Dict], horizon: int) -> Dict:
        """
        Simulate household forward for 'horizon' timesteps with given actions.
        
        Args:
            household: Current household state
            actions: List of control dictionaries for each timestep
            horizon: Number of timesteps to simulate
            
        Returns:
            Dict with projected costs, final SOCs, and constraint violations
        """
        # Create deep copy to simulate without affecting real state
        h_copy = deepcopy(household)
        
        total_cost = 0.0
        total_consumption = 0.0
        constraint_violations = []
        
        for t in range(min(horizon, len(actions))):
            # Get future prices if available
            future_t = h_copy.current_timestep + t
            if future_t < len(h_copy.buy_price_day_profile):
                buy_price = h_copy.buy_price_day_profile[future_t]
                sell_price = h_copy.sell_price_day_profile[future_t]
            else:
                # Beyond known horizon, use last known price
                buy_price = h_copy.buy_price_day_profile[-1] if h_copy.buy_price_day_profile else 0.0
                sell_price = h_copy.sell_price_day_profile[-1] if h_copy.sell_price_day_profile else 0.0
            
            # Apply controls
            controls = actions[t] if t < len(actions) else {"bess_power": 0, "ev1_power": 0, "ev2_power": 0}
            
            # Simulate BESS
            if h_copy.bess:
                bess_power = controls.get("bess_power", 0)
                if bess_power > 0:
                    h_copy.bess.charge(bess_power, 0.25)
                elif bess_power < 0:
                    h_copy.bess.discharge(-bess_power, 0.25)
            
            # Simulate EVs
            for ev_name in ['ev1', 'ev2']:
                ev = getattr(h_copy, ev_name, None)
                if ev and ev.at_home:
                    ev_power = controls.get(f"{ev_name}_power", 0)
                    if ev_power > 0:
                        ev.charge(ev_power, 0.25)
            
            # Calculate net load and cost for this timestep
            pv_gen = h_copy.pv.generation if h_copy.pv else 0.0
            base_load = h_copy.base_load  # Assumes constant, could be updated with profile
            
            ev_load = 0
            if h_copy.ev1 and h_copy.ev1.at_home:
                ev_load += controls.get("ev1_power", 0)
            if h_copy.ev2 and h_copy.ev2.at_home:
                ev_load += controls.get("ev2_power", 0)
            
            bess_load = controls.get("bess_power", 0) if controls.get("bess_power", 0) > 0 else 0
            bess_discharge = -controls.get("bess_power", 0) if controls.get("bess_power", 0) < 0 else 0
            
            net_load = base_load + ev_load + bess_load - pv_gen - bess_discharge
            
            if net_load > 0:
                step_cost = net_load * 0.25 * buy_price
                total_consumption += net_load * 0.25
            else:
                step_cost = net_load * 0.25 * sell_price
            
            total_cost += step_cost
        
        # Check constraint violations at horizon end
        requirements = h_copy.charge_requirements
        
        if h_copy.bess and 'bess' in requirements:
            req = requirements['bess']
            req_soc = req.get('soc', 0.0) * h_copy.bess.capacity
            if h_copy.bess.soc < req_soc:
                deficit = req_soc - h_copy.bess.soc
                constraint_violations.append(('bess', deficit))
        
        for ev_name in ['ev1', 'ev2']:
            ev = getattr(h_copy, ev_name, None)
            if ev and ev_name in requirements:
                req = requirements[ev_name]
                req_soc = req.get('soc', 0.0) * ev.capacity
                if ev.soc < req_soc:
                    deficit = req_soc - ev.soc
                    constraint_violations.append((ev_name, deficit))
        
        return {
            'total_cost': total_cost,
            'total_consumption': total_consumption,
            'constraint_violations': constraint_violations,
            'final_bess_soc': h_copy.bess.soc if h_copy.bess else 0,
            'final_ev1_soc': h_copy.ev1.soc if h_copy.ev1 else 0,
            'final_ev2_soc': h_copy.ev2.soc if h_copy.ev2 else 0,
        }


class ActionGenerator:
    """Generates candidate action sequences for evaluation."""
    
    @staticmethod
    def generate_action_candidates(household: Household) -> List[Dict]:
        """
        Generate discrete action candidates for current timestep.
        
        Returns list of candidate control dictionaries.
        """
        candidates = []
        
        # Baseline: do nothing
        candidates.append({"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0})
        
        # BESS actions
        if household.bess:
            # Max charge
            candidates.append({"bess_power": household.bess.max_charge, "ev1_power": 0.0, "ev2_power": 0.0})
            # Half charge
            candidates.append({"bess_power": household.bess.max_charge * 0.5, "ev1_power": 0.0, "ev2_power": 0.0})
            # Max discharge
            candidates.append({"bess_power": -household.bess.max_discharge, "ev1_power": 0.0, "ev2_power": 0.0})
            # Half discharge
            candidates.append({"bess_power": -household.bess.max_discharge * 0.5, "ev1_power": 0.0, "ev2_power": 0.0})
        
        # EV actions
        for ev_name in ['ev1', 'ev2']:
            ev = getattr(household, ev_name, None)
            if ev and ev.at_home:
                # Max charge this EV
                action = {"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0}
                action[f"{ev_name}_power"] = ev.max_charge
                candidates.append(action)
                
                # Half charge this EV
                action = {"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0}
                action[f"{ev_name}_power"] = ev.max_charge * 0.5
                candidates.append(action)
        
        # Combined actions: charge both EVs
        if household.ev1 and household.ev1.at_home and household.ev2 and household.ev2.at_home:
            candidates.append({
                "bess_power": 0.0,
                "ev1_power": household.ev1.max_charge,
                "ev2_power": household.ev2.max_charge
            })
        
        # Combined: charge EVs + BESS
        if household.bess and household.ev1 and household.ev1.at_home:
            candidates.append({
                "bess_power": household.bess.max_charge,
                "ev1_power": household.ev1.max_charge,
                "ev2_power": 0.0
            })
        
        return candidates


def calculate_urgency_score(household: Household) -> Dict[str, float]:
    """
    Calculate urgency score for each asset based on time to deadline.
    
    Returns dict mapping asset name to urgency (0=plenty of time, 1=cutting it close, >1=infeasible)
    """
    urgency = {}
    requirements = household.charge_requirements
    current_t = household.current_timestep
    
    # BESS urgency
    if household.bess and 'bess' in requirements:
        req = requirements['bess']
        deadline = req.get('timestep', 96)
        required_soc = req.get('soc', 0.0) * household.bess.capacity
        current_soc = household.bess.soc
        
        energy_deficit = max(0, required_soc - current_soc)
        timesteps_remaining = max(1, deadline - current_t)
        
        # Energy needed per timestep
        energy_per_step = energy_deficit / timesteps_remaining
        # Convert to power (kW): multiply by 4 since timestep is 15min
        power_needed = energy_per_step * 4
        
        urgency['bess'] = power_needed / household.bess.max_charge if household.bess.max_charge > 0 else 0
    
    # EV urgency
    for ev_name in ['ev1', 'ev2']:
        ev = getattr(household, ev_name, None)
        if ev and ev_name in requirements:
            req = requirements[ev_name]
            deadline = req.get('timestep', 96)
            required_soc = req.get('soc', 0.0) * ev.capacity
            current_soc = ev.soc
            
            energy_deficit = max(0, required_soc - current_soc)
            timesteps_remaining = max(1, deadline - current_t)
            
            energy_per_step = energy_deficit / timesteps_remaining
            power_needed = energy_per_step * 4 / ev.efficiency
            
            urgency[ev_name] = power_needed / ev.max_charge if ev.max_charge > 0 else 0
    
    return urgency


def lookahead_greedy(household: Household, lookahead_horizon: int = 16) -> Dict:
    """
    Lookahead greedy policy with receding horizon optimization.
    
    Strategy:
    1. Calculate urgency scores for all requirements
    2. If highly urgent (urgency > threshold), force immediate charging
    3. Otherwise, evaluate action candidates over lookahead horizon
    4. Select action that minimizes: cost + penalty for constraint violations
    5. Only execute first action (receding horizon principle)
    
    Args:
        household: Current household state
        lookahead_horizon: Number of timesteps to look ahead (default 16 = 4 hours)
        
    Returns:
        Control dictionary with power setpoints
    """
    
    # ========================================================================
    # 1. URGENCY CHECK - Handle critical constraints immediately
    # ========================================================================
    urgency_scores = calculate_urgency_score(household)
    CRITICAL_URGENCY = 0.8  # If urgency > 0.8, must act now
    
    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0
    }
    
    critical_action_taken = False
    
    # If any asset is critically urgent, charge it immediately at max rate
    for asset_name, urgency in urgency_scores.items():
        if urgency > CRITICAL_URGENCY:
            critical_action_taken = True
            if asset_name == 'bess' and household.bess:
                controls['bess_power'] = household.bess.max_charge
            elif asset_name in ['ev1', 'ev2']:
                ev = getattr(household, asset_name)
                if ev.at_home:
                    controls[f'{asset_name}_power'] = ev.max_charge
    
    # If critical action was taken, return immediately (safety first)
    if critical_action_taken:
        return controls
    
    # ========================================================================
    # 2. LOOKAHEAD OPTIMIZATION - Find best action over horizon
    # ========================================================================
    
    # Generate candidate actions for current timestep
    candidates = ActionGenerator.generate_action_candidates(household)
    
    # Evaluate each candidate by simulating forward
    best_action = None
    best_score = float('inf')
    
    CONSTRAINT_PENALTY = 1000.0  # Heavy penalty for missing requirements
    
    for candidate in candidates:
        # For simplicity, assume same action repeated over horizon
        # (More sophisticated: generate action sequences, but that's exponential)
        action_sequence = [candidate] * lookahead_horizon
        
        # Project state forward
        projection = StateProjector.project_state(household, action_sequence, lookahead_horizon)
        
        # Calculate score: cost + penalties
        cost = projection['total_cost']
        
        # Penalty for constraint violations
        violation_penalty = 0.0
        for asset, deficit in projection['constraint_violations']:
            violation_penalty += deficit * CONSTRAINT_PENALTY
        
        total_score = cost + violation_penalty
        
        # Select best
        if total_score < best_score:
            best_score = total_score
            best_action = candidate
    
    # ========================================================================
    # 3. RETURN BEST ACTION (Receding Horizon: only execute first step)
    # ========================================================================
    
    if best_action is not None:
        controls = best_action
    
    return controls


def lookahead_greedy_adaptive(household: Household) -> Dict:
    """
    Adaptive lookahead that adjusts horizon based on urgency and time remaining.
    
    - High urgency: short horizon (focus on immediate needs)
    - Low urgency: long horizon (optimize over longer period)
    """
    urgency_scores = calculate_urgency_score(household)
    max_urgency = max(urgency_scores.values()) if urgency_scores else 0.0
    
    # Adapt horizon: more urgent = shorter horizon
    if max_urgency > 0.7:
        horizon = 8  # 2 hours - focus on urgent charging
    elif max_urgency > 0.4:
        horizon = 16  # 4 hours - balanced
    else:
        horizon = 24  # 6 hours - long-term optimization
    
    return lookahead_greedy(household, lookahead_horizon=horizon)
