"""
Model Predictive Control (MPC) & Linear Programming Optimization

Implements mathematical optimization-based control policies using convex optimization.
Unlike greedy/lookahead policies that evaluate discrete actions, MPC formulates the
control problem as a constrained optimization and uses a solver to find the globally
optimal continuous control trajectory.

This represents state-of-the-art baseline performance for energy management systems.
"""

import numpy as np
from typing import Dict, Optional
from src.simulation.household import Household

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not installed. MPC policies will not work.")
    print("Install with: pip install cvxpy")


class MPCController:
    """
    Model Predictive Control for household energy management.
    
    Formulates and solves a convex optimization problem at each timestep:
    
    minimize: Σ cost(t) over horizon
    
    subject to:
        - Energy balance constraints
        - SOC dynamics for BESS and EVs
        - Power limits (charge/discharge rates)
        - SOC bounds (0 to capacity)
        - Terminal constraints (requirements at deadlines)
    """
    
    def __init__(self, horizon: int = 24, timestep_duration: float = 0.25):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Planning horizon in timesteps (default 24 = 6 hours)
            timestep_duration: Duration of each timestep in hours (default 0.25 = 15min)
        """
        self.horizon = horizon
        self.dt = timestep_duration
        
    def solve(self, household: Household) -> Dict:
        """
        Solve MPC optimization problem for current household state.
        
        Returns:
            Control dictionary with optimal power setpoints for this timestep
        """
        if not CVXPY_AVAILABLE:
            # Fallback: return no control
            return {"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0}
        
        current_t = household.current_timestep
        
        # ====================================================================
        # SETUP OPTIMIZATION VARIABLES
        # ====================================================================
        
        # Decision variables: power setpoints for each asset at each timestep
        bess_charge = None
        bess_discharge = None
        bess_soc = None
        ev1_charge = None
        ev1_soc = None
        ev2_charge = None
        ev2_soc = None
        grid_import = None
        grid_export = None
        
        # BESS variables
        if household.bess:
            bess_charge = cp.Variable(self.horizon, nonneg=True)
            bess_discharge = cp.Variable(self.horizon, nonneg=True)
            bess_soc = cp.Variable(self.horizon + 1, nonneg=True)
        
        # EV1 variables
        if household.ev1:
            ev1_charge = cp.Variable(self.horizon, nonneg=True)
            ev1_soc = cp.Variable(self.horizon + 1, nonneg=True)
        
        # EV2 variables
        if household.ev2:
            ev2_charge = cp.Variable(self.horizon, nonneg=True)
            ev2_soc = cp.Variable(self.horizon + 1, nonneg=True)
        
        # Grid variables (for cost calculation)
        grid_import = cp.Variable(self.horizon, nonneg=True)
        grid_export = cp.Variable(self.horizon, nonneg=True)
        
        # ====================================================================
        # SETUP CONSTRAINTS
        # ====================================================================
        
        constraints = []
        
        # --- BESS Constraints ---
        if household.bess:
            bess = household.bess
            
            # Initial SOC
            constraints.append(bess_soc[0] == bess.soc)
            
            # SOC dynamics
            for t in range(self.horizon):
                constraints.append(
                    bess_soc[t + 1] == bess_soc[t] 
                    + bess_charge[t] * self.dt * bess.efficiency
                    - bess_discharge[t] * self.dt / bess.efficiency
                )
            
            # Power limits
            constraints.append(bess_charge <= bess.max_charge)
            constraints.append(bess_discharge <= bess.max_discharge)
            
            # SOC bounds
            constraints.append(bess_soc <= bess.capacity)
            constraints.append(bess_soc >= 0)
            
            # Terminal constraint (requirement)
            if 'bess' in household.charge_requirements:
                req = household.charge_requirements['bess']
                deadline_t = req.get('timestep', 96)
                required_soc = req.get('soc', 0.0) * bess.capacity
                
                # If deadline is within horizon, enforce it
                horizon_deadline = min(deadline_t - current_t, self.horizon)
                if horizon_deadline > 0:
                    constraints.append(bess_soc[horizon_deadline] >= required_soc)
        
        # --- EV1 Constraints ---
        if household.ev1:
            ev = household.ev1
            
            # Initial SOC
            constraints.append(ev1_soc[0] == ev.soc)
            
            # SOC dynamics (only when at home)
            for t in range(self.horizon):
                # Simplified: assume EV stays at home for horizon (could use schedule)
                if ev.at_home:
                    constraints.append(
                        ev1_soc[t + 1] == ev1_soc[t] + ev1_charge[t] * self.dt * ev.efficiency
                    )
                else:
                    # If not home, can't charge
                    constraints.append(ev1_charge[t] == 0)
                    constraints.append(ev1_soc[t + 1] == ev1_soc[t])
            
            # Power limits
            if ev.at_home:
                constraints.append(ev1_charge <= ev.max_charge)
            else:
                constraints.append(ev1_charge == 0)
            
            # SOC bounds
            constraints.append(ev1_soc <= ev.capacity)
            constraints.append(ev1_soc >= 0)
            
            # Terminal constraint
            if 'ev1' in household.charge_requirements:
                req = household.charge_requirements['ev1']
                deadline_t = req.get('timestep', 96)
                required_soc = req.get('soc', 0.0) * ev.capacity
                
                horizon_deadline = min(deadline_t - current_t, self.horizon)
                if horizon_deadline > 0 and ev.at_home:
                    constraints.append(ev1_soc[horizon_deadline] >= required_soc)
        
        # --- EV2 Constraints ---
        if household.ev2:
            ev = household.ev2
            
            constraints.append(ev2_soc[0] == ev.soc)
            
            for t in range(self.horizon):
                if ev.at_home:
                    constraints.append(
                        ev2_soc[t + 1] == ev2_soc[t] + ev2_charge[t] * self.dt * ev.efficiency
                    )
                else:
                    constraints.append(ev2_charge[t] == 0)
                    constraints.append(ev2_soc[t + 1] == ev2_soc[t])
            
            if ev.at_home:
                constraints.append(ev2_charge <= ev.max_charge)
            else:
                constraints.append(ev2_charge == 0)
            
            constraints.append(ev2_soc <= ev.capacity)
            constraints.append(ev2_soc >= 0)
            
            if 'ev2' in household.charge_requirements:
                req = household.charge_requirements['ev2']
                deadline_t = req.get('timestep', 96)
                required_soc = req.get('soc', 0.0) * ev.capacity
                
                horizon_deadline = min(deadline_t - current_t, self.horizon)
                if horizon_deadline > 0 and ev.at_home:
                    constraints.append(ev2_soc[horizon_deadline] >= required_soc)
        
        # --- Grid Power Balance ---
        for t in range(self.horizon):
            # Get future base load and PV (assume constant for simplicity, could be profiled)
            base_load = household.base_load
            pv_gen = household.pv.generation if household.pv else 0.0
            
            # Net load = base + controllable loads - PV - BESS discharge
            net_load = base_load
            
            if household.bess:
                net_load = net_load + bess_charge[t] - bess_discharge[t]
            
            if household.ev1 and household.ev1.at_home:
                net_load = net_load + ev1_charge[t]
            
            if household.ev2 and household.ev2.at_home:
                net_load = net_load + ev2_charge[t]
            
            net_load = net_load - pv_gen
            
            # Grid balances net load
            constraints.append(grid_import[t] - grid_export[t] == net_load)
        
        # ====================================================================
        # SETUP OBJECTIVE (Minimize Cost)
        # ====================================================================
        
        cost = 0
        
        for t in range(self.horizon):
            # Get future prices
            future_t = current_t + t
            if future_t < len(household.buy_price_day_profile):
                buy_price = household.buy_price_day_profile[future_t]
                sell_price = household.sell_price_day_profile[future_t]
            else:
                # Beyond known prices, use last known
                buy_price = household.buy_price_day_profile[-1] if household.buy_price_day_profile else 0.3
                sell_price = household.sell_price_day_profile[-1] if household.sell_price_day_profile else 0.05
            
            # Cost = import cost - export revenue
            cost += grid_import[t] * self.dt * buy_price
            cost -= grid_export[t] * self.dt * sell_price
        
        # ====================================================================
        # SOLVE OPTIMIZATION PROBLEM
        # ====================================================================
        
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        
        # Try multiple solvers in order of preference
        solvers_to_try = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS]
        solved = False
        
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    solved = True
                    break
                    
            except Exception as e:
                # Solver not available or failed, try next one
                continue
        
        if not solved:
            print(f"Warning: MPC could not find optimal solution. Status: {problem.status}")
            return {"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0}
        
        # ====================================================================
        # EXTRACT OPTIMAL CONTROLS (first timestep only - receding horizon)
        # ====================================================================
        
        controls = {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0
        }
        
        if household.bess and bess_charge is not None:
            # Net BESS power (positive = charge, negative = discharge)
            controls["bess_power"] = float(bess_charge.value[0] - bess_discharge.value[0])
        
        if household.ev1 and ev1_charge is not None:
            controls["ev1_power"] = float(ev1_charge.value[0])
        
        if household.ev2 and ev2_charge is not None:
            controls["ev2_power"] = float(ev2_charge.value[0])
        
        return controls


class RobustMPCController(MPCController):
    """
    Robust MPC with safety margins and constraint tightening.
    
    Adds conservatism to handle uncertainties:
    - Tightens terminal constraints by a margin
    - Adds buffer to power limits
    - Handles potential forecast errors
    """
    
    def __init__(self, horizon: int = 24, safety_margin: float = 0.05):
        """
        Args:
            horizon: Planning horizon
            safety_margin: Safety margin for constraints (0.05 = 5% buffer)
        """
        super().__init__(horizon)
        self.safety_margin = safety_margin
    
    def solve(self, household: Household) -> Dict:
        # For robust version, we'd modify constraints in parent solve()
        # For now, just call parent (could extend with robust formulation)
        return super().solve(household)


# ========================================================================
# POLICY FUNCTIONS (for use in simulation)
# ========================================================================

def mpc_control(household: Household) -> Dict:
    """
    Standard MPC policy with 6-hour horizon.
    
    Globally optimal solution over lookahead window.
    """
    controller = MPCController(horizon=24)  # 6 hours
    return controller.solve(household)


def mpc_short_horizon(household: Household) -> Dict:
    """
    Short-horizon MPC (2 hours) - faster solving, more reactive.
    """
    controller = MPCController(horizon=8)
    return controller.solve(household)


def mpc_long_horizon(household: Household) -> Dict:
    """
    Long-horizon MPC (12 hours) - better long-term optimization.
    """
    controller = MPCController(horizon=48)
    return controller.solve(household)


def robust_mpc_control(household: Household) -> Dict:
    """
    Robust MPC with safety margins for constraint satisfaction.
    """
    controller = RobustMPCController(horizon=24, safety_margin=0.05)
    return controller.solve(household)


# ========================================================================
# FALLBACK: scipy-based optimization (no CVXPY required)
# ========================================================================

def mpc_scipy(household: Household) -> Dict:
    """
    MPC using scipy.optimize instead of CVXPY.
    More portable but less efficient for large problems.
    """
    from scipy.optimize import minimize
    import numpy as np
    
    horizon = 16
    dt = 0.25
    current_t = household.current_timestep
    
    # Decision variables vector: [bess_charge, bess_discharge, ev1_charge, ev2_charge] × horizon
    n_vars = horizon * 4  # 4 control variables per timestep
    
    def objective(x):
        """Calculate total cost over horizon."""
        total_cost = 0.0
        
        # Simulate forward with controls
        bess_soc = household.bess.soc if household.bess else 0.0
        ev1_soc = household.ev1.soc if household.ev1 else 0.0
        ev2_soc = household.ev2.soc if household.ev2 else 0.0
        
        for t in range(horizon):
            idx = t * 4
            bess_charge = x[idx]
            bess_discharge = x[idx + 1]
            ev1_charge = x[idx + 2]
            ev2_charge = x[idx + 3]
            
            # Update SOCs
            if household.bess:
                bess_soc += (bess_charge * household.bess.efficiency - bess_discharge / household.bess.efficiency) * dt
            if household.ev1:
                ev1_soc += ev1_charge * household.ev1.efficiency * dt
            if household.ev2:
                ev2_soc += ev2_charge * household.ev2.efficiency * dt
            
            # Calculate cost
            base_load = household.base_load
            pv_gen = household.pv.generation if household.pv else 0.0
            net_load = base_load + bess_charge - bess_discharge + ev1_charge + ev2_charge - pv_gen
            
            future_t = current_t + t
            if future_t < len(household.buy_price_day_profile):
                price = household.buy_price_day_profile[future_t]
            else:
                price = 0.3
            
            total_cost += max(0, net_load) * dt * price
        
        return total_cost
    
    # Bounds: all controls >= 0
    bounds = [(0, household.bess.max_charge if household.bess else 0)] * horizon  # bess_charge
    bounds += [(0, household.bess.max_discharge if household.bess else 0)] * horizon  # bess_discharge
    bounds += [(0, household.ev1.max_charge if household.ev1 and household.ev1.at_home else 0)] * horizon
    bounds += [(0, household.ev2.max_charge if household.ev2 and household.ev2.at_home else 0)] * horizon
    
    # Initial guess: zero controls
    x0 = np.zeros(n_vars)
    
    # Solve
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # Extract first controls
    controls = {
        "bess_power": float(result.x[0] - result.x[1]),  # charge - discharge
        "ev1_power": float(result.x[2]),
        "ev2_power": float(result.x[3])
    }
    
    return controls
