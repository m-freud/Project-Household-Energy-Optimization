"""
Reinforcement Learning Environment for Household Energy Management

OpenAI Gym-compatible environment wrapper for training RL agents.
Enables comparison of learning-based control policies with optimization baselines.

Requirements:
    pip install gymnasium stable-baselines3
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
from copy import deepcopy

from src.simulation.household import Household
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from src.simulation.components.PV import PV
from src.simulation.requirements.charge_requirements import basic_charge_requirements


class EnergyManagementEnv(gym.Env):
    """
    Gym environment for household energy management.
    
    Observation Space:
        - Current timestep (normalized)
        - BESS SOC (normalized)
        - EV1 SOC (normalized)
        - EV2 SOC (normalized)
        - Current buy price
        - Current sell price
        - Base load
        - PV generation
        - EV1 at home (binary)
        - EV2 at home (binary)
        - BESS requirement SOC
        - EV1 requirement SOC
        - EV2 requirement SOC
        - Timesteps until BESS deadline
        - Timesteps until EV1 deadline
        - Timesteps until EV2 deadline
    
    Action Space:
        Continuous actions for each controllable asset:
        - BESS power (-1 to 1, scaled to max discharge/charge)
        - EV1 power (0 to 1, scaled to max charge)
        - EV2 power (0 to 1, scaled to max charge)
    
    Reward:
        - Negative cost (minimize cost)
        - Large penalty for constraint violations
        - Small bonus for meeting requirements early
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        household_config: Dict = None,
        max_timesteps: int = 96,
        charge_requirements: Dict = None,
        constraint_penalty: float = 100.0,
        early_completion_bonus: float = 1.0
    ):
        """
        Initialize RL environment.
        
        Args:
            household_config: Configuration for household components
            max_timesteps: Episode length (default 96 = 24 hours)
            charge_requirements: SOC requirements for BESS/EVs
            constraint_penalty: Penalty multiplier for missing requirements
            early_completion_bonus: Reward for meeting requirements
        """
        super().__init__()
        
        self.max_timesteps = max_timesteps
        self.charge_requirements = charge_requirements or basic_charge_requirements
        self.constraint_penalty = constraint_penalty
        self.early_completion_bonus = early_completion_bonus
        
        # Store household config for reset
        self.household_config = household_config or self._default_household_config()
        
        # Initialize household (will be reset)
        self.household = None
        self.current_step = 0
        
        # Define observation space (16 features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )
        
        # Define action space (3 continuous actions)
        # [bess_power, ev1_power, ev2_power]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Tracking metrics
        self.episode_cost = 0.0
        self.episode_consumption = 0.0
        self.constraint_violations = []
    
    def _default_household_config(self) -> Dict:
        """Default household configuration."""
        return {
            'has_pv': True,
            'has_bess': True,
            'has_ev1': True,
            'has_ev2': True,
            'bess_capacity': 13.5,
            'bess_max_charge': 5.0,
            'bess_max_discharge': 5.0,
            'bess_efficiency': 0.95,
            'ev_capacity': 75.0,
            'ev_max_charge': 11.0,
            'ev_efficiency': 0.9,
            'pv_capacity': 5.0
        }
    
    def _create_household(self) -> Household:
        """Create household instance from config."""
        config = self.household_config
        
        # Create components
        pv = PV(capacity=config.get('pv_capacity', 5.0)) if config.get('has_pv') else None
        
        bess = BESS(
            capacity=config.get('bess_capacity', 13.5),
            max_charge=config.get('bess_max_charge', 5.0),
            max_discharge=config.get('bess_max_discharge', 5.0),
            efficiency=config.get('bess_efficiency', 0.95),
            soc=config.get('bess_initial_soc', 0.0)
        ) if config.get('has_bess') else None
        
        ev1 = EV(
            capacity=config.get('ev_capacity', 75.0),
            max_charge=config.get('ev_max_charge', 11.0),
            max_discharge=0.0,
            efficiency=config.get('ev_efficiency', 0.9),
            soc=config.get('ev1_initial_soc', 0.0),
            at_home=True
        ) if config.get('has_ev1') else None
        
        ev2 = EV(
            capacity=config.get('ev_capacity', 75.0),
            max_charge=config.get('ev_max_charge', 11.0),
            max_discharge=0.0,
            efficiency=config.get('ev_efficiency', 0.9),
            soc=config.get('ev2_initial_soc', 0.0),
            at_home=True
        ) if config.get('has_ev2') else None
        
        household = Household(
            player_id=0,
            start_time=0,
            pv=pv,
            bess=bess,
            ev1=ev1,
            ev2=ev2,
            charge_requirements=self.charge_requirements
        )
        
        return household
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Timestep (normalized to [0, 1])
        obs.append(self.current_step / self.max_timesteps)
        
        # BESS state
        if self.household.bess:
            obs.append(self.household.bess.soc / self.household.bess.capacity)  # Normalized SOC
        else:
            obs.append(0.0)
        
        # EV1 state
        if self.household.ev1:
            obs.append(self.household.ev1.soc / self.household.ev1.capacity)
        else:
            obs.append(0.0)
        
        # EV2 state
        if self.household.ev2:
            obs.append(self.household.ev2.soc / self.household.ev2.capacity)
        else:
            obs.append(0.0)
        
        # Prices (normalized by typical range 0-1 EUR/kWh)
        obs.append(self.household.buy_price)
        obs.append(self.household.sell_price)
        
        # Load and generation (normalized by typical values)
        obs.append(self.household.base_load / 10.0)  # Typical max ~10 kW
        pv_gen = self.household.pv.generation if self.household.pv else 0.0
        obs.append(pv_gen / 10.0)
        
        # EV availability
        obs.append(1.0 if (self.household.ev1 and self.household.ev1.at_home) else 0.0)
        obs.append(1.0 if (self.household.ev2 and self.household.ev2.at_home) else 0.0)
        
        # Requirements (normalized)
        bess_req_soc = self.charge_requirements.get('bess', {}).get('soc', 0.0)
        ev1_req_soc = self.charge_requirements.get('ev1', {}).get('soc', 0.0)
        ev2_req_soc = self.charge_requirements.get('ev2', {}).get('soc', 0.0)
        
        obs.append(bess_req_soc)
        obs.append(ev1_req_soc)
        obs.append(ev2_req_soc)
        
        # Time to deadlines (normalized)
        bess_deadline = self.charge_requirements.get('bess', {}).get('timestep', 96)
        ev1_deadline = self.charge_requirements.get('ev1', {}).get('timestep', 96)
        ev2_deadline = self.charge_requirements.get('ev2', {}).get('timestep', 96)
        
        obs.append(max(0, bess_deadline - self.current_step) / self.max_timesteps)
        obs.append(max(0, ev1_deadline - self.current_step) / self.max_timesteps)
        obs.append(max(0, ev2_deadline - self.current_step) / self.max_timesteps)
        
        return np.array(obs, dtype=np.float32)
    
    def _scale_action(self, action: np.ndarray) -> Dict:
        """
        Scale normalized action to actual power setpoints.
        
        Args:
            action: [bess_power, ev1_power, ev2_power] in [-1,1] or [0,1]
        
        Returns:
            Control dictionary with actual power values
        """
        controls = {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0
        }
        
        # BESS: action in [-1, 1] maps to [-max_discharge, max_charge]
        if self.household.bess:
            if action[0] >= 0:
                controls["bess_power"] = action[0] * self.household.bess.max_charge
            else:
                controls["bess_power"] = action[0] * self.household.bess.max_discharge
        
        # EV1: action in [0, 1] maps to [0, max_charge]
        if self.household.ev1 and self.household.ev1.at_home:
            controls["ev1_power"] = action[1] * self.household.ev1.max_charge
        
        # EV2: action in [0, 1] maps to [0, max_charge]
        if self.household.ev2 and self.household.ev2.at_home:
            controls["ev2_power"] = action[2] * self.household.ev2.max_charge
        
        return controls
    
    def _calculate_reward(self, cost: float, check_violations: bool = False) -> float:
        """
        Calculate reward for current step.
        
        Args:
            cost: Cost incurred this timestep
            check_violations: Whether to check constraint violations (at deadline)
        
        Returns:
            Scalar reward
        """
        # Base reward: negative cost (we want to minimize)
        reward = -cost
        
        # Check constraint violations if at deadline
        if check_violations:
            violations = []
            
            # Check BESS
            if self.household.bess and 'bess' in self.charge_requirements:
                req = self.charge_requirements['bess']
                if req.get('timestep', 96) == self.current_step:
                    required_soc = req['soc'] * self.household.bess.capacity
                    deficit = max(0, required_soc - self.household.bess.soc)
                    if deficit > 0:
                        violations.append(('bess', deficit))
                        reward -= deficit * self.constraint_penalty
            
            # Check EVs
            for ev_name in ['ev1', 'ev2']:
                ev = getattr(self.household, ev_name, None)
                if ev and ev_name in self.charge_requirements:
                    req = self.charge_requirements[ev_name]
                    if req.get('timestep', 96) == self.current_step:
                        required_soc = req['soc'] * ev.capacity
                        deficit = max(0, required_soc - ev.soc)
                        if deficit > 0:
                            violations.append((ev_name, deficit))
                            reward -= deficit * self.constraint_penalty
            
            self.constraint_violations.extend(violations)
        
        # Small bonus for meeting requirements
        if self._all_requirements_met():
            reward += self.early_completion_bonus
        
        return reward
    
    def _all_requirements_met(self) -> bool:
        """Check if all current requirements are satisfied."""
        # Check BESS
        if self.household.bess and 'bess' in self.charge_requirements:
            req = self.charge_requirements['bess']
            required_soc = req['soc'] * self.household.bess.capacity
            if self.household.bess.soc < required_soc - 0.1:  # Small tolerance
                return False
        
        # Check EVs
        for ev_name in ['ev1', 'ev2']:
            ev = getattr(self.household, ev_name, None)
            if ev and ev_name in self.charge_requirements:
                req = self.charge_requirements[ev_name]
                required_soc = req['soc'] * ev.capacity
                if ev.soc < required_soc - 0.1:
                    return False
        
        return True
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Create fresh household
        self.household = self._create_household()
        self.current_step = 0
        
        # Reset tracking
        self.episode_cost = 0.0
        self.episode_consumption = 0.0
        self.constraint_violations = []
        
        # TODO: Load time-series data for household (prices, loads, PV, etc.)
        # For now, using placeholder values
        self.household.buy_price = 0.3
        self.household.sell_price = 0.05
        self.household.base_load = 2.0
        if self.household.pv:
            self.household.pv.generation = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep.
        
        Args:
            action: Action from agent
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action to actual controls
        controls = self._scale_action(action)
        
        # Apply controls to household
        self.household.apply_controls(controls, duration_hours=0.25)
        
        # Calculate net load and cost
        pv_gen = self.household.pv.generation if self.household.pv else 0.0
        net_load = self.household.base_load - pv_gen
        
        # Add EV loads
        if self.household.ev1 and self.household.ev1.at_home:
            net_load += controls['ev1_power']
        if self.household.ev2 and self.household.ev2.at_home:
            net_load += controls['ev2_power']
        
        # Add BESS load
        if controls['bess_power'] > 0:
            net_load += controls['bess_power']
        else:
            net_load -= abs(controls['bess_power'])
        
        # Calculate cost
        if net_load > 0:
            cost = net_load * 0.25 * self.household.buy_price
            consumption = net_load * 0.25
        else:
            cost = net_load * 0.25 * self.household.sell_price
            consumption = 0.0
        
        self.episode_cost += cost
        self.episode_consumption += consumption
        
        # Calculate reward
        check_violations = (self.current_step == self.max_timesteps - 1)
        reward = self._calculate_reward(cost, check_violations)
        
        # Advance timestep
        self.current_step += 1
        self.household.current_timestep = self.current_step
        
        # Update household state (simplified - should load from profiles)
        # TODO: Load actual time-series data
        
        # Check if episode is done
        terminated = (self.current_step >= self.max_timesteps)
        truncated = False
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            'episode_cost': self.episode_cost if terminated else None,
            'episode_consumption': self.episode_consumption if terminated else None,
            'constraint_violations': len(self.constraint_violations) if terminated else None,
            'timestep': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render environment (optional)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
