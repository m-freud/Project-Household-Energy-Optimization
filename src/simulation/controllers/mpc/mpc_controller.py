import cvxpy as cp
import numpy as np
from typing import cast
from cvxpy.constraints.constraint import Constraint

# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.base_controller import BaseController
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.config.device_buffer_config import DeviceBufferConfig


class MPCController(BaseController):
    def __init__(
        self,
        name: str,
        household: Household,
        scenario: Scenario,
        predictor: BasePredictor,
        horizon: int = 96,
        duration_hours: float = 0.25,
        buffer_config: DeviceBufferConfig | None = None,
    ):
        super().__init__(name)
        self.household = household
        self.scenario = scenario
        self.horizon = int(horizon)
        self.predictor = predictor
        self.duration_hours = float(duration_hours)
        self.buffer_config = buffer_config or DeviceBufferConfig.disabled()

        self.bess_bounds = [-self.household.bess.max_discharge, self.household.bess.max_charge] if self.household.bess else [0,0]
        self.ev1_bounds = [0, self.household.ev1.max_charge] if self.household.ev1 else [0,0]
        self.ev2_bounds = [0, self.household.ev2.max_charge] if self.household.ev2 else [0,0]

        self.bess_soc_targets = self.scenario.bess.soc_targets if self.household.bess else {}
        self.ev1_soc_targets = self.scenario.ev1.soc_targets if self.household.ev1 else {}
        self.ev2_soc_targets = self.scenario.ev2.soc_targets if self.household.ev2 else {}

        # Built once and reused every timestep via Parameters (DPP) to avoid
        # repeated canonicalization overhead.
        self._compiled_horizon: int | None = None
        self._problem: cp.Problem | None = None
        self._params: dict[str, cp.Parameter] = {}
        self._vars: dict[str, cp.Variable | None] = {}


    def _planning_horizon(self, current_timestep: int) -> int:
        # Always return the full horizon so _ensure_compiled_problem compiles
        # the CVXPY problem exactly once and DPP warm-starting applies on every
        # subsequent solve.  Predictions are padded by _prediction_series, and
        # _target_lb places constraints at the correct relative index regardless
        # of how many real timesteps remain.
        _ = current_timestep
        return max(1, int(self.horizon))

    def _prediction_series(self, predictions: dict, key: str, horizon: int, default: float = 0.0) -> list[float]:
        values = predictions.get(key, [])
        if not isinstance(values, list):
            values = [values]

        series = [float(value) for value in values[:horizon]]
        if len(series) < horizon:
            fill_value = series[-1] if series else default
            series.extend([float(fill_value)] * (horizon - len(series)))
        return series

    def _ensure_compiled_problem(self, planning_horizon: int):
        if self._problem is not None and self._compiled_horizon == planning_horizon:
            # only if horizon < full day. no need to recompile if horizon stays the same.
            return

        h = int(planning_horizon)
        duration_hours = self.duration_hours

        # Define optimization params, start with global params and variables
        params: dict[str, cp.Parameter] = {
            "base_load": cp.Parameter(h),
            "pv_gen": cp.Parameter(h),
            "buy_price": cp.Parameter(h),
            "sell_price": cp.Parameter(h),
        }
        variables: dict[str, cp.Variable | None] = {}

        grid_import = cp.Variable(h, nonneg=True)
        grid_export = cp.Variable(h, nonneg=True)
        variables["grid_import"] = grid_import
        variables["grid_export"] = grid_export

        net_load_expr = params["base_load"] - params["pv_gen"] # initial net load, we add bess and ev charging load later
        objective = duration_hours * (
            params["buy_price"] @ grid_import - params["sell_price"] @ grid_export
        )
        constraints: list[Constraint] = [net_load_expr == grid_import - grid_export]

        if self.household.bess:
            params["bess_soc0"] = cp.Parameter()
            params["bess_target_lb"] = cp.Parameter(h + 1)

            bess_power = cp.Variable(h)
            bess_charge = cp.Variable(h, nonneg=True)
            bess_discharge = cp.Variable(h, nonneg=True)
            bess_soc = cp.Variable(h + 1, bounds=[0.0, self.household.bess.capacity])

            variables["bess_power"] = bess_power
            constraints.extend([
                bess_soc[0] == params["bess_soc0"],
                bess_charge <= self.household.bess.max_charge,
                bess_discharge <= self.household.bess.max_discharge,
                bess_power == bess_charge - bess_discharge,
                bess_soc[1:]
                == bess_soc[:-1]
                + self.household.bess.efficiency * bess_charge * duration_hours
                - bess_discharge * duration_hours / self.household.bess.efficiency,
                bess_soc[1:] >= params["bess_target_lb"][1:], # ensure targets are met with target peak function [0 0 0 target 0 0 ...]
            ])
            net_load_expr = net_load_expr + bess_power
        else:
            variables["bess_power"] = None

        if self.household.ev1:
            params["ev1_soc0"] = cp.Parameter()
            params["ev1_effective_max_charge"] = cp.Parameter(h, nonneg=True)  # = raw_max * availability
            params["ev1_home_mask"] = cp.Parameter(h)
            params["ev1_drive_load"] = cp.Parameter(h)
            params["ev1_station_price"] = cp.Parameter(h)
            params["ev1_target_lb"] = cp.Parameter(h + 1)

            ev1_charge = cp.Variable(h, nonneg=True)
            ev1_soc = cp.Variable(h + 1, bounds=[0.0, self.household.ev1.capacity])
            variables["ev1_charge"] = ev1_charge

            constraints.extend([
                ev1_soc[0] == params["ev1_soc0"],
                ev1_charge <= params["ev1_effective_max_charge"],  # DPP: param only, availability folded in
                ev1_soc[1:]
                == ev1_soc[:-1]
                + ev1_charge * duration_hours * self.household.ev1.efficiency
                - params["ev1_drive_load"] * duration_hours / self.household.ev1.efficiency,
                ev1_soc[1:] >= params["ev1_target_lb"][1:], # ensure targets are met with target peak function [0 0 0 target 0 0 ...]
            ])

            net_load_expr = net_load_expr + cp.multiply(params["ev1_home_mask"], ev1_charge)
            objective = objective + cp.sum(
                duration_hours * cp.multiply(params["ev1_station_price"], ev1_charge)
            )
        else:
            variables["ev1_charge"] = None

        if self.household.ev2:
            params["ev2_soc0"] = cp.Parameter()
            params["ev2_effective_max_charge"] = cp.Parameter(h, nonneg=True)  # = raw_max * availability
            params["ev2_home_mask"] = cp.Parameter(h)
            params["ev2_drive_load"] = cp.Parameter(h)
            params["ev2_station_price"] = cp.Parameter(h)
            params["ev2_target_lb"] = cp.Parameter(h + 1)

            ev2_charge = cp.Variable(h, nonneg=True)
            ev2_soc = cp.Variable(h + 1, bounds=[0.0, self.household.ev2.capacity])
            variables["ev2_charge"] = ev2_charge

            constraints.extend([
                ev2_soc[0] == params["ev2_soc0"],
                ev2_charge <= params["ev2_effective_max_charge"],  # DPP: param only, availability folded in
                ev2_soc[1:]
                == ev2_soc[:-1]
                + ev2_charge * duration_hours * self.household.ev2.efficiency
                - params["ev2_drive_load"] * duration_hours / self.household.ev2.efficiency,
                ev2_soc[1:] >= params["ev2_target_lb"][1:], # ensure targets are met with target peak function [0 0 0 target 0 0 ...]
            ])

            net_load_expr = net_load_expr + cp.multiply(params["ev2_home_mask"], ev2_charge)
            objective = objective + cp.sum(
                duration_hours * cp.multiply(params["ev2_station_price"], ev2_charge)
            )
        else:
            variables["ev2_charge"] = None

        constraints[0] = net_load_expr == grid_import - grid_export

        self._compiled_horizon = h
        self._params = params
        self._vars = variables
        self._problem = cp.Problem(cp.Minimize(objective), constraints)

    def _target_lb(
        self,
        targets: dict | None,
        capacity: float,
        current_timestep: int,
        horizon: int,
        time_buffer_steps: int = 0,
        energy_buffer_soc_frct: float = 0.0,
    ) -> np.ndarray:
        lb = np.zeros(horizon + 1, dtype=float)
        if not targets:
            return lb

        time_buffer_steps = max(0, int(time_buffer_steps))
        energy_buffer_soc_frct = max(0.0, float(energy_buffer_soc_frct))
        for deadline, target_soc in sorted(targets.items()):
            buffered_target_soc = min(1.0, float(target_soc) + energy_buffer_soc_frct)
            target_soc_kwh = buffered_target_soc * float(capacity)
            deadline_step = int(deadline)

            # Keep the original deadline target and copy it left for each
            # configured buffer step (e.g., buffer=4 -> d, d-1, d-2, d-3, d-4).
            for offset in range(0, time_buffer_steps + 1):
                buffered_step = deadline_step - offset
                target_index = buffered_step - int(current_timestep) + 1
                if 0 <= target_index <= horizon:
                    lb[target_index] = max(lb[target_index], target_soc_kwh)
        return lb

    def _first_value(self, variable: cp.Variable | None) -> float:
        if variable is None:
            return 0.0
        value = variable[0].value
        if value is None:
            return 0.0
        return float(np.asarray(value).item())

    def set_controls(self, household: Household, scenario: Scenario, predictor: str = 'oracle', *args, **kwargs):
        _ = (household, scenario, args, kwargs)

        current_timestep = household.current_timestep
        bess_soc = household.bess_soc
        ev1_soc = household.ev1_soc
        ev2_soc = household.ev2_soc
        base_load = household.base_load
        pv_generation = household.pv_gen
        buy_price = household.buy_price
        sell_price = household.sell_price
        ev1_at_home = household.ev1_at_home
        ev2_at_home = household.ev2_at_home
        ev1_at_charging_station = household.ev1_at_charging_station
        ev2_at_charging_station = household.ev2_at_charging_station
        ev1_load = household.ev1_load
        ev2_load = household.ev2_load

        planning_horizon = self._planning_horizon(current_timestep)
        self._ensure_compiled_problem(planning_horizon)
        predictions = self.predictor.predict(household, scenario, planning_horizon)

        base_load_profile = self._prediction_series(predictions, "base_load", planning_horizon, default=base_load)
        pv_profile = self._prediction_series(predictions, "pv_gen", planning_horizon, default=pv_generation)
        ev1_load_profile = self._prediction_series(predictions, "ev1_load", planning_horizon, default=ev1_load)
        ev2_load_profile = self._prediction_series(predictions, "ev2_load", planning_horizon, default=ev2_load)
        ev1_home_profile = self._prediction_series(predictions, "ev1_at_home", planning_horizon, default=1.0 if ev1_at_home else 0.0)
        ev2_home_profile = self._prediction_series(predictions, "ev2_at_home", planning_horizon, default=1.0 if ev2_at_home else 0.0)
        ev1_station_profile = self._prediction_series(predictions, "ev1_at_charging_station", planning_horizon, default=1.0 if ev1_at_charging_station else 0.0)
        ev2_station_profile = self._prediction_series(predictions, "ev2_at_charging_station", planning_horizon, default=1.0 if ev2_at_charging_station else 0.0)
        ev1_max_charge_profile = self._prediction_series(predictions, "ev1_max_charge", planning_horizon, default=household.ev1.max_charge if household.ev1 else 0.0)
        ev2_max_charge_profile = self._prediction_series(predictions, "ev2_max_charge", planning_horizon, default=household.ev2.max_charge if household.ev2 else 0.0)
        buy_price_profile = self._prediction_series(predictions, "buy_price", planning_horizon, default=buy_price)
        sell_price_profile = self._prediction_series(predictions, "sell_price", planning_horizon, default=sell_price)
        ev1_buy_price_profile = self._prediction_series(predictions, "ev1_buy_price", planning_horizon, default=buy_price)
        ev2_buy_price_profile = self._prediction_series(predictions, "ev2_buy_price", planning_horizon, default=buy_price)

        # update /fill the compiled problem
        # global params first
        self._params["base_load"].value = np.asarray(base_load_profile, dtype=float)
        self._params["pv_gen"].value = np.asarray(pv_profile, dtype=float)
        self._params["buy_price"].value = np.asarray(buy_price_profile, dtype=float)
        self._params["sell_price"].value = np.asarray(sell_price_profile, dtype=float)

        # device params next
        if household.bess:
            self._params["bess_soc0"].value = float(bess_soc)
            self._params["bess_target_lb"].value = self._target_lb(
                self.bess_soc_targets,
                household.bess.capacity,
                current_timestep,
                planning_horizon,
                time_buffer_steps=self.buffer_config.bess.time_buffer_steps,
                energy_buffer_soc_frct=self.buffer_config.bess.energy_buffer_soc_frct,
            )

        if household.ev1:
            ev1_home_mask = np.asarray([1.0 if value > 0 else 0.0 for value in ev1_home_profile], dtype=float)
            ev1_station_mask = np.asarray([1.0 if value > 0 else 0.0 for value in ev1_station_profile], dtype=float)
            ev1_availability = np.maximum(ev1_home_mask, ev1_station_mask)
            ev1_drive_load = np.asarray(ev1_load_profile, dtype=float) * (1.0 - ev1_availability)
            ev1_station_price = np.asarray(ev1_buy_price_profile, dtype=float) * ev1_station_mask
            ev1_profile_max_charge = np.asarray(ev1_max_charge_profile, dtype=float)
            ev1_device_max_charge = float(self.ev1_bounds[1])
            # Fold availability into effective_max_charge so the CVXPY constraint is DPP-compliant (param only, no param*param)
            ev1_effective_max_charge = np.minimum(ev1_profile_max_charge, ev1_device_max_charge) * ev1_availability

            self._params["ev1_soc0"].value = float(ev1_soc)
            self._params["ev1_home_mask"].value = ev1_home_mask
            self._params["ev1_effective_max_charge"].value = ev1_effective_max_charge
            self._params["ev1_drive_load"].value = ev1_drive_load
            self._params["ev1_station_price"].value = ev1_station_price
            self._params["ev1_target_lb"].value = self._target_lb(
                self.ev1_soc_targets,
                household.ev1.capacity,
                current_timestep,
                planning_horizon,
                time_buffer_steps=self.buffer_config.ev1.time_buffer_steps,
                energy_buffer_soc_frct=self.buffer_config.ev1.energy_buffer_soc_frct,
            )

        if household.ev2:
            ev2_home_mask = np.asarray([1.0 if value > 0 else 0.0 for value in ev2_home_profile], dtype=float)
            ev2_station_mask = np.asarray([1.0 if value > 0 else 0.0 for value in ev2_station_profile], dtype=float)
            ev2_availability = np.maximum(ev2_home_mask, ev2_station_mask)
            ev2_drive_load = np.asarray(ev2_load_profile, dtype=float) * (1.0 - ev2_availability)
            ev2_station_price = np.asarray(ev2_buy_price_profile, dtype=float) * ev2_station_mask
            ev2_profile_max_charge = np.asarray(ev2_max_charge_profile, dtype=float)
            ev2_device_max_charge = float(self.ev2_bounds[1])
            # Fold availability into effective_max_charge so the CVXPY constraint is DPP-compliant (param only, no param*param)
            ev2_effective_max_charge = np.minimum(ev2_profile_max_charge, ev2_device_max_charge) * ev2_availability

            self._params["ev2_soc0"].value = float(ev2_soc)
            self._params["ev2_home_mask"].value = ev2_home_mask
            self._params["ev2_effective_max_charge"].value = ev2_effective_max_charge
            self._params["ev2_drive_load"].value = ev2_drive_load
            self._params["ev2_station_price"].value = ev2_station_price
            self._params["ev2_target_lb"].value = self._target_lb(
                self.ev2_soc_targets,
                household.ev2.capacity,
                current_timestep,
                planning_horizon,
                time_buffer_steps=self.buffer_config.ev2.time_buffer_steps,
                energy_buffer_soc_frct=self.buffer_config.ev2.energy_buffer_soc_frct,
            )

        problem = cast(cp.Problem, self._problem)

        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        except Exception:
            problem.solve(solver=cp.SCS, warm_start=True, verbose=False)

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            return {
                "bess_power": 0.0,
                "ev1_power": 0.0,
                "ev2_power": 0.0,
            }

        bess_power = self._vars.get("bess_power")
        ev1_charge = self._vars.get("ev1_charge")
        ev2_charge = self._vars.get("ev2_charge")

        return {
            "bess_power": self._first_value(bess_power),
            "ev1_power": self._first_value(ev1_charge),
            "ev2_power": self._first_value(ev2_charge),
        }
