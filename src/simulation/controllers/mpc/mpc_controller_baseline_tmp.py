import cvxpy as cp

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


class MPCController(BaseController):
    def __init__(
        self,
        name: str,
        household: Household,
        scenario: Scenario,
        horizon: int = 96,
        predictor: BasePredictor | None = None,
    ):
        super().__init__(name)
        self.household = household
        self.scenario = scenario
        self.horizon = int(horizon)
        self.predictor = predictor

        self.bess_bounds = [-self.household.bess.max_discharge, self.household.bess.max_charge] if self.household.bess else None
        self.ev1_bounds = [0, self.household.ev1.max_charge] if self.household.ev1 else None
        self.ev2_bounds = [0, self.household.ev2.max_charge] if self.household.ev2 else None

        self.bess_soc_targets = self.scenario.bess.soc_targets if self.household.bess else None
        self.ev1_soc_targets = self.scenario.ev1.soc_targets if self.household.ev1 else None
        self.ev2_soc_targets = self.scenario.ev2.soc_targets if self.household.ev2 else None


    def _planning_horizon(self, current_timestep: int, scenario: Scenario) -> int:
        _ = (current_timestep, scenario)
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

        if self.predictor is None:
            raise NotImplementedError("No predictor configured for MPC controller")

        planning_horizon = self._planning_horizon(current_timestep, scenario)
        predictions = self.predictor.predict(household, scenario, planning_horizon)

        base_load_profile = self._prediction_series(predictions, "base_load", planning_horizon, default=base_load)
        pv_profile = self._prediction_series(predictions, "pv_gen", planning_horizon, default=pv_generation)
        ev1_load_profile = self._prediction_series(predictions, "ev1_load", planning_horizon, default=ev1_load)
        ev2_load_profile = self._prediction_series(predictions, "ev2_load", planning_horizon, default=ev2_load)
        ev1_home_profile = self._prediction_series(predictions, "ev1_at_home", planning_horizon, default=1.0 if ev1_at_home else 0.0)
        ev2_home_profile = self._prediction_series(predictions, "ev2_at_home", planning_horizon, default=1.0 if ev2_at_home else 0.0)
        ev1_station_profile = self._prediction_series(predictions, "ev1_at_charging_station", planning_horizon, default=1.0 if ev1_at_charging_station else 0.0)
        ev2_station_profile = self._prediction_series(predictions, "ev2_at_charging_station", planning_horizon, default=1.0 if ev2_at_charging_station else 0.0)
        buy_price_profile = self._prediction_series(predictions, "buy_price", planning_horizon, default=buy_price)
        sell_price_profile = self._prediction_series(predictions, "sell_price", planning_horizon, default=sell_price)
        ev1_buy_price_profile = self._prediction_series(predictions, "ev1_buy_price", planning_horizon, default=buy_price)
        ev2_buy_price_profile = self._prediction_series(predictions, "ev2_buy_price", planning_horizon, default=buy_price)

        duration_hours = 0.25
        constraints = []
        objective_terms = []
        grid_import = cp.Variable(planning_horizon, nonneg=True)
        grid_export = cp.Variable(planning_horizon, nonneg=True)

        if household.bess:
            bess_power = cp.Variable(planning_horizon, bounds=[-household.bess.max_discharge, household.bess.max_charge])
            bess_charge = cp.Variable(planning_horizon, nonneg=True)
            bess_discharge = cp.Variable(planning_horizon, nonneg=True)
            bess_soc_vars = cp.Variable(planning_horizon + 1, bounds=[0.0, household.bess.capacity])
            constraints.append(bess_soc_vars[0] == bess_soc)
            for t in range(planning_horizon):
                constraints.append(bess_charge[t] <= household.bess.max_charge)
                constraints.append(bess_discharge[t] <= household.bess.max_discharge)
                constraints.append(bess_power[t] == bess_charge[t] - bess_discharge[t])
                constraints.append(
                    bess_soc_vars[t + 1]
                    == bess_soc_vars[t]
                    + household.bess.efficiency * bess_charge[t] * duration_hours
                    - bess_discharge[t] * duration_hours / household.bess.efficiency
                )
                # we do not have to assert that charge/discharge conflict eah other, because the optimization will naturally avoid that due to efficiency losses
                # (round trip efficiency loss)
                # feel free to test this but this implicit constraint is better than another explicit boolean constrint like discharge * charge == 0

            for deadline, target_soc in sorted((self.bess_soc_targets or {}).items()):
                target_index = deadline - current_timestep + 1
                if 0 <= target_index <= planning_horizon:
                    target_soc_kwh = target_soc * household.bess.capacity
                    constraints.append(bess_soc_vars[target_index] >= target_soc_kwh)
        else:
            bess_power = None

        if household.ev1:
            ev1_charge = cp.Variable(planning_horizon, bounds=[0.0, household.ev1.max_charge])
            ev1_soc_vars = cp.Variable(planning_horizon + 1, bounds=[0.0, household.ev1.capacity])
            constraints.append(ev1_soc_vars[0] == ev1_soc)
            for t in range(planning_horizon):
                availability = 1.0 if (ev1_home_profile[t] > 0 or ev1_station_profile[t] > 0) else 0.0
                driving_load = ev1_load_profile[t] if availability <= 0.0 else 0.0
                constraints.append(ev1_charge[t] <= household.ev1.max_charge * availability)
                constraints.append(
                    ev1_soc_vars[t + 1]
                    == ev1_soc_vars[t]
                    + ev1_charge[t] * duration_hours * household.ev1.efficiency
                    - driving_load * duration_hours / household.ev1.efficiency
                )
                if ev1_home_profile[t] > 0:
                    pass
                elif ev1_station_profile[t] > 0:
                    objective_terms.append(ev1_buy_price_profile[t] * ev1_charge[t])

            for deadline, target_soc in sorted((self.ev1_soc_targets or {}).items()):
                target_index = deadline - current_timestep + 1
                if 0 <= target_index <= planning_horizon:
                    target_soc_kwh = target_soc * household.ev1.capacity
                    constraints.append(ev1_soc_vars[target_index] >= target_soc_kwh)
        else:
            ev1_charge = None

        if household.ev2:
            ev2_charge = cp.Variable(planning_horizon, bounds=[0.0, household.ev2.max_charge])
            ev2_soc_vars = cp.Variable(planning_horizon + 1, bounds=[0.0, household.ev2.capacity])
            constraints.append(ev2_soc_vars[0] == ev2_soc)
            for t in range(planning_horizon):
                availability = 1.0 if (ev2_home_profile[t] > 0 or ev2_station_profile[t] > 0) else 0.0
                driving_load = ev2_load_profile[t] if availability <= 0.0 else 0.0
                constraints.append(ev2_charge[t] <= household.ev2.max_charge * availability)
                constraints.append(
                    ev2_soc_vars[t + 1]
                    == ev2_soc_vars[t]
                    + ev2_charge[t] * duration_hours * household.ev2.efficiency
                    - driving_load * duration_hours / household.ev2.efficiency
                )
                if ev2_home_profile[t] > 0:
                    pass
                elif ev2_station_profile[t] > 0:
                    objective_terms.append(ev2_buy_price_profile[t] * ev2_charge[t])

            for deadline, target_soc in sorted((self.ev2_soc_targets or {}).items()):
                target_index = deadline - current_timestep + 1
                if 0 <= target_index <= planning_horizon:
                    target_soc_kwh = target_soc * household.ev2.capacity
                    constraints.append(ev2_soc_vars[target_index] >= target_soc_kwh)
        else:
            ev2_charge = None

        for t in range(planning_horizon):
            ev1_home_load = ev1_charge[t] if ev1_charge is not None and ev1_home_profile[t] > 0 else 0.0
            ev2_home_load = ev2_charge[t] if ev2_charge is not None and ev2_home_profile[t] > 0 else 0.0

            net_load = (
                base_load_profile[t]
                - pv_profile[t]
                + (bess_power[t] if bess_power is not None else 0.0)
                + ev1_home_load
                + ev2_home_load
            )
            constraints.append(net_load == grid_import[t] - grid_export[t])
            objective_terms.append(
                duration_hours * buy_price_profile[t] * grid_import[t]
                - duration_hours * sell_price_profile[t] * grid_export[t]
            )

        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except Exception:
            problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status not in {"optimal", "optimal_inaccurate"}:
            return {
                "bess_power": 0.0,
                "ev1_power": 0.0,
                "ev2_power": 0.0,
            }

        return {
            "bess_power": float(bess_power[0].value if bess_power is not None else 0.0),
            "ev1_power": float(ev1_charge[0].value if ev1_charge is not None else 0.0),
            "ev2_power": float(ev2_charge[0].value if ev2_charge is not None else 0.0),
        }
