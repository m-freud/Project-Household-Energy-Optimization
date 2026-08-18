from dataclasses import dataclass
from typing import Callable, cast

import cvxpy as cp
import numpy as np

from src.runtime_conig import RuntimeConfig
from src.simulation.controllers.base_controller import BaseController
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario


@dataclass
class DeviceOraclePrediction:
    at_home: np.ndarray
    at_station: np.ndarray
    load: np.ndarray
    buy_price: np.ndarray
    max_charge: np.ndarray


@dataclass
class OraclePrediction:
    start_timestep: int
    horizon: int
    base_load: np.ndarray
    pv_gen: np.ndarray
    buy_price: np.ndarray
    sell_price: np.ndarray
    ev1: DeviceOraclePrediction | None
    ev2: DeviceOraclePrediction | None


def _to_kwh(value: float, capacity: float) -> float:
    return value * capacity if value <= 1.0 else value


def _safe_clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def _is_available(at_home: np.ndarray, at_station: np.ndarray) -> np.ndarray:
    return np.clip(at_home + at_station, 0.0, 1.0)


def _as_float_array(values: list[float], length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size >= length:
        return arr[:length]
    if arr.size == 0:
        return np.zeros(length, dtype=float)
    pad = np.full(length - arr.size, float(arr[-1]), dtype=float)
    return np.concatenate([arr, pad])


class MPCController(BaseController):
    def __init__(
        self,
        name: str = "mpc_oracle",
        horizon: int = 24,
        target_slack_penalty: float = 5e4,
        power_smoothing_penalty: float = 2e-2,
        throughput_penalty: float = 1e-4,
    ):
        super().__init__(name)
        self.horizon = max(1, int(horizon))
        self.target_slack_penalty = float(target_slack_penalty)
        self.power_smoothing_penalty = float(power_smoothing_penalty)
        self.throughput_penalty = float(throughput_penalty)

    def _build_oracle_prediction(self, household: Household, horizon: int) -> OraclePrediction:
        """
        Oracle helper: reads future trajectories directly from preloaded simulation profiles.
        Replace this function with forecast models later.
        """
        profiles = getattr(household, "oracle_profiles", None)
        if profiles is None:
            raise ValueError("oracle_profiles are missing on household; cannot build MPC prediction")

        start_idx = max(0, household.current_timestep - 1)
        end_idx = start_idx + horizon

        def _slice(key: str) -> np.ndarray:
            values = profiles.get(key, [])
            return _as_float_array(values[start_idx:end_idx], horizon)

        ev1_pred = None
        if household.ev1 is not None:
            ev1_pred = DeviceOraclePrediction(
                at_home=_slice("ev1_at_home"),
                at_station=_slice("ev1_at_charging_station"),
                load=_slice("ev1_load"),
                buy_price=_slice("ev1_buy_price"),
                max_charge=_slice("ev1_max_charge"),
            )

        ev2_pred = None
        if household.ev2 is not None:
            ev2_pred = DeviceOraclePrediction(
                at_home=_slice("ev2_at_home"),
                at_station=_slice("ev2_at_charging_station"),
                load=_slice("ev2_load"),
                buy_price=_slice("ev2_buy_price"),
                max_charge=_slice("ev2_max_charge"),
            )

        return OraclePrediction(
            start_timestep=household.current_timestep,
            horizon=horizon,
            base_load=_slice("base_load"),
            pv_gen=_slice("pv_gen"),
            buy_price=_slice("buy_price"),
            sell_price=_slice("sell_price"),
            ev1=ev1_pred,
            ev2=ev2_pred,
        )

    def _target_kwh(self, scenario: Scenario, device_name: str, capacity: float) -> dict[int, float]:
        device_scenario = getattr(scenario, device_name, None)
        if device_scenario is None or not device_scenario.soc_targets:
            return {}
        return {
            int(deadline): _to_kwh(float(target), capacity)
            for deadline, target in device_scenario.soc_targets.items()
        }

    def _device_bounds_kwh(self, scenario: Scenario, device_name: str, capacity: float) -> tuple[float, float]:
        _ = (scenario, device_name)
        # Keep optimization consistent with simulator physics.
        # Device classes enforce SOC clipping in [0, capacity].
        return 0.0, float(capacity)

    def _add_target_constraints(
        self,
        constraints: list,
        stage_cost_terms: list,
        soc_var,
        target_kwh_by_deadline: dict[int, float],
        start_timestep: int,
        horizon: int,
        max_addable_from: Callable[[int, int], float],
    ) -> None:
        for deadline, target_kwh in sorted(target_kwh_by_deadline.items()):
            rel_idx = deadline - start_timestep
            if rel_idx < 0:
                continue

            # Trajectory floor prevents delaying charge too much when deadlines are outside
            # the visible MPC horizon.
            max_rel = min(horizon, max(0, rel_idx))
            for rel in range(max_rel + 1):
                current_step = start_timestep + rel
                max_addable = max(0.0, float(max_addable_from(current_step, deadline)))
                floor_soc = max(0.0, target_kwh - max_addable)
                if floor_soc <= 0:
                    continue

                slack_floor = cp.Variable(nonneg=True)
                constraints.append(soc_var[rel] + slack_floor >= floor_soc)
                stage_cost_terms.append((0.002 * self.target_slack_penalty) * slack_floor)

            if rel_idx <= horizon:
                constraints.append(soc_var[rel_idx] >= target_kwh)

    def _bess_max_addable_from(
        self,
        current_timestep: int,
        deadline: int,
        max_charge_kw: float,
        efficiency: float,
    ) -> float:
        remaining_steps = max(deadline - current_timestep, 0)
        return remaining_steps * RuntimeConfig.DURATION_TIMESTEP * max_charge_kw * efficiency

    def _ev_max_addable_from(
        self,
        household: Household,
        ev_name: str,
        current_timestep: int,
        deadline: int,
        efficiency: float,
    ) -> float:
        if deadline <= current_timestep:
            return 0.0

        profiles = getattr(household, "oracle_profiles", None)
        if profiles is None:
            return 0.0

        start_idx = max(0, current_timestep - 1)
        end_idx = max(start_idx, deadline - 1)

        at_home = _as_float_array(profiles.get(f"{ev_name}_at_home", [])[start_idx:end_idx], end_idx - start_idx)
        at_station = _as_float_array(profiles.get(f"{ev_name}_at_charging_station", [])[start_idx:end_idx], end_idx - start_idx)
        max_charge = _as_float_array(profiles.get(f"{ev_name}_max_charge", [])[start_idx:end_idx], end_idx - start_idx)

        availability = _is_available(at_home, at_station)
        return float(np.sum(availability * np.maximum(0.0, max_charge)) * RuntimeConfig.DURATION_TIMESTEP * efficiency)

    def _build_fallback_controls(self, household: Household) -> dict[str, float]:
        controls = {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0,
        }

        if household.bess is not None:
            controls["bess_power"] = _safe_clip(
                household.controls.get("bess_power", 0.0),
                -float(household.bess.max_discharge),
                float(household.bess.max_charge),
            )

        if household.ev1 is not None and (household.ev1.at_home or household.ev1.at_charging_station):
            controls["ev1_power"] = _safe_clip(
                household.controls.get("ev1_power", 0.0),
                0.0,
                float(household.ev1.max_charge),
            )

        if household.ev2 is not None and (household.ev2.at_home or household.ev2.at_charging_station):
            controls["ev2_power"] = _safe_clip(
                household.controls.get("ev2_power", 0.0),
                0.0,
                float(household.ev2.max_charge),
            )

        return controls

    def set_controls(self, household: Household, scenario: Scenario, *args, **kwargs):
        horizon = self.horizon
        pred = self._build_oracle_prediction(household, horizon)
        dt = float(RuntimeConfig.DURATION_TIMESTEP)

        constraints = []
        objective_terms = []

        # Grid exchange split: import/export >= 0, with import - export = net_load.
        p_import = cp.Variable(horizon, nonneg=True)
        p_export = cp.Variable(horizon, nonneg=True)

        # BESS variables.
        if household.bess is not None:
            bess = household.bess
            bess_charge = cp.Variable(horizon, nonneg=True)
            bess_discharge = cp.Variable(horizon, nonneg=True)
            bess_soc = cp.Variable(horizon + 1)
            constraints.append(bess_soc[0] == float(bess.soc))

            bess_lo, bess_hi = self._device_bounds_kwh(scenario, "bess", float(bess.capacity))
            constraints += [bess_soc >= bess_lo, bess_soc <= bess_hi]
            constraints += [bess_charge <= float(bess.max_charge)]
            constraints += [bess_discharge <= float(bess.max_discharge)]

            eta_bess = float(bess.efficiency)
            for k in range(horizon):
                constraints.append(
                    bess_soc[k + 1]
                    == bess_soc[k] + dt * (eta_bess * bess_charge[k] - (bess_discharge[k] / max(eta_bess, 1e-6)))
                )

            bess_power = bess_charge - bess_discharge
            objective_terms.append(self.throughput_penalty * cp.sum(bess_charge + bess_discharge))
            if horizon > 1:
                objective_terms.append(
                    self.power_smoothing_penalty
                    * cp.sum_squares(bess_power[1:] - bess_power[:-1])
                )

            bess_targets = self._target_kwh(scenario, "bess", float(bess.capacity))
            self._add_target_constraints(
                constraints,
                objective_terms,
                bess_soc,
                bess_targets,
                pred.start_timestep,
                horizon,
                max_addable_from=lambda current_step, deadline: self._bess_max_addable_from(
                    current_step,
                    deadline,
                    max_charge_kw=float(bess.max_charge),
                    efficiency=float(bess.efficiency),
                ),
            )
        else:
            bess_power = cp.Constant(np.zeros(horizon, dtype=float))

        # EV helper closure to reduce duplicated formulation.
        def _ev_terms(ev_name: str, ev, ev_pred: DeviceOraclePrediction | None):
            if ev is None or ev_pred is None:
                zeros = cp.Constant(np.zeros(horizon, dtype=float))
                return zeros, zeros

            ev_charge = cp.Variable(horizon, nonneg=True)
            ev_soc = cp.Variable(horizon + 1)
            constraints.append(ev_soc[0] == float(ev.soc))

            ev_lo, ev_hi = self._device_bounds_kwh(scenario, ev_name, float(ev.capacity))
            constraints.extend([ev_soc >= ev_lo, ev_soc <= ev_hi])

            availability = _is_available(ev_pred.at_home, ev_pred.at_station)
            max_charge = np.maximum(0.0, ev_pred.max_charge)
            constraints.extend([ev_charge <= cp.multiply(max_charge, availability)])

            unavailable = 1.0 - availability
            driving_draw = unavailable * np.maximum(0.0, ev_pred.load)
            eta_ev = float(ev.efficiency)
            for k in range(horizon):
                constraints.append(
                    ev_soc[k + 1]
                    == ev_soc[k]
                    + dt * (eta_ev * ev_charge[k])
                    - dt * (driving_draw[k] / max(eta_ev, 1e-6))
                )

            if horizon > 1:
                objective_terms.append(
                    self.power_smoothing_penalty
                    * cp.sum_squares(ev_charge[1:] - ev_charge[:-1])
                )

            ev_targets = self._target_kwh(scenario, ev_name, float(ev.capacity))
            self._add_target_constraints(
                constraints,
                objective_terms,
                ev_soc,
                ev_targets,
                pred.start_timestep,
                horizon,
                max_addable_from=lambda current_step, deadline: self._ev_max_addable_from(
                    household,
                    ev_name,
                    current_step,
                    deadline,
                    efficiency=float(ev.efficiency),
                ),
            )

            home_component = cp.multiply(ev_pred.at_home, ev_charge)
            station_component = cp.multiply(np.clip(ev_pred.at_station - ev_pred.at_home, 0.0, 1.0), ev_charge)
            station_cost = cp.sum(cp.multiply(ev_pred.buy_price, station_component)) * dt
            objective_terms.append(station_cost)

            return home_component, ev_charge

        ev1_home, ev1_total = _ev_terms("ev1", household.ev1, pred.ev1)
        ev2_home, ev2_total = _ev_terms("ev2", household.ev2, pred.ev2)

        for k in range(horizon):
            net_load_k = pred.base_load[k] + bess_power[k] + ev1_home[k] + ev2_home[k] - pred.pv_gen[k]
            constraints.append(p_import[k] - p_export[k] == net_load_k)

        grid_cost = cp.sum(cp.multiply(pred.buy_price, p_import) - cp.multiply(pred.sell_price, p_export)) * dt
        objective_terms.append(grid_cost)

        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            return self._build_fallback_controls(household)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return self._build_fallback_controls(household)

        controls = {"bess_power": 0.0, "ev1_power": 0.0, "ev2_power": 0.0}

        bess_value = cast(np.ndarray | None, getattr(bess_power, "value", None))
        if household.bess is not None and bess_value is not None:
            controls["bess_power"] = _safe_clip(
                float(bess_value[0]),
                -float(household.bess.max_discharge),
                float(household.bess.max_charge),
            )

        ev1_value = cast(np.ndarray | None, getattr(ev1_total, "value", None))
        if household.ev1 is not None and ev1_value is not None:
            ev1_limit = float(household.ev1.max_charge) if (household.ev1.at_home or household.ev1.at_charging_station) else 0.0
            controls["ev1_power"] = _safe_clip(float(ev1_value[0]), 0.0, ev1_limit)

        ev2_value = cast(np.ndarray | None, getattr(ev2_total, "value", None))
        if household.ev2 is not None and ev2_value is not None:
            ev2_limit = float(household.ev2.max_charge) if (household.ev2.at_home or household.ev2.at_charging_station) else 0.0
            controls["ev2_power"] = _safe_clip(float(ev2_value[0]), 0.0, ev2_limit)

        return controls
