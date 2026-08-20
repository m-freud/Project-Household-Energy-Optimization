# paste this to enable src. imports# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.base_controller import BaseController
from src.simulation.controllers.mpc.config.device_buffer_config import DeviceBufferConfig
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.controllers.mpc.predictors.history_avg.history_avg_predictor import HistoryAveragePredictor
from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.controllers.stepwise.step_functions.basic_examples import no_control
from src.simulation.controllers.stepwise.step_functions.linear.linear import even_linear_policy, fast_charge_policy
from src.simulation.controllers.stepwise.step_functions.linear.price_aware_linear import price_aware_linear
from src.simulation.controllers.stepwise.step_functions.waterfall.waterfall import waterfall_policy
from src.simulation.controllers.stepwise.stepwise_controller import StepwiseController
from src.simulation.devices.bess import BESS
from src.simulation.devices.ev import EV
from src.simulation.devices.pv import PV
from src.simulation.household import Household
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import Scenario, scenarios as scenario_catalog
from src.sqlite_connection import sqlite_conn, fetch_multiple_timeseries
from typing import Any, Callable, TypeVar
from xgboost import XGBClassifier, XGBRegressor
import argparse
import numpy as np
import pickle


TXGBModel = TypeVar("TXGBModel", XGBRegressor, XGBClassifier)
TLoadedModel = TypeVar("TLoadedModel")


DEFAULT_HISTORY_MEASUREMENTS = [
    "net_load",
    "net_cost",
    "total_consumption",
    "total_cost",
    "bess_soc",
    "bess_power",
    "ev1_soc",
    "ev1_power",
    "ev2_soc",
    "ev2_power",
]


class ConstantRegressor:
    """Simple fallback regressor for smoke runs when no trained model exists."""

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def predict(self, features):
        n_rows = len(features) if hasattr(features, "__len__") else 1
        return np.full(int(n_rows), self.value, dtype=float)


def build_function_controller(
    household: Household,
    scenario: Scenario,
    *,
    name: str,
    step_function: Callable[..., dict],
) -> StepwiseController:
    _ = (household, scenario)
    return StepwiseController(name=name, step_function=step_function)


def build_mpc_controller(
    household: Household,
    scenario: Scenario,
    *,
    name: str,
    horizon: int = 96,
    predictor: BasePredictor,
    duration_hours: float = 0.25,
    buffer_config: DeviceBufferConfig | None = None,
) -> MPCController:
    return MPCController(
        name=name,
        household=household,
        scenario=scenario,
        horizon=horizon,
        predictor=predictor,
        duration_hours=duration_hours,
        buffer_config=buffer_config,
    )


def make_function_controller( # for parallel runs
    name: str,
    step_function: Callable[..., dict],
) -> Callable[[Household, Scenario], StepwiseController]:
    return partial(build_function_controller, name=name, step_function=step_function)


def make_mpc_controller(
    name: str,
    horizon: int,
    predictor: BasePredictor,
    duration_hours: float = 0.25,
    buffer_config: DeviceBufferConfig | None = None,
) -> Callable[[Household, Scenario], MPCController]:
    return partial(
        build_mpc_controller,
        name=name,
        horizon=horizon,
        predictor=predictor,
        duration_hours=duration_hours,
        buffer_config=buffer_config,
    )


def _run_household_worker(player_id: int, run_context: RunContext) -> dict:
    # Worker processes create isolated DB connections; only the parent writes results.
    from src.sqlite_connection import create_sqlite_connection

    worker_conn = create_sqlite_connection()
    try:
        sim = Simulation(worker_conn, ensure_schema=False)
        scenario = run_context.scenario

        start_time = run_context.start_time
        if start_time < 1 or start_time > sim.num_timesteps:
            raise ValueError(f"start_time must be between 1 and {sim.num_timesteps}")

        household = sim.create_household(player_id, run_context)
        controller = sim.create_controller(household, run_context)

        for t in range(start_time, sim.num_timesteps + 1):
            sim.step(household, controller, scenario, duration_hours=sim.duration_hours, time=t)

        return {
            "run_id": run_context.run_id,
            "policy": controller.name,
            "scenario": scenario.name,
            "player_id": household.player_id,
            "has_pv": household.has_pv,
            "has_bess": household.has_bess,
            "total_cost": household.total_cost,
            "total_consumption": household.total_consumption,
            "net_cost": sum(household.history["net_cost"].values()) * sim.duration_hours,
            "net_load": sum(household.history["net_load"].values()) * sim.duration_hours,
            "target_met_bess": household.has_met_final_target("bess"),
            "target_met_ev1": household.has_met_final_target("ev1"),
            "target_met_ev2": household.has_met_final_target("ev2"),
            "target_met_all_bess": household.has_met_all_targets("bess"),
            "target_met_all_ev1": household.has_met_all_targets("ev1"),
            "target_met_all_ev2": household.has_met_all_targets("ev2"),
            "soc_at_deadline_bess": household.soc_at_deadline("bess"),
            "soc_at_deadline_ev1": household.soc_at_deadline("ev1"),
            "soc_at_deadline_ev2": household.soc_at_deadline("ev2"),
            "history": {
                measurement: list(household.history[measurement].items())
                for measurement in DEFAULT_HISTORY_MEASUREMENTS
            },
        }
    finally:
        worker_conn.close()


class Simulation:
    def __init__(self, sqlite_conn, ensure_schema: bool = True):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()

        self.num_households = 250
        self.num_timesteps = 96
        self.duration_hours = float(RuntimeConfig.DURATION_TIMESTEP)

        self.env_inputs = [ # time series table names
            "base_load",
            "pv_gen",
            "ev1_load",
            "ev2_load",
            "buy_price",
            "sell_price",
            "ev1_at_home",
            "ev1_at_charging_station",
            "ev1_buy_price",
            "ev1_max_charge",
            "ev2_at_home",
            "ev2_at_charging_station",
            "ev2_buy_price",
            "ev2_max_charge",
        ]

        self.household_profiles = {}

        self.current_timestep = 1  # current timestep in the simulation.

        if ensure_schema:
            self.sqlite_cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    run_id TEXT,
                    policy TEXT,
                    scenario TEXT,
                    player_id INTEGER,
                    has_pv BOOLEAN,
                    has_bess BOOLEAN,
                    total_cost REAL,
                    total_consumption REAL,
                    net_cost REAL,
                    net_load REAL
                )''')

            self._ensure_results_columns()


    def _ensure_results_columns(self):
        columns = {
            row[1]
            for row in self.sqlite_cursor.execute("PRAGMA table_info(results)").fetchall()
        }

        required_columns = [
            ("run_id", "TEXT"),
            ("net_cost", "REAL"),
            ("net_load", "REAL"),
            ("target_met_bess", "BOOLEAN"),
            ("target_met_ev1", "BOOLEAN"),
            ("target_met_ev2", "BOOLEAN"),
            ("target_met_all_bess", "BOOLEAN"),
            ("target_met_all_ev1", "BOOLEAN"),
            ("target_met_all_ev2", "BOOLEAN"),
            ("soc_at_deadline_bess", "REAL"),
            ("soc_at_deadline_ev1", "REAL"),
            ("soc_at_deadline_ev2", "REAL"),
        ]

        for column_name, column_type in required_columns:
            if column_name not in columns:
                self.sqlite_cursor.execute(
                    f"ALTER TABLE results ADD COLUMN {column_name} {column_type}"
                )

        self.sqlite_conn.commit()


    def create_household(self, player_id:int, run_context: RunContext, bess_for_all: bool=False):
        scenario = run_context.scenario
        start_time = run_context.start_time
        household = Household(player_id=player_id, start_time=start_time, scenario=scenario)

        household.base_cost = self.sqlite_cursor.execute(
            "SELECT fixed_costs FROM fixed_costs WHERE player_id = ?",
            (player_id,)
        ).fetchone()[0]

        has_pv, has_bess = self.sqlite_cursor.execute(
            "SELECT has_pv, has_bess FROM player_pv_bess WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        # plug in PV
        if has_pv:
            household.pv = PV()

        # plug in BESS
        if has_bess or bess_for_all:
            bess_data = self.sqlite_cursor.execute(
                '''SELECT capacity, charge, discharge, efficiency, initial_soc
                FROM bess
                WHERE player_id = ?''',
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data
            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, soc=initial_soc, name="bess")

        def _load_ev_data(table_name: str) -> tuple[float, float, float, float, float, float | None, float | None, float | None]:
            columns = {
                row[1]
                for row in self.sqlite_cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            }
            selected_columns = ["capacity", "charge", "discharge", "efficiency", "initial_soc"]
            if "station_max_charge" in columns:
                selected_columns.append("station_max_charge")
            elif "power" in columns:
                selected_columns.append("power")
            if "charge_slowest" in columns:
                selected_columns.append("charge_slowest")
            if "station_price" in columns:
                selected_columns.append("station_price")
            elif "price_at_public_charge_station_eur" in columns:
                selected_columns.append("price_at_public_charge_station_eur")

            row = self.sqlite_cursor.execute(
                f"SELECT {', '.join(selected_columns)} FROM {table_name} WHERE player_id = ?",
                (player_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No row found for player_id {player_id} in {table_name}")

            row_map = dict(zip(selected_columns, row))
            capacity = row_map["capacity"]
            max_charge = row_map["charge"]
            max_discharge = row_map["discharge"]
            efficiency = row_map["efficiency"]
            initial_soc = row_map["initial_soc"]
            station_max_charge = row_map.get("station_max_charge", row_map.get("power"))
            charge_slowest = row_map.get("charge_slowest")
            station_buy_price = row_map.get("station_price", row_map.get("price_at_public_charge_station_eur"))

            return (
                capacity,
                max_charge,
                max_discharge,
                efficiency,
                initial_soc,
                station_max_charge,
                charge_slowest,
                station_buy_price,
            )

        max_home_charge = None
        max_home_charge_row = self.sqlite_cursor.execute(
            "SELECT max_home_charge FROM max_home_charge WHERE player_id = ?",
            (player_id,),
        ).fetchone()
        if max_home_charge_row is not None and max_home_charge_row[0] is not None:
            max_home_charge = float(max_home_charge_row[0])

        # plug in EV1
        (
            capacity,
            max_charge,
            max_discharge,
            efficiency,
            initial_soc,
            station_max_charge,
            charge_slowest,
            station_buy_price,
        ) = _load_ev_data("ev1")
        household.ev1 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc, name="ev1")
        if max_home_charge is not None:
            household.ev1.max_home_charge = float(max_home_charge)
        if station_max_charge is not None:
            household.ev1.max_station_charge = float(station_max_charge)
        if charge_slowest is not None:
            household.ev1.charge_slowest = float(charge_slowest)
        if station_buy_price is not None:
            household.ev1_station_buy_price = float(station_buy_price)

        # plug in EV2
        (
            capacity,
            max_charge,
            max_discharge,
            efficiency,
            initial_soc,
            station_max_charge,
            charge_slowest,
            station_buy_price,
        ) = _load_ev_data("ev2")
        household.ev2 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc, name="ev2")
        if max_home_charge is not None:
            household.ev2.max_home_charge = float(max_home_charge)
        if station_max_charge is not None:
            household.ev2.max_station_charge = float(station_max_charge)
        if charge_slowest is not None:
            household.ev2.charge_slowest = float(charge_slowest)
        if station_buy_price is not None:
            household.ev2_station_buy_price = float(station_buy_price)

        # set initial SOCs from scenario
        for component in [household.bess, household.ev1, household.ev2]:
            if component:
                device_config = getattr(scenario, component.name, None)
                if device_config:
                    component.soc = device_config.start_soc * component.capacity
                    # print(f"Set initial SOC of {component.name} to {component.soc} kWh which is {device_config.start_soc*100}% of capacity")
                else:
                    print(f"No scenario config found for {component.name}, using default initial SOC of {component.soc} kWh")


        # only query once for all profiles. the simulation now knows the future
        self.household_profiles = fetch_multiple_timeseries(
            self.sqlite_cursor,
            player_id,
            measurements=self.env_inputs
        )

        # household gets access to prices over day 
        household.buy_price_day_profile = self.household_profiles["buy_price"]
        household.sell_price_day_profile = self.household_profiles["sell_price"]
        household.oracle_profiles = self.household_profiles # just pass them through for easy lookup if needed

        # set default drive load for EVs based on first non-zero value in the load profile
        # technically this is an illegal prediction, but it is a good proxy for "yesterdays last value"
        # and we have to start somewhere.
        def _first_non_zero_value(values: list[float]) -> float:
            for value in values:
                value_f = float(value)
                if value_f > 0.0:
                    return value_f
            return 0.0

        if household.ev1:
            ev1_load_profile = self.household_profiles.get("ev1_load", [])
            household.ev1.default_drive_load = _first_non_zero_value(ev1_load_profile)
        if household.ev2:
            ev2_load_profile = self.household_profiles.get("ev2_load", [])
            household.ev2.default_drive_load = _first_non_zero_value(ev2_load_profile)

        return household
    

    def create_controller(self, household: Household, run_context: RunContext) -> BaseController:
        if run_context.controller_factory is None:
            raise ValueError("RunContext must define a controller_factory")

        controller = run_context.controller_factory(household, run_context.scenario)
        if not isinstance(controller, BaseController):
            raise TypeError("Controller factory must return a BaseController instance")
        return controller


    # perform a single simulation step for 1 house
    def step(
            self,
            household: Household,
            controller: BaseController,
            scenario: Scenario,
            duration_hours: float = 0.25,
            time=1,
        ):
        self.current_timestep = time
        self.update_household_inputs(household)
        controls = controller.set_controls(household, scenario)
        household.apply_controls(controls, duration_hours=duration_hours)
        household.update_history()


    def update_household_inputs(self, household: Household):
        # update the time
        household.current_timestep = self.current_timestep
        profile_time_index = self.current_timestep - 1

        profiles = self.household_profiles
        ev1 = household.ev1
        ev2 = household.ev2
        pv = household.pv

        # update base load
        household.base_load = profiles["base_load"][profile_time_index]

        # update pv
        if pv:
            pv.generation = profiles["pv_gen"][profile_time_index]

        # update ev1
        if ev1:
            ev1.load = profiles["ev1_load"][profile_time_index]
            ev1.at_home = profiles["ev1_at_home"][profile_time_index] > 0
            ev1.at_charging_station = profiles["ev1_at_charging_station"][profile_time_index] > 0
            ev1.buy_price = profiles["ev1_buy_price"][profile_time_index]
            ev1.max_charge = profiles["ev1_max_charge"][profile_time_index]

        # update ev2
        if ev2:
            ev2.load = profiles["ev2_load"][profile_time_index]
            ev2.at_home = profiles["ev2_at_home"][profile_time_index] > 0
            ev2.at_charging_station = profiles["ev2_at_charging_station"][profile_time_index] > 0
            ev2.buy_price = profiles["ev2_buy_price"][profile_time_index]
            ev2.max_charge = profiles["ev2_max_charge"][profile_time_index]

        # update buy / sell prices
        household.buy_price = profiles["buy_price"][profile_time_index]
        household.sell_price = profiles["sell_price"][profile_time_index]


    def run_household(self, player_id, run_context: RunContext, write_results_to_sqlite: bool = True) -> Household:
        scenario = run_context.scenario
        current_run_id = run_context.run_id

        start_time = run_context.start_time
        if start_time < 1 or start_time > self.num_timesteps:
            raise ValueError(f"start_time must be between 1 and {self.num_timesteps}")

        household = self.create_household(player_id, run_context)
        controller = self.create_controller(household, run_context)

        print(f"running household {player_id} in scenario {scenario.name} with controller {controller.name}, run_id = {current_run_id}")

        for t in range(start_time, self.num_timesteps + 1):
            self.step(household, controller, scenario, duration_hours=self.duration_hours, time=t)

        if write_results_to_sqlite:
            self.load_household_history_to_sqlite(household, policy_name=controller.name, scenario_name=scenario.name)
            self.load_household_results_to_sqlite(
                household,
                policy_name=controller.name,
                scenario_name=scenario.name,
                run_id=current_run_id,
            )

        return household


    def run_all_households(
        self,
        run_context: RunContext,
        household_ids: list[int] | None = None,
        max_households: int | None = None,
        parallel_households: bool = True,
        parallel_workers: int | None = 6, # optimal number depends on CPUs of your machine
        write_results_to_sqlite: bool = True,
    ):
        if household_ids is None:
            selected_households = list(range(1, self.num_households + 1))
        else:
            selected_households = [
                player_id
                for player_id in household_ids
                if 1 <= player_id <= self.num_households
            ]

        if max_households is not None and max_households >= 0:
            selected_households = selected_households[:max_households]

        if parallel_households and len(selected_households) > 1:
            return self.run_all_households_parallel(
                run_context,
                selected_households,
                parallel_workers=parallel_workers,
            )

        results = {
            "total_costs": [],
            "total_consumptions": [],
        }

        for player_id in selected_households:
            household = self.run_household(player_id, run_context, write_results_to_sqlite=write_results_to_sqlite)
            results["total_costs"].append(float(household.total_cost))
            results["total_consumptions"].append(float(household.total_consumption))

        return results


    def run_all_households_parallel(
        self,
        run_context: RunContext,
        selected_households: list[int],
        parallel_workers: int | None = 6,
        write_results_to_sqlite: bool = True,
    ):
        results = {
            "total_costs": [],
            "total_consumptions": [],
        }

        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_household = {
                executor.submit(_run_household_worker, player_id, run_context): player_id
                for player_id in selected_households
            }

            for future in as_completed(future_to_household):
                player_id = future_to_household[future]
                payload = future.result()
                if write_results_to_sqlite:
                    self.load_history_payload_to_sqlite(payload)
                    self.load_results_payload_to_sqlite(payload)
                results["total_costs"].append(payload["total_cost"])
                results["total_consumptions"].append(payload["total_consumption"])
                print(
                    f"completed household {player_id} in scenario {payload['scenario']} "
                    f"with controller {payload['policy']}, run_id = {payload['run_id']}"
                )

        return results


    def run_batch(
        self,
        run_contexts: list[RunContext],
        household_ids: list[int] | None = None,
        max_households: int | None = None,
        parallel_households: bool = False,
        parallel_workers: int | None = 6,
        write_results_to_sqlite: bool = True,
    ):
        results = {
            "total_costs": [],
            "total_consumptions": [],
        }

        for run_context in run_contexts:
            batch_results = self.run_all_households(
                run_context,
                household_ids=household_ids,
                max_households=max_households,
                parallel_households=parallel_households,
                parallel_workers=parallel_workers,
                write_results_to_sqlite=write_results_to_sqlite,
            )
            if batch_results is not None:
                results["total_costs"].extend(batch_results.get("total_costs", []))
                results["total_consumptions"].extend(batch_results.get("total_consumptions", []))

        return results


    def load_household_results_to_sqlite(self, household: Household, policy_name:str="no_control", scenario_name:str="default_scenario", run_id:str|None=None):
        # extract dict from household, load to sqlite
        payload = {
            "run_id": run_id,
            "policy": policy_name,
            "scenario": scenario_name,
            "player_id": household.player_id,
            "has_pv": household.has_pv,
            "has_bess": household.has_bess,
            "total_cost": household.total_cost,
            "total_consumption": household.total_consumption,
            "net_cost": sum(household.history["net_cost"].values()) * 0.25,
            "net_load": sum(household.history["net_load"].values()) * 0.25,
            "target_met_bess": household.has_met_final_target("bess"),
            "target_met_ev1": household.has_met_final_target("ev1"),
            "target_met_ev2": household.has_met_final_target("ev2"),
            "target_met_all_bess": household.has_met_all_targets("bess"),
            "target_met_all_ev1": household.has_met_all_targets("ev1"),
            "target_met_all_ev2": household.has_met_all_targets("ev2"),
            "soc_at_deadline_bess": household.soc_at_deadline("bess"),
            "soc_at_deadline_ev1": household.soc_at_deadline("ev1"),
            "soc_at_deadline_ev2": household.soc_at_deadline("ev2"),
        }

        self.load_results_payload_to_sqlite(payload)


    def load_household_history_to_sqlite(self,
                               household, policy_name,
                               scenario_name, measurements=None):
        # extract dict from household, load to sqlite
        if measurements is None:
            measurements = DEFAULT_HISTORY_MEASUREMENTS

        payload = {
            "policy": policy_name,
            "scenario": scenario_name,
            "player_id": household.player_id,
            "history": {
                measurement: list(household.history[measurement].items())
                for measurement in measurements
            },
        }

        self.load_history_payload_to_sqlite(payload)

    
    def load_results_payload_to_sqlite(self, payload: dict):
        # load dict to sqlite
        self.sqlite_cursor.execute(
            '''
            INSERT INTO results (
            run_id,
            policy, scenario, player_id, has_pv, has_bess, total_cost, total_consumption,
            net_cost, net_load,
            target_met_bess, target_met_ev1, target_met_ev2,
              target_met_all_bess, target_met_all_ev1, target_met_all_ev2,
            soc_at_deadline_bess, soc_at_deadline_ev1, soc_at_deadline_ev2
            )
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                payload["run_id"],
                payload["policy"],
                payload["scenario"],
                payload["player_id"],
                payload["has_pv"],
                payload["has_bess"],
                payload["total_cost"],
                payload["total_consumption"],
                payload["net_cost"],
                payload["net_load"],
                payload["target_met_bess"],
                payload["target_met_ev1"],
                payload["target_met_ev2"],
                payload["target_met_all_bess"],
                payload["target_met_all_ev1"],
                payload["target_met_all_ev2"],
                payload["soc_at_deadline_bess"],
                payload["soc_at_deadline_ev1"],
                payload["soc_at_deadline_ev2"],
            ),
        )
        self.sqlite_conn.commit()


    def load_history_payload_to_sqlite(self, payload: dict):
        # load dict to sqlite
        policy_name = payload["policy"]
        scenario_name = payload["scenario"]
        player_id = payload["player_id"]
        history = payload["history"]

        for measurement in DEFAULT_HISTORY_MEASUREMENTS:
            self.sqlite_cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS {measurement} (
                    player_id INTEGER,
                    scenario TEXT,
                    policy TEXT,
                    period INTEGER,
                    value REAL
                )'''
            )

            rows = history.get(measurement, [])
            if not rows:
                continue

            self.sqlite_cursor.executemany(
                f'''
                INSERT INTO {measurement} (
                player_id, scenario, policy, period, value
                ) VALUES (?, ?, ?, ?, ?)
                ''',
                [
                    (player_id, scenario_name, policy_name, int(period), float(value))
                    for period, value in rows
                ],
            )

        self.sqlite_conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run household energy simulation")
    parser.add_argument(
        "--controllers",
        default="mpc_xgb,mpc_ridge,mpc_rf",
        help="Comma-separated controller names (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        default="default_scenario",
        help="Comma-separated scenario names (default: all)",
    )
    parser.add_argument(
        "--households",
        default="1",
        help="Comma-separated household ids (default: all)",
    )
    parser.add_argument(
        "--max-households",
        type=int,
        default=None,
        help="Limit number of households after filtering",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        default=1,
        help="Simulation start timestep (1..96)",
    )
    parser.add_argument(
        "--parallel-households",
        action="store_true",
        default=True,
        help="Run households in parallel using process workers",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=6,
        help="Number of worker processes for --parallel-households (default: Python decides)",
    )
    args = parser.parse_args()

    # Create a simulation instance
    # TBD this is a bad idea we actually want one simulation per household
    sim = Simulation(sqlite_conn)

    def _load_xgb_model(model_path: Path, model_class: type[TXGBModel]) -> TXGBModel:
        model = model_class()
        try:
            model.load_model(str(model_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load XGBoost model from '{model_path}'.") from exc
        return model

    def _load_pickle_model(model_path: Path):
        try:
            with open(model_path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to load pickle model from '{model_path}'.") from exc

    def _single_model_path(family: str, target: str) -> Path:
        family_key = str(family).lower()
        family_config = RuntimeConfig.MODEL_FAMILY_CONFIGS[family_key]
        model_dir = family_config.target_model_dirs[target]
        suffix = family_config.file_suffix

        preferred = Path(model_dir / f"model{suffix}")
        if preferred.exists():
            return preferred

        fallback = Path(model_dir / f"fold_1{suffix}")
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"Missing single model for {family_key}:{target}. Looked for {preferred} and {fallback}."
        )

    def _build_predictor(family: str) -> MLPredictor:
        family_key = str(family).lower()

        if family_key == "xgb":
            base_load_model = _load_xgb_model(_single_model_path("xgb", "base_load"), XGBRegressor)
            pv_gen_model = _load_xgb_model(_single_model_path("xgb", "pv_gen"), XGBRegressor)
            ev1_status_model = _load_xgb_model(_single_model_path("xgb", "ev1_status"), XGBClassifier)
            ev2_status_model = _load_xgb_model(_single_model_path("xgb", "ev2_status"), XGBClassifier)
        elif family_key in {"rf", "ridge"}:
            base_load_model = _load_pickle_model(_single_model_path(family_key, "base_load"))
            pv_gen_model = _load_pickle_model(_single_model_path(family_key, "pv_gen"))
            ev1_status_model = _load_pickle_model(_single_model_path(family_key, "ev1_status"))
            ev2_status_model = _load_pickle_model(_single_model_path(family_key, "ev2_status"))
        else:
            valid = ", ".join(sorted(RuntimeConfig.MODEL_FAMILY_CONFIGS.keys()))
            raise ValueError(f"Unknown model family '{family_key}'. Expected one of: {valid}")

        return MLPredictor(
            base_load_model=base_load_model,
            pv_gen_model=pv_gen_model,
            ev1_status_model=ev1_status_model,
            ev2_status_model=ev2_status_model,
        )

    predictor_xgb = _build_predictor("xgb")
    predictor_rf = _build_predictor("rf")
    predictor_ridge = _build_predictor("ridge")
    predictor_modular = ModularPredictor(
        default_predictor=OraclePredictor(),
        target_predictors={
            "base_load": predictor_xgb,
            "pv_gen": OraclePredictor(),
            "ev_status": HistoryAveragePredictor(conf_interval_frct=0.0),
        },
    )
    _oracle = OraclePredictor()
    predictor_modular_oracle = ModularPredictor(
        default_predictor=HistoryAveragePredictor(conf_interval_frct=0.0),
        target_predictors={
            "base_load": _oracle,
            "pv_gen": _oracle,
            "ev_status": _oracle,
        },
    )
    predictor_modular_avg = ModularPredictor(
        default_predictor=HistoryAveragePredictor(conf_interval_frct=0.0),
        target_predictors={
            "base_load": HistoryAveragePredictor(conf_interval_frct=0.0),
            "pv_gen": HistoryAveragePredictor(conf_interval_frct=0.0),
            "ev_status": HistoryAveragePredictor(conf_interval_frct=0.0),
        },
    )

    controller_factories_by_name = {
        "no_control": make_function_controller("no_control", no_control),
        "fast_charge": make_function_controller("fast_charge", fast_charge_policy),
        "even_linear": make_function_controller("even_linear", even_linear_policy),
        "price_aware_linear": make_function_controller(
            "price_aware_linear",
            partial(price_aware_linear, base_behaviour="even_linear"),
        ),
        "waterfall": make_function_controller("waterfall", waterfall_policy),
        "mpc_oracle": make_mpc_controller("mpc_oracle", horizon=96, predictor=OraclePredictor()),
        "mpc_history_avg": make_mpc_controller(
            "mpc_history_avg",
            horizon=96,
            predictor=HistoryAveragePredictor(
                conf_interval_frct=0.0,
            ),
        ),
        "mpc_xgb": make_mpc_controller(
            "mpc_xgb",
            horizon=96,
            predictor=predictor_xgb,
        ),
        "mpc_rf": make_mpc_controller(
            "mpc_rf",
            horizon=96,
            predictor=predictor_rf,
        ),
        "mpc_ridge": make_mpc_controller(
            "mpc_ridge",
            horizon=96,
            predictor=predictor_ridge,
        ),
        "mpc_modular": make_mpc_controller(
            "mpc_modular",
            horizon=96,
            predictor=predictor_modular,
        ),
        "mpc_modular_oracle": make_mpc_controller(
            "mpc_modular_oracle",
            horizon=96,
            predictor=predictor_modular_oracle,
        ),
        "mpc_modular_avg": make_mpc_controller(
            "mpc_modular_avg",
            horizon=96,
            predictor=predictor_modular_avg,
        ),
        "mpc_modular_xgb": make_mpc_controller(
            "mpc_modular_xgb",
            horizon=96,
            predictor=ModularPredictor(
                default_predictor=predictor_xgb,
                target_predictors={
                    "base_load": predictor_xgb,
                    "pv_gen": predictor_xgb,
                    "ev_status": predictor_xgb,
                },
            ),
        )
    }

    scenarios_by_name = scenario_catalog

    if args.controllers == "all":
        selected_controller_names = list(controller_factories_by_name.keys())
    else:
        requested_controller_names = [
            name.strip() for name in args.controllers.split(",") if name.strip()
        ]
        unknown_controller_names = [
            name for name in requested_controller_names if name not in controller_factories_by_name
        ]
        if unknown_controller_names:
            raise ValueError(f"Unknown controllers: {unknown_controller_names}")
        selected_controller_names = requested_controller_names

    if args.scenarios == "all":
        selected_scenarios = list(scenario_catalog.values())
    else:
        requested_scenario_names = [
            name.strip() for name in args.scenarios.split(",") if name.strip()
        ]
        unknown_scenario_names = [
            name for name in requested_scenario_names if name not in scenarios_by_name
        ]
        if unknown_scenario_names:
            raise ValueError(f"Unknown scenarios: {unknown_scenario_names}")
        selected_scenarios = [scenarios_by_name[name] for name in requested_scenario_names]

    if args.households == "all":
        selected_household_ids = None
    elif args.households == "test_set":
        selected_household_ids = list(RuntimeConfig.INDEPENDENT_TEST_SET)
    else:
        selected_household_ids = [
            int(token.strip()) for token in args.households.split(",") if token.strip()
        ]

    run_contexts = [
        RunContext(
            controller_factory=controller_factories_by_name[controller_name],
            controller_name=controller_name,
            scenario=scenario,
            start_time=args.start_time,
        )
        for controller_name in selected_controller_names
        for scenario in selected_scenarios
    ]

    sim.run_batch(
        run_contexts,
        household_ids=selected_household_ids,
        max_households=args.max_households,
        parallel_households=args.parallel_households,
        parallel_workers=args.parallel_workers,
    )
