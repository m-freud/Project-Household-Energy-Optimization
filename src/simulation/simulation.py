# paste this to enable src. imports# paste this to enable src. imports
from pathlib import Path
import sys


# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.base_controller import BaseController
from src.simulation.controllers.mpc.config.device_buffer_config import DeviceBufferConfig
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.devices.bess import BESS
from src.simulation.devices.ev import EV
from src.simulation.devices.pv import PV
from src.simulation.household import Household
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import Scenario, scenarios as scenario_catalog
from src.sqlite_connection import (
    create_results_connection,
    fetch_multiple_timeseries,
    source_sqlite_conn,
)
from src.simulation._sim_utils import build_controller_factories, build_parser, make_mpc_controller
from src.simulation._sim_utils.save_results import (
    DEFAULT_HISTORY_MEASUREMENTS,
    load_history_payload_to_sqlite,
    load_household_history_to_sqlite,
    load_household_results_to_sqlite,
    load_results_payload_to_sqlite,
)
from typing import Callable


_WORKER_RUN_CONTEXT: RunContext | None = None

def _initialize_household_worker(run_context: RunContext) -> None:
    global _WORKER_RUN_CONTEXT
    _WORKER_RUN_CONTEXT = run_context


def _run_household_worker(
    player_id: int,
    run_context: RunContext | None = None,
) -> dict:
    # Worker processes create isolated DB connections; only the parent writes results.
    from src.sqlite_connection import create_source_connection

    if run_context is None:
        run_context = _WORKER_RUN_CONTEXT
    if run_context is None:
        raise RuntimeError("Household worker has no run context.")

    worker_conn = create_source_connection()
    try:
        sim = Simulation(
            worker_conn,
            ensure_results_table=False,
            results_conn=worker_conn,
        )
        scenario = run_context.scenario

        start_time = run_context.start_time
        if start_time < 1 or start_time > sim.num_timesteps:
            raise ValueError(f"start_time must be between 1 and {sim.num_timesteps}")

        household = sim.create_household(player_id, run_context)
        controller = sim.create_controller(household, run_context)
        print(
            f"running household {player_id} in scenario {scenario.name} "
            f"with controller {controller.name}, run_id={run_context.run_id}",
            flush=True,
        )
        for t in range(start_time, sim.num_timesteps + 1):
            if controller.name in {
                "mpc_single_step",
                "mpc_multi_step",
                "mpc_composite_horizons",
            } and (t == start_time or t % 8 == 0 or t == sim.num_timesteps):
                print(
                    f"{controller.name}: household {player_id}, timestep {t}/{sim.num_timesteps}",
                    flush=True,
                )
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
    def __init__(
            self,
            sqlite_conn,
            ensure_results_table: bool = True,
            results_conn=None
            ):
        self.sqlite_conn = sqlite_conn
        self.results_conn = results_conn or create_results_connection()
        
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.results_cursor = self.results_conn.cursor()

        self.num_households = 250
        self.num_timesteps = 96
        self.duration_hours = float(RuntimeConfig.DURATION_TIMESTEP)
        self.current_timestep = 1

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

        if ensure_results_table:
            self._ensure_results_table()


    def _ensure_results_table(self):
        self.results_cursor.execute('''
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
                net_load REAL,
                target_met_bess BOOLEAN,
                target_met_ev1 BOOLEAN,
                target_met_ev2 BOOLEAN,
                target_met_all_bess BOOLEAN,
                target_met_all_ev1 BOOLEAN,
                target_met_all_ev2 BOOLEAN,
                soc_at_deadline_bess REAL,
                soc_at_deadline_ev1 REAL,
                soc_at_deadline_ev2 REAL
            )''')
        
        self.results_conn.commit()


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
                else:
                    print(f"No scenario config found for {component.name}, using default initial SOC of {component.soc} kWh")


        # query full day profiles. the simulation now knows the future
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
        # technically this is an illegal prediction, but it is a good proxy for "yesterdays last value",
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
            if controller.name in {
                "mpc_single_step",
                "mpc_multi_step",
                "mpc_composite_horizons",
            } and (t == start_time or t % 8 == 0 or t == self.num_timesteps):
                print(
                    f"{controller.name}: household {player_id}, timestep {t}/{self.num_timesteps}",
                    flush=True,
                )
            self.step(household, controller, scenario, duration_hours=self.duration_hours, time=t)

        if write_results_to_sqlite:
            load_household_history_to_sqlite(
                self.results_conn,
                household,
                policy_name=controller.name,
                scenario_name=scenario.name,
                run_id=current_run_id,
            )
            load_household_results_to_sqlite(
                self.results_conn,
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
                write_results_to_sqlite=write_results_to_sqlite,
            )

        results = {
            "net_costs": [],
            "net_loads": [],
            "total_costs": [],
            "total_consumptions": [],
        }

        for player_id in selected_households:
            household = self.run_household(player_id, run_context, write_results_to_sqlite=write_results_to_sqlite)
            results["net_costs"].append(household.net_cost)
            results["net_loads"].append(household.net_load)
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
            "net_costs": [],
            "net_loads": [],
            "total_costs": [],
            "total_consumptions": [],
        }

        with ProcessPoolExecutor(
            max_workers=parallel_workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_household_worker,
            initargs=(run_context,),
        ) as executor:
            future_to_household = {
                executor.submit(_run_household_worker, player_id): player_id
                for player_id in selected_households
            }

            for future in as_completed(future_to_household):
                player_id = future_to_household[future]
                payload = future.result()
                if write_results_to_sqlite:
                    load_history_payload_to_sqlite(self.results_conn, payload)
                    load_results_payload_to_sqlite(self.results_conn, payload)
                results["net_costs"].append(payload["net_cost"])
                results["net_loads"].append(payload["net_load"])
                results["total_costs"].append(payload["total_cost"])
                results["total_consumptions"].append(payload["total_consumption"])
                print(
                    f"completed household {player_id} in scenario {payload['scenario']} "
                    f"with controller {payload['policy']}, run_id={payload['run_id']}",
                    flush=True,
                )

        return results


    def run_batch(
        self,
        run_contexts: list[RunContext],
        household_ids: list[int] | None = None,
        max_households: int | None = None,
        parallel_households: bool = True,
        parallel_workers: int | None = 6,
        write_results_to_sqlite: bool = True,
    ):
        results = {
            "net_costs": [],
            "net_loads": [],
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
                results["net_costs"].extend(batch_results["net_costs"])
                results["net_loads"].extend(batch_results["net_loads"])
                results["total_costs"].extend(batch_results["total_costs"])
                results["total_consumptions"].extend(batch_results["total_consumptions"])

        return results


    def make_mpc_controller(self, name: str, horizon: int, predictor: BasePredictor, duration_hours: float = 0.25, buffer_config: DeviceBufferConfig | None = None) -> Callable[[Household, Scenario], MPCController]:
        # convenience / backwards compatibility
        return make_mpc_controller(name=name, horizon=horizon, predictor=predictor, duration_hours=duration_hours, buffer_config=buffer_config)




if __name__ == "__main__":
    # parse args
    args = build_parser().parse_args()

    # Create a simulation instance
    sim = Simulation(source_sqlite_conn, results_conn=create_results_connection())

    requested_controller_names = (
        None
        if args.controllers == "all"
        else [
            name.strip() for name in args.controllers.split(",") if name.strip()
        ]
    )
    controller_factories_by_name = build_controller_factories(requested_controller_names)
    selected_controller_names = (
        list(controller_factories_by_name)
        if requested_controller_names is None
        else requested_controller_names
    )

    # collect scenarios
    if args.scenarios == "all":
        selected_scenarios = list(scenario_catalog.values())
    else:
        requested_scenario_names = [
            name.strip() for name in args.scenarios.split(",") if name.strip()
        ]
        unknown_scenario_names = [
            name for name in requested_scenario_names if name not in scenario_catalog
        ]
        if unknown_scenario_names:
            raise ValueError(f"Unknown scenarios: {unknown_scenario_names}")
        selected_scenarios = [scenario_catalog[name] for name in requested_scenario_names]

    # collect households
    if args.households == "all":
        selected_household_ids = None
    else:
        selected_household_ids = args.households

    # create run contexts for each combination of controller and scenario
    run_contexts = [
        RunContext(
            controller_name=controller_name,
            controller_factory=controller_factories_by_name[controller_name],
            scenario=scenario,
            start_time=args.start_time,
        )
        for controller_name in selected_controller_names
        for scenario in selected_scenarios
    ]

    # run everything
    sim.run_batch(
        run_contexts,
        household_ids=selected_household_ids,
        max_households=args.max_households,
        parallel_households=args.parallel_households,
        parallel_workers=args.parallel_workers,
    )
