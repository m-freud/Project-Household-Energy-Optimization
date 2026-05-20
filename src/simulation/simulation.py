# paste this to enable src. imports

import argparse
from functools import partial
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.run_context import RunContext
from src.simulation.controllers.function_controller import FunctionController
from src.simulation.household import Household
from src.sqlite_connection import sqlite_conn, fetch_multiple_timeseries
from src.simulation.devices.pv import PV
from src.simulation.devices.bess import BESS
from src.simulation.devices.ev import EV

from src.simulation.scenarios.scenario import Scenario, scenarios as scenario_catalog
from src.simulation.controllers.policies.basic_examples import no_control
from src.simulation.controllers.policies.linear.linear import even_linear_policy, fast_charge_policy
from src.simulation.controllers.policies.linear.price_aware_linear import price_aware_linear
from src.simulation.controllers.policies.rule_based.priority_dispatch import priority_dispatch_policy
from src.simulation.controllers.policies.rule_based.waterfall_v1 import waterfall_v1_policy
from src.simulation.controllers.base_controller import BaseController
from src.simulation.controllers.policies.rule_based.price_aware_bess import price_aware_bess


class Simulation:
    def __init__(self, sqlite_conn):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()

        self.num_households = 250
        self.num_timesteps = 96

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


    def create_household(self, player_id:int, run_context: RunContext):
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
        if has_bess:
            bess_data = self.sqlite_cursor.execute(
                '''SELECT capacity, charge, discharge, efficiency, initial_soc
                FROM bess
                WHERE player_id = ?''',
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data
            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, soc=initial_soc, name="bess")

        # plug in EV1
        ev1_data = self.sqlite_cursor.execute(
            '''
            SELECT capacity, charge, discharge, efficiency, initial_soc
            FROM ev1
            WHERE player_id = ?
            ''',
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev1_data


        household.ev1 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc, name="ev1")

        # plug in EV2
        ev2_data = self.sqlite_cursor.execute(
            '''
            SELECT capacity, charge, discharge, efficiency, initial_soc
            FROM ev2
            WHERE player_id = ?
            ''',
            (player_id,)
        ).fetchone()

        capacity, max_charge, max_discharge, efficiency, initial_soc = ev2_data
        household.ev2 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc, name="ev2")

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
        household.ev1_buy_price_day_profile = self.household_profiles["ev1_buy_price"]
        household.ev2_buy_price_day_profile = self.household_profiles["ev2_buy_price"]

        return household
    

    def step(
            self,
            household: Household,
            controller: BaseController,
            scenario: Scenario,
            duration_hours=0.25,
            time=0):
        self.current_timestep = time
        self.update_household_inputs(household)
        controls = controller.set_controls(household, scenario)
        household.apply_controls(controls)
        household.update_history()


    def update_household_inputs(self, household: Household):
        # update time
        timestep = self.current_timestep
        household.current_timestep = self.current_timestep
        profile_time_index = timestep - 1

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
            ev1.at_home = profiles["ev1_at_home"][profile_time_index]
            ev1.at_charging_station = profiles["ev1_at_charging_station"][profile_time_index]
            ev1.buy_price = profiles["ev1_buy_price"][profile_time_index]
            ev1.max_charge = profiles["ev1_max_charge"][profile_time_index]

        # update ev2
        if ev2:
            ev2.load = profiles["ev2_load"][profile_time_index]
            ev2.at_home = profiles["ev2_at_home"][profile_time_index]
            ev2.at_charging_station = profiles["ev2_at_charging_station"][profile_time_index]
            ev2.buy_price = profiles["ev2_buy_price"][profile_time_index]
            ev2.max_charge = profiles["ev2_max_charge"][profile_time_index]

        # update buy / sell prices
        household.buy_price = profiles["buy_price"][profile_time_index]
        household.sell_price = profiles["sell_price"][profile_time_index]


    def run_household(self, player_id, run_context: RunContext):
        controller = run_context.controller
        scenario = run_context.scenario
        current_run_id = run_context.run_id

        print(f"running household {player_id} in scenario {scenario.name} with controller {controller.name}, run_id = {current_run_id}")

        start_time = run_context.start_time
        if start_time < 1 or start_time > self.num_timesteps:
            raise ValueError(f"start_time must be between 1 and {self.num_timesteps}")

        household = self.create_household(player_id, run_context)

        for t in range(start_time, self.num_timesteps + 1):
            self.step(household, controller, scenario, duration_hours=0.25, time=t)

        self.load_history_to_sqlite(household, policy_name=controller.name, scenario_name=scenario.name)
        
        self.load_results_to_sqlite(
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

        for player_id in selected_households:
            self.run_household(player_id, run_context)


    def run_batch(
        self,
        run_contexts: list[RunContext],
        household_ids: list[int] | None = None,
        max_households: int | None = None,
    ):
        for run_context in run_contexts:
            self.run_all_households(
                run_context,
                household_ids=household_ids,
                max_households=max_households,
            )


    def load_results_to_sqlite(self, household: Household, policy_name:str="no_control", scenario_name:str="default_scenario", run_id:str|None=None):
        total_cost = household.total_cost
        total_consumption = household.total_consumption
        net_cost = sum(household.history["net_cost"].values()) * 0.25
        net_load = sum(household.history["net_load"].values()) * 0.25
        target_met_bess = household.has_met_target("bess")
        target_met_ev1 = household.has_met_target("ev1")
        target_met_ev2 = household.has_met_target("ev2")
        target_met_all_bess = household.has_met_all_targets("bess")
        target_met_all_ev1 = household.has_met_all_targets("ev1")
        target_met_all_ev2 = household.has_met_all_targets("ev2")
        soc_at_deadline_bess = household.soc_at_deadline("bess")
        soc_at_deadline_ev1 = household.soc_at_deadline("ev1")
        soc_at_deadline_ev2 = household.soc_at_deadline("ev2")
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
            (run_id, policy_name, scenario_name, household.player_id, household.has_pv,
             household.has_bess, total_cost, total_consumption,
             net_cost, net_load,
             target_met_bess, target_met_ev1, target_met_ev2,
               target_met_all_bess, target_met_all_ev1, target_met_all_ev2,
             soc_at_deadline_bess, soc_at_deadline_ev1, soc_at_deadline_ev2)
        )
        self.sqlite_conn.commit()


    def load_history_to_sqlite(self,
                               household, policy_name,
                               scenario_name, measurements=None):
        if measurements is None:
            measurements = [
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

        for measurement in measurements:
            # create table if not exists
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


            points = []

            for t, value in household.history[measurement].items():
                points.append({
                    "measurement": measurement,
                    "tags": {
                        "player_id": household.player_id,
                        "policy": policy_name,
                        "scenario": scenario_name,
                    },
                    "value": value,
                    "period": t,
                })


            for point in points:
                self.sqlite_cursor.execute(
                    f'''
                    INSERT INTO {measurement} (
                    player_id, scenario, policy, period, value
                    ) VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        point["tags"]["player_id"],
                        point["tags"]["scenario"],
                        point["tags"]["policy"],
                        point["period"],
                        point["value"]
                    )
                )

            self.sqlite_conn.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run household energy simulation")
    parser.add_argument(
        "--controllers",
        default="all",
        help="Comma-separated controller names (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenario names (default: all)",
    )
    parser.add_argument(
        "--households",
        default="all",
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
    args = parser.parse_args()

    # Create a simulation instance
    sim = Simulation(sqlite_conn)

    no_control_controller = FunctionController(name="no_control", step_function=no_control)
    even_linear_controller = FunctionController(name="even_linear", step_function=even_linear_policy)
    fast_charge_controller = FunctionController(name="fast_charge", step_function=fast_charge_policy)
    price_aware_linear_1_controller = FunctionController(
        name="price_aware_linear_1",
        step_function=price_aware_linear,
    )
    price_aware_linear_2_controller = FunctionController(
        name="price_aware_linear_2",
        step_function=partial(price_aware_linear, default_behaviour="even_linear"),
    )
    priority_dispatch_controller = FunctionController(name="priority_dispatch", step_function=priority_dispatch_policy)
    waterfall_v1_controller = FunctionController(name="waterfall_v1", step_function=waterfall_v1_policy)

    controllers = [
        no_control_controller,
        fast_charge_controller,
        even_linear_controller,
        price_aware_linear_1_controller,
        price_aware_linear_2_controller,
        priority_dispatch_controller,
        waterfall_v1_controller,
    ]

    controllers_by_name = {controller.name: controller for controller in controllers}
    scenarios_by_name = {scenario.name: scenario for scenario in scenario_catalog}

    if args.controllers == "all":
        selected_controllers = controllers
    else:
        requested_controller_names = [
            name.strip() for name in args.controllers.split(",") if name.strip()
        ]
        unknown_controller_names = [
            name for name in requested_controller_names if name not in controllers_by_name
        ]
        if unknown_controller_names:
            raise ValueError(f"Unknown controllers: {unknown_controller_names}")
        selected_controllers = [controllers_by_name[name] for name in requested_controller_names]

    if args.scenarios == "all":
        selected_scenarios = scenario_catalog
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
    else:
        selected_household_ids = [
            int(token.strip()) for token in args.households.split(",") if token.strip()
        ]

    run_contexts = [
        RunContext(controller=controller, scenario=scenario, start_time=args.start_time)
        for controller in selected_controllers
        for scenario in selected_scenarios
    ]

    sim.run_batch(
        run_contexts,
        household_ids=selected_household_ids,
        max_households=args.max_households,
    )
