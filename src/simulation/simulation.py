# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household
from src.sqlite_connection import sqlite_conn, fetch_multiple_timeseries
from src.simulation.devices.pv import PV
from src.simulation.devices.bess import BESS
from src.simulation.devices.ev import EV

from src.simulation.scenarios.scenario import Scenario
from src.simulation.policies.basic_examples import no_control
from src.simulation.policies.linear import even_linear_policy, fast_charge_policy
from src.simulation.scenarios.scenario import default_scenario


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
                policy TEXT,
                scenario TEXT,
                player_id INTEGER,
                has_pv BOOLEAN,
                has_bess BOOLEAN,
                total_cost REAL,
                total_consumption REAL,
                target_met_bess BOOLEAN,
                target_met_ev1 BOOLEAN,
                target_met_ev2 BOOLEAN,
                soc_at_deadline_bess REAL,
                soc_at_deadline_ev1 REAL,
                soc_at_deadline_ev2 REAL
            )''')

        self._ensure_results_columns()


    def _ensure_results_columns(self):
        columns = {
            row[1]
            for row in self.sqlite_cursor.execute("PRAGMA table_info(results)").fetchall()
        }

        required_columns = [
            ("target_met_bess", "BOOLEAN"),
            ("target_met_ev1", "BOOLEAN"),
            ("target_met_ev2", "BOOLEAN"),
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

    
    def create_household(self, player_id:int, scenario:Scenario, start_time=0):
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

        return household
    

    def step(self, household: Household, policy=no_control, scenario: Scenario=default_scenario, duration_hours=0.25, time=0):
        self.current_timestep = time
        self.update_household_inputs(household)

        controls = policy(household)
        household.apply_controls(controls, duration_hours=duration_hours)

        household.update_history()


    def update_household_inputs(self, household: Household):
        # update time
        timestep = self.current_timestep
        household.current_timestep = self.current_timestep

        profiles = self.household_profiles
        ev1 = household.ev1
        ev2 = household.ev2
        pv = household.pv

        # update base load
        household.base_load = profiles["base_load"][timestep]

        # update pv
        if pv:
            pv.generation = profiles["pv_gen"][timestep]

        # update ev1
        if ev1:
            ev1.load = profiles["ev1_load"][timestep]
            ev1.at_home = profiles["ev1_at_home"][timestep]
            ev1.at_charging_station = profiles["ev1_at_charging_station"][timestep]
            ev1.buy_price = profiles["ev1_buy_price"][timestep]
            ev1.max_charge = profiles["ev1_max_charge"][timestep]

        # update ev2
        if ev2:
            ev2.load = profiles["ev2_load"][timestep]
            ev2.at_home = profiles["ev2_at_home"][timestep]
            ev2.at_charging_station = profiles["ev2_at_charging_station"][timestep]
            ev2.buy_price = profiles["ev2_buy_price"][timestep]
            ev2.max_charge = profiles["ev2_max_charge"][timestep]

        # update buy / sell prices
        household.buy_price = profiles["buy_price"][timestep]
        household.sell_price = profiles["sell_price"][timestep]


    def run_household(self, player_id, policy=no_control, scenario: Scenario=default_scenario, start_time=1):
        print(f"running household {player_id} with policy {policy.__name__}")

        if start_time < 1 or start_time > self.num_timesteps:
            raise ValueError(f"start_time must be between 1 and {self.num_timesteps}")

        household = self.create_household(player_id, scenario, start_time)

        for t in range(start_time, self.num_timesteps):
            self.step(household, policy=policy, scenario=scenario, duration_hours=0.25, time=t)

        self.load_history_to_sqlite(household, policy_name=policy.__name__, scenario_name=scenario.name)
        self.load_results_to_sqlite(household, policy_name=policy.__name__, scenario_name=scenario.name)

        return household


    def run_all_households(self, policy=no_control, scenario: Scenario=default_scenario, start_time=1):
        for player_id in range(1, self.num_households + 1):
            self.run_household(player_id, policy=policy, scenario=scenario, start_time=start_time)


    def load_results_to_sqlite(self, household: Household, policy_name:str="no_control", scenario_name:str="default_scenario"):
        total_cost = household.total_cost
        total_consumption = household.total_consumption
        target_met_bess = household.has_met_target("bess")
        target_met_ev1 = household.has_met_target("ev1")
        target_met_ev2 = household.has_met_target("ev2")
        soc_at_deadline_bess = household.history["bess_soc"].get(getattr(household.scenario.bess, "deadline", None), None) if household.scenario else None
        soc_at_deadline_ev1 = household.history["ev1_soc"].get(getattr(household.scenario.ev1, "deadline", None), None) if household.scenario else None
        soc_at_deadline_ev2 = household.history["ev2_soc"].get(getattr(household.scenario.ev2, "deadline", None), None) if household.scenario else None

        self.sqlite_cursor.execute(
            '''
            INSERT INTO results (
            policy, scenario, player_id, has_pv, has_bess, total_cost, total_consumption,
            target_met_bess, target_met_ev1, target_met_ev2,
            soc_at_deadline_bess, soc_at_deadline_ev1, soc_at_deadline_ev2
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (policy_name, scenario_name, household.player_id, household.has_pv,
             household.has_bess, total_cost, total_consumption,
             target_met_bess, target_met_ev1, target_met_ev2,
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
    # Create a simulation instance
    sim = Simulation(sqlite_conn)

    # Load the scenario
    scenario = default_scenario

    policies = [
        # no_control,
        # make_naive_linear_policy(urgency=1.0, delay=0.0),
        # make_naive_linear_policy(urgency=0.0, delay=0.0),
        # make_naive_linear_policy(urgency=0.0, delay=1.0),
        # make_linear_policy(urgency=0.5, delay=0.5),
        even_linear_policy,
        fast_charge_policy,
    ]

    for policy in policies:
        sim.run_all_households(policy=policy, scenario=scenario, start_time=1)