from influxdb_client.client.write_api import SYNCHRONOUS
from src.config import Config
from src.connections import fetch_multiple_timeseries
from src.ingestion.data_loader import period_to_epoch
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from src.simulation.components.PV import PV
from src.simulation.household import Household
from src.simulation.policies.blind import no_control
from src.simulation.requirements.charge_requirements import half_full_by_midnight


class Simulation:
    def __init__(self, sqlite_conn, influx_client, charge_requirements=half_full_by_midnight):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.influx_client = influx_client
        self.influx_query_api = influx_client.query_api()
        self.influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)

        self.num_households = 250
        self.num_timesteps = 96

        self.charge_requirements = charge_requirements

        self.env_inputs = [ # influx table names
            "load",
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

        self.histories = {
            "passive": {},
        } # store histories for all households if needed

        self.current_timestep = 0  # current timestep in the simulation.

        self.sqlite_cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                policy TEXT,
                player_id INTEGER,
                has_pv BOOLEAN,
                has_bess BOOLEAN,
                total_cost REAL,
                total_consumption REAL
            )''')


    def run_all_households(self, policy=no_control, start_time=0):
        for player_id in range(1, self.num_households + 1):
            self.run_household(player_id, policy=policy, start_time=start_time)


    def run_household(self, player_id, policy=no_control, start_time=0):
        print(f"running household {player_id} with policy {policy.__name__}")
        household = self.create_household(player_id, start_time)

        for t in range(start_time, self.num_timesteps):
            self.step(household, policy=policy, duration_hours=0.25, time=t)

        self.load_history_to_influx(household, policy_name=policy.__name__)
        self.load_results_to_sqlite(household, policy_name=policy.__name__)

        return household


    def create_household(self, player_id, start_time=0):
        household = Household(player_id=player_id, start_time=start_time)
        
        if self.charge_requirements:
            household.charge_requirements = self.charge_requirements

        household.fixed_cost = self.sqlite_cursor.execute(
            "SELECT fixed_costs FROM fixed_costs WHERE player_id = ?",
            (player_id,)
        ).fetchone()[0]

        has_pv, has_bess = self.sqlite_cursor.execute(
            "SELECT has_pv, has_bess FROM player_pv_bess WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        # only query once for all profiles. the simulation now knows the future
        self.household_profiles = fetch_multiple_timeseries(
            self.influx_query_api,
            player_id,
            measurements=self.env_inputs
        )

        # household gets access to prices over day
        household.buy_price_day_profile = self.household_profiles["buy_price"]
        household.sell_price_day_profile = self.household_profiles["sell_price"]

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
            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, soc=initial_soc)

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
        household.ev1 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

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
        household.ev2 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

        return household


    def step(self, household: Household, policy=no_control, duration_hours=0.25, time=0):
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
        household.base_load = profiles["load"][timestep]

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


    def load_results_to_sqlite(self, household: Household, policy_name="basic_bess"):
        total_cost = household.total_cost
        total_consumption = household.total_consumption

        self.sqlite_cursor.execute(
            '''
            INSERT INTO results (
            policy, player_id, has_pv, has_bess, total_cost, total_consumption)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (policy_name, household.player_id, household.has_pv,
             household.has_bess, total_cost, total_consumption)
        )
        self.sqlite_conn.commit()


    def load_history_to_influx(
            self,
            household: Household,
            policy_name="basic_bess",
            measurements=None):
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


        for m in measurements:
            points = []
            for t, value in household.history[m].items():
                points.append({
                    "measurement": m,
                    "tags": {"player_id": str(household.player_id), "policy": policy_name},
                    "fields": {"value": value},
                    "time": period_to_epoch(t),
                })

            try:
                self.influx_write_api.write(
                    bucket=Config.INFLUX_BUCKET,
                    org=Config.INFLUX_ORG,
                    record=points
                )
            except Exception as e:
                raise RuntimeError(f"Failed to write {len(points)} points to InfluxDB: {e}") from e
