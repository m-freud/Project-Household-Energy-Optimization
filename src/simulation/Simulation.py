from src.config import Config
from src.simulation.Household import Household
from src.simulation.components.PV import PV
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from src.simulation.policies.basic import no_control, random_control
from src.simulation.policies.rule_based import basic_bess, basic_ev, basic_ev_bess
from src.ingestion.data_loader import period_to_epoch

from influxdb_client.client.write_api import SYNCHRONOUS


class Simulation:
    def __init__(self, sqlite_conn, influx_client):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.influx_query_api = influx_client.query_api()
        self.influx_write_api = influx_client.write_api(write_options=SYNCHRONOUS)
        
        self.num_households = 250
        self.num_timesteps = 96

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

        self.time = 0  # current timestep in the simulation.


    def fetch_single_timeseries(self, player_id, measurement):
        # implement influx query to fetch timeseries data for given player_id and measurement
        query = f'''
        from(bucket: "{Config.INFLUX_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["player_id"] == "{player_id}")
          |> filter(fn: (r) => r["_measurement"] == "{measurement}")
          |> sort(columns: ["_time"])
        '''
        result = self.influx_query_api.query(org=Config.INFLUX_ORG, query=query)

        timeseries = []

        for record in result[0]:
            timeseries.append(record.get_value())

        return timeseries


    def fetch_multiple_timeseries(self, player_id, measurements:list):
        timeseries_data = {m: [] for m in measurements}

        flux_set = "[" + ",".join(f'"{m}"' for m in measurements) + "]"

        
        query = f'''
        from(bucket: "{Config.INFLUX_BUCKET}")
            |> range(start: 0)
            |> filter(fn: (r) => r["player_id"] == "{player_id}")

            |> filter(fn: (r) => contains(value: r["_measurement"], set: {flux_set}))
            |> sort(columns: ["_time"])
        '''
        result = self.influx_query_api.query(org=Config.INFLUX_ORG, query=query)


        for table in result:
            measurement = table.records[0].values['_measurement']
            series = [record.get_value() for record in table]
            timeseries_data[measurement] = series
        
        return timeseries_data


    def create_household(self, player_id, start_time=0):
        household = Household(player_id=player_id, start_time=start_time)

        household.fixed_cost = self.sqlite_cursor.execute(
            "SELECT fixed_costs FROM fixed_costs WHERE player_id = ?",
            (player_id,)
        ).fetchone()[0]

        has_pv, has_bess = self.sqlite_cursor.execute(
            "SELECT has_pv, has_bess FROM player_pv_bess WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        # only query once for all profiles. the simulation now knows the future
        self.household_profiles = self.fetch_multiple_timeseries(
            player_id,
            measurements=self.env_inputs
        )

        # plug in PV
        if has_pv:
            household.pv = PV()

        # plug in BESS
        if has_bess:
            bess_data = self.sqlite_cursor.execute(
                "SELECT capacity, charge, discharge, efficiency, initial_soc FROM bess WHERE player_id = ?",
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data
            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, soc=initial_soc)

        # plug in EV1
        ev1_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev1 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev1_data
        household.ev1 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

        # plug in EV2
        ev2_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev2 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev2_data
        household.ev2 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

        return household


    def update_household(self, household: Household):
        # update time
        household.time = self.time

        # update base load
        household.base_load = self.household_profiles["load"][household.time]
        
        # update pv
        if household.pv:
            household.pv.generation = self.household_profiles["pv_gen"][household.time]

        # update ev1
        if household.ev1:
            household.ev1.load = self.household_profiles["ev1_load"][household.time]
            household.ev1.at_home = self.household_profiles["ev1_at_home"][household.time]
            household.ev1.at_charging_station = self.household_profiles["ev1_at_charging_station"][household.time]
            household.ev1.buy_price = self.household_profiles["ev1_buy_price"][household.time]
            household.ev1.max_charge = self.household_profiles["ev1_max_charge"][household.time]

        # update ev2
        if household.ev2:
            household.ev2.load = self.household_profiles["ev2_load"][household.time]
            household.ev2.at_home = self.household_profiles["ev2_at_home"][household.time]
            household.ev2.at_charging_station = self.household_profiles["ev2_at_charging_station"][household.time]
            household.ev2.buy_price = self.household_profiles["ev2_buy_price"][household.time]
            household.ev2.max_charge = self.household_profiles["ev2_max_charge"][household.time]

        # update buy / sell prices
        household.buy_price = self.household_profiles["buy_price"][household.time]
        household.sell_price = self.household_profiles["sell_price"][household.time]


    def step(self, household: Household, policy=no_control, duration_hours=0.25, time=0):
        self.time = time
        self.update_household(household)

        controls = policy(household, household.time)
        household.apply_controls(controls, duration_hours=duration_hours)

        household.update_history()


    def run_household(self, player_id, policy=no_control, start_time=0):
        household = self.create_household(player_id, start_time)

        for t in range(start_time, self.num_timesteps):
            self.step(household, policy=policy, duration_hours=0.25, time=t)

        return household


if __name__ == "__main__":
    import src.connections as connections
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client
    )



    for i in range(1, 2):
        household = simulation.run_household(i, policy=basic_ev_bess)
        household.plot_history_all(plots=['ev1_soc', 'ev1_power', 'ev1_at_home', 'net_load','bess_soc', 'bess_power'])
