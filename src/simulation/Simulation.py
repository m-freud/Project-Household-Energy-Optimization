from src.config import Config
from src.simulation.Household import Household
from src.simulation.components.PV import PV
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from src.simulation.controller.policies.basic import no_control, random_control


class Simulation:
    def __init__(self, sqlite_conn, influx_query_api): # it makes sense to pass multiple policies here. we get more runs with the same query
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.influx_query_api = influx_query_api
        
        self.num_households = 250
        self.num_timesteps = 96

        self.env_inputs = [
            "load",
            "pv_gen",
            "ev1_load",
            "ev2_load",
            "buy_price",
            "sell_price",
            "ev1_at_home",
            "ev1_at_charging_station",
            "ev1_buy_price",
            "ev2_at_home",
            "ev2_at_charging_station",
            "ev2_buy_price",
        ]

        self.household_profiles = {} # store profiles for current household being simulated
        # the simulation knows the whole day.
        # the household knows the day up to current timestep

        self.timestep_current_run = 0  # current timestep in the simulation.


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
        household = Household()

        has_pv, has_bess = self.sqlite_cursor.execute(
            "SELECT has_pv, has_bess FROM player_pv_bess WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        # only query once for all profiles. the simulation now knows the future
        self.household_profiles = self.fetch_multiple_timeseries(
            player_id,
            self.exogenous_inputs
        )

        # Player
        household.player = Player()

        # PV
        if has_pv:
            household.pv = PV()

        # BESS
        if has_bess:
            bess_data = self.sqlite_cursor.execute(
                "SELECT capacity, charge, discharge, efficiency, initial_soc FROM bess WHERE player_id = ?",
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data
            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, soc=initial_soc)

        # EV1
        ev1_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev1 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev1_data
        household.ev1 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

        # EV2
        ev2_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev2 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev2_data
        household.ev2 = EV(capacity, max_charge, max_discharge, efficiency, initial_soc)

        return household


    def update_household(self, household: Household):
        # update all participants based on current timestep.
        # step 1: get exogenous inputs for this timestep (give current situation to household)
        # step 2: update controllable participants based on policy (household sets controls based on situation and policy)

        time = self.timestep_current_run
        household.time = time

        # exogenous inputs
        # update player
        if household.player:
            household.player.load = self.household_profiles["load"][time]
        
        # update pv
        if household.pv:
            household.pv.generation = self.household_profiles["pv_gen"][time]

        # update ev1
        if household.ev1:
            household.ev1.load = self.household_profiles["ev1_load"][time]
            household.ev1.at_home = self.household_profiles["ev1_at_home"][time]
            household.ev1.at_charging_station = self.household_profiles["ev1_at_charging_station"][time]

        # update ev2
        if household.ev2:
            household.ev2.load = self.household_profiles["ev2_load"][time]
            household.ev2.at_home = self.household_profiles["ev2_at_home"][time]
            household.ev2.at_charging_station = self.household_profiles["ev2_at_charging_station"][time]

        # update buy / sell prices
        household.buy_price = self.household_profiles["buy_price"][time]
        household.sell_price = self.household_profiles["sell_price"][time]

        self.timestep_current_run += 1


    def run_household(self, player_id, policy):
        household = self.create_household(player_id)

        for t in range(96):
            self.timestep_current_run = t

            # 1️⃣ exogen
            self.update_household(household)

            # 2️⃣ decide
            controls = policy(household, t)

            # 3️⃣ apply physics
            household.set_controls(controls)
            household.apply_controls(duration_hours=0.25)

            # 4️⃣ log
            household.update_history()

        return household



    def run_all_households(self, policy=None):
        for player_id in range(1, self.num_households + 1):
            self.run_household(player_id, policy)


if __name__ == "__main__":
    import src.connections as connections
    sqlite_conn = connections.create_sqlite_connection()
    influx_query_api = connections.get_influx_query_api()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_query_api=influx_query_api
    )

    household = simulation.run_household(1, policy=random_policy)
    household.plot_history_all(plots=[])
    # household.plot_history()
    # household.plot_history_all(plots=['ev1_load', 'ev1_power', 'ev1_soc', 'bess_power', 'bess_soc'])
    # household.plot_history_all(plots=['load', 'bess_soc', 'ev1_soc', 'ev2_soc'])