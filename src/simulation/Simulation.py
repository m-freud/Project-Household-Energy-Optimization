from src.config import Config
from src.simulation.participants.Household import Household
from src.simulation.participants.fixed.Player import Player
from src.simulation.participants.fixed.PV import PV
from src.simulation.participants.controllable.BESS import BESS
from src.simulation.participants.controllable.EV import EV


class Simulation:
    def __init__(self, sqlite_conn, influx_query_api, strategies=['no_optimization', 'self_consumption', 'cost_optimization']):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.influx_query_api = influx_query_api
        self.strategies = strategies

        # query the length of the player_pv_bess table
        self.sqlite_cursor.execute("SELECT COUNT(*) FROM player_pv_bess")
        self.num_households = self.sqlite_cursor.fetchone()[0]


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


    def create_household(self, player_id):
        household = Household()

        has_pv, has_bess = self.sqlite_cursor.execute(
            "SELECT has_pv, has_bess FROM player_pv_bess WHERE player_id = ?",
            (player_id,)
        ).fetchone()

        profiles_to_fetch = [
            "load",
            "pv_gen"
            "ev1_load",
            "ev2_load",
            "buy_price",
            "sell_price",
            "ev1_buy_price",
            "ev2_buy_price",
            "ev1_at_home",
            "ev2_at_home",
            "ev1_at_charging_station",
            "ev2_at_charging_station"
        ]

        # only query once for all profiles. the simulation now knows the future
        profiles = self.fetch_multiple_timeseries(
            player_id,
            profiles_to_fetch
        )

        # Player
        household.player = Player(profiles["load"], profiles["ev1_load"], profiles["ev2_load"])

        # PV
        if has_pv:
            household.pv = PV(profiles["pv_gen"])

        # BESS
        if has_bess:
            bess_data = self.sqlite_cursor.execute(
                "SELECT capacity, charge, discharge, efficiency, initial_soc FROM bess WHERE player_id = ?",
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data

            household.bess = BESS(capacity, max_charge, max_discharge, efficiency, initial_soc)

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


    def update_household(self, household: Household, time_step: int, policy=None):
        # get current profiles
        load = household.player.get_load()[time_step]
        pv_gen = household.pv.get_generation()[time_step] if household.pv else 0

        net_load = load - pv_gen

        if policy:
            bess_power, ev1_power, ev2_power = policy(time_step, household.profiles, household.bess, household.ev1, household.ev2) # what do we pass here?
        else:
            bess_power, ev1_power, ev2_power = 0, 0, 0

        return net_load, bess_power, ev1_power, ev2_power, net_cost

    def run_household(self, player_id, optimizer=None):
        # create is part of run
        household = self.create_household(player_id)

        for t in range(96):  # assuming 96 time steps (15-minute intervals in a day)
            pass

        return household







    def run_all_households(self, strategy=None):
        for player_id in range(1, self.num_households + 1):
            self.run_household(player_id, strategy)



if __name__ == "__main__":
    import src.connections as connections
    from src.simulation.Simulation import Simulation
    sqlite_conn = connections.create_sqlite_connection()
    influx_query_api = connections.get_influx_query_api()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_query_api=influx_query_api
    )

    simulation.run_household(1)
    exit()