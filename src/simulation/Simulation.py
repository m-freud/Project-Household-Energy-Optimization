from src.config import Config
from src.simulation.Household import Household
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

        profiles_to_fetch = ["load", "ev1_load", "ev2_load", "buy_price", "sell_price"]
        if has_pv:
            profiles_to_fetch.append("pv_gen")

        # only query once for all profiles
        profiles = self.fetch_multiple_timeseries(
            player_id,
            profiles_to_fetch
        )

        # Player
        household.add_player(Player(profiles["load"], profiles["ev1_load"], profiles["ev2_load"]))

        # PV
        if has_pv:
            household.add_pv(PV(profiles["pv_gen"]))

        # BESS
        if has_bess:
            bess_data = self.sqlite_cursor.execute(
                "SELECT capacity, charge, discharge, efficiency, initial_soc FROM bess WHERE player_id = ?",
                (player_id,)
            ).fetchone()

            capacity, max_charge, max_discharge, efficiency, initial_soc = bess_data

            household.add_bess(BESS(capacity, max_charge, max_discharge, efficiency, initial_soc))

        # EV1
        ev1_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev1 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev1_data
        household.add_ev1(EV(capacity, max_charge, max_discharge, efficiency, initial_soc))

        # EV2
        ev2_data = self.sqlite_cursor.execute(
            "SELECT capacity, charge, discharge, efficiency, initial_soc FROM ev2 WHERE player_id = ?",
            (player_id,)
        ).fetchone()
        capacity, max_charge, max_discharge, efficiency, initial_soc = ev2_data
        household.add_ev2(EV(capacity, max_charge, max_discharge, efficiency, initial_soc))

        return household


    def run_household(self, player_id):
        # Implementation of the simulation logic goes here
        household = self.create_household(player_id)

        # For each strategy, simulate the household behavior
        for strategy in self.strategies:
            break

            #print(f"Simulating household {player_id} with strategy {strategy}")
            # Here you would implement the actual simulation logic
            # For now, we just print the action
            #load_profile = household.generate_new_load_profile(strategy)
            #print(f"Generated load profile for household {player_id} with strategy {strategy}: {load_profile}")

            # and then load it to influxdb, with a nice tag structure.



    def run_all_households(self):
        for player_id in range(1, self.num_households + 1):
            self.run_household(player_id)





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