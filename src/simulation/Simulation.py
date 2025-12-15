from src.simulation.Household import Household


class Simulation:
    def __init__(self, sqlite_conn, influx_query_api, strategies=['no_optimization', 'self_consumption', 'cost_optimization']):
        self.sqlite_conn = sqlite_conn
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.influx_query_api = influx_query_api
        self.strategies = strategies

        # query the length of the player_pv_bess table
        self.sqlite_cursor.execute("SELECT COUNT(*) FROM player_pv_bess")
        self.num_households = self.sqlite_cursor.fetchone()[0]


    def create_household(self, player_id):
        household = Household(player_id=player_id)
        # we could fetch and fill the household here.
        # x, y, z = fetch from db
        # household.player = Player(load_profile, ev1_profile, ev2_profile)
        # household.pv = PV(pv_profile)
        # household.bess = BESS(capacity, max_charge, max_discharge, efficiency, initial_soc)
        # household.ev1 = EV(ev1_profile, capacity, max_charge, max_discharge, efficiency, initial_soc)
        # household.ev2 = EV(ev2_profile, capacity, max_charge, max_discharge

        return household


    def run_household(self, player_id):
        # Implementation of the simulation logic goes here
        household = self.create_household(player_id)

        # For each strategy, simulate the household behavior
        for strategy in self.strategies:
            print(f"Simulating household {player_id} with strategy {strategy}")
            # Here you would implement the actual simulation logic
            # For now, we just print the action
            load_profile = household.generate_new_load_profile(strategy)
            print(f"Generated load profile for household {player_id} with strategy {strategy}: {load_profile}")

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

    simulation.run_household()