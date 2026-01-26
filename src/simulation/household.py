import matplotlib.pyplot as plt
from src.simulation.components.BESS import BESS
from src.simulation.components.EV import EV
from src.simulation.components.PV import PV
from src.simulation.requirements.charge_requirements import half_full_by_midnight


class Household:
    '''
    Represents a household with various energy components and their states.
    '''
    def __init__(
            self,
            player_id=1,
            start_time=0,
            pv:PV|None=None,
            bess:BESS|None=None,
            ev1:EV|None=None,
            ev2:EV|None=None,
            fixed_cost=0.0,
            charge_requirements=half_full_by_midnight):
        # timing info
        self.current_timestep = start_time  # start time of the simulation for this household
        self.player_id = player_id

        # you can create empty or full
        self.pv = pv
        self.bess = bess
        self.ev1 = ev1
        self.ev2 = ev2
        self.charge_requirements = charge_requirements

        self.base_load = 0.0  # current base load
        self.buy_price = 0.0  # current buy price for electricity
        self.sell_price = 0.0  # current sell price for electricity
        self.fixed_cost = fixed_cost  # fixed cost per day

        self.buy_price_day_profile = []  # store buy price profile for the day
        self.sell_price_day_profile = []  # store sell price profile for the day

        self.history = {
            "base_load": {},            
            "pv_gen": {},
            "bess_soc": {},
            "buy_price": {},
            "sell_price": {},
            "net_load": {},
            "net_cost": {},
            "ev1_soc": {},
            "ev2_soc": {},
            "ev1_at_home": {},
            "ev2_at_home": {},
            "ev1_at_charging_station": {},
            "ev2_at_charging_station": {},
            "ev1_load": {},
            "ev2_load": {},
            "ev1_power": {},
            "ev2_power": {},
            "bess_power": {},
            "total_generation": {},
            "total_consumption": {},
            "total_cost": {}
        } # histories from start_time to now

        self.controls = {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0
        }


    def apply_controls(self, controls, duration_hours=0.25):
        if controls is not None:
            self.controls = controls

        # apply controls to controllable participants
        if self.bess and "bess_power" in controls:
            power = controls["bess_power"]
            if power > 0:
                self.bess.charge(power, duration_hours)
            elif power < 0:
                self.bess.discharge(-power, duration_hours)

        if self.ev1 and "ev1_power" in controls:
            if (self.ev1.at_home or self.ev1.at_charging_station):
                power = controls["ev1_power"]
                if power > 0:
                    self.ev1.charge(power, duration_hours)
                # EVs cannot discharge to the grid in this model

        if self.ev2 and "ev2_power" in controls:
            if (self.ev2.at_home or self.ev2.at_charging_station):
                power = controls["ev2_power"]
                if power > 0:
                    self.ev2.charge(power, duration_hours)
                # EVs cannot discharge to the grid in this model

        for ev in [self.ev1, self.ev2]:
            if ev and not (ev.at_home or ev.at_charging_station):
                # apply driving load (load = 0 when idle)
                ev.discharge(ev.load, duration_hours)


    def update_history(self):
        '''
        logs current states to history for current timestep
        '''
        # base load
        self.history["base_load"][self.current_timestep] = self.base_load
        self.history["buy_price"][self.current_timestep] = self.buy_price
        self.history["sell_price"][self.current_timestep] = self.sell_price

        # PV
        if self.pv:
            self.history["pv_gen"][self.current_timestep] = self.pv.generation
        else:
            self.history["pv_gen"][self.current_timestep] = 0.0

        # BESS
        if self.bess:
            self.history["bess_soc"][self.current_timestep] = self.bess.soc
        else:
            self.history["bess_soc"][self.current_timestep] = 0.0

        # EV1
        if self.ev1:
            self.history["ev1_soc"][self.current_timestep] = self.ev1.soc
            self.history["ev1_load"][self.current_timestep] = self.ev1.load
            self.history["ev1_at_home"][self.current_timestep] = int(self.ev1.at_home)
            self.history["ev1_at_charging_station"][self.current_timestep] = int(self.ev1.at_charging_station)
        else:
            self.history["ev1_soc"][self.current_timestep] = 0.0
            self.history["ev1_load"][self.current_timestep] = 0.0
            self.history["ev1_at_home"][self.current_timestep] = 0
            self.history["ev1_at_charging_station"][self.current_timestep] = 0

        # EV2
        if self.ev2:
            self.history["ev2_soc"][self.current_timestep] = self.ev2.soc
            self.history["ev2_load"][self.current_timestep] = self.ev2.load
            self.history["ev2_at_home"][self.current_timestep] = int(self.ev2.at_home)
            self.history["ev2_at_charging_station"][self.current_timestep] = int(self.ev2.at_charging_station)
        else:
            self.history["ev2_soc"][self.current_timestep] = 0.0
            self.history["ev2_load"][self.current_timestep] = 0.0
            self.history["ev2_at_home"][self.current_timestep] = 0
            self.history["ev2_at_charging_station"][self.current_timestep] = 0

        # controls
        for control_key in self.controls:
            self.history[control_key][self.current_timestep] = self.controls[control_key]

        # net load and cost
        self.history["net_cost"][self.current_timestep] = self.net_cost
        self.history["net_load"][self.current_timestep] = self.net_load

        # totals
        self.history["total_generation"][self.current_timestep] = self.total_generation
        self.history["total_consumption"][self.current_timestep] = self.total_consumption
        self.history["total_cost"][self.current_timestep] = self.total_cost


    def plot_history(self):
        '''
        plots each measurement in its own figure
        '''
        for key, series in self.history.items():
            if not series:
                continue
            t = list(series.keys())
            y = list(series.values())

            plt.figure()
            plt.plot(t, y)
            plt.title(key)
            plt.xlabel("time step")
            plt.ylabel(key)
            plt.grid(True)

        plt.show()


    def plot_history_all(self, plots:list|None=None):
        '''
        plots: list of measurement names to plot. if None, plot all.'''
        plt.figure(figsize=(12, 6))

        for key, series in self.history.items():
            if not series:
                continue

            if plots and key not in plots:
                continue

            t = list(series.keys())
            y = list(series.values())
            plt.plot(t, y, label=key)

        plt.xlabel("time step")
        plt.ylabel("value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @property
    def has_pv(self):
        return self.pv is not None

    @property
    def has_bess(self):
        return self.bess is not None

    @property
    def net_cost(self):
        if self.net_load > 0:
            return self.net_load * self.buy_price
        else:
            return self.net_load * -self.sell_price

    @property
    def net_load(self):
        base_load = self.base_load if self.base_load else 0.0
        pv_generation = self.pv.generation if self.pv else 0.0
        bess_load = self.controls.get("bess_power", 0.0) if self.bess else 0.0
        ev1_load = self.controls.get("ev1_power", 0.0) if self.ev1 and self.ev1.at_home else 0.0
        ev2_load = self.controls.get("ev2_power", 0.0) if self.ev2 and self.ev2.at_home else 0.0
        return base_load + bess_load + ev1_load + ev2_load - pv_generation

    @property
    def total_generation(self):
        return sum(self.history["pv_gen"].values()) * 0.25 if self.pv else 0.0

    @property
    def total_consumption(self):
        return sum(self.history["net_load"].values()) * 0.25

    @property
    def total_cost(self):
        return sum(self.history["net_cost"].values()) * 0.25 + self.fixed_cost
