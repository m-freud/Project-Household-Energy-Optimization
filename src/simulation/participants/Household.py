from src.simulation.participants.fixed.Player import Player
from src.simulation.participants.fixed.PV import PV
from src.simulation.participants.controllable.BESS import BESS
from src.simulation.participants.controllable.EV import EV


class Household:
    def __init__(self, start_time=0, player=None, pv=None, bess=None, ev1=None, ev2=None):
        # timing info
        self.time = start_time  # start time of the simulation for this household

        # you can create empty or full
        self.player = player
        self.pv = pv
        self.bess = bess
        self.ev1 = ev1
        self.ev2 = ev2

        self.buy_price = 0.0  # current buy price for electricity
        self.sell_price = 0.0  # current sell price for electricity

        self.history = {
            "base_load": {},            
            "pv_gen": {},
            "bess_soc": {},
            "ev1_soc": {},
            "ev2_soc": {},
            "buy_price": {},
            "sell_price": {},
            "net_load": {},
            "cost": {},
            "bess_power": {},
            "ev1_power": {},
            "ev2_power": {},
            "total_generation": {},
            "total_consumption": {},
            "total_cost": {}
        } # histories from 0 to now

        self.controls = {
            "bess_power": 0,
            "ev1_power": 0,
            "ev2_power": 0
        }


    def set_controls(self, controls):
        # set controls for controllable participants based on policy
        if controls is not None:
            self.controls = controls

    
    def apply_controls(self, duration_hours=0.25):
        # apply controls to controllable participants
        if self.bess and "bess_power" in self.controls:
            power = self.controls["bess_power"]
            if power > 0:
                self.bess.charge(power, duration_hours)
            elif power < 0:
                self.bess.discharge(-power, duration_hours)

        if self.ev1 and "ev1_power" in self.controls and (self.ev1.at_home or self.ev1.at_charging_station):
            power = self.controls["ev1_power"]
            if power > 0:
                self.ev1.charge(power, duration_hours)
            # EVs cannot discharge to the grid in this model

        if self.ev2 and "ev2_power" in self.controls and (self.ev2.at_home or self.ev2.at_charging_station):
            power = self.controls["ev2_power"]
            if power > 0:
                self.ev2.charge(power, duration_hours)
            # EVs cannot discharge to the grid in this model

        for ev in [self.ev1, self.ev2]:
            if ev and not (ev.at_home or ev.at_charging_station):
                # apply driving load (load = 0 when idle)
                ev.discharge(ev.load, duration_hours)

    
    def update_history(self):
        # update history at each timestep
        if self.player: # TODO make sure base_load is tied to player everywhere
            self.history["base_load"][self.time] = self.player.load
        else:
            self.history["base_load"][self.time] = 0

        if self.pv:
            self.history["pv_gen"][self.time] = self.pv.generation
        else:
            self.history["pv_gen"][self.time] = 0

        if self.bess:
            self.history["bess_soc"][self.time] = self.bess.soc
        else:
            self.history["bess_soc"][self.time] = None

        if self.ev1:
            self.history["ev1_soc"][self.time] = self.ev1.soc
        else:
            self.history["ev1_soc"][self.time] = None

        if self.ev2:
            self.history["ev2_soc"][self.time] = self.ev2.soc
        else:
            self.history["ev2_soc"][self.time] = None

        self.history["buy_price"][self.time] = self.buy_price
        self.history["sell_price"][self.time] = self.sell_price

        self.history["cost"][self.time] = self.cost

        self.history["net_load"][self.time] = self.net_load

        self.history["bess_power"][self.time] = self.controls.get("bess_power", 0)
        self.history["ev1_power"][self.time] = self.controls.get("ev1_power", 0)
        self.history["ev2_power"][self.time] = self.controls.get("ev2_power", 0)

        self.history["total_generation"][self.time] = sum(self.history["pv_gen"].values()) * 0.25 if self.pv else 0
        self.history["total_consumption"][self.time] = sum(self.history["net_load"].values()) * 0.25
        self.history["total_cost"][self.time] = sum(self.history["cost"].values()) * 0.25


    def plot_history(self):
        import matplotlib.pyplot as plt
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

        import matplotlib.pyplot as plt

    def plot_history_all(self, plots=[]):
        import matplotlib.pyplot as plt
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




    # add participants
    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def player(self):
        return self._player

    @player.setter
    def player(self, player: Player|None):
        self._player = player

    @property
    def pv(self):
        return self._pv

    @pv.setter
    def pv(self, pv: PV|None):
        self._pv = pv

    @property
    def bess(self):
        return self._bess

    @bess.setter
    def bess(self, bess: BESS|None):
        self._bess = bess

    @property
    def ev1(self):
        return self._ev1

    @ev1.setter
    def ev1(self, ev: EV|None):
        self._ev1 = ev

    @property
    def ev2(self):
        return self._ev2

    @ev2.setter
    def ev2(self, ev: EV|None):
        self._ev2 = ev

    @property
    def buy_price(self):
        return self._buy_price if self._buy_price is not None else 0.0
    
    @buy_price.setter
    def buy_price(self, price):
        self._buy_price = price

    @property
    def sell_price(self):
        return self._sell_price if self._sell_price is not None else 0.0
    
    @sell_price.setter
    def sell_price(self, price):
        self._sell_price = price

    @property
    def cost(self):
        if self.net_load > 0:
            return self.net_load * self.buy_price
        else:
            return self.net_load * self.sell_price

    @property
    def net_load(self):
        base_load = self.player.load if self.player else 0
        bess_load = self.controls.get("bess_power", 0) if self.bess else 0
        ev1_load = self.controls.get("ev1_power", 0) if self.ev1 else 0
        ev2_load = self.controls.get("ev2_power", 0) if self.ev2 else 0
        generation = self.pv.generation if self.pv else 0

        return base_load + bess_load + ev1_load + ev2_load - generation

    
