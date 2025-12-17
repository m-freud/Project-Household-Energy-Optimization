

class EV:
    def __init__(self, capacity, max_charge, max_discharge, efficiency, soc=0, load=0, at_home=False, at_charging_station=False):
        self.capacity = capacity  # in kWh
        self.max_charge = max_charge  # in kW (per hour)
        self.max_discharge = max_discharge  # in kW (per hour)
        self.efficiency = efficiency  # as a decimal
        self._soc = soc  # state of charge in kWh
        self._load = load  # current load in kW
        self._at_home = at_home  # boolean
        self._at_charging_station = at_charging_station  # boolean
        self._buy_price = 0.0  # current (white) power price for charging

    def charge(self, power, duration_hours):
        charge_power = min(power, self.max_charge / (duration_hours * self.efficiency))
        energy_added = charge_power * duration_hours * self.efficiency
        self.soc = min(self.capacity, self.soc + energy_added)
        return energy_added
    
    def discharge(self, power, duration_hours):
        energy_removed = power * duration_hours
        self.soc = max(0, self.soc - energy_removed)
    
    @property
    def buy_price(self):
        return self._buy_price

    @buy_price.setter
    def buy_price(self, price):
        self._buy_price = price
    
    @property
    def soc(self):
        return self._soc
    
    @soc.setter
    def soc(self, value):
        self._soc = max(0, min(value, self.capacity))

    @property
    def load(self):
        return self._load
    
    @load.setter
    def load(self, value):
        self._load = value

    @property
    def at_home(self):
        return self._at_home

    @at_home.setter
    def at_home(self, value):
        self._at_home = value

    @property
    def at_charging_station(self):
        return self._at_charging_station
    
    @at_charging_station.setter
    def at_charging_station(self, value):
        self._at_charging_station = value

    
    def get_capacity(self):
        return self.capacity
    
    def get_max_charge(self):
        return self.max_charge
    
    def get_max_discharge(self):
        return self.max_discharge
    
    def get_efficiency(self):
        return self.efficiency
