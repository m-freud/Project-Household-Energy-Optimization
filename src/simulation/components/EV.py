

class EV:
    def __init__(self, capacity,
                 max_charge, max_discharge,
                 efficiency, soc=0.0, load=0.0,
                 at_home=False, at_charging_station=False,
                 name="ev"):
        self.capacity = capacity  # in kWh
        self.max_charge = max_charge  # in kW (per hour)
        self.max_discharge = max_discharge  # in kW (per hour)
        self.efficiency = efficiency  # as a decimal
        self._soc = soc  # state of charge in kWh
        self.load = load  # current load in kW
        self.at_home = at_home  # boolean
        self.at_charging_station = at_charging_station  # boolean
        self.buy_price = 0.0  # current (white) power price for charging
        self.name = name


    def charge(self, power, duration_hours):
        charge_power = min(power, self.max_charge)
        energy_added = charge_power * duration_hours * self.efficiency
        self.soc = min(self.capacity, self.soc + energy_added)

        return energy_added
    

    def discharge(self, power, duration_hours):
        discharge_power = min(power, self.max_discharge)
        energy_removed = discharge_power * duration_hours / self.efficiency
        self.soc = max(0, self.soc - energy_removed)

        return energy_removed
    
    @property
    def soc(self):
        return float(self._soc)

    @soc.setter
    def soc(self, value):
        self._soc = max(0, min(value, self.capacity))
        
    @property
    def soc_fraction(self):
        return self.soc / self.capacity
