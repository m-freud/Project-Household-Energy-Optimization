
class BESS:
    def __init__(
            self,
            capacity,
            max_charge,
            max_discharge,
            efficiency,
            soc=0.0,
            name="bess",
            ):
        self.capacity = capacity  # in kWh
        self.max_charge = max_charge  # in kW (per hour)
        self.max_discharge = max_discharge  # in kW (per hour)
        self.efficiency = efficiency  # as a decimal
        self.soc = soc  # state of charge in kWh
        self.name = name


    def charge(self, power, duration):
        charge_power = min(power, self.max_charge)
        energy_added = float(charge_power * duration * self.efficiency)
        self.soc += energy_added
        return energy_added

    
    def discharge(self, power, duration):
        discharge_power = min(power, self.max_discharge)
        energy_removed = discharge_power * duration / self.efficiency
        self.soc = max(0, self.soc - energy_removed)
        return energy_removed

    @property
    def soc(self):
        # convert to float
        return float(self._soc)

    @soc.setter
    def soc(self, value):
        self._soc = max(0, min(value, self.capacity))

    @property
    def soc_fraction(self):
        return self.soc / self.capacity
