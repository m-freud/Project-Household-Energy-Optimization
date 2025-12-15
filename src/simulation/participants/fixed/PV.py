
class PV:
    def __init__(self, pv_generation):
        self.pv_generation = pv_generation  # list of PV generation values over time

    @property
    def generation(self):
        return self.pv_generation
    
    @generation.setter
    def generation(self, pv_generation):
        self.pv_generation = pv_generation