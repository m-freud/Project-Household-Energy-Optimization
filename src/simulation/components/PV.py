
class PV:
    def __init__(self, generation=0):
        self._generation = generation  # list of PV generation values over time

    @property
    def generation(self):
        return self._generation
    
    @generation.setter
    def generation(self, pv_generation):
        self._generation = pv_generation