

class PV:
    def __init__(self, pv_profile):
        self.pv_profile = pv_profile  # list of PV generation values over time

    
    def get_generation(self, timestep):
        if timestep < len(self.pv_profile):
            return self.pv_profile[timestep]
        else:
            return 0  # No generation beyond the profile length