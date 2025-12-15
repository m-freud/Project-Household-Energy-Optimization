
class Player:
    def __init__(self, load_profile=None, ev1_load_profile=None, ev2_load_profile=None):
        self.load_profile = load_profile
        self.ev1_load_profile = ev1_load_profile
        self.ev2_load_profile = ev2_load_profile


    def set_load_profile(self, profile):
        self.load_profile = profile


    def set_ev1_load_profile(self, profile):
        self.ev1_load_profile = profile


    def set_ev2_load_profile(self, profile):
        self.ev2_load_profile = profile
