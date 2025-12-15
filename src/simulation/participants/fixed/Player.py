

class Player:
    def __init__(self, load=None, ev1_load=None, ev2_load=None):
        self.load = load
        self.ev1_load = ev1_load
        self.ev2_load = ev2_load

    def set_load(self, load):
        self.load = load

    def get_load(self):
        return self.load

    def set_ev1_load(self, ev1_load):
        self.ev1_load = ev1_load

    def get_ev1_load(self):
        return self.ev1_load

    def set_ev2_load(self, ev2_load):
        self.ev2_load = ev2_load

    def get_ev2_load(self):
        return self.ev2_load
