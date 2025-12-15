from src.simulation.participants.fixed.Player import Player
from src.simulation.participants.fixed.PV import PV
from src.simulation.participants.controllable.BESS import BESS
from src.simulation.participants.controllable.EV import EV




class Household:
    # could also be Controller. The point is the household contains all components and makes all energy management decisions.
    def __init__(self, player=None, pv=None, bess=None, ev1=None, ev2=None):
        # you can create empty or full
        self.player = player
        self.pv = pv
        self.bess = bess
        self.ev1 = ev1
        self.ev2 = ev2


    # add participants
    def add_player(self, player: Player):
        self.player = player

    def add_pv(self, pv: PV):
        self.pv = pv
    
    def add_bess(self, bess: BESS):
        self.bess = bess
    
    def add_ev1(self, ev: EV):
        self.ev1 = ev
    
    def add_ev2(self, ev: EV):
        self.ev2 = ev
