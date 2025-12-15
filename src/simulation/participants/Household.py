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

        self.profiles = {} # histories from 0 to now

        self.timestep = 0  # current timestep in the simulation. maybe we need it, maybe not


    # add participants
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
