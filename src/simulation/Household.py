from src.simulation.participants.fixed.Player import Player
from src.simulation.participants.fixed.PV import PV
from src.simulation.participants.controllable.BESS import BESS
from src.simulation.participants.controllable.EV import EV




class Household:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.load_profile = []
        self.pv_profile = []


    def generate_new_load_profile(self, strategy: str):
        pass  
    
    # where do we actually want to get this from? sim level? household level? player level? then we need to pass the connections down.

    # it seems we have two options: let the simulation load everything and pass components to the household,
    # or let each component perform its own loading.