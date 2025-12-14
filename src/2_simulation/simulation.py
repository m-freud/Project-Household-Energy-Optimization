# Lets simulate for household 1. Write high level shit here and connect the dots.
from components import Household


household_1 = Household(player_id=1) # this already has ESS, PV, EV components inside.

optimizer = "optimizer1"

cost_profile1 = household_1.generate_cost_profile(load_profile)

load_to_influx(cost_profile1)