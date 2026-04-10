"""
create forecasts for:
price
load
pv generation
ev position

and then combine them modularly 

define cost function based on forecasts
use bookshelf optimizer to minimize cost function



"""


def model_predictive_control(household, policy):
    