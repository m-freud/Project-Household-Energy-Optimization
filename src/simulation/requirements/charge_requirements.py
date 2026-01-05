# goal: minimize consumption.
# requirements: this

no_requirements = {
    "ev1": {
        "soc": 0.0,  # required state of charge (fraction of capacity)
        "deadline": 0,  # timestep by which required_soc must be met
    },
    "ev2": {
        "soc": 0.0,
        "deadline": 0,
    },
    "bess": {
        "soc": 0.0,  # required state of charge (fraction of capacity)
        "deadline": 0,  # timestep by which required_soc must be met
    },
}


basic_charge_requirements = {
    "ev1": {
        "soc": 0.5,  # required state of charge (fraction of capacity)
        "deadline": 96,  # timestep by which required_soc must be met
    },
    "ev2": {
        "soc": 0.5,
        "deadline": 96,
    },
    "bess": {
        "soc": 0.5,  # required state of charge (fraction of capacity)
        "deadline": 96,  # timestep by which required_soc must be met
    },
}