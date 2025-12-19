from src.simulation.Household import Household


def no_control(household:Household, t):
    """A policy that does not control anything, all controls are set to zero."""
    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0,
    }

    return controls


def random_control(household:Household, t):
    """A policy that sets random controls within the allowed limits."""
    import random

    controls = {
        "bess_power": random.uniform(-household.bess.max_discharge, household.bess.max_charge) if household.bess else 0.0,
        "ev1_power": random.uniform(0, household.ev1.max_charge) if household.ev1 else 0.0,
        "ev2_power": random.uniform(0, household.ev2.max_charge) if household.ev2 else 0.0,
    }

    return controls
