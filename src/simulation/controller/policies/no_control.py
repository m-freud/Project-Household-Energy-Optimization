from src.simulation.participants.Household import Household

def no_control_policy(household:Household, t):
    """A policy that does not control anything, all controls are set to zero."""
    controls = {
        "bess_power": 0.0,
        "ev1_power": 0.0,
        "ev2_power": 0.0,
    }

    return controls
