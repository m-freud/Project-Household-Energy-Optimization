import math

def encode_time_cyclic(timestep: int, total_timesteps: int) -> tuple[float, float]:
    """
    Encode a timestep as cyclic features (sine and cosine).

    Args:
        timestep (int): The current timestep to encode.
        total_timesteps (int): The total number of timesteps in the cycle.

    Returns:
        tuple[float, float]: A tuple containing the sine and cosine values for the encoded timestep.
    """
    angle = 2 * math.pi * (timestep / total_timesteps)
    return math.sin(angle), math.cos(angle)
