import math
from src.runtime_config import RuntimeConfig

def encode_time_cyclic(timestep: int) -> tuple[float, float]:
    """
    Encode a timestep as cyclic features (sine and cosine).

    Args:
        timestep (int): The current timestep to encode.

    Returns:
        tuple[float, float]: A tuple containing the sine and cosine values for the encoded timestep.
    """
    total_timesteps = RuntimeConfig.TOTAL_TIMESTEPS_DAY
    angle = 2 * math.pi * (timestep / total_timesteps)
    return math.sin(angle), math.cos(angle)
