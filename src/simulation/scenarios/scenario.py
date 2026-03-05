from dataclasses import dataclass, field

from src.sqlite_connection import load_attribute


@dataclass
class DeviceScenario:
    start_soc: float = 0.0
    target_soc: float = 0.0
    deadline: int = 0


@dataclass
class Scenario:
    name: str
    ev1: DeviceScenario = field(default_factory=DeviceScenario)
    ev2: DeviceScenario = field(default_factory=DeviceScenario)
    bess: DeviceScenario = field(default_factory=DeviceScenario)

no_requirements = Scenario(
    name="no_requirements"
)

default_scenario = Scenario(
    name="default_scenario",
    ev1=DeviceScenario(start_soc=0.2, target_soc=0.5, deadline=96),
    ev2=DeviceScenario(start_soc=0.2, target_soc=0.5, deadline=96),
    bess=DeviceScenario(start_soc=0.0, target_soc=0.5, deadline=96)
)


def get_scenario_value(
    scenario_name: str,
    device_name: str,
    player_id: int,
    value,
):
    if scenario_name == default_scenario.name:
        scenario = default_scenario
    elif scenario_name == no_requirements.name:
        scenario = no_requirements
    else:
        return None

    device_scenario = getattr(scenario, device_name, None)
    if device_scenario is None:
        return None

    if callable(value):
        raw_value = value(device_scenario)
    else:
        raw_value = getattr(device_scenario, str(value), None)

    if raw_value is None:
        return None

    if str(value) in {"target_soc", "start_soc"}:
        capacity = load_attribute(device_name, player_id, "capacity")
        if capacity is None:
            return None
        return float(raw_value) * float(capacity)

    return raw_value
