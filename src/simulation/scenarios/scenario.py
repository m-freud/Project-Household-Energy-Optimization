from dataclasses import dataclass, field

from src.sqlite_connection import load_attribute


@dataclass
class DeviceScenario:
    start_soc: float
    soc_allowed_range: tuple[float, float]
    soc_targets: dict[int, float]

@dataclass
class Scenario:
    name: str
    ev1: DeviceScenario = field(default_factory=DeviceScenario)
    ev2: DeviceScenario = field(default_factory=DeviceScenario)
    bess: DeviceScenario = field(default_factory=DeviceScenario)


default_ev_scenario = DeviceScenario(
    start_soc=0.2,
    soc_allowed_range=(0.1, 0.9),
    soc_targets={96: 0.8}
)

default_bess_scenario = DeviceScenario(
    start_soc=0.2,
    soc_allowed_range=(0.1, 0.9),
    soc_targets={96: 0.8}
)

default_scenario = Scenario(
    name="default_scenario",
    ev1=default_ev_scenario,
    ev2=default_ev_scenario,
    bess=default_bess_scenario
)


def get_scenario_value(
    scenario_name: str,
    device_name: str,
    player_id: int,
    value,
):
    for scenario in [default_scenario]:
        if scenario_name == scenario.name:
            break
    else:
        return None

    device_scenario = getattr(scenario, device_name, None)
    if device_scenario is None:
        return None

    return getattr(device_scenario, str(value), None)
