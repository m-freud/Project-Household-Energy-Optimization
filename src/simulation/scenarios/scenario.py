from dataclasses import dataclass, field


@dataclass
class DeviceScenario:
    start_soc: float
    soc_allowed_range: tuple[float, float]
    soc_targets: dict[int, float]


def _empty_device_scenario() -> DeviceScenario:
    return DeviceScenario(start_soc=0.0, soc_allowed_range=(0.0, 1.0), soc_targets={})

@dataclass
class Scenario:
    name: str
    ev1: DeviceScenario = field(default_factory=_empty_device_scenario)
    ev2: DeviceScenario = field(default_factory=_empty_device_scenario)
    bess: DeviceScenario = field(default_factory=_empty_device_scenario)


scenarios = [
    Scenario(
        name="default_scenario",
        ev1=DeviceScenario(
            start_soc=0.1,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.9},
        ),
        ev2=DeviceScenario(
            start_soc=0.1,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.9},
        ),
        bess=DeviceScenario(
            start_soc=0.1,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.9},
        ),
    ),
    Scenario(
        name="low_start_wide",
        ev1=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={32: 0.3, 64: 0.6, 96: 0.8},
        ),
        ev2=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={32: 0.3, 64: 0.6, 96: 0.8},
        ),
        bess=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.8},
        ),
    ),
    Scenario(
        name="mid_start_normal",
        ev1=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.4, 64: 0.65, 96: 0.8},
        ),
        ev2=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.4, 64: 0.65, 96: 0.8},
        ),
        bess=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={96: 0.8},
        ),
    ),
    Scenario(
        name="high_start_narrow",
        ev1=DeviceScenario(
            start_soc=0.7,
            soc_allowed_range=(0.3, 0.7),
            soc_targets={32: 0.5, 64: 0.65, 96: 0.7},
        ),
        ev2=DeviceScenario(
            start_soc=0.7,
            soc_allowed_range=(0.3, 0.7),
            soc_targets={32: 0.5, 64: 0.65, 96: 0.7},
        ),
        bess=DeviceScenario(
            start_soc=0.7,
            soc_allowed_range=(0.3, 0.7),
            soc_targets={96: 0.7},
        ),
    ),
    Scenario(
        name="early_urgency",
        ev1=DeviceScenario(
            start_soc=0.3,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.65, 64: 0.75, 96: 0.85},
        ),
        ev2=DeviceScenario(
            start_soc=0.3,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.65, 64: 0.75, 96: 0.85},
        ),
        bess=DeviceScenario(
            start_soc=0.3,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={96: 0.85},
        ),
    ),
    Scenario(
        name="late_relaxed",
        ev1=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={32: 0.35, 64: 0.5, 96: 0.75},
        ),
        ev2=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={32: 0.35, 64: 0.5, 96: 0.75},
        ),
        bess=DeviceScenario(
            start_soc=0.5,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.75},
        ),
    ),
    Scenario(
        name="stressed_ev_buffered_bess",
        ev1=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.5, 64: 0.75, 96: 0.85},
        ),
        ev2=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.2, 0.8),
            soc_targets={32: 0.5, 64: 0.75, 96: 0.85},
        ),
        bess=DeviceScenario(
            start_soc=0.2,
            soc_allowed_range=(0.1, 0.9),
            soc_targets={96: 0.8},
        ),
    ),
]

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in scenarios}
default_scenario = SCENARIOS_BY_NAME["default_scenario"]


def get_scenario_names() -> list[str]:
    return [scenario.name for scenario in scenarios]


def get_scenario_value(
    scenario_name: str,
    device_name: str,
    value,
):
    scenario = SCENARIOS_BY_NAME.get(scenario_name)
    if scenario is None:
        return None

    device_scenario = getattr(scenario, device_name, None)
    if device_scenario is None:
        return None

    if value == "deadline":
        if not device_scenario.soc_targets:
            return None
        return max(device_scenario.soc_targets.keys())

    if value == "target_soc":
        if not device_scenario.soc_targets:
            return None
        deadline = max(device_scenario.soc_targets.keys())
        return device_scenario.soc_targets.get(deadline)

    return getattr(device_scenario, str(value), None)
