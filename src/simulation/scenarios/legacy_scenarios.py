from src.simulation.scenarios.scenario import DeviceScenario, Scenario


legacy_scenarios = [
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

LEGACY_SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in legacy_scenarios}


def get_legacy_scenario_names() -> list[str]:
    return [scenario.name for scenario in legacy_scenarios]


def get_legacy_scenario(scenario_name: str) -> Scenario | None:
    return LEGACY_SCENARIOS_BY_NAME.get(scenario_name)
