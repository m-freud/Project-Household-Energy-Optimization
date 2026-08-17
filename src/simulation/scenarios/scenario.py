from dataclasses import dataclass, field


@dataclass
class DeviceScenario:
    start_soc: float
    soc_allowed_range: tuple[float, float]
    soc_targets: dict[int, float]


def _empty_device_scenario() -> DeviceScenario:
    return DeviceScenario(start_soc=0.0, soc_allowed_range=(0.0, 1.0), soc_targets={})


def _make_uniform_device_scenario(
    *,
    start_soc: float,
    soc_allowed_range: tuple[float, float],
    ev_targets: dict[int, float],  # uniform across EVs
    bess_final_target: float,
) -> tuple[DeviceScenario, DeviceScenario, DeviceScenario]:
    """Create matching EV profiles with a simpler BESS final target profile."""
    ev1 = DeviceScenario(
        start_soc=start_soc,
        soc_allowed_range=soc_allowed_range,
        soc_targets=dict(ev_targets),
    )
    ev2 = DeviceScenario(
        start_soc=start_soc,
        soc_allowed_range=soc_allowed_range,
        soc_targets=dict(ev_targets),
    )
    bess = DeviceScenario(
        start_soc=start_soc,
        soc_allowed_range=soc_allowed_range,
        soc_targets={96: bess_final_target},
    )
    return ev1, ev2, bess


def _build_uniform_scenario( # all scenarios are uniform across devices
    *,
    name: str,
    start_soc: float,
    soc_allowed_range: tuple[float, float],
    ev_targets: dict[int, float],
    bess_final_target: float=0.5,
) -> "Scenario":
    ev1, ev2, bess = _make_uniform_device_scenario(
        start_soc=start_soc,
        soc_allowed_range=soc_allowed_range,
        ev_targets=ev_targets,
        bess_final_target=bess_final_target,
    )
    return Scenario(name=name, ev1=ev1, ev2=ev2, bess=bess)


@dataclass
class Scenario:
    name: str
    ev1: DeviceScenario = field(default_factory=_empty_device_scenario)
    ev2: DeviceScenario = field(default_factory=_empty_device_scenario)
    bess: DeviceScenario = field(default_factory=_empty_device_scenario)


FIXED_SOC_ALLOWED_RANGE = (0.2, 0.8)
FIXED_BESS_FINAL_TARGET = 0.5

START_SOC_LEVELS = {
    "low": 0.1,
    "mid": 0.5,
    "high": 0.8,
}

EV_TARGET_PROFILES = {
    "relaxed": {32: 0.35, 64: 0.5, 96: 0.8},
    "stressed": {32: 0.6, 64: 0.75, 96: 0.9},
}



DEFAULT_SOC_RANGE = (0.2, 0.8)
LOW = DEFAULT_SOC_RANGE[0]
MID = 0.5
HIGH = DEFAULT_SOC_RANGE[1]


scenarios = {
    "00_no_targets_baseline": _build_uniform_scenario(
        name="00_no_targets_baseline",
        start_soc=0.5,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={},
    ),
    "01_relaxed_flexible": _build_uniform_scenario(
        name="01_relaxed_flexible",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: LOW,
            96: MID
            },
    ),
    "02": _build_uniform_scenario(
        name="02",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "02a": _build_uniform_scenario(
        name="02a",
        start_soc=LOW,
        soc_allowed_range=(0.05, 0.95),
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "02b": _build_uniform_scenario(
        name="02b",
        start_soc=LOW,
        soc_allowed_range=(0.3, 0.7),
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "12": _build_uniform_scenario(
        name="12",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: MID,
            96: HIGH
            },
    ),
    "22": _build_uniform_scenario(
        name="22",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: HIGH,
            96: HIGH
            },
    ),
}


default_scenario = scenarios["02"]


def get_scenario_names() -> list[str]:
    return list(scenarios.keys())


def get_scenario_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_name, scenario in scenarios.items():
        rows.append(
            {
                "scenario": scenario_name,
                "start_soc": scenario.ev1.start_soc,
                "soc_allowed_range": f"{scenario.bess.soc_allowed_range[0]:.1f}-{scenario.bess.soc_allowed_range[1]:.1f}",
                "bess_target_eod": scenario.bess.soc_targets.get(96),
                "ev_targets": str(scenario.ev1.soc_targets),
            }
        )
    return rows


def get_scenario_value(
    scenario_name: str,
    device_name: str,
    value,
):
    scenario = scenarios.get(scenario_name)
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

    if value == "soc_targets":
        return dict(device_scenario.soc_targets)

    return getattr(device_scenario, str(value), None)


if __name__ == "__main__":
    for row in get_scenario_summary_rows():
        print(row)

    example_scenario_name = "02"
    example_device_name = "ev1"
    example_value = "target_soc"
    example_value_result = get_scenario_value(
        scenario_name=example_scenario_name,
        device_name=example_device_name,
        value=example_value,
    )

    print(
        f"Example: scenario={example_scenario_name}, device={example_device_name}, value={example_value} => {example_value_result}"
    )