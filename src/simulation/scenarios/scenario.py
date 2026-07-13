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
    ev_targets: dict[int, float],
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


def _build_uniform_scenario(
    *,
    name: str,
    start_soc: float,
    soc_allowed_range: tuple[float, float],
    ev_targets: dict[int, float],
    bess_final_target: float,
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
    "low": 0.2,
    "mid": 0.5,
    "high": 0.8,
}

EV_TARGET_PROFILES = {
    "relaxed": {32: 0.35, 64: 0.5, 96: 0.8},
    "stressed": {32: 0.6, 64: 0.75, 96: 0.9},
}

SCENARIO_METADATA = {
    "default_scenario": {"start_level": "mid", "urgency": "relaxed"},
    "relaxed_low_start": {"start_level": "low", "urgency": "relaxed"},
    "relaxed_high_start": {"start_level": "high", "urgency": "relaxed"},
    "stressed_low_start": {"start_level": "low", "urgency": "stressed"},
    "stressed_mid_start": {"start_level": "mid", "urgency": "stressed"},
    "stressed_high_start": {"start_level": "high", "urgency": "stressed"},
}


scenarios = [
    _build_uniform_scenario(
        name="default_scenario",
        start_soc=START_SOC_LEVELS["mid"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["relaxed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
    _build_uniform_scenario(
        name="relaxed_low_start",
        start_soc=START_SOC_LEVELS["low"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["relaxed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
    _build_uniform_scenario(
        name="relaxed_high_start",
        start_soc=START_SOC_LEVELS["high"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["relaxed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
    _build_uniform_scenario(
        name="stressed_low_start",
        start_soc=START_SOC_LEVELS["low"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["stressed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
    _build_uniform_scenario(
        name="stressed_mid_start",
        start_soc=START_SOC_LEVELS["mid"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["stressed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
    _build_uniform_scenario(
        name="stressed_high_start",
        start_soc=START_SOC_LEVELS["high"],
        soc_allowed_range=FIXED_SOC_ALLOWED_RANGE,
        ev_targets=EV_TARGET_PROFILES["stressed"],
        bess_final_target=FIXED_BESS_FINAL_TARGET,
    ),
]

SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in scenarios}
default_scenario = SCENARIOS_BY_NAME["default_scenario"]


def get_scenario_names() -> list[str]:
    return [scenario.name for scenario in scenarios]


def get_scenario_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        metadata = SCENARIO_METADATA.get(scenario.name, {})
        rows.append(
            {
                "scenario": scenario.name,
                "urgency": metadata.get("urgency", "unknown"),
                "start_level": metadata.get("start_level", "unknown"),
                "start_soc": scenario.bess.start_soc,
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
