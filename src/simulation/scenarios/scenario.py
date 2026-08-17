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


@dataclass(frozen=True)
class ScenarioMetadata:
    label: str
    description: str


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
    "00_baseline": _build_uniform_scenario(
        name="00_baseline",
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
    "02_default_benchmark": _build_uniform_scenario(
        name="02_default_benchmark",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "02a_wide_soc_range": _build_uniform_scenario(
        name="02a_wide_soc_range",
        start_soc=LOW,
        soc_allowed_range=(0.05, 0.95),
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "02b_tight_soc_range": _build_uniform_scenario(
        name="02b_tight_soc_range",
        start_soc=LOW,
        soc_allowed_range=(0.3, 0.7),
        ev_targets={
            70: LOW,
            96: HIGH
            },
    ),
    "12_even_ramp_up": _build_uniform_scenario(
        name="12_even_ramp_up",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: MID,
            96: HIGH
            },
    ),
    "22_early_stress": _build_uniform_scenario(
        name="22_early_stress",
        start_soc=LOW,
        soc_allowed_range=DEFAULT_SOC_RANGE,
        ev_targets={
            70: HIGH,
            96: HIGH
            },
    ),
}


default_scenario = scenarios["02_default_benchmark"]


SCENARIO_METADATA = {
    "00_baseline": ScenarioMetadata(
        label="00 Baseline",
        description="No EV targets, default SOC range.",
    ),
    "01_relaxed_flexible": ScenarioMetadata(
        label="01 Relaxed Flexible",
        description="Low start SOC with a moderate final EV target.",
    ),
    "02_default_benchmark": ScenarioMetadata(
        label="02 Default Benchmark",
        description="Low start SOC with a high end-of-day EV target.",
    ),
    "02a_wide_soc_range": ScenarioMetadata(
        label="02a Wide SOC Range",
        description="Scenario 02 with a wider SOC operating band.",
    ),
    "02b_tight_soc_range": ScenarioMetadata(
        label="02b Tight SOC Range",
        description="Scenario 02 with a tighter SOC operating band.",
    ),
    "12_even_ramp_up": ScenarioMetadata(
        label="12 Even Ramp Up",
        description="Intermediate EV checkpoint before a high end-of-day target.",
    ),
    "22_early_stress": ScenarioMetadata(
        label="22 Early Stress",
        description="High EV targets at both checkpoints.",
    ),
}


def _validate_scenario_metadata() -> None:
    scenario_names = set(scenarios)
    metadata_names = set(SCENARIO_METADATA)

    missing_metadata = sorted(scenario_names - metadata_names)
    extra_metadata = sorted(metadata_names - scenario_names)
    if missing_metadata or extra_metadata:
        details: list[str] = []
        if missing_metadata:
            details.append(f"missing metadata for {missing_metadata}")
        if extra_metadata:
            details.append(f"unused metadata for {extra_metadata}")
        raise ValueError("Scenario metadata mismatch: " + "; ".join(details))


_validate_scenario_metadata()


def get_scenario_names() -> list[str]:
    return list(scenarios.keys())


def get_scenario_metadata_map() -> dict[str, ScenarioMetadata]:
    return dict(SCENARIO_METADATA)


def get_scenario_label(scenario_name: str) -> str:
    metadata = SCENARIO_METADATA.get(scenario_name)
    if metadata is None:
        return scenario_name
    return metadata.label


def get_scenario_summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_name, scenario in scenarios.items():
        metadata = SCENARIO_METADATA[scenario_name]
        rows.append(
            {
                "scenario": scenario_name,
                "label": metadata.label,
                "description": metadata.description,
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

    example_scenario_name = "02_default_benchmark"
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