from dataclasses import dataclass, field


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

no_requirements = Scenario(name="no_requirements")

default_scenario = Scenario(
    name="default_scenario",
    ev1=DeviceScenario(start_soc=0.2, target_soc=0.5, deadline=96),
    ev2=DeviceScenario(start_soc=0.2, target_soc=0.5, deadline=96),
    bess=DeviceScenario(start_soc=0.2, target_soc=0.5, deadline=96)
)
