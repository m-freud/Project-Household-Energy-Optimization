from dataclasses import dataclass, field


@dataclass
class DeviceRequirement:
    start_soc: float = 0.0
    target_soc: float = 0.0
    deadline: int = 0


@dataclass
class Scenario:
    name: str
    ev1: DeviceRequirement = field(default_factory=DeviceRequirement)
    ev2: DeviceRequirement = field(default_factory=DeviceRequirement)
    bess: DeviceRequirement = field(default_factory=DeviceRequirement)
