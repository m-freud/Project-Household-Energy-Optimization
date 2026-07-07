from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceBuffer:
    # Pull deadline constraints earlier by this many 15-minute steps.
    time_buffer_steps: int = 0
    # Reserved for later tightening in SOC-percent points (0.05 = +5% SOC).
    energy_buffer_soc_frct: float = 0.0


@dataclass(frozen=True)
class DeviceBufferConfig:
    bess: DeviceBuffer = DeviceBuffer()
    ev1: DeviceBuffer = DeviceBuffer()
    ev2: DeviceBuffer = DeviceBuffer()

    @classmethod
    def with_universal_time_buffer(cls, steps: int) -> "DeviceBufferConfig":
        device = DeviceBuffer(time_buffer_steps=max(0, int(steps)))
        return cls(bess=device, ev1=device, ev2=device)
    
    @classmethod
    def with_universal_energy_buffer(cls, soc_frct: float) -> "DeviceBufferConfig":
        device = DeviceBuffer(energy_buffer_soc_frct=max(0.0, float(soc_frct)))
        return cls(bess=device, ev1=device, ev2=device)
    
    @classmethod
    def with_universal_buffer(cls, steps: int, soc_frct: float) -> "DeviceBufferConfig":
        device = DeviceBuffer(
            time_buffer_steps=max(0, int(steps)),
            energy_buffer_soc_frct=max(0.0, float(soc_frct)),
        )
        return cls(bess=device, ev1=device, ev2=device)

    @classmethod
    def disabled(cls) -> "DeviceBufferConfig":
        return cls()