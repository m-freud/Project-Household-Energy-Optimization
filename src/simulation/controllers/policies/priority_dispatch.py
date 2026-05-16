from pathlib import Path
import sys

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.config import Config


# ── helpers ──────────────────────────────────────────────────────────────────

def _next_target(timestep: int, soc_targets: dict) -> tuple[float, int]:
    deadline = min((t for t in soc_targets if t >= timestep), default=96)
    return soc_targets.get(deadline, 0.0), deadline


def _to_kwh(value: float, capacity: float) -> float:
    return value * capacity if value <= 1.0 else value


def _is_urgent(timestep: int, deadline: int, threshold: int = 4) -> bool:
    """True when fewer than `threshold` steps remain before the deadline."""
    return (deadline - timestep) <= threshold


def _price_quartiles(profile: list) -> tuple[float | None, float | None]:
    """Return (25th-pct, 75th-pct) price thresholds from the day profile."""
    if not profile:
        return None, None
    s = sorted(profile)
    n = len(s)
    return s[max(0, int(n * 0.25) - 1)], s[min(int(n * 0.75), n - 1)]
# so this sorts the profile by value
# and then returns the values at 0.25 and 0.75 of len(profile)


# ── main policy ──────────────────────────────────────────────────────────────

def priority_dispatch_policy(household: Household, scenario: Scenario) -> dict:
    """
    Priority-based (waterfall) energy management controller.

    Rule priority per device:
      1. Target already met AND deadline not urgent  →  idle
      2. Unallocated PV surplus available            →  charge with surplus
      3. Buy price ≤ 25th-pct of day profile        →  fast charge
      4. Buy price ≥ 75th-pct AND not urgent        →  idle (BESS: discharge to cover load)
      5. Fallback                                    →  even-linear to next target
    """
    controls = {"ev1_power": 0.0, "ev2_power": 0.0, "bess_power": 0.0}

    t = household.current_timestep
    buy_price = household.buy_price
    q25, q75 = _price_quartiles(household.buy_price_day_profile)

    price_cheap = q25 is not None and buy_price <= q25
    price_expensive = q75 is not None and buy_price >= q75

    pv_gen   = household.pv.generation if household.pv else 0.0
    base_load = household.base_load or 0.0
    # PV power available beyond fixed base load, shrinks as storage absorbs it
    pv_surplus = max(0.0, pv_gen - base_load)

    # ── EVs ──────────────────────────────────────────────────────────────────
    for ev_name, ev, dev_scenario in [
        ("ev1_power", household.ev1, scenario.ev1),
        ("ev2_power", household.ev2, scenario.ev2),
    ]:
        if ev is None or not (ev.at_home or ev.at_charging_station):
            continue

        ev_profile = getattr(
            household,
            f"{ev.name}_buy_price_day_profile",
            household.buy_price_day_profile,
        )
        ev_q25, ev_q75 = _price_quartiles(ev_profile)
        ev_price_cheap = ev_q25 is not None and ev.buy_price <= ev_q25
        ev_price_expensive = ev_q75 is not None and ev.buy_price >= ev_q75

        target_kwh, deadline = _next_target(t, dev_scenario.soc_targets)
        target_kwh = _to_kwh(target_kwh, ev.capacity)
        deficit = target_kwh - ev.soc
        urgent = _is_urgent(t, deadline)
        remaining = max(deadline - t, 1)

        if deficit <= 0 and not urgent:
            # Rule 1 – already on track
            pass
        elif pv_surplus > 0 and deficit > 0:
            # Rule 2 – absorb PV surplus
            power = min(pv_surplus, ev.max_charge)
            controls[ev_name] = power
            pv_surplus -= power
        elif ev_price_cheap and deficit > 0:
            # Rule 3 – cheap grid energy
            controls[ev_name] = min(ev.max_charge,
                                    deficit / Config.DURATION_TIMESTEP)
        elif ev_price_expensive and not urgent:
            # Rule 4 – hold off
            pass
        elif deficit > 0:
            # Rule 5 – even-linear guarantee
            controls[ev_name] = min(
                (deficit / (remaining * Config.DURATION_TIMESTEP)) / ev.efficiency,
                ev.max_charge,
            )

    # ── BESS ─────────────────────────────────────────────────────────────────
    if household.bess:
        bess = household.bess
        target_kwh, deadline = _next_target(t, scenario.bess.soc_targets)
        target_kwh = _to_kwh(target_kwh, bess.capacity)
        deficit = target_kwh - bess.soc
        urgent = _is_urgent(t, deadline)
        remaining = max(deadline - t, 1)

        # net load after EV decisions, before BESS
        net_load_excl_bess = (base_load
                              + controls["ev1_power"]
                              + controls["ev2_power"]
                              - pv_gen)

        def _discharge_power() -> float:
            return min(
                max(0.0, net_load_excl_bess),
                bess.max_discharge,
                bess.soc * 4 * bess.capacity,
            )

        if deficit <= 0 and not urgent:
            # Rule 1 – target met; opportunistically discharge at peak price
            if price_expensive and net_load_excl_bess > 0:
                controls["bess_power"] = -_discharge_power()
        elif pv_surplus > 0 and deficit > 0:
            # Rule 2 – soak up remaining PV surplus
            controls["bess_power"] = min(pv_surplus, bess.max_charge)
        elif price_cheap and deficit > 0:
            # Rule 3 – cheap grid charging
            controls["bess_power"] = min(bess.max_charge,
                                         deficit / Config.DURATION_TIMESTEP)
        elif price_expensive and not urgent:
            # Rule 4 – discharge to offset expensive grid import
            if net_load_excl_bess > 0:
                controls["bess_power"] = -_discharge_power()
        else:
            # Rule 5 – even-linear fallback
            if deficit > 0:
                base_charge = min(
                    (deficit / (remaining * Config.DURATION_TIMESTEP)) / bess.efficiency,
                    bess.max_charge,
                )
                # also absorb any remaining PV surplus on top
                controls["bess_power"] = min(base_charge + pv_surplus, bess.max_charge)
            elif net_load_excl_bess > 0:
                controls["bess_power"] = -_discharge_power()

    return controls
