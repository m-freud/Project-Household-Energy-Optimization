from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.policies.linear import even_linear_policy, fast_charge_policy
from src.simulation.controllers.policies.basic_examples import no_control as no_control_policy
from src.config import Config


def _q25_threshold(profile: list[float]) -> float | None:
	if not profile:
		return None
	sorted_profile = sorted(profile)
	idx = max(0, int(len(sorted_profile) * 0.25) - 1)
	return sorted_profile[idx]


def _next_target(timestep: int, soc_targets: dict) -> tuple[float, int]:
	"""Find the next deadline and its SOC target."""
	deadline = min((t for t in soc_targets if t >= timestep), default=96)
	return soc_targets.get(deadline, 0.0), deadline


def _is_trajectory_urgent(
	current_soc: float,
	target_soc: float,
	current_timestep: int,
	deadline: int,
	max_charge: float,
	efficiency: float,
	buffer_hours: float = 1.0,
) -> bool:
	"""Check whether the remaining window is too small for max-charge trajectory with a buffer."""
	if target_soc <= current_soc:
		return False

	if max_charge <= 0 or efficiency <= 0:
		return True

	remaining_hours = max(deadline - current_timestep, 0) * Config.DURATION_TIMESTEP
	required_hours = (target_soc - current_soc) / (max_charge * efficiency)
	return remaining_hours <= (required_hours + buffer_hours)


def price_aware_linear_1(household: Household, scenario: Scenario) -> dict:
	"""
	Default behavior: even_linear.
	Override: if a device price is in the cheapest quartile (<= 25th percentile), use fast_charge for that device.
	"""
	controls = even_linear_policy(household, scenario)
	fast_controls = fast_charge_policy(household, scenario)

	# EV1: compare ev1.buy_price against ev1 day-profile quartile
	ev1_profile = getattr(household, "ev1_buy_price_day_profile", household.buy_price_day_profile)
	ev1_q25 = _q25_threshold(ev1_profile)
	if household.ev1 and ev1_q25 is not None and household.ev1.buy_price <= ev1_q25:
		controls["ev1_power"] = fast_controls["ev1_power"]

	# EV2: compare ev2.buy_price against ev2 day-profile quartile
	ev2_profile = getattr(household, "ev2_buy_price_day_profile", household.buy_price_day_profile)
	ev2_q25 = _q25_threshold(ev2_profile)
	if household.ev2 and ev2_q25 is not None and household.ev2.buy_price <= ev2_q25:
		controls["ev2_power"] = fast_controls["ev2_power"]

	# Keep BESS behavior price-aware against household buy price.
	hh_q25 = _q25_threshold(household.buy_price_day_profile)
	if hh_q25 is not None and household.buy_price <= hh_q25:
		controls["bess_power"] = fast_controls["bess_power"]

	return controls

def price_aware_linear_2(household: Household, scenario: Scenario) -> dict:
	"""
	Default behavior: no_control.
	Urgency override: if deadline <= 3 hours away, use even_linear to guarantee target.
	Price override: if a device price is in the cheapest quartile (<= 25th percentile), use fast_charge for that device.
	"""
	controls = no_control_policy(household, scenario)
	even_controls = even_linear_policy(household, scenario)
	fast_controls = fast_charge_policy(household, scenario)
	
	t = household.current_timestep

	# EV1
	if household.ev1:
		ev1_target_soc, ev1_deadline = _next_target(t, scenario.ev1.soc_targets)
		ev1_target_soc = ev1_target_soc * household.ev1.capacity if ev1_target_soc <= 1.0 else ev1_target_soc
		if _is_trajectory_urgent(household.ev1.soc, ev1_target_soc, t, ev1_deadline, household.ev1.max_charge, household.ev1.efficiency):
			# Deadline urgent: use even_linear to guarantee target
			controls["ev1_power"] = even_controls["ev1_power"]
		else:
			# Not urgent: apply price logic
			ev1_profile = getattr(household, "ev1_buy_price_day_profile", household.buy_price_day_profile)
			ev1_q25 = _q25_threshold(ev1_profile)
			if ev1_q25 is not None and household.ev1.buy_price <= ev1_q25:
				controls["ev1_power"] = fast_controls["ev1_power"]

	# EV2
	if household.ev2:
		ev2_target_soc, ev2_deadline = _next_target(t, scenario.ev2.soc_targets)
		ev2_target_soc = ev2_target_soc * household.ev2.capacity if ev2_target_soc <= 1.0 else ev2_target_soc
		if _is_trajectory_urgent(household.ev2.soc, ev2_target_soc, t, ev2_deadline, household.ev2.max_charge, household.ev2.efficiency):
			# Deadline urgent: use even_linear to guarantee target
			controls["ev2_power"] = even_controls["ev2_power"]
		else:
			# Not urgent: apply price logic
			ev2_profile = getattr(household, "ev2_buy_price_day_profile", household.buy_price_day_profile)
			ev2_q25 = _q25_threshold(ev2_profile)
			if ev2_q25 is not None and household.ev2.buy_price <= ev2_q25:
				controls["ev2_power"] = fast_controls["ev2_power"]
			elif household.ev2.soc < ev2_target_soc and (household.ev2.at_home or household.ev2.at_charging_station):
				controls["ev2_power"] = even_controls["ev2_power"]

	# BESS
	if household.bess:
		bess_target_soc, bess_deadline = _next_target(t, scenario.bess.soc_targets)
		bess_target_soc = bess_target_soc * household.bess.capacity if bess_target_soc <= 1.0 else bess_target_soc
		if _is_trajectory_urgent(household.bess.soc, bess_target_soc, t, bess_deadline, household.bess.max_charge, household.bess.efficiency):
			# Deadline urgent: use even_linear to guarantee target
			controls["bess_power"] = even_controls["bess_power"]
		else:
			# Not urgent: apply price logic
			hh_q25 = _q25_threshold(household.buy_price_day_profile)
			if hh_q25 is not None and household.buy_price <= hh_q25:
				controls["bess_power"] = fast_controls["bess_power"]

	return controls