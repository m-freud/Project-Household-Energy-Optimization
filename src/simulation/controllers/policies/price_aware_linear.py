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


def _is_deadline_urgent(timestep: int, deadline: int, hours_threshold: float = 3.0) -> bool:
	"""Check if deadline is within threshold hours. Default 3 hours = 12 timesteps (15-min each)."""
	timesteps_threshold = int(hours_threshold / Config.DURATION_TIMESTEP)
	return (deadline - timestep) <= timesteps_threshold


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
		_, ev1_deadline = _next_target(t, scenario.ev1.soc_targets)
		if _is_deadline_urgent(t, ev1_deadline):
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
		_, ev2_deadline = _next_target(t, scenario.ev2.soc_targets)
		if _is_deadline_urgent(t, ev2_deadline):
			# Deadline urgent: use even_linear to guarantee target
			controls["ev2_power"] = even_controls["ev2_power"]
		else:
			# Not urgent: apply price logic
			ev2_profile = getattr(household, "ev2_buy_price_day_profile", household.buy_price_day_profile)
			ev2_q25 = _q25_threshold(ev2_profile)
			if ev2_q25 is not None and household.ev2.buy_price <= ev2_q25:
				controls["ev2_power"] = fast_controls["ev2_power"]

	# BESS
	if household.bess:
		_, bess_deadline = _next_target(t, scenario.bess.soc_targets)
		if _is_deadline_urgent(t, bess_deadline):
			# Deadline urgent: use even_linear to guarantee target
			controls["bess_power"] = even_controls["bess_power"]
		else:
			# Not urgent: apply price logic
			hh_q25 = _q25_threshold(household.buy_price_day_profile)
			if hh_q25 is not None and household.buy_price <= hh_q25:
				controls["bess_power"] = fast_controls["bess_power"]

	return controls