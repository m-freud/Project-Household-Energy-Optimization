from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.policies.linear import even_linear_policy, fast_charge_policy


def _q25_threshold(profile: list[float]) -> float | None:
	if not profile:
		return None
	sorted_profile = sorted(profile)
	idx = max(0, int(len(sorted_profile) * 0.25) - 1)
	return sorted_profile[idx]


def price_aware_linear_1(household: Household, scenario: Scenario) -> dict:
	"""
	Default behavior: even_linear.
	Override: if a device price is in the cheapest quartile (<= 25th percentile), use fast_charge for that device.
	"""
	even_controls = even_linear_policy(household, scenario)
	fast_controls = fast_charge_policy(household, scenario)

	# EV1: compare ev1.buy_price against ev1 day-profile quartile
	ev1_profile = getattr(household, "ev1_buy_price_day_profile", household.buy_price_day_profile)
	ev1_q25 = _q25_threshold(ev1_profile)
	if household.ev1 and ev1_q25 is not None and household.ev1.buy_price <= ev1_q25:
		even_controls["ev1_power"] = fast_controls["ev1_power"]

	# EV2: compare ev2.buy_price against ev2 day-profile quartile
	ev2_profile = getattr(household, "ev2_buy_price_day_profile", household.buy_price_day_profile)
	ev2_q25 = _q25_threshold(ev2_profile)
	if household.ev2 and ev2_q25 is not None and household.ev2.buy_price <= ev2_q25:
		even_controls["ev2_power"] = fast_controls["ev2_power"]

	# Keep BESS behavior price-aware against household buy price.
	hh_q25 = _q25_threshold(household.buy_price_day_profile)
	if hh_q25 is not None and household.buy_price <= hh_q25:
		even_controls["bess_power"] = fast_controls["bess_power"]

	return even_controls