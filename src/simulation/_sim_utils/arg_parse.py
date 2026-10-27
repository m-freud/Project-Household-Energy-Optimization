

import argparse
import re


def _parse_household_selection(value: str) -> str | list[int]:
    if value in {"all", "test_set"}:
        return value

    household_ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            start, end = map(int, range_match.groups())
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Invalid household range '{token}': start must not exceed end"
                )
            household_ids.extend(range(start, end + 1))
            continue

        try:
            household_ids.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid household selection '{token}'; use IDs or inclusive ranges "
                "such as '5-25'"
            ) from exc

    return household_ids

    
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HEMS simulation")
    parser.add_argument(
        "--controllers",
        default="mpc_oracle,mpc_history_avg,mpc_single_step",
        help="Comma-separated controller names (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        default="default_scenario",
        help="Comma-separated scenario names (default: all)",
    )
    parser.add_argument(
        "--households",
        type=_parse_household_selection,
        default="1",
        help="Household IDs/ranges (e.g. 5-25 or 1,3,5-8), 'all', or 'test_set' (default: 1)",
    )
    parser.add_argument(
        "--max-households",
        type=int,
        default=None,
        help="Limit number of households after filtering",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        default=1,
        help="Simulation start timestep (1..96)",
    )
    parser.add_argument(
        "--parallel-households",
        action="store_true",
        default=True,
        help="Run households in parallel using process workers",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=6,
        help="Number of worker processes for --parallel-households (default: Python decides)",
    )
    return parser

