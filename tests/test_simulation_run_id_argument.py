from src.simulation._sim_utils.arg_parse import build_parser
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import default_scenario


def test_run_id_argument_defaults_to_automatic_id():
    args = build_parser().parse_args([])

    assert args.run_id is None


def test_run_id_argument_can_override_run_context_id():
    args = build_parser().parse_args(["--run-id", "manual-batch-42"])
    context = RunContext(scenario=default_scenario, run_id=args.run_id)

    assert context.run_id == "manual-batch-42"
