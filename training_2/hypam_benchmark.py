
import argparse


class HypamBenchmark:
    def __init__(self, models, targets, scenarios, n_test_ids):
        self.models = models
        self.targets = targets
        self.scenarios = scenarios
        self.n_test_ids = n_test_ids

    def run_benchmark(self):
        for model in self.models:
            for target in self.targets:
                for scenario in self.scenarios:
                    print(f"Running benchmark for model: {model}, target: {target}, scenario: {scenario}")
                    # Here you would call the actual benchmarking function for the model, target, and scenario
                    # For example:
                    # benchmark_model(model, target, scenario, self.n_test_ids)
                    # This is a placeholder for demonstration purposes.
                    print(f"Completed benchmark for model: {model}, target: {target}, scenario: {scenario}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter benchmarking for selected models and targets.")
    parser.add_argument(
        "--models",
        default="ridge",
        help="Comma-separated model names (default: ridge). Options: ridge, xgb, rf.",
    )
    parser.add_argument(
        "--targets",
        default="base_load",
        help="Comma-separated target names (default: base_load)",
    )
    parser.add_argument(
        "--scenarios",
        default="default_scenario",
        help="Comma-separated scenario names (default: default_scenario)",
    )
    parser.add_argument(
        "--n_test_ids",
        type=int,
        default=18,
        help="Number of test households to use for benchmarking (default: 18)",
    )
    args = parser.parse_args()

    models = args.models.split(",")
    targets = args.targets.split(",")
    scenarios = args.scenarios.split(",")
    n_test_ids = args.n_test_ids

    benchmark = HypamBenchmark(models=models, targets=targets, scenarios=scenarios, n_test_ids=n_test_ids)

    benchmark.run_benchmark()
    