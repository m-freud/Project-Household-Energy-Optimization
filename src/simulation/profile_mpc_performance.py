from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cvxpy as cp

from src.runtime_config import RuntimeConfig
from src.simulation._sim_utils import build_controller_factories
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import default_scenario
from src.simulation.simulation import Simulation
from src.sqlite_connection import create_source_connection


CONTROLLERS = (
    "mpc_single_step",
    "mpc_multi_step",
    "mpc_composite_horizons",
)
PREDICTOR_METHODS = {
    "predict_ev_status": "ev_status",
    "predict_base_load": "base_load",
    "predict_pv_gen": "pv_gen",
}
MODEL_ATTRIBUTES = {
    "base_load": ("base_load_model", "base_load_model_bank"),
    "pv_gen": ("pv_gen_model", "pv_gen_model_bank"),
    "ev1_status": ("ev1_status_model", "ev1_model", "ev1_model_bank"),
    "ev2_status": ("ev2_status_model", "ev2_model", "ev2_model_bank"),
}
TIMING_FIELDS = (
    "step_seconds",
    "input_update_seconds",
    "set_controls_seconds",
    "predictor_seconds",
    "predictor_non_model_seconds",
    "predictor_orchestration_seconds",
    "predict_ev_status_seconds",
    "predict_ev_status_non_model_seconds",
    "predict_base_load_seconds",
    "predict_base_load_non_model_seconds",
    "predict_pv_gen_seconds",
    "predict_pv_gen_non_model_seconds",
    "model_inference_seconds",
    "model_inference_calls",
    "solver_seconds",
    "solver_calls",
    "control_preparation_seconds",
    "apply_controls_seconds",
    "history_update_seconds",
)


def _model_entries(predictor: Any) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for label, attributes in MODEL_ATTRIBUTES.items():
        for attribute in attributes:
            model_or_bank = getattr(predictor, attribute, None)
            if isinstance(model_or_bank, dict):
                for horizon, model in model_or_bank.items():
                    if id(model) not in seen:
                        entries.append((f"{label}:h{horizon}", model))
                        seen.add(id(model))
            elif model_or_bank is not None and id(model_or_bank) not in seen:
                entries.append((label, model_or_bank))
                seen.add(id(model_or_bank))
    return entries


class _Profiler:
    def __init__(self, predictor: Any) -> None:
        self.metrics: dict[str, Any] = self._new_metrics()
        self._restore: list[tuple[Any, str, Any]] = []
        self._wrap_predictor(predictor)
        self._wrap_models(predictor)

    @staticmethod
    def _new_metrics() -> dict[str, Any]:
        return {
            "predictor_seconds": 0.0,
            "predictor_methods": defaultdict(float),
            "model_seconds": defaultdict(float),
            "model_calls": defaultdict(int),
            "solver_seconds": 0.0,
            "solver_calls": 0,
        }

    def reset(self) -> None:
        self.metrics = self._new_metrics()

    def _wrap(self, obj: Any, name: str, replacement: Callable[..., Any]) -> None:
        original = getattr(obj, name)
        setattr(obj, name, replacement)
        self._restore.append((obj, name, original))

    def _wrap_predictor(self, predictor: Any) -> None:
        original_predict = predictor.predict

        def timed_predict(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original_predict(*args, **kwargs)
            finally:
                self.metrics["predictor_seconds"] += time.perf_counter() - started

        self._wrap(predictor, "predict", timed_predict)

        for method_name, metric_name in PREDICTOR_METHODS.items():
            original_method = getattr(predictor, method_name, None)
            if original_method is None:
                continue

            def timed_method(
                *args: Any,
                _original: Callable[..., Any] = original_method,
                _metric_name: str = metric_name,
                **kwargs: Any,
            ) -> Any:
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.metrics["predictor_methods"][_metric_name] += (
                        time.perf_counter() - started
                    )

            self._wrap(predictor, method_name, timed_method)

    def _wrap_models(self, predictor: Any) -> None:
        for label, model in _model_entries(predictor):
            original_predict = model.predict

            def timed_model_predict(
                *args: Any,
                _original: Callable[..., Any] = original_predict,
                _label: str = label,
                **kwargs: Any,
            ) -> Any:
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.metrics["model_seconds"][_label] += (
                        time.perf_counter() - started
                    )
                    self.metrics["model_calls"][_label] += 1

            self._wrap(model, "predict", timed_model_predict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "predictor_seconds": self.metrics["predictor_seconds"],
            "predictor_methods": dict(self.metrics["predictor_methods"]),
            "model_seconds": dict(self.metrics["model_seconds"]),
            "model_calls": dict(self.metrics["model_calls"]),
            "solver_seconds": self.metrics["solver_seconds"],
            "solver_calls": self.metrics["solver_calls"],
        }

    def restore(self) -> None:
        for obj, name, original in reversed(self._restore):
            setattr(obj, name, original)
        self._restore.clear()


def _install_solver_timer(profiler: _Profiler) -> Callable[[], None]:
    original_solve = cp.Problem.solve

    def timed_solve(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_solve(problem, *args, **kwargs)
        finally:
            profiler.metrics["solver_seconds"] += time.perf_counter() - started
            profiler.metrics["solver_calls"] += 1

    cp.Problem.solve = timed_solve

    def restore() -> None:
        cp.Problem.solve = original_solve

    return restore


def _step(
    simulation: Simulation,
    controller: MPCController,
    household: Any,
    scenario: Any,
    timestep: int,
    profiler: _Profiler,
) -> dict[str, Any]:
    profiler.reset()
    step_started = time.perf_counter()

    simulation.current_timestep = timestep
    input_started = time.perf_counter()
    simulation.update_household_inputs(household)
    input_update_seconds = time.perf_counter() - input_started

    control_started = time.perf_counter()
    controls = controller.set_controls(household, scenario)
    set_controls_seconds = time.perf_counter() - control_started
    metrics = profiler.snapshot()

    apply_started = time.perf_counter()
    household.apply_controls(controls, duration_hours=simulation.duration_hours)
    apply_controls_seconds = time.perf_counter() - apply_started

    history_started = time.perf_counter()
    household.update_history()
    history_update_seconds = time.perf_counter() - history_started
    step_seconds = time.perf_counter() - step_started

    target_method_seconds = sum(metrics["predictor_methods"].values())
    model_inference_seconds = sum(metrics["model_seconds"].values())
    model_target_seconds = {
        "ev_status": sum(
            seconds
            for label, seconds in metrics["model_seconds"].items()
            if label.startswith(("ev1_status", "ev2_status"))
        ),
        "base_load": sum(
            seconds
            for label, seconds in metrics["model_seconds"].items()
            if label.startswith("base_load")
        ),
        "pv_gen": sum(
            seconds
            for label, seconds in metrics["model_seconds"].items()
            if label.startswith("pv_gen")
        ),
    }
    method_seconds = metrics["predictor_methods"]
    row: dict[str, Any] = {
        "timestep": timestep,
        "step_seconds": step_seconds,
        "input_update_seconds": input_update_seconds,
        "set_controls_seconds": set_controls_seconds,
        "predictor_seconds": metrics["predictor_seconds"],
        "predictor_non_model_seconds": max(
            0.0, metrics["predictor_seconds"] - model_inference_seconds
        ),
        "predictor_orchestration_seconds": max(
            0.0, metrics["predictor_seconds"] - target_method_seconds
        ),
        "predict_ev_status_seconds": method_seconds.get("ev_status", 0.0),
        "predict_ev_status_non_model_seconds": max(
            0.0, method_seconds.get("ev_status", 0.0) - model_target_seconds["ev_status"]
        ),
        "predict_base_load_seconds": method_seconds.get("base_load", 0.0),
        "predict_base_load_non_model_seconds": max(
            0.0, method_seconds.get("base_load", 0.0) - model_target_seconds["base_load"]
        ),
        "predict_pv_gen_seconds": method_seconds.get("pv_gen", 0.0),
        "predict_pv_gen_non_model_seconds": max(
            0.0, method_seconds.get("pv_gen", 0.0) - model_target_seconds["pv_gen"]
        ),
        "model_inference_seconds": model_inference_seconds,
        "model_inference_calls": sum(metrics["model_calls"].values()),
        "solver_seconds": metrics["solver_seconds"],
        "solver_calls": metrics["solver_calls"],
        "control_preparation_seconds": max(
            0.0,
            set_controls_seconds
            - metrics["predictor_seconds"]
            - metrics["solver_seconds"],
        ),
        "apply_controls_seconds": apply_controls_seconds,
        "history_update_seconds": history_update_seconds,
    }
    for label, seconds in metrics["model_seconds"].items():
        safe_label = label.replace(":", "_")
        row[f"model_{safe_label}_seconds"] = seconds
        row[f"model_{safe_label}_calls"] = metrics["model_calls"][label]
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for field in TIMING_FIELDS:
        values = [float(row[field]) for row in rows]
        summary[f"{field}_total"] = sum(values)
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_median"] = statistics.median(values)
        summary[f"{field}_max"] = max(values)
    return summary


def run_profile(household_ids: list[int], output_path: Path) -> None:
    factories = build_controller_factories(list(CONTROLLERS))
    detailed_rows: list[dict[str, Any]] = []
    restore_solver: Callable[[], None] | None = None

    source_conn = create_source_connection()
    results_conn = sqlite3.connect(":memory:")
    simulation = Simulation(
        source_conn,
        ensure_results_table=False,
        results_conn=results_conn,
    )
    try:
        for controller_name in CONTROLLERS:
            for household_id in household_ids:
                scenario = default_scenario
                context = RunContext(
                    controller_name=controller_name,
                    controller_factory=factories[controller_name],
                    scenario=scenario,
                    run_id="profile",
                )
                setup_started = time.perf_counter()
                household = simulation.create_household(household_id, context)
                controller = simulation.create_controller(household, context)
                if not isinstance(controller, MPCController):
                    raise TypeError(f"{controller_name} did not create an MPCController")
                profiler = _Profiler(controller.predictor)
                compile_started = time.perf_counter()
                controller._ensure_compiled_problem(controller.horizon)
                compile_seconds = time.perf_counter() - compile_started
                setup_seconds = time.perf_counter() - setup_started - compile_seconds
                restore_solver = _install_solver_timer(profiler)

                rows: list[dict[str, Any]] = []
                try:
                    for timestep in range(context.start_time, simulation.num_timesteps + 1):
                        row = _step(
                            simulation,
                            controller,
                            household,
                            scenario,
                            timestep,
                            profiler,
                        )
                        row.update(
                            controller=controller_name,
                            household_id=household_id,
                        )
                        rows.append(row)
                        detailed_rows.append(row)
                finally:
                    profiler.restore()
                    if restore_solver is not None:
                        restore_solver()
                        restore_solver = None

                summary = _summarize(rows)
                print(
                    f"{controller_name} household={household_id}: "
                    f"setup={setup_seconds:.3f}s compile={compile_seconds:.3f}s "
                    f"step_mean={summary['step_seconds_mean']:.4f}s "
                    f"predictor_mean={summary['predictor_seconds_mean']:.4f}s "
                    f"solver_mean={summary['solver_seconds_mean']:.4f}s "
                    f"model_calls={summary['model_inference_calls_total']:.0f}"
                )
                model_labels = sorted(
                    {
                        field.removeprefix("model_").removesuffix("_seconds")
                        for row in rows
                        for field in row
                        if field.startswith("model_")
                        and field.endswith("_seconds")
                        and field != "model_inference_seconds"
                    }
                )
                if model_labels:
                    model_details = ", ".join(
                        f"{label}={sum(float(row.get(f'model_{label}_seconds', 0.0)) for row in rows):.2f}s/"
                        f"{sum(int(row.get(f'model_{label}_calls', 0)) for row in rows)} calls"
                        for label in model_labels
                    )
                    print(f"  model inference: {model_details}")
    finally:
        if restore_solver is not None:
            restore_solver()
        results_conn.close()
        source_conn.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in detailed_rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"Detailed per-timestep measurements written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile MPC prediction and solver runtime by timestep."
    )
    parser.add_argument(
        "--households",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Household IDs to profile (default: 1 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RuntimeConfig.ROOT_DIR / "mpc_performance_profile.csv",
        help="Output CSV path (default: mpc_performance_profile.csv in repository root)",
    )
    args = parser.parse_args()
    run_profile(args.households, args.output)


if __name__ == "__main__":
    main()
