from pathlib import Path

import pytest

from src.simulation._sim_utils import controller_factories


class StubModel:
    def predict(self, features):
        return [0.0] * len(features)


@pytest.mark.parametrize(
    "builder",
    [
        controller_factories.build_multi_step_predictor,
        controller_factories.build_multi_model_predictor,
    ],
)
def test_multistep_predictor_builders_configure_load_and_pv_interpolation(
    monkeypatch,
    builder,
):
    model_directories = {
        target: Path(target)
        for target in ("base_load", "pv_gen", "ev1_status", "ev2_status")
    }
    stub_model = StubModel()

    def load_model(_path, _model_class):
        _ = (_path, _model_class)
        return stub_model

    def load_model_bank(_target, _model_class, _directories):
        _ = (_target, _model_class, _directories)
        return {1: stub_model}

    monkeypatch.setattr(controller_factories, "_load_xgb_model", load_model)
    monkeypatch.setattr(controller_factories, "_load_composite_model_bank", load_model_bank)

    linear_predictor = builder(model_directories)
    recursive_predictor = builder(model_directories, interpolation="recursive")

    assert linear_predictor.base_load_interpolation == "linear"
    assert linear_predictor.pv_gen_interpolation == "linear"
    assert recursive_predictor.base_load_interpolation == "recursive"
    assert recursive_predictor.pv_gen_interpolation == "recursive"


def test_controller_factories_expose_renamed_and_recursive_mpc_names(monkeypatch):
    stub_predictor = object()

    def build_stub_predictor(*args, **kwargs):
        _ = (args, kwargs)
        return stub_predictor

    for builder_name in (
        "build_single_step_predictor",
        "build_multi_step_predictor",
        "build_multi_model_predictor",
    ):
        monkeypatch.setattr(
            controller_factories,
            builder_name,
            build_stub_predictor,
        )

    requested_names = [
        "mpc_single_step",
        "mpc_multi_step",
        "mpc_multi_step_recursive",
        "mpc_multi_model",
        "mpc_multi_model_recursive",
    ]

    factories = controller_factories.build_controller_factories(requested_names)

    assert set(factories) == set(requested_names)
    with pytest.raises(ValueError, match="mpc_composite_horizons"):
        controller_factories.build_controller_factories(["mpc_composite_horizons"])
