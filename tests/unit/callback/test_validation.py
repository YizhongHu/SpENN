"""Tests for the train-end Validation callback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spenn.callback import Event, Validation
from spenn.diagnostics import EnergyEvaluation
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_sampler,
    build_tiny_spenn,
)
from tests.unit.callback.support import FakeState, RecordingContext


class ValidationContext(RecordingContext):
    """RecordingContext plus the runtime metadata the callback reads."""

    def __init__(self) -> None:
        super().__init__()
        self.metadata = SimpleNamespace(device=None)


def _make_callback(**overrides) -> Validation:
    kwargs = {
        "sampler": build_tiny_sampler(),
        "hamiltonian_terms": build_tiny_hamiltonian_terms(),
        "diagnostics": [EnergyEvaluation(name="energy")],
    }
    kwargs.update(overrides)
    return Validation(["train_end"], **kwargs)


def _train_end_event(context, model, *, state=None, step: int = 2) -> Event:
    return Event(name="train_end", context=context, state=state, payload={"model": model, "step": step})


def test_validation_responds_to_train_end_and_logs_validation_metrics() -> None:
    model = build_tiny_spenn()
    callback = _make_callback()
    context = ValidationContext()

    callback.handle(_train_end_event(context, model))

    metrics = context.latest("validation")
    assert "energy" in metrics
    assert "energy_variance" in metrics
    assert "energy_stderr" in metrics
    assert "local_energy_finite_fraction" in metrics
    assert context.by_namespace("validation")[-1]["step"] == 2


def test_validation_logs_sampler_metadata_and_geometry() -> None:
    model = build_tiny_spenn()
    callback = _make_callback()
    context = ValidationContext()

    callback.handle(_train_end_event(context, model))

    sampler_metrics = context.latest("validation/sampler")
    for key in (
        "acceptance_rate",
        "n_walkers",
        "burn_in",
        "n_steps",
        "proposal_scale",
        "seed",
        "radius_mean",
        "radius_q99",
        "electron_distance_q01",
    ):
        assert key in sampler_metrics, f"missing {key}"
    perf = context.latest("validation/perf")
    assert perf["wall_time_sec"] > 0.0


def test_validation_never_logs_exact_reference_metrics() -> None:
    model = build_tiny_spenn()
    callback = _make_callback()
    context = ValidationContext()

    callback.handle(_train_end_event(context, model))

    for record in context.records:
        assert record["namespace"].startswith("validation")
        assert "energy_error" not in record["metrics"]
        assert "energy_abs_error" not in record["metrics"]
        assert "reference_energy" not in record["metrics"]


def test_validation_rejects_reference_energy_diagnostics() -> None:
    with pytest.raises(ValueError, match="reference_energy"):
        _make_callback(diagnostics=[EnergyEvaluation(name="energy", reference_energy=2.0)])


def test_validation_fails_loudly_on_forbidden_metric_keys() -> None:
    class LeakyDiagnostic:
        name = "leaky"

        def evaluate(self, context):
            return {"energy": 1.0, "energy_error": 0.1}

    model = build_tiny_spenn()
    callback = _make_callback(diagnostics=[LeakyDiagnostic()])
    context = ValidationContext()

    with pytest.raises(ValueError, match="energy_error"):
        callback.handle(_train_end_event(context, model))


def test_validation_requires_diagnostics() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        _make_callback(diagnostics=None)


def test_validation_rejects_reserved_namespaces() -> None:
    for namespace in ("train", "eval", "checks", "eval/holdout"):
        with pytest.raises(ValueError, match="reserved"):
            _make_callback(namespace=namespace)


def test_validation_rejects_the_training_sampler_object() -> None:
    model = build_tiny_spenn()
    shared_sampler = build_tiny_sampler()
    callback = _make_callback(sampler=shared_sampler)
    state = FakeState(step=2, model=model, sampler=shared_sampler)
    context = ValidationContext()

    with pytest.raises(ValueError, match="independent"):
        callback.handle(_train_end_event(context, model, state=state))


def test_validation_uses_a_sampler_distinct_from_training(

) -> None:
    model = build_tiny_spenn()
    training_sampler = build_tiny_sampler()
    callback = _make_callback()
    assert callback.sampler is not training_sampler
    state = FakeState(step=2, model=model, sampler=training_sampler)
    context = ValidationContext()

    callback.handle(_train_end_event(context, model, state=state))

    # The training chain was never touched by validation.
    assert training_sampler.walkers is None
    assert callback.sampler.walkers is not None


def test_validation_leaves_model_parameters_unchanged() -> None:
    model = build_tiny_spenn()
    model.train()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    callback = _make_callback()
    context = ValidationContext()

    callback.handle(_train_end_event(context, model))

    after = dict(model.named_parameters())
    assert before.keys() == after.keys()
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name].detach()), name
    # Training mode is restored after the eval-mode validation pass.
    assert model.training


def test_validation_requires_a_model() -> None:
    callback = _make_callback()
    context = ValidationContext()
    event = Event(name="train_end", context=context, state=None, payload={"step": 2})

    with pytest.raises(ValueError, match="model"):
        callback.handle(event)


def test_train_end_bypasses_every_n_steps_cadence() -> None:
    callback = _make_callback(every_n_steps=1000)
    context = ValidationContext()
    event = _train_end_event(context, model=object(), step=7)

    assert callback.should_run(event)


def test_custom_namespace_is_used_for_all_records() -> None:
    model = build_tiny_spenn()
    callback = _make_callback(namespace="validation/holdout")
    context = ValidationContext()

    callback.handle(_train_end_event(context, model))

    namespaces = {record["namespace"] for record in context.records}
    assert namespaces == {
        "validation/holdout",
        "validation/holdout/sampler",
        "validation/holdout/perf",
    }
