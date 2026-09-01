"""Contract tests for typed VMC update inputs and the legacy adapter."""

from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError
from io import BytesIO
from typing import Any

import pytest
import torch

from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.physics.kinetic import KineticEnergy
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    ModelParameterBinding,
    ScoreUpdateInput,
    VMCStepData,
    VMCUpdateMethod,
    VMCUpdateResult,
)
from tests.unit.training.test_vmc_trainer_tpen_smoke import _StubContext


def _batch(*, n_electrons: int = 1, sample_shape: tuple[int, ...] = (1,)) -> ElectronBatch:
    """Build a small float64 batch with explicit spin labels."""

    positions = torch.zeros((*sample_shape, n_electrons, 1), dtype=torch.float64)
    spins = torch.ones((*sample_shape, n_electrons), dtype=torch.float64)
    return ElectronBatch(positions=positions, spins=spins)


def _output(batch: ElectronBatch, value: torch.Tensor | None = None) -> WavefunctionOutput:
    """Build an output using the readout's flattened sample convention."""

    shape = (batch.batch_size,)
    logabs = torch.zeros(shape, dtype=batch.dtype) if value is None else value.reshape(shape)
    return WavefunctionOutput(logabs=logabs, sign=torch.ones(shape, dtype=batch.dtype))


def _optimizer_update_input(parameter: torch.nn.Parameter, *, step: int) -> AutogradUpdateInput:
    """Build the same differentiable update input at a given model state."""

    objective = (parameter.square() * 3.0).sum()
    batch = _batch()
    return AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=step,
        objective=objective,
    )


def _binding(parameter: torch.nn.Parameter) -> ParameterBinding:
    """Build the direct one-slot binding used by score-input tests."""

    slot = ParameterSlot(
        ordinal=0,
        shape=tuple(parameter.shape),
        numel=parameter.numel(),
        dtype=parameter.dtype,
    )
    return ParameterBinding(
        layout=ParameterLayout(slots=(slot,)),
        parameters=(parameter,),
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Compare an optimizer state payload without relying on object identity."""

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_step_records_are_frozen_and_accept_flattened_sample_axes() -> None:
    """The common record follows flattened primal output, not raw sample rank."""

    batch = _batch(sample_shape=(2, 3))
    output = _output(batch)
    data = VMCStepData(
        batch=batch,
        wavefunction=output,
        local_energy=torch.zeros(6, dtype=torch.float64),
    )
    assert data.validate() is data

    with pytest.raises(FrozenInstanceError):
        data.batch = batch  # type: ignore[misc]

    objective = torch.tensor(0.0, dtype=torch.float64)
    autograd_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=data.local_energy,
        step=4,
        objective=objective,
    )
    assert autograd_input.step == 4

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = _binding(parameter)
    scores = MaterializedParameterLogScores(
        layout=binding.layout,
        blocks=(torch.zeros(6, 2, dtype=torch.float64),),
    )
    score_input = ScoreUpdateInput(
        batch=batch,
        wavefunction=output,
        local_energy=data.local_energy,
        step=4,
        parameter_scores=scores,
        parameter_binding=binding,
    )
    assert score_input.parameter_binding.parameters[0] is parameter


def test_step_records_reject_shape_and_layout_drift() -> None:
    batch = _batch(sample_shape=(2, 3))
    output = _output(batch)

    with pytest.raises(ValueError, match="same shape"):
        VMCStepData(
            batch=batch,
            wavefunction=output,
            local_energy=torch.zeros(2, 3, dtype=torch.float64),
        )

    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = _binding(parameter)
    mismatched_slot = ParameterSlot(
        ordinal=0,
        shape=(3,),
        numel=3,
        dtype=torch.float64,
    )
    mismatched_scores = MaterializedParameterLogScores(
        layout=ParameterLayout(slots=(mismatched_slot,)),
        blocks=(torch.zeros(6, 3, dtype=torch.float64),),
    )
    with pytest.raises(ValueError, match="layouts do not match"):
        ScoreUpdateInput(
            batch=batch,
            wavefunction=output,
            local_energy=torch.zeros(6, dtype=torch.float64),
            step=0,
            parameter_scores=mismatched_scores,
            parameter_binding=binding,
        )


def test_legacy_adapter_matches_current_zero_grad_backward_clip_step_sequence() -> None:
    """An independent current-style sequence pins the adapter's arithmetic."""

    control = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    adapted = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.Adam([control], lr=0.1)
    adapted_optimizer = torch.optim.Adam([adapted], lr=0.1)

    control_objective = (control.square() * 3.0).sum()
    control_optimizer.zero_grad(set_to_none=True)
    control_objective.backward()
    torch.nn.utils.clip_grad_norm_([control], 0.5)
    control_grad = control.grad.detach().clone()
    control_optimizer.step()
    control_state = control_optimizer.state_dict()

    adapted_objective = (adapted.square() * 3.0).sum()
    batch = _batch()
    update_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, adapted_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=adapted_objective,
    )
    result = LegacyAutogradUpdate(adapted_optimizer, gradient_clip_norm=0.5).update(update_input)

    assert result == VMCUpdateResult(applied=True, grad_norm=float(control_grad.norm().item()))
    assert torch.equal(adapted, control)
    assert adapted.grad is not None
    assert torch.equal(adapted.grad, control_grad)
    _assert_nested_equal(adapted_optimizer.state_dict(), control_state)


def test_legacy_adapter_matches_pre_f4_model_gradient_domain_for_subset_optimizer() -> None:
    """A subset optimizer must retain the predecessor model-wide gradient domain."""

    clipped_selected = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    clipped_unselected = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    clipped_optimizer = torch.optim.SGD([clipped_selected], lr=0.1)
    clipped_objective = clipped_selected + 2.0 * clipped_unselected
    clipped_batch = _batch()
    clipped_input = AutogradUpdateInput(
        batch=clipped_batch,
        wavefunction=_output(clipped_batch, clipped_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=clipped_objective,
    )
    clipped_result = LegacyAutogradUpdate(
        clipped_optimizer,
        gradient_clip_norm=0.5,
        model_parameters=ModelParameterBinding(
            parameters=(clipped_selected, clipped_unselected)
        ),
    ).update(clipped_input)

    # These values are the observed pre-F4 trainer result for this fixed
    # objective at the F3 predecessor b2c01970: clip_grad_norm_ operated on
    # both model parameters before the optimizer stepped only ``selected``.
    assert clipped_result.grad_norm == pytest.approx(0.5)
    assert clipped_selected.grad is not None
    assert clipped_unselected.grad is not None
    assert clipped_selected.grad.item() == pytest.approx(0.22360679774997896)
    assert clipped_unselected.grad.item() == pytest.approx(0.4472135954999579)
    assert clipped_selected.item() == pytest.approx(0.9776393202250021)
    assert clipped_unselected.item() == pytest.approx(2.0)

    unclipped_selected = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    unclipped_unselected = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    unclipped_optimizer = torch.optim.SGD([unclipped_selected], lr=0.1)
    unclipped_objective = unclipped_selected + 2.0 * unclipped_unselected
    unclipped_batch = _batch()
    unclipped_input = AutogradUpdateInput(
        batch=unclipped_batch,
        wavefunction=_output(unclipped_batch, unclipped_objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=unclipped_objective,
    )
    unclipped_result = LegacyAutogradUpdate(
        unclipped_optimizer,
        model_parameters=ModelParameterBinding(
            parameters=(unclipped_selected, unclipped_unselected)
        ),
    ).update(unclipped_input)

    # The unclipped norm is also model-wide in the predecessor; a
    # subset-only implementation reports 1.0 here instead of sqrt(5).
    assert unclipped_result.grad_norm == pytest.approx(2.23606797749979)


def test_legacy_adapter_loads_raw_checkpoint_and_preserves_next_update() -> None:
    """Loading an on-disk optimizer payload must affect the next Adam update."""

    control = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    control_optimizer = torch.optim.Adam([control], lr=0.1)
    control_adapter = LegacyAutogradUpdate(control_optimizer, gradient_clip_norm=0.5)
    control_adapter.update(_optimizer_update_input(control, step=0))

    # This is the raw payload written by the legacy checkpoint callback.
    raw_optimizer_checkpoint = copy.deepcopy(control_optimizer.state_dict())
    assert raw_optimizer_checkpoint["state"]
    first_state = next(iter(raw_optimizer_checkpoint["state"].values()))
    assert first_state["step"].item() == 1.0
    assert first_state["exp_avg"].abs().sum().item() > 0.0
    _assert_nested_equal(control_adapter.state_dict(), raw_optimizer_checkpoint)

    restored = torch.nn.Parameter(control.detach().clone())
    restored_optimizer = torch.optim.Adam([restored], lr=0.1)
    restored_adapter = LegacyAutogradUpdate(restored_optimizer, gradient_clip_norm=0.5)
    restored_adapter.load_state_dict(raw_optimizer_checkpoint)
    _assert_nested_equal(restored_adapter.state_dict(), raw_optimizer_checkpoint)

    control_result = control_adapter.update(_optimizer_update_input(control, step=1))
    restored_result = restored_adapter.update(_optimizer_update_input(restored, step=1))

    assert restored_result == control_result
    assert torch.equal(restored, control)
    assert restored.grad is not None
    assert control.grad is not None
    assert torch.equal(restored.grad, control.grad)
    _assert_nested_equal(restored_adapter.state_dict(), control_adapter.state_dict())


def test_legacy_adapter_skips_vacuum_and_errors_for_disconnected_nonvacuum() -> None:
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    adapter = LegacyAutogradUpdate(optimizer)

    vacuum = _batch(n_electrons=0)
    skipped = adapter.update(
        AutogradUpdateInput(
            batch=vacuum,
            wavefunction=_output(vacuum),
            local_energy=torch.zeros(1, dtype=torch.float64),
            step=0,
            objective=torch.tensor(1.0, dtype=torch.float64),
        )
    )
    assert skipped == VMCUpdateResult(applied=False, grad_norm=0.0)
    assert parameter.grad is None

    nonvacuum = _batch()
    with pytest.raises(RuntimeError, match="disconnected"):
        adapter.update(
            AutogradUpdateInput(
                batch=nonvacuum,
                wavefunction=_output(nonvacuum),
                local_energy=torch.zeros(1, dtype=torch.float64),
                step=1,
                objective=torch.tensor(1.0, dtype=torch.float64),
            )
        )
    assert parameter.grad is None


def test_live_update_inputs_cannot_be_serialized() -> None:
    batch = _batch()
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    objective = parameter.square().sum()
    update_input = AutogradUpdateInput(
        batch=batch,
        wavefunction=_output(batch, objective.reshape(1)),
        local_energy=torch.zeros(1, dtype=torch.float64),
        step=0,
        objective=objective,
    )

    with pytest.raises(RuntimeError, match="graph-bearing"):
        pickle.dumps(update_input)

    with pytest.raises(RuntimeError, match="graph-bearing"):
        torch.save(update_input, BytesIO())


class _RecordingUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """Test updater proving that the trainer delegates an exact input record."""

    def __init__(self) -> None:
        self.received: AutogradUpdateInput | None = None

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        self.received = update_input
        return VMCUpdateResult(applied=True, grad_norm=2.5)


class _OneElectronWalkers:
    def make_batch(self) -> ElectronBatch:
        return _batch()


class _OneElectronSampler:
    def collect_samples(self, model, *, device=None):
        del model, device
        return _OneElectronWalkers(), None


class _OneParameterWavefunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        value = self.weight.expand(batch.batch_size)
        return WavefunctionOutput(logabs=value, sign=torch.ones_like(value))


def test_trainer_delegates_without_publishing_live_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """The update record stays local while counters and completion events remain."""

    import tpen.training.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "local_energy",
        lambda terms, model, batch, return_terms: torch.zeros(batch.batch_size, dtype=batch.dtype),
    )
    update = _RecordingUpdate()
    trainer = VMCTrainer(max_steps=1, update_method=update)
    context = _StubContext()
    state = trainer.fit(
        model=_OneParameterWavefunction(),
        sampler=_OneElectronSampler(),
        hamiltonian_terms=[KineticEnergy()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))], lr=0.1),
        context=context,
        emit=lambda **_: None,
    )

    assert update.received is not None
    assert trainer.state_dict() == {"next_iteration": 1, "completed_updates": 1}
    assert all(value is not update.received for value in vars(state).values())
    assert state.wavefunction_output is update.received.wavefunction
