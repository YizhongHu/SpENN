"""Typed VMC update inputs and the behavior-preserving legacy adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from tpen.data.batch import (
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterBinding,
    WavefunctionOutput,
)
from tpen.dependencies import require_torch

torch = require_torch(feature="VMC update methods")


InputT = TypeVar("InputT")
ScopeFactory = Callable[[int], AbstractContextManager[Any]]


@dataclass(frozen=True, kw_only=True)
class VMCStepData:
    """Common live data produced for one VMC iteration.

    Parameters
    ----------
    batch : ElectronBatch
        Electron configurations used to produce the wavefunction and energy.
    wavefunction : WavefunctionOutput
        The model output for ``batch``.  Its leading shape may be the batch's
        flattened size because the TPEN readout flattens multidimensional
        sample axes.
    local_energy : torch.Tensor
        Per-sample total local energies with the same shape, dtype, and device
        as ``wavefunction.logabs``.

    Notes
    -----
    This record is deliberately live for exactly one update call.  It is never
    assigned to :class:`tpen.training.state.TrainerState`; its serialization
    guard also prevents a graph-bearing record from crossing an artifact
    boundary by accident.
    """

    batch: ElectronBatch
    wavefunction: WavefunctionOutput
    local_energy: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "VMCStepData":
        """Validate the common batch, output, and local-energy contract."""

        if not isinstance(self.batch, ElectronBatch):
            raise TypeError("VMCStepData.batch must be an ElectronBatch")
        if not isinstance(self.wavefunction, WavefunctionOutput):
            raise TypeError("VMCStepData.wavefunction must be a WavefunctionOutput")
        if not isinstance(self.local_energy, torch.Tensor):
            raise TypeError("VMCStepData.local_energy must be a torch.Tensor")
        self.batch.validate()
        # ``batch_size`` rather than ``sample_shape`` is intentional: the
        # Pfaffian readout returns a flat primal output for a multidimensional
        # sample shape, while local energy follows that output shape.
        self.wavefunction.validate(batch_size=self.batch.batch_size)
        if self.local_energy.shape != self.wavefunction.logabs.shape:
            raise ValueError(
                "VMCStepData.local_energy must have the same shape as wavefunction.logabs, "
                f"got {tuple(self.local_energy.shape)} and {tuple(self.wavefunction.logabs.shape)}"
            )
        if not self.local_energy.is_floating_point():
            raise TypeError("VMCStepData.local_energy must have a real floating dtype")
        if self.local_energy.device != self.wavefunction.logabs.device:
            raise ValueError("VMCStepData local_energy and wavefunction must share one device")
        if self.local_energy.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("VMCStepData local_energy and wavefunction must share one dtype")
        if self.batch.device != self.wavefunction.logabs.device:
            raise ValueError("VMCStepData batch and wavefunction must share one device")
        if self.batch.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("VMCStepData batch and wavefunction must share one dtype")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization while any common step value is graph-live."""

        if _batch_requires_grad(self.batch) or _output_requires_grad(self.wavefunction):
            raise RuntimeError("graph-bearing VMCStepData cannot be serialized")
        if self.local_energy.requires_grad:
            raise RuntimeError("graph-bearing VMCStepData cannot be serialized")
        return self.__dict__


@dataclass(frozen=True, kw_only=True)
class AutogradUpdateInput(VMCStepData):
    """Typed input for an autograd-backed VMC update."""

    step: int
    objective: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "AutogradUpdateInput":
        """Validate the common step data and differentiable scalar objective."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.objective, torch.Tensor):
            raise TypeError("AutogradUpdateInput.objective must be a torch.Tensor")
        if self.objective.ndim != 0:
            raise ValueError(
                f"AutogradUpdateInput.objective must be scalar, got shape {tuple(self.objective.shape)}"
            )
        if not self.objective.is_floating_point():
            raise TypeError("AutogradUpdateInput.objective must have a real floating dtype")
        if self.objective.device != self.wavefunction.logabs.device:
            raise ValueError("AutogradUpdateInput objective and wavefunction must share one device")
        if self.objective.dtype != self.wavefunction.logabs.dtype:
            raise ValueError("AutogradUpdateInput objective and wavefunction must share one dtype")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization while an objective or common value is live."""

        super().__getstate__()
        if self.objective.requires_grad:
            raise RuntimeError("graph-bearing AutogradUpdateInput cannot be serialized")
        return self.__dict__


@dataclass(frozen=True, kw_only=True)
class ScoreUpdateInput(VMCStepData):
    """Typed input reserved for score-based VMC update methods.

    The first score consumer is intentionally not implemented in this slice.
    Keeping its exact input record here prevents a future SR implementation from
    inventing an optional capability bag or a string-keyed parameter lookup.
    """

    step: int
    parameter_scores: MaterializedParameterLogScores
    parameter_binding: ParameterBinding

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ScoreUpdateInput":
        """Validate score blocks against the direct model parameter binding."""

        super().validate()
        _validate_step(self.step)
        if not isinstance(self.parameter_scores, MaterializedParameterLogScores):
            raise TypeError(
                "ScoreUpdateInput.parameter_scores must be MaterializedParameterLogScores"
            )
        if not isinstance(self.parameter_binding, ParameterBinding):
            raise TypeError("ScoreUpdateInput.parameter_binding must be a ParameterBinding")
        self.parameter_binding.validate()
        self.parameter_scores.validate(sample_shape=tuple(self.wavefunction.logabs.shape))
        if not self.parameter_scores.layout.compare(self.parameter_binding.layout)[0]:
            raise ValueError("ScoreUpdateInput parameter scores and binding layouts do not match")
        if self.parameter_binding.parameters:
            binding_device = self.parameter_binding.parameters[0].device
            if self.parameter_scores.device != binding_device:
                raise ValueError("ScoreUpdateInput parameter scores and binding must share one device")
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Reject serialization of a record holding direct live parameters."""

        super().__getstate__()
        raise RuntimeError("live ScoreUpdateInput cannot be serialized")


@dataclass(frozen=True, kw_only=True)
class VMCUpdateResult:
    """Result of one update-method invocation."""

    applied: bool
    grad_norm: float

    def __post_init__(self) -> None:
        if type(self.applied) is not bool:
            raise TypeError("VMCUpdateResult.applied must be a bool")
        object.__setattr__(self, "grad_norm", float(self.grad_norm))


@dataclass(frozen=True, kw_only=True)
class ModelParameterBinding:
    """Bind the legacy gradient domain to direct model parameters."""

    parameters: tuple[torch.nn.Parameter, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if any(not isinstance(parameter, torch.nn.Parameter) for parameter in self.parameters):
            raise TypeError("ModelParameterBinding.parameters must contain direct parameters")


class VMCUpdateMethod(Generic[InputT], ABC):
    """Nominal typed contract for VMC update strategies.

    Concrete methods own their optimizer/preconditioner state.  The default
    state surface is empty, which lets stateless future methods satisfy the
    contract without inventing checkpoint payloads.  ``set_step_scopes`` is a
    narrow trainer integration hook: it preserves typed Backward and
    OptimizerUpdate event boundaries without adding a context or callback bag
    to the live update input.
    """

    @abstractmethod
    def update(self, update_input: InputT) -> VMCUpdateResult:
        """Apply one update and report whether it returned an applied step."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return this method's checkpointable state, empty by default."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore this method's state, empty by default."""

        if not isinstance(state, Mapping):
            raise TypeError("VMCUpdateMethod state must be a mapping")
        if state:
            raise ValueError("stateless VMCUpdateMethod cannot load non-empty state")

    def set_step_scopes(
        self,
        *,
        backward_scope: ScopeFactory | None = None,
        optimizer_scope: ScopeFactory | None = None,
    ) -> None:
        """Install optional typed event scopes for one trainer invocation."""

        del backward_scope, optimizer_scope


class LegacyAutogradUpdate(VMCUpdateMethod[AutogradUpdateInput]):
    """Exact adapter for TPEN's historical zero-grad/backward/step sequence."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_norm: float | None = None,
        *,
        model_parameters: ModelParameterBinding,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("LegacyAutogradUpdate.optimizer must be a torch.optim.Optimizer")
        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError(
                "LegacyAutogradUpdate.model_parameters must be a ModelParameterBinding"
            )
        self.optimizer = optimizer
        self.gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)
        self.model_parameters = model_parameters
        self._backward_scope: ScopeFactory | None = None
        self._optimizer_scope: ScopeFactory | None = None

    def set_step_scopes(
        self,
        *,
        backward_scope: ScopeFactory | None = None,
        optimizer_scope: ScopeFactory | None = None,
    ) -> None:
        """Preserve the trainer's typed phase boundaries around legacy work."""

        self._backward_scope = backward_scope
        self._optimizer_scope = optimizer_scope

    def update(self, update_input: AutogradUpdateInput) -> VMCUpdateResult:
        """Run the historical update sequence without clearing post-step grads."""

        if not isinstance(update_input, AutogradUpdateInput):
            raise TypeError("LegacyAutogradUpdate requires AutogradUpdateInput")

        # This ordering is observable: GradientStats reads the gradients after
        # optimizer.step(), so there is deliberately no post-step zero_grad.
        self.optimizer.zero_grad(set_to_none=True)
        objective = update_input.objective
        if not objective.requires_grad:
            if update_input.batch.n_electrons == 0:
                return VMCUpdateResult(applied=False, grad_norm=0.0)
            raise RuntimeError(
                "VMC loss is disconnected from model parameters for a "
                "nonzero-electron batch"
            )

        self._run_backward(update_input)
        gradient_parameters = self.gradient_params()
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                gradient_parameters, self.gradient_clip_norm
            )
        grad_norm = _gradient_norm(gradient_parameters)
        self._run_optimizer_step(update_input)
        return VMCUpdateResult(applied=True, grad_norm=grad_norm)

    def optimizer_params(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the optimizer's direct parameter references in group order."""

        parameters: list[torch.nn.Parameter] = []
        for group in self.optimizer.param_groups:
            parameters.extend(group["params"])
        return tuple(parameters)

    def gradient_params(self) -> tuple[torch.nn.Parameter, ...]:
        """Return the exact model parameter domain used by legacy gradients."""

        return self.model_parameters.parameters

    def state_dict(self) -> Mapping[str, Any]:
        """Return the raw PyTorch optimizer payload unchanged."""

        return self.optimizer.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load a raw PyTorch optimizer payload for checkpoint compatibility."""

        if not isinstance(state, Mapping):
            raise TypeError("LegacyAutogradUpdate state must be a mapping")
        self.optimizer.load_state_dict(state)

    def _run_backward(self, update_input: AutogradUpdateInput) -> None:
        if self._backward_scope is None:
            update_input.objective.backward()
            return
        with self._backward_scope(update_input.step):
            update_input.objective.backward()

    def _run_optimizer_step(self, update_input: AutogradUpdateInput) -> None:
        if self._optimizer_scope is None:
            self.optimizer.step()
            return
        with self._optimizer_scope(update_input.step):
            self.optimizer.step()


def _validate_step(step: int) -> None:
    if type(step) is not int or step < 0:
        raise ValueError("VMC update step must be a non-negative integer")


def _batch_requires_grad(batch: ElectronBatch) -> bool:
    tensors = (batch.positions, batch.nuclear_positions, batch.nuclear_charges, batch.spins)
    return any(tensor is not None and tensor.requires_grad for tensor in tensors)


def _output_requires_grad(output: WavefunctionOutput) -> bool:
    tensors = (output.logabs, output.sign, output.phase, *output.aux.values())
    return any(isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors)


def _gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


__all__ = [
    "AutogradUpdateInput",
    "LegacyAutogradUpdate",
    "ModelParameterBinding",
    "ScoreUpdateInput",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
]
