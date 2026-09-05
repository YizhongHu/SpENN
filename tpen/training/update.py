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
    ParameterLayout,
    ParameterSlot,
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
    """Bind the legacy gradient domain to direct model parameters.

    The static layout is retained separately from the live parameter
    references.  Checkpoint restore can therefore compare the recorded layout
    before rebuilding the binding against the model objects that are live
    after ``load_state_dict``.
    """

    parameters: tuple[torch.nn.Parameter, ...]
    layout: ParameterLayout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if any(not isinstance(parameter, torch.nn.Parameter) for parameter in self.parameters):
            raise TypeError("ModelParameterBinding.parameters must contain direct parameters")
        if self.layout is None:
            object.__setattr__(self, "layout", _parameter_layout(self.parameters))
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("ModelParameterBinding.layout must be a ParameterLayout")
        self.validate()

    def validate(self) -> "ModelParameterBinding":
        """Validate that the live references have the recorded layout."""

        assert self.layout is not None
        self.layout.validate()
        if len(self.parameters) != len(self.layout.slots):
            raise ValueError(
                "ModelParameterBinding.parameters must have one reference per layout slot"
            )
        for slot, parameter in zip(self.layout.slots, self.parameters, strict=True):
            if tuple(parameter.shape) != slot.shape:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected shape {slot.shape}, "
                    f"got {tuple(parameter.shape)}"
                )
            if parameter.numel() != slot.numel:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected numel {slot.numel}, "
                    f"got {parameter.numel()}"
                )
            if parameter.dtype != slot.dtype:
                raise ValueError(
                    f"ModelParameterBinding slot {slot.ordinal} expected dtype {slot.dtype}, "
                    f"got {parameter.dtype}"
                )
        return self

    def compare(
        self,
        other: "ModelParameterBinding",
    ) -> tuple[bool, dict[str, float]]:
        """Compare layout metadata and direct parameter-reference identity."""

        if type(self) is not type(other) or not self.layout.compare(other.layout)[0]:
            return False, {"max_abs_error": float("inf")}
        close = len(self.parameters) == len(other.parameters) and all(
            left is right for left, right in zip(self.parameters, other.parameters, strict=True)
        )
        return close, {"max_abs_error": 0.0 if close else float("inf")}

    @classmethod
    def from_parameters(
        cls,
        parameters: tuple[torch.nn.Parameter, ...],
    ) -> "ModelParameterBinding":
        """Build a binding whose layout is derived from live parameters."""

        return cls(parameters=tuple(parameters))

    def rebind(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        *,
        layout: ParameterLayout | None = None,
    ) -> "ModelParameterBinding":
        """Rebuild direct references after checking the expected layout.

        Parameters
        ----------
        parameters : tuple of torch.nn.Parameter
            The current model-owned parameter objects.
        layout : ParameterLayout, optional
            Recorded layout to enforce.  Defaults to this binding's layout.
        """

        parameters = tuple(parameters)
        expected = self.layout if layout is None else layout
        current = _parameter_layout(parameters)
        if not expected.compare(current)[0]:
            layout_mismatch_message = "checkpoint parameter layout does not match live model"
            raise ValueError(layout_mismatch_message)
        return type(self)(layout=current, parameters=parameters)


@dataclass(frozen=True, kw_only=True)
class VMCUpdateState:
    """The single optimizer and parameter binding owned by an update method."""

    optimizer: torch.optim.Optimizer
    model_parameters: ModelParameterBinding

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError("VMCUpdateState.optimizer must be a torch.optim.Optimizer")
        if not isinstance(self.model_parameters, ModelParameterBinding):
            raise TypeError("VMCUpdateState.model_parameters must be a ModelParameterBinding")


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

    def update_state(self) -> VMCUpdateState | None:
        """Return owned optimizer state, or ``None`` for a stateless method.

        Returning ``None`` delegates the authority to the optimizer supplied
        to ``VMCTrainer.fit``.  A stateful method must return its one typed
        authority so the trainer can reject an ambiguous legacy optimizer
        before restore or update work begins.
        """

        return None

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Accept a rebuilt direct parameter binding after checkpoint restore.

        Stateless methods do not retain a binding.  A stateful method that
        keeps direct parameter references must override this hook so the
        references used by its next update are the restored model's live
        objects.
        """

        del model_parameters

    def forward_request(self) -> Any | None:
        """Return the typed forward request this method's input requires.

        Returns
        -------
        WavefunctionForwardRequest or None
            ``None``, the default, means an ordinary value forward
            (``model(batch)``) supplies everything the method needs.  A method
            consuming derivative payloads returns the exact request describing
            them, for example a
            :class:`~tpen.nn.forward.MaterializedParameterScoreRequest`.

        Notes
        -----
        This exists so the trainer can produce the score-bearing forward packet
        **once**, in the single forward it already performs, rather than
        running an ordinary forward and then recomputing derivatives.  A design
        that did the latter would double the forward and derivative work of
        every step.

        The return type is deliberately the request object itself rather than a
        capability flag or a name.  The trainer dispatches on the request's own
        type, so adding a future derivative payload cannot be done by inventing
        a string, and a method cannot claim a capability whose payload it does
        not then receive.
        """

        return None

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

    def update_state(self) -> VMCUpdateState:
        """Return the optimizer and direct gradient binding owned by the adapter."""

        return VMCUpdateState(
            optimizer=self.optimizer,
            model_parameters=self.model_parameters,
        )

    def rebind_model_parameters(self, model_parameters: ModelParameterBinding) -> None:
        """Replace the legacy gradient domain with restored model references."""

        if not isinstance(model_parameters, ModelParameterBinding):
            raise TypeError("LegacyAutogradUpdate model_parameters must be a ModelParameterBinding")
        self.model_parameters = model_parameters

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


def _parameter_layout(
    parameters: tuple[torch.nn.Parameter, ...],
) -> ParameterLayout:
    """Derive static layout metadata from direct live parameters."""

    return ParameterLayout(
        slots=tuple(
            ParameterSlot(
                ordinal=ordinal,
                shape=tuple(parameter.shape),
                numel=parameter.numel(),
                dtype=parameter.dtype,
            )
            for ordinal, parameter in enumerate(parameters)
        )
    )


def serialize_parameter_layout(layout: ParameterLayout) -> dict[str, Any]:
    """Return JSON-safe immutable metadata for a parameter layout."""

    if not isinstance(layout, ParameterLayout):
        raise TypeError("parameter layout must be a ParameterLayout")
    layout.validate()
    return {
        "slots": [
            {
                "ordinal": slot.ordinal,
                "shape": list(slot.shape),
                "numel": slot.numel,
                "dtype": str(slot.dtype),
            }
            for slot in layout.slots
        ]
    }


def deserialize_parameter_layout(state: Mapping[str, Any]) -> ParameterLayout:
    """Parse strict JSON-safe parameter-layout metadata."""

    if not isinstance(state, Mapping):
        raise TypeError("parameter layout state must be a mapping")
    slots = state.get("slots")
    if not isinstance(slots, list):
        raise ValueError("parameter layout state must contain a slots list")
    parsed: list[ParameterSlot] = []
    for raw in slots:
        if not isinstance(raw, Mapping):
            raise TypeError("parameter layout slots must be mappings")
        dtype_name = raw.get("dtype")
        if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
            raise ValueError("parameter layout slot dtype must be a torch dtype name")
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("parameter layout slot dtype must be a real floating torch.dtype")
        shape = raw.get("shape")
        if not isinstance(shape, list):
            raise TypeError("parameter layout slot shape must be a list")
        parsed.append(
            ParameterSlot(
                ordinal=raw.get("ordinal"),
                shape=tuple(shape),
                numel=raw.get("numel"),
                dtype=dtype,
            )
        )
    return ParameterLayout(slots=tuple(parsed))


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
    "VMCUpdateState",
    "deserialize_parameter_layout",
    "serialize_parameter_layout",
]
