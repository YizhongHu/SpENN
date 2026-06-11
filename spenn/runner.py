"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Event
from spenn.diagnostics import Diagnostic, EvaluationContext, JsonScalar
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy, normalize_hamiltonian_terms, summarize_local_energy
from spenn.training.optim import make_optimizer


class Runner:
    """Base runner with callback lifecycle dispatch.

    Callbacks and loggers are owned by the `RunContext` (configured at the
    config root); ``emit`` dispatches lifecycle events into ``context.callbacks``
    and runners log through ``context.log``.
    """

    def emit(
        self,
        name: str,
        context: RunContext,
        *,
        state: object | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one lifecycle event to the context's callbacks."""

        event = Event(name=name, context=context, state=state, payload={} if payload is None else payload)
        for callback in context.callbacks:
            callback.handle(event)

    def run(self, context: RunContext) -> RunResult:
        """Execute a configured run."""

        raise NotImplementedError


class Train(Runner):
    """Config-driven VMC training runner.

    Builds the optimizer, drives the configured trainer through the VMC loop,
    and emits lifecycle events. Callbacks and loggers are owned by the
    `RunContext`; the runner adds no exception handling (``run_from_config``
    owns that) and only emits events while the trainer logs through the context.

    Parameters
    ----------
    model : torch.nn.Module
        Wavefunction model to optimize.
    sampler : object
        Sampler exposing ``collect_samples(model, device=...) -> (walkers, stats)``.
    hamiltonian_terms : sequence or mapping
        Hamiltonian terms summed by `local_energy`. A
        ``dict[str, HamiltonianTerm]`` uses its non-empty string keys as the
        public term names for decomposition and metrics; a sequence derives
        unique names from term class names.
    optimizer : Any
        Configured optimizer spec/factory (typically a ``_partial_`` optimizer
        constructor) applied to ``model.parameters()`` by `make_optimizer`.
    trainer : object
        Trainer exposing ``fit(*, model, sampler, hamiltonian_terms, optimizer,
        context, emit) -> TrainerState``.
    """

    def __init__(self, model, sampler, hamiltonian_terms, optimizer, trainer) -> None:
        self.model = model
        self.sampler = sampler
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
        self.optimizer = optimizer
        self.trainer = trainer

    def run(self, context: RunContext) -> RunResult:
        """Build the optimizer and run the configured VMC training loop."""

        self.emit("run_start", context)
        if isinstance(self.model, torch.nn.Module):
            _place_module_for_runtime(self.model, context)
            self.model.train()

        optimizer = make_optimizer(self.optimizer, self.model.parameters())
        self.emit("model_built", context, payload={"model": self.model, "optimizer": optimizer})

        self.emit("train_start", context)
        final_state = self.trainer.fit(
            model=self.model,
            sampler=self.sampler,
            hamiltonian_terms=self.hamiltonian_terms,
            optimizer=optimizer,
            context=context,
            emit=lambda name, *, state=None, payload=None: self.emit(name, context, state=state, payload=payload),
        )
        self.emit("train_end", context, state=final_state)
        self.emit("run_end", context)
        return RunResult(status="completed")


class Evaluate(Runner):
    """Generic sampled diagnostic evaluation runner.

    `Evaluate` owns sampling and shared evaluation work for one configured
    model/system. Diagnostics consume an `EvaluationContext`; they do not
    resample, log, emit callbacks, or write artifacts.

    Sampler contract assumed by this runner::

        walkers, sampler_stats = sampler.collect_samples(model, device=runtime_device)
        batch = walkers.make_batch()

    Parameters
    ----------
    model : callable
        Wavefunction model returning ``WavefunctionOutput``.
    sampler : object
        Sampler exposing ``collect_samples(model, device=...) -> (walkers, stats)``.
    hamiltonian_terms : sequence or mapping
        Hamiltonian terms summed by `local_energy`. A
        ``dict[str, HamiltonianTerm]`` uses its non-empty string keys as the
        public term names for decomposition and metrics; a sequence derives
        unique names from term class names.
    diagnostics : sequence of Diagnostic, optional
        Diagnostics to evaluate from the shared `EvaluationContext`. An empty
        sequence preserves the minimal pre-diagnostics local-energy summary.
    return_terms : bool, optional
        Whether to request per-term local-energy components from `local_energy`.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        diagnostics: Sequence[object] | None = None,
        return_terms: bool = False,
    ) -> None:
        self.model = model
        self.sampler = sampler
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
        self.diagnostics = _validate_diagnostics(diagnostics)
        self.return_terms = bool(return_terms)

    def run(self, context: RunContext) -> RunResult:
        """Sample configurations, evaluate local energy, and log metrics."""

        self.emit("run_start", context)
        self.emit("evaluate_start", context)

        if isinstance(self.model, torch.nn.Module):
            _place_module_for_runtime(self.model, context)
            self.model.eval()
            _assert_eager_initialized(self.model)

        # No torch.no_grad: local-energy evaluation needs position derivatives.
        walkers, sampler_stats = self.sampler.collect_samples(
            self.model,
            device=context.metadata.device,
        )
        batch = walkers.make_batch()

        self.emit("samples_collected", context, payload={"sampler_stats": dict(sampler_stats)})

        normalized_terms = normalize_hamiltonian_terms(self.hamiltonian_terms)
        energy_result = local_energy(normalized_terms, self.model, batch, return_terms=self.return_terms)
        total_energy, term_energies = _split_local_energy_result(energy_result)

        metrics: dict[str, JsonScalar]
        if self.diagnostics:
            with torch.no_grad():
                wavefunction_output = self.model(batch)
            evaluation = EvaluationContext(
                model=self.model,
                batch=batch,
                wavefunction_output=wavefunction_output,
                local_energy=total_energy,
                local_energy_terms=term_energies,
                sampler_stats=dict(sampler_stats),
                hamiltonian_terms=normalized_terms,
            )
            metrics = _evaluate_diagnostics(self.diagnostics, evaluation)
        else:
            metrics = summarize_local_energy(energy_result)

        context.log(metrics, step=0, namespace="eval")
        if sampler_stats:
            context.log(dict(sampler_stats), step=0, namespace="eval/sampler")

        self.emit("evaluate_end", context, payload={"metrics": metrics})
        self.emit("run_end", context)
        return RunResult(status="completed")


def _place_module_for_runtime(module: torch.nn.Module, context: RunContext) -> None:
    """Move a configured module to the run's device and floating dtype."""

    module.to(device=torch.device(context.metadata.device), dtype=_runtime_dtype(context.metadata.dtype))


def _assert_eager_initialized(module: torch.nn.Module) -> None:
    """Fail before evaluation if a model still contains lazy state."""

    for name, parameter in module.named_parameters():
        if isinstance(parameter, UninitializedParameter):
            raise RuntimeError(f"model parameter {name!r} is uninitialized before evaluation")
    for name, buffer in module.named_buffers():
        if isinstance(buffer, UninitializedBuffer):
            raise RuntimeError(f"model buffer {name!r} is uninitialized before evaluation")


def _validate_diagnostics(diagnostics: Sequence[object] | None) -> tuple[Diagnostic, ...]:
    """Validate configured diagnostics without invoking them."""

    if diagnostics is None:
        return ()

    validated: list[Diagnostic] = []
    for index, diagnostic in enumerate(diagnostics):
        if not callable(getattr(diagnostic, "evaluate", None)):
            raise TypeError(
                f"diagnostics[{index}] must be an instantiated diagnostic object with an evaluate(...) "
                f"method, got {type(diagnostic)!r}. This usually means the diagnostic config was not "
                "recursively instantiated by Hydra. Put diagnostics inside the Evaluate runner config "
                "or pass instantiated diagnostic objects."
            )
        name = getattr(diagnostic, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"diagnostics[{index}] must expose a non-empty string name")
        validated.append(diagnostic)
    return tuple(validated)


def _split_local_energy_result(
    result: LocalEnergyResult | torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor] | None]:
    """Return ``(total, terms_or_none)`` from a local-energy result."""

    if isinstance(result, LocalEnergyResult):
        return result.total, result.terms
    return result, None


def _evaluate_diagnostics(
    diagnostics: Sequence[Diagnostic],
    context: EvaluationContext,
) -> dict[str, JsonScalar]:
    """Evaluate diagnostics and merge their flat metric mappings."""

    metrics: dict[str, JsonScalar] = {}
    for diagnostic in diagnostics:
        result = diagnostic.evaluate(context)
        if not isinstance(result, Mapping):
            raise TypeError(f"diagnostic {diagnostic.name!r} must return a mapping of metric names to scalars")
        for key, value in result.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"diagnostic {diagnostic.name!r} returned an empty metric name")
            if key in metrics:
                raise ValueError(f"diagnostic metric key collision for {key!r}")
            _validate_json_scalar(diagnostic.name, key, value)
            metrics[key] = value
    return metrics


def _validate_json_scalar(diagnostic_name: str, key: str, value: object) -> None:
    """Fail loudly when a diagnostic returns a non-scalar metric value."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return
    raise TypeError(
        f"diagnostic {diagnostic_name!r} metric {key!r} must be a JSON scalar, "
        f"got {type(value).__name__}"
    )


def _runtime_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, str(name))
    except AttributeError as exc:
        raise ValueError(f"Unsupported runtime dtype {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported runtime dtype {name!r}")
    return dtype


__all__ = ["Evaluate", "Runner", "Train"]
