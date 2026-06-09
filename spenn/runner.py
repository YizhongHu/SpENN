"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from typing import Any

import torch

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Event
from spenn.physics.hamiltonian import local_energy, summarize_local_energy
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
        Sampler exposing ``collect_samples(model) -> (walkers, stats)``.
    hamiltonian_terms : sequence
        Hamiltonian terms summed by `local_energy`.
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
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.optimizer = optimizer
        self.trainer = trainer

    def run(self, context: RunContext) -> RunResult:
        """Build the optimizer and run the configured VMC training loop."""

        self.emit("run_start", context)
        if isinstance(self.model, torch.nn.Module):
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
    """Generic sampled local-energy evaluation runner.

    This runner intentionally does not own diagnostics yet. PR6 will add the
    real diagnostics interface. Until then, Evaluate only samples configurations,
    computes intrinsic local-energy summary metrics, and logs them through the
    run context. It does not read reference energy and does not accept
    callbacks/loggers/diagnostics (callbacks and loggers are RunContext-owned).

    Sampler contract assumed by this runner::

        walkers, sampler_stats = sampler.collect_samples(model)
        batch = walkers.make_batch()

    Parameters
    ----------
    model : callable
        Wavefunction model returning ``WavefunctionOutput``.
    sampler : object
        Sampler exposing ``collect_samples(model) -> (walkers, stats)``.
    hamiltonian_terms : sequence
        Hamiltonian terms summed by `local_energy`.
    return_terms : bool, optional
        Whether to request per-term local-energy components from `local_energy`.
    """

    def __init__(self, model, sampler, hamiltonian_terms, return_terms: bool = False) -> None:
        self.model = model
        self.sampler = sampler
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.return_terms = bool(return_terms)

    def run(self, context: RunContext) -> RunResult:
        """Sample configurations, evaluate local energy, and log metrics."""

        self.emit("run_start", context)
        self.emit("evaluate_start", context)

        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        # No torch.no_grad: local-energy evaluation needs position derivatives.
        walkers, sampler_stats = self.sampler.collect_samples(self.model)
        batch = walkers.make_batch()

        self.emit("samples_collected", context, payload={"sampler_stats": dict(sampler_stats)})

        result = local_energy(self.hamiltonian_terms, self.model, batch, return_terms=self.return_terms)

        metrics = summarize_local_energy(result)
        metrics.update({f"sampler.{key}": value for key, value in sampler_stats.items()})

        context.log(metrics, step=0, namespace="eval")

        self.emit("evaluate_end", context, payload={"metrics": metrics})
        self.emit("run_end", context)
        return RunResult(status="completed")


__all__ = ["Evaluate", "Runner", "Train"]
