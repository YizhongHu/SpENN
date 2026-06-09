"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Callback, Event
from spenn.logging import Logger
from spenn.physics.hamiltonian import local_energy, reference_energy_metrics, summarize_local_energy
from spenn.training.optim import make_optimizer


class Runner:
    """Base runner with callback lifecycle dispatch.

    Runners own their callbacks and loggers (instantiated from the runner's own
    config block). ``run_from_config`` mirrors them onto the `RunContext` so
    ``context.log`` and lifecycle dispatch operate on the runner's objects.
    """

    def __init__(
        self,
        callbacks: Iterable[Callback] | None = None,
        loggers: Iterable[Logger] | None = None,
    ) -> None:
        self.callbacks = list(callbacks or [])
        self.loggers = list(loggers or [])

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


class Scaffold(Runner):
    """Runner that validates generic run-management plumbing."""

    def run(self, context: RunContext) -> RunResult:
        """Execute the PR1 scaffold lifecycle."""

        self.emit("run_start", context)
        context.log({"scaffold_completed": True}, step=0, namespace="scaffold")
        self.emit("run_end", context)
        return RunResult(status="completed")


class Train(Runner):
    """Config-driven VMC training runner.

    Builds the optimizer, drives the configured trainer through the VMC loop,
    and emits lifecycle events. It owns its callbacks and loggers (and adds no
    exception handling -- ``run_from_config`` owns that); it only emits events
    and lets the trainer log through the context.

    Parameters
    ----------
    model : torch.nn.Module
        Wavefunction model to optimize.
    sampler : object
        Sampler exposing ``collect_samples(model) -> (walkers, stats)``.
    hamiltonian_terms : sequence
        Hamiltonian terms summed by `local_energy`.
    optimizer_factory : Any
        Optimizer factory or config consumed by `make_optimizer`.
    trainer : object
        Trainer exposing ``fit(*, model, sampler, hamiltonian_terms, optimizer,
        context, emit) -> TrainerState``.
    callbacks : sequence, optional
        Lifecycle callbacks owned by this runner.
    loggers : sequence, optional
        Metric loggers owned by this runner.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        optimizer_factory,
        trainer,
        callbacks=None,
        loggers=None,
    ) -> None:
        super().__init__(callbacks=callbacks, loggers=loggers)
        self.model = model
        self.sampler = sampler
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.optimizer_factory = optimizer_factory
        self.trainer = trainer

    def run(self, context: RunContext) -> RunResult:
        """Build the optimizer and run the configured VMC training loop."""

        self.emit("run_start", context)
        if isinstance(self.model, torch.nn.Module):
            self.model.train()

        optimizer = make_optimizer(self.optimizer_factory, self.model.parameters())
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


class Load(Runner):
    """Placeholder for future load/evaluate runner configs."""

    def run(self, context: RunContext) -> RunResult:
        """Raise until load runner support is implemented."""

        raise NotImplementedError("spenn.runner.Load will be implemented in a later PR.")


class Evaluate(Runner):
    """Generic sampled energy-evaluation runner (no training).

    Samples configurations with the configured sampler, evaluates the local
    energy of the configured Hamiltonian terms, summarizes the energy estimate,
    and logs it through the context. It knows nothing about any specific system.

    Sampler contract assumed by this runner::

        walkers, sampler_stats = sampler.collect_samples(model)
        batch = walkers.make_batch()

    where ``sampler_stats`` is a dict of scalar logging values.

    Parameters
    ----------
    model : callable
        Wavefunction model returning ``WavefunctionOutput``.
    sampler : object
        Sampler exposing ``collect_samples(model) -> (walkers, stats)``.
    hamiltonian_terms : sequence
        Hamiltonian terms summed by `local_energy`.
    evaluation : Mapping or None, optional
        Evaluation settings (``return_terms``, ``reference_energy``). When
        ``reference_energy`` is set, reference-comparison metrics are merged in.
    callbacks : sequence, optional
        Lifecycle callbacks owned by this runner.
    loggers : sequence, optional
        Metric loggers owned by this runner.
    diagnostics : sequence, optional
        Diagnostics owned by this runner (Evaluate-only; not consumed by Train).
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        evaluation: Any = None,
        callbacks=None,
        loggers=None,
        diagnostics=None,
    ) -> None:
        super().__init__(callbacks=callbacks, loggers=loggers)
        self.model = model
        self.sampler = sampler
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.evaluation = evaluation
        self.diagnostics = list(diagnostics or [])

    def _eval_get(self, key: str, default: Any) -> Any:
        ev = self.evaluation
        if ev is None:
            return default
        if hasattr(ev, "get"):
            return ev.get(key, default)
        return getattr(ev, key, default)

    def run(self, context: RunContext) -> RunResult:
        """Sample, evaluate the local energy, and log the energy estimate."""

        self.emit("run_start", context)
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
        return_terms = bool(self._eval_get("return_terms", False))
        reference_energy = self._eval_get("reference_energy", None)

        walkers, sampler_stats = self.sampler.collect_samples(self.model)
        result = local_energy(self.hamiltonian_terms, self.model, walkers.make_batch(), return_terms=return_terms)

        metrics = summarize_local_energy(result)
        if reference_energy is not None:
            metrics.update(
                reference_energy_metrics(
                    energy_mean=metrics["energy_mean"],
                    reference_energy=float(reference_energy),
                )
            )
        metrics.update({f"sampler.{key}": value for key, value in sampler_stats.items()})
        context.log(metrics, step=0, namespace="eval")

        self.emit("run_end", context)
        return RunResult(status="completed")


__all__ = ["Evaluate", "Load", "Runner", "Scaffold", "Train"]
