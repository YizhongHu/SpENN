"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Callback, Event
from spenn.logging import Logger
from spenn.physics.hamiltonian import local_energy, summarize_local_energy


class Runner:
    """Base runner with callback lifecycle dispatch.

    Callbacks and loggers are owned by the `RunContext`; `emit` dispatches
    lifecycle events into ``context.callbacks``.
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
    """Placeholder for future training runner configs."""

    def run(self, context: RunContext) -> RunResult:
        """Raise until training runner support is implemented."""

        raise NotImplementedError("spenn.runner.Train will be implemented in a later PR.")


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
        Evaluation settings (``return_terms``, ``expected_energy``).
    """

    def __init__(self, model, sampler, hamiltonian_terms, evaluation: Any = None) -> None:
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.hamiltonian_terms = list(hamiltonian_terms)
        self.evaluation = evaluation

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
        expected_energy = self._eval_get("expected_energy", None)

        walkers, sampler_stats = self.sampler.collect_samples(self.model)
        result = local_energy(self.hamiltonian_terms, self.model, walkers.make_batch(), return_terms=return_terms)

        metrics = summarize_local_energy(result, expected_energy=expected_energy)
        metrics.update({f"sampler.{key}": value for key, value in sampler_stats.items()})
        context.log(metrics, step=0, namespace="eval")

        self.emit("run_end", context)
        return RunResult(status="completed")


__all__ = ["Evaluate", "Load", "Runner", "Scaffold", "Train"]
