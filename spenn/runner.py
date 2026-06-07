"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from spenn.artifacts import RunContext, RunResult
from spenn.callback import Callback, Event
from spenn.logging import Logger
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy


class Runner:
    """Base runner with callback lifecycle dispatch."""

    def __init__(
        self,
        callbacks: Iterable[Callback] | None = None,
        loggers: Iterable[Logger] | None = None,
    ) -> None:
        self.callbacks = list(callbacks or [])
        self.loggers = list(loggers or [])
        self.callback_registry = self.build_callback_registry(self.callbacks)

    def build_callback_registry(self, callbacks: Iterable[Callback]) -> dict[str, list[Callback]]:
        """Group callbacks by subscribed event name."""

        registry: dict[str, list[Callback]] = {}
        for callback in callbacks:
            for trigger in callback.triggers:
                registry.setdefault(trigger, []).append(callback)
        return registry

    def emit(
        self,
        name: str,
        context: RunContext,
        *,
        state: object | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit one lifecycle event to subscribed callbacks."""

        event = Event(name=name, context=context, state=state, payload={} if payload is None else payload)
        for callback in self.callback_registry.get(name, []):
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
    and logs it. It knows nothing about any specific system.

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
    callbacks, loggers : iterable, optional
        Lifecycle callbacks and metric loggers.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        evaluation: Any = None,
        callbacks: Iterable[Callback] | None = None,
        loggers: Iterable[Logger] | None = None,
    ) -> None:
        super().__init__(callbacks=callbacks, loggers=loggers)
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
        return_terms = bool(self._eval_get("return_terms", False))
        expected_energy = self._eval_get("expected_energy", None)

        walkers, sampler_stats = self.sampler.collect_samples(self.model)
        result = local_energy(self.hamiltonian_terms, self.model, walkers.make_batch(), return_terms=return_terms)

        metrics = summarize_energy(result, expected_energy=expected_energy)
        metrics.update({f"sampler.{key}": value for key, value in sampler_stats.items()})
        context.log(metrics, step=0, namespace="eval")

        self.emit("run_end", context)
        return RunResult(status="completed")


def summarize_energy(result: LocalEnergyResult | torch.Tensor, *, expected_energy: float | None = None) -> dict[str, Any]:
    """Summarize a local-energy sample into scalar metrics.

    Parameters
    ----------
    result : LocalEnergyResult or torch.Tensor
        Per-sample local energy, optionally with a per-term decomposition.
    expected_energy : float or None, optional
        Known exact energy; when given, error metrics are included.

    Returns
    -------
    dict
        Scalar energy metrics suitable for logging.
    """

    if isinstance(result, LocalEnergyResult):
        eloc, terms = result.total, result.terms
    else:
        eloc, terms = result, {}
    n = int(eloc.numel())
    finite_mask = torch.isfinite(eloc)
    n_finite = int(finite_mask.sum().item())
    finite_eloc = eloc[finite_mask]
    mean = finite_eloc.mean()
    std = finite_eloc.std(unbiased=False)
    metrics: dict[str, Any] = {
        "energy_mean": float(mean.item()),
        "energy_stderr": float((std / math.sqrt(n_finite)).item()) if n_finite else float("inf"),
        "energy_variance": float(finite_eloc.var(unbiased=False).item()),
        "n_samples": n,
        "nonfinite_energy_fraction": float((n - n_finite) / n) if n else 0.0,
    }
    if expected_energy is not None:
        expected = float(expected_energy)
        metrics["expected_energy"] = expected
        metrics["energy_error"] = metrics["energy_mean"] - expected
        metrics["abs_energy_error"] = abs(metrics["energy_mean"] - expected)
    for name, value in terms.items():
        metrics[f"terms.{name}_mean"] = float(value[torch.isfinite(value)].mean().item())
    return metrics


__all__ = ["Evaluate", "Load", "Runner", "Scaffold", "Train", "summarize_energy"]
