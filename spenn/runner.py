"""Public runner targets for configured SpENN executions."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parameter import UninitializedParameter

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
    construction_seed : int or None, optional
        Seed used only to materialize lazy model parameters before sampling and
        optimizer construction. This controls parameter initialization, kept
        separate from the sampler's Markov-chain RNG. When ``None``, lazy
        parameters are still materialized first, but under the ambient RNG.
    """

    def __init__(self, model, sampler, hamiltonian_terms, optimizer, trainer, construction_seed: int | None = None) -> None:
        self.model = model
        self.sampler = sampler
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
        self.optimizer = optimizer
        self.trainer = trainer
        self.construction_seed = None if construction_seed is None else int(construction_seed)

    def _materialize_model(self) -> None:
        """Materialize lazy model parameters before sampling/optimizer build.

        Lazy modules (e.g. ``nn.LazyLinear``) only allocate parameters on their
        first forward. If that first forward were the sampler's model
        evaluation, parameter initialization would be coupled to the sampler's
        RNG and ordering. This forces one deterministic example forward under
        the explicit construction seed inside an isolated RNG fork (so global
        CPU/CUDA RNG state is unchanged for everything else).

        Fails loudly: it raises if lazy parameters exist but the sampler cannot
        provide an example batch, and if any lazy parameter remains
        uninitialized after the materialization forward.
        """

        if not isinstance(self.model, torch.nn.Module):
            return
        uninitialized = [
            name for name, param in self.model.named_parameters() if isinstance(param, UninitializedParameter)
        ]
        if not uninitialized:
            return
        example = getattr(self.sampler, "example_batch", None)
        if not callable(example):
            raise RuntimeError(
                "model has uninitialized (lazy) parameters but the sampler provides no "
                "example_batch() to materialize them before optimizer construction; "
                f"uninitialized parameters: {uninitialized}"
            )
        batch = example()
        cuda_devices = sorted(
            {
                param.device.index
                for param in self.model.parameters()
                if param.device.type == "cuda" and param.device.index is not None
            }
        )
        # Isolate CPU (and any model CUDA devices) RNG so construction seeding
        # never leaks into sampler/global randomness.
        with torch.random.fork_rng(devices=cuda_devices):
            if self.construction_seed is not None:
                torch.manual_seed(self.construction_seed)
            with torch.no_grad():
                self.model(batch)
        remaining = [
            name for name, param in self.model.named_parameters() if isinstance(param, UninitializedParameter)
        ]
        if remaining:
            raise RuntimeError(
                f"lazy parameters remain uninitialized after the materialization forward: {remaining}"
            )

    def run(self, context: RunContext) -> RunResult:
        """Build the optimizer and run the configured VMC training loop."""

        self.emit("run_start", context)
        if isinstance(self.model, torch.nn.Module):
            self.model.train()

        # Materialize lazy parameters before optimizer/sampling so model init is
        # decoupled from sampler RNG (construction seed, not sampler seed).
        self._materialize_model()
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
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
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
