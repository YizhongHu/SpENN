"""Minimal event-driven VMC trainer."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.nn.parameter import UninitializedParameter

from spenn.artifacts import RunContext
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy
from spenn.training.state import TrainerState
from spenn.training.vmc import compute_vmc_objective, summarize_local_energy_terms, summarize_logabs


def _parameter_norm(model) -> float:
    """Return the L2 norm of trainable initialized parameters."""

    total = None
    for param in model.parameters():
        if not param.requires_grad or isinstance(param, UninitializedParameter):
            continue
        value = param.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


def _gradient_norm(model) -> float:
    """Return the L2 norm of available gradients."""

    total = None
    for param in model.parameters():
        if param.grad is None or isinstance(param, UninitializedParameter):
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


class VMCTrainer:
    """Run a fixed number of VMC optimization steps over an event stream.

    The trainer is configuration-only: ``fit`` receives the model, sampler,
    Hamiltonian terms, optimizer, run context, and an ``emit`` callable, and
    drives the sample -> local-energy -> surrogate-loss -> step loop while
    logging metrics and emitting lifecycle events.

    Parameters
    ----------
    max_steps : int
        Number of optimization steps to run.
    log_every_n_steps : int, optional
        Log metrics every ``log_every_n_steps`` steps.
    return_terms : bool, optional
        Whether to request and summarize the per-term local-energy decomposition.
    gradient_clip_norm : float or None, optional
        Maximum gradient norm. When ``None``, gradients are not clipped.
    """

    def __init__(
        self,
        max_steps: int,
        log_every_n_steps: int = 1,
        return_terms: bool = False,
        gradient_clip_norm: float | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.return_terms = bool(return_terms)
        self.gradient_clip_norm = None if gradient_clip_norm is None else float(gradient_clip_norm)

    def fit(
        self,
        *,
        model,
        sampler,
        hamiltonian_terms,
        optimizer: torch.optim.Optimizer,
        context: RunContext,
        emit: Callable[..., None],
    ) -> TrainerState:
        """Run the training loop and return the final `TrainerState`."""

        state = TrainerState(model=model, optimizer=optimizer, sampler=sampler)
        for step in range(1, self.max_steps + 1):
            emit("step_start", payload={"step": step})

            walkers, sampler_stats = sampler.collect_samples(model)
            batch = walkers.make_batch()
            result = local_energy(hamiltonian_terms, model, batch, return_terms=self.return_terms)
            if isinstance(result, LocalEnergyResult):
                total_local_energy = result.total
                term_components = result.components
            else:
                total_local_energy = result
                term_components = None

            output = model(batch)
            objective = compute_vmc_objective(output.logabs, total_local_energy)
            loss = objective.loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
            grad_norm = _gradient_norm(model)
            optimizer.step()

            # Canonical VMC-native metrics come from the objective helper; the
            # trainer only adds trainer-owned mechanics and optional per-term
            # local-energy metrics (metrics only, never part of the objective).
            metrics: dict[str, Any] = dict(objective.metrics)
            metrics.update(summarize_logabs(output.logabs))
            if term_components is not None:
                metrics.update(summarize_local_energy_terms(term_components, hamiltonian_terms))
            metrics["grad_norm"] = grad_norm
            metrics["param_norm"] = _parameter_norm(model)
            metrics.update({f"sampler.{key}": value for key, value in sampler_stats.items()})

            state.step = step
            state.metrics = metrics
            state.samples = walkers
            state.batch = batch
            state.local_energy = total_local_energy.detach()
            state.loss = loss.detach()
            state.wavefunction_output = output
            state.sampler_stats = dict(sampler_stats)

            if self.log_every_n_steps and step % self.log_every_n_steps == 0:
                context.log(metrics, step=step, namespace="train")

            emit("step_end", state=state, payload={"step": step})

        return state


__all__ = ["VMCTrainer"]
