"""Minimal event-driven VMC trainer."""

from __future__ import annotations

from typing import Any, Callable

from spenn.artifacts import RunContext
from spenn.dependencies import require_torch
from spenn.physics.hamiltonian import LocalEnergyResult, local_energy
from spenn.training.state import TrainerState
from spenn.training.vmc import compute_vmc_objective, summarize_local_energy_terms, summarize_logabs

torch = require_torch(feature="VMC training")


def _parameter_norm(model) -> float:
    """Return the L2 norm of trainable initialized parameters."""

    total = None
    for param in model.parameters():
        if not param.requires_grad:
            continue
        value = param.detach().pow(2).sum()
        total = value if total is None else total + value
    return float(torch.sqrt(total).item()) if total is not None else 0.0


def _gradient_norm(model) -> float:
    """Return the L2 norm of available gradients."""

    total = None
    for param in model.parameters():
        if param.grad is None:
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
        self.global_step = 0

    def state_dict(self) -> dict[str, int]:
        """Return checkpointable trainer progress state."""

        return {
            "global_step": int(self.global_step),
            "completed_steps": int(self.global_step),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trainer progress state for ``train_resume``."""

        global_step = state.get("global_step", state.get("completed_steps", 0))
        self.global_step = int(global_step)

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

        state = TrainerState(model=model, optimizer=optimizer, trainer=self, sampler=sampler)
        # Steps are 0-indexed: the first step always satisfies the
        # step % every_n_steps == 0 cadence gates in callbacks and logging.
        for step in range(self.global_step, self.max_steps):
            emit("step_start", payload={"step": step})

            walkers, sampler_stats = sampler.collect_samples(model, device=context.metadata.device)
            batch = walkers.make_batch()
            result = local_energy(hamiltonian_terms, model, batch, return_terms=self.return_terms)
            if isinstance(result, LocalEnergyResult):
                total_local_energy = result.total
                term_energies = result.terms
            else:
                total_local_energy = result
                term_energies = None

            output = model(batch)
            objective = compute_vmc_objective(output.logabs, total_local_energy)
            loss = objective.loss

            optimizer.zero_grad(set_to_none=True)
            optimizer_step = False
            if loss.requires_grad:
                loss.backward()
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                grad_norm = _gradient_norm(model)
                optimizer.step()
                optimizer_step = True
            elif batch.n_electrons == 0:
                # The zero-electron vacuum has no sampled coordinate degrees of
                # freedom, so the current Pfaffian readout yields a constant
                # wavefunction and a no-op optimizer step is the correct loop
                # behavior. Nonzero disconnected losses still fail below.
                grad_norm = 0.0
            else:
                raise RuntimeError(
                    "VMC loss is disconnected from model parameters for a nonzero-electron batch"
                )

            # Canonical VMC-native metrics come from the objective helper; the
            # trainer only adds trainer-owned mechanics and optional per-term
            # local-energy metrics (metrics only, never part of the objective).
            metrics: dict[str, Any] = dict(objective.metrics)
            metrics.update(summarize_logabs(output.logabs))
            if term_energies is not None:
                metrics.update(summarize_local_energy_terms(term_energies))
            metrics["grad_norm"] = grad_norm
            metrics["param_norm"] = _parameter_norm(model)
            metrics["loss_has_grad"] = bool(loss.requires_grad)
            metrics["optimizer_step"] = optimizer_step

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
                if sampler_stats:
                    context.log(dict(sampler_stats), step=step, namespace="train/sampler")

            self.global_step = step + 1
            emit(
                "step_end",
                state=state,
                payload={
                    "step": step,
                    "model": model,
                    "optimizer": optimizer,
                    "trainer": self,
                    "sampler": sampler,
                },
            )

        return state


__all__ = ["VMCTrainer"]
