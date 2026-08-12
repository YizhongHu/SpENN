"""Minimal event-driven VMC trainer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.training.events import (
    Backward,
    BuildBatch,
    CollectSamples,
    Forward,
    LocalEnergy,
    Metrics,
    Objective,
    OptimizerUpdate,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
    UpdateSkipped,
)
from tpen.training.state import TrainerState
from tpen.training.vmc import compute_vmc_objective, summarize_local_energy_terms, summarize_logabs

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

    Notes
    -----
    Progress uses two independent counters. ``next_iteration`` is the durable
    resume cursor and is assigned near the end of the loop body, so it advances
    once per iteration that *ran to completion*: an iteration that raises
    part-way through (say inside ``local_energy`` or ``optimizer.step()``) never
    advances it and is retried from the same cursor on resume. It advances
    whether or not that iteration applied an optimizer update.
    ``completed_updates`` counts optimizer updates that actually returned, so the
    two diverge whenever a completed iteration skips its update (the
    zero-electron vacuum).

    Every key in the ``train`` record logged at loop step ``k`` describes the
    *pre-update* model -- the one that produced step ``k``'s samples. That
    includes ``param_norm``, which is therefore read before ``optimizer.step()``
    rather than after it. Keeping the record single-version is a contract: a new
    trainer-owned metric must be computed before the update, or it does not
    belong in this record.
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
        # Durable resume cursor: the next iteration this trainer will attempt.
        self.next_iteration = 0
        # Optimizer updates that actually returned; skipped updates never count.
        self.completed_updates = 0

    def state_dict(self) -> dict[str, int]:
        """Return checkpointable trainer progress state.

        Returns
        -------
        dict
            ``next_iteration`` (durable resume cursor) and
            ``completed_updates`` (applied optimizer updates). The two diverge
            whenever a completed iteration skipped its optimizer update.
        """

        return {
            "next_iteration": int(self.next_iteration),
            "completed_updates": int(self.completed_updates),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore trainer progress state for ``train_resume``.

        Parameters
        ----------
        state : Mapping
            Trainer state read from a checkpoint's ``trainer.json``. Both
            ``next_iteration`` and ``completed_updates`` are required.

        Raises
        ------
        KeyError
            If either required key is missing. Checkpoints written before the
            rename (``global_step``/``completed_steps``) are unsupported and
            fail here rather than silently resuming from step 0.
        ValueError or TypeError
            If either value cannot be coerced to ``int``. Both coercions run
            before either assignment, so a rejected state leaves the trainer's
            progress counters unmutated.
        """

        for key in ("next_iteration", "completed_updates"):
            if key not in state:
                raise KeyError(
                    f"trainer state is missing required key {key!r}; checkpoints "
                    "written with 'global_step'/'completed_steps' are unsupported"
                )
        # Coerce both counters before assigning either one, so a malformed value
        # cannot leave the trainer half-restored at a bogus resume cursor.
        next_iteration = int(state["next_iteration"])
        completed_updates = int(state["completed_updates"])
        self.next_iteration = next_iteration
        self.completed_updates = completed_updates

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
        # One `TrainerState` instance is passed beside every typed occurrence
        # this loop emits, so a typed handler reads it at the moment it is
        # delivered. Its reference fields (`model`, `optimizer`, `trainer`,
        # `sampler`) are live everywhere; its value fields are assigned near the
        # end of the body, so a handler at a boundary *above* that assignment
        # reads the previous iteration's values -- and the constructor defaults,
        # including `step == -1`, on the first iteration.
        # Training-loop steps are 0-indexed for metrics and most callbacks.
        # Checkpoint callbacks reuse the resume cursor `next_iteration`.
        for step in range(self.next_iteration, self.max_steps):
            iteration = TrainingIteration(step=step)
            with context.scope(iteration, state=state):
                emit("step_start", step=step)

                with context.scope(CollectSamples(step=step), state=state):
                    walkers, sampler_stats = sampler.collect_samples(
                        model, device=context.metadata.device
                    )
                with context.scope(BuildBatch(step=step), state=state):
                    batch = walkers.make_batch()
                with context.scope(LocalEnergy(step=step), state=state):
                    result = local_energy(
                        hamiltonian_terms,
                        model,
                        batch,
                        return_terms=self.return_terms,
                    )
                if isinstance(result, LocalEnergyResult):
                    total_local_energy = result.total
                    term_energies = result.terms
                else:
                    total_local_energy = result
                    term_energies = None

                with context.scope(Forward(step=step), state=state):
                    output = model(batch)
                with context.scope(Objective(step=step), state=state):
                    objective = compute_vmc_objective(output.logabs, total_local_energy)
                loss = objective.loss

                # Read the parameter norm here, before any update, so the whole
                # `train` record describes exactly one model version: the model
                # that produced these samples, this loss, and this gradient.
                # Reading it after `optimizer.step()` would make `param_norm`
                # the sole post-update key in an otherwise pre-update record.
                # Both branches below reach the metrics block, so the skip path
                # reports this same pre-update value.
                param_norm = _parameter_norm(model)

                optimizer.zero_grad(set_to_none=True)
                optimizer_step = False
                if loss.requires_grad:
                    with context.scope(Backward(step=step), state=state):
                        loss.backward()
                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip_norm
                        )
                    grad_norm = _gradient_norm(model)
                    with context.scope(OptimizerUpdate(step=step), state=state):
                        optimizer.step()
                    # The update counts only once `optimizer.step()` has
                    # returned, so this always follows Ended[OptimizerUpdate].
                    self.completed_updates += 1
                    context.emit(UpdateCompleted(iteration=iteration), state=state)
                    optimizer_step = True
                elif batch.n_electrons == 0:
                    # The zero-electron vacuum has no sampled coordinate degrees
                    # of freedom, so the current Pfaffian readout yields a
                    # constant wavefunction and a no-op optimizer step is the
                    # correct loop behavior. No OptimizerUpdate scope opens on
                    # this path. Nonzero disconnected losses still fail below.
                    grad_norm = 0.0
                    context.emit(UpdateSkipped(iteration=iteration), state=state)
                else:
                    raise RuntimeError(
                        "VMC loss is disconnected from model parameters for a "
                        "nonzero-electron batch"
                    )

                # Canonical VMC-native metrics come from the objective helper;
                # the trainer only adds trainer-owned mechanics and optional
                # per-term local-energy metrics (metrics only, never part of the
                # objective).
                with context.scope(Metrics(step=step), state=state):
                    metrics: dict[str, Any] = dict(objective.metrics)
                    metrics.update(summarize_logabs(output.logabs))
                    if term_energies is not None:
                        metrics.update(summarize_local_energy_terms(term_energies))
                    metrics["grad_norm"] = grad_norm
                    metrics["param_norm"] = param_norm
                    metrics["loss_has_grad"] = bool(loss.requires_grad)
                    metrics["optimizer_step"] = optimizer_step

                state.step = step
                state.metrics = metrics
                # Published as `train/optimizer_step` above and carried here as
                # a typed field for the same reason: a callback observing update
                # by-products (gradients) must be able to tell "this iteration
                # applied no update, so there is nothing to see" from "an update
                # ran and I still saw nothing", which is a broken observation.
                state.optimizer_step = optimizer_step
                state.samples = walkers
                state.batch = batch
                state.local_energy = total_local_energy.detach()
                state.loss = loss.detach()
                state.wavefunction_output = output
                state.sampler_stats = sampler_stats

                if self.log_every_n_steps and step % self.log_every_n_steps == 0:
                    context.log(metrics, step=step, namespace="train")
                    # Explicit None check: SamplerStats defines no __bool__, so a
                    # truthiness test would always pass and could not express
                    # "this sampler produced no diagnostics". The typed record
                    # owns the metric-name composition, so the trainer never
                    # re-spells a train/sampler key.
                    if sampler_stats is not None:
                        context.log(
                            sampler_stats.as_metrics(),
                            step=step,
                            namespace="train/sampler",
                        )

                # Assigned here, near the end of the body, so only an iteration
                # that ran to completion advances the resume cursor: a crash
                # above retries this same step on resume. An iteration that
                # applied no optimizer update still advances it.
                self.next_iteration = step + 1
                emit(
                    "step_end",
                    state=state,
                    step=step,
                    payload={
                        "model": model,
                        "optimizer": optimizer,
                        "trainer": self,
                        "sampler": sampler,
                    },
                )
                context.emit(TrainingIterationCompleted(iteration=iteration), state=state)

        return state


__all__ = ["VMCTrainer"]
