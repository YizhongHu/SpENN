"""Training runner target."""

from __future__ import annotations

from tpen.artifacts import RunContext, RunResult
from tpen.checkpoint import CheckpointRestored, restore_checkpoint_with_events
from tpen.training.events import ModelBuilt, TrainingCompleted, TrainingStarted
from tpen.training.optim import make_optimizer

from .base import Runner, _assert_eager_initialized, _is_torch_module, _place_module_for_runtime


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
        Sampler exposing
        ``collect_samples(model, device=...) -> (walkers, SamplerStats)``.
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
        context, emit) -> TrainerState`` and a ``next_iteration`` resume
        cursor, which labels the terminal ``train_end`` artifacts.
    """

    def __init__(
        self,
        model,
        sampler,
        hamiltonian_terms,
        optimizer,
        trainer,
        load=None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        # Keep the configured form (sequence or ``dict[str, term]``);
        # ``local_energy`` normalizes it (see ``normalize_hamiltonian_terms``).
        self.hamiltonian_terms = hamiltonian_terms
        self.optimizer = optimizer
        self.trainer = trainer
        self.load = load

    def run(self, context: RunContext) -> RunResult:
        """Build the optimizer and run the configured VMC training loop."""

        if _is_torch_module(self.model):
            _place_module_for_runtime(self.model, context)
            _assert_eager_initialized(self.model)
            self.model.train()

        optimizer = make_optimizer(self.optimizer, self.model.parameters())
        context.emit(ModelBuilt())
        mode = _load_mode(self.load)
        if mode == "model_only":
            raise ValueError("Train rejects load.mode='model_only'; use train_resume")
        if mode == "train_resume":
            report = restore_checkpoint_with_events(
                load=self.load,
                model=self.model,
                optimizer=optimizer,
                trainer=self.trainer,
                sampler=self.sampler,
                context=context,
                emit=context.emit,
            )
            context.emit(CheckpointRestored(report=report))

        context.emit(TrainingStarted())
        final_state = self.trainer.fit(
            model=self.model,
            sampler=self.sampler,
            hamiltonian_terms=self.hamiltonian_terms,
            optimizer=optimizer,
            context=context,
            emit=lambda **_: None,
        )
        # train_end carries the trained model and the durable resume cursor so
        # lifecycle callbacks can label terminal artifacts consistently. The
        # cursor is a hard requirement on the trainer: guessing it from
        # `final_state.step + 1` silently produces a different terminal
        # checkpoint identity whenever the two disagree.
        # Typed counterpart of the legacy ``train_end`` above, emitted at the
        # same point in the same order the trainer pairs its own two channels
        # (legacy ``step_end`` first, then `TrainingIterationCompleted`). It
        # carries the loop's state so `tpen.callback.Checkpoint` can write the
        # terminal checkpoint through typed delivery.
        #
        # Emitting it HERE rather than inside `fit` is what makes it fire when
        # the loop body never ran: `max_steps=0` and a fully-resumed run both
        # return from `fit` without executing an iteration, and both still owe a
        # terminal checkpoint. Deferred import because importing `tpen.training`
        # pulls in torch, and `tpen.runner` must stay importable without it.
        from tpen.training.events import TrainingCompleted

        context.emit(TrainingCompleted(), state=final_state)
        return RunResult(status="completed")


def _load_mode(load) -> str:
    if load is None:
        return "none"
    if hasattr(load, "get"):
        return str(load.get("mode", "none"))
    return "none"


__all__ = ["Train"]
