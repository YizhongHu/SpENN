"""Training runner target."""

from __future__ import annotations

from spenn.artifacts import RunContext, RunResult
from spenn.checkpoint import restore_checkpoint_with_events
from spenn.training.optim import make_optimizer

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

        self.emit("run_start", context)
        if _is_torch_module(self.model):
            _place_module_for_runtime(self.model, context)
            _assert_eager_initialized(self.model)
            self.model.train()

        optimizer = make_optimizer(self.optimizer, self.model.parameters())
        self.emit("model_built", context, payload={"model": self.model, "optimizer": optimizer})
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
                emit=self.emit,
            )
            self.emit("checkpoint_restored", context, payload={"restore_report": report.to_dict()})

        self.emit("train_start", context)
        final_state = self.trainer.fit(
            model=self.model,
            sampler=self.sampler,
            hamiltonian_terms=self.hamiltonian_terms,
            optimizer=optimizer,
            context=context,
            emit=lambda name, *, state=None, payload=None: self.emit(name, context, state=state, payload=payload),
        )
        # train_end carries the trained model and final step so lifecycle
        # callbacks do not depend on trainer internals.
        self.emit(
            "train_end",
            context,
            state=final_state,
            payload={"model": self.model, "step": int(final_state.step)},
        )
        self.emit("run_end", context)
        return RunResult(status="completed")


def _load_mode(load) -> str:
    if load is None:
        return "none"
    if hasattr(load, "get"):
        return str(load.get("mode", "none"))
    return "none"


__all__ = ["Train"]
