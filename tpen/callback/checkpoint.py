"""Training checkpoint callback.

Checkpoint artifact format, hashing, and restore behavior are owned by
``tpen.checkpoint``. This callback is only the lifecycle adapter that receives
runner state through events and asks the package-owned saver to write a
directory checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tpen.artifacts import RunContext
from tpen.checkpoint import checkpoint_step_dir_name, save_checkpoint
from tpen.checkpoint.artifact import is_complete_checkpoint_dir
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription

from .base import Callback, Event
from .cadence import SubscriptionGroup


class Checkpoint(Callback):
    """Write directory checkpoints from explicit event state.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger checkpointing (typically ``step_end``).
    output_dir : str or pathlib.Path
        Directory into which checkpoints are written.
    keep_last : int or None, optional
        Keep only the latest ``keep_last`` complete checkpoint directories.
    save_optimizer, save_trainer, save_sampler, save_rng : bool, optional
        Whether to include train-resume state components.
    **kwargs
        Forwarded to `Callback` (e.g. ``every_n_steps``).

    Notes
    -----
    The callback reads the trainer's two progress counters -- both required
    keys of ``trainer.state_dict()`` -- and uses each for exactly one job.

    ``next_iteration`` is *identity*: it names the checkpoint directory, so a
    restored run continues from exactly the iteration the name encodes. A run
    with ``max_steps=500`` writes its terminal checkpoint as ``step_000500``,
    while training metrics and other step callbacks keep their existing 0-based
    loop step indices.

    ``completed_updates`` is *cadence* for the periodic ``step_end`` path:
    ``every_n_steps`` counts applied optimizer updates, so periodic checkpoints
    are spaced by real training progress rather than by attempted iterations.

    Periodic checkpointing is driven by the typed
    `tpen.training.events.UpdateCompleted` event, which the trainer emits in
    the one place ``completed_updates`` is incremented -- immediately after
    ``optimizer.step()`` returns. That gives exactly one candidate firing per
    completed update *by construction*. A vacuum iteration opens no optimizer
    scope and emits ``UpdateSkipped`` instead, which this callback does not
    subscribe to, so no periodic checkpoint fires for it. This selection is
    unconditional: with no ``every_n_steps`` configured, ``step_end`` writes on
    every completed update, not on every iteration. No "has the counter moved"
    defence is needed -- the non-injectivity of ``completed_updates`` across
    iterations is handled structurally.

    The typed event is the *trigger*; the *cadence coordinate* is the durable
    ``completed_updates`` value, never the run-local ``Occurrence.count``.
    Occurrence counts restart at 1 after a checkpoint restore, so a resumed run
    that gated on them would silently shift its checkpoint phase relative to an
    uninterrupted run. Gating on the durable counter keeps the two aligned.

    The write itself still happens at the ``step_end`` trigger rather than at
    the ``UpdateCompleted`` occurrence, because the resume cursor has not yet
    advanced when the update completes -- the trainer assigns
    ``next_iteration`` at the end of the loop body -- and the typed occurrence
    carries no model, optimizer, or sampler to save. ``UpdateCompleted``
    therefore selects which ``step_end`` may write.

    ``train_end`` is a *terminal* trigger, not a periodic one, and update
    selection deliberately does not apply to it: a run whose final iteration
    skipped its update, and a run whose loop body never executed, must both
    still write their terminal checkpoint. Its ``every_n_steps`` window, when
    one is configured, keeps the resume-cursor coordinate it has always used.

    Both counters are read independently of ``save_trainer``, because the v2
    manifest records both whether or not ``trainer.json`` is written.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        output_dir: str | Path,
        *,
        keep_last: int | None = None,
        save_optimizer: bool = True,
        save_trainer: bool = True,
        save_sampler: bool = True,
        save_rng: bool = True,
        **kwargs: Any,
    ) -> None:
        # Importing ``tpen.callback`` must stay torch-free, and the
        # ``tpen.training`` package __init__ pulls in torch. Resolve the
        # training-owned event types only when this callback is constructed.
        from tpen.training.events import UpdateCompleted

        typed_groups = (
            # One group with one selector, so `validate_subscription_groups`
            # has nothing to overlap. `UpdateCompleted` is a plain Event rather
            # than a Started/Ended boundary, so it could not collide with a
            # lifecycle selector even if a second group were added later.
            #
            # `cadence=None` is deliberate: a group `Cadence` gates on the
            # run-local `Occurrence.count`, which restarts after a restore.
            # The scheduling window is applied to the durable
            # `completed_updates` counter instead, in `_legacy_cadence_step`.
            SubscriptionGroup(selectors=(Subscription.of(UpdateCompleted),)),
        )
        super().__init__(triggers, typed_groups=typed_groups, **kwargs)
        self.output_dir = Path(output_dir)
        self.keep_last = keep_last
        self.save_optimizer = bool(save_optimizer)
        self.save_trainer = bool(save_trainer)
        self.save_sampler = bool(save_sampler)
        self.save_rng = bool(save_rng)
        self._update_completed_type = UpdateCompleted
        # Trainer step of the most recent iteration that applied an optimizer
        # update. Matched against the step reported at `step_end`, so it is
        # self-clearing: a later iteration that skipped its update simply never
        # matches, and no explicit per-iteration reset is required.
        self._updated_iteration_step: int | None = None

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Mark this iteration as having applied its optimizer update."""

        del context
        event = occurrence.event
        if isinstance(event, self._update_completed_type):
            # `occurrence.count` is deliberately unused: it is run-local and
            # restarts after a restore. The durable coordinate is read from the
            # trainer at `step_end`.
            self._updated_iteration_step = int(event.iteration.step)

    def on_step_end(self, event: Event) -> None:
        """Write a periodic checkpoint for an iteration that applied an update.

        Update selection lives here rather than in `_legacy_cadence_step` for
        two reasons. It must apply even when no ``every_n_steps`` is configured
        -- the base class consults the cadence hook only when one is set, so
        "no cadence" would otherwise mean "every iteration" instead of "every
        completed update". And it must apply *only* to this periodic path: a
        terminal ``train_end`` checkpoint is not periodic and must not be
        dropped because the final iteration happened to skip its update.
        """

        state = _require_state(event)
        if self._updated_iteration_step != state.step:
            return
        self._save(event, state)

    def on_train_end(self, event: Event) -> None:
        """Write the final completed training step checkpoint."""

        self._save(event, _require_state(event))

    def _save(self, event: Event, state: Any) -> None:
        next_iteration, completed_updates = _trainer_progress(state)
        final_dir = self.output_dir / checkpoint_step_dir_name(next_iteration)
        if is_complete_checkpoint_dir(final_dir):
            return
        model = state.model
        if model is None:
            raise ValueError("Checkpoint callback requires event.state.model")
        save_checkpoint(
            output_dir=self.output_dir,
            next_iteration=next_iteration,
            completed_updates=completed_updates,
            model=model,
            optimizer=state.optimizer if self.save_optimizer else None,
            trainer=state.trainer if self.save_trainer else None,
            sampler=state.sampler if self.save_sampler else None,
            context=event.context,
            save_optimizer=self.save_optimizer,
            save_trainer=self.save_trainer,
            save_sampler=self.save_sampler,
            save_rng=self.save_rng,
            keep_last=self.keep_last,
        )

    def _legacy_cadence_step(self, event: Event) -> int | None:
        """Return the durable coordinate the inherited ``every_n_steps`` window uses.

        This hook applies only the periodic *window*. Whether an iteration is
        eligible at all is decided by `on_step_end`; see its docstring.

        Parameters
        ----------
        event : Event
            Lifecycle event carrying the loop's `tpen.training.TrainerState`.

        Returns
        -------
        int
            ``completed_updates`` for the periodic ``step_end`` path, so
            periodic checkpoints are spaced by optimizer updates that actually
            ran. ``next_iteration`` for the terminal ``train_end`` path, which
            keeps the resume-cursor coordinate it has always used.
        """

        state = _require_state(event)
        # The trainer is a hard requirement of this callback on every path, so
        # it is read before any branching: a missing trainer must fail loudly.
        next_iteration, completed_updates = _trainer_progress(state)
        if event.name == "train_end":
            # A terminal checkpoint is not periodic. It keeps its pre-existing
            # resume-cursor coordinate and is never filtered by whether the
            # final iteration applied an optimizer update -- a run whose last
            # iteration was a vacuum, and a run whose loop body never executed
            # at all (`max_steps=0`, or a fully-resumed run, where
            # ``TrainerState.step`` is still -1), must both still write it.
            return next_iteration
        return completed_updates

    def _reset_typed_state(self) -> None:
        """Drop the armed update when the owning RunContext identity changes."""

        self._updated_iteration_step = None


def _require_state(event: Event) -> Any:
    """Return the event's training state, failing loudly when it is absent."""

    state = event.state
    if state is None:
        raise ValueError("Checkpoint callback requires event.state")
    return state


def _trainer_progress(state: Any) -> tuple[int, int]:
    """Read the trainer's two required progress counters from typed state.

    Parameters
    ----------
    state : Any
        The loop's `tpen.training.TrainerState`. Its ``trainer`` field is read
        by name; no payload keys and no arbitrary attributes are probed.

    Returns
    -------
    tuple of int
        ``(next_iteration, completed_updates)``.

    Raises
    ------
    ValueError
        If the state carries no trainer.
    TypeError
        If the trainer does not expose a ``state_dict()`` returning a mapping.
    KeyError
        If either counter is missing. Both are required keys with no default:
        a trainer that cannot report its resume cursor must not be allowed to
        mislabel a checkpoint directory.
    """

    trainer = state.trainer
    if trainer is None:
        raise ValueError("Checkpoint callback requires event.state.trainer")
    state_dict = getattr(trainer, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("trainer must expose state_dict() for checkpoint progress")
    progress = state_dict()
    if not isinstance(progress, Mapping):
        raise TypeError("trainer.state_dict() must return a mapping")
    return int(progress["next_iteration"]), int(progress["completed_updates"])


__all__ = [
    "Checkpoint",
]
