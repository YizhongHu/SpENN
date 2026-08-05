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

from tpen.checkpoint import checkpoint_step_dir_name, save_checkpoint
from tpen.checkpoint.artifact import is_complete_checkpoint_dir
from .base import Callback, Event


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
    Checkpoint step numbers count completed optimizer updates. A training run
    with ``max_steps=500`` writes its terminal checkpoint as
    ``step_000500``, while training metrics and other step callbacks keep
    their existing 0-based loop step indices.
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
        super().__init__(triggers, **kwargs)
        self.output_dir = Path(output_dir)
        self.keep_last = keep_last
        self.save_optimizer = bool(save_optimizer)
        self.save_trainer = bool(save_trainer)
        self.save_sampler = bool(save_sampler)
        self.save_rng = bool(save_rng)

    def on_step_end(self, event: Event) -> None:
        """Write the current step's checkpoint."""

        self._save(event)

    def on_train_end(self, event: Event) -> None:
        """Write the final completed training step checkpoint."""

        self._save(event)

    def _save(self, event: Event) -> None:
        state = event.state
        if state is None:
            raise ValueError("Checkpoint callback requires event.state")
        step = _checkpoint_step(event)
        if step is None:
            raise ValueError("Checkpoint callback requires a step in event payload or state")
        final_dir = self.output_dir / checkpoint_step_dir_name(int(step))
        if is_complete_checkpoint_dir(final_dir):
            return
        save_checkpoint(
            output_dir=self.output_dir,
            step=int(step),
            model=_event_value(event, "model"),
            optimizer=_event_value(event, "optimizer") if self.save_optimizer else None,
            trainer=_event_value(event, "trainer") if self.save_trainer else None,
            sampler=_event_value(event, "sampler") if self.save_sampler else None,
            context=event.context,
            save_optimizer=self.save_optimizer,
            save_trainer=self.save_trainer,
            save_sampler=self.save_sampler,
            save_rng=self.save_rng,
            keep_last=self.keep_last,
        )

    def should_run(self, event: Event) -> bool:
        """Return whether this checkpoint should handle `event`.

        Checkpoint cadence is based on completed updates, not the trainer's
        0-based loop index.
        """

        if event.name not in self.triggers:
            return False
        if self.max_calls is not None and self.num_calls >= self.max_calls:
            return False
        if self.every_n_steps is not None:
            step = _checkpoint_step(event)
            if step is None or step < self.start_step:
                return False
            if (step - self.start_step) % self.every_n_steps != 0:
                return False
        return self._draw_probability()


def _event_value(event: Event, name: str) -> Any:
    if name in event.payload:
        value = event.payload[name]
    else:
        value = getattr(event.state, name, None)
    if value is None:
        raise ValueError(f"Checkpoint callback requires {name!r} in event payload or state")
    return value


def _checkpoint_step(event: Event) -> int | None:
    state = event.state
    trainer = event.payload.get("trainer")
    if trainer is None and state is not None:
        trainer = getattr(state, "trainer", None)
    completed_step = _completed_step_from_trainer(trainer)
    if completed_step is not None:
        return completed_step

    step = event.step
    if step is None and state is not None:
        step = getattr(state, "step", None)
    if step is None:
        return None
    if event.name == "step_end":
        return int(step) + 1
    return int(step)


def _completed_step_from_trainer(trainer: Any) -> int | None:
    state_dict = getattr(trainer, "state_dict", None)
    if not callable(state_dict):
        return None
    state = state_dict()
    if not isinstance(state, Mapping):
        return None
    value = state.get("global_step", state.get("completed_steps"))
    return None if value is None else int(value)


__all__ = [
    "Checkpoint",
]
