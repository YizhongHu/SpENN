"""Training checkpoint callback.

Checkpoint artifact format, hashing, and restore behavior are owned by
``spenn.checkpoint``. This callback is only the lifecycle adapter that receives
runner state through events and asks the package-owned saver to write a
directory checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spenn.checkpoint import save_checkpoint
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

        state = event.state
        if state is None:
            raise ValueError("Checkpoint callback requires event.state")
        step = event.step
        if step is None:
            step = getattr(state, "step", None)
        if step is None:
            raise ValueError("Checkpoint callback requires a step in event payload or state")
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


def _event_value(event: Event, name: str) -> Any:
    if name in event.payload:
        value = event.payload[name]
    else:
        value = getattr(event.state, name, None)
    if value is None:
        raise ValueError(f"Checkpoint callback requires {name!r} in event payload or state")
    return value


__all__ = [
    "Checkpoint",
]
