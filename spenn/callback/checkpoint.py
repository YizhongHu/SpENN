"""Training checkpoint callback."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .base import Callback, Event


class Checkpoint(Callback):
    """Write training checkpoints from the loop `TrainerState`.

    Reads ``event.state`` (a `spenn.training.state.TrainerState`) and writes a
    ``torch.save`` payload to ``output_dir/step_<step>.pt`` and
    ``output_dir/latest.pt``. PR3 only writes checkpoints; it does not resume.

    Parameters
    ----------
    triggers : iterable of str
        Event names that should trigger checkpointing (typically ``step_end``).
    output_dir : str or pathlib.Path
        Directory into which checkpoints are written.
    **kwargs
        Forwarded to `Callback` (e.g. ``every_n_steps``).
    """

    def __init__(self, triggers: Iterable[str], output_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(triggers, **kwargs)
        self.output_dir = Path(output_dir)

    def on_step_end(self, event: Event) -> None:
        """Write the current step's checkpoint."""

        import torch

        state = event.state
        sampler = getattr(state, "sampler", None)
        sampler_mcmc_state = getattr(sampler, "mcmc_state_dict", None)
        payload = {
            "step": state.step,
            "model_state_dict": state.model.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "sampler_mcmc_state": sampler_mcmc_state() if callable(sampler_mcmc_state) else None,
            "metrics": state.metrics,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_dir / f"step_{state.step}.pt")
        torch.save(payload, self.output_dir / "latest.pt")



__all__ = ["Checkpoint"]
