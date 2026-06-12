"""Training checkpoint callback and checkpoint restore helper."""

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


def load_model_checkpoint(model, path: str | Path, strict: bool = True):
    """Restore trained weights from a `Checkpoint` payload and return the model.

    Hydra-instantiable wrapper for evaluation configs: nest the model spec
    under ``model`` and point ``path`` at a checkpoint written by `Checkpoint`
    (e.g. ``<train_run_dir>/checkpoints/latest.pt``)::

        runner:
          _target_: spenn.runner.Evaluate
          model:
            _target_: spenn.callback.checkpoint.load_model_checkpoint
            model: ${model}
            path: ${evaluation.checkpoint}

    Parameters
    ----------
    model : torch.nn.Module
        Freshly instantiated model matching the checkpointed architecture.
    path : str or pathlib.Path
        Checkpoint file written by `Checkpoint` (a ``torch.save`` payload
        holding ``model_state_dict``).
    strict : bool, optional
        Forwarded to ``load_state_dict``; keep True so architecture mismatches
        fail loudly instead of silently evaluating partial weights.

    Returns
    -------
    torch.nn.Module
        The same model instance with restored weights.
    """

    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    # Checkpoints are trusted local artifacts written by `Checkpoint` above;
    # the payload holds more than tensors (step, metrics), so weights_only
    # deserialization is not applicable.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{checkpoint_path} is not a Checkpoint payload (missing model_state_dict)"
        )
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return model


__all__ = ["Checkpoint", "load_model_checkpoint"]
