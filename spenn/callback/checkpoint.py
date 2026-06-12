"""Training checkpoint callback and checkpoint restore helper.

Checkpoint schema (``schema_version`` 1)
----------------------------------------

A checkpoint is a structured ``torch.save`` payload, never a pickled model
object. New checkpoints carry the model weights plus the provenance needed to
re-evaluate them without guessing architecture from state-dict keys::

    {
        "schema_version": 1,
        "kind": "spenn.model_checkpoint",

        "step": int,
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "sampler_mcmc_state": ...,
        "metrics": {...},

        "model_config": resolved model component spec (plain container),
        "model_config_hash": sha256 of the canonical model_config JSON,
        "resolved_config_hash": sha256 of the full resolved run config,
        "config_id": study.config_id,

        "runtime": {"device": str, "dtype": str},
        "git": {"sha": str, "branch": str, "dirty": bool},
        "versions": {"python": str, "torch": str, "spenn": str},
    }

Legacy checkpoints (written before ``schema_version`` existed) hold only the
state dicts; they can still be restored, but only into an explicitly
configured model — architecture is never inferred from checkpoint keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .base import Callback, Event

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "spenn.model_checkpoint"


def model_config_hash(config: Any) -> str:
    """Canonical sha256 hash of a resolved config container.

    Hashes the canonical JSON encoding (sorted keys, compact separators) of a
    plain config container, so the same resolved config always hashes the same
    regardless of key order or OmegaConf wrapping. Used for both
    ``model_config_hash`` and ``resolved_config_hash`` in checkpoint payloads,
    and by evaluation tooling to pair checkpoints with eval configs.

    Parameters
    ----------
    config : mapping, sequence, or omegaconf container
        Resolved configuration. OmegaConf containers are resolved and
        converted; plain containers are hashed as-is.

    Returns
    -------
    str
        Hex sha256 digest of the canonical JSON encoding.
    """

    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_provenance(context: Any) -> dict[str, Any]:
    """Provenance fields for a checkpoint payload from the run context.

    Every field degrades to ``None`` when the context (or a piece of it) is
    unavailable, so checkpointing never fails on missing provenance.
    """

    cfg = getattr(context, "cfg", None)
    metadata = getattr(context, "metadata", None)

    resolved_config: dict[str, Any] | None = None
    if OmegaConf.is_config(cfg):
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        resolved_config = cfg
    model_config = resolved_config.get("model") if resolved_config else None
    study = (resolved_config.get("study") if resolved_config else None) or {}

    extra = getattr(metadata, "extra", None) or {}
    from spenn import __version__ as spenn_version

    return {
        "model_config": model_config,
        "model_config_hash": None if model_config is None else model_config_hash(model_config),
        "resolved_config_hash": (
            None if resolved_config is None else model_config_hash(resolved_config)
        ),
        "config_id": study.get("config_id"),
        "runtime": {
            "device": getattr(metadata, "device", None),
            "dtype": getattr(metadata, "dtype", None),
        },
        "git": {
            "sha": getattr(metadata, "git_commit", None),
            "branch": getattr(metadata, "git_branch", None),
            "dirty": getattr(metadata, "dirty_worktree", None),
        },
        "versions": {
            "python": extra.get("python_version"),
            "torch": extra.get("torch_version"),
            "spenn": spenn_version,
        },
    }


class Checkpoint(Callback):
    """Write training checkpoints from the loop `TrainerState`.

    Reads ``event.state`` (a `spenn.training.state.TrainerState`) and writes a
    structured ``torch.save`` payload (see the module docstring for the
    schema) to ``output_dir/step_<step>.pt`` and ``output_dir/latest.pt``.
    Model config and provenance come from ``event.context``; training state
    (weights, optimizer, sampler MCMC state) comes from ``event.state``.

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
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "step": state.step,
            "model_state_dict": state.model.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict(),
            "sampler_mcmc_state": sampler_mcmc_state() if callable(sampler_mcmc_state) else None,
            "metrics": state.metrics,
            **_checkpoint_provenance(event.context),
        }
        payload["versions"]["torch"] = payload["versions"]["torch"] or torch.__version__
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, self.output_dir / f"step_{state.step}.pt")
        torch.save(payload, self.output_dir / "latest.pt")


def load_model_checkpoint(
    model,
    path: str | Path,
    strict: bool = True,
    expected_model_config_hash: str | None = None,
    allow_model_config_mismatch: bool = False,
):
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
            strict: ${evaluation.checkpoint_strict}
            expected_model_config_hash: ${evaluation.expected_model_config_hash}

    Parameters
    ----------
    model : torch.nn.Module
        Freshly instantiated model matching the checkpointed architecture.
        Architecture always comes from explicit configuration, never from
        checkpoint keys.
    path : str or pathlib.Path
        Checkpoint file written by `Checkpoint` (a ``torch.save`` payload
        holding ``model_state_dict``).
    strict : bool, optional
        Forwarded to ``load_state_dict``; keep True so missing or unexpected
        keys fail loudly instead of silently evaluating partial weights.
    expected_model_config_hash : str or None, optional
        When set, the checkpoint's stored ``model_config_hash`` must match it.
        A structured checkpoint with a different hash, or a legacy checkpoint
        without model_config metadata, fails loudly.
    allow_model_config_mismatch : bool, optional
        Explicit escape hatch that downgrades the hash check to a no-op.
        Canonical benchmark configs must not set this.

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

    if expected_model_config_hash and not allow_model_config_mismatch:
        stored_hash = payload.get("model_config_hash")
        if stored_hash is None:
            raise ValueError(
                f"{checkpoint_path}: this checkpoint does not contain model_config "
                "metadata. Provide an explicit model config in the evaluation config "
                "or retrain with the new checkpoint schema."
            )
        if stored_hash != expected_model_config_hash:
            raise ValueError(
                f"{checkpoint_path}: model_config_hash mismatch "
                f"(checkpoint {stored_hash}, expected {expected_model_config_hash}); "
                "the checkpoint was trained with a different model config. Set "
                "allow_model_config_mismatch only as an explicit escape hatch."
            )

    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return model


__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "Checkpoint",
    "load_model_checkpoint",
    "model_config_hash",
]
