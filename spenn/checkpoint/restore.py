"""Checkpoint restore modes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .artifact import resolve_checkpoint_dir
from .hashing import checkpoint_hashes
from .schema import read_manifest

RESTORE_MODES = ("none", "model_only", "train_resume")


@dataclass(frozen=True)
class RestoreReport:
    """Summary of the state restored from a checkpoint."""

    restore_mode: str
    checkpoint_dir: str | None = None
    step: int | None = None
    loaded_model: bool = False
    loaded_optimizer: bool = False
    loaded_trainer: bool = False
    loaded_sampler: bool = False
    loaded_rng: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report mapping."""

        return {
            "restore_mode": self.restore_mode,
            "checkpoint_dir": self.checkpoint_dir,
            "step": self.step,
            "loaded_model": self.loaded_model,
            "loaded_optimizer": self.loaded_optimizer,
            "loaded_trainer": self.loaded_trainer,
            "loaded_sampler": self.loaded_sampler,
            "loaded_rng": self.loaded_rng,
        }


def restore_checkpoint(
    *,
    checkpoint: Any,
    model: Any,
    context: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
    restore_mode: str | None = None,
    strict: bool | None = None,
    allow_protocol_mismatch: bool | None = None,
) -> RestoreReport:
    """Restore checkpoint state into explicitly configured objects."""

    config = _restore_config(checkpoint)
    mode = str(restore_mode or config.get("restore_mode", "none"))
    if mode not in RESTORE_MODES:
        raise ValueError(f"checkpoint.restore_mode must be one of {RESTORE_MODES}, got {mode!r}")
    if mode == "none":
        return RestoreReport(restore_mode="none")

    path = config.get("path")
    if path in (None, ""):
        raise ValueError(f"checkpoint.path is required for restore_mode={mode!r}")
    strict_load = bool(config.get("strict", True) if strict is None else strict)
    allow_mismatch = bool(
        config.get("allow_protocol_mismatch", False)
        if allow_protocol_mismatch is None
        else allow_protocol_mismatch
    )

    checkpoint_dir = resolve_checkpoint_dir(path)
    manifest = read_manifest(checkpoint_dir / "manifest.json")
    current_hashes = checkpoint_hashes(getattr(context, "cfg", {}))

    _verify_hash(manifest.hashes, current_hashes, "model_config", checkpoint_dir)
    if mode == "model_only":
        _verify_hash(
            manifest.hashes,
            current_hashes,
            "hamiltonian_config",
            checkpoint_dir,
            allow_mismatch=allow_mismatch,
        )
        _load_model(checkpoint_dir, manifest.files, model, strict=strict_load, context=context)
        return RestoreReport(
            restore_mode=mode,
            checkpoint_dir=str(checkpoint_dir),
            step=manifest.step,
            loaded_model=True,
        )

    for hash_name in (
        "optimizer_config",
        "trainer_config",
        "sampler_config",
        "hamiltonian_config",
    ):
        _verify_hash(
            manifest.hashes,
            current_hashes,
            hash_name,
            checkpoint_dir,
            allow_mismatch=(hash_name == "hamiltonian_config" and allow_mismatch),
        )

    _load_model(checkpoint_dir, manifest.files, model, strict=strict_load, context=context)
    _load_optimizer(checkpoint_dir, manifest.files, optimizer)
    _load_trainer(checkpoint_dir, manifest.files, trainer)
    _load_sampler(checkpoint_dir, manifest.files, sampler)
    _load_rng(checkpoint_dir, manifest.files)
    return RestoreReport(
        restore_mode=mode,
        checkpoint_dir=str(checkpoint_dir),
        step=manifest.step,
        loaded_model=True,
        loaded_optimizer=True,
        loaded_trainer=True,
        loaded_sampler=True,
        loaded_rng=True,
    )


def _restore_config(checkpoint: Any) -> dict[str, Any]:
    if checkpoint is None:
        return {"restore_mode": "none"}
    if OmegaConf.is_config(checkpoint):
        return dict(OmegaConf.to_container(checkpoint, resolve=True))
    if isinstance(checkpoint, dict):
        return dict(checkpoint)
    raise TypeError("checkpoint config must be a mapping or OmegaConf container")


def _verify_hash(
    stored: dict[str, str | None],
    current: dict[str, str | None],
    name: str,
    checkpoint_dir: Path,
    *,
    allow_mismatch: bool = False,
) -> None:
    stored_hash = stored.get(name)
    current_hash = current.get(name)
    if stored_hash is None:
        raise ValueError(f"{checkpoint_dir}: manifest missing {name}")
    if current_hash is None:
        raise ValueError(f"current config is missing {name.removesuffix('_config')} for restore")
    if stored_hash != current_hash and not allow_mismatch:
        raise ValueError(
            f"{checkpoint_dir}: {name} mismatch "
            f"(checkpoint {stored_hash}, current {current_hash})"
        )


def _load_model(
    checkpoint_dir: Path,
    files: dict[str, str],
    model: Any,
    *,
    strict: bool,
    context: Any,
) -> None:
    import torch

    path = _required_file(checkpoint_dir, files, "model")
    map_location = getattr(getattr(context, "metadata", None), "device", "cpu")
    state_dict = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state_dict, strict=strict)
    _assert_model_runtime(model, context)


def _load_optimizer(checkpoint_dir: Path, files: dict[str, str], optimizer: Any) -> None:
    import torch

    if optimizer is None:
        raise ValueError("train_resume restore requires an optimizer")
    path = _required_file(checkpoint_dir, files, "optimizer")
    optimizer.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))


def _load_trainer(checkpoint_dir: Path, files: dict[str, str], trainer: Any) -> None:
    if trainer is None:
        raise ValueError("train_resume restore requires a trainer")
    load_state_dict = getattr(trainer, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("trainer must expose load_state_dict() for train_resume restore")
    path = _required_file(checkpoint_dir, files, "trainer")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    load_state_dict(state)


def _load_sampler(checkpoint_dir: Path, files: dict[str, str], sampler: Any) -> None:
    import torch

    if sampler is None:
        raise ValueError("train_resume restore requires a sampler")
    load_state = getattr(sampler, "load_mcmc_state_dict", None)
    if not callable(load_state):
        raise TypeError("sampler must expose load_mcmc_state_dict() for train_resume restore")
    path = _required_file(checkpoint_dir, files, "sampler")
    load_state(torch.load(path, map_location="cpu", weights_only=False))


def _load_rng(checkpoint_dir: Path, files: dict[str, str]) -> None:
    import torch

    path = _required_file(checkpoint_dir, files, "rng")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    cuda = getattr(torch, "cuda", None)
    if cuda_states is not None and cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            cuda.set_rng_state_all(cuda_states)
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.set_state(state["numpy"])


def _required_file(checkpoint_dir: Path, files: dict[str, str], key: str) -> Path:
    relative = files.get(key)
    if not relative:
        raise FileNotFoundError(f"{checkpoint_dir}: checkpoint manifest lacks file entry {key!r}")
    path = checkpoint_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"{checkpoint_dir}: missing checkpoint file {relative}")
    return path


def _assert_model_runtime(model: Any, context: Any) -> None:
    import torch

    metadata = getattr(context, "metadata", None)
    expected_device = getattr(metadata, "device", None)
    expected_dtype_name = getattr(metadata, "dtype", None)
    if expected_device is None or expected_dtype_name is None:
        return
    expected_dtype = getattr(torch, str(expected_dtype_name))
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device != torch.device(expected_device):
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} on {tensor.device}, "
                f"expected {expected_device}"
            )
        if tensor.is_floating_point() and tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} with dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )
