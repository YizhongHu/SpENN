"""Checkpoint restore modes."""

from __future__ import annotations

import inspect
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
    """Summary of the state restored from a checkpoint.

    Parameters
    ----------
    mode : str
        Restore mode that produced this report.
    checkpoint_dir : str or None, optional
        Resolved checkpoint step directory, or ``None`` for ``mode="none"``.
    schema_version : int or None, optional
        Manifest schema version that was read.
    next_iteration : int or None, optional
        Resume cursor recorded by the manifest: the iteration a resumed run
        continues from. ``None`` for ``mode="none"``.
    completed_updates : int or None, optional
        Applied optimizer updates recorded by the manifest. ``None`` for
        ``mode="none"`` and for a v1 manifest under ``model_only``, which never
        recorded the counter. Always populated under ``train_resume``, which
        only admits v2.
    loaded_model, loaded_optimizer, loaded_trainer, loaded_sampler, loaded_rng : bool, optional
        Which components were restored.
    """

    mode: str
    checkpoint_dir: str | None = None
    schema_version: int | None = None
    next_iteration: int | None = None
    completed_updates: int | None = None
    loaded_model: bool = False
    loaded_optimizer: bool = False
    loaded_trainer: bool = False
    loaded_sampler: bool = False
    loaded_rng: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report mapping."""

        return {
            "mode": self.mode,
            "checkpoint_dir": self.checkpoint_dir,
            "schema_version": self.schema_version,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "loaded_model": self.loaded_model,
            "loaded_optimizer": self.loaded_optimizer,
            "loaded_trainer": self.loaded_trainer,
            "loaded_sampler": self.loaded_sampler,
            "loaded_rng": self.loaded_rng,
        }


def restore_checkpoint(
    *,
    load: Any,
    model: Any,
    context: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
    mode: str | None = None,
    strict: bool | None = None,
    allow_protocol_mismatch: bool | None = None,
) -> RestoreReport:
    """Restore checkpoint state into explicitly configured objects."""

    config = _load_config(load)
    mode = str(mode or config.get("mode", "none"))
    if mode not in RESTORE_MODES:
        raise ValueError(f"load.mode must be one of {RESTORE_MODES}, got {mode!r}")
    if mode == "none":
        return RestoreReport(mode="none")

    path = config.get("path")
    if path in (None, ""):
        raise ValueError(f"load.path is required for mode={mode!r}")
    strict_load = bool(config.get("strict", True) if strict is None else strict)
    allow_mismatch = bool(
        config.get("allow_protocol_mismatch", False)
        if allow_protocol_mismatch is None
        else allow_protocol_mismatch
    )

    checkpoint_dir = resolve_checkpoint_dir(path)
    # Schema acceptance is mode-dependent, so the mode is decided before the
    # manifest is read: a v1 artifact is refused for `train_resume` at the gate.
    manifest = read_manifest(checkpoint_dir / "manifest.json", mode=mode)
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
            mode=mode,
            checkpoint_dir=str(checkpoint_dir),
            schema_version=manifest.schema_version,
            next_iteration=manifest.next_iteration,
            # `None` for a v1 manifest, which never recorded the counter.
            completed_updates=manifest.completed_updates,
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
    _load_sampler(checkpoint_dir, manifest.files, sampler, context)
    _load_rng(checkpoint_dir, manifest.files)
    return RestoreReport(
        mode=mode,
        checkpoint_dir=str(checkpoint_dir),
        schema_version=manifest.schema_version,
        next_iteration=manifest.next_iteration,
        # The schema gate admits only v2 here, so both counters are real.
        completed_updates=manifest.completed_updates,
        loaded_model=True,
        loaded_optimizer=True,
        loaded_trainer=True,
        loaded_sampler=True,
        loaded_rng=True,
    )


def restore_checkpoint_with_events(
    *,
    load: Any,
    model: Any,
    context: Any,
    emit: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
) -> RestoreReport:
    """Restore a checkpoint while emitting durable load lifecycle events."""

    config = _load_config(load)
    mode = str(config.get("mode", "none"))
    if mode == "none":
        return RestoreReport(mode="none")
    path = config.get("path")
    strict = bool(config.get("strict", True))
    emit(
        "load_start",
        context,
        payload={
            "path": path,
            "mode": mode,
            "strict": strict,
        },
    )
    try:
        report = restore_checkpoint(
            load=load,
            model=model,
            context=context,
            optimizer=optimizer,
            trainer=trainer,
            sampler=sampler,
        )
    except Exception as exc:
        setattr(exc, "_spenn_failure_phase", "load")
        setattr(exc, "_spenn_load_path", path)
        setattr(exc, "_spenn_load_mode", mode)
        emit(
            "load_failed",
            context,
            payload={
                "path": path,
                "mode": mode,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise

    emit(
        "load_success",
        context,
        payload={
            "path": path,
            "resolved_checkpoint_dir": report.checkpoint_dir,
            "schema_version": report.schema_version,
            "next_iteration": report.next_iteration,
            "completed_updates": report.completed_updates,
            "loaded_model": report.loaded_model,
            "loaded_optimizer": report.loaded_optimizer,
            "loaded_trainer": report.loaded_trainer,
            "loaded_sampler": report.loaded_sampler,
            "loaded_rng": report.loaded_rng,
        },
    )
    return report


def _load_config(load: Any) -> dict[str, Any]:
    if load is None:
        return {"mode": "none"}
    if OmegaConf.is_config(load):
        return dict(OmegaConf.to_container(load, resolve=True))
    if isinstance(load, dict):
        return dict(load)
    raise TypeError("load config must be a mapping or OmegaConf container")


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


def _load_sampler(checkpoint_dir: Path, files: dict[str, str], sampler: Any, context: Any) -> None:
    import torch

    if sampler is None:
        raise ValueError("train_resume restore requires a sampler")
    load_state = getattr(sampler, "load_mcmc_state_dict", None)
    if not callable(load_state):
        raise TypeError("sampler must expose load_mcmc_state_dict() for train_resume restore")
    path = _required_file(checkpoint_dir, files, "sampler")
    state = torch.load(path, map_location="cpu", weights_only=False)
    target_device = getattr(getattr(context, "metadata", None), "device", None)
    signature = inspect.signature(load_state)
    if "device" in signature.parameters:
        load_state(state, device=target_device)
    else:
        load_state(state)


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
    expected_torch_device = _canonical_runtime_device(expected_device)
    expected_dtype = getattr(torch, str(expected_dtype_name))
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if _canonical_runtime_device(tensor.device) != expected_torch_device:
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} on {tensor.device}, "
                f"expected {expected_device}"
            )
        if tensor.is_floating_point() and tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"checkpoint restore left model tensor {name!r} with dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )


def _canonical_runtime_device(device: Any) -> Any:
    """Return a comparable torch device for runtime checks."""

    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return resolved
