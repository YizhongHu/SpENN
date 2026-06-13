"""Structured checkpoint saving."""

from __future__ import annotations

import random
import shutil
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from spenn import __version__ as spenn_version

from .artifact import checkpoint_step_dir_name, prune_old_checkpoints, write_latest
from .hashing import checkpoint_hashes
from .manifest import CHECKPOINT_KIND, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest


def save_checkpoint(
    *,
    output_dir: str | Path,
    step: int,
    model: Any,
    context: Any,
    optimizer: Any | None = None,
    trainer: Any | None = None,
    sampler: Any | None = None,
    save_optimizer: bool = True,
    save_trainer: bool = True,
    save_sampler: bool = True,
    save_rng: bool = True,
    keep_last: int | None = None,
) -> Path:
    """Write one complete directory checkpoint and update ``latest.json``."""

    import torch

    cfg = _require_config(context)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    created_at = time.time()
    final_dir = root / checkpoint_step_dir_name(step)
    tmp_dir = root / f"{final_dir.name}.tmp"
    if final_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {final_dir}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    files: dict[str, str] = {}
    try:
        _write_resolved_config(tmp_dir / "resolved_config.yaml", cfg)
        files["resolved_config"] = "resolved_config.yaml"

        torch.save(model.state_dict(), tmp_dir / "model.pt")
        files["model"] = "model.pt"

        if save_optimizer:
            if optimizer is None:
                raise ValueError("save_optimizer=True requires optimizer in the checkpoint event")
            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            files["optimizer"] = "optimizer.pt"

        if save_trainer:
            trainer_state = _state_dict_from(trainer, "trainer")
            _write_json_mapping(tmp_dir / "trainer.json", trainer_state)
            files["trainer"] = "trainer.json"

        if save_sampler:
            sampler_state = _sampler_state_dict(sampler)
            torch.save(sampler_state, tmp_dir / "sampler.pt")
            files["sampler"] = "sampler.pt"

        if save_rng:
            torch.save(_rng_state_dict(), tmp_dir / "rng.pt")
            files["rng"] = "rng.pt"

        manifest = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            kind=CHECKPOINT_KIND,
            step=int(step),
            created_at_unix=created_at,
            files=files,
            hashes=checkpoint_hashes(cfg),
            runtime=_runtime_metadata(context),
            provenance=_provenance_metadata(context),
        )
        manifest.write(tmp_dir / "manifest.json")
        (tmp_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
        tmp_dir.rename(final_dir)
        write_latest(root, final_dir, step=int(step), created_at_unix=created_at)
        prune_old_checkpoints(root, keep_last=keep_last)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return final_dir


def _require_config(context: Any) -> Any:
    cfg = getattr(context, "cfg", None)
    if cfg is None:
        raise ValueError("checkpoint saving requires event.context.cfg")
    return cfg


def _write_resolved_config(path: Path, cfg: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if OmegaConf.is_config(cfg):
        OmegaConf.save(config=cfg, f=path, resolve=True)
        return
    OmegaConf.save(config=OmegaConf.create(cfg), f=path, resolve=True)


def _write_json_mapping(path: Path, data: Mapping[str, Any]) -> None:
    from spenn.artifacts import write_json

    write_json(path, data)


def _state_dict_from(value: Any, owner: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"save_{owner}=True requires {owner} in the checkpoint event")
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{owner} must expose state_dict() for checkpoint saving")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError(f"{owner}.state_dict() must return a mapping")
    return state


def _sampler_state_dict(sampler: Any) -> Mapping[str, Any]:
    if sampler is None:
        raise ValueError("save_sampler=True requires sampler in the checkpoint event")
    state_dict = getattr(sampler, "mcmc_state_dict", None)
    if not callable(state_dict):
        raise TypeError("sampler must expose mcmc_state_dict() for checkpoint saving")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise TypeError("sampler.mcmc_state_dict() must return a mapping")
    return state


def _rng_state_dict() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        state["torch_cuda"] = cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        state["numpy"] = None
    else:
        state["numpy"] = np.random.get_state()
    return state


def _runtime_metadata(context: Any) -> dict[str, Any]:
    import torch

    metadata = getattr(context, "metadata", None)
    return {
        "dtype": getattr(metadata, "dtype", None),
        "device": getattr(metadata, "device", None),
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    }


def _provenance_metadata(context: Any) -> dict[str, Any]:
    cfg = _require_config(context)
    metadata = getattr(context, "metadata", None)
    extra = getattr(metadata, "extra", None) or {}
    study = OmegaConf.select(cfg, "study", default={}) or {}
    if OmegaConf.is_config(study):
        study = OmegaConf.to_container(study, resolve=True)
    if not isinstance(study, Mapping):
        study = {}
    return {
        "run_id": getattr(metadata, "run_id", None),
        "run_dir": str(getattr(context, "run_dir", "")),
        "config_id": study.get("config_id"),
        "study_name": study.get("name"),
        "git_sha": getattr(metadata, "git_commit", None),
        "git_branch": getattr(metadata, "git_branch", None),
        "git_dirty": getattr(metadata, "dirty_worktree", None),
        "command": getattr(metadata, "command", None),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "spenn_version": spenn_version,
        "slurm": extra.get("slurm", {}),
    }
