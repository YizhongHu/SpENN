"""Regression tests for checkpoint save and train-resume failure boundaries."""

from __future__ import annotations

import io
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import (
    CheckpointPayload,
    TrainResume,
    restore_checkpoint,
    save_checkpoint,
)
from tpen.checkpoint.hashing import file_sha256
from tpen.checkpoint.rng import DEVICE_KEY
from tpen.sampling import MetropolisSampler
from tpen.training import VMCTrainer


@dataclass(frozen=True)
class _SaveBoundarySnapshot:
    """Exact filesystem state relevant to save validation admission."""

    root_exists: bool
    root_directories: tuple[str, ...]
    root_files: tuple[tuple[str, bytes], ...]
    temporary_directories: tuple[str, ...]
    complete_directories: tuple[str, ...]
    catalog_bytes: bytes | None
    latest_bytes: bytes | None


def _save_boundary_snapshot(root: Path) -> _SaveBoundarySnapshot:
    if not root.exists():
        return _SaveBoundarySnapshot(
            root_exists=False,
            root_directories=(),
            root_files=(),
            temporary_directories=(),
            complete_directories=(),
            catalog_bytes=None,
            latest_bytes=None,
        )

    return _SaveBoundarySnapshot(
        root_exists=True,
        root_directories=tuple(
            sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_dir())
        ),
        root_files=tuple(
            (str(path.relative_to(root)), path.read_bytes())
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ),
        temporary_directories=tuple(
            sorted(path.name for path in root.glob("*.tmp") if path.is_dir())
        ),
        complete_directories=tuple(
            sorted(path.name for path in root.glob("step_*") if path.is_dir())
        ),
        catalog_bytes=_optional_bytes(root / "publications.jsonl"),
        latest_bytes=_optional_bytes(root / "latest.json"),
    )


def _optional_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.is_file() else None


def _context() -> SimpleNamespace:
    """Return the same tiny resolved config for save and restore calls."""

    cfg = OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"_target_": "tests.VMCTrainer", "max_steps": 1},
            "sampler": {"_target_": "tests.MetropolisSampler", "n_steps": 1},
            "hamiltonian_terms": {"constant": {"_target_": "tests.ConstantHamiltonian"}},
            "run": {"run_id": "failure-protocol", "dir": "run"},
            "study": {"name": "unit", "config_id": "failure-protocol"},
        }
    )
    return SimpleNamespace(
        cfg=cfg,
        metadata=SimpleNamespace(
            run_id="failure-protocol",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("run"),
    )


def test_noncanonical_public_payload_is_rejected_before_any_save_surface_mutates(
    tmp_path: Path,
) -> None:
    """A constructible public payload cannot create even an unpublished root."""

    output_dir = tmp_path / "checkpoints"
    payload = CheckpointPayload(
        profile="train_resume",
        required_files=TrainResume().required_files,
        required_state=("completed_updates", "next_iteration"),
        restore_intents=("model_only", "train_resume"),
    )
    before = _save_boundary_snapshot(output_dir)

    with pytest.raises(ValueError, match="not canonical"):
        save_checkpoint(
            output_dir=output_dir,
            next_iteration=2,
            completed_updates=2,
            model=torch.nn.Linear(2, 1).double(),
            context=_context(),
            payload=payload,
        )

    assert _save_boundary_snapshot(output_dir) == before


@dataclass(frozen=True)
class _LateComponentFailure:
    """One concrete late-component preflight failure and its public error."""

    component: str
    kind: str
    exception_type: type[Exception]
    match: str


_LATE_COMPONENT_FAILURES = (
    _LateComponentFailure("optimizer", "missing", FileNotFoundError, "missing checkpoint file"),
    _LateComponentFailure("sampler", "corrupt", ValueError, "sampler checkpoint file digest mismatch"),
    _LateComponentFailure("trainer", "malformed", ValueError, "completed_updates"),
    _LateComponentFailure("rng", "incompatible", ValueError, "backend"),
)


@dataclass(frozen=True)
class _ConsumerState:
    """Frozen pre-call snapshot of every live train-resume consumer."""

    model: tuple[tuple[str, torch.Tensor], ...]
    optimizer: bytes
    trainer: tuple[int, int]
    sampler: bytes
    torch_rng: torch.Tensor
    python_rng: bytes
    numpy_rng: bytes


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _consumer_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: VMCTrainer,
    sampler: MetropolisSampler,
) -> _ConsumerState:
    return _ConsumerState(
        model=tuple(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        ),
        optimizer=_torch_bytes(optimizer.state_dict()),
        trainer=(trainer.next_iteration, trainer.completed_updates),
        sampler=_torch_bytes(sampler.mcmc_state_dict()),
        torch_rng=torch.get_rng_state().clone(),
        python_rng=pickle.dumps(random.getstate()),
        numpy_rng=pickle.dumps(np.random.get_state()),
    )


def _assert_consumer_state_unchanged(
    before: _ConsumerState,
    after: _ConsumerState,
) -> None:
    assert before.optimizer == after.optimizer
    assert before.trainer == after.trainer
    assert before.sampler == after.sampler
    assert torch.equal(before.torch_rng, after.torch_rng)
    assert before.python_rng == after.python_rng
    assert before.numpy_rng == after.numpy_rng
    assert len(before.model) == len(after.model)
    for (before_name, before_value), (after_name, after_value) in zip(
        before.model, after.model, strict=True
    ):
        assert before_name == after_name
        assert torch.equal(before_value, after_value)


def _refresh_component_digest(checkpoint_dir: Path, component: str) -> None:
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_path = checkpoint_dir / manifest["files"][component]
    manifest["hashes"][f"{component}_sha256"] = file_sha256(component_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _apply_late_component_failure(
    checkpoint_dir: Path,
    failure: _LateComponentFailure,
) -> None:
    manifest = json.loads(
        (checkpoint_dir / "manifest.json").read_text(encoding="utf-8")
    )
    component_path = checkpoint_dir / manifest["files"][failure.component]
    if failure.kind == "missing":
        component_path.unlink()
        return
    if failure.kind == "corrupt":
        component_path.write_bytes(b"corrupt checkpoint bytes")
        return
    if failure.kind == "malformed":
        trainer_state = json.loads(component_path.read_text(encoding="utf-8"))
        trainer_state["completed_updates"] = -1
        component_path.write_text(json.dumps(trainer_state), encoding="utf-8")
        _refresh_component_digest(checkpoint_dir, failure.component)
        return
    if failure.kind == "incompatible":
        rng_state = torch.load(component_path, map_location="cpu", weights_only=False)
        rng_state[DEVICE_KEY] = "meta"
        torch.save(rng_state, component_path)
        _refresh_component_digest(checkpoint_dir, failure.component)
        return
    raise AssertionError(f"unhandled test failure kind: {failure.kind}")


@pytest.mark.parametrize(
    "failure",
    _LATE_COMPONENT_FAILURES,
    ids=lambda failure: f"{failure.kind}-{failure.component}",
)
def test_train_resume_preflight_refusal_preserves_every_consumer(
    tmp_path: Path,
    failure: _LateComponentFailure,
) -> None:
    """Late TrainResume refusal leaves model, consumers, and RNG untouched."""

    context = _context()
    source_model = torch.nn.Linear(2, 1).double()
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    (source_model.weight.sum() + source_model.bias.sum()).backward()
    source_optimizer.step()
    source_optimizer.zero_grad()
    source_trainer = VMCTrainer(max_steps=1)
    source_trainer.next_iteration = 2
    source_trainer.completed_updates = 2
    source_sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=7,
        dtype=torch.float64,
    )
    source_sampler.initialize(device="cpu")
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=2,
        completed_updates=2,
        model=source_model,
        context=context,
        optimizer=source_optimizer,
        trainer=source_trainer,
        sampler=source_sampler,
        payload=TrainResume(),
    )
    _apply_late_component_failure(checkpoint_dir, failure)

    target_model = torch.nn.Linear(2, 1).double()
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    (target_model.weight.sum() + target_model.bias.sum()).backward()
    target_optimizer.step()
    target_optimizer.zero_grad()
    target_trainer = VMCTrainer(max_steps=1)
    target_trainer.next_iteration = 9
    target_trainer.completed_updates = 8
    target_sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=11,
        dtype=torch.float64,
    )
    target_sampler.initialize(device="cpu")
    before = _consumer_snapshot(
        target_model, target_optimizer, target_trainer, target_sampler
    )

    with pytest.raises(failure.exception_type, match=failure.match):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(checkpoint_dir)},
            model=target_model,
            context=context,
            optimizer=target_optimizer,
            trainer=target_trainer,
            sampler=target_sampler,
        )

    after = _consumer_snapshot(
        target_model, target_optimizer, target_trainer, target_sampler
    )
    _assert_consumer_state_unchanged(before, after)
