"""Regression tests for callback recovery across publication commit windows."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import RunContext
from tpen.callback import Checkpoint
from tpen.checkpoint import read_publications
from tpen.checkpoint.save import write_latest as imported_write_latest
import tpen.checkpoint.save as save_module
from tpen.events import Occurrence
from tpen.sampling import MetropolisSampler
from tpen.training import VMCTrainer
from tpen.training.events import TrainingCompleted
from tpen.training.state import TrainerState


class _CheckpointContext(RunContext):
    """Minimal real RunContext accepted by typed callback dispatch."""

    def __init__(self, cfg, metadata, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.metadata = metadata
        self._run_dir = Path(run_dir)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def __repr__(self) -> str:
        return f"_CheckpointContext(run_dir={self._run_dir!r})"


def _context(tmp_path: Path) -> _CheckpointContext:
    cfg = OmegaConf.create(
        {
            "study": {"name": "publication-recovery", "config_id": "unit"},
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )
    metadata = SimpleNamespace(
        run_id="publication-recovery",
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="codex/checkpoint",
        dirty_worktree=False,
        command="pytest",
        extra={"slurm": {}},
    )
    return _CheckpointContext(cfg, metadata, tmp_path / "run")


def _state() -> TrainerState:
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = VMCTrainer(max_steps=1)
    trainer.next_iteration = 2
    trainer.completed_updates = 2
    sampler = MetropolisSampler(
        n_walkers=2,
        burn_in=0,
        n_steps=1,
        n_electrons=1,
        spatial_dim=1,
        seed=7,
        dtype=torch.float64,
    )
    return TrainerState(
        step=1,
        metrics={},
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
    )


def _finish(callback: Checkpoint, state: TrainerState, context: _CheckpointContext) -> None:
    callback.handle_occurrence(
        Occurrence(event=TrainingCompleted(), count=1),
        context,
        state,
    )


@dataclass(frozen=True)
class _CheckpointSnapshot:
    """Byte snapshot proving callback retry does not rewrite the payload."""

    files: tuple[tuple[str, bytes], ...]


def _checkpoint_snapshot(checkpoint_dir: Path) -> _CheckpointSnapshot:
    return _CheckpointSnapshot(
        tuple(
            (str(path.relative_to(checkpoint_dir)), path.read_bytes())
            for path in sorted(checkpoint_dir.rglob("*"))
            if path.is_file()
        )
    )


@dataclass
class _FailOnceLatest:
    """Direct-reference fail-once fault at the catalog/latest boundary."""

    delegate: Callable[..., None]
    failures_remaining: int = 1

    def __call__(self, *args, **kwargs) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("latest pointer write failed once")
        self.delegate(*args, **kwargs)


def test_callback_retry_repairs_latest_after_catalog_commit_without_rewriting_payload(
    tmp_path: Path,
) -> None:
    """A fail-once latest update leaves one row and retry repairs the pointer."""

    root = tmp_path / "checkpoints"
    state = _state()
    context = _context(tmp_path)
    callback = Checkpoint(output_dir=root, periodic=False)
    original_write_latest = save_module.write_latest
    save_module.write_latest = _FailOnceLatest(imported_write_latest)
    try:
        with pytest.raises(OSError, match="latest pointer write failed once"):
            _finish(callback, state, context)
    finally:
        save_module.write_latest = original_write_latest

    final_dir = root / "step_000002"
    assert final_dir.is_dir()
    rows_before_retry = read_publications(root / "publications.jsonl")
    assert len(rows_before_retry) == 1
    assert rows_before_retry[0].checkpoint_dir == final_dir
    assert not (root / "latest.json").exists()
    payload_before = _checkpoint_snapshot(final_dir)

    _finish(callback, state, context)

    rows_after_retry = read_publications(root / "publications.jsonl")
    assert len(rows_after_retry) == 1
    assert rows_after_retry[0].checkpoint_dir == final_dir
    latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    assert latest["checkpoint_dir"] == final_dir.name
    assert _checkpoint_snapshot(final_dir) == payload_before
