"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import tpen
from tpen.callback import Checkpoint, Event
from tpen.checkpoint import checkpoint_hashes
from tpen.training.state import TrainerState


def _state(step: int, *, completed_steps: int | None = None) -> TrainerState:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = _Trainer(completed_steps=step + 1 if completed_steps is None else completed_steps)
    return TrainerState(
        step=step,
        metrics={"loss": 0.5, "energy": 1.25},
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=_SamplerWithMCMCState(),
    )


class _Trainer:
    def __init__(self, *, completed_steps: int) -> None:
        self.completed_steps = int(completed_steps)

    def state_dict(self) -> dict[str, int]:
        return {"global_step": self.completed_steps}


class _SamplerWithMCMCState:
    def mcmc_state_dict(self) -> dict:
        return {"has_burned_in": True}


def _event(state: TrainerState, *, context=None, name: str = "step_end") -> Event:
    return Event(
        name=name,
        context=_context() if context is None else context,
        state=state,
        payload={"step": state.step},
    )


def test_checkpoint_writes_step_directory_and_latest_pointer(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path / "checkpoints", every_n_steps=1)

    callback.handle(_event(_state(2)))

    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "step_000003"
    assert step_dir.is_dir()
    assert (step_dir / "manifest.json").exists()
    assert (step_dir / "resolved_config.yaml").exists()
    assert (step_dir / "model.pt").exists()
    assert (step_dir / "optimizer.pt").exists()
    assert (step_dir / "trainer.json").exists()
    assert (step_dir / "sampler.pt").exists()
    assert (step_dir / "rng.pt").exists()
    assert (step_dir / "COMPLETE").exists()
    assert (ckpt_dir / "latest.json").exists()
    assert not (ckpt_dir / "step_000003.tmp").exists()


def test_checkpoint_payload_contains_expected_keys(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(3)

    callback.handle(_event(state))

    manifest = torch.load(tmp_path / "step_000004" / "model.pt", weights_only=False)
    assert set(manifest) == set(state.model.state_dict())
    sampler_state = torch.load(tmp_path / "step_000004" / "sampler.pt", weights_only=False)
    assert sampler_state == {"has_burned_in": True}


def test_checkpoint_respects_every_n_steps_filter(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)

    callback.handle(_event(_state(0)))
    assert not (tmp_path / "step_000001").exists()

    callback.handle(_event(_state(1)))
    assert (tmp_path / "step_000002").exists()


def test_checkpoint_cadence_counts_completed_updates(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=5)

    callback.handle(_event(_state(4, completed_steps=5)))

    assert (tmp_path / "step_000005").exists()


def test_checkpoint_writes_train_end_checkpoint_without_step_cadence(tmp_path) -> None:
    callback = Checkpoint(triggers=["train_end"], output_dir=tmp_path)
    state = _state(3, completed_steps=4)

    callback.handle(_event(state, name="train_end"))

    assert (tmp_path / "step_000004" / "COMPLETE").exists()
    assert (tmp_path / "latest.json").exists()


def test_checkpoint_train_end_skips_existing_complete_checkpoint(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end", "train_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(1, completed_steps=2)

    callback.handle(_event(state))
    callback.handle(_event(state, name="train_end"))

    assert sorted(path.name for path in tmp_path.glob("step_*")) == ["step_000002"]


def test_train_end_updates_latest_when_cadence_misses_terminal_step(tmp_path) -> None:
    periodic = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)
    terminal = Checkpoint(triggers=["train_end"], output_dir=tmp_path)

    periodic.handle(_event(_state(1, completed_steps=2)))
    final_state = _state(2, completed_steps=3)
    periodic.handle(_event(final_state))
    terminal.handle(_event(final_state, name="train_end"))

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000002",
        "step_000003",
    ]
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["checkpoint_dir"] == "step_000003"
    assert latest["step"] == 3


def _context() -> SimpleNamespace:
    """Minimal RunContext stand-in carrying resolved config and metadata."""

    cfg = OmegaConf.create(
        {
            "study": {"name": "test_study", "config_id": "lr=0.001_channels=4"},
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )
    metadata = SimpleNamespace(
        run_id="run",
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="main",
        dirty_worktree=False,
        command="pytest",
        extra={"python_version": "3.12.0", "torch_version": torch.__version__},
    )
    return SimpleNamespace(cfg=cfg, metadata=metadata, run_dir="/tmp/run")


def test_checkpoint_payload_uses_structured_schema(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    context = _context()
    state = _state(1)

    callback.handle(Event(name="step_end", context=context, state=state, payload={"step": 1}))

    manifest = json.loads((tmp_path / "step_000002" / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "spenn.checkpoint"
    assert manifest["files"]["model"] == "model.pt"
    assert manifest["hashes"] == checkpoint_hashes(context.cfg)
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["dtype"] == "float64"
    assert manifest["runtime"]["torch_version"] == torch.__version__
    assert manifest["provenance"]["config_id"] == "lr=0.001_channels=4"
    assert manifest["provenance"]["study_name"] == "test_study"
    assert manifest["provenance"]["git_sha"] == "deadbeef"
    assert manifest["provenance"]["spenn_version"] == tpen.__version__


def test_checkpoint_fails_loudly_when_required_state_is_missing(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(1)
    state.trainer = None

    with pytest.raises(ValueError, match="trainer"):
        callback.handle(_event(state))
