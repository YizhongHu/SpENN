"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import spenn
from spenn.callback import Checkpoint, Event
from spenn.checkpoint import checkpoint_hashes
from spenn.training.state import TrainerState


def _state(step: int) -> TrainerState:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = _Trainer()
    return TrainerState(
        step=step,
        metrics={"loss": 0.5, "energy": 1.25},
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=_SamplerWithMCMCState(),
    )


class _Trainer:
    def state_dict(self) -> dict[str, int]:
        return {"global_step": 4}


class _SamplerWithMCMCState:
    def mcmc_state_dict(self) -> dict:
        return {"has_burned_in": True}


def _event(state: TrainerState, *, context=None) -> Event:
    return Event(
        name="step_end",
        context=_context() if context is None else context,
        state=state,
        payload={"step": state.step},
    )


def test_checkpoint_writes_step_directory_and_latest_pointer(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path / "checkpoints", every_n_steps=1)

    callback.handle(_event(_state(2)))

    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "step_000002"
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
    assert not (ckpt_dir / "step_000002.tmp").exists()


def test_checkpoint_payload_contains_expected_keys(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(3)

    callback.handle(_event(state))

    manifest = torch.load(tmp_path / "step_000003" / "model.pt", weights_only=False)
    assert set(manifest) == set(state.model.state_dict())
    sampler_state = torch.load(tmp_path / "step_000003" / "sampler.pt", weights_only=False)
    assert sampler_state == {"has_burned_in": True}


def test_checkpoint_respects_every_n_steps_filter(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)

    callback.handle(_event(_state(1)))
    assert not (tmp_path / "step_000001").exists()

    callback.handle(_event(_state(2)))
    assert (tmp_path / "step_000002").exists()


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

    import json

    manifest = json.loads((tmp_path / "step_000001" / "manifest.json").read_text())
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
    assert manifest["provenance"]["spenn_version"] == spenn.__version__


def test_checkpoint_fails_loudly_when_required_state_is_missing(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(1)
    state.trainer = None

    with pytest.raises(ValueError, match="trainer"):
        callback.handle(_event(state))
