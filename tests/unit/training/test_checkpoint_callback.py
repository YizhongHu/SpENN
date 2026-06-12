"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import spenn
from spenn.callback import Checkpoint, Event, model_config_hash
from spenn.training.state import TrainerState


def _state(step: int) -> TrainerState:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return TrainerState(
        step=step,
        metrics={"loss": 0.5, "energy": 1.25},
        model=model,
        optimizer=optimizer,
        sampler=None,
    )


def _event(state: TrainerState) -> Event:
    return Event(name="step_end", context=None, state=state, payload={"step": state.step})


def test_checkpoint_writes_step_and_latest_files(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path / "checkpoints", every_n_steps=1)

    callback.handle(_event(_state(2)))

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "step_2.pt").exists()
    assert (ckpt_dir / "latest.pt").exists()


def test_checkpoint_payload_contains_expected_keys(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(3)

    callback.handle(_event(state))

    payload = torch.load(tmp_path / "step_3.pt", weights_only=False)
    assert payload["step"] == 3
    assert set(payload["model_state_dict"]) == set(state.model.state_dict())
    assert "optimizer_state_dict" in payload
    assert payload["metrics"] == {"loss": 0.5, "energy": 1.25}
    assert payload["sampler_mcmc_state"] is None


def test_checkpoint_respects_every_n_steps_filter(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=2)

    callback.handle(_event(_state(1)))
    assert not (tmp_path / "step_1.pt").exists()

    callback.handle(_event(_state(2)))
    assert (tmp_path / "step_2.pt").exists()


def test_checkpoint_saves_sampler_mcmc_state_when_available(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    state = _state(1)

    class _SamplerWithMCMCState:
        def mcmc_state_dict(self) -> dict:
            return {"has_burned_in": True}

    state.sampler = _SamplerWithMCMCState()

    callback.handle(_event(state))

    payload = torch.load(tmp_path / "step_1.pt", weights_only=False)
    assert payload["sampler_mcmc_state"] == {"has_burned_in": True}


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
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="main",
        dirty_worktree=False,
        extra={"python_version": "3.12.0", "torch_version": torch.__version__},
    )
    return SimpleNamespace(cfg=cfg, metadata=metadata)


def test_checkpoint_payload_uses_structured_schema(tmp_path) -> None:
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)
    context = _context()
    state = _state(1)

    callback.handle(Event(name="step_end", context=context, state=state, payload={"step": 1}))

    payload = torch.load(tmp_path / "step_1.pt", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["kind"] == "spenn.model_checkpoint"
    assert payload["model_config"] == OmegaConf.to_container(context.cfg.model)
    assert payload["model_config_hash"] == model_config_hash(context.cfg.model)
    assert payload["resolved_config_hash"] == model_config_hash(context.cfg)
    assert payload["config_id"] == "lr=0.001_channels=4"
    assert payload["runtime"] == {"device": "cpu", "dtype": "float64"}
    assert payload["git"] == {"sha": "deadbeef", "branch": "main", "dirty": False}
    assert payload["versions"] == {
        "python": "3.12.0",
        "torch": torch.__version__,
        "spenn": spenn.__version__,
    }


def test_checkpoint_provenance_degrades_without_context(tmp_path) -> None:
    # Existing unit tests run with context=None; provenance must degrade to
    # None fields instead of failing the checkpoint write.
    callback = Checkpoint(triggers=["step_end"], output_dir=tmp_path, every_n_steps=1)

    callback.handle(_event(_state(1)))

    payload = torch.load(tmp_path / "step_1.pt", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["model_config"] is None
    assert payload["model_config_hash"] is None
    assert payload["config_id"] is None
    assert payload["versions"]["spenn"] == spenn.__version__
