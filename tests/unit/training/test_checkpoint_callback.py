"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

import torch

from spenn.callback import Checkpoint, Event
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
