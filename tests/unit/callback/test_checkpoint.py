"""Tests for checkpoint writing and the load_model_checkpoint restore helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spenn.callback import load_model_checkpoint


def _write_payload(path: Path, model: torch.nn.Module) -> None:
    """Write a minimal payload matching the Checkpoint callback contract."""

    payload = {
        "step": 7,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "sampler_mcmc_state": None,
        "metrics": {"train/loss": 0.5},
    }
    torch.save(payload, path)


def test_load_model_checkpoint_restores_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "latest.pt"
    _write_payload(checkpoint, trained)

    torch.manual_seed(1)
    fresh = torch.nn.Linear(3, 2)
    assert not torch.equal(fresh.weight, trained.weight)

    restored = load_model_checkpoint(fresh, checkpoint)

    assert restored is fresh  # restores in place and returns the same module
    assert torch.equal(restored.weight, trained.weight)
    assert torch.equal(restored.bias, trained.bias)


def test_load_model_checkpoint_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_model_checkpoint(torch.nn.Linear(3, 2), tmp_path / "absent.pt")


def test_load_model_checkpoint_rejects_non_checkpoint_payload(tmp_path: Path) -> None:
    bogus = tmp_path / "weights.pt"
    torch.save({"weights": torch.zeros(2)}, bogus)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_model_checkpoint(torch.nn.Linear(3, 2), bogus)


def test_load_model_checkpoint_strict_architecture_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest.pt"
    _write_payload(checkpoint, torch.nn.Linear(3, 2))

    with pytest.raises(RuntimeError):
        load_model_checkpoint(torch.nn.Linear(4, 2), checkpoint)
