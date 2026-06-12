"""Tests for checkpoint writing and the load_model_checkpoint restore helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from spenn.callback import load_model_checkpoint, model_config_hash


def _write_payload(path: Path, model: torch.nn.Module, **extra: object) -> None:
    """Write a payload matching the Checkpoint callback contract.

    Without ``extra`` this is a legacy (pre-schema) payload; pass
    ``model_config_hash=...`` etc. to emulate a structured checkpoint.
    """

    payload = {
        "step": 7,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "sampler_mcmc_state": None,
        "metrics": {"train/loss": 0.5},
        **extra,
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


def test_load_model_checkpoint_strict_missing_and_unexpected_keys(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)

    # Unexpected key: the checkpoint carries a parameter the model lacks.
    state = dict(model.state_dict())
    state["ghost"] = torch.zeros(1)
    unexpected = tmp_path / "unexpected.pt"
    torch.save({"model_state_dict": state}, unexpected)
    with pytest.raises(RuntimeError, match="ghost"):
        load_model_checkpoint(torch.nn.Linear(3, 2), unexpected)

    # Missing key: the checkpoint lacks a parameter the model expects.
    state = dict(model.state_dict())
    del state["bias"]
    missing = tmp_path / "missing.pt"
    torch.save({"model_state_dict": state}, missing)
    with pytest.raises(RuntimeError, match="bias"):
        load_model_checkpoint(torch.nn.Linear(3, 2), missing)


def test_model_config_hash_is_canonical() -> None:
    config = {"_target_": "spenn.nn.SpENNWaveFunction", "channels": 4, "layers": [1, 2]}
    reordered = {"layers": [1, 2], "channels": 4, "_target_": "spenn.nn.SpENNWaveFunction"}

    assert model_config_hash(config) == model_config_hash(reordered)
    assert model_config_hash(config) == model_config_hash(OmegaConf.create(config))
    assert model_config_hash(config) != model_config_hash({**config, "channels": 8})


def test_load_model_checkpoint_verifies_model_config_hash(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    model_config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2}
    expected = model_config_hash(model_config)
    checkpoint = tmp_path / "latest.pt"
    _write_payload(
        checkpoint, model, model_config=model_config, model_config_hash=expected, schema_version=1
    )

    restored = load_model_checkpoint(
        torch.nn.Linear(3, 2), checkpoint, expected_model_config_hash=expected
    )
    assert torch.equal(restored.weight, model.weight)

    with pytest.raises(ValueError, match="model_config_hash mismatch"):
        load_model_checkpoint(
            torch.nn.Linear(3, 2), checkpoint, expected_model_config_hash="deadbeef"
        )


def test_load_model_checkpoint_legacy_payload_lacks_model_config_metadata(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "legacy.pt"
    _write_payload(checkpoint, model)  # legacy payload: no model_config_hash

    # Demanding a hash from a legacy checkpoint must fail loudly...
    with pytest.raises(ValueError, match="does not contain model_config metadata"):
        load_model_checkpoint(
            torch.nn.Linear(3, 2), checkpoint, expected_model_config_hash="abc123"
        )

    # ...while loading without a hash expectation stays supported, because the
    # model architecture is explicitly configured (never inferred from keys).
    restored = load_model_checkpoint(torch.nn.Linear(3, 2), checkpoint)
    assert torch.equal(restored.weight, model.weight)


def test_load_model_checkpoint_mismatch_escape_hatch(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "latest.pt"
    _write_payload(checkpoint, model, model_config_hash="stored-hash")

    restored = load_model_checkpoint(
        torch.nn.Linear(3, 2),
        checkpoint,
        expected_model_config_hash="other-hash",
        allow_model_config_mismatch=True,
    )
    assert torch.equal(restored.weight, model.weight)
