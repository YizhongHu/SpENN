"""Tests for package-owned checkpoint restore helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from spenn.checkpoint import checkpoint_hashes, restore_checkpoint, save_checkpoint, stable_config_hash


def _cfg(*, model_out: int = 2):
    return OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": model_out},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"_target_": "tests.Trainer", "max_steps": 2},
            "sampler": {"_target_": "tests.Sampler", "n_steps": 5},
            "hamiltonian_terms": {"constant": {"_target_": "tests.ConstantHamiltonian"}},
            "run": {"run_id": "run", "dir": "/tmp/run"},
            "study": {"name": "unit", "config_id": "cfg"},
        }
    )


def _context(cfg=None):
    return SimpleNamespace(
        cfg=_cfg() if cfg is None else cfg,
        metadata=SimpleNamespace(
            run_id="run",
            device="cpu",
            dtype="float64",
            git_commit="deadbeef",
            git_branch="codex/checkpoint",
            dirty_worktree=False,
            command="pytest",
            extra={"slurm": {}},
        ),
        run_dir=Path("/tmp/run"),
    )


class _Trainer:
    def __init__(self) -> None:
        self.loaded = None

    def state_dict(self) -> dict[str, int]:
        return {"global_step": 3}

    def load_state_dict(self, state) -> None:
        self.loaded = dict(state)


class _Sampler:
    def __init__(self) -> None:
        self.loaded = None

    def mcmc_state_dict(self) -> dict[str, object]:
        return {"has_burned_in": True, "position": torch.ones(1)}

    def load_mcmc_state_dict(self, state) -> None:
        self.loaded = dict(state)


def _write_checkpoint(tmp_path: Path, model: torch.nn.Module | None = None, **kwargs) -> Path:
    model = torch.nn.Linear(3, 2).double() if model is None else model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        step=3,
        model=model,
        optimizer=optimizer,
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        **kwargs,
    )


def test_model_only_restore_loads_weights_into_configured_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = torch.nn.Linear(3, 2).double()
    root = _write_checkpoint(tmp_path, model=trained).parent

    torch.manual_seed(1)
    fresh = torch.nn.Linear(3, 2).double()
    assert not torch.equal(fresh.weight, trained.weight)

    report = restore_checkpoint(
        load={"path": str(root), "mode": "model_only", "strict": True},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False
    assert report.loaded_sampler is False


def test_restore_rejects_checkpoint_without_complete_marker(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    (checkpoint_dir / "COMPLETE").unlink()

    with pytest.raises(ValueError, match="COMPLETE"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_restore_rejects_model_config_hash_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="model_config"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only"},
            model=torch.nn.Linear(3, 4).double(),
            context=_context(_cfg(model_out=4)),
        )


def test_model_only_does_not_require_train_resume_files(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        step=1,
        model=trained,
        context=_context(),
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )

    fresh = torch.nn.Linear(3, 2).double()
    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "model_only"},
        model=fresh,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert report.loaded_model is True
    assert report.loaded_optimizer is False


def test_train_resume_restores_all_train_state(tmp_path: Path) -> None:
    trained = torch.nn.Linear(3, 2).double()
    checkpoint_dir = _write_checkpoint(tmp_path, model=trained)
    fresh = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)
    trainer = _Trainer()
    sampler = _Sampler()

    report = restore_checkpoint(
        load={"path": str(checkpoint_dir), "mode": "train_resume"},
        model=fresh,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        context=_context(),
    )

    assert torch.equal(fresh.weight, trained.weight)
    assert trainer.loaded == {"global_step": 3}
    assert sampler.loaded["has_burned_in"] is True
    assert report.loaded_optimizer is True
    assert report.loaded_trainer is True
    assert report.loaded_sampler is True
    assert report.loaded_rng is True


def test_train_resume_fails_when_required_file_is_missing(tmp_path: Path) -> None:
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        step=1,
        model=torch.nn.Linear(3, 2).double(),
        optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
        trainer=_Trainer(),
        sampler=_Sampler(),
        context=_context(),
        save_optimizer=False,
    )

    with pytest.raises(FileNotFoundError, match="optimizer"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "train_resume"},
            model=torch.nn.Linear(3, 2).double(),
            optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
            trainer=_Trainer(),
            sampler=_Sampler(),
            context=_context(),
        )


def test_restore_strict_load_fails_on_unexpected_keys(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    state = torch.load(checkpoint_dir / "model.pt", weights_only=False)
    state["ghost"] = torch.zeros(1)
    torch.save(state, checkpoint_dir / "model.pt")

    with pytest.raises(RuntimeError, match="ghost"):
        restore_checkpoint(
            load={"path": str(checkpoint_dir), "mode": "model_only", "strict": True},
            model=torch.nn.Linear(3, 2).double(),
            context=_context(),
        )


def test_stable_config_hash_is_canonical_and_strict() -> None:
    config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2}
    reordered = {"out_features": 2, "_target_": "torch.nn.Linear", "in_features": 3}

    assert stable_config_hash(config) == stable_config_hash(reordered)
    assert stable_config_hash(config) == stable_config_hash(OmegaConf.create(config))
    assert stable_config_hash(config) != stable_config_hash({**config, "out_features": 4})

    with pytest.raises(TypeError, match="JSON/YAML-safe"):
        stable_config_hash({"path": Path("not-json-safe")})


def test_checkpoint_hashes_resolve_interpolations_and_track_components() -> None:
    cfg = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "a", "dir": "/tmp/a"},
            "model": {"channels": "${width}"},
            "sampler": {"n_steps": "${steps}"},
            "hamiltonian_terms": {"trap": {"omega": "${omega}"}},
        }
    )
    same = OmegaConf.create(
        {
            "width": 4,
            "steps": 5,
            "omega": 0.5,
            "run": {"run_id": "b", "dir": "/tmp/b"},
            "model": {"channels": 4},
            "sampler": {"n_steps": 5},
            "hamiltonian_terms": {"trap": {"omega": 0.5}},
        }
    )
    changed_sampler = OmegaConf.merge(same, {"sampler": {"n_steps": 6}})
    changed_hamiltonian = OmegaConf.merge(same, {"hamiltonian_terms": {"trap": {"omega": 0.7}}})

    hashes = checkpoint_hashes(cfg)

    assert hashes["model_config"] == checkpoint_hashes(same)["model_config"]
    assert hashes["resolved_config"] == checkpoint_hashes(same)["resolved_config"]
    assert hashes["sampler_config"] != checkpoint_hashes(changed_sampler)["sampler_config"]
    assert hashes["hamiltonian_config"] != checkpoint_hashes(changed_hamiltonian)["hamiltonian_config"]
