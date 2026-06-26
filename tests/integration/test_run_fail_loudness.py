"""Fail-loudness behavior of run_from_config and interface validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from spenn.run import run_from_config


def _cfg(tmp_path: Path, **extra) -> OmegaConf:
    base = {
        "experiment": {"name": "f", "sector": "f", "run_name": "f"},
        "run": {"root": str(tmp_path), "run_id": None, "dir": None},
        "runtime": {"seed": 0},
        "runner": {
            "_target_": "spenn.runner.Evaluate",
            "model": None,
            "sampler": None,
            "hamiltonian_terms": [],
        },
    }
    base.update(extra)
    return OmegaConf.create(base)


def test_run_from_config_returns_one_on_handled_failure(tmp_path: Path) -> None:
    # Evaluate with sampler=None fails inside run(); default behavior returns 1.
    assert run_from_config(_cfg(tmp_path), config_path="x", command="t") == 1


def test_run_from_config_raise_exceptions_reraises_original(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        run_from_config(_cfg(tmp_path), config_path="x", command="t", raise_exceptions=True)


def test_prepare_run_context_rejects_callback_without_handle(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.callbacks = [{"_target_": "builtins.object"}]
    with pytest.raises(TypeError, match="handle"):
        run_from_config(cfg, config_path="x", command="t")


def test_prepare_run_context_rejects_logger_without_log(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.loggers = [{"_target_": "builtins.object"}]
    with pytest.raises(TypeError, match="log"):
        run_from_config(cfg, config_path="x", command="t")
