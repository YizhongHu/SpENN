"""Fail-loudness behavior of run_from_config and interface validation."""

from __future__ import annotations

import json
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
        run_from_config(cfg, config_path="x", command="t", raise_exceptions=True)


def test_prepare_run_context_rejects_logger_without_log(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.loggers = [{"_target_": "builtins.object"}]
    with pytest.raises(TypeError, match="log"):
        run_from_config(cfg, config_path="x", command="t", raise_exceptions=True)


def test_invalid_load_path_is_fatal_and_durable_with_terminal_disabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing" / "latest.json"
    cfg = _cfg(
        tmp_path,
        terminal={"enabled": False},
        load={
            "path": str(missing),
            "mode": "model_only",
            "strict": True,
            "allow_protocol_mismatch": False,
        },
        loggers=[
            {
                "_target_": "spenn.logging.JSONL",
                "path": "${run.dir}/metrics.jsonl",
            }
        ],
        runner={
            "_target_": "spenn.runner.Evaluate",
            "model": None,
            "sampler": None,
            "hamiltonian_terms": [],
            "load": "${load}",
            "diagnostics": [
                {
                    "_target_": "spenn.diagnostics.EnergyEvaluation",
                    "name": "energy",
                }
            ],
        },
    )

    assert run_from_config(cfg, config_path="invalid-load.yaml", command="run.py --config invalid-load.yaml") == 1

    captured = capsys.readouterr()
    assert "FATAL load error" in captured.err
    assert str(missing) in captured.err
    assert "load.path" in captured.err
    assert captured.out == ""

    run_dirs = list(tmp_path.glob("f/f/*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "run_start.json").is_file()

    error = json.loads((run_dir / "error.json").read_text())
    assert error["phase"] == "load"
    assert error["exception_type"] == "FileNotFoundError"
    assert str(missing) in error["exception_message"]

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    event_names = [event["event"] for event in events]
    assert event_names[:2] == ["run_start", "load_start"]
    assert "load_failed" in event_names
    assert "run_failed" in event_names

    load_start = next(event for event in events if event["event"] == "load_start")
    assert load_start["payload"] == {
        "mode": "model_only",
        "path": str(missing),
        "strict": True,
    }
    load_failed = next(event for event in events if event["event"] == "load_failed")
    assert load_failed["payload"]["mode"] == "model_only"
    assert load_failed["payload"]["path"] == str(missing)
    assert load_failed["payload"]["exception_type"] == "FileNotFoundError"
    assert str(missing) in load_failed["payload"]["message"]
