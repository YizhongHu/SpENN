"""Tests for optional experiment tracking helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from omegaconf import OmegaConf

from spenn.training.tracking import NullTracker, WandbTracker, build_tracker


def test_build_tracker_returns_noop_when_wandb_disabled(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"tracking": {"wandb": {"enabled": False}}})

    tracker = build_tracker(cfg, output_dir=tmp_path, git={})

    assert isinstance(tracker, NullTracker)


def test_wandb_tracker_uses_config_and_logs_numeric_payloads(tmp_path: Path, monkeypatch) -> None:
    runs: list[_FakeRun] = []

    def init(**kwargs):
        run = _FakeRun()
        runs.append(run)
        run.init_kwargs = kwargs
        return run

    wandb = types.ModuleType("wandb")
    wandb.init = init
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    cfg = OmegaConf.create(
        {
            "experiment_name": "hooke_multibody_smoke",
            "run_id": "run_01",
            "run": {"mode": "train"},
            "tracking": {
                "tags": ["hooke_multibody", "smoke"],
                "wandb": {
                    "enabled": True,
                    "project": "spenn",
                    "entity": None,
                    "mode": "offline",
                    "group": "hooke_multibody",
                    "name": "named_run",
                    "id": "run_01",
                    "job_type": "train",
                    "resume": "allow",
                },
            },
        }
    )

    tracker = build_tracker(cfg, output_dir=tmp_path, git={"git_commit": "abc123"})
    tracker.log_rows([{"step": 2, "energy": 1.25, "flag": True, "bad": float("nan"), "note": "skip"}])
    tracker.log_metrics({"final_energy": 1.1, "count": 3, "ok": True})
    tracker.finish()

    assert isinstance(tracker, WandbTracker)
    assert len(runs) == 1
    run = runs[0]
    assert run.init_kwargs["project"] == "spenn"
    assert run.init_kwargs["name"] == "named_run"
    assert run.init_kwargs["id"] == "run_01"
    assert run.init_kwargs["mode"] == "offline"
    assert run.init_kwargs["dir"] == str(tmp_path)
    assert run.init_kwargs["tags"] == ["hooke_multibody", "smoke"]
    assert run.init_kwargs["config"]["git"]["git_commit"] == "abc123"
    assert run.logs[0] == ({"energy": 1.25}, 2)
    assert run.logs[1] == ({"final_energy": 1.1, "count": 3}, None)
    assert run.summary == {"final_energy": 1.1, "count": 3}
    assert run.finished is True


class _FakeRun:
    def __init__(self) -> None:
        self.init_kwargs: dict[str, object] = {}
        self.logs: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logs.append((payload, step))

    def finish(self) -> None:
        self.finished = True
