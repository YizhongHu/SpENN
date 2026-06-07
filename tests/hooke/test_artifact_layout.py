"""Artifact layout tests for the Hooke scaffold smoke run."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from spenn.artifacts import REQUIRED_RUN_DIRS
from spenn.run import load_config, run_from_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "hooke" / "configs" / "smoke" / "scaffold.yaml"


def test_scaffold_run_writes_required_artifact_layout(tmp_path: Path) -> None:
    """The scaffold runner writes the required artifact skeleton."""

    code = run_from_config(load_config(str(CONFIG), [f"run.root={tmp_path}"]), config_path=str(CONFIG), command="pytest run")

    assert code == 0
    run_dir = _single_run_dir(tmp_path)
    assert {path.name for path in run_dir.iterdir()} >= {
        *REQUIRED_RUN_DIRS,
        "config.yaml",
        "resolved_config.yaml",
        "metadata.json",
        "status.json",
        "metrics.csv",
        "metrics.jsonl",
        "report.md",
    }

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = OmegaConf.load(run_dir / "config.yaml")
    resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert status["status"] == "completed"
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == run_dir.name
    assert metadata["run_dir"] == str(run_dir)
    assert metadata["config_path"] == str(CONFIG)
    assert config.run.run_id is None
    assert config.run.dir is None
    assert resolved.run.dir == str(run_dir)
    assert resolved.run.run_id == run_dir.name
    assert "No Hooke physics" in report


def _single_run_dir(root: Path) -> Path:
    run_dirs = sorted((root / "hooke_scaffold" / "scaffold").iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]
