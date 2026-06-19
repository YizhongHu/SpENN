"""Entrypoint tests for configured runs (via a real Evaluate config)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tests" / "integration" / "artifacts" / "hooke" / "exact_singlet_eval.yaml"
RUN_GLOB = "hooke_exact/singlet"


def test_run_py_is_the_only_root_entrypoint() -> None:
    """The repository uses run.py instead of train.py for configured runs."""

    assert (ROOT / "run.py").exists()
    assert not (ROOT / "train.py").exists()


def test_run_cli_writes_rerunnable_config(tmp_path: Path) -> None:
    """The public run.py command writes a config that launches a fresh run."""

    _run_cli(CONFIG, tmp_path)
    first_run = _single_run_dir(tmp_path)
    config_snapshot = OmegaConf.load(first_run / "config.yaml")
    resolved_snapshot = OmegaConf.load(first_run / "resolved_config.yaml")

    assert config_snapshot.run.run_id is None
    assert config_snapshot.run.dir is None
    assert resolved_snapshot.run.run_id == first_run.name
    assert resolved_snapshot.run.dir == str(first_run)

    _run_cli(first_run / "config.yaml", None)
    run_dirs = sorted((tmp_path / RUN_GLOB).iterdir())
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    assert all((run_dir / "status.json").exists() for run_dir in run_dirs)


def _run_cli(config: Path, root: Path | None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "run.py", "--config", str(config)]
    if root is not None:
        cmd.append(f"run.root={root}")
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def _single_run_dir(root: Path) -> Path:
    run_dirs = sorted((root / RUN_GLOB).iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]
