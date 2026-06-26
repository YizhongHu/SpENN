"""Optional dependency gateway behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from spenn.dependencies import OptionalDependencyError, require_torch


def test_require_torch_rejects_partial_namespace_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A namespace-only torch package is not a usable PyTorch install."""

    def fake_import_module(name: str):
        if name == "torch":
            return SimpleNamespace(__file__=None, __version__=None)
        if name == "torch.nn":
            raise ImportError(name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("spenn.dependencies.importlib.import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError, match="uv sync --extra cpu"):
        require_torch(feature="configured SpENN run")


def test_run_cli_preflight_reports_missing_torch_without_hydra_traceback(tmp_path: Path) -> None:
    """Torch-required configs fail early with the dependency gateway message."""

    config = tmp_path / "torch_required.yaml"
    config.write_text(
        """
experiment:
  name: optional_dep
  sector: smoke
run:
  root: outputs
runtime:
  seed: 0
runner:
  _target_: spenn.runner.Train
""",
        encoding="utf-8",
    )
    fake_torch = tmp_path / "fake_torch"
    fake_torch.mkdir()
    (fake_torch / "torch.py").write_text("__version__ = None\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(fake_torch), str(repo), env.get("PYTHONPATH", "")])

    result = subprocess.run(
        [sys.executable, "run.py", "--config", str(config)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "configured SpENN run requires a complete `torch` installation" in result.stderr
    assert "hydra.errors.InstantiationException" not in result.stderr
