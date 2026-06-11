"""Regression guards for the PR5.1 post-smoke cleanup.

These assert that transitional surfaces stay removed and that the runtime
contracts (run-dir layout, runner-owned vs RunContext-owned config) hold.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from spenn.run import run_from_config


def test_scaffold_and_load_are_not_public_runners() -> None:
    import spenn.runner as runner

    assert "Scaffold" not in runner.__all__
    assert "Load" not in runner.__all__
    assert not hasattr(runner, "Scaffold")
    assert not hasattr(runner, "Load")


def test_reference_energy_callback_is_removed() -> None:
    import spenn.callback as callback

    assert "ReferenceEnergy" not in callback.__all__
    assert not hasattr(callback, "ReferenceEnergy")


def test_report_skeleton_callback_is_removed() -> None:
    import spenn.callback as callback

    assert "ReportSkeleton" not in callback.__all__
    assert not hasattr(callback, "ReportSkeleton")


def test_concatenated_state_is_removed() -> None:
    import spenn.data.equivariant_state as module

    assert "ConcatenatedState" not in module.__all__
    assert not hasattr(module, "ConcatenatedState")


def test_equivariant_state_has_no_validate_contract() -> None:
    import spenn.data.equivariant_state as module

    assert "validate_tree" not in module.__all__
    assert not hasattr(module.EquivariantState, "validate")


def test_data_validity_has_no_recursive_tensor_probe() -> None:
    import spenn.callback as callback

    assert not hasattr(callback, "_iter_tensors")
    assert not hasattr(callback, "_nonfinite_tensor_count")


def test_runtime_qol_modules_are_split_packages() -> None:
    """Keep callback, logging, and runner implementations in owner modules."""

    importable_modules = (
        "spenn.callback.base",
        "spenn.callback.status",
        "spenn.callback.snapshot",
        "spenn.callback.metadata",
        "spenn.callback.checkpoint",
        "spenn.callback.equivariance",
        "spenn.callback.health.data_validity",
        "spenn.callback.health.sampler_health",
        "spenn.callback.health.gradient_stats",
        "spenn.logging.base",
        "spenn.logging.csv",
        "spenn.logging.jsonl",
        "spenn.logging.wandb",
        "spenn.runner.base",
    )
    owner_modules = (
        "spenn.callback.timing",
        "spenn.runner.train",
        "spenn.runner.evaluate",
    )

    for module in importable_modules:
        assert importlib.import_module(module)
    for module in owner_modules:
        assert importlib.util.find_spec(module) is not None

    from spenn.callback import DataValidity
    from spenn.callback.health.data_validity import DataValidity as OwnedDataValidity
    from spenn.logging import WandB
    from spenn.logging.wandb import WandB as OwnedWandB

    assert DataValidity is OwnedDataValidity
    assert WandB is OwnedWandB


def test_runner_import_does_not_require_torch_nn(tmp_path: Path) -> None:
    """Importing the runner base target should not eagerly import ``torch.nn``."""

    (tmp_path / "torch.py").write_text('__version__ = "partial-torch"\n', encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    pythonpath = [str(tmp_path), str(repo)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from spenn.run import main; from spenn.runner import Runner; print(Runner.__name__)",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Runner"


def test_required_run_dirs_are_checks_diagnostics_and_checkpoints() -> None:
    from spenn.artifacts import REQUIRED_RUN_DIRS

    assert REQUIRED_RUN_DIRS == ("checkpoints", "checks", "diagnostics")


def test_permutable_lives_in_data_permutation() -> None:
    import spenn.data.permutation as permutation

    assert "Permutable" in permutation.__all__
    assert hasattr(permutation, "Permutable")


@pytest.mark.parametrize("forbidden", ["callbacks", "loggers"])
def test_runner_owned_callbacks_or_loggers_are_rejected(tmp_path: Path, forbidden: str) -> None:
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "reject", "sector": "reject", "run_name": "reject"},
            "run": {"root": str(tmp_path), "run_id": None, "dir": None},
            "runtime": {"seed": 0},
            "runner": {
                "_target_": "spenn.runner.Evaluate",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                forbidden: [],
            },
        }
    )

    # The runner must not own callbacks/loggers -> run_from_config fails (exit 1).
    assert run_from_config(cfg, config_path="x", command="pytest") == 1
