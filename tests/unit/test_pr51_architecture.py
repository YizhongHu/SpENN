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

from tpen.run import run_from_config


def test_scaffold_and_load_are_not_public_runners() -> None:
    import tpen.runner as runner

    assert "Scaffold" not in runner.__all__
    assert "Load" not in runner.__all__
    assert not hasattr(runner, "Scaffold")
    assert not hasattr(runner, "Load")


def test_reference_energy_callback_is_removed() -> None:
    import tpen.callback as callback

    assert "ReferenceEnergy" not in callback.__all__
    assert not hasattr(callback, "ReferenceEnergy")


def test_report_skeleton_callback_is_removed() -> None:
    import tpen.callback as callback

    assert "ReportSkeleton" not in callback.__all__
    assert not hasattr(callback, "ReportSkeleton")


def test_concatenated_state_is_removed() -> None:
    import tpen.data.equivariant_state as module

    assert "ConcatenatedState" not in module.__all__
    assert not hasattr(module, "ConcatenatedState")


def test_equivariant_state_has_no_validate_contract() -> None:
    import tpen.data.equivariant_state as module

    assert "validate_tree" not in module.__all__
    assert not hasattr(module.EquivariantState, "validate")


def test_data_integrity_has_no_recursive_tensor_probe() -> None:
    import tpen.callback as callback

    assert not hasattr(callback, "_iter_tensors")
    assert not hasattr(callback, "_nonfinite_tensor_count")


def test_runtime_qol_modules_are_split_packages() -> None:
    """Keep callback, logging, and runner implementations in owner modules."""

    importable_modules = (
        "tpen.callback.base",
        "tpen.callback.cadence",
        "tpen.callback.status",
        "tpen.callback.snapshot",
        "tpen.callback.metadata",
        "tpen.callback.checkpoint",
        "tpen.callback.equivariance",
        "tpen.callback.health.data_integrity",
        "tpen.callback.health.sampler_health",
        "tpen.callback.health.gradient_stats",
        "tpen.callback.timing.base",
        "tpen.callback.timing.run_timing",
        "tpen.callback.timing.train_step_timing",
        "tpen.callback.timing.evaluation_timing",
        "tpen.callback.timing.diagnostic_timing",
        "tpen.logging.base",
        "tpen.logging.csv",
        "tpen.logging.jsonl",
        "tpen.logging.wandb",
        "tpen.runner.base",
    )
    owner_modules = (
        "tpen.callback.timing",
        "tpen.runner.train",
        "tpen.runner.evaluate",
    )

    for module in importable_modules:
        assert importlib.import_module(module)
    for module in owner_modules:
        assert importlib.util.find_spec(module) is not None

    from tpen.callback import DataIntegrity
    from tpen.callback.health.data_integrity import DataIntegrity as OwnedDataIntegrity
    from tpen.callback import DiagnosticTiming, EvaluationTiming, RunTiming, TrainStepTiming
    from tpen.callback.timing.diagnostic_timing import DiagnosticTiming as OwnedDiagnosticTiming
    from tpen.callback.timing.evaluation_timing import EvaluationTiming as OwnedEvaluationTiming
    from tpen.callback.timing.run_timing import RunTiming as OwnedRunTiming
    from tpen.callback.timing.train_step_timing import TrainStepTiming as OwnedTrainStepTiming
    from tpen.logging import WandB
    from tpen.logging.wandb import WandB as OwnedWandB

    assert DataIntegrity is OwnedDataIntegrity
    assert DiagnosticTiming is OwnedDiagnosticTiming
    assert EvaluationTiming is OwnedEvaluationTiming
    assert RunTiming is OwnedRunTiming
    assert TrainStepTiming is OwnedTrainStepTiming
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
            "from tpen.run import main; from tpen.runner import Runner; print(Runner.__name__)",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Runner"


def test_callback_timing_import_stays_torch_free(tmp_path: Path) -> None:
    """Importing callback timing must not resolve training or Torch modules."""

    (tmp_path / "torch.py").write_text(
        'raise AssertionError("torch imported")\n',
        encoding="utf-8",
    )
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
            (
                "import sys; import tpen.callback.timing; "
                "print('torch' in sys.modules, 'tpen.training' in sys.modules)"
            ),
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_required_run_dirs_are_checks_diagnostics_and_checkpoints() -> None:
    from tpen.artifacts import REQUIRED_RUN_DIRS

    assert REQUIRED_RUN_DIRS == ("checkpoints", "checks", "diagnostics")


def test_permutable_lives_in_data_permutation() -> None:
    import tpen.data.permutation as permutation

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
                "_target_": "tpen.runner.Evaluate",
                "model": None,
                "sampler": None,
                "hamiltonian_terms": [],
                forbidden: [],
            },
        }
    )

    # The runner must not own callbacks/loggers -> run_from_config fails (exit 1).
    assert run_from_config(cfg, config_path="x", command="pytest") == 1
