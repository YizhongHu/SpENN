"""Regression guards for the PR5.1 post-smoke cleanup.

These assert that transitional surfaces stay removed and that the runtime
contracts (run-dir layout, runner-owned vs RunContext-owned config) hold.
"""

from __future__ import annotations

from pathlib import Path

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
