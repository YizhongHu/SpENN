"""Exact-SHA, stratum, immutable-script, and submission-failure tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

STUDY_DIR = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_diagnostic_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch = _load("diagnostic_launch")


def _row(*, smoke: bool = False) -> dict[str, Any]:
    resources = (
        {
            "partition": "gpu_test",
            "stratum": "a100_mig",
            "constraint": None,
            "timeout_min": 120,
            "cpus": 4,
            "mem_gb": 32,
            "gpus": 1,
        }
        if smoke
        else {
            "partition": "kozinsky_gpu",
            "stratum": "a100",
            "constraint": "a100",
            "timeout_min": 720,
            "cpus": 4,
            "mem_gb": 32,
            "gpus": 1,
        }
    )
    return {
        "row_id": "step_025000-primary_256x4096-seed1000",
        "kind": "diagnostic_eval",
        "resources": resources,
    }


def _manifest(row: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    return {
        "study": "he-v1-diagnostic-v1",
        "scale": "smoke" if smoke else "production",
        "plan_sha256": "b" * 64,
        "evaluation_git_sha": "a" * 40,
        "rows": [row],
    }


def test_production_is_pinned_and_smoke_uses_declared_mig_partition(
    tmp_path: Path,
) -> None:
    production = launch.sbatch_directives(
        _row(), job_name="p", log_dir=tmp_path, account=None
    )
    smoke = launch.sbatch_directives(
        _row(smoke=True), job_name="s", log_dir=tmp_path, account=None
    )
    assert "#SBATCH --constraint=a100" in production
    assert not any("--constraint" in line for line in smoke)
    assert "#SBATCH --partition=gpu_test" in smoke
    assert "#SBATCH --no-requeue" in production
    assert "#SBATCH --no-requeue" in smoke


def test_changed_resource_coordinate_is_rejected(tmp_path: Path) -> None:
    row = _row()
    row["resources"]["cpus"] = 8
    with pytest.raises(launch.DiagnosticLaunchError, match="frozen grid"):
        launch.sbatch_directives(row, job_name="bad", log_dir=tmp_path, account=None)


def test_script_uses_absolute_uv_locked_sync_and_exact_sha(tmp_path: Path) -> None:
    row = _row(smoke=True)
    script = launch.build_script(
        row,
        directives=launch.sbatch_directives(
            row, job_name="smoke", log_dir=tmp_path, account=None
        ),
        command=launch.driver_command(
            row,
            results_root=tmp_path,
            plan_attempt_id="P1",
            launch_attempt_id="L1",
        ),
        repo_root=STUDY_DIR.parents[2],
        evaluation_git_sha="a" * 40,
        uv_bin="/home/test/.local/bin/uv",
        uv_extra="cu128",
        uv_environment_root=tmp_path / "envs",
        uv_cache_root=tmp_path / "cache",
    )
    assert "/home/test/.local/bin/uv run --locked --extra cu128" in script
    assert "--nosync" not in script
    assert "git rev-parse HEAD" in script
    assert "git status --porcelain --untracked-files=no" in script
    assert "diagnostic.py" in script
    assert '"${SLURM_JOB_ID}"' in script


def test_bare_uv_and_unknown_rows_fail_loudly(tmp_path: Path) -> None:
    row = _row()
    with pytest.raises(launch.DiagnosticLaunchError, match="absolute"):
        launch.build_script(
            row,
            directives=[],
            command=["python", "diagnostic.py"],
            repo_root=tmp_path,
            evaluation_git_sha="a" * 40,
            uv_bin="uv",
            uv_extra="cu128",
            uv_environment_root=tmp_path / "envs",
            uv_cache_root=tmp_path / "cache",
        )
    with pytest.raises(launch.DiagnosticLaunchError, match="not in"):
        launch.select_rows(_manifest(row), row_ids=["missing"])


def test_submission_failure_leaves_immutable_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row()
    manifest = _manifest(row)
    monkeypatch.setattr(launch, "_require_repo_sha", lambda *_args: None)

    def fail(_path: Path) -> str:
        raise launch.DiagnosticLaunchError("synthetic sbatch failure")

    with pytest.raises(launch.DiagnosticLaunchError, match="synthetic"):
        launch.launch(
            manifest,
            results_root=tmp_path,
            repo_root=STUDY_DIR.parents[2],
            plan_attempt_id="P1",
            launch_attempt_id="L1",
            rows=[row],
            uv_bin="/home/test/.local/bin/uv",
            uv_extra="cu128",
            uv_environment_root=tmp_path / "envs",
            uv_cache_root=tmp_path / "cache",
            submit=True,
            submitter=fail,
            submit_interval_seconds=0.5,
        )
    failure = json.loads(
        (tmp_path / "01_launch" / "L1" / "launch_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["error_type"] == "DiagnosticLaunchError"
    assert failure["completed_rows"] == []
    with pytest.raises(FileExistsError):
        launch.launch(
            manifest,
            results_root=tmp_path,
            repo_root=STUDY_DIR.parents[2],
            plan_attempt_id="P1",
            launch_attempt_id="L1",
            rows=[row],
            uv_bin="/home/test/.local/bin/uv",
            uv_extra="cu128",
            uv_environment_root=tmp_path / "envs",
            uv_cache_root=tmp_path / "cache",
        )
