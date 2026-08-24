"""Allocation, real-format restore, task selection, and driver-failure tests."""

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


driver = _load("diagnostic")
plan = driver.plan_stage


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "step_025000"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"real-format-model")
    (path / "resolved_config.yaml").write_text("model: {}\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "completed_updates": 25000,
        "files": {"model": "model.pt", "resolved_config": "resolved_config.yaml"},
        "provenance": {"git_sha": "c" * 40, "tpen_version": "0.1.0"},
    }
    (path / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    row = {
        "row_id": "step_025000-checkpoint-diagnostics",
        "kind": "diagnostic_eval",
        "stage": "03_eval",
        "profile": "checkpoint_diagnostics",
        "task_names": [
            "he_en_numerical_atlas",
            "full_model_antisymmetry",
            "rotation_consistency",
            "feature_trace",
        ],
        "protocol": "checkpoint_diagnostics",
        "comparison_kind": "checkpoint_diagnostics",
        "seed": 6000,
        "n_walkers": 4,
        "n_draws": 1,
        "burn_in": 1,
        "stride": 1,
        "record_capacity": 4,
        "diagnostic_samples": 4,
        "factor_arm": None,
        "checkpoint_label": "step_025000",
        "checkpoint_step": 25000,
        "checkpoint_model_sha256": plan.file_sha256(path / "model.pt"),
        "checkpoint_manifest_sha256": plan.file_sha256(path / "manifest.json"),
        "checkpoint_complete_sha256": plan.file_sha256(path / "COMPLETE"),
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_source_git_sha": "c" * 40,
        "checkpoint_source_tpen_version": "0.1.0",
        "checkpoint_source_dir": str(path),
        "resources": {
            "partition": "gpu_test",
            "stratum": "a100_mig",
            "constraint": None,
            "timeout_min": 120,
            "cpus": 4,
            "mem_gb": 32,
            "gpus": 1,
        },
    }
    return path, row


def _manifest(row: dict[str, Any]) -> dict[str, Any]:
    repo_root = STUDY_DIR.parents[2]
    base = "experiments/atomistic/he-v1/configs/eval.yaml"
    overlay = "experiments/atomistic/he-v1/configs/diagnostic_eval.yaml"
    return {
        "study": "he-v1-diagnostic-v1",
        "scale": "smoke",
        "scale_overrides": {
            "n_walkers": 4,
            "n_draws": 2,
            "burn_in": 1,
            "stride": 1,
            "diagnostic_samples": 4,
            "atlas_max_refinement_steps": 4,
            "atlas_radii": [2.0],
        },
        "plan_sha256": "d" * 64,
        "evaluation_git_sha": "e" * 40,
        "base_eval_config": base,
        "base_eval_config_sha256": plan.file_sha256(repo_root / base),
        "overlay_config": overlay,
        "overlay_config_sha256": plan.file_sha256(repo_root / overlay),
        "rows": [row],
    }


def test_scheduler_and_delivered_strata_are_fail_closed(tmp_path: Path) -> None:
    _path, row = _checkpoint(tmp_path / "inputs")
    with pytest.raises(driver.DiagnosticDriverError, match="SLURM_JOB_ID"):
        driver.require_scheduler({})
    assert driver.require_scheduler({"SLURM_JOB_ID": "42"}) == "42"
    receipt = driver.verify_delivered_device(
        row,
        receipt_dir=tmp_path / "matching",
        job_id="42",
        device_reader=lambda: "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
        environ={"SLURM_JOB_PARTITION": "gpu_test"},
    )
    assert receipt["delivered_matches_requested"] is True
    full = dict(row)
    full["resources"] = {
        **row["resources"],
        "partition": "kozinsky_gpu",
        "stratum": "a100",
        "constraint": "a100",
    }
    with pytest.raises(driver.DiagnosticDriverError, match="full NVIDIA"):
        driver.verify_delivered_device(
            full,
            receipt_dir=tmp_path / "mismatch",
            job_id="43",
            device_reader=lambda: "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
            environ={"SLURM_JOB_PARTITION": "kozinsky_gpu"},
        )
    written = json.loads(
        (tmp_path / "mismatch" / driver.ALLOCATION_RECEIPT).read_text(
            encoding="utf-8"
        )
    )
    assert written["delivered_matches_requested"] is False


def test_real_format_checkpoint_is_rehashed_and_partial_restore_fails(
    tmp_path: Path,
) -> None:
    path, row = _checkpoint(tmp_path)
    binding = driver.reconcile_checkpoint(row)
    assert binding["checkpoint_model_file"] == str(path / "model.pt")
    assert binding["hashes"]["model"] == row["checkpoint_model_sha256"]
    (path / "COMPLETE").unlink()
    with pytest.raises(driver.DiagnosticDriverError, match="incomplete"):
        driver.reconcile_checkpoint(row)


def test_config_selects_exact_tasks_and_smoke_scales_same_graph(tmp_path: Path) -> None:
    _path, row = _checkpoint(tmp_path)
    manifest = _manifest(row)
    cfg, config_sha = driver.build_diagnostic_config(
        manifest,
        row,
        results_root=tmp_path / "results",
        plan_attempt_id="P1",
    )
    assert len(config_sha) == 64
    assert [task.name for task in cfg.evaluator.tasks] == row["task_names"]
    assert cfg.evaluation_tasks.he_en_numerical_atlas.generator.max_refinement_steps == 4
    assert list(cfg.evaluation_tasks.he_one_electron_tail_atlas.generator.radii) == [2.0]
    callback_targets = [callback["_target_"] for callback in cfg.callbacks]
    assert "tpen.callback.ArtifactIndex" in callback_targets
    assert "tpen.callback.FailureLog" in callback_targets
    assert "callbacks" not in cfg.runner and "loggers" not in cfg.runner
    assert cfg.load.mode == "model_only"
    assert cfg.load.strict is True
    assert cfg.load.path == str(_path)


def test_unknown_planned_task_fails_before_restore(tmp_path: Path) -> None:
    _path, row = _checkpoint(tmp_path)
    row["task_names"] = ["not_a_task"]
    with pytest.raises(driver.DiagnosticDriverError, match="not declared"):
        driver.build_diagnostic_config(
            _manifest(row),
            row,
            results_root=tmp_path / "results",
            plan_attempt_id="P1",
        )


def test_checkpoint_hash_change_after_plan_fails(tmp_path: Path) -> None:
    path, row = _checkpoint(tmp_path)
    (path / "model.pt").write_bytes(b"mutated")
    with pytest.raises(driver.DiagnosticDriverError, match="content mismatch"):
        driver.reconcile_checkpoint(row)


def test_run_row_passes_real_format_config_to_sanctioned_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, row = _checkpoint(tmp_path / "inputs")
    manifest = _manifest(row)
    monkeypatch.setattr(driver, "require_evaluation_checkout", lambda _manifest: "e" * 40)
    observed: dict[str, Any] = {}

    def runner(cfg: Any, *, config_path: str, command: str) -> int:
        observed["load_path"] = cfg.load.path
        observed["strict"] = cfg.load.strict
        observed["tasks"] = [task.name for task in cfg.evaluator.tasks]
        observed["config_path"] = config_path
        observed["command"] = command
        return 0

    result = driver.run_row(
        manifest,
        row,
        results_root=tmp_path / "results",
        plan_attempt_id="P1",
        launch_attempt_id="L1",
        device_reader=lambda: "NVIDIA A100-SXM4-40GB MIG 3g.20gb",
        runner=runner,
        environ={
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_PARTITION": "gpu_test",
        },
    )
    assert result == 0
    assert observed["load_path"] == row["checkpoint_source_dir"]
    assert observed["strict"] is True
    assert observed["tasks"] == row["task_names"]
    binding = json.loads(
        (
            tmp_path
            / "results"
            / "03_eval"
            / row["row_id"]
            / "P1"
            / "checkpoint_binding.json"
        ).read_text(encoding="utf-8")
    )
    assert binding["hashes"]["model"] == row["checkpoint_model_sha256"]
