"""Exact row reconciliation and durable collection-failure tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parent


def _load(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_diagnostic_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collector = _load("diagnostic_collect")
plan = collector.plan_stage


def _row(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "checkpoint"
    source.mkdir(parents=True)
    (source / "model.pt").write_bytes(b"model")
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (source / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return {
        "row_id": "step_025000-primary_256x4096-seed1000",
        "kind": "diagnostic_eval",
        "stage": "03_eval",
        "profile": "retained_energy",
        "task_names": ["retained_energy"],
        "protocol": "primary_256x4096",
        "comparison_kind": "primary_headline",
        "seed": 1000,
        "n_walkers": 4,
        "n_draws": 2,
        "burn_in": 1,
        "stride": 1,
        "record_capacity": 8,
        "diagnostic_samples": 4,
        "factor_arm": None,
        "checkpoint_label": "step_025000",
        "checkpoint_step": 25000,
        "checkpoint_model_sha256": plan.file_sha256(source / "model.pt"),
        "checkpoint_manifest_sha256": plan.file_sha256(source / "manifest.json"),
        "checkpoint_complete_sha256": plan.file_sha256(source / "COMPLETE"),
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_source_git_sha": "c" * 40,
        "checkpoint_source_tpen_version": "0.1.0",
        "checkpoint_source_dir": str(source),
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


def _manifest(row: dict[str, Any]) -> dict[str, Any]:
    production_hash = plan.file_sha256(
        STUDY_DIR / "configs" / "production_grid.yaml"
    )
    return {
        "study": "he-v1-diagnostic-v1",
        "scale": "smoke",
        "plan_sha256": "d" * 64,
        "evaluation_git_sha": "e" * 40,
        "production_grid_sha256_before": production_hash,
        "checkpoints": [
            {"label": "step_025000", "model_sha256": row["checkpoint_model_sha256"]},
            {"label": "step_050000", "model_sha256": "f" * 64},
        ],
        "rows": [row],
    }


def _materialize(tmp_path: Path, manifest: dict[str, Any], row: dict[str, Any]) -> Path:
    row_id = row["row_id"]
    launch_dir = tmp_path / "01_launch" / "L1" / "rows" / row_id
    launch_dir.mkdir(parents=True)
    submission = {
        "row_id": row_id,
        "plan_sha256": manifest["plan_sha256"],
        "evaluation_git_sha": manifest["evaluation_git_sha"],
        "launch_attempt_id": "L1",
        "submitted": True,
        "job_id": "123",
    }
    (launch_dir / "submission.json").write_text(
        json.dumps(submission), encoding="utf-8"
    )
    outer = tmp_path / "03_eval" / row_id / "P1"
    run_dir = outer / row_id
    (run_dir / "diagnostics").mkdir(parents=True)
    (outer / "row.json").write_text(
        json.dumps(
            {
                "row": row,
                "plan_sha256": manifest["plan_sha256"],
                "launch_attempt_id": "L1",
                "job_id": "123",
            }
        ),
        encoding="utf-8",
    )
    (outer / "allocation_receipt.json").write_text(
        json.dumps(
            {
                "row_id": row_id,
                "job_id": "123",
                "requested_stratum": "a100_mig",
                "delivered_matches_requested": True,
                "delivered_device": "NVIDIA A100 MIG 3g.20gb",
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "checkpoint_label": row["checkpoint_label"],
        "checkpoint_step": row["checkpoint_step"],
        "checkpoint_dir": row["checkpoint_source_dir"],
        "hashes": {
            "model": row["checkpoint_model_sha256"],
            "manifest": row["checkpoint_manifest_sha256"],
            "complete": row["checkpoint_complete_sha256"],
        },
        "source_git_sha": row["checkpoint_source_git_sha"],
    }
    (outer / "checkpoint_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "git_commit": manifest["evaluation_git_sha"],
                "dirty_worktree": False,
            }
        ),
        encoding="utf-8",
    )
    config_sha = "9" * 64
    resolved = {
        "trajectory_identity": {"config_sha256": config_sha},
        "diagnostic": {
            "protocol": row["protocol"],
            "comparison_kind": row["comparison_kind"],
            "n_walkers": row["n_walkers"],
            "n_draws": row["n_draws"],
            "burn_in": row["burn_in"],
            "stride": row["stride"],
        },
        "load": {"path": row["checkpoint_source_dir"]},
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    (run_dir / "checkpoint_replay_semantics.json").write_text(
        json.dumps(
            {
                "checkpoint_model_sha256": row["checkpoint_model_sha256"],
                "evaluation_config_sha256": config_sha,
                "source_git_sha": row["checkpoint_source_git_sha"],
            }
        ),
        encoding="utf-8",
    )
    task_dir = run_dir / "retained_energy"
    task_dir.mkdir()
    artifact_specs = [
        (
            "sampled_eval_table",
            "records.csv",
            {
                "rows": 8,
                "truncated": False,
                "selection": "complete_draw_walker_grid",
            },
        ),
        (
            "local_energy_trajectory_statistics",
            "trajectory_statistics.jsonl",
            {
                "checkpoint_sha256": row["checkpoint_model_sha256"],
                "evaluator_id": "he-v1-diagnostic-v1",
            },
        ),
        ("conditioned_local_energy", "conditioned.json", {}),
    ]
    artifacts = []
    for name, filename, metadata in artifact_specs:
        path = task_dir / filename
        path.write_text("artifact\n", encoding="utf-8")
        artifacts.append(
            {"name": name, "kind": "test", "path": str(path), "metadata": metadata}
        )
    (run_dir / "diagnostics" / "index.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "retained_energy",
                        "namespace": "he_v1_diagnostic_v1/primary/retained_energy",
                        "status": "success",
                        "artifacts": artifacts,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"namespace": "eval/perf", "metrics": {"seconds": 1.0}}),
                json.dumps({"namespace": "runtime", "metrics": {"cpu_seconds": 1.0}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_complete_row_reconciles_by_identity_and_content(tmp_path: Path) -> None:
    row = _row(tmp_path)
    manifest = _manifest(row)
    _materialize(tmp_path, manifest, row)
    result = collector.reconcile_row(
        manifest,
        row,
        results_root=tmp_path,
        plan_attempt_id="P1",
        launch_attempt_id="L1",
    )
    assert result["row_id"] == row["row_id"]
    assert result["comparison_kind"] == "primary_headline"
    assert result["artifact_count"] == 3
    assert {artifact["name"] for artifact in result["artifacts"]} == {
        "sampled_eval_table",
        "local_energy_trajectory_statistics",
        "conditioned_local_energy",
    }


def test_missing_or_short_raw_grid_fails_loudly(tmp_path: Path) -> None:
    row = _row(tmp_path)
    manifest = _manifest(row)
    run_dir = _materialize(tmp_path, manifest, row)
    index_path = run_dir / "diagnostics" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["tasks"][0]["artifacts"][0]["metadata"]["rows"] = 7
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(collector.DiagnosticCollectError, match="complete grid"):
        collector.reconcile_row(
            manifest,
            row,
            results_root=tmp_path,
            plan_attempt_id="P1",
            launch_attempt_id="L1",
        )


def test_collect_writes_failure_receipt_before_raising(tmp_path: Path) -> None:
    row = _row(tmp_path)
    manifest = _manifest(row)
    with pytest.raises(collector.DiagnosticCollectError, match="receipt="):
        collector.collect(
            manifest,
            results_root=tmp_path,
            plan_attempt_id="P1",
            launch_attempt_id="L1",
            collect_attempt_id="C1",
        )
    receipt = json.loads(
        (tmp_path / "04_collect" / "C1" / "collected.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "failed"
    assert receipt["n_collected_rows"] == 0
    assert receipt["errors"]
    assert receipt["checkpoint_reporting"] == "both_without_selection"
    assert receipt["selection_policy"] == "none"
