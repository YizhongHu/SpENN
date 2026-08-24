"""Frozen-grid, real-checkpoint, and immutable-plan regression tests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parent
GRID_PATH = STUDY_DIR / "configs" / "diagnostic_grid.yaml"
EVALUATION_SHA = "a" * 40


def _load(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"he_v1_diagnostic_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = _load("diagnostic_plan")


def _checkpoint(
    root: Path,
    *,
    label: str,
    completed_updates: int,
) -> tuple[Path, dict[str, str | int]]:
    path = root / label
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(f"weights-{label}".encode())
    (path / "resolved_config.yaml").write_text("model: {}\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "completed_updates": completed_updates,
        "files": {"model": "model.pt", "resolved_config": "resolved_config.yaml"},
        "provenance": {
            "git_sha": "418accf153368aab45586dc2a2cc97c18472691c",
            "tpen_version": "0.1.0",
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return path, {
        "label": label,
        "completed_updates": completed_updates,
        "model_sha256": plan.file_sha256(path / "model.pt"),
        "manifest_sha256": plan.file_sha256(path / "manifest.json"),
    }


def _inputs(tmp_path: Path) -> tuple[dict, Path, list[dict]]:
    grid = plan.load_grid(GRID_PATH)
    first_path, first = _checkpoint(
        tmp_path, label="step_025000", completed_updates=25000
    )
    second_path, second = _checkpoint(
        tmp_path, label="step_050000", completed_updates=50000
    )
    grid["checkpoints"] = [first, second]
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(
        yaml.safe_dump(
            {
                "schema": plan.SOURCE_SCHEMA,
                "checkpoints": {
                    "step_025000": str(first_path),
                    "step_050000": str(second_path),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoints = plan.reconcile_checkpoint_sources(
        grid, plan.load_sources(sources_path)
    )
    return grid, sources_path, checkpoints


def _manifest(tmp_path: Path, *, scale: str = "production") -> dict:
    grid, sources, checkpoints = _inputs(tmp_path)
    return plan.build_manifest(
        grid,
        checkpoints,
        grid_path=GRID_PATH,
        source_map_path=sources,
        scale=scale,
        evaluation_git_sha=EVALUATION_SHA,
        created_at="2026-08-24T12:00:00-04:00",
    )


def test_frozen_grid_expands_every_declared_row_for_both_checkpoints(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    assert len(manifest["rows"]) == 42
    for label in ("step_025000", "step_050000"):
        rows = [row for row in manifest["rows"] if row["checkpoint_label"] == label]
        assert len(rows) == 21
        assert sum(row["profile"] == "retained_energy" for row in rows) == 12
        assert sum(row["profile"] == "reequilibrated_energy" for row in rows) == 7
        assert sum(row["profile"] == "common_factor_response" for row in rows) == 1
        assert sum(row["profile"] == "checkpoint_diagnostics" for row in rows) == 1


def test_smoke_changes_only_declared_scale_and_resources(tmp_path: Path) -> None:
    production = _manifest(tmp_path / "production")
    smoke = _manifest(tmp_path / "smoke", scale="smoke")
    assert smoke["scale_overrides"]
    assert production["scale_overrides"] == {}
    assert [row["row_id"] for row in smoke["rows"]] == [
        row["row_id"] for row in production["rows"]
    ]
    for full, small in zip(production["rows"], smoke["rows"]):
        for key in (
            "profile",
            "task_names",
            "protocol",
            "comparison_kind",
            "seed",
            "factor_arm",
            "checkpoint_model_sha256",
        ):
            assert small[key] == full[key]
        assert small["resources"]["partition"] == "gpu_test"
        assert small["n_walkers"] <= full["n_walkers"]


@pytest.mark.parametrize("bad_sha", ["short", "A" * 40, "g" * 40])
def test_manifest_requires_full_lowercase_evaluator_sha(
    tmp_path: Path, bad_sha: str
) -> None:
    grid, sources, checkpoints = _inputs(tmp_path)
    with pytest.raises(plan.DiagnosticPlanError, match="full Git SHA"):
        plan.build_manifest(
            grid,
            checkpoints,
            grid_path=GRID_PATH,
            source_map_path=sources,
            scale="production",
            evaluation_git_sha=bad_sha,
        )


def test_partial_and_content_mismatched_checkpoints_fail_before_plan(
    tmp_path: Path,
) -> None:
    grid, sources_path, _checkpoints = _inputs(tmp_path)
    sources = plan.load_sources(sources_path)
    (sources["step_025000"] / "COMPLETE").unlink()
    with pytest.raises(plan.DiagnosticPlanError, match="incomplete"):
        plan.reconcile_checkpoint_sources(grid, sources)
    (sources["step_025000"] / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (sources["step_050000"] / "model.pt").write_bytes(b"changed")
    with pytest.raises(plan.DiagnosticPlanError, match="content mismatch"):
        plan.reconcile_checkpoint_sources(grid, sources)


def test_real_format_manifest_and_resolved_config_are_required(tmp_path: Path) -> None:
    grid, sources_path, _checkpoints = _inputs(tmp_path)
    sources = plan.load_sources(sources_path)
    manifest_path = sources["step_025000"] / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["kind"] = "legacy"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    grid["checkpoints"][0]["manifest_sha256"] = plan.file_sha256(manifest_path)
    with pytest.raises(plan.DiagnosticPlanError, match="real format"):
        plan.reconcile_checkpoint_sources(grid, sources)


def test_plan_attempt_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "inputs")
    production_grid_before = plan.file_sha256(
        STUDY_DIR / "configs" / "production_grid.yaml"
    )
    plan.write_plan(manifest, results_root=tmp_path / "results", attempt_id="P1")
    with pytest.raises(FileExistsError):
        plan.write_plan(manifest, results_root=tmp_path / "results", attempt_id="P1")
    assert plan.file_sha256(STUDY_DIR / "configs" / "production_grid.yaml") == (
        production_grid_before
    )
    path = tmp_path / "results" / "00_plan" / "P1" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(plan.DiagnosticPlanError, match="hash mismatch"):
        plan.read_manifest(tmp_path / "results", "P1")


def test_unknown_grid_key_and_changed_frozen_arm_fail(tmp_path: Path) -> None:
    payload = yaml.safe_load(GRID_PATH.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(plan.DiagnosticPlanError, match="keys mismatch"):
        plan.load_grid(path)
    changed = copy.deepcopy(payload)
    changed.pop("surprise")
    changed["factor_arms"][1]["b_ee"] = 0.8
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(plan.DiagnosticPlanError, match="coordinates changed"):
        plan.load_grid(path)
