"""Focused contract tests for the two-row real-checkpoint He-v1 canary."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parent
GRID_PATH = STUDY_DIR / "configs" / "eval_canary.yaml"
EVALUATION_SHA = "a" * 40


def _load_collect() -> ModuleType:
    path = STUDY_DIR / "collect.py"
    spec = importlib.util.spec_from_file_location("he_v1_canary_collect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect = _load_collect()
canary = collect.canary
eval_stage = collect.eval_stage
launch = collect.launch_stage
plan = collect.plan_stage


def _checkpoint(root: Path, source_id: str, step: int) -> tuple[Path, dict[str, Any]]:
    checkpoint_dir = root / source_id
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(f"real model {step}\n".encode())
    (checkpoint_dir / "resolved_config.yaml").write_text(
        "experiment:\n  name: real-he-v1-training\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        # Deliberately independent of the checkpoint coordinate: skipped
        # updates are allowed, but this real count must remain immutable.
        "completed_updates": step - 3,
        "files": {"model": "model.pt", "resolved_config": "resolved_config.yaml"},
        "provenance": {"git_sha": "b" * 40, "tpen_version": "1.2.3"},
    }
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    source = {
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "next_iteration": step,
        "completed_updates": step - 3,
        "model_sha256": canary.file_sha256(checkpoint_dir / "model.pt"),
        "manifest_sha256": canary.file_sha256(checkpoint_dir / "manifest.json"),
        "complete_sha256": canary.file_sha256(checkpoint_dir / "COMPLETE"),
        "training_source_sha": "b" * 40,
    }
    return checkpoint_dir, source


def _case(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    source_root = tmp_path / "external-real-checkpoints"
    _, source_25 = _checkpoint(source_root, "actual-step-025000", 25_000)
    _, source_50 = _checkpoint(source_root, "actual-step-050000", 50_000)
    source_map = {
        "schema": canary.SOURCE_SCHEMA,
        "sources": {
            "actual-step-025000": source_25,
            "actual-step-050000": source_50,
        },
    }
    source_map_path = tmp_path / "checkpoint-sources.yaml"
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    grid = canary.load_grid(GRID_PATH)
    sources = canary.load_source_map(source_map_path)
    rows = canary.expand_rows(grid, canary.reconcile_grid_sources(grid, sources))
    manifest = canary.build_manifest(
        grid=grid,
        rows=rows,
        attempt_id="20260825T120000",
        results_root=tmp_path / "results",
        grid_path=GRID_PATH,
        source_map_path=source_map_path,
        evaluation_git_sha=EVALUATION_SHA,
        created_at="2026-08-25T12:00:00-04:00",
    )
    return manifest, source_map_path, source_map


def test_canary_expands_exactly_two_separately_addressable_energy_rows(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)

    assert manifest["n_rows"] == manifest["n_eval_rows"] == 2
    assert [row["checkpoint_step"] for row in manifest["rows"]] == [25_000, 50_000]
    assert [row["row_id"] for row in manifest["rows"]] == [
        "eval-canary-step000025000",
        "eval-canary-step000050000",
    ]
    assert all(row["task_names"] == ["mcmc_energy"] for row in manifest["rows"])
    assert all(row["record_capacity"] == 64 for row in manifest["rows"])
    assert all(row["depends_on"] == [] for row in manifest["rows"])
    assert all("checkpoint_dir" not in row for row in manifest["rows"])
    assert all(
        "checkpoint_dir" not in json.dumps(row["checkpoint_source"])
        for row in manifest["rows"]
    )
    assert canary.file_sha256(source_map_path) == manifest["source_map_sha256"]
    selected = launch.select_rows(
        manifest, kinds=[], row_ids=["eval-canary-step000050000"]
    )
    assert [row["checkpoint_step"] for row in selected] == [50_000]


@pytest.mark.parametrize(
    "fault, match",
    [
        ("partial", "source map ids mismatch"),
        ("model", "content mismatch"),
        ("manifest", "content mismatch"),
        ("complete", "content mismatch"),
        ("updates", "completed_updates mismatch"),
        ("training_sha", "training source SHA mismatch"),
    ],
)
def test_immutable_source_validation_fails_closed(
    tmp_path: Path, fault: str, match: str
) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    source_id = "actual-step-025000"
    checkpoint_dir = Path(source_map["sources"][source_id]["checkpoint_dir"])
    if fault == "partial":
        del source_map["sources"][source_id]
    elif fault in {"model", "manifest", "complete"}:
        filename = {"model": "model.pt", "manifest": "manifest.json", "complete": "COMPLETE"}[
            fault
        ]
        with (checkpoint_dir / filename).open("ab") as handle:
            handle.write(b"mutation")
    elif fault == "updates":
        source_map["sources"][source_id]["completed_updates"] -= 1
    else:
        source_map["sources"][source_id]["training_source_sha"] = "c" * 40
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    # This test targets source semantics rather than the map-file hash guard.
    manifest["source_map_sha256"] = canary.file_sha256(source_map_path)
    with pytest.raises(canary.CanaryError, match=match):
        canary.reconcile_manifest_sources(manifest, source_map_path)


def test_selected_row_launch_validates_both_sources_and_emits_exact_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    monkeypatch.setattr(launch, "_require_repo_identity", lambda *_: None)
    selected = launch.select_rows(
        manifest, kinds=[], row_ids=["eval-canary-step000050000"]
    )
    summary = launch.launch(
        manifest=manifest,
        results_root=tmp_path / "results",
        repo_root=tmp_path / "clean-checkout",
        rows=selected,
        launch_attempt_id="20260825T130000",
        uv_bin="/home/test/.local/bin/uv",
        uv_extras=["cu128"],
        uv_project_environment="/work/test/env",
        uv_cache_root="/work/test/cache",
        account="kozinsky_lab",
        submit=False,
        checkpoint_source_map=source_map_path,
    )
    assert [record["row_id"] for record in summary["rows"]] == [
        "eval-canary-step000050000"
    ]
    record = summary["rows"][0]
    script = Path(record["script_path"]).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_test" in script
    assert "#SBATCH --constraint" not in script
    assert f'test "$(git rev-parse HEAD)" = {EVALUATION_SHA}' in script
    assert 'git status --porcelain --untracked-files=no' in script
    assert record["command"][-2:] == ["--checkpoint-source-map", str(source_map_path)]

    # Selection cannot make an invalid unselected source disappear.
    Path(source_map["sources"]["actual-step-025000"]["checkpoint_dir"], "model.pt").write_bytes(
        b"changed"
    )
    with pytest.raises(launch.LaunchError, match="content mismatch"):
        launch.launch(
            manifest=manifest,
            results_root=tmp_path / "other-results",
            repo_root=tmp_path / "clean-checkout",
            rows=selected,
            launch_attempt_id="other",
            uv_bin="/home/test/.local/bin/uv",
            uv_extras=["cu128"],
            uv_project_environment="/work/test/env",
            uv_cache_root="/work/test/cache",
            account=None,
            submit=False,
            checkpoint_source_map=source_map_path,
        )


def test_runtime_transform_keeps_the_real_graph_and_only_reduces_scale(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _case(tmp_path)
    row = manifest["rows"][0]
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)

    assert len(configured.evaluator.tasks) == 1
    assert configured.evaluator.tasks[0].name == "mcmc_energy"
    assert configured.evaluation_sampler.n_walkers == row["n_walkers"]
    assert configured.evaluation_sampler.burn_in == row["burn_in"]
    assert configured.evaluation_sampler.n_steps == row["stride"]
    assert configured.evaluator.tasks[0].generator.n_draws == row["n_draws"]
    targets = [callback._target_ for callback in configured.callbacks]
    assert targets[-2:] == ["tpen.callback.ArtifactIndex", "tpen.callback.FailureLog"]


def _write_canary_outputs(
    tmp_path: Path,
    manifest: dict[str, Any],
    source_map_path: Path,
    *,
    row_index: int = 0,
) -> tuple[dict[str, Any], Any, Path, Path, dict[str, Any], dict[str, Any]]:
    row = manifest["rows"][row_index]
    sources = canary.reconcile_manifest_sources(manifest, source_map_path)
    source = canary.source_for_row(row, sources)
    result_dir = collect.layout.row_dir(
        tmp_path, row["stage"], row["row_id"], manifest["attempt_id"]
    )
    run_dir = result_dir / row["row_id"]
    task_dir = run_dir / "mcmc_energy"
    task_dir.mkdir(parents=True)
    (run_dir / "diagnostics").mkdir()

    collect.layout.write_json(result_dir / eval_stage.CHECKPOINT_BINDING_RECEIPT, source.receipt())
    row_record = {
        "row": row,
        "plan_attempt_id": manifest["attempt_id"],
        "launch_attempt_id": "launch-1",
        "job_id": "123",
    }
    metadata = {
        "run_id": row["row_id"],
        "git_commit": EVALUATION_SHA,
        "dirty_worktree": False,
    }
    collect.layout.write_json(result_dir / "row.json", row_record)
    collect.layout.write_json(run_dir / "metadata.json", metadata)

    replay = {
        "source_git_sha": source.training_source_sha,
        "source_tpen_version": row["checkpoint_source"]["source_tpen_version"],
        "checkpoint_schema_version": 2,
        "checkpoint_kind": "tpen.checkpoint",
        "checkpoint_model_sha256": source.model_sha256,
    }
    resolved_config = {
        "evaluator": {
            "tasks": [
                {
                    "name": "mcmc_energy",
                    "generator": {
                        "n_draws": row["n_draws"],
                        "chunk_size": row["chunk_size"],
                    },
                    "summaries": [
                        {
                            "_target_": "tpen.evaluation.summaries.SampledRecordWriter",
                            "max_samples": row["record_capacity"],
                        }
                    ],
                }
            ]
        },
        "evaluation_sampler": {
            "n_walkers": row["n_walkers"],
            "burn_in": row["burn_in"],
            "n_steps": row["stride"],
        },
        "load": {"replay_semantics": replay},
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )

    csv_path = task_dir / "sampled_eval_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_index", "draw_index", "walker_index"]
        )
        writer.writeheader()
        for draw in range(row["n_draws"]):
            for walker in range(row["n_walkers"]):
                writer.writerow(
                    {
                        "sample_index": draw * row["n_walkers"] + walker,
                        "draw_index": draw,
                        "walker_index": walker,
                    }
                )
    csv_sha = plan.file_sha256(csv_path)
    record_metadata = {
        "schema": "trajectory_records/v1",
        "row_semantics": "complete_draw_walker_grid",
        "observable": "local_energy",
        "draw_count": row["n_draws"],
        "walker_count": row["n_walkers"],
        "draw_stride": row["stride"],
        "burn_in_draws": row["burn_in"],
        "row_count": row["record_capacity"],
        "csv_sha256": csv_sha,
        "byte_count": csv_path.stat().st_size,
    }
    collect.layout.write_json(task_dir / "sampled_eval_table.metadata.json", record_metadata)

    config_sha = eval_stage.config_identity_hash(
        STUDY_DIR.parents[2] / row["config"],
        row["overrides"],
        identity_values={
            key: row.get(key)
            for key in (
                "canary_protocol",
                "checkpoint_source",
                "task_names",
                "n_walkers",
                "n_draws",
                "burn_in",
                "stride",
                "chunk_size",
                "record_capacity",
            )
        },
    )
    identity = {
        "stage": row["stage"],
        "run_id": row["row_id"],
        "attempt_id": manifest["attempt_id"],
        "checkpoint_sha256": source.model_sha256,
        "config_sha256": config_sha,
        "observable": "local_energy",
        "evaluator_id": eval_stage.EVALUATOR_ID,
    }
    trajectory = {
        **identity,
        "status": "unresolved",
        "shape": {
            "walker_count": row["n_walkers"],
            "draw_count": row["n_draws"],
            "total_draws": row["record_capacity"],
            "draw_stride": row["stride"],
            "burn_in_draws": row["burn_in"],
        },
    }
    trajectory_path = task_dir / "trajectory_statistics.jsonl"
    trajectory_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    index = {
        "tasks": [
            {
                "name": "mcmc_energy",
                "namespace": "eval/mcmc_energy",
                "output_dir": str(task_dir.resolve()),
                "status": "success",
                "artifacts": [
                    {
                        "name": "sampled_eval_table",
                        "path": str(csv_path.resolve()),
                        "metadata": {
                            "rows": row["record_capacity"],
                            "n_total": row["record_capacity"],
                            "draw_count": row["n_draws"],
                            "walker_count": row["n_walkers"],
                            "truncated": False,
                            "selection": "complete_draw_walker_grid",
                            "content_id": csv_sha,
                            "bytes": csv_path.stat().st_size,
                        },
                    },
                    {
                        "name": "local_energy_trajectory_statistics",
                        "path": str(trajectory_path.resolve()),
                        "metadata": {**identity, "status": "unresolved"},
                    },
                ],
            }
        ]
    }
    collect.layout.write_json(run_dir / "diagnostics" / "index.json", index)
    return row, source, result_dir, run_dir, row_record, metadata


@pytest.mark.parametrize("mutation", ["record_grid", "trajectory_identity", "missing_record"])
def test_collection_reconciliation_fails_closed_on_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    row, source, result_dir, run_dir, row_record, metadata = _write_canary_outputs(
        tmp_path, manifest, source_map_path
    )
    baseline = collect.reconcile_canary_row(
        row,
        source=source,
        result_dir=result_dir,
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
    )
    assert baseline == []

    csv_path = run_dir / "mcmc_energy" / "sampled_eval_table.csv"
    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    if mutation == "record_grid":
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        fields = lines[1].split(",")
        fields[0] = "99"
        lines[1] = ",".join(fields)
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif mutation == "trajectory_identity":
        receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
        receipt["checkpoint_sha256"] = "f" * 64
        trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    else:
        csv_path.rename(csv_path.with_suffix(".preserved-partial.csv"))

    reasons = collect.reconcile_canary_row(
        row,
        source=source,
        result_dir=result_dir,
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
    )
    assert reasons


def test_collection_refuses_partial_source_map_and_partial_row_set(tmp_path: Path) -> None:
    manifest, source_map_path, source_map = _case(tmp_path)
    del source_map["sources"]["actual-step-025000"]
    source_map_path.write_text(yaml.safe_dump(source_map, sort_keys=True), encoding="utf-8")
    manifest["source_map_sha256"] = canary.file_sha256(source_map_path)
    results_root = tmp_path / "results"
    plan.write_plan(manifest, results_root=results_root)
    with pytest.raises(collect.CollectError, match="incomplete"):
        collect.collect(
            results_root=results_root,
            plan_attempt_id=manifest["attempt_id"],
            collect_attempt_id="collect-1",
            gate_spec={},
            gate_spec_source="plan_manifest",
            checkpoint_source_map=source_map_path,
        )

    with pytest.raises(collect.CollectError, match="both 25k and 50k"):
        collect.require_complete_canary_collection(
            {
                "canary_complete": False,
                "rows": [
                    {"identity": {"row_id": "eval-canary-step000025000"}, "status": "pass"}
                ],
            }
        )
