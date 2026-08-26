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
GRID_42_PATH = STUDY_DIR / "configs" / "eval_42.yaml"
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
    assert all(row["burn_in"] == 4 for row in manifest["rows"])
    assert all(row["discard_draws"] == 0 for row in manifest["rows"])
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


def test_frozen_grid_declares_42_rows_and_all_required_arms() -> None:
    grid = canary.load_grid(GRID_42_PATH)
    rows = grid["rows"]
    assert len(rows) == 42
    assert {step: sum(row["checkpoint_step"] == step for row in rows) for step in (25_000, 50_000)} == {
        25_000: 21, 50_000: 21
    }
    for step in (25_000, 50_000):
        subset = [row for row in rows if row["checkpoint_step"] == step]
        assert [row["seed"] for row in subset[:4]] == [1000, 1001, 1002, 1003]
        assert [row["seed"] for row in subset[4:8]] == [2000, 2001, 2002, 2003]
        assert [row["seed"] for row in subset[8:12]] == [3000, 3001, 3100, 3101]
        assert subset[12]["seed"] == 5000
        assert [row["seed"] for row in subset[13:20]] == list(range(4000, 4007))
        assert subset[20]["seed"] == 6000
    assert all(row["resources"]["constraint"] == "a100" for row in rows)
    assert sum("factor_response" in row["task_names"] for row in rows) == 4
    assert all(
        row["task_names"] == ["mcmc_energy"]
        for row in rows
        if row["seed"] in {1000, 1001, 1002, 1003, 2000, 2001, 2002, 2003, 3000, 3001, 3100, 3101}
    )


def test_frozen_grid_schema_matches_canary_consumer_constant() -> None:
    payload = yaml.safe_load(GRID_42_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == canary.GRID_SCHEMA
    assert canary.load_grid(GRID_42_PATH)["schema"] == canary.GRID_SCHEMA


def test_frozen_grid_rejects_a_real_gpu_constraint_drift() -> None:
    payload = yaml.safe_load(GRID_42_PATH.read_text(encoding="utf-8"))
    payload["rows"][0]["resources"]["constraint"] = None
    with pytest.raises(canary.CanaryError, match="constraint disagrees"):
        canary._load_grid_v2(payload)


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
    assert configured.evaluator.tasks[0].generator.discard_draws == row["discard_draws"]
    targets = [callback._target_ for callback in configured.callbacks]
    assert targets[-2:] == ["tpen.callback.ArtifactIndex", "tpen.callback.FailureLog"]


def test_runtime_transform_selects_declared_multi_task_row_and_factor_task_is_inert(
    tmp_path: Path,
) -> None:
    """An unselected factor declaration cannot enter the evaluator graph."""

    manifest, _, _ = _case(tmp_path)
    row = dict(manifest["rows"][0])
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)
    assert [task.name for task in configured.evaluator.tasks] == ["mcmc_energy"]
    assert all(task.name != "factor_response" for task in configured.evaluator.tasks)

    row["task_names"] = ["mcmc_energy", "factor_response"]
    cfg = collect.eval_stage.driver.build_config(
        STUDY_DIR.parents[2] / row["config"], row["overrides"]
    )
    configured = eval_stage.configure_canary_evaluation(cfg, row)
    assert [task.name for task in configured.evaluator.tasks] == [
        "mcmc_energy", "factor_response"
    ]


def test_runtime_transform_rejects_unknown_or_duplicate_declared_tasks(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _case(tmp_path)
    row = dict(manifest["rows"][0])
    for task_names, match in [
        (["does_not_exist"], "unknown evaluation task"),
        (["mcmc_energy", "mcmc_energy"], "must be unique"),
    ]:
        row["task_names"] = task_names
        cfg = collect.eval_stage.driver.build_config(
            STUDY_DIR.parents[2] / row["config"], row["overrides"]
        )
        with pytest.raises(eval_stage.driver.DriverError, match=match):
            eval_stage.configure_canary_evaluation(cfg, row)


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
                        "discard_draws": row["discard_draws"],
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
        "burn_in_draws": row["discard_draws"],
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
                "discard_draws",
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
            "burn_in_draws": row["discard_draws"],
        },
    }
    trajectory_path = task_dir / "trajectory_statistics.jsonl"
    trajectory_path.write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    proposal_scale = 0.5
    discarded_draws = [
        _sampler_diagnostic_draw(
            row,
            collection_index=index,
            region_index=index,
            proposal_scale=proposal_scale,
        )
        for index in range(row["discard_draws"])
    ]
    retained_draws = [
        _sampler_diagnostic_draw(
            row,
            collection_index=row["discard_draws"] + index,
            region_index=index,
            proposal_scale=proposal_scale,
        )
        for index in range(row["n_draws"])
    ]
    sampler_diagnostics = {
        "schema": "sampler_trajectory_diagnostics/v1",
        "n_walkers": row["n_walkers"],
        "draw_stride": row["stride"],
        "sampler_burn_in": row["burn_in"],
        "proposal_scale": proposal_scale,
        "intermediate_sampler_steps_observed": False,
        "intermediate_sampler_steps_unobserved_reason": (
            "collector sees only stride-spaced states"
        ),
        "sampler_internal_burn_in_states_observed": False,
        "discarded_draw_acceptance_rate_series": [
            draw["acceptance_rate"] for draw in discarded_draws
        ],
        "retained_draw_acceptance_rate_series": [
            draw["acceptance_rate"] for draw in retained_draws
        ],
        "discarded_draws": discarded_draws,
        "retained_draws": retained_draws,
        "metrics": {
            "trajectory_retained_draw_count": row["n_draws"],
            "trajectory_discarded_draw_count": row["discard_draws"],
            "trajectory_n_walkers": row["n_walkers"],
            "trajectory_draw_stride": row["stride"],
            "trajectory_sampler_burn_in": row["burn_in"],
            "trajectory_proposal_scale": proposal_scale,
            "trajectory_retained_value_count": row["record_capacity"],
            "trajectory_discarded_value_count": row["discard_draws"]
            * row["n_walkers"],
            "trajectory_retained_transition_count": row["record_capacity"]
            * row["stride"],
            "trajectory_discarded_transition_count": row["discard_draws"]
            * row["n_walkers"]
            * row["stride"],
            "trajectory_retained_draw_acceptance_rate_mean": 0.5,
            "trajectory_intermediate_sampler_steps_observed": False,
        },
    }
    sampler_diagnostics_path = task_dir / "sampler_trajectory_diagnostics.json"
    collect.layout.write_json(sampler_diagnostics_path, sampler_diagnostics)
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
                        "kind": "csv",
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
                        "kind": "trajectory_statistics_sidecar",
                        "path": str(trajectory_path.resolve()),
                        "metadata": {**identity, "status": "unresolved"},
                    },
                    {
                        "name": "sampler_trajectory_diagnostics",
                        "kind": "sampler_trajectory_diagnostics",
                        "path": str(sampler_diagnostics_path.resolve()),
                        "metadata": {
                            "schema": "sampler_trajectory_diagnostics/v1",
                            "retained_draw_count": row["n_draws"],
                            "discarded_draw_count": row["discard_draws"],
                            "draw_stride": row["stride"],
                            "intermediate_sampler_steps_observed": False,
                        },
                    },
                ],
            }
        ]
    }
    collect.layout.write_json(run_dir / "diagnostics" / "index.json", index)
    return row, source, result_dir, run_dir, row_record, metadata


def _sampler_diagnostic_draw(
    row: dict[str, Any],
    *,
    collection_index: int,
    region_index: int,
    proposal_scale: float,
) -> dict[str, Any]:
    return {
        "collection_index": collection_index,
        "region_index": region_index,
        "acceptance_rate": 0.5,
        "n_walkers": row["n_walkers"],
        "burn_in": row["burn_in"],
        "draw_stride": row["stride"],
        "transition_count": row["n_walkers"] * row["stride"],
        "proposal_scale": proposal_scale,
        "seed": row["seed"],
        "minimum_electron_nucleus_radius": 0.25,
    }


def _reconcile_written_outputs(
    manifest: dict[str, Any],
    written: tuple[dict[str, Any], Any, Path, Path, dict[str, Any], dict[str, Any]],
) -> list[str]:
    row, source, result_dir, run_dir, row_record, metadata = written
    return collect.reconcile_canary_row(
        row,
        source=source,
        result_dir=result_dir,
        run_dir=run_dir,
        plan_attempt_id=manifest["attempt_id"],
        manifest=manifest,
        row_record=row_record,
        metadata=metadata,
    )


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "diagnostics" / "index.json").read_text(encoding="utf-8"))


def _write_artifact_index(run_dir: Path, index: dict[str, Any]) -> None:
    collect.layout.write_json(run_dir / "diagnostics" / "index.json", index)


def _artifact(index: dict[str, Any], name: str) -> dict[str, Any]:
    return next(
        artifact
        for artifact in index["tasks"][0]["artifacts"]
        if artifact["name"] == name
    )


def _refresh_record_identity(run_dir: Path) -> None:
    task_dir = run_dir / "mcmc_energy"
    csv_path = task_dir / "sampled_eval_table.csv"
    csv_sha = plan.file_sha256(csv_path)
    record_metadata_path = task_dir / "sampled_eval_table.metadata.json"
    record_metadata = json.loads(record_metadata_path.read_text(encoding="utf-8"))
    record_metadata["csv_sha256"] = csv_sha
    record_metadata["byte_count"] = csv_path.stat().st_size
    collect.layout.write_json(record_metadata_path, record_metadata)
    index = _artifact_index(run_dir)
    metadata = _artifact(index, "sampled_eval_table")["metadata"]
    metadata["content_id"] = csv_sha
    metadata["bytes"] = csv_path.stat().st_size
    _write_artifact_index(run_dir, index)


def test_collection_accepts_exact_three_artifact_contract(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(
        tmp_path, manifest, source_map_path
    )
    run_dir = written[3]
    names = [artifact["name"] for artifact in _artifact_index(run_dir)["tasks"][0]["artifacts"]]

    assert set(names) == set(collect.CANARY_ARTIFACT_FILENAMES)
    assert len(names) == 3
    assert _reconcile_written_outputs(manifest, written) == []


def test_collection_reconciles_artifacts_for_each_declared_task(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append(
        {
            "name": "factor_response",
            "namespace": "eval/factor_response",
            "output_dir": str(factor_dir.resolve()),
            "status": "success",
            "artifacts": [
                {
                    "name": "factor_response_common_configuration",
                    "kind": "csv",
                    "path": str(factor_path.resolve()),
                    "metadata": {
                        "comparison_kind": "common_configuration",
                        "rows": row["record_capacity"] * 7,
                        "arm_count": 7,
                        "configuration_count": row["record_capacity"],
                        "model_state_restored": True,
                    },
                }
            ],
        }
    )
    _write_artifact_index(run_dir, index)

    assert _reconcile_written_outputs(manifest, written) == []


def test_collection_rejects_declared_task_namespace_mismatch(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    index["tasks"][0]["namespace"] = "eval/spatial_exchange_symmetry"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("namespace disagrees" in reason for reason in reasons)


def test_collection_rejects_factor_content_shape_mismatch(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append({
        "name": "factor_response", "namespace": "eval/factor_response",
        "output_dir": str(factor_dir.resolve()), "status": "success",
        "artifacts": [{
            "name": "factor_response_common_configuration", "kind": "csv",
            "path": str(factor_path.resolve()),
            "metadata": {"comparison_kind": "common_configuration", "rows": 1,
                         "arm_count": 1, "configuration_count": row["record_capacity"],
                         "model_state_restored": True},
        }],
    })
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("factor_response artifact metadata disagrees" in reason for reason in reasons)


def test_collection_rejects_a_declared_task_missing_from_artifact_index(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("artifact index task list disagrees" in reason for reason in reasons)


def test_collection_rejects_declared_task_artifact_outside_task_directory(
    tmp_path: Path,
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    row["task_names"].append("factor_response")
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evaluator"]["tasks"].append({"name": "factor_response"})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    factor_dir = run_dir / "factor_response"
    factor_dir.mkdir()
    factor_path = factor_dir / "factor_response.csv"
    factor_path.write_text("arm,energy\nbaseline,0\n", encoding="utf-8")
    index = _artifact_index(run_dir)
    index["tasks"].append(
        {
            "name": "factor_response",
            "namespace": "eval/factor_response",
            "output_dir": str(factor_dir.resolve()),
            "status": "success",
            "artifacts": [{"name": "factor_response", "kind": "csv", "path": str((tmp_path / "escaped.csv").resolve())}],
        }
    )
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("outside its indexed task directory" in reason for reason in reasons)


@pytest.mark.parametrize("artifact_name", sorted(collect.CANARY_ARTIFACT_FILENAMES))
@pytest.mark.parametrize("mutation", ["missing", "renamed"])
def test_collection_rejects_each_missing_or_renamed_artifact(
    tmp_path: Path, artifact_name: str, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    artifacts = index["tasks"][0]["artifacts"]
    if mutation == "missing":
        index["tasks"][0]["artifacts"] = [
            artifact for artifact in artifacts if artifact["name"] != artifact_name
        ]
    else:
        _artifact(index, artifact_name)["name"] = f"renamed_{artifact_name}"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("mcmc_energy artifacts mismatch" in reason for reason in reasons)


def test_collection_rejects_unknown_extra_artifact(tmp_path: Path) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    unknown = dict(index["tasks"][0]["artifacts"][0])
    unknown["name"] = "unknown_canary_artifact"
    index["tasks"][0]["artifacts"].append(unknown)
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("mcmc_energy artifacts mismatch" in reason for reason in reasons)


@pytest.mark.parametrize("artifact_name", sorted(collect.CANARY_ARTIFACT_FILENAMES))
def test_collection_rejects_artifact_provenance_mismatch(
    tmp_path: Path, artifact_name: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    index = _artifact_index(run_dir)
    _artifact(index, artifact_name)["kind"] = "wrong_kind"
    _write_artifact_index(run_dir, index)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("kind disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("channel", ["sampler_burn_in", "trajectory_discard"])
def test_collection_keeps_sampler_burn_in_and_trajectory_discard_distinct(
    tmp_path: Path, channel: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    config_path = run_dir / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if channel == "sampler_burn_in":
        config["evaluation_sampler"]["burn_in"] = row["discard_draws"]
    else:
        config["evaluator"]["tasks"][0]["generator"]["discard_draws"] = row["burn_in"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("resolved" in reason and "scale disagrees" in reason for reason in reasons)


@pytest.mark.parametrize(
    "field",
    ["draw_count", "walker_count", "row_count", "burn_in_draws"],
)
def test_collection_rejects_each_record_shape_mismatch(
    tmp_path: Path, field: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    path = run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json"
    record_metadata = json.loads(path.read_text(encoding="utf-8"))
    record_metadata[field] = (
        row["burn_in"] if field == "burn_in_draws" else record_metadata[field] + 1
    )
    collect.layout.write_json(path, record_metadata)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("record metadata disagrees" in reason for reason in reasons)


@pytest.mark.parametrize("coordinate", ["sample_index", "draw_index", "walker_index"])
def test_collection_rejects_each_record_coordinate_mismatch(
    tmp_path: Path, coordinate: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    csv_path = run_dir / "mcmc_energy" / "sampled_eval_table.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    records[0][coordinate] = "99"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _refresh_record_identity(run_dir)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("complete ordered draw/walker grid" in reason for reason in reasons)


@pytest.mark.parametrize(
    "field",
    ["walker_count", "draw_count", "total_draws", "draw_stride", "burn_in_draws"],
)
def test_collection_rejects_each_trajectory_statistics_shape_mismatch(
    tmp_path: Path, field: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    row, _, _, run_dir, _, _ = written
    path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["shape"][field] = (
        row["burn_in"] if field == "burn_in_draws" else receipt["shape"][field] + 1
    )
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("trajectory-statistics shape disagrees" in reason for reason in reasons)


@pytest.mark.parametrize(
    "mutation",
    ["retained_draw", "discarded_draw", "walker_count", "sampler_burn_in"],
)
def test_collection_rejects_sampler_diagnostics_shape_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]
    path = run_dir / "mcmc_energy" / "sampler_trajectory_diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "retained_draw":
        diagnostics["retained_draws"].pop()
    elif mutation == "discarded_draw":
        diagnostics["discarded_draws"].append(dict(diagnostics["retained_draws"][0]))
    elif mutation == "walker_count":
        diagnostics["n_walkers"] += 1
    else:
        diagnostics["sampler_burn_in"] = diagnostics["sampler_burn_in"] + 1
    collect.layout.write_json(path, diagnostics)

    reasons = _reconcile_written_outputs(manifest, written)

    assert any("diagnostics disagree" in reason for reason in reasons)


@pytest.mark.parametrize("mutation", ["trajectory_identity", "missing_record"])
def test_collection_reconciliation_fails_closed_on_partial_or_identity_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    manifest, source_map_path, _ = _case(tmp_path)
    written = _write_canary_outputs(tmp_path, manifest, source_map_path)
    run_dir = written[3]

    trajectory_path = run_dir / "mcmc_energy" / "trajectory_statistics.jsonl"
    if mutation == "trajectory_identity":
        receipt = json.loads(trajectory_path.read_text(encoding="utf-8"))
        receipt["checkpoint_sha256"] = "f" * 64
        trajectory_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    else:
        csv_path = run_dir / "mcmc_energy" / "sampled_eval_table.csv"
        csv_path.rename(csv_path.with_suffix(".preserved-partial.csv"))

    reasons = _reconcile_written_outputs(manifest, written)

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
