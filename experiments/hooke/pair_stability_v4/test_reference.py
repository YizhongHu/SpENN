"""Tests for V4-0 completed-lineage audit contracts."""

from __future__ import annotations

import csv
import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import audit  # noqa: E402
import reference  # noqa: E402
import roots  # noqa: E402
from experiments.toolkit import (  # noqa: E402
    CompletionSpec,
    ExecutionRecord,
    ResourceSpec,
    StagePlan,
    TaskSpec,
    write_execution_records,
)


@pytest.fixture(autouse=True)
def _test_reference_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep freeze tests in a private, direct-child reference namespace."""

    monkeypatch.setattr(reference, "REFERENCE_OWNER_ROOT", tmp_path.absolute())


def test_completed_lineage_audit_accepts_real_shaped_v4_fixture(
    tmp_path: Path,
) -> None:
    """Typed plans, submissions, terminal outputs, and populations all agree."""

    root, attempts = _completed_lineage(tmp_path)

    assert audit.audit_completed_lineage(root, attempts=attempts) == ()
    assert (
        root
        / "05_final_grid"
        / attempts["final_grid"]
        / "manifest.json"
    ).is_file()
    for stage in audit.STAGE_EXPECTATIONS:
        assert (
            root
            / stage
            / "stage_plans"
            / attempts["grid"]
            / "stage_manifest.json"
        ).is_file()
    final_eval_plan = (
        root
        / "07_final_eval"
        / "stage_plans"
        / attempts["final_eval"]
        / "tasks.jsonl"
    )
    final_eval_task = json.loads(final_eval_plan.read_text().splitlines()[0])
    planned_job = final_eval_task["metadata"]["job"]
    assert "source_champion_id" not in planned_job
    assert set(planned_job) == {
        "final_run_id",
        "final_grid_attempt_id",
        "final_train_attempt_id",
        "final_eval_attempt_id",
        "final_eval_attempt_dir",
        "checkpoint",
        "command",
        "command_parts",
    }


def test_completed_lineage_audit_keeps_submission_and_terminal_facts_distinct(
    tmp_path: Path,
) -> None:
    """Submission identity, launcher completion, and run completion fail alone."""

    root, attempts = _completed_lineage(tmp_path)
    plan_dir = root / "01_train" / "stage_plans" / attempts["train"]
    records_path = plan_dir / "execution_records.jsonl"
    original_records = records_path.read_text()
    records = [json.loads(line) for line in original_records.splitlines()]
    records[0]["run_id"] = "wrong-run"
    records_path.write_text(
        "\n".join(json.dumps(row) for row in records) + "\n"
    )
    assert any(
        "invalid ExecutionRecord" in error
        or "identity sets differ" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    records_path.write_text(original_records)

    first_task = json.loads((plan_dir / "tasks.jsonl").read_text().splitlines()[0])
    launcher = Path(first_task["logs"][0])
    original_launcher = launcher.read_text()
    _write_json(launcher, {"status": "running", "returncode": None})
    assert any(
        "launcher status is not success" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    launcher.write_text(original_launcher)

    status = Path(first_task["completion"]["status_path"])
    original_status = status.read_text()
    _write_json(status, {"status": "running"})
    assert any(
        "run status is not completed" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    status.write_text(original_status)

    run_start_path = Path(first_task["result_dir"]) / "run_start.json"
    original_run_start = run_start_path.read_text()
    run_start = json.loads(original_run_start)
    run_start["environment"] = {}
    _write_json(run_start_path, run_start)
    assert any(
        "run_start environment lacks" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    run_start_path.write_text(original_run_start)


def test_completed_lineage_audit_rejects_checkpoint_metric_resource_and_population_drift(
    tmp_path: Path,
) -> None:
    """Independent scientific readiness and profile mutations are detected."""

    root, attempts = _completed_lineage(tmp_path)
    train_plan = root / "01_train" / "stage_plans" / attempts["train"]
    train_task = json.loads(
        (train_plan / "tasks.jsonl").read_text().splitlines()[0]
    )
    pointer = Path(train_task["completion"]["checkpoint_path"])
    original_pointer = pointer.read_text()
    pointer.write_text(
        json.dumps(
            {
                "checkpoint_dir": "step_000000",
                "step": 1,
                "created_at_unix": 0.0,
            }
        )
    )
    assert any(
        "checkpoint pointer is not concretely complete" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    pointer.write_text(original_pointer)

    eval_plan = root / "02_validation" / "stage_plans" / attempts["validation"]
    eval_task = json.loads(
        (eval_plan / "tasks.jsonl").read_text().splitlines()[0]
    )
    metrics = Path(eval_task["result_dir"]) / "metrics.jsonl"
    original_metrics = metrics.read_text()
    metrics.write_text("")
    assert any(
        "no valid nonempty metrics" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    metrics.write_text(original_metrics)

    metric_rows = [
        json.loads(line)
        for line in original_metrics.splitlines()
        if line.strip()
    ]
    metric_rows.insert(
        1,
        {
            "namespace": "eval/status",
            "step": 0,
            "metrics": {
                "suite_success": False,
                "suite_failed": True,
            },
        },
    )
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows)
    )
    duplicate_errors = audit.audit_completed_lineage(
        root,
        attempts=attempts,
    )
    assert any(
        "suite_failed=true" in error for error in duplicate_errors
    )
    assert any(
        "eval/status count is not exactly one" in error
        for error in duplicate_errors
    )
    metrics.write_text(original_metrics)

    tasks_path = train_plan / "tasks.jsonl"
    original_tasks = tasks_path.read_text()
    tasks = [json.loads(line) for line in original_tasks.splitlines()]
    tasks[0]["resources"]["partition"] = "seas_gpu"
    tasks_path.write_text("\n".join(json.dumps(row) for row in tasks) + "\n")
    assert any(
        "resources differ" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    tasks_path.write_text(original_tasks)

    grid_path = root / "00_grid" / attempts["grid"] / "manifest.json"
    grid = json.loads(grid_path.read_text())
    grid["jobs"][0]["run_id"] = "same-count-wrong-run"
    _write_json(grid_path, grid)
    assert any(
        "run-id population differs" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )


def test_science_collection_runtime_and_selector_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    """Science rows, CSV projection, runtime, and selection are independent."""

    root, attempts = _completed_lineage(tmp_path)
    validation_plan = (
        root / "02_validation" / "stage_plans" / attempts["validation"]
    )
    first_task = json.loads(
        (validation_plan / "tasks.jsonl").read_text().splitlines()[0]
    )
    result_dir = Path(first_task["result_dir"])

    metrics_path = result_dir / "metrics.jsonl"
    original_metrics = metrics_path.read_text()
    metric_rows = [
        json.loads(line) for line in original_metrics.splitlines() if line
    ]
    science_row = next(
        row
        for row in metric_rows
        if row["namespace"] == "eval/cusp"
    )
    metrics_path.write_text(
        original_metrics + json.dumps(science_row) + "\n"
    )
    assert any(
        "eval/cusp science count is not exactly one" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    metrics_path.write_text(original_metrics)

    perf_row = next(
        row
        for row in metric_rows
        if row["namespace"] == "eval/perf/cusp"
    )
    metrics_path.write_text(
        original_metrics + json.dumps(perf_row) + "\n"
    )
    assert any(
        "eval/perf/cusp count is not exactly one" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    metrics_path.write_text(original_metrics)

    metadata_path = result_dir / "metadata.json"
    original_metadata = metadata_path.read_text()
    metadata = json.loads(original_metadata)
    metadata["runtime"]["cuda_visible_devices"] = "different-device"
    _write_json(metadata_path, metadata)
    assert any(
        "CUDA_VISIBLE_DEVICES differs" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    metadata_path.write_text(original_metadata)

    collect_dir = root / "03_collect" / attempts["collection"]
    failures_path = collect_dir / "failures.csv"
    original_failures = failures_path.read_bytes()
    _write_csv(failures_path, [], ("wrong_header",))
    assert any(
        "failures.csv header differs" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    failures_path.write_bytes(original_failures)

    summary_path = collect_dir / "summary.csv"
    original_summary = summary_path.read_bytes()
    with summary_path.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    for row in summary_rows:
        row["eval/unconfigured/value"] = "1"
    _write_csv(
        summary_path,
        summary_rows,
        tuple(summary_rows[0]),
    )
    assert any(
        "header differs from replayed projection" in error
        or "metric_columns differ from replayed projection" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    summary_path.write_bytes(original_summary)

    selection_path = (
        root
        / "04_select"
        / attempts["selection"]
        / "selection_report.json"
    )
    original_selection = selection_path.read_text()
    selection = json.loads(original_selection)
    selection["configs"][0]["config_id"] = "tampered-config"
    _write_json(selection_path, selection)
    assert any(
        "selection replay differs for report field configs" in error
        for error in audit.audit_completed_lineage(root, attempts=attempts)
    )
    selection_path.write_text(original_selection)


def test_reference_evidence_receipt_detects_mid_freeze_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw worker evidence cannot change after audit but before publication."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    plan = (
        source
        / "02_validation"
        / "stage_plans"
        / attempts["validation"]
        / "tasks.jsonl"
    )
    first_task = json.loads(plan.read_text().splitlines()[0])
    target = Path(first_task["result_dir"]) / "metrics.jsonl"
    original_store = reference._store_artifact
    mutated = False

    def mutate_after_copy(*args: object, **kwargs: object) -> object:
        nonlocal mutated
        artifact = original_store(*args, **kwargs)
        if not mutated:
            target.write_text(
                target.read_text()
                + json.dumps(
                    {
                        "namespace": "eval/cusp",
                        "step": 1,
                        "metrics": {"local_energy_mean": 99.0},
                    }
                )
                + "\n"
            )
            mutated = True
        return artifact

    monkeypatch.setattr(reference, "_store_artifact", mutate_after_copy)
    with pytest.raises(RuntimeError, match="raw audit evidence mutated"):
        reference.freeze_reference(
            source,
            (tmp_path / "reference").absolute(),
            attempts=attempts,
        )


@pytest.mark.parametrize("operation", ("add", "remove", "rename"))
def test_reference_directory_receipt_detects_entry_population_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Directory add/remove/rename races cannot escape the freeze read set."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    grid_jobs = source / "00_grid" / attempts["grid"] / "jobs"
    final_eval_plan = (
        source
        / "07_final_eval"
        / "stage_plans"
        / attempts["final_eval"]
        / "tasks.jsonl"
    )
    first_task = json.loads(final_eval_plan.read_text().splitlines()[0])
    diagnostic_index = json.loads(
        (
            Path(first_task["result_dir"])
            / "diagnostics"
            / "index.json"
        ).read_text()
    )
    diagnostic_artifact = Path(
        diagnostic_index["tasks"][0]["artifacts"][0]["path"]
    )
    original_figures = reference._figure_metadata
    mutated = False

    def mutate_after_first_receipt(*args: object, **kwargs: object) -> object:
        nonlocal mutated
        figures = original_figures(*args, **kwargs)
        if not mutated:
            if operation == "add":
                (grid_jobs / "unexpected.json").write_text("{}\n")
            elif operation == "remove":
                diagnostic_artifact.unlink()
            else:
                diagnostic_artifact.rename(
                    diagnostic_artifact.with_name("renamed.csv")
                )
            mutated = True
        return figures

    monkeypatch.setattr(reference, "_figure_metadata", mutate_after_first_receipt)
    with pytest.raises(
        RuntimeError,
        match="source read set mutated before reference publication",
    ):
        reference.freeze_reference(
            source,
            (tmp_path / "reference").absolute(),
            attempts=attempts,
        )


def test_reference_verifier_rejects_audit_evidence_digest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime assignments and every selector digest are integrity checked."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "reference").absolute(),
        attempts=attempts,
    )
    descriptor_path = frozen / "reference.json"
    original = descriptor_path.read_text()

    descriptor = json.loads(original)
    descriptor["audit_evidence"]["worker_runtime"]["assignments"][0][
        "profile_sha256"
    ] = "b" * 64
    _write_json(descriptor_path, descriptor)
    assert any(
        "worker-runtime" in error
        for error in reference.verify_reference(frozen)
    )

    selector_mutations = (
        (("contract_sha256",), "b" * 64),
        (("producer", "source_sha256"), "b" * 64),
        (("replay_sha256",), "b" * 64),
        (("policy_sha256",), "b" * 64),
        (("artifact_sha256", "summary_csv"), "b" * 64),
        (("verifier", "source_sha256"), "b" * 64),
        (("verifier", "id"), "unknown-selector-version"),
    )
    for path, replacement in selector_mutations:
        descriptor = json.loads(original)
        target = descriptor["audit_evidence"]["selection"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        _write_json(descriptor_path, descriptor)
        assert any(
            "selection" in error or "selector" in error
            for error in reference.verify_reference(frozen)
        ), path
    descriptor_path.write_text(original)


def test_selector_v1_is_independent_and_freeze_cross_checks_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 verifies without toolkit replay, while freeze requires agreement."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "reference").absolute(),
        attempts=attempts,
    )
    producer_replay = audit.replay_selection

    def evolving_selector_must_not_run(*args, **kwargs):
        raise AssertionError("mutable producer was called during V1 verification")

    monkeypatch.setattr(audit, "replay_selection", evolving_selector_must_not_run)
    assert reference.verify_reference(frozen) == ()

    monkeypatch.setattr(audit, "replay_selection", producer_replay)
    source_two, attempts_two = _v3_reference_source(
        tmp_path / "second",
        monkeypatch,
    )

    def incompatible_producer(*args, **kwargs):
        replay = producer_replay(*args, **kwargs)
        return {**replay, "overall_champion": "producer-drift"}

    monkeypatch.setattr(audit, "replay_selection", incompatible_producer)
    with pytest.raises(
        ValueError,
        match="producer replay differs from immutable V1 verifier",
    ):
        reference.freeze_reference(
            source_two,
            (tmp_path / "reference-two").absolute(),
            attempts=attempts_two,
        )


def test_reference_contracts_derive_read_set_science_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coherently rehashed omissions cannot redefine frozen V1 expectations."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "reference").absolute(),
        attempts=attempts,
    )
    descriptor_path = frozen / "reference.json"
    original = descriptor_path.read_text()

    descriptor = json.loads(original)
    receipt = descriptor["evidence_inputs"]
    required_suffixes = {
        "status.json",
        "launcher_status.json",
        "source_final_job.json",
        "source_final_train_attempt.json",
        "evaluated_checkpoint.json",
        "selected_checkpoint.json",
        "COMPLETE",
        "manifest.json",
        "model.pt",
        "summary.png",
    }
    observed_suffixes = {
        Path(row["source_path"]).name for row in receipt["files"]
    }
    assert required_suffixes <= observed_suffixes
    assert {row["role"] for row in receipt["files"]} == {
        "protected_artifact",
        "audit_evidence",
    }
    assert {
        row["role"] for row in receipt["directory_projections"]
    } == {
        "grid_jobs",
        "final_grid_jobs",
        "02_validation_diagnostic_output",
        "07_final_eval_diagnostic_output",
    }

    receipt["files"] = receipt["files"][1:]
    receipt["file_count"] = len(receipt["files"])
    receipt["aggregate_sha256"] = reference.canonical_sha256(
        receipt["files"]
    )
    _write_json(descriptor_path, descriptor)
    assert any(
        "path/role population differs" in error
        for error in reference.verify_reference(frozen)
    )

    descriptor = json.loads(original)
    receipt = descriptor["evidence_inputs"]
    receipt["directory_projections"] = receipt[
        "directory_projections"
    ][1:]
    receipt["directory_aggregate_sha256"] = reference.canonical_sha256(
        receipt["directory_projections"]
    )
    _write_json(descriptor_path, descriptor)
    assert any(
        "directory projections differ" in error
        for error in reference.verify_reference(frozen)
    )

    descriptor = json.loads(original)
    science = descriptor["audit_evidence"]["science_metrics"]
    del science["stages"]["02_validation"]["cusp"]
    science["schema_sha256"] = reference.canonical_sha256(
        science["stages"]
    )
    _write_json(descriptor_path, descriptor)
    assert any(
        "science-metric task population differs" in error
        for error in reference.verify_reference(frozen)
    )

    descriptor = json.loads(original)
    runtime = descriptor["audit_evidence"]["worker_runtime"]
    profile_row = runtime["stages"]["01_train"]["profiles"][0]
    profile_row["profile"]["dtype"] = "float32"
    changed_digest = reference.canonical_sha256(profile_row["profile"])
    original_digest = profile_row["profile_sha256"]
    profile_row["profile_sha256"] = changed_digest
    for assignment in runtime["assignments"]:
        if assignment["profile_sha256"] == original_digest:
            assignment["profile_sha256"] = changed_digest
    runtime["aggregate_sha256"] = reference.canonical_sha256(
        sorted(
            (
                row["stage"],
                row["run_id"],
                row["profile_sha256"],
            )
            for row in runtime["assignments"]
        )
    )
    _write_json(descriptor_path, descriptor)
    assert any(
        "dtype is not float64" in error
        for error in reference.verify_reference(frozen)
    )
    descriptor_path.write_text(original)


def test_lineage_root_resolution_distinguishes_v3_reference_and_v4_purposes(
    tmp_path: Path,
) -> None:
    """External v3 references need no sentinel; v4 audit roots cannot freeze."""

    v3_root = (tmp_path / "v3-reference").absolute()
    v3_root.mkdir()
    assert audit._resolve_lineage_root(v3_root) == (
        v3_root.resolve(),
        "pair_stability_v3",
    )

    v4_root = roots.initialize_root(
        (tmp_path / "v4-candidate").absolute(),
        lineage_id="lineage",
    )
    assert audit._resolve_lineage_root(v4_root) == (
        v4_root,
        "pair_stability_v4",
    )

    ownership_root = roots.initialize_root(
        (tmp_path / "ownership").absolute(),
        lineage_id="lineage",
        purpose=roots.PURPOSE_OWNERSHIP_AUDIT,
    )
    with pytest.raises(ValueError, match="purpose"):
        audit._resolve_lineage_root(ownership_root)


def test_reference_freeze_is_repeatable_and_independently_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freezing the same audited source twice yields the same portable bytes."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    first = reference.freeze_reference(
        source,
        (tmp_path / "reference-a").absolute(),
        attempts=attempts,
    )
    second = reference.freeze_reference(
        source,
        (tmp_path / "reference-b").absolute(),
        attempts=attempts,
    )

    assert reference.verify_reference(first) == ()
    assert reference.verify_reference(second) == ()
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_reference_captures_real_clean_git_provenance_before_owner_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real clean checkout is sampled before V4-owned sibling staging exists.

    The science fixture still substitutes its V3-only audit mechanics, but this
    test deliberately leaves ``_source_provenance`` unmocked.  It proves the
    ordering needed when the default reference owner itself lives in a checkout:
    provenance is captured while clean, then the sibling staging/publication
    makes that checkout dirty without invalidating the frozen reference.
    """

    real_source_provenance = reference._source_provenance
    real_lineage_summary = reference._lineage_summary
    source, attempts = _v3_reference_source(tmp_path / "source", monkeypatch)
    clean_repo = tmp_path / "clean-repo"
    clean_repo.mkdir()
    (clean_repo / "tracked.txt").write_text("tracked\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "tracked.txt"],
        [
            "git",
            "-c",
            "user.name=V4 Test",
            "-c",
            "user.email=v4-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
    ):
        subprocess.run(command, cwd=clean_repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clean_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=clean_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    legacy = {
        "schema_version": "pair-stability-v4/legacy-source/v1",
        "manifest_path": "legacy-source.json",
        "manifest_sha256": "a" * 64,
        "closure_sha256": "b" * 64,
        "files": [{"path": "legacy.py", "sha256": "c" * 64}],
    }
    runtime = {
        "schema_version": "pair-stability-v4/runtime-source/v1",
        "closure_sha256": "d" * 64,
        "n_files": 1,
        "git_commit": commit,
        "git_branch": branch,
        "dirty": False,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "uv_project_environment": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
    }
    config = {
        "schema_version": "pair-stability-v4/config-source/v1",
        "closure_sha256": "e" * 64,
        "files": [{"path": "config.yaml", "sha256": "f" * 64}],
    }
    owner = clean_repo / "reference" / "v3_smoke"
    monkeypatch.setattr(reference, "REPO_ROOT", clean_repo)
    monkeypatch.setattr(reference, "REFERENCE_OWNER_ROOT", owner)
    monkeypatch.setattr(reference, "_source_provenance", real_source_provenance)
    monkeypatch.setattr(reference, "legacy_source_receipt", lambda _root: legacy)
    monkeypatch.setattr(reference, "runtime_source_receipt", lambda _root: runtime)
    monkeypatch.setattr(reference, "config_source_receipt", lambda _root: config)
    monkeypatch.setattr(reference, "_require_worker_commit", lambda *_args: None)
    monkeypatch.setattr(
        reference,
        "_lineage_summary",
        lambda root, selected_attempts: {
            **real_lineage_summary(root, selected_attempts),
            "worker_commits": [commit],
        },
    )

    frozen = reference.freeze_reference(
        source,
        owner / "clean-reference",
        attempts=attempts,
    )
    descriptor = json.loads((frozen / "reference.json").read_text())
    assert descriptor["source"] == {
        "commit": commit,
        "branch": branch,
        "dirty": False,
    }
    assert reference.verify_reference(frozen) == ()
    assert subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=clean_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.startswith("?? reference/")


def test_reference_comparison_contract_pins_static_layout_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen references fail when their reviewed map binding is changed."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "reference").absolute(),
        attempts=attempts,
    )
    descriptor_path = frozen / "reference.json"
    descriptor = json.loads(descriptor_path.read_text())
    contract = descriptor["comparison_contract"]
    assert contract == reference._comparison_contract()
    contract["layout_map_sha256"] = "0" * 64
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    )

    assert any(
        "comparison contract/layout digest mismatch" in error
        for error in reference.verify_reference(frozen)
    )


def test_reference_cli_has_typed_freeze_verify_and_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reference creation is explicit; verification mismatches return one."""

    attempts = {
        key: "lineage-a"
        for key in (
            "grid",
            "train",
            "validation",
            "collection",
            "selection",
            "final_grid",
            "final_train",
            "final_eval",
            "final_collect",
            "report",
        )
    }
    destination = (tmp_path / "reference").absolute()
    monkeypatch.setattr(
        reference,
        "freeze_reference",
        lambda root, target, *, attempts: destination,
    )
    assert (
        reference.main(
            [
                "freeze",
                "--results-root",
                str((tmp_path / "source").absolute()),
                "--destination",
                str(destination),
                "--attempts",
                json.dumps(attempts),
            ]
        )
        == 0
    )

    monkeypatch.setattr(reference, "verify_reference", lambda _path: ())
    assert (
        reference.main(["verify", "--reference", str(destination)])
        == 0
    )
    monkeypatch.setattr(
        reference,
        "verify_reference",
        lambda _path: ("digest mismatch",),
    )
    assert (
        reference.main(["verify", "--reference", str(destination)])
        == 1
    )
    with pytest.raises(SystemExit) as exc_info:
        reference.main(
            [
                "freeze",
                "--results-root",
                str(tmp_path),
                "--destination",
                str(destination),
                "--attempts",
                "{}",
            ]
        )
    assert exc_info.value.code == 2


def test_reference_owner_requires_one_new_direct_child(tmp_path: Path) -> None:
    """The fixed production owner admits neither traversal nor replacement."""

    assert "reference_owner" not in inspect.signature(reference.freeze_reference).parameters
    with pytest.raises(ValueError, match="direct child"):
        reference._reference_destination(
            (tmp_path / "nested" / "reference").absolute(),
            owner=reference.REFERENCE_OWNER_ROOT,
        )
    existing = (tmp_path / "existing").absolute()
    existing.mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        reference._reference_destination(
            existing,
            owner=reference.REFERENCE_OWNER_ROOT,
        )
    escaped = (tmp_path / "escaped").absolute()
    escaped.symlink_to(tmp_path / "outside")
    with pytest.raises(FileExistsError, match="exists"):
        reference._reference_destination(
            escaped,
            owner=reference.REFERENCE_OWNER_ROOT,
        )


def test_reference_verifier_rejects_descriptor_and_file_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected completeness is derived from frozen manifests, not row trust."""

    source, attempts = _v3_reference_source(tmp_path, monkeypatch)
    frozen = reference.freeze_reference(
        source,
        (tmp_path / "reference").absolute(),
        attempts=attempts,
    )
    descriptor_path = frozen / "reference.json"
    descriptor = json.loads(descriptor_path.read_text())
    target = next(
        row
        for row in descriptor["artifacts"]
        if row["logical_path"] == "00_grid/{grid}/commands.sh"
    )
    descriptor["artifacts"].remove(target)
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n"
    )
    (frozen / target["stored_path"]).rename(tmp_path / "held-commands.sh")

    errors = reference.verify_reference(frozen)

    assert any("inventory contract mismatch" in error for error in errors)
    assert any("protected logical-path population mismatch" in error for error in errors)


def test_reference_encoding_threshold_and_metadata_are_fail_closed(
    tmp_path: Path,
) -> None:
    """Encoding policy includes the exact threshold and deterministic gzip."""

    source_root = (tmp_path / "source").absolute()
    destination = (tmp_path / "stored").absolute()
    source_root.mkdir()
    destination.mkdir()
    at_limit = source_root / "at_limit.csv"
    over_limit = source_root / "over_limit.csv"
    at_limit.write_bytes(
        b"value\n" + (b"a\n" * ((reference.RAW_TABLE_LIMIT - 6) // 2))
    )
    over_limit.write_bytes(at_limit.read_bytes() + b"b\n")

    raw = reference._store_artifact(
        at_limit,
        root=source_root,
        destination=destination,
        logical_role="table",
        logical_path="at_limit.csv",
    )
    compressed = reference._store_artifact(
        over_limit,
        root=source_root,
        destination=destination,
        logical_role="table",
        logical_path="over_limit.csv",
    )

    assert raw.encoding == "raw"
    assert raw.raw_size == reference.RAW_TABLE_LIMIT
    assert compressed.encoding == "gzip"
    with (destination / compressed.stored_path).open("rb") as handle:
        header = handle.read(10)
    assert int.from_bytes(header[4:8], "little") == 0
    assert header[3] & 0x08 == 0
    assert compressed.table_header == ("value",)
    assert compressed.row_count == 524_286
    assert compressed.column_types == {"value": "string"}
    assert reference._verify_artifact(raw) == []
    assert reference._verify_artifact(compressed) == []


def _v3_reference_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage: str = "lineage-a",
) -> tuple[Path, dict[str, str]]:
    source, attempts = _completed_lineage(
        tmp_path / "fixture",
        guarded=False,
        lineage=lineage,
    )
    provenance = {
        "commit": "a" * 40,
        "branch": "codex/test",
        "dirty": False,
    }
    runtime = {
        "schema_version": "pair-stability-v4/runtime-source/v1",
        "closure_sha256": "b" * 64,
        "n_files": 1,
        "git_commit": provenance["commit"],
        "git_branch": provenance["branch"],
        "dirty": False,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "uv_project_environment": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
    }
    monkeypatch.setattr(
        reference,
        "audit_completed_lineage",
        lambda _root, *, attempts: (),
    )
    monkeypatch.setattr(reference, "_source_provenance", lambda: provenance)
    monkeypatch.setattr(
        reference,
        "runtime_source_receipt",
        lambda _root: runtime,
    )
    return source, attempts


def _completed_lineage(
    tmp_path: Path,
    *,
    guarded: bool = True,
    lineage: str = "lineage-a",
) -> tuple[Path, dict[str, str]]:
    root = (tmp_path / "candidate").absolute()
    if guarded:
        root = roots.initialize_root(root, lineage_id=lineage)
    else:
        root.mkdir(parents=True)
    attempts = {
        "grid": lineage,
        "train": lineage,
        "validation": lineage,
        "collection": lineage,
        "selection": lineage,
        "final_grid": lineage,
        "final_train": lineage,
        "final_eval": lineage,
        "final_collect": lineage,
        "report": lineage,
    }
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(
                STUDY_DIR.parent
                / "pair_stability_v3"
                / "plan.py"
            ),
            "--grid",
            str(STUDY_DIR / "configs" / "smoke.yaml"),
            "--config",
            str(STUDY_DIR / "configs" / "pair_stability.yaml"),
            "--results-root",
            str(root),
            "--attempt-id",
            lineage,
            "--timezone",
            "America/New_York",
            "--blind",
            "--blind-seed",
            "811",
            "--python",
            "python",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    grid_dir = root / "00_grid" / lineage
    grid_manifest = json.loads((grid_dir / "manifest.json").read_text())
    grid_jobs = list(grid_manifest["jobs"])
    scan_ids = [str(job["run_id"]) for job in grid_jobs]
    validation_tasks = audit._configured_evaluation_tasks(
        grid_dir / "validation_config.yaml",
        suite="validation",
    )
    final_eval_tasks = audit._configured_evaluation_tasks(
        grid_dir / "validation_config.yaml",
        suite="final_eval",
    )
    validation_metric_columns = sorted(
        {
            "eval/perf/wall_time_sec",
            "eval/status/suite_success",
            "eval/status/suite_failed",
            *(
                f"eval/{task}/{audit.SCIENCE_METRIC_ANCHORS[task]}"
                for task in validation_tasks
            ),
            *(
                f"eval/{task}/status/task_success"
                for task in validation_tasks
            ),
            *(
                f"eval/{task}/status/task_failed"
                for task in validation_tasks
            ),
            *(
                f"eval/perf/{task}/generator_time_sec"
                for task in validation_tasks
            ),
        }
    )
    summary_rows = [
        {
            "run_id": str(job["run_id"]),
            "validation_attempt_id": lineage,
            "validation_attempt_dir": str(
                root / "02_validation" / str(job["run_id"]) / lineage
            ),
            "status": "completed",
            "major_id": str(job["major_id"]),
            "minor_id": str(job["minor_id"]),
            "config_id": str(job["config_id"]),
            "train_attempt_id": lineage,
            "checkpoint_path": str(
                root
                / "01_train"
                / str(job["run_id"])
                / lineage
                / "checkpoints"
            ),
            "n_diagnostics": len(validation_tasks),
            **{
                axis: job["choices"][axis]
                for axis in (
                    *grid_manifest["major_axes"],
                    *grid_manifest["minor_axes"],
                    grid_manifest["scan_seed_axis"],
                )
            },
            "eval/status/suite_success": True,
            "eval/status/suite_failed": False,
            "eval/perf/wall_time_sec": 0.1,
            **{
                (
                    f"eval/{task}/"
                    f"{audit.SCIENCE_METRIC_ANCHORS[task]}"
                ): 2.0
                for task in validation_tasks
            },
            **{
                f"eval/{task}/status/task_success": True
                for task in validation_tasks
            },
            **{
                f"eval/{task}/status/task_failed": False
                for task in validation_tasks
            },
            **{
                f"eval/perf/{task}/generator_time_sec": 0.01
                for task in validation_tasks
            },
        }
        for job in grid_jobs
    ]
    selection_replay = audit.replay_selection(
        grid=grid_manifest,
        summary_rows=[
            {
                key: audit._metric_csv_scalar(value)
                for key, value in row.items()
            }
            for row in summary_rows
        ],
    )
    champion_rows = selection_replay["champions"]
    final_ids = [
        f"{row['config_id']}_winner-{row['winner_kind']}_rep-0"
        for row in champion_rows
    ]

    selection = root / "04_select" / lineage
    selection.mkdir(parents=True)
    champion_fields = audit.selection_champion_columns(selection_replay)
    _write_csv(selection / "champions.csv", champion_rows, champion_fields)

    final_grid = root / "05_final_grid" / lineage
    final_grid.mkdir(parents=True)
    final_job_rows = [
        {
            "final_run_id": run_id,
            "source_selection_attempt_id": lineage,
            "source_champion_id": f"champion-{index:04d}",
            "source_champion_row_index": index,
            "source_scan_run_id": champion_rows[index]["config_id"],
            "source_scan_run_ids": champion_rows[index]["run_ids"],
            "source_scan_seeds": champion_rows[index]["seeds"],
            "replicate_index": 0,
            "winner_kind": champion_rows[index]["winner_kind"],
            "basis": champion_rows[index]["basis"],
            "update_normalization": champion_rows[index][
                "update_normalization"
            ],
            "feature_normalization": champion_rows[index][
                "feature_normalization"
            ],
            "metric": champion_rows[index]["metric"],
            "metric_value": 2.0,
            "final_train_sampler_seed": 1000,
            "final_train_model_seed": 100,
            "final_eval_sampler_seed": 10000,
            "source_champion": champion_rows[index],
        }
        for index, run_id in enumerate(final_ids)
    ]
    final_manifest = {
        "study": "pair_stability_v4",
        "stage": "05_final_grid",
        "attempt_id": lineage,
        "results_root": str(root),
        "source_selection_attempt_id": lineage,
        "source_selection_attempt_dir": str(selection),
        "train_config": str(grid_dir / "train_config.yaml"),
        "eval_config": str(grid_dir / "validation_config.yaml"),
        "config_snapshots": grid_manifest["config_snapshots"],
        "replicates": 1,
        "final_replicates": 1,
        "n_source_champions": len(champion_rows),
        "n_jobs": len(final_ids),
        "major_axes": grid_manifest["major_axes"],
        "minor_axes": grid_manifest["minor_axes"],
        "axis_id_labels": grid_manifest["axis_id_labels"],
        "axis_overrides": grid_manifest["axis_overrides"],
        "champion_kinds": sorted(
            {row["winner_kind"] for row in champion_rows}
        ),
        "seed_overrides": grid_manifest["seed_overrides"],
        "final_seed_sequences": grid_manifest["final_seed_sequences"],
        "static_overrides": grid_manifest["static_overrides"],
    }
    _write_json(
        final_grid / "manifest.json",
        final_manifest,
    )
    (final_grid / "manifest.yaml").write_text(
        yaml.safe_dump(final_manifest, sort_keys=False)
    )
    _write_csv(
        final_grid / "final_jobs.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key != "source_champion"
            }
            for row in final_job_rows
        ],
        tuple(key for key in final_job_rows[0] if key != "source_champion"),
    )
    for row in final_job_rows:
        _write_json(
            final_grid / "jobs" / f"{row['final_run_id']}.json",
            row,
        )
    _write_json(
        final_grid / "source_selection_attempt.json",
        {
            "selection_attempt_id": lineage,
            "selection_attempt_dir": str(selection),
            "champions_path": str(selection / "champions.csv"),
        },
    )
    (final_grid / "source_champions.csv").write_bytes(
        (selection / "champions.csv").read_bytes()
    )
    (final_grid / "task_lineage.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "row_id": run_id,
                    "task_ids": {
                        "train": [
                            (
                                f"01_train:"
                                f"{champion_rows[index]['run_ids']}:{lineage}"
                            )
                        ],
                        "validation": [
                            (
                                f"02_validation:"
                                f"{champion_rows[index]['run_ids']}:{lineage}"
                            )
                        ],
                    },
                }
            )
            + "\n"
            for index, run_id in enumerate(final_ids)
        )
    )

    _write_stage(root, "01_train", scan_ids, lineage)
    _write_stage(
        root,
        "02_validation",
        scan_ids,
        lineage,
        evaluation_tasks=validation_tasks,
    )
    _write_stage(root, "06_final_train", final_ids, lineage)
    _write_stage(
        root,
        "07_final_eval",
        final_ids,
        lineage,
        evaluation_tasks=final_eval_tasks,
    )

    collection = root / "03_collect" / lineage
    collection.mkdir(parents=True)
    _write_json(
        collection / "collection_report.json",
        {
            "study": "pair_stability_v4",
            "stage": "03_collect",
            "attempt_id": lineage,
            "grid_attempt_id": lineage,
            "major_axes": grid_manifest["major_axes"],
            "minor_axes": grid_manifest["minor_axes"],
            "scan_seed_axis": grid_manifest["scan_seed_axis"],
            "config_keys": [
                *grid_manifest["major_axes"],
                *grid_manifest["minor_axes"],
            ],
            "axis_id_labels": grid_manifest["axis_id_labels"],
            "n_collected": 64,
            "n_failures": 0,
            "metric_columns": validation_metric_columns,
            "required_train_metrics": [],
        },
    )
    _write_csv(
        collection / "summary.csv",
        summary_rows,
        (
            "run_id",
            "validation_attempt_id",
            "validation_attempt_dir",
            "status",
            "major_id",
            "minor_id",
            "config_id",
            "train_attempt_id",
            "checkpoint_path",
            "n_diagnostics",
            *grid_manifest["major_axes"],
            *grid_manifest["minor_axes"],
            grid_manifest["scan_seed_axis"],
            *validation_metric_columns,
        ),
    )
    _write_json(
        collection / "source_grid_attempt.json",
        {
            "grid_attempt_id": lineage,
            "grid_attempt_dir": str(grid_dir),
            "manifest_path": str(grid_dir / "manifest.json"),
        },
    )
    _write_json(
        collection / "source_validation_attempts.json",
        [
            {
                "run_id": run_id,
                "validation_attempt_id": lineage,
                "validation_attempt_dir": str(
                    root / "02_validation" / run_id / lineage
                ),
            }
            for run_id in scan_ids
        ],
    )
    (collection / "task_lineage.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "row_id": run_id,
                    "task_ids": {
                        "validation": f"02_validation:{run_id}:{lineage}",
                        "train": f"01_train:{run_id}:{lineage}",
                    },
                }
            )
            + "\n"
            for run_id in scan_ids
        )
    )
    summary_header = tuple(
        next(csv.reader((collection / "summary.csv").open(newline="")))
    )
    _write_csv(collection / "failures.csv", [], summary_header)
    for name in ("cost_by_run.csv", "cost_by_axis.csv", "cost_by_task.csv"):
        _write_csv(collection / name, [], ("metric", "value"))

    _write_json(
        selection / "source_collection_attempt.json",
        {
            "collection_attempt_id": lineage,
            "collection_attempt_dir": str(collection),
        },
    )
    _write_json(
        selection / "selection_report.json",
        {
            "study": "pair_stability_v4",
            "stage": "04_select",
            "attempt_id": lineage,
            "collection_attempt_id": lineage,
            "metric": None,
            "mode": "min",
            "group_by": selection_replay["group_by"],
            "major_axes": grid_manifest["major_axes"],
            "minor_axes": grid_manifest["minor_axes"],
            "scan_seed_axis": grid_manifest["scan_seed_axis"],
            "axis_id_labels": grid_manifest["axis_id_labels"],
            "champion_kinds": selection_replay["champion_kinds"],
            "champion_specs": selection_replay["champion_specs"],
            "config_keys": selection_replay["config_keys"],
            "seed_aggregation": {
                "value": "median of successful seed rows",
                "error_bar": (
                    "sample standard error across successful seed rows"
                ),
                "mean": "arithmetic mean across successful seed rows",
            },
            "reference_metrics": selection_replay["reference_metrics"],
            "reference_statistics": list(
                audit.SELECTION_REFERENCE_STATISTICS
            ),
            "wall_time_metrics": [audit.TRAIN_WALL_TIME_METRIC],
            "n_candidates": selection_replay["n_candidates"],
            "n_configs": selection_replay["n_candidates"],
            "n_champions": len(champion_rows),
            "overall_champion": selection_replay["overall_champion"],
            "overall_metric": selection_replay["overall_metric"],
            "overall_metric_value": selection_replay[
                "overall_metric_value"
            ],
            "overall_decisions": selection_replay["overall_decisions"],
            "secondary_champion_kind": selection_replay[
                "secondary_champion_kind"
            ],
            "secondary_metric": selection_replay["secondary_metric"],
            "secondary_champion": selection_replay[
                "secondary_champion"
            ],
            "secondary_metric_value": selection_replay[
                "secondary_metric_value"
            ],
            "secondary_decisions": selection_replay[
                "secondary_decisions"
            ],
            "decisions_by_group": selection_replay[
                "decisions_by_group"
            ],
            "used_status_fallback": selection_replay[
                "used_status_fallback"
            ],
            "champions": selection_replay["champions"],
            "configs": selection_replay["configs"],
        },
    )
    (selection / "task_lineage.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "row_id": (
                        f"energy:basis={row['basis']}|"
                        f"update_normalization={row['update_normalization']}|"
                        f"feature_normalization={row['feature_normalization']}"
                    ),
                    "task_ids": {
                        "validation": [
                            f"02_validation:{row['run_ids']}:{lineage}"
                        ],
                        "train": [
                            f"01_train:{row['run_ids']}:{lineage}"
                        ],
                    },
                }
            )
            + "\n"
            for row in champion_rows
        )
    )

    final_collect = root / "08_final_collect" / lineage
    final_collect.mkdir(parents=True)
    _write_csv(
        final_collect / "run_index.csv",
        [
            {
                "final_run_id": run_id,
                "final_eval_attempt_id": lineage,
                "train_status": "checkpoint_selected",
                "eval_status": "completed",
                "n_eval_tasks_success": len(final_eval_tasks),
                "n_eval_tasks_failed": 0,
            }
            for run_id in final_ids
        ],
        (
            "final_run_id",
            "final_eval_attempt_id",
            "train_status",
            "eval_status",
            "n_eval_tasks_success",
            "n_eval_tasks_failed",
        ),
    )
    _write_csv(
        final_collect / "failure_modes.csv",
        [],
        ("final_run_id", "severity", "failure_mode"),
    )
    final_tables = {
        name: (8 if name == "run_index.csv" else 0)
        for name in sorted(audit.EXPECTED_FINAL_COLLECT_TABLES)
    }
    for name in final_tables:
        path = final_collect / name
        if path.exists():
            continue
        _write_csv(path, [], ("final_run_id", "value"))
    (final_collect / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "study": "pair_stability_v4",
                "stage": "08_final_collect",
                "attempt_id": lineage,
                "final_grid_attempt_id": lineage,
                "final_eval_attempt_id": lineage,
                "final_eval_attempt_ids": [lineage],
                "final_eval_attempts": {
                    run_id: lineage for run_id in final_ids
                },
                "n_final_eval_attempts": 8,
                "major_axes": grid_manifest["major_axes"],
                "minor_axes": grid_manifest["minor_axes"],
                "axis_columns": [
                    *grid_manifest["major_axes"],
                    *grid_manifest["minor_axes"],
                ],
                "report_row_key": "basis_update",
                "report_col_key": "feature_normalization",
                "expected_final_replicates": 1,
                "tables": final_tables,
                "source_stages": {
                    "final_grid": "05_final_grid",
                    "final_train": "06_final_train",
                    "final_eval": "07_final_eval",
                },
            },
            sort_keys=False,
        )
    )
    report = root / "09_final_report" / lineage
    report.mkdir(parents=True)
    for name in final_tables:
        target = report / "tables" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((final_collect / name).read_bytes())
    figure_name = "summary.png"
    figure = report / "figures" / figure_name
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"not-a-real-png-but-nonempty")
    _write_json(
        report / "final_report.json",
        {
            "study": "pair_stability_v4",
            "stage": "09_final_report",
            "attempt_id": lineage,
            "final_collect_attempt_id": lineage,
            "final_collect_dir": str(final_collect),
            "report_axes": {
                "row": "basis_update",
                "column": "feature_normalization",
            },
            "tables": final_tables,
            "figures": [figure_name],
            "caveats": [],
        },
    )
    (report / "report.md").write_text("# report\n")
    return root, attempts


def _write_stage(
    root: Path,
    stage: str,
    run_ids: list[str],
    lineage: str,
    evaluation_tasks: tuple[str, ...] = (),
) -> None:
    expectation = audit.STAGE_EXPECTATIONS[stage]
    resource = ResourceSpec(
        profile="cuda",
        device="cuda",
        partition="gpu_test",
        threads=4,
        mem_gb=32,
        gpus=1,
        timeout_min=int(expectation["timeout_min"]),
        uv_environment=".venv-gpu",
        uv_extras=("cu126",),
    )
    tasks: list[TaskSpec] = []
    records: list[ExecutionRecord] = []
    for index, run_id in enumerate(run_ids):
        result_dir = root / stage / run_id / lineage
        result_dir.mkdir(parents=True)
        command_parts = [
            "python",
            "-u",
            "run.py",
            "study.name=pair_stability_v4",
            f"run.run_id={run_id}/{lineage}",
        ]
        if stage in {"06_final_train", "07_final_eval"}:
            command_parts.append(f"study.config_id=champion-{index:04d}")
        command = tuple(command_parts)
        launcher_status = result_dir / "launcher_status.json"
        status = result_dir / "status.json"
        checkpoint = (
            result_dir / "checkpoints" / "latest.json"
            if stage in {"01_train", "06_final_train"}
            else None
        )
        completion = CompletionSpec(
            policy=str(expectation["completion_policy"]),
            status_path=str(status),
            checkpoint_path=None if checkpoint is None else str(checkpoint),
        )
        source_train: dict[str, object] | None = None
        if stage == "02_validation":
            train_dir = root / "01_train" / run_id / lineage
            source_train = {
                "run_id": run_id,
                "grid_attempt_id": lineage,
                "train_attempt_id": lineage,
                "train_dir": str(root / "01_train" / run_id),
                "train_attempt_dir": str(train_dir),
                "checkpoint_path": str(train_dir / "checkpoints"),
            }
        if stage == "06_final_train":
            job_metadata = json.loads(
                (
                    root
                    / "05_final_grid"
                    / lineage
                    / "jobs"
                    / f"{run_id}.json"
                ).read_text()
            )
        elif stage == "07_final_eval":
            train_dir = root / "06_final_train" / run_id / lineage
            pointer_path = train_dir / "checkpoints" / "latest.json"
            pointer_data = json.loads(pointer_path.read_text())
            checkpoint_record = {
                "selection_path": str(
                    train_dir / "selected_checkpoint.json"
                ),
                "selection_policy": "latest_checkpoint_pointer",
                "checkpoint_pointer": str(pointer_path),
                "checkpoint_pointer_data": pointer_data,
                "resolved_checkpoint_dir": str(
                    train_dir
                    / "checkpoints"
                    / pointer_data["checkpoint_dir"]
                ),
            }
            # Match final_eval.py's deliberately reduced planned-job shape:
            # source champion identity remains in source_final_job.json.
            job_metadata = {
                "final_run_id": run_id,
                "final_grid_attempt_id": lineage,
                "final_train_attempt_id": lineage,
                "final_eval_attempt_id": lineage,
                "final_eval_attempt_dir": str(result_dir),
                "checkpoint": checkpoint_record,
                "command": shlex.join(command),
                "command_parts": list(command),
            }
        else:
            job_metadata = {
                "run_id": run_id,
                **(
                    {"source_train_attempt": source_train}
                    if source_train is not None
                    else {}
                ),
            }
        task = TaskSpec(
            task_id=f"{stage}:{run_id}:{lineage}",
            stage=stage,
            attempt_id=lineage,
            run_id=run_id,
            command=command,
            result_dir=str(result_dir),
            outputs=(() if checkpoint is None else (str(checkpoint),)),
            logs=(str(launcher_status),),
            resources=resource,
            completion=completion,
            metadata={"job": job_metadata},
        )
        submitted = tuple(audit._expected_submitted_command(list(command)))
        chunk_index = index // int(expectation["chunk_size"])
        launcher_id = (
            "12345"
            if int(expectation["count"]) == int(expectation["chunk_size"])
            else f"12345_{chunk_index}"
        )
        record = ExecutionRecord(
            task_id=task.task_id,
            run_id=run_id,
            stage=stage,
            attempt_id=lineage,
            backend="submitit",
            launcher_job_id=launcher_id,
            submitted_command=submitted,
            status_path=str(launcher_status),
        )
        _write_json(
            status,
            {
                "status": "completed",
                "current_event": "run_end",
                "exception_type": None,
                "exception_message": None,
            },
        )
        _write_json(
            launcher_status,
            {
                "status": "success",
                "returncode": 0,
                "command": shlex.join(submitted),
            },
        )
        _write_json(
            result_dir / "run_start.json",
            {
                "run_id": f"{run_id}/{lineage}",
                "run_dir": str(result_dir),
                "study": {
                    "name": "pair_stability_v4",
                    "config_id": (
                        f"champion-{index:04d}"
                        if stage in {"06_final_train", "07_final_eval"}
                        else None
                    ),
                },
                "command": shlex.join(command),
                "git": {
                    "sha": "a" * 40,
                    "branch": "codex/test",
                    "dirty": False,
                },
                "slurm": {
                    "job_partition": "gpu_test",
                    "cpus_per_task": "4",
                    "job_id": "22345",
                    **(
                        {"array_task_id": str(chunk_index)}
                        if "_" in launcher_id
                        else {}
                    ),
                },
                "environment": {
                    "SLURM_JOB_PARTITION": "gpu_test",
                    "SLURM_CPUS_PER_TASK": "4",
                    "SLURM_JOB_ID": "22345",
                    "CUDA_VISIBLE_DEVICES": f"GPU-{chunk_index}",
                    **(
                        {"SLURM_ARRAY_TASK_ID": str(chunk_index)}
                        if "_" in launcher_id
                        else {}
                    ),
                },
            },
        )
        _write_json(
            result_dir / "metadata.json",
            {
                "status": "completed",
                "dirty_worktree": False,
                "git_commit": "a" * 40,
                "run_id": f"{run_id}/{lineage}",
                "run_dir": str(result_dir),
                "command": shlex.join(command),
                "device": "cuda",
                "dtype": "float64",
                "runtime": {
                    "device": "cuda",
                    "dtype": "float64",
                    "python_version": "3.14.test",
                    "python_executable": str(
                        REPO_ROOT / ".venv-gpu" / "bin" / "python"
                    ),
                    "torch_version": "2.test+cu126",
                    "torch_cuda_version": "12.6",
                    "cuda_visible_devices": f"GPU-{chunk_index}",
                },
                "hardware": {
                    "cuda_available": True,
                    "cuda_device_count": 1,
                    "cuda_devices": [
                        {
                            "index": 0,
                            "name": "Test GPU",
                            "total_memory_bytes": 1_000_000,
                            "capability": "8.0",
                        }
                    ],
                },
                "slurm": {
                    "job_partition": "gpu_test",
                    "cpus_per_task": "4",
                    "job_id": "22345",
                    **(
                        {"array_task_id": str(chunk_index)}
                        if "_" in launcher_id
                        else {}
                    ),
                },
            },
        )
        submission = {
            (
                "final_run_id"
                if stage in {"06_final_train", "07_final_eval"}
                else "run_id"
            ): run_id,
            "launcher": "submitit",
            "launcher_job_id": launcher_id,
            "command": shlex.join(command),
            "submitted_command": shlex.join(submitted),
        }
        submission.update(
            {
                "01_train": {"grid_attempt_id": lineage},
                "02_validation": {
                    "grid_attempt_id": lineage,
                    "train_attempt_id": lineage,
                    "validation_attempt_id": lineage,
                },
                "06_final_train": {
                    "final_grid_attempt_id": lineage,
                    "final_train_attempt_id": lineage,
                },
                "07_final_eval": {
                    "final_grid_attempt_id": lineage,
                    "final_train_attempt_id": lineage,
                    "final_eval_attempt_id": lineage,
                },
            }[stage]
        )
        _write_json(result_dir / "submission.json", submission)
        metrics = (
            []
            if evaluation_tasks
            else [
                {
                    "namespace": "train",
                    "step": 0,
                    "metrics": {"energy": 2.0},
                }
            ]
        )
        if evaluation_tasks:
            metrics.append(
                {
                    "namespace": "eval/perf",
                    "step": 0,
                    "metrics": {"wall_time_sec": 0.1},
                }
            )
            metrics.append(
                {
                    "namespace": "eval/status",
                    "step": 0,
                    "metrics": {
                        "suite_success": True,
                        "suite_failed": False,
                    },
                }
            )
            metrics.extend(
                row
                for task in evaluation_tasks
                for row in (
                    {
                        "namespace": f"eval/{task}",
                        "step": 0,
                        "metrics": {
                            audit.SCIENCE_METRIC_ANCHORS[task]: 2.0
                        },
                    },
                    {
                        "namespace": f"eval/perf/{task}",
                        "step": 0,
                        "metrics": {"generator_time_sec": 0.01},
                    },
                    {
                        "namespace": f"eval/{task}/status",
                        "step": 0,
                        "metrics": {
                            "task_success": True,
                            "task_failed": False,
                        },
                    },
                )
            )
        (result_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metrics)
        )
        if evaluation_tasks:
            diagnostic_rows = []
            record_filenames = {
                "cusp": "cusp_profiles.csv",
                "tail": "tail_profiles.csv",
                "stratified_geometry": "stratified_metrics.csv",
                "hooke_orbital": "hooke_orbital_metrics.csv",
                "energy": "mcmc_energy_samples.csv",
                "full_model_antisymmetry": "transform_records.csv",
                "spatial_exchange_symmetry": "transform_records.csv",
                "rotation_consistency": "transform_records.csv",
                "trace_equivariance": "trace_records.csv",
                "feature_trace_stability": "trace_records.csv",
                "readout_trace_stability": "trace_records.csv",
            }
            for task_name in evaluation_tasks:
                output_dir = result_dir / task_name
                output_dir.mkdir(parents=True)
                artifacts = []
                if stage == "07_final_eval":
                    filename = record_filenames[task_name]
                    artifact_path = output_dir / filename
                    artifact_path.write_text("value\n1\n")
                    artifacts.append(
                        {
                            "kind": "csv",
                            "metadata": {"rows": 1},
                            "name": "records",
                            "path": str(artifact_path),
                        }
                    )
                diagnostic_rows.append(
                    {
                        "name": task_name,
                        "namespace": f"eval/{task_name}",
                        "output_dir": str(output_dir),
                        "status": "success",
                        "artifacts": artifacts,
                    }
                )
            _write_json(
                result_dir / "diagnostics" / "index.json",
                {"tasks": diagnostic_rows},
            )
        if checkpoint is not None:
            concrete = result_dir / "checkpoints" / "step_000000"
            concrete.mkdir(parents=True)
            (concrete / "model.pt").write_bytes(b"model")
            _write_json(
                concrete / "manifest.json",
                {
                    "schema_version": 1,
                    "kind": "spenn.checkpoint",
                    "step": 0,
                    "created_at_unix": 0.0,
                    "files": {"model": "model.pt"},
                    "hashes": {},
                    "runtime": {},
                    "provenance": {},
                },
            )
            (concrete / "COMPLETE").write_text("complete\n")
            _write_json(
                checkpoint,
                {
                    "checkpoint_dir": concrete.name,
                    "step": 0,
                    "created_at_unix": 0.0,
                },
            )
            if stage == "06_final_train":
                _write_json(
                    result_dir / "selected_checkpoint.json",
                    {
                        "selection_policy": "latest_checkpoint_pointer",
                        "checkpoint_dir": str(checkpoint.parent),
                        "checkpoint_pointer": str(checkpoint),
                        "resolved_checkpoint_dir": None,
                    },
                )
        if stage in {"01_train", "02_validation"}:
            source_grid = {
                "run_id": run_id,
                "grid_attempt_id": lineage,
                "grid_attempt_dir": str(root / "00_grid" / lineage),
            }
            if stage == "01_train":
                source_grid["manifest_path"] = str(
                    root / "00_grid" / lineage / "manifest.json"
                )
            _write_json(
                result_dir / "source_grid_attempt.json",
                source_grid,
            )
        if stage == "02_validation":
            assert source_train is not None
            _write_json(
                result_dir / "source_train_attempt.json",
                source_train,
            )
        if stage in {"06_final_train", "07_final_eval"}:
            final_grid_dir = root / "05_final_grid" / lineage
            final_job = json.loads(
                (final_grid_dir / "jobs" / f"{run_id}.json").read_text()
            )
            _write_json(
                result_dir / "source_final_grid_attempt.json",
                {
                    "final_grid_attempt_id": lineage,
                    "final_grid_attempt_dir": str(final_grid_dir),
                    "final_jobs_path": str(
                        final_grid_dir / "final_jobs.csv"
                    ),
                },
            )
            _write_json(result_dir / "source_final_job.json", final_job)
            _write_json(
                result_dir / "source_champion.json",
                final_job["source_champion"],
            )
        if stage == "07_final_eval":
            train_dir = root / "06_final_train" / run_id / lineage
            pointer_path = train_dir / "checkpoints" / "latest.json"
            pointer_data = json.loads(pointer_path.read_text())
            checkpoint_record = {
                "selection_path": str(train_dir / "selected_checkpoint.json"),
                "selection_policy": "latest_checkpoint_pointer",
                "checkpoint_pointer": str(pointer_path),
                "checkpoint_pointer_data": pointer_data,
                "resolved_checkpoint_dir": str(
                    train_dir / "checkpoints" / pointer_data["checkpoint_dir"]
                ),
            }
            _write_json(
                result_dir / "source_final_train_attempt.json",
                {
                    "final_run_id": run_id,
                    "final_train_attempt_id": lineage,
                    "final_train_attempt_dir": str(train_dir),
                    "checkpoint": checkpoint_record,
                },
            )
            _write_json(
                result_dir / "evaluated_checkpoint.json",
                checkpoint_record,
            )
        tasks.append(task)
        records.append(record)

    plan = StagePlan(
        study="pair_stability_v4",
        stage=stage,
        attempt_id=lineage,
        results_root=str(root),
        source_attempts={str(expectation["source_attempt"]): lineage},
        smoke=False,
        metadata={
            "backend": "submitit",
            "device": "cuda",
            "chunk_size": int(expectation["chunk_size"]),
        },
        tasks=tuple(tasks),
    )
    plan_dir = root / stage / "stage_plans" / lineage
    plan.write(plan_dir)
    write_execution_records(plan_dir, records)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
