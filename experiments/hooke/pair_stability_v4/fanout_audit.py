"""Exact task, submission, worker, and terminal audits for V4-0 fan-out."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.toolkit import ExecutionRecord, StagePlan

from routes import REPO_ROOT, load_routes

EXPECTED_SCAN_COUNT = 64
EXPECTED_FINAL_COUNT = 8
GPU_TEST_ARRAY_TASK_CAP = 2
WORKER_RUNTIME_SCHEMA_VERSION = "pair-stability-v4/worker-runtime/v1"
WORKER_RUNTIME_VOLATILE_POINTERS = ("/runtime/cuda_visible_devices",)
SCIENCE_METRIC_ANCHORS = {
    "cusp": "local_energy_mean",
    "tail": "local_energy_mean",
    "stratified_geometry": "local_energy_mean",
    "hooke_orbital": "local_energy_mean",
    "full_model_antisymmetry": "logabs_max_abs_error",
    "trace_equivariance": "max_abs_error",
    "feature_trace_stability": "feature_rms_q95",
    "readout_trace_stability": "condition_number_q95",
    "energy": "local_energy_mean",
    "spatial_exchange_symmetry": "logabs_max_abs_error",
    "rotation_consistency": "local_energy_max_abs_error",
}
STAGE_EXPECTATIONS = {
    "01_train": {
        "count": EXPECTED_SCAN_COUNT,
        "chunk_size": 32,
        "timeout_min": 60,
        "completion_policy": "status_completed_with_checkpoint",
        "source_attempt": "grid",
    },
    "02_validation": {
        "count": EXPECTED_SCAN_COUNT,
        "chunk_size": 32,
        "timeout_min": 120,
        "completion_policy": "status_completed",
        "source_attempt": "grid",
    },
    "06_final_train": {
        "count": EXPECTED_FINAL_COUNT,
        "chunk_size": 8,
        "timeout_min": 60,
        "completion_policy": "status_completed_with_checkpoint",
        "source_attempt": "final_grid",
    },
    "07_final_eval": {
        "count": EXPECTED_FINAL_COUNT,
        "chunk_size": 8,
        "timeout_min": 120,
        "completion_policy": "status_completed",
        "source_attempt": "final_grid",
    },
}

_FANOUT_STAGE_ROLES = {
    "01_train": "screen_train",
    "02_validation": "screen_eval",
    "06_final_train": "confirm_train",
    "07_final_eval": "confirm_eval",
}


def gpu_test_array_task_count(*, population: int, chunk_size: int) -> int:
    """Return Submitit ``map_array`` element count for one fan-out route."""

    if population <= 0 or chunk_size <= 0:
        raise ValueError("fan-out population and chunk_size must be positive")
    return (population + chunk_size - 1) // chunk_size


def audit_gpu_test_fanout_profile() -> tuple[str, ...]:
    """Prove every approved V4-0 fan-out submission respects gpu_test cap."""

    errors: list[str] = []
    routes = load_routes()
    for stage, role in _FANOUT_STAGE_ROLES.items():
        expectation = STAGE_EXPECTATIONS[stage]
        route = routes[role]
        arguments = route.arguments
        try:
            partition_flag = (
                "--slurm-cuda-partition"
                if "--slurm-cuda-partition" in arguments
                else "--slurm-partition"
            )
            partition = _route_argument(arguments, partition_flag)
            chunk_size = int(_route_argument(arguments, "--chunk-size"))
        except ValueError as exc:
            errors.append(f"{role} route profile is invalid: {exc}")
            continue
        if route.kind != "fanout" or partition != "gpu_test":
            errors.append(f"{role} is not one gpu_test fan-out submission")
        if chunk_size != int(expectation["chunk_size"]):
            errors.append(f"{role} chunk size differs from approved profile")
        tasks = gpu_test_array_task_count(
            population=int(expectation["count"]),
            chunk_size=chunk_size,
        )
        if tasks > GPU_TEST_ARRAY_TASK_CAP:
            errors.append(
                f"{role} submits {tasks} gpu_test array tasks, cap is "
                f"{GPU_TEST_ARRAY_TASK_CAP}"
            )
    return tuple(errors)


def _route_argument(arguments: Sequence[str], flag: str) -> str:
    try:
        index = list(arguments).index(flag)
    except ValueError as exc:
        raise ValueError(f"missing {flag}") from exc
    try:
        value = arguments[index + 1]
    except IndexError as exc:
        raise ValueError(f"missing value for {flag}") from exc
    if value.startswith("--"):
        raise ValueError(f"missing value for {flag}")
    return value


def audit_fanout_stages(
    root: Path,
    *,
    attempt: str,
    expected_study: str,
    evaluation_tasks: Mapping[str, Sequence[str]] | None = None,
) -> tuple[
    dict[str, frozenset[str]],
    frozenset[str],
    tuple[str, ...],
    dict[str, Any],
]:
    """Audit all four canonical fan-out stages as one immutable result."""

    errors: list[str] = []
    worker_commits: set[str] = set()
    stage_run_ids: dict[str, frozenset[str]] = {}
    science_observations: dict[str, dict[str, list[tuple[str, ...]]]] = {}
    runtime_rows: list[dict[str, Any]] = []
    task_contracts = evaluation_tasks or {}
    for stage, expectation in STAGE_EXPECTATIONS.items():
        run_ids = _audit_fanout_stage(
            root,
            stage=stage,
            attempt=attempt,
            expected_study=expected_study,
            expectation=expectation,
            required_eval_tasks=tuple(task_contracts.get(stage, ())),
            worker_commits=worker_commits,
            science_observations=science_observations,
            runtime_rows=runtime_rows,
            errors=errors,
        )
        stage_run_ids[stage] = frozenset(run_ids)
    _audit_science_schema_consistency(science_observations, errors)
    _audit_worker_runtime_uniformity(runtime_rows, errors)
    evidence = {
        "science_metrics": _science_schema_summary(science_observations),
        "worker_runtime": _worker_runtime_summary(runtime_rows),
    }
    return (
        stage_run_ids,
        frozenset(worker_commits),
        tuple(dict.fromkeys(errors)),
        evidence,
    )


def _audit_fanout_stage(
    root: Path,
    *,
    stage: str,
    attempt: str,
    expected_study: str,
    expectation: Mapping[str, Any],
    required_eval_tasks: Sequence[str],
    worker_commits: set[str],
    science_observations: dict[str, dict[str, list[tuple[str, ...]]]],
    runtime_rows: list[dict[str, Any]],
    errors: list[str],
) -> set[str]:
    plan_dir = root / stage / "stage_plans" / attempt
    manifest = _read_json_for_audit(plan_dir / "stage_manifest.json", errors)
    try:
        plan = StagePlan.read(plan_dir)
    except (OSError, ValueError) as exc:
        errors.append(f"{stage} invalid StagePlan: {exc}")
        tasks: list[dict[str, Any]] = []
    else:
        tasks = [task.to_dict() for task in plan.tasks]
    raw_records = _read_jsonl(plan_dir / "execution_records.jsonl", errors)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(raw_records):
        try:
            records.append(ExecutionRecord.from_dict(row).to_dict())
        except ValueError as exc:
            errors.append(f"{stage} invalid ExecutionRecord {index}: {exc}")
    expected_count = int(expectation["count"])

    expected_manifest = {
        "schema_version": "experiment-toolkit/v1",
        "study": expected_study,
        "stage": stage,
        "attempt_id": attempt,
        "results_root": str(root),
        "smoke": False,
        "tasks_path": "tasks.jsonl",
        "n_tasks": expected_count,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            errors.append(
                f"{stage} stage manifest {key}={manifest.get(key)!r}, "
                f"expected {expected!r}"
            )
    source_key = str(expectation["source_attempt"])
    if manifest.get("source_attempts") != {source_key: attempt}:
        errors.append(f"{stage} source_attempts do not name {source_key}={attempt}")
    metadata = manifest.get("metadata")
    expected_metadata = {
        "backend": "submitit",
        "device": "cuda",
        "chunk_size": int(expectation["chunk_size"]),
    }
    if not isinstance(metadata, dict):
        errors.append(f"{stage} stage metadata is not an object")
    else:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                errors.append(
                    f"{stage} stage metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )

    if len(tasks) != expected_count:
        errors.append(
            f"{stage} task population is {len(tasks)}, expected {expected_count}"
        )
    if len(records) != expected_count:
        errors.append(
            f"{stage} execution-record population is {len(records)}, "
            f"expected {expected_count}"
        )
    _audit_launcher_groups(
        records,
        stage=stage,
        expected_count=expected_count,
        chunk_size=int(expectation["chunk_size"]),
        errors=errors,
    )
    task_by_identity = _rows_by_identity(tasks, stage, "task", errors)
    record_by_identity = _rows_by_identity(
        records,
        stage,
        "execution record",
        errors,
    )
    if set(task_by_identity) != set(record_by_identity):
        errors.append(f"{stage} task/submission identity sets differ")

    for index, identity in enumerate(sorted(set(task_by_identity) & set(record_by_identity))):
        _audit_task_and_submission(
            root,
            stage=stage,
            attempt=attempt,
            index=index,
            task=task_by_identity[identity],
            record=record_by_identity[identity],
            expected_study=expected_study,
            expectation=expectation,
            required_eval_tasks=required_eval_tasks,
            worker_commits=worker_commits,
            science_observations=science_observations,
            runtime_rows=runtime_rows,
            errors=errors,
        )
    return {identity[1] for identity in task_by_identity}


def _audit_task_and_submission(
    root: Path,
    *,
    stage: str,
    attempt: str,
    index: int,
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_study: str,
    expectation: Mapping[str, Any],
    required_eval_tasks: Sequence[str],
    worker_commits: set[str],
    science_observations: dict[str, dict[str, list[tuple[str, ...]]]],
    runtime_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    context = f"{stage} task {index}"
    run_id = str(task.get("run_id") or "")
    task_id = f"{stage}:{run_id}:{attempt}"
    expected_task = {
        "task_id": task_id,
        "stage": stage,
        "attempt_id": attempt,
    }
    for key, expected in expected_task.items():
        if task.get(key) != expected:
            errors.append(
                f"{context} {key}={task.get(key)!r}, expected {expected!r}"
            )
    expected_record = {
        "task_id": task_id,
        "run_id": run_id,
        "stage": stage,
        "attempt_id": attempt,
        "backend": "submitit",
    }
    for key, expected in expected_record.items():
        if record.get(key) != expected:
            errors.append(
                f"{context} execution {key}={record.get(key)!r}, "
                f"expected {expected!r}"
            )
    launcher_job_id = record.get("launcher_job_id")
    if not isinstance(launcher_job_id, str) or not re.fullmatch(
        r"[0-9]+(?:_[0-9]+)?",
        launcher_job_id,
    ):
        errors.append(f"{context} has no launcher_job_id")
    submitted = record.get("submitted_command")
    if not isinstance(submitted, list) or not submitted or not all(
        isinstance(part, str) and part for part in submitted
    ):
        errors.append(f"{context} submitted_command is invalid")
    elif submitted != _expected_submitted_command(task.get("command")):
        errors.append(f"{context} submitted command differs from planned CUDA wrapper")

    resources = task.get("resources")
    expected_resources = {
        "profile": "cuda",
        "device": "cuda",
        "partition": "gpu_test",
        "threads": 4,
        "mem_gb": 32,
        "gpus": 1,
        "timeout_min": int(expectation["timeout_min"]),
        "uv_environment": ".venv-gpu",
        "uv_extras": ["cu126"],
        "metadata": {},
    }
    if resources != expected_resources:
        errors.append(
            f"{context} resources differ: {resources!r} != "
            f"{expected_resources!r}"
        )

    result_dir = _task_path(root, task.get("result_dir"), context, errors)
    expected_result_dir = root / stage / run_id / attempt
    if result_dir != expected_result_dir:
        errors.append(
            f"{context} result_dir={result_dir}, "
            f"expected {expected_result_dir}"
        )
    logs = task.get("logs")
    expected_launcher_status = (
        result_dir / "launcher_status.json" if result_dir is not None else None
    )
    if not isinstance(logs, list) or logs != [str(expected_launcher_status)]:
        errors.append(f"{context} launcher log path is not exact")
    if record.get("status_path") != (
        str(expected_launcher_status) if expected_launcher_status else None
    ):
        errors.append(f"{context} execution status_path is not task launcher log")

    completion = task.get("completion")
    expected_policy = str(expectation["completion_policy"])
    if not isinstance(completion, dict) or completion.get("policy") != expected_policy:
        errors.append(f"{context} completion policy is not {expected_policy}")
        return
    if result_dir is None:
        return
    status_path = _task_path(root, completion.get("status_path"), context, errors)
    if status_path != result_dir / "status.json":
        errors.append(f"{context} completion status path is not result/status.json")
    status = _read_json_for_audit(status_path, errors) if status_path else {}
    if status.get("status") != "completed":
        errors.append(f"{context} run status is not completed")
    if status.get("current_event") != "run_end":
        errors.append(f"{context} run status current_event is not run_end")
    if status.get("exception_type") is not None:
        errors.append(f"{context} run status exception_type is not null")
    if status.get("exception_message") is not None:
        errors.append(f"{context} run status exception_message is not null")
    launcher_status = (
        _read_json_for_audit(expected_launcher_status, errors)
        if expected_launcher_status
        else {}
    )
    if (
        launcher_status.get("status") != "success"
        or launcher_status.get("returncode") != 0
    ):
        errors.append(f"{context} launcher status is not success")
    if isinstance(submitted, list) and launcher_status.get("command") != shlex.join(
        submitted
    ):
        errors.append(f"{context} launcher command differs from execution record")

    checkpoint_value = completion.get("checkpoint_path")
    if expected_policy == "status_completed_with_checkpoint":
        concrete_checkpoint: Path | None = None
        checkpoint_path = _task_path(root, checkpoint_value, context, errors)
        expected_checkpoint = result_dir / "checkpoints" / "latest.json"
        if checkpoint_path != expected_checkpoint:
            errors.append(f"{context} checkpoint path is not checkpoints/latest.json")
        else:
            concrete_checkpoint = _complete_checkpoint_pointer(checkpoint_path)
        if checkpoint_path == expected_checkpoint and concrete_checkpoint is None:
            errors.append(f"{context} checkpoint pointer is not concretely complete")
        if (
            stage == "06_final_train"
            and concrete_checkpoint is not None
            and not _final_selection_matches(result_dir, concrete_checkpoint)
        ):
            errors.append(
                f"{context} selected_checkpoint does not resolve to latest complete checkpoint"
            )
    elif checkpoint_value not in {None, ""}:
        errors.append(f"{context} status-only completion declares a checkpoint")

    metrics_path = result_dir / "metrics.jsonl"
    if not _valid_metrics(metrics_path, require_eval=stage in {"02_validation", "07_final_eval"}):
        errors.append(f"{context} has no valid nonempty metrics.jsonl evidence")
    if stage in {"02_validation", "07_final_eval"}:
        schemas = _audit_evaluation_status_metrics(
            metrics_path,
            required_tasks=required_eval_tasks,
            context=context,
            errors=errors,
        )
        for task_name, keys in schemas.items():
            science_observations.setdefault(stage, {}).setdefault(
                task_name,
                [],
            ).append(keys)
        _audit_diagnostics_index(
            result_dir,
            required_tasks=required_eval_tasks,
            context=context,
            errors=errors,
        )
    if stage == "02_validation":
        _audit_scan_validation_source(
            result_dir,
            task=task,
            run_id=run_id,
            attempt=attempt,
            root=root,
            context=context,
            errors=errors,
        )
    elif stage == "07_final_eval":
        _audit_final_evaluation_source(
            result_dir,
            run_id=run_id,
            attempt=attempt,
            root=root,
            context=context,
            errors=errors,
        )
    _audit_stage_source_edges(
        root,
        result_dir=result_dir,
        stage=stage,
        run_id=run_id,
        attempt=attempt,
        context=context,
        errors=errors,
    )
    _audit_submission_record(
        result_dir,
        stage=stage,
        attempt=attempt,
        run_id=run_id,
        task_command=task.get("command"),
        launcher_job_id=str(launcher_job_id or ""),
        submitted=submitted,
        context=context,
        errors=errors,
    )
    runtime_profile = _audit_worker_provenance(
        result_dir,
        context=context,
        expected_study=expected_study,
        stage=stage,
        run_id=run_id,
        attempt=attempt,
        task_command=task.get("command"),
        task_metadata=task.get("metadata"),
        task_resources=resources,
        launcher_job_id=str(launcher_job_id or ""),
        worker_commits=worker_commits,
        errors=errors,
    )
    if runtime_profile is not None:
        runtime_rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "profile": runtime_profile,
            }
        )


def _audit_worker_provenance(
    result_dir: Path,
    *,
    context: str,
    expected_study: str,
    stage: str,
    run_id: str,
    attempt: str,
    task_command: object,
    task_metadata: object,
    task_resources: object,
    launcher_job_id: str,
    worker_commits: set[str],
    errors: list[str],
) -> dict[str, Any] | None:
    run_start = _read_json_for_audit(result_dir / "run_start.json", errors)
    if run_start.get("run_id") != f"{run_id}/{attempt}":
        errors.append(f"{context} run_start run_id mismatch")
    if run_start.get("run_dir") != str(result_dir):
        errors.append(f"{context} run_start run_dir mismatch")
    study = run_start.get("study")
    if not isinstance(study, dict) or study.get("name") != expected_study:
        errors.append(f"{context} run_start study identity mismatch")
    elif stage in {"01_train", "02_validation"}:
        if study.get("config_id") is not None:
            errors.append(f"{context} scan run_start config_id is not null")
    else:
        command_config_id = _command_override(
            task_command,
            "study.config_id",
        )
        if stage == "06_final_train":
            metadata_job = (
                task_metadata.get("job")
                if isinstance(task_metadata, dict)
                else None
            )
            expected_config_id = (
                metadata_job.get("source_champion_id")
                if isinstance(metadata_job, dict)
                else None
            )
        else:
            # final_eval intentionally plans a reduced job record.  Its
            # immutable per-run source copy, which is separately checked
            # against the final-grid job, owns the champion identity.
            source_final_job = _read_json_for_audit(
                result_dir / "source_final_job.json",
                errors,
            )
            expected_config_id = source_final_job.get(
                "source_champion_id"
            )
        if (
            not expected_config_id
            or command_config_id != expected_config_id
            or study.get("config_id") != expected_config_id
        ):
            errors.append(f"{context} final run_start config_id mismatch")
    git = run_start.get("git")
    if not isinstance(git, dict):
        errors.append(f"{context} run_start git provenance is missing")
    else:
        sha = str(git.get("sha") or "")
        if not sha:
            errors.append(f"{context} worker git commit is empty")
        else:
            worker_commits.add(sha)
        if git.get("dirty") is not False:
            errors.append(f"{context} worker source is dirty")
    metadata = _read_json_for_audit(result_dir / "metadata.json", errors)
    expected_runtime_id = f"{run_id}/{attempt}"
    if metadata.get("run_id") != expected_runtime_id:
        errors.append(f"{context} metadata run_id mismatch")
    if metadata.get("run_dir") != str(result_dir):
        errors.append(f"{context} metadata run_dir mismatch")
    if metadata.get("command") != run_start.get("command"):
        errors.append(f"{context} run_start/metadata command mismatch")
    if (
        isinstance(git, dict)
        and metadata.get("git_commit") != git.get("sha")
    ):
        errors.append(f"{context} run_start/metadata git commit mismatch")
    if metadata.get("dirty_worktree") is not False:
        errors.append(f"{context} metadata records a dirty worker")
    if metadata.get("status") != "completed":
        errors.append(f"{context} metadata status is not completed")
    device = metadata.get("device")
    runtime = metadata.get("runtime")
    if isinstance(runtime, dict):
        device = runtime.get("device", device)
    if device != "cuda":
        errors.append(f"{context} metadata device is not cuda")
    runtime_profile = _worker_runtime_profile(
        metadata,
        run_start=run_start,
        task_resources=task_resources,
        context=context,
        errors=errors,
    )
    slurm = metadata.get("slurm")
    run_start_slurm = run_start.get("slurm")
    if not isinstance(slurm, dict) or not isinstance(run_start_slurm, dict):
        errors.append(f"{context} metadata Slurm provenance is missing")
    else:
        for key in (
            "job_id",
            "array_task_id",
            "job_partition",
            "cpus_per_task",
        ):
            if slurm.get(key) != run_start_slurm.get(key):
                errors.append(
                    f"{context} run_start/metadata Slurm {key} mismatch"
                )
        expected_slurm = {
            "job_partition": "gpu_test",
            "cpus_per_task": "4",
        }
        for key, expected in expected_slurm.items():
            if str(slurm.get(key) or "") != expected:
                errors.append(
                    f"{context} Slurm {key}={slurm.get(key)!r}, "
                    f"expected {expected!r}"
                )
        job_id = str(slurm.get("job_id") or "")
        array_task_id = str(slurm.get("array_task_id") or "")
        if not re.fullmatch(r"[0-9]+", job_id):
            errors.append(
                f"{context} worker Slurm job_id is invalid: {job_id!r}"
            )
        if array_task_id and not re.fullmatch(r"[0-9]+", array_task_id):
            errors.append(
                f"{context} worker Slurm array_task_id is invalid: "
                f"{array_task_id!r}"
            )
        launcher_match = re.fullmatch(r"[0-9]+(?:_([0-9]+))?", launcher_job_id)
        launcher_array = (
            launcher_match.group(1) if launcher_match is not None else None
        )
        if launcher_array is None:
            if array_task_id:
                errors.append(
                    f"{context} non-array launcher has worker array_task_id"
                )
        elif array_task_id != launcher_array:
            errors.append(
                f"{context} launcher chunk and worker array_task_id differ"
            )
        environment = run_start.get("environment")
        if not isinstance(environment, dict):
            errors.append(f"{context} run_start environment is missing")
        else:
            environment_keys = {
                "job_id": "SLURM_JOB_ID",
                "array_task_id": "SLURM_ARRAY_TASK_ID",
                "job_partition": "SLURM_JOB_PARTITION",
                "cpus_per_task": "SLURM_CPUS_PER_TASK",
            }
            required_environment = {
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
            }
            missing_environment = required_environment - set(environment)
            if missing_environment:
                errors.append(
                    f"{context} run_start environment lacks "
                    f"{sorted(missing_environment)!r}"
                )
            for key, environment_key in environment_keys.items():
                if (
                    environment_key in environment
                    and environment[environment_key] != run_start_slurm.get(key)
                ):
                    errors.append(
                        f"{context} environment {environment_key} "
                        "differs from run_start Slurm"
                    )
            array_environment = environment.get("SLURM_ARRAY_TASK_ID")
            if launcher_array is None:
                if array_environment not in {None, ""}:
                    errors.append(
                        f"{context} non-array launcher records "
                        "SLURM_ARRAY_TASK_ID"
                    )
            elif array_environment != launcher_array:
                errors.append(
                    f"{context} environment SLURM_ARRAY_TASK_ID "
                    "differs from launcher chunk"
                )
    return runtime_profile


def _task_path(
    root: Path,
    value: object,
    context: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{context} has an empty artifact path")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{context} artifact path is not absolute: {value}")
        return None
    resolved = path.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        errors.append(f"{context} artifact path escapes lineage root: {value}")
        return None
    return resolved


def _complete_checkpoint_pointer(path: Path) -> Path | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        pointer = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(pointer, dict):
        return None
    directory_name = pointer.get("checkpoint_dir")
    if not isinstance(directory_name, str) or not directory_name:
        return None
    if Path(directory_name).is_absolute() or ".." in Path(directory_name).parts:
        return None
    concrete = (path.parent / directory_name).resolve(strict=False)
    if concrete.parent != path.parent.resolve():
        return None
    if not (
        concrete.is_dir()
        and not concrete.is_symlink()
        and (concrete / "COMPLETE").is_file()
        and not (concrete / "COMPLETE").is_symlink()
        and (concrete / "manifest.json").is_file()
        and not (concrete / "manifest.json").is_symlink()
    ):
        return None
    try:
        step = int(pointer["step"])
        manifest = json.loads((concrete / "manifest.json").read_text())
        manifest_step = int(manifest["step"])
        files = manifest["files"]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if directory_name != f"step_{step:06d}" or manifest_step != step:
        return None
    if not isinstance(files, dict) or not files:
        return None
    for value in files.values():
        if not isinstance(value, str) or not value:
            return None
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        artifact = (concrete / relative).resolve(strict=False)
        if concrete not in artifact.parents or not artifact.is_file():
            return None
    return concrete


def _final_selection_matches(result_dir: Path, checkpoint_dir: Path) -> bool:
    selection_path = result_dir / "selected_checkpoint.json"
    if not selection_path.is_file() or selection_path.is_symlink():
        return False
    try:
        selection = json.loads(selection_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(selection, dict):
        return False
    pointer = result_dir / "checkpoints" / "latest.json"
    if selection.get("selection_policy") != "latest_checkpoint_pointer":
        return False
    if Path(str(selection.get("checkpoint_pointer") or "")).resolve(
        strict=False
    ) != pointer.resolve(strict=False):
        return False
    if Path(str(selection.get("checkpoint_dir") or "")).resolve(
        strict=False
    ) != pointer.parent.resolve(strict=False):
        return False
    resolved = selection.get("resolved_checkpoint_dir")
    return resolved in {None, ""} or Path(str(resolved)).resolve(
        strict=False
    ) == checkpoint_dir


def _valid_metrics(path: Path, *, require_eval: bool) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        return False
    found_metric = False
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return False
            namespace = value.get("namespace")
            metrics = value.get("metrics")
            if isinstance(metrics, dict) and metrics:
                if not require_eval or (
                    isinstance(namespace, str) and namespace.startswith("eval/")
                ):
                    found_metric = True
    except (OSError, json.JSONDecodeError):
        return False
    return found_metric


def _audit_evaluation_status_metrics(
    path: Path,
    *,
    required_tasks: Sequence[str],
    context: str,
    errors: list[str],
) -> dict[str, tuple[str, ...]]:
    """Require successful suite/task terminal metrics from the evaluator."""

    schemas: dict[str, tuple[str, ...]] = {}
    by_namespace: dict[str, list[dict[str, Any]]] = {}
    if not path.is_file() or path.is_symlink():
        errors.append(f"{context} evaluation status metrics are missing")
        return schemas
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("metric row is not an object")
            namespace = value.get("namespace")
            metrics = value.get("metrics")
            if isinstance(namespace, str) and isinstance(metrics, dict):
                by_namespace.setdefault(namespace, []).append(metrics)
                if namespace == "eval/status":
                    if metrics.get("suite_failed") is True:
                        errors.append(
                            f"{context} eval/status records suite_failed=true"
                        )
                    if metrics.get("suite_success") is False:
                        errors.append(
                            f"{context} eval/status records suite_success=false"
                        )
                elif (
                    namespace.startswith("eval/")
                    and namespace.endswith("/status")
                ):
                    if metrics.get("task_failed") is True:
                        errors.append(
                            f"{context} {namespace} records task_failed=true"
                        )
                    if metrics.get("task_success") is False:
                        errors.append(
                            f"{context} {namespace} records task_success=false"
                        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{context} cannot parse evaluation statuses: {exc}")
        return schemas
    suite_rows = by_namespace.get("eval/status", [])
    if len(suite_rows) != 1:
        errors.append(f"{context} eval/status count is not exactly one")
    suite = suite_rows[0] if len(suite_rows) == 1 else {}
    if suite.get("suite_success") is not True:
        errors.append(f"{context} eval suite_success is not true")
    if suite.get("suite_failed") is not False:
        errors.append(f"{context} eval suite_failed is not false")
    if not required_tasks:
        errors.append(f"{context} has no configured evaluation task contract")
        return schemas
    aggregate_perf = by_namespace.get("eval/perf", [])
    if len(aggregate_perf) != 1:
        errors.append(f"{context} eval/perf count is not exactly one")
    aggregate_metrics = (
        aggregate_perf[0] if len(aggregate_perf) == 1 else {}
    )
    aggregate_keys = _scalar_metric_keys(
        aggregate_metrics,
        context=f"{context} eval/perf",
        errors=errors,
    )
    schemas["perf"] = aggregate_keys
    for task in required_tasks:
        science_rows = by_namespace.get(f"eval/{task}", [])
        if len(science_rows) != 1:
            errors.append(
                f"{context} eval/{task} science count is not exactly one"
            )
            science = {}
        else:
            science = science_rows[0]
        if not science:
            errors.append(f"{context} {task} has no science metrics")
        scalar_keys = _scalar_metric_keys(
            science,
            context=f"{context} {task} science",
            errors=errors,
        )
        anchor = SCIENCE_METRIC_ANCHORS.get(task)
        if anchor is None:
            errors.append(f"{context} {task} has no frozen science anchor")
        elif anchor not in science:
            errors.append(
                f"{context} {task} lacks required science metric {anchor}"
            )
        schemas[task] = tuple(sorted(str(key) for key in science))
        perf_namespace = f"eval/perf/{task}"
        perf_rows = by_namespace.get(perf_namespace, [])
        if len(perf_rows) != 1:
            errors.append(
                f"{context} {perf_namespace} count is not exactly one"
            )
        perf_metrics = perf_rows[0] if len(perf_rows) == 1 else {}
        schemas[f"perf/{task}"] = _scalar_metric_keys(
            perf_metrics,
            context=f"{context} {perf_namespace}",
            errors=errors,
        )
        namespace = f"eval/{task}/status"
        status_rows = by_namespace.get(namespace, [])
        if len(status_rows) != 1:
            errors.append(f"{context} {namespace} count is not exactly one")
        status = status_rows[0] if len(status_rows) == 1 else {}
        if status.get("task_success") is not True:
            errors.append(f"{context} {task} task_success is not true")
        if status.get("task_failed") is not False:
            errors.append(f"{context} {task} task_failed is not false")
    for namespace, status_rows in by_namespace.items():
        if (
            namespace.startswith("eval/")
            and namespace.endswith("/status")
            and len(status_rows) > 1
            and namespace != "eval/status"
            and namespace
            not in {f"eval/{task}/status" for task in required_tasks}
        ):
            errors.append(f"{context} {namespace} status is duplicated")
    unexpected_science = {
        namespace
        for namespace in by_namespace
        if namespace.startswith("eval/")
        and namespace not in {"eval/status"}
        and not namespace.endswith("/status")
        and namespace != "eval/perf"
        and namespace.removeprefix("eval/perf/") not in set(required_tasks)
        and namespace.removeprefix("eval/") not in set(required_tasks)
    }
    if unexpected_science:
        errors.append(
            f"{context} has unexpected science namespaces "
            f"{sorted(unexpected_science)!r}"
        )
    return schemas


def _is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, bool | int | float | str)


def _scalar_metric_keys(
    metrics: Mapping[str, Any],
    *,
    context: str,
    errors: list[str],
) -> tuple[str, ...]:
    if not metrics:
        errors.append(f"{context} has no metrics")
        return ()
    keys = tuple(
        sorted(
            key
            for key, value in metrics.items()
            if isinstance(key, str) and key and _is_json_scalar(value)
        )
    )
    if len(keys) != len(metrics):
        errors.append(f"{context} metrics are not nonempty scalar fields")
    return keys


def _audit_science_schema_consistency(
    observations: Mapping[str, Mapping[str, Sequence[tuple[str, ...]]]],
    errors: list[str],
) -> None:
    """Require every run in a stage to expose one identical task schema."""

    for stage in ("02_validation", "07_final_eval"):
        expected_count = int(STAGE_EXPECTATIONS[stage]["count"])
        tasks = observations.get(stage, {})
        for task, schemas in tasks.items():
            if len(schemas) != expected_count:
                errors.append(
                    f"{stage} {task} science schema count is {len(schemas)}, "
                    f"expected {expected_count}"
                )
            unique = {tuple(schema) for schema in schemas}
            if len(unique) != 1:
                errors.append(
                    f"{stage} {task} science metric key schema differs across runs"
                )


def _worker_runtime_profile(
    metadata: Mapping[str, Any],
    *,
    run_start: Mapping[str, Any],
    task_resources: object,
    context: str,
    errors: list[str],
) -> dict[str, Any] | None:
    """Validate and return the stable GPU worker-runtime projection."""

    runtime = metadata.get("runtime")
    hardware = metadata.get("hardware")
    resources = task_resources if isinstance(task_resources, Mapping) else {}
    if not isinstance(runtime, dict):
        errors.append(f"{context} metadata runtime is missing")
        return None
    if not isinstance(hardware, dict):
        errors.append(f"{context} metadata hardware is missing")
        return None
    for owner, value in (
        ("metadata", metadata.get("device")),
        ("runtime", runtime.get("device")),
    ):
        if value != "cuda":
            errors.append(f"{context} {owner} device is not cuda")
    for owner, value in (
        ("metadata", metadata.get("dtype")),
        ("runtime", runtime.get("dtype")),
    ):
        if value != "float64":
            errors.append(f"{context} {owner} dtype is not float64")

    python_version = runtime.get("python_version")
    python_executable = runtime.get("python_executable")
    torch_version = runtime.get("torch_version")
    torch_cuda_version = runtime.get("torch_cuda_version")
    for key, value in (
        ("python_version", python_version),
        ("python_executable", python_executable),
        ("torch_version", torch_version),
        ("torch_cuda_version", torch_cuda_version),
    ):
        if not isinstance(value, str) or not value:
            errors.append(f"{context} runtime {key} is empty")
    executable = (
        Path(python_executable)
        if isinstance(python_executable, str) and python_executable
        else None
    )
    environment_name = (
        executable.parent.parent.name if executable is not None else ""
    )
    expected_environment = resources.get("uv_environment")
    if executable is None or not executable.is_absolute():
        errors.append(f"{context} runtime Python executable is not absolute")
    if environment_name != ".venv-gpu" or expected_environment != ".venv-gpu":
        errors.append(f"{context} runtime Python environment is not .venv-gpu")

    cuda_available = hardware.get("cuda_available")
    cuda_device_count = hardware.get("cuda_device_count")
    cuda_devices = hardware.get("cuda_devices")
    expected_gpus = resources.get("gpus")
    if cuda_available is not True:
        errors.append(f"{context} hardware CUDA is unavailable")
    if cuda_device_count != 1 or expected_gpus != 1:
        errors.append(f"{context} CUDA device/resource count is not one")
    if not isinstance(cuda_devices, list) or len(cuda_devices) != 1:
        errors.append(f"{context} CUDA device inventory is not one device")
        stable_devices: list[dict[str, Any]] = []
    else:
        stable_devices = []
        for device in cuda_devices:
            if not isinstance(device, dict):
                errors.append(f"{context} CUDA device row is not an object")
                continue
            required = {"index", "name", "total_memory_bytes", "capability"}
            if set(device) != required:
                errors.append(f"{context} CUDA device fields mismatch")
            if device.get("index") != 0:
                errors.append(f"{context} CUDA device index is not zero")
            if not isinstance(device.get("name"), str) or not device.get("name"):
                errors.append(f"{context} CUDA device name is empty")
            if (
                not isinstance(device.get("total_memory_bytes"), int)
                or device["total_memory_bytes"] <= 0
            ):
                errors.append(f"{context} CUDA device memory is invalid")
            if (
                not isinstance(device.get("capability"), str)
                or not device.get("capability")
            ):
                errors.append(f"{context} CUDA device capability is empty")
            stable_devices.append(dict(device))

    environment = run_start.get("environment")
    visible = runtime.get("cuda_visible_devices")
    start_visible = (
        environment.get("CUDA_VISIBLE_DEVICES")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(visible, str) or not visible:
        errors.append(f"{context} runtime CUDA_VISIBLE_DEVICES is empty")
    if visible != start_visible:
        errors.append(
            f"{context} runtime CUDA_VISIBLE_DEVICES differs from run_start"
        )

    return {
        "device": runtime.get("device"),
        "dtype": runtime.get("dtype"),
        "python_version": python_version,
        "python_executable": python_executable,
        "python_environment": environment_name,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": stable_devices,
    }


def _audit_worker_runtime_uniformity(
    runtime_rows: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    """Require all 144 fan-out tasks to use one stable GPU profile."""

    expected_total = sum(
        int(expectation["count"])
        for expectation in STAGE_EXPECTATIONS.values()
    )
    if len(runtime_rows) != expected_total:
        errors.append(
            f"worker runtime profile population is {len(runtime_rows)}, "
            f"expected {expected_total}"
        )
    profile_digests: set[str] = set()
    stage_counts: dict[str, int] = {}
    for row in runtime_rows:
        stage = str(row.get("stage") or "")
        profile = row.get("profile")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if isinstance(profile, Mapping):
            profile_digests.add(_canonical_sha256(profile))
    for stage, expectation in STAGE_EXPECTATIONS.items():
        expected_count = int(expectation["count"])
        if stage_counts.get(stage, 0) != expected_count:
            errors.append(
                f"{stage} worker runtime profile count is "
                f"{stage_counts.get(stage, 0)}, expected {expected_count}"
            )
    if len(profile_digests) != 1:
        errors.append(
            f"fan-out workers expose {len(profile_digests)} stable runtime profiles"
        )


def _science_schema_summary(
    observations: Mapping[str, Mapping[str, Sequence[tuple[str, ...]]]],
) -> dict[str, Any]:
    stages: dict[str, dict[str, list[str]]] = {}
    for stage in ("02_validation", "07_final_eval"):
        stage_tasks: dict[str, list[str]] = {}
        for task, schemas in sorted(observations.get(stage, {}).items()):
            unique = sorted({tuple(schema) for schema in schemas})
            stage_tasks[task] = list(unique[0]) if len(unique) == 1 else []
        stages[stage] = stage_tasks
    return {
        "schema_version": "pair-stability-v4/science-metric-schema/v1",
        "stages": stages,
        "schema_sha256": _canonical_sha256(stages),
    }


def _worker_runtime_summary(
    runtime_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {}
    aggregate_rows: list[tuple[str, str, str]] = []
    for stage in STAGE_EXPECTATIONS:
        profile_counts: dict[str, dict[str, Any]] = {}
        stage_rows = [
            row for row in runtime_rows if row.get("stage") == stage
        ]
        for row in stage_rows:
            profile = row.get("profile")
            if not isinstance(profile, Mapping):
                continue
            digest = _canonical_sha256(profile)
            entry = profile_counts.setdefault(
                digest,
                {
                    "profile_sha256": digest,
                    "count": 0,
                    "profile": dict(profile),
                },
            )
            entry["count"] += 1
            aggregate_rows.append(
                (stage, str(row.get("run_id") or ""), digest)
            )
        stages[stage] = {
            "task_count": len(stage_rows),
            "profiles": [
                profile_counts[digest] for digest in sorted(profile_counts)
            ],
        }
    aggregate_rows.sort()
    return {
        "schema_version": WORKER_RUNTIME_SCHEMA_VERSION,
        "volatile_json_pointers": list(WORKER_RUNTIME_VOLATILE_POINTERS),
        "stages": stages,
        "assignments": [
            {
                "stage": stage,
                "run_id": run_id,
                "profile_sha256": digest,
            }
            for stage, run_id, digest in aggregate_rows
        ],
        "aggregate_sha256": _canonical_sha256(aggregate_rows),
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_diagnostics_index(
    result_dir: Path,
    *,
    required_tasks: Sequence[str],
    context: str,
    errors: list[str],
) -> None:
    index = _read_json_for_audit(
        result_dir / "diagnostics" / "index.json",
        errors,
    )
    tasks = index.get("tasks")
    if not isinstance(tasks, list):
        errors.append(f"{context} diagnostic task index is not a list")
        return
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in tasks:
        if not isinstance(row, dict):
            errors.append(f"{context} diagnostic task row is not an object")
            continue
        name = str(row.get("name") or "")
        namespace = str(row.get("namespace") or "")
        if not name or name in by_name:
            errors.append(f"{context} diagnostic task names are invalid/duplicate")
            continue
        if namespace != f"eval/{name}":
            errors.append(f"{context} diagnostic namespace mismatch for {name}")
        if row.get("status") != "success":
            errors.append(f"{context} diagnostic task {name} is not successful")
        output = Path(str(row.get("output_dir") or "")).resolve(strict=False)
        expected_output = (result_dir / name).resolve(strict=False)
        if output != expected_output or result_dir not in output.parents:
            errors.append(f"{context} diagnostic output_dir mismatch for {name}")
        by_name[name] = row
    if set(by_name) != set(required_tasks):
        errors.append(f"{context} diagnostic task population differs from suite")


def _audit_scan_validation_source(
    result_dir: Path,
    *,
    task: Mapping[str, Any],
    run_id: str,
    attempt: str,
    root: Path,
    context: str,
    errors: list[str],
) -> None:
    source = _read_json_for_audit(
        result_dir / "source_train_attempt.json",
        errors,
    )
    expected_dir = root / "01_train" / run_id / attempt
    expected = {
        "run_id": run_id,
        "grid_attempt_id": attempt,
        "train_attempt_id": attempt,
        "train_dir": str(root / "01_train" / run_id),
        "train_attempt_dir": str(expected_dir),
        "checkpoint_path": str(expected_dir / "checkpoints"),
    }
    for key, value in expected.items():
        if source.get(key) != value:
            errors.append(
                f"{context} scan source {key}={source.get(key)!r}, "
                f"expected {value!r}"
            )
    metadata = task.get("metadata")
    job = metadata.get("job") if isinstance(metadata, dict) else None
    planned_source = (
        job.get("source_train_attempt")
        if isinstance(job, dict)
        else None
    )
    if planned_source != source:
        errors.append(
            f"{context} task metadata source_train_attempt differs from worker"
        )


def _audit_final_evaluation_source(
    result_dir: Path,
    *,
    run_id: str,
    attempt: str,
    root: Path,
    context: str,
    errors: list[str],
) -> None:
    source = _read_json_for_audit(
        result_dir / "source_final_train_attempt.json",
        errors,
    )
    evaluated = _read_json_for_audit(
        result_dir / "evaluated_checkpoint.json",
        errors,
    )
    train_dir = root / "06_final_train" / run_id / attempt
    concrete = _complete_checkpoint_pointer(
        train_dir / "checkpoints" / "latest.json"
    )
    if source.get("final_run_id") != run_id:
        errors.append(f"{context} final-train source run id mismatch")
    if source.get("final_train_attempt_id") != attempt:
        errors.append(f"{context} final-train source attempt mismatch")
    if source.get("final_train_attempt_dir") != str(train_dir):
        errors.append(f"{context} final-train source directory mismatch")
    checkpoint = source.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint != evaluated:
        errors.append(f"{context} evaluated checkpoint differs from source record")
    if concrete is None or evaluated.get("resolved_checkpoint_dir") != str(concrete):
        errors.append(f"{context} evaluated checkpoint is not selected complete checkpoint")
    if concrete is not None and not _final_selection_matches(train_dir, concrete):
        errors.append(f"{context} source final-train selection is invalid")


def _audit_stage_source_edges(
    root: Path,
    *,
    result_dir: Path,
    stage: str,
    run_id: str,
    attempt: str,
    context: str,
    errors: list[str],
) -> None:
    if stage in {"01_train", "02_validation"}:
        source = _read_json_for_audit(
            result_dir / "source_grid_attempt.json",
            errors,
        )
        expected = {
            "run_id": run_id,
            "grid_attempt_id": attempt,
            "grid_attempt_dir": str(root / "00_grid" / attempt),
        }
        if stage == "01_train":
            expected["manifest_path"] = str(
                root / "00_grid" / attempt / "manifest.json"
            )
        _expect_exact_record(source, expected, f"{context} source-grid", errors)
        return

    final_grid_dir = root / "05_final_grid" / attempt
    source_grid = _read_json_for_audit(
        result_dir / "source_final_grid_attempt.json",
        errors,
    )
    _expect_exact_record(
        source_grid,
        {
            "final_grid_attempt_id": attempt,
            "final_grid_attempt_dir": str(final_grid_dir),
            "final_jobs_path": str(final_grid_dir / "final_jobs.csv"),
        },
        f"{context} source-final-grid",
        errors,
    )
    job_path = final_grid_dir / "jobs" / f"{run_id}.json"
    job = _read_json_for_audit(job_path, errors)
    source_job = _read_json_for_audit(
        result_dir / "source_final_job.json",
        errors,
    )
    if source_job != job:
        errors.append(f"{context} source final job differs from final-grid job")
    champion = _read_json_for_audit(
        result_dir / "source_champion.json",
        errors,
    )
    expected_champion = job.get("source_champion")
    if not isinstance(expected_champion, dict) or champion != expected_champion:
        errors.append(f"{context} source champion differs from final-grid job")


def _audit_submission_record(
    result_dir: Path,
    *,
    stage: str,
    attempt: str,
    run_id: str,
    task_command: object,
    launcher_job_id: str,
    submitted: object,
    context: str,
    errors: list[str],
) -> None:
    submission = _read_json_for_audit(
        result_dir / "submission.json",
        errors,
    )
    run_key = (
        "final_run_id"
        if stage in {"06_final_train", "07_final_eval"}
        else "run_id"
    )
    expected_attempts = {
        "01_train": {"grid_attempt_id": attempt},
        "02_validation": {
            "grid_attempt_id": attempt,
            "train_attempt_id": attempt,
            "validation_attempt_id": attempt,
        },
        "06_final_train": {
            "final_grid_attempt_id": attempt,
            "final_train_attempt_id": attempt,
        },
        "07_final_eval": {
            "final_grid_attempt_id": attempt,
            "final_train_attempt_id": attempt,
            "final_eval_attempt_id": attempt,
        },
    }[stage]
    expected_fields = {
        run_key,
        *expected_attempts,
        "launcher",
        "launcher_job_id",
        "command",
        "submitted_command",
    }
    if set(submission) != expected_fields:
        errors.append(f"{context} submission fields mismatch")
    if submission.get(run_key) != run_id:
        errors.append(f"{context} submission run identity mismatch")
    if submission.get("launcher") != "submitit":
        errors.append(f"{context} submission launcher is not submitit")
    if submission.get("launcher_job_id") != launcher_job_id:
        errors.append(f"{context} submission launcher identity mismatch")
    if submission.get("command") != (
        shlex.join(task_command)
        if isinstance(task_command, list)
        else ""
    ):
        errors.append(f"{context} submission base command differs from task")
    if (
        isinstance(submitted, list)
        and submission.get("submitted_command") != shlex.join(submitted)
    ):
        errors.append(f"{context} submission command differs from execution record")
    for key, value in expected_attempts.items():
        if submission.get(key) != value:
            errors.append(
                f"{context} submission {key}={submission.get(key)!r}, "
                f"expected {value!r}"
            )


def _expect_exact_record(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if set(actual) != set(expected):
        errors.append(f"{label} fields mismatch")
    for key, value in expected.items():
        if actual.get(key) != value:
            errors.append(
                f"{label} {key}={actual.get(key)!r}, expected {value!r}"
            )


def _expected_submitted_command(command: object) -> list[str]:
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        return []
    run_command = [
        part
        for part in command
        if not part.startswith("runtime.device=")
    ]
    run_command.append("runtime.device=cuda")
    if Path(run_command[0]).name.startswith("python"):
        run_command[0] = "python"
    sync_lock = REPO_ROOT.parent / f".{REPO_ROOT.name}.uv-sync.lock"
    script = "\n".join(
        (
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export UV_PROJECT_ENVIRONMENT=.venv-gpu",
            "# Submitit chunks share this checkout and must not sync concurrently.",
            f"flock {shlex.quote(str(sync_lock))} uv sync --extra cu126",
            "source .venv-gpu/bin/activate",
            f"exec {shlex.join(run_command)}",
        )
    )
    return ["bash", "-lc", script]


def _rows_by_identity(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
    label: str,
    errors: list[str],
) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = (
            str(row.get("task_id") or ""),
            str(row.get("run_id") or ""),
            str(row.get("stage") or ""),
            str(row.get("attempt_id") or ""),
        )
        if not all(identity):
            errors.append(f"{stage} {label} {index} has incomplete identity")
        if identity in result:
            errors.append(f"{stage} has duplicate {label} identity {identity!r}")
        result[identity] = row
    return result


def _audit_launcher_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    expected_count: int,
    chunk_size: int,
    errors: list[str],
) -> None:
    expected_groups = expected_count // chunk_size
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get("launcher_job_id") or "")
        counts[value] = counts.get(value, 0) + 1
    if len(counts) != expected_groups or sorted(counts.values()) != [
        chunk_size
    ] * expected_groups:
        errors.append(
            f"{stage} launcher chunk grouping differs: {counts!r}"
        )
        return
    if expected_groups == 1:
        value = next(iter(counts), "")
        if re.fullmatch(r"[0-9]+", value) is None:
            errors.append(
                f"{stage} single-chunk launcher id is invalid: {value!r}"
            )
        return
    masters: set[str] = set()
    suffixes: set[int] = set()
    for value in counts:
        match = re.fullmatch(r"([0-9]+)_([0-9]+)", value)
        if match is None:
            errors.append(f"{stage} launcher chunk id is invalid: {value!r}")
            continue
        masters.add(match.group(1))
        suffixes.add(int(match.group(2)))
    if len(masters) != 1 or suffixes != set(range(expected_groups)):
        errors.append(
            f"{stage} launcher chunk master/suffix grouping differs"
        )


def _command_override(command: object, key: str) -> str | None:
    if not isinstance(command, list):
        return None
    prefix = f"{key}="
    values = [
        part[len(prefix) :]
        for part in command
        if isinstance(part, str) and part.startswith(prefix)
    ]
    if len(values) != 1 or not values[0]:
        return None
    return values[0]


def _read_json_for_audit(
    path: Path,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        errors.append(f"invalid JSON: {path}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object: {path}")
        return {}
    return value


def _read_jsonl(path: Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing JSONL: {path}")
        return []
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"invalid JSONL {path}:{line_number}: {exc}")
    return rows
