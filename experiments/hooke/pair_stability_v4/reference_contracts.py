"""Versioned structural verification for frozen reference evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml

from fanout_audit import (
    SCIENCE_METRIC_ANCHORS,
    STAGE_EXPECTATIONS,
    WORKER_RUNTIME_SCHEMA_VERSION,
    WORKER_RUNTIME_VOLATILE_POINTERS,
)
from reference_evidence import (
    CHECKPOINT_EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_INPUT_SCHEMA_VERSION,
    FANOUT_ATTEMPTS,
)
from roots import validate_lineage_id
from strict_data import loads_json, loads_yaml


class ReferenceArtifact(Protocol):
    """Artifact fields needed by structural contract verification."""

    logical_path: str
    source_path: str
    stored_path: str
    encoding: str
    raw_sha256: str
    raw_size: int
    reference_dir: Path | None


_REQUIRED_ATTEMPTS = {
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
}


def _artifact_bytes(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> bytes:
    artifact = by_logical.get(logical_path)
    if artifact is None or artifact.reference_dir is None:
        raise ValueError(f"missing protected contract artifact: {logical_path}")
    stored = artifact.reference_dir / artifact.stored_path
    if stored.is_symlink() or not stored.is_file():
        raise ValueError(f"stored contract artifact is unavailable: {logical_path}")
    if artifact.encoding == "raw":
        raw = stored.read_bytes()
    elif artifact.encoding == "gzip":
        with gzip.open(stored, "rb") as handle:
            raw = handle.read()
    else:
        raise ValueError(f"unsupported contract artifact encoding: {logical_path}")
    if len(raw) != artifact.raw_size or hashlib.sha256(raw).hexdigest() != artifact.raw_sha256:
        raise ValueError(f"raw contract artifact digest differs: {logical_path}")
    return raw


def _artifact_json(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> dict[str, Any]:
    value = loads_json(
        _artifact_bytes(by_logical, logical_path),
        source=f"frozen contract {logical_path}",
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected frozen JSON object: {logical_path}")
    return value


def _artifact_jsonl(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _artifact_bytes(by_logical, logical_path).decode().splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = loads_json(
            line,
            source=f"frozen contract {logical_path}:{line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"frozen JSONL row is not an object: {logical_path}:{line_number}"
            )
        rows.append(value)
    return rows


def _artifact_yaml(
    by_logical: Mapping[str, ReferenceArtifact],
    logical_path: str,
) -> dict[str, Any]:
    value = loads_yaml(
        _artifact_bytes(by_logical, logical_path),
        source=f"frozen contract {logical_path}",
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected frozen YAML object: {logical_path}")
    return value


def _normalize_attempts(value: Mapping[str, object]) -> dict[str, str]:
    if set(value) != _REQUIRED_ATTEMPTS:
        raise ValueError("reference attempt keys mismatch")
    return {
        key: validate_lineage_id(str(value[key]))
        for key in sorted(_REQUIRED_ATTEMPTS)
    }


def _safe_nested_path(value: object) -> Path:
    path = Path(str(value))
    if (
        not str(value)
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise ValueError(f"unsafe nested artifact path: {value!r}")
    return path


def _safe_relative_text(value: object) -> str:
    return _safe_nested_path(value).as_posix()


def _safe_component(value: object, name: str) -> str:
    text = str(value or "")
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"invalid {name}: {text!r}")
    return text


def _digest_text(value: object) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError("invalid SHA-256 digest")
    return text


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_science_metric_summary(
    value: object,
    *,
    artifacts: Sequence[ReferenceArtifact],
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "stages",
        "schema_sha256",
    }:
        return ["reference science-metric summary schema mismatch"]
    errors: list[str] = []
    if (
        value.get("schema_version")
        != "pair-stability-v4/science-metric-schema/v1"
    ):
        errors.append("reference science-metric summary version mismatch")
    stages = value.get("stages")
    try:
        by_logical = {
            artifact.logical_path: artifact for artifact in artifacts
        }
        validation_config = _artifact_yaml(
            by_logical,
            "00_grid/{grid}/validation_config.yaml",
        )
        configured_tasks = {
            "02_validation": _configured_task_names(
                validation_config,
                suite="validation",
            ),
            "07_final_eval": _configured_task_names(
                validation_config,
                suite="final_eval",
            ),
        }
        expected_tasks = {
            stage: (
                *names,
                "perf",
                *(f"perf/{name}" for name in names),
            )
            for stage, names in configured_tasks.items()
        }
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(
            f"cannot derive frozen science task contract: {exc}"
        )
        expected_tasks = {}
    if not isinstance(stages, dict) or set(stages) != set(expected_tasks):
        errors.append("reference science-metric stage set mismatch")
    else:
        for stage, tasks in stages.items():
            expected_stage_tasks = expected_tasks[stage]
            if (
                not isinstance(tasks, dict)
                or set(tasks) != set(expected_stage_tasks)
            ):
                errors.append(
                    f"reference science-metric task population differs "
                    f"for {stage}"
                )
                continue
            for task, keys in tasks.items():
                if (
                    not isinstance(task, str)
                    or not task
                    or not isinstance(keys, list)
                    or not keys
                    or keys != sorted(set(keys))
                    or not all(isinstance(key, str) and key for key in keys)
                ):
                    errors.append(
                        f"reference science-metric keys invalid for "
                        f"{stage}/{task}"
                    )
                    continue
                anchor = SCIENCE_METRIC_ANCHORS.get(task)
                if task == "perf":
                    anchor = "wall_time_sec"
                if task.startswith("perf/"):
                    anchor = None
                if anchor is not None and anchor not in keys:
                    errors.append(
                        f"reference science-metric anchor missing for "
                        f"{stage}/{task}"
                    )
        if value.get("schema_sha256") != canonical_sha256(stages):
            errors.append("reference science-metric digest mismatch")
    return errors


def _configured_task_names(
    config: Mapping[str, Any],
    *,
    suite: str,
) -> tuple[str, ...]:
    """Resolve an evaluation suite from a frozen validation config."""

    suites = config.get("evaluation_suites")
    specifications = config.get("evaluation_tasks")
    if not isinstance(suites, Mapping) or not isinstance(
        specifications,
        Mapping,
    ):
        raise ValueError("validation evaluation contracts are missing")
    suite_spec = suites.get(suite)
    if not isinstance(suite_spec, Mapping) or not isinstance(
        suite_spec.get("tasks"),
        list,
    ):
        raise ValueError(f"evaluation suite {suite!r} is invalid")
    names: list[str] = []
    for reference in suite_spec["tasks"]:
        text = str(reference)
        prefix = "${evaluation_tasks."
        if not text.startswith(prefix) or not text.endswith("}"):
            raise ValueError(
                f"evaluation task reference is not explicit: {text}"
            )
        key = text[len(prefix) : -1]
        task = specifications.get(key)
        if not isinstance(task, Mapping):
            raise ValueError(f"evaluation task {key!r} is missing")
        name = str(task.get("name") or "")
        if not name or "/" in name:
            raise ValueError(f"evaluation task {key!r} has invalid name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"evaluation suite {suite!r} has duplicate names")
    return tuple(names)


def _verify_worker_runtime_summary(
    value: object,
    *,
    artifacts: Sequence[ReferenceArtifact],
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "volatile_json_pointers",
        "stages",
        "assignments",
        "aggregate_sha256",
    }:
        return ["reference worker-runtime summary schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != WORKER_RUNTIME_SCHEMA_VERSION:
        errors.append("reference worker-runtime version mismatch")
    if value.get("volatile_json_pointers") != list(
        WORKER_RUNTIME_VOLATILE_POINTERS
    ):
        errors.append("reference worker-runtime volatile allowlist mismatch")
    stages = value.get("stages")
    assignments = value.get("assignments")
    if not isinstance(stages, dict) or set(stages) != set(
        STAGE_EXPECTATIONS
    ):
        errors.append("reference worker-runtime stage set mismatch")
        return errors
    if not isinstance(assignments, list):
        errors.append("reference worker-runtime assignments are not a list")
        return errors
    assignment_tuples: list[tuple[str, str, str]] = []
    assignment_counts: dict[tuple[str, str], int] = {}
    for row in assignments:
        if not isinstance(row, dict) or set(row) != {
            "stage",
            "run_id",
            "profile_sha256",
        }:
            errors.append("reference worker-runtime assignment fields mismatch")
            continue
        stage = str(row.get("stage") or "")
        run_id = str(row.get("run_id") or "")
        digest = str(row.get("profile_sha256") or "")
        try:
            _digest_text(digest)
        except ValueError:
            errors.append("reference worker-runtime assignment digest invalid")
        assignment_tuples.append((stage, run_id, digest))
        assignment_counts[(stage, digest)] = (
            assignment_counts.get((stage, digest), 0) + 1
        )
    if assignment_tuples != sorted(assignment_tuples):
        errors.append("reference worker-runtime assignments are not sorted")
    if len(set(assignment_tuples)) != len(assignment_tuples):
        errors.append("reference worker-runtime assignments are duplicated")
    if value.get("aggregate_sha256") != canonical_sha256(
        sorted(assignment_tuples)
    ):
        errors.append("reference worker-runtime aggregate digest mismatch")
    try:
        expected_assignments = _runtime_task_contract(artifacts)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"cannot derive frozen runtime task contract: {exc}")
        expected_assignments = set()
    if {
        (stage, run_id) for stage, run_id, _digest in assignment_tuples
    } != expected_assignments:
        errors.append(
            "reference worker-runtime assignments differ from frozen tasks"
        )

    all_profiles: set[str] = set()
    for stage, expectation in STAGE_EXPECTATIONS.items():
        stage_value = stages.get(stage)
        if not isinstance(stage_value, dict) or set(stage_value) != {
            "task_count",
            "profiles",
        }:
            errors.append(
                f"reference worker-runtime stage schema invalid for {stage}"
            )
            continue
        expected_count = int(expectation["count"])
        if stage_value.get("task_count") != expected_count:
            errors.append(
                f"reference worker-runtime task count differs for {stage}"
            )
        profiles = stage_value.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            errors.append(
                f"reference worker-runtime profiles missing for {stage}"
            )
            continue
        observed_count = 0
        for profile_row in profiles:
            if not isinstance(profile_row, dict) or set(profile_row) != {
                "profile_sha256",
                "count",
                "profile",
            }:
                errors.append(
                    f"reference worker-runtime profile fields invalid for {stage}"
                )
                continue
            profile = profile_row.get("profile")
            digest = str(profile_row.get("profile_sha256") or "")
            if not isinstance(profile, dict):
                errors.append(
                    f"reference worker-runtime profile is invalid for {stage}"
                )
            else:
                errors.extend(
                    _verify_worker_runtime_profile(
                        profile,
                        context=f"{stage}/{digest}",
                    )
                )
            if not isinstance(profile, dict) or canonical_sha256(profile) != digest:
                errors.append(
                    f"reference worker-runtime profile digest differs for {stage}"
                )
            count = profile_row.get("count")
            if not isinstance(count, int) or count <= 0:
                errors.append(
                    f"reference worker-runtime profile count invalid for {stage}"
                )
                continue
            observed_count += count
            if assignment_counts.get((stage, digest), 0) != count:
                errors.append(
                    f"reference worker-runtime assignment count differs for {stage}"
                )
            all_profiles.add(digest)
        if observed_count != expected_count:
            errors.append(
                f"reference worker-runtime profile population differs for {stage}"
            )
    if len(all_profiles) != 1:
        errors.append("reference workers do not share one stable runtime profile")
    return errors


def _runtime_task_contract(
    artifacts: Sequence[ReferenceArtifact],
) -> set[tuple[str, str]]:
    """Return exact GPU worker assignments derived from protected plans."""

    by_logical = {
        artifact.logical_path: artifact for artifact in artifacts
    }
    for logical_path in (
        "00_grid/{grid}/train_config.yaml",
        "00_grid/{grid}/validation_config.yaml",
    ):
        config = _artifact_yaml(by_logical, logical_path)
        runtime = config.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("dtype") != "float64":
            raise ValueError(
                f"frozen runtime dtype is not float64: {logical_path}"
            )
    assignments: set[tuple[str, str]] = set()
    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        tasks = _artifact_jsonl(
            by_logical,
            f"{stage}/stage_plans/{{{attempt_key}}}/tasks.jsonl",
        )
        for task in tasks:
            run_id = _safe_component(
                task.get("run_id"),
                f"frozen {stage} runtime run_id",
            )
            resources = task.get("resources")
            if not isinstance(resources, Mapping):
                raise ValueError(f"frozen {stage} resources are missing")
            if (
                resources.get("device") != "cuda"
                or resources.get("gpus") != 1
                or resources.get("uv_environment") != ".venv-gpu"
            ):
                raise ValueError(
                    f"frozen {stage} GPU runtime contract differs"
                )
            assignment = (stage, run_id)
            if assignment in assignments:
                raise ValueError(
                    f"frozen runtime assignment is duplicated: {assignment}"
                )
            assignments.add(assignment)
    return assignments


def _verify_worker_runtime_profile(
    profile: Mapping[str, Any],
    *,
    context: str,
) -> list[str]:
    """Validate the immutable stable-worker profile schema and invariants."""

    expected_fields = {
        "device",
        "dtype",
        "python_version",
        "python_executable",
        "python_environment",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_devices",
    }
    if set(profile) != expected_fields:
        return [f"reference worker-runtime profile fields differ for {context}"]
    errors: list[str] = []
    if profile.get("device") != "cuda":
        errors.append(
            f"reference worker-runtime device is not cuda for {context}"
        )
    if profile.get("dtype") != "float64":
        errors.append(
            f"reference worker-runtime dtype is not float64 for {context}"
        )
    for key in (
        "python_version",
        "python_executable",
        "torch_version",
        "torch_cuda_version",
    ):
        if not isinstance(profile.get(key), str) or not profile[key]:
            errors.append(
                f"reference worker-runtime {key} is empty for {context}"
            )
    executable_value = profile.get("python_executable")
    executable = (
        Path(executable_value)
        if isinstance(executable_value, str)
        else Path()
    )
    if (
        not executable.is_absolute()
        or executable.parent.parent.name != ".venv-gpu"
        or profile.get("python_environment") != ".venv-gpu"
    ):
        errors.append(
            f"reference worker-runtime environment is not .venv-gpu "
            f"for {context}"
        )
    if (
        profile.get("cuda_available") is not True
        or profile.get("cuda_device_count") != 1
    ):
        errors.append(
            f"reference worker-runtime CUDA availability/count differs "
            f"for {context}"
        )
    devices = profile.get("cuda_devices")
    if not isinstance(devices, list) or len(devices) != 1:
        errors.append(
            f"reference worker-runtime device inventory differs for {context}"
        )
        return errors
    device = devices[0]
    required_device_fields = {
        "index",
        "name",
        "total_memory_bytes",
        "capability",
    }
    if not isinstance(device, Mapping) or set(device) != required_device_fields:
        errors.append(
            f"reference worker-runtime device fields differ for {context}"
        )
        return errors
    if device.get("index") != 0:
        errors.append(
            f"reference worker-runtime device index differs for {context}"
        )
    if not isinstance(device.get("name"), str) or not device["name"]:
        errors.append(
            f"reference worker-runtime device name is empty for {context}"
        )
    if (
        not isinstance(device.get("total_memory_bytes"), int)
        or device["total_memory_bytes"] <= 0
    ):
        errors.append(
            f"reference worker-runtime device memory is invalid for {context}"
        )
    if (
        not isinstance(device.get("capability"), str)
        or not device["capability"]
    ):
        errors.append(
            f"reference worker-runtime device capability is empty for {context}"
        )
    return errors


def _verify_evidence_input_receipt(
    value: object,
    *,
    artifacts: Sequence[ReferenceArtifact],
    attempts: object,
    source_results: object,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "file_count",
        "files",
        "checkpoint_contracts",
        "directory_projections",
        "directory_aggregate_sha256",
        "aggregate_sha256",
    }:
        return ["reference evidence-input receipt schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != EVIDENCE_INPUT_SCHEMA_VERSION:
        errors.append("reference evidence-input receipt version mismatch")
    rows = value.get("files")
    if not isinstance(rows, list):
        return [*errors, "reference evidence-input files are not a list"]
    if value.get("file_count") != len(rows):
        errors.append("reference evidence-input file count mismatch")
    paths: list[str] = []
    required = {
        "role",
        "source_path",
        "size",
        "mtime_ns",
        "ctime_ns",
        "mode",
        "device",
        "inode",
        "sha256",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            errors.append("reference evidence-input row fields mismatch")
            continue
        try:
            path = _safe_relative_text(row["source_path"])
            _digest_text(row["sha256"])
        except ValueError:
            errors.append("reference evidence-input row path/digest invalid")
            continue
        paths.append(path)
        if row.get("role") not in {
            "protected_artifact",
            "audit_evidence",
        }:
            errors.append(
                f"reference evidence-input role is invalid for {path}"
            )
        for key in (
            "size",
            "mtime_ns",
            "ctime_ns",
            "mode",
            "device",
            "inode",
        ):
            if not isinstance(row.get(key), int) or row[key] < 0:
                errors.append(
                    f"reference evidence-input {key} is invalid for {path}"
                )
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        errors.append("reference evidence-input paths are not sorted/unique")
    if value.get("aggregate_sha256") != canonical_sha256(rows):
        errors.append("reference evidence-input aggregate digest mismatch")
    directory_projections = value.get("directory_projections")
    try:
        observed_directories = _validate_directory_projections(
            directory_projections
        )
    except ValueError as exc:
        errors.append(f"reference directory projections are invalid: {exc}")
        observed_directories = []
    if value.get("directory_aggregate_sha256") != canonical_sha256(
        directory_projections
    ):
        errors.append("reference directory-projection digest mismatch")
    try:
        expected_roles = _derive_evidence_input_paths(
            artifacts,
            attempts=attempts,
            source_results=source_results,
            checkpoint_contracts=value.get("checkpoint_contracts"),
            receipt_digests={
                str(row.get("source_path")): str(row.get("sha256"))
                for row in rows
                if isinstance(row, Mapping)
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"cannot derive reference evidence-input paths: {exc}")
    else:
        observed_roles = {
            str(row.get("source_path")): str(row.get("role"))
            for row in rows
            if isinstance(row, Mapping)
        }
        if observed_roles != expected_roles:
            errors.append(
                "reference evidence-input path/role population differs from "
                "frozen task/source contracts"
            )
    try:
        expected_directories = _derive_directory_projections(
            artifacts,
            attempts=attempts,
            source_results=source_results,
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(
            f"cannot derive reference directory projections: {exc}"
        )
    else:
        if observed_directories != expected_directories:
            errors.append(
                "reference directory projections differ from frozen "
                "manifest/task/config contracts"
            )
    return errors


def _validate_directory_projections(
    value: object,
) -> list[dict[str, Any]]:
    """Validate canonical entry/type snapshots without trusting producer code."""

    if not isinstance(value, list):
        raise ValueError("directory projections are not a list")
    required = {"role", "source_path", "entries", "entries_sha256"}
    allowed_roles = {
        "grid_jobs",
        "final_grid_jobs",
        "02_validation_diagnostic_output",
        "07_final_eval_diagnostic_output",
    }
    result: list[dict[str, Any]] = []
    paths: list[str] = []
    for projection in value:
        if not isinstance(projection, dict) or set(projection) != required:
            raise ValueError("directory projection fields mismatch")
        role = str(projection.get("role") or "")
        if role not in allowed_roles:
            raise ValueError(f"unknown directory projection role {role!r}")
        source_path = _safe_relative_text(projection.get("source_path"))
        raw_entries = projection.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("directory projection entries are not a list")
        entries: list[dict[str, str]] = []
        entry_paths: list[str] = []
        for entry in raw_entries:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "type"}
            ):
                raise ValueError("directory projection entry fields mismatch")
            path = _safe_relative_text(entry.get("path"))
            entry_type = str(entry.get("type") or "")
            if entry_type not in {"file", "directory"}:
                raise ValueError("directory projection entry type is invalid")
            entries.append({"path": path, "type": entry_type})
            entry_paths.append(path)
        if (
            entry_paths != sorted(entry_paths)
            or len(set(entry_paths)) != len(entry_paths)
        ):
            raise ValueError(
                "directory projection entries are not sorted/unique"
            )
        if projection.get("entries_sha256") != canonical_sha256(entries):
            raise ValueError("directory projection entry digest mismatch")
        paths.append(source_path)
        result.append(
            {
                "role": role,
                "source_path": source_path,
                "entries": entries,
                "entries_sha256": canonical_sha256(entries),
            }
        )
    expected_order = sorted(
        result,
        key=lambda row: (str(row["source_path"]), str(row["role"])),
    )
    if result != expected_order or len(set(paths)) != len(paths):
        raise ValueError("directory projections are not sorted/unique")
    return result


def _derive_directory_projections(
    artifacts: Sequence[ReferenceArtifact],
    *,
    attempts: object,
    source_results: object,
) -> list[dict[str, Any]]:
    """Reconstruct exact directory entry contracts from frozen inputs."""

    normalized = _normalize_attempts(
        attempts if isinstance(attempts, Mapping) else {}
    )
    if not isinstance(source_results, Mapping):
        raise ValueError("source-results identity is not an object")
    source_root_value = source_results.get("canonical_root")
    if (
        not isinstance(source_root_value, str)
        or not Path(source_root_value).is_absolute()
    ):
        raise ValueError("source-results canonical root is invalid")
    source_root = Path(source_root_value)
    by_logical = {
        artifact.logical_path: artifact for artifact in artifacts
    }
    result: list[dict[str, Any]] = []

    grid = _artifact_json(by_logical, "00_grid/{grid}/manifest.json")
    raw_jobs = grid.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("frozen grid jobs are not a list")
    grid_names = [
        f"{_safe_component(row.get('run_id'), 'grid run_id')}.json"
        for row in raw_jobs
        if isinstance(row, Mapping)
    ]
    if len(grid_names) != len(raw_jobs):
        raise ValueError("frozen grid job row is not an object")
    result.append(
        _expected_directory_projection(
            role="grid_jobs",
            source_path=(
                Path("00_grid") / normalized["grid"] / "jobs"
            ).as_posix(),
            file_paths=grid_names,
        )
    )

    final_job_prefix = (
        Path("05_final_grid") / normalized["final_grid"] / "jobs"
    )
    final_job_names = sorted(
        Path(artifact.source_path).name
        for artifact in artifacts
        if Path(artifact.source_path).parent == final_job_prefix
    )
    if not final_job_names:
        raise ValueError("frozen final-grid job population is empty")
    result.append(
        _expected_directory_projection(
            role="final_grid_jobs",
            source_path=final_job_prefix.as_posix(),
            file_paths=final_job_names,
        )
    )

    validation_config = _artifact_yaml(
        by_logical,
        "00_grid/{grid}/validation_config.yaml",
    )
    suite_files = {
        "02_validation": _configured_diagnostic_files(
            validation_config,
            suite="validation",
        ),
        "07_final_eval": _configured_diagnostic_files(
            validation_config,
            suite="final_eval",
        ),
    }
    for stage, attempt_key in (
        ("02_validation", "validation"),
        ("07_final_eval", "final_eval"),
    ):
        tasks = _artifact_jsonl(
            by_logical,
            f"{stage}/stage_plans/{{{attempt_key}}}/tasks.jsonl",
        )
        for task in tasks:
            run_id = _safe_component(
                task.get("run_id"),
                f"frozen {stage} diagnostic run_id",
            )
            result_dir = _relative_to_source_root(
                task.get("result_dir"),
                source_root=source_root,
                context=f"frozen {stage} diagnostic result_dir",
            )
            expected_result = (
                Path(stage) / run_id / normalized[attempt_key]
            )
            if result_dir != expected_result:
                raise ValueError(
                    f"frozen {stage} diagnostic result_dir differs"
                )
            for task_name, filenames in suite_files[stage].items():
                result.append(
                    _expected_directory_projection(
                        role=f"{stage}_diagnostic_output",
                        source_path=(result_dir / task_name).as_posix(),
                        file_paths=filenames,
                    )
                )
    return sorted(
        result,
        key=lambda row: (str(row["source_path"]), str(row["role"])),
    )


def _expected_directory_projection(
    *,
    role: str,
    source_path: str,
    file_paths: Sequence[str],
) -> dict[str, Any]:
    entries_by_path: dict[str, str] = {}
    for value in file_paths:
        file_path = _safe_nested_path(value)
        parents = list(file_path.parents)
        for parent in reversed(parents[:-1]):
            entries_by_path[parent.as_posix()] = "directory"
        if (
            file_path.as_posix() in entries_by_path
            and entries_by_path[file_path.as_posix()] != "file"
        ):
            raise ValueError("diagnostic artifact file/directory collision")
        entries_by_path[file_path.as_posix()] = "file"
    entries = [
        {"path": path, "type": entry_type}
        for path, entry_type in sorted(entries_by_path.items())
    ]
    return {
        "role": role,
        "source_path": _safe_relative_text(source_path),
        "entries": entries,
        "entries_sha256": canonical_sha256(entries),
    }


def _configured_diagnostic_files(
    config: Mapping[str, Any],
    *,
    suite: str,
) -> dict[str, tuple[str, ...]]:
    """Derive output filenames from one frozen suite and writer policy."""

    suites = config.get("evaluation_suites")
    specifications = config.get("evaluation_tasks")
    if not isinstance(suites, Mapping) or not isinstance(
        specifications,
        Mapping,
    ):
        raise ValueError("validation diagnostic contracts are missing")
    suite_spec = suites.get(suite)
    if not isinstance(suite_spec, Mapping):
        raise ValueError(f"evaluation suite {suite!r} is invalid")
    references = suite_spec.get("tasks")
    artifact_level = suite_spec.get("artifact_level")
    if not isinstance(references, list) or artifact_level not in {
        "summaries",
        "records",
    }:
        raise ValueError(f"evaluation suite {suite!r} artifact policy invalid")
    result: dict[str, tuple[str, ...]] = {}
    for reference in references:
        text = str(reference)
        prefix = "${evaluation_tasks."
        if not text.startswith(prefix) or not text.endswith("}"):
            raise ValueError(
                f"evaluation task reference is not explicit: {text}"
            )
        key = text[len(prefix) : -1]
        task = specifications.get(key)
        if not isinstance(task, Mapping):
            raise ValueError(f"evaluation task {key!r} is missing")
        name = _safe_component(
            task.get("name"),
            f"evaluation task {key!r} name",
        )
        if name in result:
            raise ValueError(f"evaluation suite {suite!r} duplicates {name}")
        filenames: list[str] = []
        summaries = task.get("summaries")
        if not isinstance(summaries, list):
            raise ValueError(f"evaluation task {key!r} summaries invalid")
        if artifact_level == "records":
            for summary in summaries:
                if not isinstance(summary, Mapping):
                    raise ValueError(
                        f"evaluation task {key!r} summary is invalid"
                    )
                target = str(summary.get("_target_") or "")
                filename: object | None = None
                if target.endswith(".SampledRecordWriter"):
                    filename = summary.get("filename")
                    if filename is None:
                        raise ValueError(
                            f"evaluation task {key!r} sampled writer "
                            "has no explicit filename"
                        )
                elif target.endswith(".TransformRecordWriter"):
                    filename = summary.get(
                        "filename",
                        "transform_records.csv",
                    )
                elif target.endswith(".TraceRecordWriter"):
                    filename = summary.get(
                        "filename",
                        "trace_records.csv",
                    )
                elif target.endswith("RecordWriter"):
                    raise ValueError(
                        f"evaluation task {key!r} has unknown record writer"
                    )
                if filename is not None:
                    filenames.append(_safe_relative_text(filename))
        if len(set(filenames)) != len(filenames):
            raise ValueError(
                f"evaluation task {key!r} artifact filenames duplicate"
            )
        result[name] = tuple(sorted(filenames))
    return result


def _derive_evidence_input_paths(
    artifacts: Sequence[ReferenceArtifact],
    *,
    attempts: object,
    source_results: object,
    checkpoint_contracts: object,
    receipt_digests: Mapping[str, str],
) -> dict[str, str]:
    """Derive every raw audit input from frozen plans and source identity."""

    normalized = _normalize_attempts(
        attempts if isinstance(attempts, Mapping) else {}
    )
    if not isinstance(source_results, Mapping):
        raise ValueError("source-results identity is not an object")
    source_root_value = source_results.get("canonical_root")
    if (
        not isinstance(source_root_value, str)
        or not Path(source_root_value).is_absolute()
    ):
        raise ValueError("source-results canonical root is invalid")
    source_root = Path(source_root_value)
    by_logical = {
        artifact.logical_path: artifact for artifact in artifacts
    }
    expected: set[str] = set()
    checkpoint_tasks: dict[str, str] = {}
    for stage, attempt_key in FANOUT_ATTEMPTS.items():
        tasks = _artifact_jsonl(
            by_logical,
            f"{stage}/stage_plans/{{{attempt_key}}}/tasks.jsonl",
        )
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            run_id = _safe_component(
                task.get("run_id"),
                f"frozen {stage} evidence run_id",
            )
            attempt = normalized[attempt_key]
            expected_result = Path(stage) / run_id / attempt
            result = _relative_to_source_root(
                task.get("result_dir"),
                source_root=source_root,
                context=f"frozen {stage} result_dir",
            )
            if result != expected_result:
                raise ValueError(
                    f"frozen {stage} result_dir differs from task identity"
                )
            expected.update(
                (result / name).as_posix()
                for name in (
                    "status.json",
                    "launcher_status.json",
                    "metrics.jsonl",
                    "metadata.json",
                    "run_start.json",
                )
            )
            if stage in {"01_train", "02_validation"}:
                expected.add((result / "source_grid_attempt.json").as_posix())
            else:
                expected.update(
                    (result / name).as_posix()
                    for name in (
                        "source_final_grid_attempt.json",
                        "source_final_job.json",
                        "source_champion.json",
                    )
                )
            if stage == "02_validation":
                expected.add(
                    (result / "source_train_attempt.json").as_posix()
                )
            if stage in {"02_validation", "07_final_eval"}:
                expected.add(
                    (result / "diagnostics" / "index.json").as_posix()
                )
            if stage == "06_final_train":
                expected.add(
                    (result / "selected_checkpoint.json").as_posix()
                )
            if stage == "07_final_eval":
                expected.update(
                    (result / name).as_posix()
                    for name in (
                        "source_final_train_attempt.json",
                        "evaluated_checkpoint.json",
                    )
                )

            completion = task.get("completion")
            if not isinstance(completion, Mapping):
                raise ValueError(f"frozen {stage} completion is invalid")
            status_path = _relative_to_source_root(
                completion.get("status_path"),
                source_root=source_root,
                context=f"frozen {stage} status_path",
            )
            if status_path != result / "status.json":
                raise ValueError(
                    f"frozen {stage} status path differs from task identity"
                )
            logs = task.get("logs")
            if not isinstance(logs, list) or len(logs) != 1:
                raise ValueError(f"frozen {stage} launcher logs are invalid")
            launcher = _relative_to_source_root(
                logs[0],
                source_root=source_root,
                context=f"frozen {stage} launcher log",
            )
            if launcher != result / "launcher_status.json":
                raise ValueError(
                    f"frozen {stage} launcher path differs from task identity"
                )
            if (
                completion.get("policy")
                == "status_completed_with_checkpoint"
            ):
                pointer = _relative_to_source_root(
                    completion.get("checkpoint_path"),
                    source_root=source_root,
                    context=f"frozen {stage} checkpoint pointer",
                )
                expected_pointer = result / "checkpoints" / "latest.json"
                if pointer != expected_pointer:
                    raise ValueError(
                        f"frozen {stage} checkpoint differs from task identity"
                    )
                if not task_id or task_id in checkpoint_tasks:
                    raise ValueError("frozen checkpoint task ids are invalid")
                checkpoint_tasks[task_id] = pointer.as_posix()

    report = _artifact_json(
        by_logical,
        "09_final_report/{report}/final_report.json",
    )
    figures = report.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError("frozen final report figure contract is empty")
    expected.update(
        (
            Path("09_final_report")
            / normalized["report"]
            / "figures"
            / _safe_nested_path(name)
        ).as_posix()
        for name in figures
    )
    expected.update(
        _verify_checkpoint_evidence_contracts(
            checkpoint_contracts,
            checkpoint_tasks=checkpoint_tasks,
            receipt_digests=receipt_digests,
        )
    )
    protected = {
        artifact.source_path: "protected_artifact"
        for artifact in artifacts
    }
    overlap = set(protected) & expected
    if overlap:
        raise ValueError(
            f"protected/audit evidence roles overlap: {sorted(overlap)!r}"
        )
    return {
        **protected,
        **{path: "audit_evidence" for path in expected},
    }


def _verify_checkpoint_evidence_contracts(
    value: object,
    *,
    checkpoint_tasks: Mapping[str, str],
    receipt_digests: Mapping[str, str],
) -> set[str]:
    if not isinstance(value, list):
        raise ValueError("checkpoint evidence contracts are not a list")
    expected_paths: set[str] = set()
    observed_tasks: set[str] = set()
    required = {
        "schema_version",
        "task_id",
        "pointer_path",
        "pointer_projection",
        "step",
        "checkpoint_dir",
        "manifest_projection",
        "payload_paths",
        "file_sha256",
    }
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("checkpoint evidence contract fields mismatch")
        if row.get("schema_version") != CHECKPOINT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("checkpoint evidence contract version mismatch")
        task_id = str(row.get("task_id") or "")
        if task_id in observed_tasks or task_id not in checkpoint_tasks:
            raise ValueError("checkpoint evidence task identity is invalid")
        observed_tasks.add(task_id)
        pointer = _safe_relative_text(row.get("pointer_path"))
        if pointer != checkpoint_tasks[task_id]:
            raise ValueError("checkpoint evidence pointer differs from task")
        pointer_projection = row.get("pointer_projection")
        if not isinstance(pointer_projection, Mapping):
            raise ValueError("checkpoint pointer projection is invalid")
        step = row.get("step")
        if not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint evidence step is invalid")
        if (
            pointer_projection.get("step") != step
            or pointer_projection.get("checkpoint_dir")
            != f"step_{step:06d}"
        ):
            raise ValueError(
                "checkpoint pointer projection differs from identity"
            )
        checkpoint_dir = _safe_relative_text(row.get("checkpoint_dir"))
        expected_dir = (
            Path(pointer).parent / f"step_{step:06d}"
        ).as_posix()
        if checkpoint_dir != expected_dir:
            raise ValueError(
                "checkpoint evidence concrete directory differs from step"
            )
        manifest_projection = row.get("manifest_projection")
        if not isinstance(manifest_projection, Mapping):
            raise ValueError("checkpoint manifest projection is invalid")
        manifest_files = manifest_projection.get("files")
        if (
            manifest_projection.get("step") != step
            or not isinstance(manifest_files, Mapping)
            or not manifest_files
        ):
            raise ValueError(
                "checkpoint manifest projection differs from identity"
            )
        projected_payloads = sorted(
            (
                Path(checkpoint_dir)
                / _safe_nested_path(path)
            ).as_posix()
            for path in manifest_files.values()
        )
        payloads = row.get("payload_paths")
        if (
            not isinstance(payloads, list)
            or not payloads
            or payloads != sorted(set(payloads))
            or payloads != projected_payloads
        ):
            raise ValueError("checkpoint evidence payload paths are invalid")
        concrete = Path(checkpoint_dir)
        for payload_value in payloads:
            payload = Path(_safe_relative_text(payload_value))
            if concrete not in payload.parents:
                raise ValueError("checkpoint evidence payload escapes checkpoint")
            expected_paths.add(payload.as_posix())
        expected_paths.update(
            {
                pointer,
                (concrete / "COMPLETE").as_posix(),
                (concrete / "manifest.json").as_posix(),
            }
        )
        contract_digests = row.get("file_sha256")
        checkpoint_paths = {
            pointer,
            (concrete / "COMPLETE").as_posix(),
            (concrete / "manifest.json").as_posix(),
            *payloads,
        }
        if (
            not isinstance(contract_digests, Mapping)
            or set(contract_digests) != checkpoint_paths
        ):
            raise ValueError("checkpoint evidence digest population differs")
        for path, digest in contract_digests.items():
            _digest_text(digest)
            if receipt_digests.get(path) != digest:
                raise ValueError(
                    "checkpoint evidence digest differs from read-set receipt"
                )
    if observed_tasks != set(checkpoint_tasks):
        raise ValueError(
            "checkpoint evidence population differs from frozen tasks"
        )
    return expected_paths


def _relative_to_source_root(
    value: object,
    *,
    source_root: Path,
    context: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} is empty")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} is not a safe absolute path")
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes source results") from exc
    _safe_relative_text(relative.as_posix())
    return relative
