"""Fail-closed V4-owned control-evidence contracts for V4-0 acceptance.

Scientific artifacts prove a route produced expected data.  This module proves
that one guarded V4 controller invoked exactly one pinned legacy subprocess
route for every V4-0 role and left the legacy source/results trees unchanged.
It deliberately models only V4-0's no-retry ten-stage stack.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from roots import PURPOSE_EXPERIMENT, require_beneath_root, require_v4_root, root_metadata
from routes import (
    REPO_ROOT,
    ROLE_TO_STAGE,
    STUDY_DIR,
    config_source_receipt,
    legacy_source_receipt,
    load_routes,
    render_legacy_argv,
    runtime_source_receipt,
)
from strict_data import StrictDataError, load_json


CONTROL_SCHEMA_VERSION = "pair-stability-v4/control-closure/v1"
PRE_CLOSE_CONTROL_SCHEMA_VERSION = "pair-stability-v4/control-preclose/v1"
CONTROLLER_SCHEMA_VERSION = "pair-stability-v4/controller/v1"
LEGACY_INVENTORY_SCHEMA_VERSION = "pair-stability-v4/legacy-inventory/v2"
DISPATCH_MANIFEST_SCHEMA_VERSION = "pair-stability-v4/dispatch-manifest/v1"
CANONICAL_CONTROLLER_PARTITION = "sapphire"
CANONICAL_CONTROLLER_MAX_SECONDS = 3 * 24 * 60 * 60
_JOB_ID_PATTERN = re.compile(r"^[0-9]+$")


def write_controller_request(
    root: Path,
    *,
    lineage_id: str,
    partition: str,
    walltime: str,
    cpus: int,
    mem_per_cpu_gb: int,
) -> Path:
    """Record immutable controller intent before submitting Slurm work."""

    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    profile = _controller_profile(
        partition=partition,
        walltime=walltime,
        cpus=cpus,
        mem_per_cpu_gb=mem_per_cpu_gb,
    )
    destination = _stack_path(root, lineage_id, "controller-request.json")
    _write_new_json(
        destination,
        {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "phase": "request",
            "lineage_id": lineage_id,
            "results_root": str(root),
            "root_metadata": root_metadata(root),
            "controller": profile,
            "legacy_source": legacy_source_receipt(REPO_ROOT),
            "runtime_source": runtime_source_receipt(REPO_ROOT),
            "config_source": config_source_receipt(REPO_ROOT),
            "created_at": _timestamp(),
        },
    )
    return destination


def write_controller_submission(
    root: Path,
    *,
    lineage_id: str,
    job_id: str,
) -> Path:
    """Record one controller job identity after successful ``sbatch``."""

    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    request = _load_stack_record(root, lineage_id, "controller-request.json")
    errors = _validate_controller_request(request, root=root, lineage_id=lineage_id)
    if errors:
        raise ValueError("invalid controller request: " + "; ".join(errors))
    job_id = _validate_job_id(job_id, "controller job id")
    destination = _stack_path(root, lineage_id, "controller-submission.json")
    _write_new_json(
        destination,
        {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "phase": "submission",
            "lineage_id": lineage_id,
            "results_root": str(root),
            "request_sha256": _sha256_file(
                _stack_path(root, lineage_id, "controller-request.json")
            ),
            "controller_job_id": job_id,
            "submitted_at": _timestamp(),
        },
    )
    return destination


def write_dispatch_manifest(root: Path, *, lineage_id: str) -> Path:
    """Freeze exact terminal dispatch-result population after controller work."""

    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    dispatch_root = require_beneath_root(
        root / "_v4" / "dispatch" / lineage_id,
        root,
    )
    results: list[str] = []
    if dispatch_root.exists():
        for path in sorted(dispatch_root.rglob("result.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unsafe dispatch result entry: {path}")
            results.append(path.relative_to(root).as_posix())
    destination = _stack_path(root, lineage_id, "dispatch-results.json")
    _write_new_json(
        destination,
        {
            "schema_version": DISPATCH_MANIFEST_SCHEMA_VERSION,
            "lineage_id": lineage_id,
            "results_root": str(root),
            "result_paths": results,
            "created_at": _timestamp(),
        },
    )
    return destination


def write_controller_result(
    root: Path,
    *,
    lineage_id: str,
    stage: str,
    stage_exit_code: int,
    exit_code: int,
    finalization_errors: Sequence[str],
    worker_job_id: str,
    worker_partition: str,
    effective_cpus_per_task: str,
    effective_mem_per_cpu_mb: str,
    effective_time_limit: str,
) -> Path:
    """Record one terminal controller state after all finalization attempts.

    A malformed effective Slurm allocation is itself terminal evidence.  Write
    an immutable ``incomplete`` result first, then raise so the controller
    exits nonzero without erasing why its finalization contract failed.
    """

    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    request_path = _stack_path(root, lineage_id, "controller-request.json")
    submission_path = _stack_path(root, lineage_id, "controller-submission.json")
    request, request_load_error = _load_terminal_record(request_path)
    submission, submission_load_error = _load_terminal_record(submission_path)
    terminal_errors: list[str] = []
    if request_load_error is not None:
        terminal_errors.append(f"controller request unavailable: {request_load_error}")
    else:
        terminal_errors.extend(
            _validate_controller_request(request, root=root, lineage_id=lineage_id)
        )
    if submission_load_error is not None:
        terminal_errors.append(
            f"controller submission unavailable: {submission_load_error}"
        )
    else:
        terminal_errors.extend(
            _validate_controller_submission(
                submission,
                root=root,
                lineage_id=lineage_id,
                request_path=request_path,
            )
        )
    try:
        _validate_job_id(worker_job_id, "worker controller job id")
    except ValueError as exc:
        terminal_errors.append(str(exc))
    if submission.get("controller_job_id") != worker_job_id:
        terminal_errors.append("worker controller job id differs from submission")
    if worker_partition != CANONICAL_CONTROLLER_PARTITION:
        terminal_errors.append("worker controller partition is not canonical Sapphire")
    effective = {
        "partition": _record_text(worker_partition),
        "cpus_per_task": _record_text(effective_cpus_per_task),
        "mem_per_cpu_mb": _record_text(effective_mem_per_cpu_mb),
        "time_limit": _record_text(effective_time_limit),
    }
    terminal_errors.extend(
        _validate_effective_controller(effective, request.get("controller"))
    )
    if type(stage_exit_code) is not int or stage_exit_code < 0:
        terminal_errors.append("controller stage exit code is invalid")
    if type(exit_code) is not int or exit_code < 0:
        terminal_errors.append("controller exit code is invalid")
    if not isinstance(stage, str) or not stage:
        terminal_errors.append("controller terminal stage is invalid")
    normalized_errors: list[str] = []
    for item in finalization_errors:
        if not isinstance(item, str) or not item:
            terminal_errors.append("controller finalization error is invalid")
        else:
            normalized_errors.append(item)
    normalized_errors.extend(terminal_errors)
    if normalized_errors:
        status = "incomplete"
    elif stage == "complete" and stage_exit_code == 0 and exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    destination = _stack_path(root, lineage_id, "controller-result.json")
    _write_new_json(
        destination,
        {
            "schema_version": CONTROLLER_SCHEMA_VERSION,
            "phase": "result",
            "lineage_id": lineage_id,
            "results_root": str(root),
            "request_sha256": _safe_file_sha256(request_path),
            "submission_sha256": _safe_file_sha256(submission_path),
            "controller_job_id": _record_text(worker_job_id),
            "worker_partition": _record_text(worker_partition),
            "effective_controller": effective,
            "stage": stage,
            "stage_exit_code": stage_exit_code,
            "exit_code": exit_code,
            "status": status,
            "finalization_errors": normalized_errors,
            "completed_at": _timestamp(),
        },
    )
    if terminal_errors:
        raise ValueError(
            "controller terminal evidence is incomplete: "
            + "; ".join(terminal_errors)
        )
    return destination


def audit_control_closure(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[str, ...]:
    """Return every reason a V4 candidate lacks complete control evidence."""

    errors: list[str] = []
    try:
        lineage_id = _single_lineage(attempts)
        root = require_v4_root(
            root,
            lineage_id=lineage_id,
            purpose=PURPOSE_EXPERIMENT,
        )
    except (OSError, ValueError) as exc:
        return (str(exc),)
    try:
        projection = _collect_control_projection(
            root,
            lineage_id=lineage_id,
            require_controller_result=True,
        )
    except (OSError, StrictDataError, ValueError) as exc:
        return (f"invalid control evidence: {exc}",)
    errors.extend(projection["errors"])
    return tuple(dict.fromkeys(errors))


def control_provenance(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> dict[str, object]:
    """Return verified compact control/closure facts for comparison provenance."""

    lineage_id = _single_lineage(attempts)
    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    projection = _collect_control_projection(
        root,
        lineage_id=lineage_id,
        require_controller_result=True,
    )
    errors = tuple(dict.fromkeys(projection["errors"]))
    if errors:
        raise ValueError("control evidence failed: " + "; ".join(errors))
    return {
        key: value
        for key, value in projection.items()
        if key != "errors"
    }


def audit_preclose_control(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate control evidence available before ``controller-result.json``.

    This deliberately does not infer controller success.  It is the public
    pre-close seam used by V4-only finalizers that must complete before the
    immutable controller result records a finalization failure.
    """

    try:
        lineage_id = _single_lineage(attempts)
        root = require_v4_root(
            root,
            lineage_id=lineage_id,
            purpose=PURPOSE_EXPERIMENT,
        )
    except (OSError, ValueError) as exc:
        return (str(exc),)
    try:
        projection = _collect_control_projection(
            root,
            lineage_id=lineage_id,
            require_controller_result=False,
        )
    except (OSError, StrictDataError, ValueError) as exc:
        return (f"invalid pre-close control evidence: {exc}",)
    return tuple(dict.fromkeys(projection["errors"]))


def preclose_control_provenance(
    root: Path,
    *,
    attempts: Mapping[str, str],
) -> dict[str, object]:
    """Return verified pre-close facts without fabricating terminal success."""

    lineage_id = _single_lineage(attempts)
    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    projection = _collect_control_projection(
        root,
        lineage_id=lineage_id,
        require_controller_result=False,
    )
    errors = tuple(dict.fromkeys(projection["errors"]))
    if errors:
        raise ValueError("pre-close control evidence failed: " + "; ".join(errors))
    return {key: value for key, value in projection.items() if key != "errors"}


def _collect_control_projection(
    root: Path,
    *,
    lineage_id: str,
    require_controller_result: bool,
) -> dict[str, Any]:
    """Validate records and build one equality-safe closure projection."""

    errors: list[str] = []
    stack = _stack_directory(root, lineage_id)
    required_stack = {
        "controller-request.json",
        "controller-submission.json",
        "dispatch-results.json",
        "legacy-pre.json",
        "legacy-post.json",
    }
    if require_controller_result:
        required_stack.add("controller-result.json")
    actual_stack = {
        path.name
        for path in stack.iterdir()
        if path.is_file() or path.is_symlink()
    }
    missing = required_stack - actual_stack
    if missing:
        errors.append(f"missing control records: {sorted(missing)}")
    request: dict[str, Any] = {}
    submission: dict[str, Any] = {}
    result: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    pre: dict[str, Any] = {}
    post: dict[str, Any] = {}
    records_to_load = (
        ("controller-request.json", request),
        ("controller-submission.json", submission),
        ("dispatch-results.json", manifest),
        ("legacy-pre.json", pre),
        ("legacy-post.json", post),
    )
    if require_controller_result:
        records_to_load += (("controller-result.json", result),)
    for name, sink in records_to_load:
        path = _stack_path(root, lineage_id, name)
        if not path.is_file() or path.is_symlink():
            continue
        try:
            sink.update(_load_json_object(path))
        except (OSError, StrictDataError, ValueError) as exc:
            errors.append(f"invalid {name}: {exc}")
    if request:
        errors.extend(_validate_controller_request(request, root=root, lineage_id=lineage_id))
    if submission:
        errors.extend(
            _validate_controller_submission(
                submission,
                root=root,
                lineage_id=lineage_id,
                request_path=_stack_path(root, lineage_id, "controller-request.json"),
            )
        )
    if require_controller_result and result:
        errors.extend(
            _validate_controller_result(
                result,
                root=root,
                lineage_id=lineage_id,
                request_path=_stack_path(root, lineage_id, "controller-request.json"),
                submission_path=_stack_path(root, lineage_id, "controller-submission.json"),
                request=request,
                submission=submission,
            )
        )
    errors.extend(
        _validate_legacy_inventories(
            pre,
            post,
            root=root,
            lineage_id=lineage_id,
            controller_request=request,
        )
    )

    dispatch_projection, dispatch_errors = _validate_dispatch_records(
        root,
        lineage_id=lineage_id,
        request=request,
        manifest=manifest,
    )
    errors.extend(dispatch_errors)
    if request and dispatch_projection:
        errors.extend(_validate_closure_projection(request, dispatch_projection))

    record_paths = [
        _stack_path(root, lineage_id, name)
        for name in sorted(required_stack)
        if _stack_path(root, lineage_id, name).is_file()
    ]
    dispatch_paths = dispatch_projection.get("receipt_paths", [])
    record_paths.extend(root / path for path in dispatch_paths)
    record_digests = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(set(record_paths))
        if path.is_file() and not path.is_symlink()
    ]
    closures = dispatch_projection.get("closures", {})
    return {
        "errors": errors,
        "schema_version": (
            CONTROL_SCHEMA_VERSION
            if require_controller_result
            else PRE_CLOSE_CONTROL_SCHEMA_VERSION
        ),
        "lineage_id": lineage_id,
        "root": str(root),
        "controller": request.get("controller", {}),
        "effective_controller": result.get("effective_controller", {}),
        "controller_status": result.get("status"),
        "closure_projection": closures,
        "dispatch_runtime_profiles": dispatch_projection.get("runtime_profiles", {}),
        "legacy_inventory_sha256": _canonical_sha256(
            {"source": pre.get("source"), "results": pre.get("results")}
        ) if pre else None,
        "record_digests": record_digests,
        "verification_sha256": _canonical_sha256(record_digests),
    }


def _validate_dispatch_records(
    root: Path,
    *,
    lineage_id: str,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    routes = load_routes()
    dispatch_root = root / "_v4" / "dispatch" / lineage_id
    expected_roles = set(routes)
    actual_roles = (
        {path.name for path in dispatch_root.iterdir()}
        if dispatch_root.is_dir() and not dispatch_root.is_symlink()
        else set()
    )
    if actual_roles != expected_roles:
        errors.append(
            "dispatch role population mismatch; "
            f"missing={sorted(expected_roles - actual_roles)}, "
            f"extra={sorted(actual_roles - expected_roles)}"
        )
    actual_result_paths: list[str] = []
    actual_receipt_paths: list[str] = []
    runtime_by_kind: dict[str, list[dict[str, object]]] = {}
    closures: dict[str, object] = {}
    for role, route in routes.items():
        role_root = dispatch_root / role
        if not role_root.is_dir() or role_root.is_symlink():
            errors.append(f"missing dispatch receipt directory for {role}")
            continue
        invocations = sorted(role_root.iterdir())
        if len(invocations) != 1 or any(
            not item.is_dir() or item.is_symlink() for item in invocations
        ):
            errors.append(f"{role} must have exactly one immutable receipt pair")
            continue
        receipt = invocations[0]
        unexpected = {entry.name for entry in receipt.iterdir()} - {"request.json", "result.json"}
        if unexpected:
            errors.append(f"{role} receipt contains unexpected entries: {sorted(unexpected)}")
        request_path = receipt / "request.json"
        result_path = receipt / "result.json"
        if not request_path.is_file() or request_path.is_symlink() or not result_path.is_file() or result_path.is_symlink():
            errors.append(f"{role} request/result pair is incomplete")
            continue
        try:
            dispatch_request = _load_json_object(request_path)
            dispatch_result = _load_json_object(result_path)
        except (OSError, StrictDataError, ValueError) as exc:
            errors.append(f"{role} receipt is invalid: {exc}")
            continue
        errors.extend(
            _validate_dispatch_pair(
                dispatch_request,
                dispatch_result,
                root=root,
                lineage_id=lineage_id,
                role=role,
                physical_stage=route.physical_stage,
                route_kind=route.kind,
                controller_request=request,
                receipt_id=receipt.name,
            )
        )
        actual_result_paths.append(result_path.relative_to(root).as_posix())
        actual_receipt_paths.extend(
            (
                request_path.relative_to(root).as_posix(),
                result_path.relative_to(root).as_posix(),
            )
        )
        runtime = dispatch_request.get("runtime_source")
        if isinstance(runtime, dict):
            runtime_by_kind.setdefault(route.kind, []).append(_runtime_variant(runtime))
        if not closures:
            closures = _closure_projection(dispatch_request)
    expected_manifest = {
        "schema_version": DISPATCH_MANIFEST_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "results_root": str(root),
        "result_paths": sorted(actual_result_paths),
        "created_at": manifest.get("created_at"),
    }
    if not manifest:
        errors.append("dispatch result manifest is missing")
    elif manifest != expected_manifest or not _is_timestamp(manifest.get("created_at")):
        errors.append("dispatch result manifest does not bind exact receipt population")
    for kind, variants in runtime_by_kind.items():
        if len({ _canonical_sha256(item) for item in variants }) != 1:
            errors.append(f"dispatch runtime environment is mixed for {kind} routes")
    return (
        {
            "result_paths": sorted(actual_result_paths),
            "receipt_paths": sorted(actual_receipt_paths),
            "closures": closures,
            "runtime_profiles": {
                kind: values[0]
                for kind, values in sorted(runtime_by_kind.items())
                if values
            },
        },
        errors,
    )


def _validate_dispatch_pair(
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    root: Path,
    lineage_id: str,
    role: str,
    physical_stage: str,
    route_kind: str,
    controller_request: Mapping[str, Any],
    receipt_id: str,
) -> list[str]:
    errors: list[str] = []
    required_request = {
        "schema_version", "invocation_id", "logical_role", "physical_stage",
        "route_kind", "profile", "purpose", "results_root", "output_attempt",
        "input_attempts", "config_paths", "argv", "cwd", "started_at",
        "legacy_source", "runtime_source", "config_source", "pre_run_unsafe_links",
    }
    required_result = {
        "schema_version", "invocation_id", "logical_role", "status", "returncode",
        "signal", "error", "argv", "cwd", "legacy_source", "runtime_source",
        "config_source", "started_at", "completed_at", "post_run_unsafe_links",
    }
    if set(request) != required_request:
        errors.append(f"{role} request schema mismatch")
        return errors
    if set(result) != required_result:
        errors.append(f"{role} result schema mismatch")
        return errors
    expected_inputs = {key: lineage_id for key in load_routes()[role].required_input_attempts}
    for key, expected in {
        "schema_version": "pair-stability-v4/dispatch/v1",
        "logical_role": role,
        "physical_stage": physical_stage,
        "route_kind": route_kind,
        "profile": "canonical",
        "purpose": PURPOSE_EXPERIMENT,
        "results_root": str(root),
        "output_attempt": lineage_id,
        "input_attempts": expected_inputs,
        "cwd": str(REPO_ROOT),
        "pre_run_unsafe_links": [],
    }.items():
        if request.get(key) != expected:
            errors.append(f"{role} request {key} differs from canonical control identity")
    if request.get("invocation_id") != receipt_id:
        errors.append(f"{role} request invocation id differs from receipt directory")
    if not isinstance(request.get("invocation_id"), str) or not request["invocation_id"]:
        errors.append(f"{role} request invocation id is invalid")
    if not _is_timestamp(request.get("started_at")):
        errors.append(f"{role} request timestamp is invalid")
    for key in ("invocation_id", "logical_role", "argv", "cwd", "legacy_source", "runtime_source", "config_source", "started_at"):
        if result.get(key) != request.get(key):
            errors.append(f"{role} request/result {key} differs")
    if (
        result.get("status") != "completed"
        or type(result.get("returncode")) is not int
        or result.get("returncode") != 0
        or result.get("signal") is not None
        or result.get("error") is not None
    ):
        errors.append(f"{role} result is not terminal success")
    if not _is_timestamp(result.get("completed_at")):
        errors.append(f"{role} result timestamp is invalid")
    if result.get("post_run_unsafe_links") != []:
        errors.append(f"{role} result records unsafe V4 links")
    if not isinstance(request.get("argv"), list) or not all(isinstance(item, str) for item in request["argv"]):
        errors.append(f"{role} request argv is invalid")
    configs = request.get("config_paths")
    try:
        expected_configs = _expected_config_paths(
            root,
            role=role,
            lineage_id=lineage_id,
        )
    except (OSError, StrictDataError, ValueError) as exc:
        errors.append(f"{role} cannot derive canonical config paths: {exc}")
    else:
        expected_config_strings = {
            key: str(value) for key, value in sorted(expected_configs.items())
        }
        if configs != expected_config_strings:
            errors.append(f"{role} request config paths differ from canonical route")
        try:
            expected_argv = list(
                render_legacy_argv(
                    load_routes()[role],
                    results_root=root,
                    output_attempt=lineage_id,
                    input_attempts=expected_inputs,
                    config_paths=expected_configs,
                    repo_root=REPO_ROOT,
                )
            )
        except (OSError, ValueError) as exc:
            errors.append(f"{role} cannot render receipt argv: {exc}")
        else:
            if request.get("argv") != expected_argv:
                errors.append(f"{role} request argv differs from pinned route")
    errors.extend(
        _validate_source_receipts(
            request.get("legacy_source"),
            request.get("runtime_source"),
            request.get("config_source"),
            label=f"{role} request",
        )
    )
    if controller_request:
        for field in ("legacy_source", "config_source"):
            if request.get(field) != controller_request.get(field):
                errors.append(f"{role} {field} differs from controller request")
        controller_runtime = controller_request.get("runtime_source")
        request_runtime = request.get("runtime_source")
        if not isinstance(controller_runtime, dict) or not isinstance(request_runtime, dict):
            errors.append(f"{role} runtime source is malformed")
        elif {
            "schema_version": request_runtime.get("schema_version"),
            "closure_sha256": request_runtime.get("closure_sha256"),
            "dirty": request_runtime.get("dirty"),
        } != {
            "schema_version": controller_runtime.get("schema_version"),
            "closure_sha256": controller_runtime.get("closure_sha256"),
            "dirty": controller_runtime.get("dirty"),
        }:
            errors.append(f"{role} runtime closure differs from controller request")
    return errors


def _expected_config_paths(
    root: Path,
    *,
    role: str,
    lineage_id: str,
) -> dict[str, Path]:
    """Derive route config paths from guarded V4 artifacts, never receipts."""

    route = load_routes()[role]
    if not route.required_configs:
        return {}
    if role == "screen_plan":
        values = {
            "smoke": STUDY_DIR / "configs" / "smoke.yaml",
            "train": STUDY_DIR / "configs" / "pair_stability.yaml",
        }
    elif role in {"screen_eval", "confirm_plan"}:
        grid = require_beneath_root(root / "00_grid" / lineage_id, root)
        manifest = _load_json_object(grid / "manifest.json")
        snapshots = manifest.get("config_snapshots")
        if not isinstance(snapshots, dict):
            raise ValueError("grid config snapshots are missing")
        values = {}
        for key in ("train", "validation"):
            relative = snapshots.get(key)
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"grid {key} snapshot is invalid")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"grid {key} snapshot escapes grid directory")
            values[key] = require_beneath_root(grid / path, root)
    elif role in {"confirm_train", "confirm_eval"}:
        manifest = _load_json_object(
            require_beneath_root(
                root / "05_final_grid" / lineage_id / "manifest.json",
                root,
            )
        )
        values = {}
        for key, field in (("train", "train_config"), ("validation", "eval_config")):
            raw = manifest.get(field)
            if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
                raise ValueError(f"final-grid {field} is invalid")
            values[key] = require_beneath_root(Path(raw), root)
    else:
        raise ValueError(f"unexpected config-bearing V4 route: {role}")
    selected = {key: values[key] for key in route.required_configs}
    for key, path in selected.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"canonical {key} config is not a regular file")
    return selected


def _validate_closure_projection(
    controller_request: Mapping[str, Any],
    dispatch_projection: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    expected = _closure_projection(controller_request)
    actual = dispatch_projection.get("closures")
    if actual != expected:
        errors.append("dispatch closure projection differs from controller request")
    return errors


def _validate_legacy_inventories(
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
    *,
    root: Path,
    lineage_id: str,
    controller_request: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_fields = {
        "schema_version", "phase", "lineage_id", "captured_at", "source",
        "results", "legacy_source", "runtime_source", "config_source",
    }
    for phase, value in (("pre", pre), ("post", post)):
        if set(value) != expected_fields:
            errors.append(f"legacy-{phase} inventory schema mismatch")
            continue
        if value.get("schema_version") != LEGACY_INVENTORY_SCHEMA_VERSION:
            errors.append(f"legacy-{phase} inventory version mismatch")
        if value.get("phase") != phase or value.get("lineage_id") != lineage_id:
            errors.append(f"legacy-{phase} inventory lineage/phase mismatch")
        if not _is_timestamp(value.get("captured_at")):
            errors.append(f"legacy-{phase} inventory timestamp is invalid")
        if not isinstance(value.get("source"), list) or not value["source"]:
            errors.append(f"legacy-{phase} source inventory is empty or invalid")
        else:
            errors.extend(
                _validate_inventory_rows(
                    value["source"],
                    label=f"legacy-{phase} source",
                    require_file_hash=True,
                )
            )
        if not isinstance(value.get("results"), list):
            errors.append(f"legacy-{phase} results inventory is invalid")
        else:
            errors.extend(
                _validate_inventory_rows(
                    value["results"],
                    label=f"legacy-{phase} results",
                    require_file_hash=False,
                )
            )
        for field in ("legacy_source", "runtime_source", "config_source"):
            if value.get(field) != controller_request.get(field):
                errors.append(
                    f"legacy-{phase} inventory {field} differs from controller request"
                )
        errors.extend(
            _validate_source_receipts(
                value.get("legacy_source"),
                value.get("runtime_source"),
                value.get("config_source"),
                label=f"legacy-{phase} inventory",
            )
        )
        if isinstance(value.get("source"), list):
            errors.extend(
                _validate_source_inventory_binding(
                    value["source"],
                    controller_request.get("legacy_source"),
                    label=f"legacy-{phase}",
                )
            )
    if pre and post:
        if pre.get("source") != post.get("source"):
            errors.append("legacy source inventory changed during V4 candidate")
        if pre.get("results") != post.get("results"):
            errors.append("legacy results inventory changed during V4 candidate")
        for field in ("legacy_source", "runtime_source", "config_source"):
            if pre.get(field) != post.get(field):
                errors.append(f"legacy inventory {field} differs between pre/post")
    return errors


def _validate_source_inventory_binding(
    inventory: Sequence[object],
    legacy_receipt: object,
    *,
    label: str,
) -> list[str]:
    """Bind source-tree inventory rows to the recorded legacy closure facts."""

    if not isinstance(legacy_receipt, dict) or not isinstance(
        legacy_receipt.get("files"), list
    ):
        return [f"{label} source inventory cannot bind malformed legacy receipt"]
    expected: list[dict[str, str]] = []
    for row in legacy_receipt["files"]:
        if not isinstance(row, dict):
            return [f"{label} source inventory cannot bind malformed legacy receipt"]
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return [f"{label} source inventory cannot bind malformed legacy receipt"]
        expected.append({"path": path, "sha256": digest})
    actual: list[dict[str, str]] = []
    for row in inventory:
        if not isinstance(row, dict):
            return [f"{label} source inventory is malformed"]
        if row.get("type") != "file":
            return [f"{label} source inventory contains non-file entry"]
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return [f"{label} source inventory file binding is malformed"]
        actual.append({"path": path, "sha256": digest})
    if sorted(actual, key=lambda row: row["path"]) != sorted(
        expected,
        key=lambda row: row["path"],
    ):
        return [f"{label} source inventory differs from recorded legacy closure"]
    return []


def _validate_inventory_rows(
    rows: Sequence[object],
    *,
    label: str,
    require_file_hash: bool,
) -> list[str]:
    """Validate deterministic inventory receipts without rereading live trees."""

    errors: list[str] = []
    paths: list[str] = []
    for index, row in enumerate(rows):
        prefix = f"{label} inventory row {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix} is not an object")
            continue
        kind = row.get("type")
        base = {"path", "type", "size", "mtime_ns", "mode"}
        if kind == "file":
            expected = base | ({"sha256"} if require_file_hash else set())
        elif kind == "directory":
            expected = base
        elif kind == "symlink":
            expected = base | {"link_target"}
        else:
            errors.append(f"{prefix} type is invalid")
            continue
        if set(row) != expected:
            errors.append(f"{prefix} fields mismatch")
            continue
        path = row.get("path")
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or Path(path).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(path).parts)
        ):
            errors.append(f"{prefix} path is unsafe")
        else:
            paths.append(path)
        for field in ("size", "mtime_ns", "mode"):
            number = row.get(field)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                errors.append(f"{prefix} {field} is invalid")
        if kind == "file" and require_file_hash:
            digest = row.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                errors.append(f"{prefix} sha256 is invalid")
        if kind == "symlink":
            target = row.get("link_target")
            if not isinstance(target, str) or not target or "\x00" in target:
                errors.append(f"{prefix} link target is invalid")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        errors.append(f"{label} inventory paths are not sorted and unique")
    return errors


def _validate_controller_request(
    value: Mapping[str, Any],
    *,
    root: Path,
    lineage_id: str,
) -> list[str]:
    required = {
        "schema_version", "phase", "lineage_id", "results_root", "root_metadata",
        "controller", "legacy_source", "runtime_source", "config_source", "created_at",
    }
    errors: list[str] = []
    if set(value) != required:
        return ["controller request schema mismatch"]
    for key, expected in {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "phase": "request",
        "lineage_id": lineage_id,
        "results_root": str(root),
        "root_metadata": root_metadata(root),
    }.items():
        if value.get(key) != expected:
            errors.append(f"controller request {key} mismatch")
    try:
        _controller_profile(**dict(value.get("controller") or {}))
    except (TypeError, ValueError) as exc:
        errors.append(f"controller request profile invalid: {exc}")
    if not _is_timestamp(value.get("created_at")):
        errors.append("controller request timestamp is invalid")
    errors.extend(
        _validate_source_receipts(
            value.get("legacy_source"),
            value.get("runtime_source"),
            value.get("config_source"),
            label="controller request",
        )
    )
    return errors


def _validate_controller_submission(
    value: Mapping[str, Any],
    *,
    root: Path,
    lineage_id: str,
    request_path: Path,
) -> list[str]:
    required = {
        "schema_version", "phase", "lineage_id", "results_root", "request_sha256",
        "controller_job_id", "submitted_at",
    }
    if set(value) != required:
        return ["controller submission schema mismatch"]
    errors: list[str] = []
    for key, expected in {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "phase": "submission",
        "lineage_id": lineage_id,
        "results_root": str(root),
        "request_sha256": _sha256_file(request_path) if request_path.is_file() else None,
    }.items():
        if value.get(key) != expected:
            errors.append(f"controller submission {key} mismatch")
    try:
        _validate_job_id(value.get("controller_job_id"), "controller job id")
    except ValueError as exc:
        errors.append(str(exc))
    if not _is_timestamp(value.get("submitted_at")):
        errors.append("controller submission timestamp is invalid")
    return errors


def _validate_controller_result(
    value: Mapping[str, Any],
    *,
    root: Path,
    lineage_id: str,
    request_path: Path,
    submission_path: Path,
    request: Mapping[str, Any],
    submission: Mapping[str, Any],
) -> list[str]:
    required = {
        "schema_version", "phase", "lineage_id", "results_root", "request_sha256",
        "submission_sha256", "controller_job_id", "worker_partition", "effective_controller", "stage",
        "stage_exit_code", "exit_code", "status", "finalization_errors", "completed_at",
    }
    if set(value) != required:
        return ["controller result schema mismatch"]
    errors: list[str] = []
    expected = {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "phase": "result",
        "lineage_id": lineage_id,
        "results_root": str(root),
        "request_sha256": _sha256_file(request_path) if request_path.is_file() else None,
        "submission_sha256": _sha256_file(submission_path) if submission_path.is_file() else None,
        "controller_job_id": submission.get("controller_job_id"),
        "worker_partition": CANONICAL_CONTROLLER_PARTITION,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            errors.append(f"controller result {key} mismatch")
    try:
        _validate_job_id(value.get("controller_job_id"), "controller result job id")
    except ValueError as exc:
        errors.append(str(exc))
    if value.get("status") != "completed":
        errors.append("controller result is not terminal completed evidence")
    if (
        type(value.get("stage_exit_code")) is not int
        or type(value.get("exit_code")) is not int
        or value.get("stage_exit_code") != 0
        or value.get("exit_code") != 0
    ):
        errors.append("controller result records nonzero stage or controller exit")
    if value.get("finalization_errors") != []:
        errors.append("controller result records finalization failures")
    if not isinstance(value.get("stage"), str) or value.get("stage") != "complete":
        errors.append("controller result did not reach complete stage")
    if not _is_timestamp(value.get("completed_at")):
        errors.append("controller result timestamp is invalid")
    errors.extend(
        _validate_effective_controller(
            value.get("effective_controller"),
            request.get("controller"),
        )
    )
    return errors


def _validate_effective_controller(
    effective: object,
    requested: object,
) -> list[str]:
    """Validate Slurm's effective controller allocation against request facts."""

    if not isinstance(effective, dict) or set(effective) != {
        "partition", "cpus_per_task", "mem_per_cpu_mb", "time_limit"
    }:
        return ["controller result effective profile schema mismatch"]
    try:
        requested_profile = _controller_profile(**dict(requested or {}))
        if effective.get("partition") != CANONICAL_CONTROLLER_PARTITION:
            raise ValueError("partition is not Sapphire")
        cpus = _nonempty_text(
            effective.get("cpus_per_task"),
            "effective controller CPUs per task",
        )
        memory = _nonempty_text(
            effective.get("mem_per_cpu_mb"),
            "effective controller memory per CPU",
        )
        walltime = _nonempty_text(
            effective.get("time_limit"),
            "effective controller time limit",
        )
        if not cpus.isdecimal() or int(cpus) != requested_profile["cpus"]:
            raise ValueError("CPU allocation differs from requested profile")
        expected_memory = int(requested_profile["mem_per_cpu_gb"]) * 1024
        if not memory.isdecimal() or int(memory) != expected_memory:
            raise ValueError("memory allocation differs from requested profile")
        if _slurm_seconds(walltime) != _slurm_seconds(str(requested_profile["walltime"])):
            raise ValueError("time limit differs from requested profile")
    except (TypeError, ValueError) as exc:
        return [f"controller result effective profile is invalid: {exc}"]
    return []


def _validate_source_receipts(
    legacy: object,
    runtime: object,
    config: object,
    *,
    label: str,
) -> list[str]:
    """Validate recorded closure shapes without rereading a later checkout."""

    errors: list[str] = []
    errors.extend(_validate_legacy_receipt(legacy, label=label))
    errors.extend(_validate_runtime_receipt(runtime, label=label))
    errors.extend(_validate_config_receipt(config, label=label))
    return errors


def _validate_legacy_receipt(value: object, *, label: str) -> list[str]:
    required = {
        "schema_version", "manifest_path", "manifest_sha256", "closure_sha256", "files",
    }
    if not isinstance(value, dict) or set(value) != required:
        return [f"{label} legacy source schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/legacy-source/v1":
        errors.append(f"{label} legacy source version mismatch")
    manifest_path = value.get("manifest_path")
    if not _safe_repository_path(manifest_path):
        errors.append(f"{label} legacy source manifest path is invalid")
    for field in ("manifest_sha256", "closure_sha256"):
        if not _is_sha256(value.get(field)):
            errors.append(f"{label} legacy source {field} is invalid")
    files = value.get("files")
    file_errors, canonical_files = _validate_receipt_files(
        files,
        label=f"{label} legacy source",
    )
    errors.extend(file_errors)
    if canonical_files and value.get("closure_sha256") != _receipt_digest(canonical_files):
        errors.append(f"{label} legacy source closure digest mismatch")
    return errors


def _validate_config_receipt(value: object, *, label: str) -> list[str]:
    required = {"schema_version", "closure_sha256", "files"}
    if not isinstance(value, dict) or set(value) != required:
        return [f"{label} config source schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/config-source/v1":
        errors.append(f"{label} config source version mismatch")
    if not _is_sha256(value.get("closure_sha256")):
        errors.append(f"{label} config source closure_sha256 is invalid")
    file_errors, canonical_files = _validate_receipt_files(
        value.get("files"),
        label=f"{label} config source",
    )
    errors.extend(file_errors)
    if canonical_files and value.get("closure_sha256") != _receipt_digest(canonical_files):
        errors.append(f"{label} config source closure digest mismatch")
    return errors


def _validate_runtime_receipt(value: object, *, label: str) -> list[str]:
    required = {
        "schema_version", "closure_sha256", "n_files", "git_commit", "git_branch",
        "dirty", "python_executable", "python_version", "uv_project_environment",
        "torch_version", "torch_cuda_version", "cuda_available",
    }
    if not isinstance(value, dict) or set(value) != required:
        return [f"{label} runtime source schema mismatch"]
    errors: list[str] = []
    if value.get("schema_version") != "pair-stability-v4/runtime-source/v1":
        errors.append(f"{label} runtime source version mismatch")
    if not _is_sha256(value.get("closure_sha256")):
        errors.append(f"{label} runtime source closure_sha256 is invalid")
    if type(value.get("n_files")) is not int or value["n_files"] <= 0:
        errors.append(f"{label} runtime source file count is invalid")
    commit = value.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        errors.append(f"{label} runtime source git commit is invalid")
    for field in ("git_branch", "python_executable", "python_version"):
        if not isinstance(value.get(field), str) or not value[field]:
            errors.append(f"{label} runtime source {field} is invalid")
    for field in ("uv_project_environment", "torch_version", "torch_cuda_version"):
        if value.get(field) is not None and (
            not isinstance(value.get(field), str) or not value[field]
        ):
            errors.append(f"{label} runtime source {field} is invalid")
    if value.get("dirty") is not False:
        errors.append(f"{label} runtime source is dirty or invalid")
    if type(value.get("cuda_available")) is not bool:
        errors.append(f"{label} runtime source CUDA availability is invalid")
    return errors


def _validate_receipt_files(
    value: object,
    *,
    label: str,
) -> tuple[list[str], list[dict[str, str]]]:
    if not isinstance(value, list) or not value:
        return [f"{label} files are invalid"], []
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    paths: list[str] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append(f"{label} file row is invalid")
            continue
        path = row.get("path")
        digest = row.get("sha256")
        if not _safe_repository_path(path) or not _is_sha256(digest):
            errors.append(f"{label} file value is invalid")
            continue
        assert isinstance(path, str)
        assert isinstance(digest, str)
        paths.append(path)
        rows.append({"path": path, "sha256": digest})
    if len(paths) != len(set(paths)):
        errors.append(f"{label} file paths are not unique")
    return errors, rows


def _safe_repository_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _receipt_digest(files: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(files, key=lambda item: item["path"]):
        digest.update(row["path"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _closure_projection(value: Mapping[str, Any]) -> dict[str, object]:
    """Return equality facts; commits/env paths remain recorded, not equated."""

    legacy = value.get("legacy_source")
    runtime = value.get("runtime_source")
    config = value.get("config_source")
    if not isinstance(legacy, dict) or not isinstance(runtime, dict) or not isinstance(config, dict):
        return {}
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "legacy_closure_sha256": legacy.get("closure_sha256"),
        "runtime_closure_sha256": runtime.get("closure_sha256"),
        "config_closure_sha256": config.get("closure_sha256"),
    }


def _runtime_variant(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value.get(key)
        for key in (
            "python_executable", "python_version", "uv_project_environment",
            "torch_version", "torch_cuda_version", "cuda_available",
        )
    }


def _controller_profile(
    *,
    partition: object,
    walltime: object,
    cpus: object,
    mem_per_cpu_gb: object,
) -> dict[str, object]:
    if not isinstance(partition, str) or partition != CANONICAL_CONTROLLER_PARTITION:
        raise ValueError("canonical V4-0 controller partition must be sapphire")
    if (
        not isinstance(walltime, str)
        or not walltime
        or _slurm_seconds(walltime) > CANONICAL_CONTROLLER_MAX_SECONDS
    ):
        raise ValueError("canonical V4-0 controller walltime exceeds Sapphire three-day limit")
    if (
        type(cpus) is not int
        or type(mem_per_cpu_gb) is not int
        or cpus != 4
        or mem_per_cpu_gb != 8
    ):
        raise ValueError("canonical V4-0 controller resources must be 4 CPU and 8 GiB/CPU")
    return {
        "partition": CANONICAL_CONTROLLER_PARTITION,
        "walltime": walltime,
        "cpus": 4,
        "mem_per_cpu_gb": 8,
    }


def _slurm_seconds(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError("controller walltime must use [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError("controller walltime fields are invalid")
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def _single_lineage(attempts: Mapping[str, str]) -> str:
    values = {str(value) for value in attempts.values()}
    if not values or len(values) != 1:
        raise ValueError("control evidence requires exactly one V4-0 lineage id")
    return values.pop()


def _stack_directory(root: Path, lineage_id: str) -> Path:
    path = require_beneath_root(root / "_v4" / "stack" / lineage_id, root)
    if not path.is_dir() or path.is_symlink():
        raise ValueError("stack evidence directory is missing or unsafe")
    return path


def _stack_path(root: Path, lineage_id: str, name: str) -> Path:
    if "/" in name or "\\" in name:
        raise ValueError("stack evidence name is unsafe")
    return require_beneath_root(root / "_v4" / "stack" / lineage_id / name, root)


def _load_stack_record(root: Path, lineage_id: str, name: str) -> dict[str, Any]:
    path = _stack_path(root, lineage_id, name)
    return _load_json_object(path)


def _load_terminal_record(path: Path) -> tuple[dict[str, Any], str | None]:
    """Load prior controller evidence without preventing terminal failure proof."""

    try:
        return _load_json_object(path), None
    except (OSError, StrictDataError, ValueError) as exc:
        return {}, str(exc)


def _safe_file_sha256(path: Path) -> str | None:
    if path.is_file() and not path.is_symlink():
        return _sha256_file(path)
    return None


def _record_text(value: object) -> str:
    """Serialize a terminal-input value without allowing non-JSON objects."""

    return value if isinstance(value, str) else repr(value)


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"control record is not a regular file: {path}")
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"control record is not an object: {path}")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_job_id(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is invalid")
    text = value
    if _JOB_ID_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} is invalid")
    return text


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is unavailable")
    text = value
    if not text or text == "missing":
        raise ValueError(f"{name} is unavailable")
    return text


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} is invalid")
    return value


def _is_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="microseconds")
