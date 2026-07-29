"""Typed V4-0 dispatcher for independently invocable pinned v3 stage CLIs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import audit
import control_audit
from audit_receipts import inventory_results_tree, inventory_source_tree
from roots import (
    PURPOSE_EXPERIMENT,
    PURPOSE_OWNERSHIP_AUDIT,
    ROOT_PURPOSES,
    initialize_root,
    require_beneath_root,
    require_v4_root,
    root_metadata,
    validate_lineage_id,
    validate_root_links,
)
from routes import (
    REPO_ROOT,
    STUDY_DIR,
    V3_STUDY_DIR,
    LegacyStageRoute,
    config_source_receipt,
    legacy_source_receipt,
    load_routes,
    render_legacy_argv,
    require_launcher_environment,
    runtime_source_receipt,
    verify_legacy_source_manifest,
)
from strict_data import StrictDataError, load_json

DISPATCH_SCHEMA_VERSION = "pair-stability-v4/dispatch/v1"
AUDIT_SCHEMA_VERSION = "pair-stability-v4/ownership-audit/v1"
INPUT_FLAGS = {
    "grid": "--grid-attempt",
    "train": "--train-attempt",
    "collection": "--collection-attempt",
    "selection": "--selection-attempt",
    "final_grid": "--final-grid-attempt",
    "final_train": "--final-train-attempt",
    "final_eval": "--final-eval-attempt",
    "final_collect": "--final-collect-attempt",
}


def dispatch_stage(
    role: str,
    *,
    results_root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
) -> int:
    """Validate ownership, persist receipts, and invoke one pinned v3 CLI."""

    routes = load_routes()
    if role not in routes:
        raise ValueError(f"unknown V4-0 logical role: {role!r}")
    route = routes[role]
    root, normalized_inputs, configs = _prepare_dispatch(
        route,
        results_root=results_root,
        output_attempt=output_attempt,
        input_attempts=input_attempts,
        purpose=PURPOSE_EXPERIMENT,
    )
    require_launcher_environment(route, REPO_ROOT)
    argv = render_legacy_argv(
        route,
        results_root=root,
        output_attempt=output_attempt,
        input_attempts=normalized_inputs,
        config_paths=configs,
        repo_root=REPO_ROOT,
    )
    return _execute_route(
        route,
        root=root,
        output_attempt=output_attempt,
        input_attempts=normalized_inputs,
        config_paths=configs,
        argv=argv,
        purpose=PURPOSE_EXPERIMENT,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Render or run one independently invocable V4-0 stage."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "initialize":
        root = initialize_root(
            Path(args.results_root),
            lineage_id=args.lineage_id,
            purpose=args.purpose,
        )
        print(root)
        return 0
    if args.command == "audit":
        return _run_ownership_audit(
            Path(args.results_root),
            lineage_id=args.lineage_id,
        )
    if args.command == "stack-inventory":
        return _write_stack_inventory(
            Path(args.results_root),
            lineage_id=args.lineage_id,
            phase=args.phase,
        )
    if args.command == "controller-request":
        destination = control_audit.write_controller_request(
            Path(args.results_root),
            lineage_id=args.lineage_id,
            partition=args.partition,
            walltime=args.walltime,
            cpus=args.cpus,
            mem_per_cpu_gb=args.mem_per_cpu_gb,
        )
        print(destination)
        return 0
    if args.command == "controller-submission":
        destination = control_audit.write_controller_submission(
            Path(args.results_root),
            lineage_id=args.lineage_id,
            job_id=args.job_id,
        )
        print(destination)
        return 0
    if args.command == "dispatch-manifest":
        destination = control_audit.write_dispatch_manifest(
            Path(args.results_root),
            lineage_id=args.lineage_id,
        )
        print(destination)
        return 0
    if args.command == "controller-result":
        destination = control_audit.write_controller_result(
            Path(args.results_root),
            lineage_id=args.lineage_id,
            stage=args.stage,
            stage_exit_code=args.stage_exit_code,
            exit_code=args.exit_code,
            finalization_errors=args.finalization_error,
            worker_job_id=args.worker_job_id,
            worker_partition=args.worker_partition,
            effective_cpus_per_task=args.effective_cpus_per_task,
            effective_mem_per_cpu_mb=args.effective_mem_per_cpu_mb,
            effective_time_limit=args.effective_time_limit,
        )
        print(destination)
        return 0
    if args.command == "profile-check":
        errors = audit.audit_gpu_test_fanout_profile()
        if errors:
            raise ValueError("V4-0 gpu_test profile check failed: " + "; ".join(errors))
        print("V4-0 gpu_test profile check passed")
        return 0
    if args.command in {"render", "run"}:
        input_attempts = _input_attempts_from_args(args)
        if args.command == "run":
            return dispatch_stage(
                args.role,
                results_root=Path(args.results_root),
                output_attempt=args.output_attempt,
                input_attempts=input_attempts,
            )
        route = load_routes().get(args.role)
        if route is None:
            parser.error(f"unknown V4-0 logical role: {args.role}")
        root, normalized_inputs, configs = _prepare_dispatch(
            route,
            results_root=Path(args.results_root),
            output_attempt=args.output_attempt,
            input_attempts=input_attempts,
            purpose=PURPOSE_EXPERIMENT,
        )
        command = render_legacy_argv(
            route,
            results_root=root,
            output_attempt=args.output_attempt,
            input_attempts=normalized_inputs,
            config_paths=configs,
            repo_root=REPO_ROOT,
        )
        print(json.dumps(list(command)))
        return 0
    parser.error("missing command")


def _prepare_dispatch(
    route: LegacyStageRoute,
    *,
    results_root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
    purpose: str,
) -> tuple[Path, dict[str, str], dict[str, Path]]:
    output_attempt = validate_lineage_id(output_attempt)
    if route.kind == "fanout":
        profile_errors = audit.audit_gpu_test_fanout_profile()
        if profile_errors:
            raise ValueError(
                "V4-0 gpu_test profile check failed before fan-out submission: "
                + "; ".join(profile_errors)
            )
    root = require_v4_root(
        results_root,
        lineage_id=output_attempt,
        purpose=purpose,
    )
    source_errors = verify_legacy_source_manifest(REPO_ROOT)
    if source_errors:
        raise ValueError("; ".join(source_errors))
    config_source_receipt(REPO_ROOT)
    if validate_root_links(root):
        raise ValueError("guarded v4 root contains an external symlink")

    normalized_inputs = {
        str(key): validate_lineage_id(str(value))
        for key, value in input_attempts.items()
    }
    if set(normalized_inputs) != set(route.required_input_attempts):
        raise ValueError(
            f"{route.logical_role} input attempts mismatch; "
            f"expected={list(route.required_input_attempts)!r}, "
            f"received={sorted(normalized_inputs)!r}"
        )
    if any(attempt != output_attempt for attempt in normalized_inputs.values()):
        raise ValueError(
            "V4-0 requires one fixed lineage/attempt id across every stage"
        )
    _validate_upstream_attempts(root, route, normalized_inputs)
    configs = _resolve_config_paths(root, route, normalized_inputs)
    return root, normalized_inputs, configs


def _execute_route(
    route: LegacyStageRoute,
    *,
    root: Path,
    output_attempt: str,
    input_attempts: Mapping[str, str],
    config_paths: Mapping[str, Path],
    argv: Sequence[str],
    purpose: str,
    profile: str = "canonical",
) -> int:
    invocation_id = (
        datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%dT%H%M%S%f%z")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    receipt_dir = require_beneath_root(
        root
        / "_v4"
        / "dispatch"
        / output_attempt
        / route.logical_role
        / invocation_id,
        root,
    )
    receipt_dir.mkdir(parents=True, exist_ok=False)
    started_at = _timestamp()
    exact_argv = [str(part) for part in argv]
    request = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "logical_role": route.logical_role,
        "physical_stage": route.physical_stage,
        "route_kind": route.kind,
        "profile": profile,
        "purpose": purpose,
        "results_root": str(root),
        "output_attempt": output_attempt,
        "input_attempts": dict(sorted(input_attempts.items())),
        "config_paths": {
            key: str(value) for key, value in sorted(config_paths.items())
        },
        "argv": exact_argv,
        "cwd": str(REPO_ROOT),
        "started_at": started_at,
        "legacy_source": legacy_source_receipt(REPO_ROOT),
        "runtime_source": runtime_source_receipt(REPO_ROOT),
        "config_source": config_source_receipt(REPO_ROOT),
        "pre_run_unsafe_links": list(validate_root_links(root)),
    }
    _write_new_json(receipt_dir / "request.json", request)

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    status = "launch_error"
    returncode: int | None = None
    error: str | None = None
    try:
        result = subprocess.run(
            exact_argv,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        returncode = int(result.returncode)
        status = "completed" if returncode == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        error = "KeyboardInterrupt"
        _write_dispatch_result(
            receipt_dir,
            request=request,
            status=status,
            returncode=returncode,
            error=error,
            root=root,
        )
        raise
    except OSError as exc:
        error = repr(exc)
    _write_dispatch_result(
        receipt_dir,
        request=request,
        status=status,
        returncode=returncode,
        error=error,
        root=root,
    )
    if returncode is None:
        raise RuntimeError(f"could not launch {route.logical_role}: {error}")
    return returncode


def _write_dispatch_result(
    receipt_dir: Path,
    *,
    request: Mapping[str, Any],
    status: str,
    returncode: int | None,
    error: str | None,
    root: Path,
) -> None:
    signal = -returncode if returncode is not None and returncode < 0 else None
    result = {
        "schema_version": DISPATCH_SCHEMA_VERSION,
        "invocation_id": request["invocation_id"],
        "logical_role": request["logical_role"],
        "status": status,
        "returncode": returncode,
        "signal": signal,
        "error": error,
        "argv": request["argv"],
        "cwd": request["cwd"],
        "legacy_source": request["legacy_source"],
        "runtime_source": request["runtime_source"],
        "config_source": request["config_source"],
        "started_at": request["started_at"],
        "completed_at": _timestamp(),
        "post_run_unsafe_links": list(validate_root_links(root)),
    }
    _write_new_json(receipt_dir / "result.json", result)


def _resolve_config_paths(
    root: Path,
    route: LegacyStageRoute,
    inputs: Mapping[str, str],
) -> dict[str, Path]:
    if not route.required_configs:
        return {}
    if route.logical_role == "screen_plan":
        available = {
            "smoke": STUDY_DIR / "configs" / "smoke.yaml",
            "train": STUDY_DIR / "configs" / "pair_stability.yaml",
        }
    elif route.logical_role in {"screen_eval", "confirm_plan"}:
        grid_dir = require_beneath_root(
            root / "00_grid" / inputs["grid"],
            root,
        )
        manifest = _read_json_object(grid_dir / "manifest.json")
        snapshots = manifest.get("config_snapshots")
        if not isinstance(snapshots, dict):
            raise ValueError("grid manifest config_snapshots must be an object")
        available = {
            "train": _guarded_snapshot(grid_dir, snapshots, "train", root),
            "validation": _guarded_snapshot(
                grid_dir,
                snapshots,
                "validation",
                root,
            ),
        }
    elif route.logical_role in {"confirm_train", "confirm_eval"}:
        manifest_path = require_beneath_root(
            root / "05_final_grid" / inputs["final_grid"] / "manifest.json",
            root,
        )
        manifest = _read_json_object(manifest_path)
        available = {
            "train": _guarded_manifest_path(manifest, "train_config", root),
            "validation": _guarded_manifest_path(manifest, "eval_config", root),
        }
    else:
        raise ValueError(
            f"route {route.logical_role} unexpectedly declares configs"
        )
    selected = {name: available[name] for name in route.required_configs}
    for name, path in selected.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"missing materialized {name} config for "
                f"{route.logical_role}: {path}"
            )
    return selected


def _validate_upstream_attempts(
    root: Path,
    route: LegacyStageRoute,
    inputs: Mapping[str, str],
) -> None:
    direct = {
        "grid": ("00_grid", "manifest.json"),
        "collection": ("03_collect", "collection_report.json"),
        "selection": ("04_select", "selection_report.json"),
        "final_grid": ("05_final_grid", "manifest.json"),
        "final_collect": ("08_final_collect", "manifest.yaml"),
    }
    for key, attempt in inputs.items():
        if key in direct:
            stage, filename = direct[key]
            path = require_beneath_root(root / stage / attempt / filename, root)
            if not path.is_file() or path.is_symlink():
                raise ValueError(
                    f"missing {key} upstream artifact: {path.relative_to(root)}"
                )
        elif key == "train":
            _validate_per_run_attempts(
                root,
                grid_attempt=inputs["grid"],
                stage="01_train",
                run_key="run_id",
                attempt=attempt,
            )
        elif key in {"final_train", "final_eval"}:
            stage = "06_final_train" if key == "final_train" else "07_final_eval"
            _validate_per_run_attempts(
                root,
                grid_attempt=inputs["final_grid"],
                stage=stage,
                run_key="final_run_id",
                attempt=attempt,
                final=True,
            )
        else:
            raise ValueError(f"unsupported upstream attempt key: {key!r}")


def _validate_per_run_attempts(
    root: Path,
    *,
    grid_attempt: str,
    stage: str,
    run_key: str,
    attempt: str,
    final: bool = False,
) -> None:
    grid_stage = "05_final_grid" if final else "00_grid"
    manifest = _read_json_object(
        require_beneath_root(
            root / grid_stage / grid_attempt / "manifest.json",
            root,
        )
    )
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        if final:
            jobs_dir = root / grid_stage / grid_attempt / "jobs"
            jobs = [
                _read_json_object(path)
                for path in sorted(jobs_dir.iterdir())
                if path.suffix == ".json" and path.is_file()
            ]
        else:
            raise ValueError(f"{grid_stage} manifest jobs must be a list")
    for row in jobs:
        if not isinstance(row, dict) or not row.get(run_key):
            raise ValueError(f"{grid_stage} job is missing {run_key}")
        run_id = str(row[run_key])
        attempt_dir = require_beneath_root(
            root / stage / run_id / attempt,
            root,
        )
        if not attempt_dir.is_dir() or attempt_dir.is_symlink():
            raise ValueError(
                f"missing per-run upstream attempt: "
                f"{attempt_dir.relative_to(root)}"
            )


def _guarded_snapshot(
    directory: Path,
    snapshots: Mapping[str, Any],
    key: str,
    root: Path,
) -> Path:
    value = snapshots.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"grid manifest does not declare {key} snapshot")
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"invalid grid {key} snapshot path: {value}")
    return require_beneath_root(directory / value, root)


def _guarded_manifest_path(
    manifest: Mapping[str, Any],
    key: str,
    root: Path,
) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"final-grid manifest does not declare {key}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"final-grid {key} must be an absolute materialized path")
    return require_beneath_root(path, root)


def _run_ownership_audit(results_root: Path, *, lineage_id: str) -> int:
    lineage_id = validate_lineage_id(lineage_id)
    source_errors = verify_legacy_source_manifest(REPO_ROOT)
    if source_errors:
        raise ValueError(
            "ownership audit source preflight failed: "
            + "; ".join(source_errors)
        )
    config_source_receipt(REPO_ROOT)
    before_source = inventory_source_tree(REPO_ROOT)
    before_results = inventory_results_tree(V3_STUDY_DIR / "results")
    root = initialize_root(
        results_root,
        lineage_id=lineage_id,
        purpose=PURPOSE_OWNERSHIP_AUDIT,
    )
    rendered_routes: dict[str, list[str]] = {}
    plan_code: int | None = None
    train_code: int | None = None
    train_state = "not_started"
    execution_error: BaseException | None = None
    try:
        routes = load_routes()
        attempts = {name: lineage_id for name in INPUT_FLAGS}
        future_configs = {
            "smoke": STUDY_DIR / "configs" / "smoke.yaml",
            "train": root / "00_grid" / lineage_id / "train_config.yaml",
            "validation": root
            / "00_grid"
            / lineage_id
            / "validation_config.yaml",
        }
        for role, route in routes.items():
            role_inputs = {
                key: attempts[key] for key in route.required_input_attempts
            }
            role_configs = {
                key: future_configs[key] for key in route.required_configs
            }
            rendered_routes[role] = list(
                render_legacy_argv(
                    route,
                    results_root=root,
                    output_attempt=lineage_id,
                    input_attempts=role_inputs,
                    config_paths=role_configs,
                    repo_root=REPO_ROOT,
                )
            )

        plan_route = replace(
            routes["screen_plan"],
            arguments=(*routes["screen_plan"].arguments, "--limit", "1"),
        )
        plan_configs = {
            "smoke": STUDY_DIR / "configs" / "smoke.yaml",
            "train": STUDY_DIR / "configs" / "pair_stability.yaml",
        }
        plan_argv = render_legacy_argv(
            plan_route,
            results_root=root,
            output_attempt=lineage_id,
            input_attempts={},
            config_paths=plan_configs,
            repo_root=REPO_ROOT,
        )
        plan_code = _execute_route(
            plan_route,
            root=root,
            output_attempt=lineage_id,
            input_attempts={},
            config_paths=plan_configs,
            argv=plan_argv,
            purpose=PURPOSE_OWNERSHIP_AUDIT,
            profile="one-row-local-cpu",
        )
        train_state = "skipped_after_plan_failure"
        if plan_code == 0:
            train_route = _local_audit_train_route(routes["screen_train"])
            train_argv = render_legacy_argv(
                train_route,
                results_root=root,
                output_attempt=lineage_id,
                input_attempts={"grid": lineage_id},
                config_paths={},
                repo_root=REPO_ROOT,
            )
            train_code = _execute_route(
                train_route,
                root=root,
                output_attempt=lineage_id,
                input_attempts={"grid": lineage_id},
                config_paths={},
                argv=train_argv,
                purpose=PURPOSE_OWNERSHIP_AUDIT,
                profile="one-row-local-cpu",
            )
            train_state = "completed" if train_code == 0 else "failed"
    except BaseException as exc:
        execution_error = exc
        train_state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "exception"

    after_source = inventory_source_tree(REPO_ROOT)
    after_results = inventory_results_tree(V3_STUDY_DIR / "results")
    try:
        identity_errors = audit.audit_identity(
            root,
            attempts={"grid": lineage_id},
        )
    except (OSError, ValueError) as exc:
        identity_errors = [f"ownership audit identity check failed: {exc}"]
    errors = [
        *(["pinned v3 source changed during audit"] if before_source != after_source else []),
        *(["live v3 results changed during audit"] if before_results != after_results else []),
        *(f"unsafe v4 root link: {item}" for item in validate_root_links(root)),
        *identity_errors,
        *(
            _audit_ownership_train_result(root, lineage_id=lineage_id)
            if plan_code == 0
            else ("ownership audit train unavailable after plan failure",)
        ),
        *(
            [
                "ownership audit execution interrupted: "
                f"{type(execution_error).__name__}: {execution_error}"
            ]
            if execution_error is not None
            else []
        ),
    ]
    receipt_dir = require_beneath_root(
        root / "_v4" / "ownership_audit" / lineage_id,
        root,
    )
    receipt_dir.mkdir(parents=True, exist_ok=False)
    _write_new_json(
        receipt_dir / "audit.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "lineage_id": lineage_id,
            "purpose": PURPOSE_OWNERSHIP_AUDIT,
            "results_root": str(root),
            "completed_at": _timestamp(),
            "plan_returncode": plan_code,
            "train_returncode": train_code,
            "train_state": train_state,
            "rendered_routes": rendered_routes,
            "source_before": list(before_source),
            "source_after": list(after_source),
            "legacy_results_before": list(before_results),
            "legacy_results_after": list(after_results),
            "errors": errors,
            "outcome": "passed"
            if plan_code == 0
            and train_code == 0
            and execution_error is None
            and not errors
            else "failed",
        },
    )
    if execution_error is not None:
        raise execution_error.with_traceback(execution_error.__traceback__)
    if plan_code not in {None, 0}:
        return plan_code
    if train_code not in {None, 0}:
        return train_code
    return 0 if not errors else 1


def _audit_ownership_train_result(
    root: Path,
    *,
    lineage_id: str,
) -> tuple[str, ...]:
    """Audit the representative local train result produced by ownership audit."""

    errors: list[str] = []
    root = require_v4_root(
        root,
        lineage_id=lineage_id,
        purpose=PURPOSE_OWNERSHIP_AUDIT,
    )
    grid_dir = root / "00_grid" / lineage_id
    grid = _read_audit_json(grid_dir / "manifest.json", errors)
    jobs = grid.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1 or not isinstance(
        jobs[0],
        dict,
    ):
        errors.append("ownership audit grid does not contain exactly one job")
        return tuple(errors)
    run_id = str(jobs[0].get("run_id") or "")
    if not run_id or "/" in run_id or "\\" in run_id:
        errors.append("ownership audit grid run identity is invalid")
        return tuple(errors)
    result_dir = root / "01_train" / run_id / lineage_id
    try:
        result_dir = require_beneath_root(result_dir, root)
    except ValueError as exc:
        errors.append(f"ownership audit train result path is unsafe: {exc}")
        return tuple(errors)
    if not result_dir.is_dir() or result_dir.is_symlink():
        errors.append("ownership audit train result directory is missing")
        return tuple(errors)

    expected_source = {
        "run_id": run_id,
        "grid_attempt_id": lineage_id,
        "grid_attempt_dir": str(grid_dir),
        "manifest_path": str(grid_dir / "manifest.json"),
    }
    source = _read_audit_json(
        result_dir / "source_grid_attempt.json",
        errors,
    )
    if source != expected_source:
        errors.append("ownership audit train source-grid record mismatch")

    submission = _read_audit_json(result_dir / "submission.json", errors)
    expected_submission_fields = {
        "run_id",
        "grid_attempt_id",
        "launcher",
        "launcher_job_id",
        "command",
        "submitted_command",
    }
    if set(submission) != expected_submission_fields:
        errors.append("ownership audit train submission fields mismatch")
    if submission.get("run_id") != run_id:
        errors.append("ownership audit train submission run id mismatch")
    if submission.get("grid_attempt_id") != lineage_id:
        errors.append("ownership audit train submission grid attempt mismatch")
    if submission.get("launcher") != "local":
        errors.append("ownership audit train submission is not local")

    status = _read_audit_json(result_dir / "status.json", errors)
    if (
        status.get("status") != "completed"
        or status.get("current_event") != "run_end"
        or status.get("exception_type") is not None
        or status.get("exception_message") is not None
    ):
        errors.append("ownership audit train status is not terminal success")

    run_start = _read_audit_json(result_dir / "run_start.json", errors)
    metadata = _read_audit_json(result_dir / "metadata.json", errors)
    expected_runtime_id = f"{run_id}/{lineage_id}"
    if run_start.get("run_id") != expected_runtime_id:
        errors.append("ownership audit run_start run id mismatch")
    if run_start.get("run_dir") != str(result_dir):
        errors.append("ownership audit run_start run directory mismatch")
    study = run_start.get("study")
    if not isinstance(study, dict) or study != {
        "name": "pair_stability_v4",
        "config_id": None,
    }:
        errors.append("ownership audit run_start study identity mismatch")
    command = run_start.get("command")
    if not isinstance(command, str) or not _ownership_command_identifies_v4(
        command,
        root=root,
        run_id=run_id,
        lineage_id=lineage_id,
    ):
        errors.append("ownership audit run_start command identity mismatch")
    if metadata.get("run_id") != expected_runtime_id:
        errors.append("ownership audit metadata run id mismatch")
    if metadata.get("run_dir") != str(result_dir):
        errors.append("ownership audit metadata run directory mismatch")
    if metadata.get("command") != command:
        errors.append("ownership audit run_start/metadata command mismatch")
    if metadata.get("status") != "completed":
        errors.append("ownership audit metadata status is not completed")
    runtime = metadata.get("runtime")
    device = runtime.get("device") if isinstance(runtime, dict) else metadata.get(
        "device"
    )
    if device != "cpu":
        errors.append("ownership audit train did not execute on CPU")

    command_path = result_dir / "command.txt"
    try:
        submitted_command = command_path.read_text().strip()
    except OSError:
        submitted_command = ""
    if (
        not submitted_command
        or submission.get("submitted_command") != submitted_command
        or "pair_stability_v3" in submitted_command
        or str(root) not in submitted_command
        or expected_runtime_id not in submitted_command
    ):
        errors.append("ownership audit submitted command identity mismatch")

    pointer = result_dir / "checkpoints" / "latest.json"
    if not _audit_complete_checkpoint(pointer):
        errors.append("ownership audit train checkpoint is incomplete")
    return tuple(dict.fromkeys(errors))


def _ownership_command_identifies_v4(
    command: str,
    *,
    root: Path,
    run_id: str,
    lineage_id: str,
) -> bool:
    return (
        "study.name=pair_stability_v4" in command
        and "pair_stability_v3" not in command
        and f"run.run_id={run_id}/{lineage_id}" in command
        and str(root / "01_train") in command
    )


def _audit_complete_checkpoint(pointer_path: Path) -> bool:
    pointer = _read_audit_json(pointer_path, [])
    directory_name = pointer.get("checkpoint_dir")
    if (
        not isinstance(directory_name, str)
        or not directory_name
        or Path(directory_name).is_absolute()
        or ".." in Path(directory_name).parts
    ):
        return False
    concrete = pointer_path.parent / directory_name
    return (
        concrete.is_dir()
        and not concrete.is_symlink()
        and (concrete / "COMPLETE").is_file()
        and not (concrete / "COMPLETE").is_symlink()
        and (concrete / "manifest.json").is_file()
        and not (concrete / "manifest.json").is_symlink()
    )


def _read_audit_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (OSError, StrictDataError) as exc:
        errors.append(f"invalid ownership audit JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"ownership audit JSON is not an object: {path}")
        return {}
    return value


def _local_audit_train_route(route: LegacyStageRoute) -> LegacyStageRoute:
    arguments = list(route.arguments)
    replacements = {
        "--backend": "local",
        "--device": "cpu",
        "--chunk-size": "1",
    }
    removed_flags = {
        "--slurm-cpus",
        "--slurm-partition",
        "--slurm-mem-per-cpu-gb",
        "--slurm-timeout-min",
    }
    result: list[str] = []
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in replacements:
            result.extend((flag, replacements[flag]))
            index += 2
        elif flag in removed_flags:
            index += 2
        else:
            result.append(flag)
            index += 1
    return replace(route, arguments=tuple(result))


def _write_stack_inventory(
    results_root: Path,
    *,
    lineage_id: str,
    phase: str,
) -> int:
    root = require_v4_root(
        results_root,
        lineage_id=lineage_id,
        purpose=PURPOSE_EXPERIMENT,
    )
    if phase not in {"pre", "post"}:
        raise ValueError("stack inventory phase must be pre or post")
    destination = require_beneath_root(
        root / "_v4" / "stack" / lineage_id / f"legacy-{phase}.json",
        root,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(
        destination,
        {
            "schema_version": control_audit.LEGACY_INVENTORY_SCHEMA_VERSION,
            "phase": phase,
            "lineage_id": lineage_id,
            "captured_at": _timestamp(),
            "source": list(inventory_source_tree(REPO_ROOT)),
            "results": list(inventory_results_tree(V3_STUDY_DIR / "results")),
            "legacy_source": legacy_source_receipt(REPO_ROOT),
            "runtime_source": runtime_source_receipt(REPO_ROOT),
            "config_source": config_source_receipt(REPO_ROOT),
        },
    )
    print(destination)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--results-root", required=True)
    initialize.add_argument("--lineage-id", required=True)
    initialize.add_argument(
        "--purpose",
        choices=sorted(ROOT_PURPOSES),
        default=PURPOSE_EXPERIMENT,
    )

    for command in ("render", "run"):
        stage = subparsers.add_parser(command)
        stage.add_argument("role", choices=tuple(load_routes()))
        stage.add_argument("--results-root", required=True)
        stage.add_argument("--output-attempt", required=True)
        for name, flag in INPUT_FLAGS.items():
            stage.add_argument(flag, dest=f"{name}_attempt")

    ownership = subparsers.add_parser("audit")
    ownership.add_argument("--results-root", required=True)
    ownership.add_argument("--lineage-id", required=True)

    inventory = subparsers.add_parser("stack-inventory")
    inventory.add_argument("--results-root", required=True)
    inventory.add_argument("--lineage-id", required=True)
    inventory.add_argument("--phase", choices=("pre", "post"), required=True)

    controller_request = subparsers.add_parser("controller-request")
    controller_request.add_argument("--results-root", required=True)
    controller_request.add_argument("--lineage-id", required=True)
    controller_request.add_argument("--partition", required=True)
    controller_request.add_argument("--walltime", required=True)
    controller_request.add_argument("--cpus", type=int, required=True)
    controller_request.add_argument("--mem-per-cpu-gb", type=int, required=True)

    controller_submission = subparsers.add_parser("controller-submission")
    controller_submission.add_argument("--results-root", required=True)
    controller_submission.add_argument("--lineage-id", required=True)
    controller_submission.add_argument("--job-id", required=True)

    dispatch_manifest = subparsers.add_parser("dispatch-manifest")
    dispatch_manifest.add_argument("--results-root", required=True)
    dispatch_manifest.add_argument("--lineage-id", required=True)

    controller_result = subparsers.add_parser("controller-result")
    controller_result.add_argument("--results-root", required=True)
    controller_result.add_argument("--lineage-id", required=True)
    controller_result.add_argument("--stage", required=True)
    controller_result.add_argument("--stage-exit-code", type=int, required=True)
    controller_result.add_argument("--exit-code", type=int, required=True)
    controller_result.add_argument(
        "--finalization-error",
        action="append",
        default=[],
    )
    controller_result.add_argument("--worker-job-id", required=True)
    controller_result.add_argument("--worker-partition", required=True)
    controller_result.add_argument("--effective-cpus-per-task", required=True)
    controller_result.add_argument("--effective-mem-per-cpu-mb", required=True)
    controller_result.add_argument("--effective-time-limit", required=True)

    profile_check = subparsers.add_parser("profile-check")
    profile_check.add_argument("--results-root", required=False)
    return parser


def _input_attempts_from_args(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in INPUT_FLAGS:
        value = getattr(args, f"{name}_attempt")
        if value is not None:
            values[name] = value
    return values


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = load_json(Path(path))
    except (OSError, StrictDataError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _timestamp() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat(
        timespec="microseconds"
    )


if __name__ == "__main__":
    raise SystemExit(main())
