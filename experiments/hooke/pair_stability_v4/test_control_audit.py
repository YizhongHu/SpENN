"""Focused fail-closed tests for V4-0 controller evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import control_audit  # noqa: E402
import routes  # noqa: E402
from roots import PURPOSE_EXPERIMENT, root_metadata  # noqa: E402
from test_reference import _completed_lineage  # noqa: E402


_TIME = "2026-07-24T19:00:00-04:00"


def test_control_closure_accepts_one_local_and_one_fanout_runtime_profile(
    tmp_path: Path,
) -> None:
    """Local/controller and Submitit runtimes may differ by route kind only."""

    root, attempts = _completed_lineage(tmp_path)
    _write_valid_control_evidence(root, attempts)

    assert control_audit.audit_control_closure(root, attempts=attempts) == ()
    provenance = control_audit.control_provenance(root, attempts=attempts)
    profiles = provenance["dispatch_runtime_profiles"]
    assert profiles["local"]["uv_project_environment"] == "/tmp/.venv"
    assert profiles["fanout"]["uv_project_environment"] == "/tmp/.venv-submitit"
    paths = {row["path"] for row in provenance["record_digests"]}
    assert any(path.endswith("/request.json") for path in paths)
    assert any(path.endswith("/result.json") for path in paths)


def test_control_closure_rejects_mixed_kind_runtime_and_malformed_inventory(
    tmp_path: Path,
) -> None:
    """A route-kind runtime mixture and equal junk inventories both fail closed."""

    root, attempts = _completed_lineage(tmp_path)
    _write_valid_control_evidence(root, attempts)
    lineage = attempts["grid"]
    receipt = next(
        (root / "_v4" / "dispatch" / lineage / "screen_train").iterdir()
    )
    for filename in ("request.json", "result.json"):
        path = receipt / filename
        value = json.loads(path.read_text())
        value["runtime_source"]["python_executable"] = "/tmp/other-python"
        _write_json(path, value)
    errors = control_audit.audit_control_closure(root, attempts=attempts)
    assert any("runtime environment is mixed for fanout" in error for error in errors)

    # Matching, structurally valid pre/post junk still cannot claim source-tree
    # identity because it is bound to the recorded legacy closure.
    for phase in ("pre", "post"):
        path = root / "_v4" / "stack" / lineage / f"legacy-{phase}.json"
        value = json.loads(path.read_text())
        value["source"] = [
            {
                "path": "junk.py",
                "type": "file",
                "size": 1,
                "mtime_ns": 1,
                "mode": 33188,
                "sha256": "f" * 64,
            }
        ]
        _write_json(path, value)
    errors = control_audit.audit_control_closure(root, attempts=attempts)
    assert any("source inventory differs from recorded legacy closure" in error for error in errors)


def test_terminal_effective_profile_mismatch_writes_incomplete_receipt(
    tmp_path: Path,
) -> None:
    """A bad actual allocation is recorded before the controller fails."""

    root, attempts = _completed_lineage(tmp_path)
    _write_valid_control_evidence(root, attempts, write_terminal_result=False)
    lineage = attempts["grid"]

    with pytest.raises(ValueError, match="terminal evidence is incomplete"):
        control_audit.write_controller_result(
            root,
            lineage_id=lineage,
            stage="complete",
            stage_exit_code=0,
            exit_code=0,
            finalization_errors=(),
            worker_job_id="12345",
            worker_partition="sapphire",
            effective_cpus_per_task="4",
            effective_mem_per_cpu_mb="4096",
            effective_time_limit="3-00:00:00",
        )
    result = json.loads(
        (root / "_v4" / "stack" / lineage / "controller-result.json").read_text()
    )
    assert result["status"] == "incomplete"
    assert any("memory allocation differs" in item for item in result["finalization_errors"])
    errors = control_audit.audit_control_closure(root, attempts=attempts)
    assert any("effective profile" in error for error in errors)


def test_control_closure_rejects_boolean_exit_and_truncated_receipt(
    tmp_path: Path,
) -> None:
    """JSON booleans and copied digest stubs cannot masquerade as evidence."""

    root, attempts = _completed_lineage(tmp_path)
    _write_valid_control_evidence(root, attempts)
    lineage = attempts["grid"]
    result_path = root / "_v4" / "stack" / lineage / "controller-result.json"
    result = json.loads(result_path.read_text())
    result["exit_code"] = False
    _write_json(result_path, result)
    errors = control_audit.audit_control_closure(root, attempts=attempts)
    assert any("nonzero stage or controller exit" in error for error in errors)

    request_path = root / "_v4" / "stack" / lineage / "controller-request.json"
    request = json.loads(request_path.read_text())
    request["legacy_source"] = {
        "closure_sha256": request["legacy_source"]["closure_sha256"]
    }
    _write_json(request_path, request)
    errors = control_audit.audit_control_closure(root, attempts=attempts)
    assert any("legacy source schema mismatch" in error for error in errors)


@pytest.mark.parametrize(
    ("partition", "walltime"),
    (("kozinsky", "3-00:00:00"), ("sapphire", "4-00:00:00")),
)
def test_controller_request_rejects_noncanonical_cluster_profile(
    tmp_path: Path,
    partition: str,
    walltime: str,
) -> None:
    """The observed Sapphire controller constraint is an executable contract."""

    root, attempts = _completed_lineage(tmp_path)
    with pytest.raises(ValueError):
        control_audit.write_controller_request(
            root,
            lineage_id=attempts["grid"],
            partition=partition,
            walltime=walltime,
            cpus=4,
            mem_per_cpu_gb=8,
        )


def _write_valid_control_evidence(
    root: Path,
    attempts: dict[str, str],
    *,
    write_terminal_result: bool = True,
    source_receipts: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ] | None = None,
) -> None:
    """Materialize one fully valid, synthetic V4-only control closure."""

    lineage = attempts["grid"]
    stack = root / "_v4" / "stack" / lineage
    stack.mkdir(parents=True, exist_ok=True)
    legacy, runtime_local, runtime_fanout, config = (
        _source_receipts() if source_receipts is None else source_receipts
    )
    controller = {
        "partition": "sapphire",
        "walltime": "3-00:00:00",
        "cpus": 4,
        "mem_per_cpu_gb": 8,
    }
    controller_request = {
        "schema_version": control_audit.CONTROLLER_SCHEMA_VERSION,
        "phase": "request",
        "lineage_id": lineage,
        "results_root": str(root),
        "root_metadata": root_metadata(root),
        "controller": controller,
        "legacy_source": legacy,
        "runtime_source": runtime_local,
        "config_source": config,
        "created_at": _TIME,
    }
    request_path = stack / "controller-request.json"
    _write_json(request_path, controller_request)
    submission = {
        "schema_version": control_audit.CONTROLLER_SCHEMA_VERSION,
        "phase": "submission",
        "lineage_id": lineage,
        "results_root": str(root),
        "request_sha256": control_audit._sha256_file(request_path),
        "controller_job_id": "12345",
        "submitted_at": _TIME,
    }
    _write_json(stack / "controller-submission.json", submission)
    source_inventory = [
        {
            "path": row["path"],
            "type": "file",
            "size": 1,
            "mtime_ns": 1,
            "mode": 33188,
            "sha256": row["sha256"],
        }
        for row in sorted(legacy["files"], key=lambda item: str(item["path"]))
    ]
    for phase in ("pre", "post"):
        _write_json(
            stack / f"legacy-{phase}.json",
            {
                "schema_version": control_audit.LEGACY_INVENTORY_SCHEMA_VERSION,
                "phase": phase,
                "lineage_id": lineage,
                "captured_at": _TIME,
                "source": source_inventory,
                "results": [],
                "legacy_source": legacy,
                "runtime_source": runtime_local,
                "config_source": config,
            },
        )

    result_paths: list[str] = []
    for role, route in routes.load_routes().items():
        receipt = root / "_v4" / "dispatch" / lineage / role / f"receipt-{role}"
        receipt.mkdir(parents=True)
        configs = control_audit._expected_config_paths(
            root,
            role=role,
            lineage_id=lineage,
        )
        input_attempts = {
            key: lineage for key in route.required_input_attempts
        }
        argv = list(
            routes.render_legacy_argv(
                route,
                results_root=root,
                output_attempt=lineage,
                input_attempts=input_attempts,
                config_paths=configs,
                repo_root=routes.REPO_ROOT,
            )
        )
        runtime = runtime_fanout if route.kind == "fanout" else runtime_local
        request = {
            "schema_version": "pair-stability-v4/dispatch/v1",
            "invocation_id": receipt.name,
            "logical_role": role,
            "physical_stage": route.physical_stage,
            "route_kind": route.kind,
            "profile": "canonical",
            "purpose": PURPOSE_EXPERIMENT,
            "results_root": str(root),
            "output_attempt": lineage,
            "input_attempts": input_attempts,
            "config_paths": {key: str(value) for key, value in sorted(configs.items())},
            "argv": argv,
            "cwd": str(routes.REPO_ROOT),
            "started_at": _TIME,
            "legacy_source": legacy,
            "runtime_source": runtime,
            "config_source": config,
            "pre_run_unsafe_links": [],
        }
        _write_json(receipt / "request.json", request)
        _write_json(
            receipt / "result.json",
            {
                "schema_version": "pair-stability-v4/dispatch/v1",
                "invocation_id": receipt.name,
                "logical_role": role,
                "status": "completed",
                "returncode": 0,
                "signal": None,
                "error": None,
                "argv": argv,
                "cwd": str(routes.REPO_ROOT),
                "legacy_source": legacy,
                "runtime_source": runtime,
                "config_source": config,
                "started_at": _TIME,
                "completed_at": _TIME,
                "post_run_unsafe_links": [],
            },
        )
        result_paths.append((receipt / "result.json").relative_to(root).as_posix())
    _write_json(
        stack / "dispatch-results.json",
        {
            "schema_version": control_audit.DISPATCH_MANIFEST_SCHEMA_VERSION,
            "lineage_id": lineage,
            "results_root": str(root),
            "result_paths": sorted(result_paths),
            "created_at": _TIME,
        },
    )
    if write_terminal_result:
        control_audit.write_controller_result(
            root,
            lineage_id=lineage,
            stage="complete",
            stage_exit_code=0,
            exit_code=0,
            finalization_errors=(),
            worker_job_id="12345",
            worker_partition="sapphire",
            effective_cpus_per_task="4",
            effective_mem_per_cpu_mb="8192",
            effective_time_limit="3-00:00:00",
        )


def _source_receipts() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Return valid minimal closure facts with local/fanout runtime variants."""

    legacy_files = [{"path": "experiments/hooke/pair_stability_v3/plan.py", "sha256": "a" * 64}]
    config_files = [{"path": "experiments/hooke/pair_stability_v4/configs/smoke.yaml", "sha256": "b" * 64}]
    legacy = {
        "schema_version": "pair-stability-v4/legacy-source/v1",
        "manifest_path": "experiments/hooke/pair_stability_v4/legacy_source_v1.json",
        "manifest_sha256": "c" * 64,
        "closure_sha256": control_audit._receipt_digest(legacy_files),
        "files": legacy_files,
    }
    config = {
        "schema_version": "pair-stability-v4/config-source/v1",
        "closure_sha256": control_audit._receipt_digest(config_files),
        "files": config_files,
    }

    def runtime(environment: str, executable: str) -> dict[str, object]:
        return {
            "schema_version": "pair-stability-v4/runtime-source/v1",
            "closure_sha256": "d" * 64,
            "n_files": 1,
            "git_commit": "e" * 40,
            "git_branch": "codex/test",
            "dirty": False,
            "python_executable": executable,
            "python_version": "3.12.0",
            "uv_project_environment": environment,
            "torch_version": None,
            "torch_cuda_version": None,
            "cuda_available": False,
        }

    return legacy, runtime("/tmp/.venv", "/tmp/.venv/bin/python"), runtime(
        "/tmp/.venv-submitit",
        "/tmp/.venv-submitit/bin/python",
    ), config


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
