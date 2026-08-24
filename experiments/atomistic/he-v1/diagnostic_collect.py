"""Fail-loud reconciliation for the frozen He-v1 diagnostic study.

Collection joins rows only by immutable plan identity. It requires every row,
checkpoint binding, allocation receipt, successful task, complete raw artifact,
timing/resource metric stream, and replay-provenance record before publishing a
two-checkpoint result bundle. No checkpoint or arm is selected as preferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import diagnostic_plan as plan_stage  # noqa: E402
import layout  # noqa: E402

COLLECT_SCHEMA = "he-v1-diagnostic-collection/v1"


class DiagnosticCollectError(RuntimeError):
    """One or more planned diagnostic rows failed immutable reconciliation."""


def collect(
    manifest: Mapping[str, Any],
    *,
    results_root: str | Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
    collect_attempt_id: str,
) -> dict[str, Any]:
    """Collect every planned row, write one receipt, and raise if any fail."""

    results_root = Path(results_root)
    output_dir = layout.collect_attempt_dir(results_root, collect_attempt_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    production_grid = STUDY_DIR / "configs" / "production_grid.yaml"
    production_grid_after = plan_stage.file_sha256(production_grid)
    errors: list[dict[str, str]] = []
    if production_grid_after != manifest["production_grid_sha256_before"]:
        errors.append(
            {
                "row_id": "<study>",
                "error": "production_grid.yaml changed after diagnostic planning",
            }
        )
    rows: list[dict[str, Any]] = []
    for planned in manifest["rows"]:
        row_id = str(planned["row_id"])
        try:
            rows.append(
                reconcile_row(
                    manifest,
                    planned,
                    results_root=results_root,
                    plan_attempt_id=plan_attempt_id,
                    launch_attempt_id=launch_attempt_id,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "row_id": row_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    checkpoint_labels = [str(item["label"]) for item in manifest["checkpoints"]]
    if checkpoint_labels != ["step_025000", "step_050000"]:
        errors.append(
            {
                "row_id": "<study>",
                "error": f"checkpoint set/order changed: {checkpoint_labels}",
            }
        )
    if len(manifest["rows"]) != 42:
        errors.append(
            {
                "row_id": "<study>",
                "error": f"frozen study requires 42 rows, found {len(manifest['rows'])}",
            }
        )
    for label in checkpoint_labels:
        count = sum(
            row["checkpoint_label"] == label for row in manifest["rows"]
        )
        if count != 21:
            errors.append(
                {
                    "row_id": "<study>",
                    "error": f"checkpoint {label} requires 21 planned rows, found {count}",
                }
            )
    checkpoint_reports = [
        {
            "checkpoint_label": label,
            "checkpoint_model_sha256": next(
                checkpoint["model_sha256"]
                for checkpoint in manifest["checkpoints"]
                if checkpoint["label"] == label
            ),
            "row_ids": [
                row["row_id"] for row in rows if row["checkpoint_label"] == label
            ],
        }
        for label in checkpoint_labels
    ]
    report = {
        "schema": COLLECT_SCHEMA,
        "study": manifest["study"],
        "scale": manifest["scale"],
        "plan_attempt_id": plan_attempt_id,
        "plan_sha256": manifest["plan_sha256"],
        "launch_attempt_id": launch_attempt_id,
        "collect_attempt_id": collect_attempt_id,
        "evaluation_git_sha": manifest["evaluation_git_sha"],
        "created_at": datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "checkpoint_reporting": "both_without_selection",
        "selection_policy": "none",
        "production_run_mutation_authorized": False,
        "production_grid_sha256_before": manifest["production_grid_sha256_before"],
        "production_grid_sha256_after": production_grid_after,
        "status": "failed" if errors else "success",
        "n_planned_rows": len(manifest["rows"]),
        "n_collected_rows": len(rows),
        "checkpoints": checkpoint_reports,
        "rows": rows,
        "errors": errors,
    }
    _write_new_json(output_dir / "collected.json", report)
    _write_rows_csv(output_dir / "rows.csv", rows)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_COLLECT), collect_attempt_id)
    if errors:
        raise DiagnosticCollectError(
            f"diagnostic collection failed with {len(errors)} reconciliation error(s); "
            f"receipt={output_dir / 'collected.json'}"
        )
    return report


def reconcile_row(
    manifest: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    results_root: Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
) -> dict[str, Any]:
    """Return a content-addressed row record after exact reconciliation."""

    row_id = str(planned["row_id"])
    outer = layout.row_dir(
        results_root,
        layout.STAGE_EVAL,
        row_id,
        plan_attempt_id,
    )
    run_dir = outer / row_id
    launch_dir = (
        layout.launch_attempt_dir(results_root, launch_attempt_id) / "rows" / row_id
    )
    submission = _read_json(launch_dir / "submission.json")
    if (
        submission.get("row_id") != row_id
        or submission.get("plan_sha256") != manifest["plan_sha256"]
        or submission.get("evaluation_git_sha") != manifest["evaluation_git_sha"]
        or submission.get("launch_attempt_id") != launch_attempt_id
        or not submission.get("submitted")
        or not submission.get("job_id")
    ):
        raise DiagnosticCollectError("submission receipt does not match the planned row")
    row_receipt = _read_json(outer / "row.json")
    if row_receipt.get("row") != planned:
        raise DiagnosticCollectError("executed row payload differs from the plan")
    if (
        row_receipt.get("plan_sha256") != manifest["plan_sha256"]
        or row_receipt.get("launch_attempt_id") != launch_attempt_id
        or str(row_receipt.get("job_id")) != str(submission["job_id"])
    ):
        raise DiagnosticCollectError("row execution identity does not match submission")
    allocation = _read_json(outer / "allocation_receipt.json")
    if (
        allocation.get("row_id") != row_id
        or not allocation.get("delivered_matches_requested")
        or allocation.get("requested_stratum") != planned["resources"]["stratum"]
        or str(allocation.get("job_id")) != str(submission["job_id"])
    ):
        raise DiagnosticCollectError("allocation receipt does not match the planned stratum")
    binding = _read_json(outer / "checkpoint_binding.json")
    expected_hashes = {
        "model": planned["checkpoint_model_sha256"],
        "manifest": planned["checkpoint_manifest_sha256"],
        "complete": planned["checkpoint_complete_sha256"],
    }
    if (
        binding.get("checkpoint_label") != planned["checkpoint_label"]
        or binding.get("checkpoint_step") != planned["checkpoint_step"]
        or binding.get("hashes") != expected_hashes
        or binding.get("source_git_sha") != planned["checkpoint_source_git_sha"]
    ):
        raise DiagnosticCollectError("checkpoint binding differs from the planned bytes")

    status = _read_json(run_dir / "status.json")
    metadata = _read_json(run_dir / "metadata.json")
    if status.get("status") != "completed" or metadata.get("status") != "completed":
        raise DiagnosticCollectError(
            f"run is not successful: status={status.get('status')!r}, "
            f"metadata={metadata.get('status')!r}"
        )
    if (
        metadata.get("git_commit") != manifest["evaluation_git_sha"]
        or metadata.get("dirty_worktree") is not False
    ):
        raise DiagnosticCollectError("run metadata is not clean exact-SHA provenance")
    failures = run_dir / "diagnostics" / "failures.jsonl"
    if failures.is_file() and failures.read_text(encoding="utf-8").strip():
        raise DiagnosticCollectError("diagnostic failure log is non-empty")

    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    config_sha = resolved["trajectory_identity"]["config_sha256"]
    if (
        resolved["diagnostic"]["protocol"] != planned["protocol"]
        or resolved["diagnostic"]["comparison_kind"] != planned["comparison_kind"]
        or resolved["diagnostic"]["n_walkers"] != planned["n_walkers"]
        or resolved["diagnostic"]["n_draws"] != planned["n_draws"]
        or resolved["diagnostic"]["burn_in"] != planned["burn_in"]
        or resolved["diagnostic"]["stride"] != planned["stride"]
        or resolved["load"]["path"] != binding["checkpoint_dir"]
    ):
        raise DiagnosticCollectError("resolved config differs from the planned row controls")
    replay = _read_json(run_dir / "checkpoint_replay_semantics.json")
    if (
        replay.get("checkpoint_model_sha256") != planned["checkpoint_model_sha256"]
        or replay.get("evaluation_config_sha256") != config_sha
        or replay.get("source_git_sha") != planned["checkpoint_source_git_sha"]
    ):
        raise DiagnosticCollectError("checkpoint replay semantics do not match the row")

    index = _read_json(run_dir / "diagnostics" / "index.json")
    tasks = index.get("tasks")
    if not isinstance(tasks, list):
        raise DiagnosticCollectError("diagnostics index has no task list")
    if [task.get("name") for task in tasks] != list(planned["task_names"]):
        raise DiagnosticCollectError("executed diagnostic task graph differs from the plan")
    if any(task.get("status") != "success" for task in tasks):
        raise DiagnosticCollectError("one or more diagnostic tasks did not succeed")
    artifacts = _reconcile_artifacts(run_dir, tasks)
    _require_profile_artifacts(planned, artifacts)
    metric_records = _read_metrics(run_dir / "metrics.jsonl")
    metric_namespaces = {record["namespace"] for record in metric_records}
    if "eval/perf" not in metric_namespaces or "runtime" not in metric_namespaces:
        raise DiagnosticCollectError("cost/resource metric namespaces are incomplete")

    return {
        "row_id": row_id,
        "checkpoint_label": planned["checkpoint_label"],
        "checkpoint_model_sha256": planned["checkpoint_model_sha256"],
        "profile": planned["profile"],
        "protocol": planned["protocol"],
        "comparison_kind": planned["comparison_kind"],
        "factor_arm": planned.get("factor_arm"),
        "seed": planned["seed"],
        "n_walkers": planned["n_walkers"],
        "n_draws": planned["n_draws"],
        "burn_in": planned["burn_in"],
        "stride": planned["stride"],
        "config_sha256": config_sha,
        "job_id": str(submission["job_id"]),
        "delivered_device": allocation["delivered_device"],
        "task_names": list(planned["task_names"]),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "metrics": metric_records,
    }


def _reconcile_artifacts(
    run_dir: Path,
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_artifacts = task.get("artifacts")
        if not isinstance(task_artifacts, list) or not task_artifacts:
            raise DiagnosticCollectError(
                f"task {task.get('name')!r} published no durable artifacts"
            )
        for artifact in task_artifacts:
            path = Path(str(artifact.get("path")))
            resolved = path.resolve()
            try:
                resolved.relative_to(run_dir.resolve())
            except ValueError as exc:
                raise DiagnosticCollectError(
                    f"artifact path escapes row run directory: {path}"
                ) from exc
            if not path.is_file():
                raise DiagnosticCollectError(f"artifact is missing: {path}")
            records.append(
                {
                    "task": task["name"],
                    "namespace": task["namespace"],
                    "name": artifact["name"],
                    "kind": artifact["kind"],
                    "path": str(path),
                    "sha256": plan_stage.file_sha256(path),
                    "bytes": path.stat().st_size,
                    "metadata": artifact.get("metadata", {}),
                }
            )
    return records


def _require_profile_artifacts(
    planned: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    by_name = {str(artifact["name"]): artifact for artifact in artifacts}
    profile = str(planned["profile"])
    if profile in {"retained_energy", "reequilibrated_energy"}:
        required = {
            "sampled_eval_table",
            "local_energy_trajectory_statistics",
        }
        if profile == "retained_energy":
            required.add("conditioned_local_energy")
        missing = sorted(required - set(by_name))
        if missing:
            raise DiagnosticCollectError(f"energy row artifacts are incomplete: {missing}")
        sampled = by_name["sampled_eval_table"]["metadata"]
        if (
            sampled.get("rows") != planned["record_capacity"]
            or sampled.get("truncated") is not False
            or sampled.get("selection") != "complete_draw_walker_grid"
        ):
            raise DiagnosticCollectError("energy trajectory artifact is not the complete grid")
        trajectory = by_name["local_energy_trajectory_statistics"]["metadata"]
        if (
            trajectory.get("checkpoint_sha256")
            != planned["checkpoint_model_sha256"]
            or trajectory.get("evaluator_id") != "he-v1-diagnostic-v1"
        ):
            raise DiagnosticCollectError("energy/MCSE identity is not checkpoint-aligned")
    elif profile == "common_factor_response":
        artifact = by_name.get("factor_response_common_configuration")
        expected_rows = int(planned["record_capacity"]) * 7
        if artifact is None or artifact["metadata"].get("rows") != expected_rows:
            raise DiagnosticCollectError("common-configuration factor grid is incomplete")
        if artifact["metadata"].get("comparison_kind") != "common_configuration":
            raise DiagnosticCollectError("common factor response is mislabeled")
    elif profile == "checkpoint_diagnostics":
        artifact_tasks = {str(artifact["task"]) for artifact in artifacts}
        missing_tasks = sorted(set(planned["task_names"]) - artifact_tasks)
        if missing_tasks:
            raise DiagnosticCollectError(
                f"checkpoint diagnostic tasks lack records: {missing_tasks}"
            )
    else:
        raise DiagnosticCollectError(f"unknown diagnostic profile {profile!r}")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DiagnosticCollectError(f"metrics stream is missing: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("namespace"), str) or not isinstance(
            record.get("metrics"), Mapping
        ):
            raise DiagnosticCollectError(f"malformed metrics record at line {line_number}")
        records.append(record)
    if not records:
        raise DiagnosticCollectError("metrics stream is empty")
    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticCollectError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticCollectError(f"required JSON artifact is not a mapping: {path}")
    return value


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "row_id",
        "checkpoint_label",
        "checkpoint_model_sha256",
        "profile",
        "protocol",
        "comparison_kind",
        "seed",
        "n_walkers",
        "n_draws",
        "burn_in",
        "stride",
        "config_sha256",
        "job_id",
        "delivered_device",
        "artifact_count",
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    parser.add_argument("--launch-attempt-id", required=True)
    parser.add_argument("--collect-attempt-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    manifest = plan_stage.read_manifest(results_root, args.plan_attempt_id)
    collect(
        manifest,
        results_root=results_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
        collect_attempt_id=args.collect_attempt_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLECT_SCHEMA",
    "DiagnosticCollectError",
    "collect",
    "reconcile_row",
]
