"""Materialize and optionally submit frozen diagnostic rows to Cannon Slurm.

The launcher writes immutable scripts and submission receipts before spending
GPU time. Production rows are constrained to full A100 cards; smoke rows use
the declared ``gpu_test`` MIG stratum without inventing a nonexistent feature.
Both scales invoke the same row driver and task graph.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import diagnostic_plan as plan_stage  # noqa: E402
import layout  # noqa: E402

Submitter = Callable[[Path], str]


class DiagnosticLaunchError(RuntimeError):
    """A planned diagnostic launch cannot be materialized or submitted."""


def select_rows(
    manifest: Mapping[str, Any],
    *,
    row_ids: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Select rows in manifest order and reject every unknown identity."""

    rows = [dict(row) for row in manifest["rows"]]
    if not row_ids:
        return rows
    wanted = list(dict.fromkeys(str(row_id) for row_id in row_ids))
    known = {str(row["row_id"]) for row in rows}
    unknown = [row_id for row_id in wanted if row_id not in known]
    if unknown:
        raise DiagnosticLaunchError(f"row ids are not in this diagnostic plan: {unknown}")
    wanted_set = set(wanted)
    return [row for row in rows if str(row["row_id"]) in wanted_set]


def sbatch_directives(
    row: Mapping[str, Any],
    *,
    job_name: str,
    log_dir: Path,
    account: str | None,
) -> list[str]:
    """Return validated Slurm directives for one declared diagnostic stratum."""

    resources = row["resources"]
    scale_coordinates = (
        str(resources["partition"]),
        str(resources["stratum"]),
        resources.get("constraint"),
        int(resources["timeout_min"]),
        int(resources["cpus"]),
        int(resources["mem_gb"]),
        int(resources["gpus"]),
    )
    if scale_coordinates == ("kozinsky_gpu", "a100", "a100", 720, 4, 32, 1):
        constraint = "a100"
    elif scale_coordinates == ("gpu_test", "a100_mig", None, 120, 4, 32, 1):
        constraint = None
    else:
        raise DiagnosticLaunchError(
            f"row {row['row_id']!r} resources changed from the frozen grid: "
            f"{scale_coordinates!r}"
        )
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources['partition']}",
        f"#SBATCH --gres=gpu:{int(resources['gpus'])}",
        f"#SBATCH --cpus-per-task={int(resources['cpus'])}",
        f"#SBATCH --mem={int(resources['mem_gb'])}G",
        f"#SBATCH --time={_slurm_time(int(resources['timeout_min']))}",
        "#SBATCH --no-requeue",
        f"#SBATCH --output={log_dir / 'slurm-%j.out'}",
        f"#SBATCH --error={log_dir / 'slurm-%j.err'}",
    ]
    if constraint is not None:
        directives.insert(2, f"#SBATCH --constraint={constraint}")
    if account:
        directives.insert(1, f"#SBATCH --account={account}")
    return directives


def driver_command(
    row: Mapping[str, Any],
    *,
    results_root: str | Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
) -> list[str]:
    """Return the scale-independent in-allocation command for one row."""

    return [
        "python",
        str(STUDY_DIR / "diagnostic.py"),
        "--results-root",
        str(results_root),
        "--plan-attempt-id",
        plan_attempt_id,
        "--launch-attempt-id",
        launch_attempt_id,
        "--row-id",
        str(row["row_id"]),
    ]


def build_script(
    row: Mapping[str, Any],
    *,
    directives: Sequence[str],
    command: Sequence[str],
    repo_root: Path,
    evaluation_git_sha: str,
    uv_bin: str,
    uv_extra: str,
    uv_environment_root: str | Path,
    uv_cache_root: str | Path,
) -> str:
    """Build one fail-closed exact-SHA Slurm script."""

    uv_path = Path(uv_bin)
    if not uv_path.is_absolute():
        raise DiagnosticLaunchError(
            f"uv binary must be absolute on Cannon, got {uv_bin!r}"
        )
    if not str(uv_extra).strip():
        raise DiagnosticLaunchError("the CUDA uv extra must be explicit")
    quoted_repo = shlex.quote(str(repo_root))
    quoted_sha = shlex.quote(evaluation_git_sha)
    uv_command = [str(uv_path), "run", "--locked", "--extra", str(uv_extra), *command]
    lines = ["#!/bin/bash", *directives, "", "set -euo pipefail", ""]
    lines.extend(
        [
            'test -n "${SLURM_JOB_ID:-}" || { echo "missing SLURM_JOB_ID" >&2; exit 90; }',
            f"cd {quoted_repo}",
            f'test "$(git rev-parse HEAD)" = {quoted_sha} || '
            '{ echo "evaluator SHA mismatch" >&2; exit 91; }',
            'test -z "$(git status --porcelain --untracked-files=no)" || '
            '{ echo "tracked checkout is dirty" >&2; exit 92; }',
            f"export UV_PROJECT_ENVIRONMENT={shlex.quote(str(uv_environment_root))}/"
            '"${SLURM_JOB_ID}"',
            f"export UV_CACHE_DIR={shlex.quote(str(uv_cache_root))}/"
            '"${SLURM_JOB_ID}"',
            'mkdir -p "${UV_PROJECT_ENVIRONMENT}" "${UV_CACHE_DIR}"',
            f'echo "[he-v1-diagnostic-v1] row={row["row_id"]} '
            f'stratum={row["resources"]["stratum"]} job=${{SLURM_JOB_ID}}"',
            shlex.join(uv_command),
            "",
        ]
    )
    return "\n".join(lines)


def submit_script(path: Path) -> str:
    """Submit one script with ``sbatch --parsable`` and return its job id."""

    completed = subprocess.run(
        ["sbatch", "--parsable", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DiagnosticLaunchError(
            f"sbatch failed for {path} with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise DiagnosticLaunchError(f"sbatch returned no job id for {path}")
    return job_id


def launch(
    manifest: Mapping[str, Any],
    *,
    results_root: str | Path,
    repo_root: str | Path,
    plan_attempt_id: str,
    launch_attempt_id: str,
    rows: Sequence[Mapping[str, Any]],
    uv_bin: str,
    uv_extra: str,
    uv_environment_root: str | Path,
    uv_cache_root: str | Path,
    account: str | None = None,
    submit: bool = False,
    submitter: Submitter = submit_script,
    submit_interval_seconds: float = 0.75,
) -> dict[str, Any]:
    """Write one immutable launch attempt and optionally submit its rows."""

    repo_root = Path(repo_root).resolve()
    _require_repo_sha(repo_root, str(manifest["evaluation_git_sha"]))
    attempt_dir = layout.launch_attempt_dir(results_root, launch_attempt_id)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows):
            row_id = str(row["row_id"])
            row_dir = attempt_dir / "rows" / row_id
            log_dir = row_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=False)
            command = driver_command(
                row,
                results_root=results_root,
                plan_attempt_id=plan_attempt_id,
                launch_attempt_id=launch_attempt_id,
            )
            directives = sbatch_directives(
                row,
                job_name=f"he-diag-{row_id}",
                log_dir=log_dir,
                account=account,
            )
            script = build_script(
                row,
                directives=directives,
                command=command,
                repo_root=repo_root,
                evaluation_git_sha=str(manifest["evaluation_git_sha"]),
                uv_bin=uv_bin,
                uv_extra=uv_extra,
                uv_environment_root=uv_environment_root,
                uv_cache_root=uv_cache_root,
            )
            script_path = row_dir / "sbatch.sh"
            with script_path.open("x", encoding="utf-8") as handle:
                handle.write(script)
            script_path.chmod(0o755)
            job_id = submitter(script_path) if submit else None
            record = {
                "schema": "he-v1-diagnostic-submission/v1",
                "row_id": row_id,
                "plan_attempt_id": plan_attempt_id,
                "plan_sha256": manifest["plan_sha256"],
                "launch_attempt_id": launch_attempt_id,
                "evaluation_git_sha": manifest["evaluation_git_sha"],
                "scale": manifest["scale"],
                "partition": row["resources"]["partition"],
                "requested_stratum": row["resources"]["stratum"],
                "requested_constraint": row["resources"].get("constraint"),
                "script_path": str(script_path),
                "log_dir": str(log_dir),
                "submitted": job_id is not None,
                "job_id": job_id,
                "requeue": False,
                "resume": False,
            }
            _write_new_json(row_dir / "submission.json", record)
            records.append(record)
            if submit and index + 1 < len(rows):
                time.sleep(min(max(float(submit_interval_seconds), 0.5), 1.0))
    except Exception as exc:
        _write_new_json(
            attempt_dir / "launch_failure.json",
            {
                "schema": "he-v1-diagnostic-launch-failure/v1",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "completed_rows": [record["row_id"] for record in records],
            },
        )
        raise
    summary = {
        "schema": "he-v1-diagnostic-launch/v1",
        "study": manifest["study"],
        "scale": manifest["scale"],
        "plan_attempt_id": plan_attempt_id,
        "plan_sha256": manifest["plan_sha256"],
        "launch_attempt_id": launch_attempt_id,
        "evaluation_git_sha": manifest["evaluation_git_sha"],
        "created_at": datetime.now(ZoneInfo(plan_stage.STUDY_TIMEZONE)).isoformat(),
        "submitted": bool(submit),
        "n_rows": len(records),
        "n_submitted": sum(record["submitted"] for record in records),
        "rows": records,
    }
    _write_new_json(attempt_dir / "submissions.json", summary)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_LAUNCH), launch_attempt_id)
    return summary


def _require_repo_sha(repo_root: Path, expected: str) -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or completed.stdout.strip() != expected:
        raise DiagnosticLaunchError(
            f"launch checkout is not evaluator SHA {expected}: "
            f"{completed.stdout.strip() or completed.stderr.strip()}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DiagnosticLaunchError("launch checkout has tracked modifications")


def _slurm_time(minutes: int) -> str:
    if minutes <= 0:
        raise DiagnosticLaunchError("wall time must be positive")
    hours, remainder = divmod(minutes, 60)
    return f"{hours:02d}:{remainder:02d}:00"


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plan-attempt-id", required=True)
    parser.add_argument("--launch-attempt-id", required=True)
    parser.add_argument("--row-id", action="append", default=[])
    parser.add_argument("--uv-bin", required=True)
    parser.add_argument("--uv-extra", required=True)
    parser.add_argument("--uv-environment-root", required=True)
    parser.add_argument("--uv-cache-root", required=True)
    parser.add_argument("--account", default=None)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    manifest = plan_stage.read_manifest(results_root, args.plan_attempt_id)
    rows = select_rows(manifest, row_ids=args.row_id)
    launch(
        manifest,
        results_root=results_root,
        repo_root=args.repo_root,
        plan_attempt_id=args.plan_attempt_id,
        launch_attempt_id=args.launch_attempt_id,
        rows=rows,
        uv_bin=args.uv_bin,
        uv_extra=args.uv_extra,
        uv_environment_root=args.uv_environment_root,
        uv_cache_root=args.uv_cache_root,
        account=args.account,
        submit=args.submit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiagnosticLaunchError",
    "build_script",
    "driver_command",
    "launch",
    "sbatch_directives",
    "select_rows",
    "submit_script",
]
