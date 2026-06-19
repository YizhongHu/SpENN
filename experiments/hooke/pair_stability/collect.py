"""Collect pair-stability validation attempts into a summary table (PR8.8).

Walks ``02_validation/{run_id}/attempts/*`` (the latest attempt per run id),
reads each attempt's status, evaluation metrics, and recorded train-attempt
provenance, and writes a ``03_collect`` attempt with ``summary.csv``,
``failures.csv``, ``collection_report.json``, and explicit source pointers to
the exact validation (and grid) attempts consumed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from orchestrator import (
    STAGE_COLLECT,
    STAGE_GRID,
    STAGE_VALIDATION,
    grid_attempt_dir,
    parse_run_id,
    read_json,
    stage_dir,
    utc_attempt_id,
    validation_run_dir,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

# Physical-correctness evaluation tasks whose presence we record per attempt.
TASK_NAMES = ("cusp", "tail", "stratified_geometry", "hooke_orbital", "energy")
SUCCESS_STATUSES = {"completed", "success"}

CORE_COLUMNS = (
    "run_id",
    "validation_attempt_id",
    "validation_attempt_dir",
    "status",
    "architecture",
    "normalization",
    "lr",
    "channels",
    "seed",
    "train_attempt_id",
    "checkpoint_path",
    "n_diagnostics",
)


def latest_attempt_id(run_dir: Path) -> str | None:
    """Return the most recent attempt id under ``run_dir/attempts``."""

    attempts = run_dir / "attempts"
    if not attempts.is_dir():
        return None
    ids = sorted(child.name for child in attempts.iterdir() if child.is_dir())
    return ids[-1] if ids else None


def read_metrics_jsonl(path: Path) -> dict[str, Any]:
    """Flatten ``metrics.jsonl`` records into ``namespace/key -> value``."""

    metrics: dict[str, Any] = {}
    if not path.is_file():
        return metrics
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        namespace = str(record.get("namespace", "")).strip("/")
        values = record.get("metrics", {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            metrics[f"{namespace}/{key}" if namespace else str(key)] = value
    return metrics


def _status_of(attempt_dir: Path) -> str:
    status = attempt_dir / "status.json"
    if not status.is_file():
        return "missing_status"
    return str(read_json(status).get("status", "unknown"))


def _run_parameters(attempt_dir: Path, run_id: str) -> dict[str, Any]:
    """Recover run parameters from resolved_config.yaml, falling back to run id."""

    resolved = attempt_dir / "resolved_config.yaml"
    if resolved.is_file():
        cfg = OmegaConf.load(resolved)
        params = OmegaConf.select(cfg, "run_parameters")
        if params is not None:
            return OmegaConf.to_container(params, resolve=True)
    return parse_run_id(run_id)


def _count_diagnostics(attempt_dir: Path) -> int:
    """Count evaluation task outputs recorded for this attempt."""

    index = attempt_dir / "diagnostics" / "index.json"
    if index.is_file():
        data = read_json(index)
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            return len(data["artifacts"])
        if isinstance(data, list):
            return len(data)
    return sum(1 for name in TASK_NAMES if (attempt_dir / name).is_dir())


def collect_validation_attempt(run_id: str, attempt_id: str, attempt_dir: Path) -> dict[str, Any]:
    """Build one summary row from a validation attempt directory."""

    params = _run_parameters(attempt_dir, run_id)
    source_path = attempt_dir / "source_train_attempt.json"
    source = read_json(source_path) if source_path.is_file() else {}
    metrics = read_metrics_jsonl(attempt_dir / "metrics.jsonl")

    row: dict[str, Any] = {column: "" for column in CORE_COLUMNS}
    row.update(
        run_id=run_id,
        validation_attempt_id=attempt_id,
        validation_attempt_dir=str(attempt_dir),
        status=_status_of(attempt_dir),
        architecture=params.get("architecture", ""),
        normalization=params.get("normalization", ""),
        lr=params.get("lr", ""),
        channels=params.get("channels", ""),
        seed=params.get("seed", ""),
        train_attempt_id=source.get("train_attempt_id", ""),
        checkpoint_path=source.get("checkpoint_path", ""),
        n_diagnostics=_count_diagnostics(attempt_dir),
    )
    row.update(metrics)
    return row


def _resolve_grid_attempt(results_root: Path, grid_attempt_id: str | None) -> str | None:
    if grid_attempt_id is not None:
        return grid_attempt_id
    latest = stage_dir(results_root, STAGE_GRID) / "latest.json"
    if latest.is_file():
        return str(read_json(latest).get("attempt_id"))
    return None


def _run_ids(results_root: Path, grid_attempt_id: str | None) -> list[str]:
    """Return run ids to collect, from the grid manifest if available."""

    if grid_attempt_id is not None:
        manifest = read_json(grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json")
        return [str(job["run_id"]) for job in manifest.get("jobs", [])]
    validation_root = stage_dir(results_root, STAGE_VALIDATION)
    if not validation_root.is_dir():
        return []
    return sorted(child.name for child in validation_root.iterdir() if (child / "attempts").is_dir())


def collect(
    *,
    results_root: str | Path,
    collect_attempt_id: str | None = None,
    grid_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Collect validation attempts and write a ``03_collect`` attempt."""

    results_root = Path(results_root)
    collect_attempt_id = collect_attempt_id or utc_attempt_id()
    grid_attempt_id = _resolve_grid_attempt(results_root, grid_attempt_id)

    rows: list[dict[str, Any]] = []
    consumed: list[dict[str, Any]] = []
    for run_id in _run_ids(results_root, grid_attempt_id):
        run_dir = validation_run_dir(results_root, run_id)
        attempt_id = latest_attempt_id(run_dir)
        if attempt_id is None:
            continue
        attempt_dir = run_dir / "attempts" / attempt_id
        rows.append(collect_validation_attempt(run_id, attempt_id, attempt_dir))
        consumed.append(
            {"run_id": run_id, "validation_attempt_id": attempt_id, "validation_attempt_dir": str(attempt_dir)}
        )

    attempt = stage_dir(results_root, STAGE_COLLECT) / "attempts" / collect_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    metric_columns = sorted({key for row in rows for key in row if key not in CORE_COLUMNS})
    columns = list(CORE_COLUMNS) + metric_columns
    failures = [row for row in rows if str(row["status"]) not in SUCCESS_STATUSES]

    _write_csv(attempt / "summary.csv", columns, rows)
    _write_csv(attempt / "failures.csv", columns, failures)
    write_json(attempt / "source_grid_attempt.json", {"grid_attempt_id": grid_attempt_id})
    write_json(attempt / "source_validation_attempts.json", consumed)
    report = {
        "study": "pair_stability",
        "stage": STAGE_COLLECT,
        "attempt_id": collect_attempt_id,
        "grid_attempt_id": grid_attempt_id,
        "n_collected": len(rows),
        "n_failures": len(failures),
        "metric_columns": metric_columns,
    }
    write_json(attempt / "collection_report.json", report)
    return {"attempt_dir": str(attempt), "report": report, "rows": rows}


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse collect command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt whose run ids to collect.")
    parser.add_argument("--attempt-id", default=None, help="Collect attempt id (defaults to now).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect validation attempts from the command line."""

    args = parse_args(argv)
    result = collect(
        results_root=args.results_root,
        collect_attempt_id=args.attempt_id,
        grid_attempt_id=args.grid_attempt_id,
    )
    report = result["report"]
    print(
        f"[pair_stability] collected {report['n_collected']} runs "
        f"({report['n_failures']} failures) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
