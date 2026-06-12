#!/usr/bin/env python
"""Collect Hooke pair validation-scan run outputs into a normalized table.

Reads completed (and failed/incomplete) run directories and produces
``runs.csv`` and ``runs.jsonl``. The collector only normalizes local run
outputs; it never selects a winner, never reads W&B, and never imports
``spenn``.

Inputs
------
manifest.yaml
    Study protocol (grid keys, metric names).
run roots
    One or more directories scanned recursively for run directories. A run
    directory is any directory containing ``metadata.json``.

Outputs
-------
runs.csv, runs.jsonl
    One row per run directory with status, hyperparameters, validation
    metrics, sampler geometry diagnostics, and provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

# Metric namespaces copied into the normalized table. Everything else in
# metrics.jsonl stays in the run directory.
_METRIC_PREFIXES = ("validation/",)
_EXTRA_METRICS = ("runtime/wall_time_sec",)

_BASE_COLUMNS = ("run_dir", "status", "study_name", "config_id")
_PROVENANCE_COLUMNS = ("git/sha", "wandb/run_id")


def load_manifest(path: Path) -> dict:
    """Load and minimally validate the study manifest."""

    with open(path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    for required in ("study", "grid", "seed_key", "validation"):
        if required not in manifest:
            raise ValueError(f"manifest {path} is missing the {required!r} section")
    if manifest["seed_key"] not in manifest["grid"]:
        raise ValueError(f"manifest seed_key {manifest['seed_key']!r} is not a grid key")
    return manifest


def grid_keys(manifest: dict) -> list[str]:
    """Return the grid axes in manifest declaration order."""

    return list(manifest["grid"])


def group_keys(manifest: dict) -> list[str]:
    """Return the non-seed grid axes that define a config group."""

    seed_key = manifest["seed_key"]
    return [key for key in grid_keys(manifest) if key != seed_key]


def format_value(value: object) -> str:
    """Format one hyperparameter value deterministically for config IDs."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def config_id_from_values(manifest: dict, values: dict[str, object]) -> str:
    """Build the deterministic config ID for one non-seed grid point.

    Uses the last dotted component of each non-seed grid key, in manifest
    order, e.g. ``lr=0.001_channels=32_layers=1_gate_activation=silu``.
    """

    parts = []
    for key in group_keys(manifest):
        short = key.rsplit(".", 1)[-1]
        parts.append(f"{short}={format_value(values.get(key))}")
    return "_".join(parts)


def lookup_dotted(config: dict, dotted_key: str) -> object:
    """Resolve a dotted path like ``optimizer_params.lr`` in a nested mapping."""

    node: object = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def read_last_metrics(metrics_jsonl: Path) -> dict[str, object]:
    """Return the last logged value per flat ``namespace/key`` metric path."""

    metrics: dict[str, object] = {}
    if not metrics_jsonl.is_file():
        return metrics
    with open(metrics_jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed run is not fatal
            namespace = str(record.get("namespace", "")).strip("/")
            payload = record.get("metrics")
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                flat = f"{namespace}/{key}" if namespace else str(key)
                metrics[flat] = value
    return metrics


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (yaml.YAMLError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _run_status(status_file: dict, metrics: dict[str, object], validation_metric: str) -> str:
    """Classify a run as completed, failed, or incomplete.

    A run only counts as completed when its lifecycle finished cleanly *and*
    the validation metric was actually logged.
    """

    status = str(status_file.get("status", "")).lower()
    if status in ("failed", "exception", "error"):
        return "failed"
    if status == "completed" and validation_metric in metrics:
        return "completed"
    return "incomplete"


def collect_run(run_dir: Path, manifest: dict) -> dict[str, object]:
    """Normalize one run directory into a flat row."""

    resolved = _read_yaml(run_dir / "resolved_config.yaml")
    metadata = _read_json(run_dir / "metadata.json")
    status_file = _read_json(run_dir / "status.json")
    metrics = read_last_metrics(run_dir / "metrics.jsonl")

    validation_metric = str(manifest["validation"]["metric"])
    row: dict[str, object] = {
        "run_dir": str(run_dir),
        "status": _run_status(status_file, metrics, validation_metric),
        "study_name": lookup_dotted(resolved, "study.name") or "",
    }

    grid_values = {key: lookup_dotted(resolved, key) for key in grid_keys(manifest)}
    row.update(grid_values)
    recorded_id = lookup_dotted(resolved, "study.config_id")
    row["config_id"] = recorded_id or config_id_from_values(manifest, grid_values)

    eligibility = manifest.get("eligibility", {})
    wanted_checks = tuple(eligibility.get("require", ()))
    for flat, value in metrics.items():
        if flat.startswith(_METRIC_PREFIXES) or flat in _EXTRA_METRICS or flat in wanted_checks:
            row[flat] = value

    row["git/sha"] = metadata.get("git_commit") or ""
    row["wandb/run_id"] = metadata.get("wandb_run_id") or ""
    return row


def discover_run_dirs(roots: list[Path]) -> list[Path]:
    """Find run directories (holding metadata.json) under the given roots."""

    found: set[Path] = set()
    for root in roots:
        if (root / "metadata.json").is_file():
            found.add(root)
            continue
        found.update(path.parent for path in root.rglob("metadata.json"))
    return sorted(found)


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return str(value)


def write_outputs(rows: list[dict[str, object]], manifest: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write runs.csv and runs.jsonl with a deterministic column order."""

    metric_columns = sorted(
        {key for row in rows for key in row}
        - set(_BASE_COLUMNS)
        - set(_PROVENANCE_COLUMNS)
        - set(grid_keys(manifest))
    )
    columns = [*_BASE_COLUMNS, *grid_keys(manifest), *metric_columns, *_PROVENANCE_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runs.csv"
    jsonl_path = output_dir / "runs.jsonl"

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row.get(column)) for column in columns])

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")

    return csv_path, jsonl_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifest.yaml",
        help="Study manifest path.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="Run root (or single run directory). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for runs.csv and runs.jsonl.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    run_dirs = discover_run_dirs(args.run_root)
    if not run_dirs:
        print(f"no run directories found under {[str(r) for r in args.run_root]}", file=sys.stderr)
        return 1

    rows = [collect_run(run_dir, manifest) for run_dir in run_dirs]
    csv_path, jsonl_path = write_outputs(rows, manifest, args.output_dir)

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    print(f"collected {len(rows)} runs {by_status} -> {csv_path}, {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
