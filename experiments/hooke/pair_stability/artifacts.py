"""Study-local artifact readers and writers for pair-stability stages."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def csv_value(value: Any) -> Any:
    """Return ``value`` in the scalar/JSON form used by study CSV files."""

    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a CSV file into dictionaries, returning an empty list if absent."""

    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write rows to CSV with stable columns and ignored extra fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns is not None else sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def load_json_if_present(path: Path, default: Any = None) -> Any:
    """Read JSON from ``path`` when present, otherwise return ``default``."""

    if not path.is_file():
        return default
    return json.loads(path.read_text())


def load_json_dict_if_present(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path`` when present, otherwise return ``{}``."""

    payload = load_json_if_present(path, {})
    return payload if isinstance(payload, dict) else {}


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    """Expand a metrics JSONL file into ``step, namespace, metric, value`` rows."""

    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        namespace = str(record.get("namespace", "")).strip("/")
        step = record.get("step", "")
        if "metrics" in record:
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                continue
            items = metrics.items()
        elif "metric" in record and "value" in record:
            items = [(record["metric"], record["value"])]
        else:
            continue
        for key, value in items:
            rows.append({"step": step, "namespace": namespace, "metric": str(key), "value": csv_value(value)})
    return rows


def metric_key(namespace: Any, metric: Any) -> str:
    """Return the public ``namespace/metric`` key for one metric row."""

    namespace_text = str(namespace).strip("/")
    metric_text = str(metric)
    return f"{namespace_text}/{metric_text}" if namespace_text else metric_text


def metric_map(rows: Sequence[Mapping[str, Any]], *, prefix: str | None = None) -> dict[str, Any]:
    """Return ``namespace/metric -> value`` entries from metric rows."""

    output: dict[str, Any] = {}
    for row in rows:
        key = metric_key(row.get("namespace", ""), row.get("metric", ""))
        if prefix:
            key = f"{prefix}/{key}" if key else prefix
        output[key] = row.get("value")
    return output


def read_metrics_map(path: Path, *, prefix: str | None = None) -> dict[str, Any]:
    """Read metrics JSONL and return its ``namespace/metric`` map."""

    return metric_map(read_metrics_jsonl(path), prefix=prefix)


def parse_time(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning ``None`` for missing or malformed text."""

    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def duration_from_status(status: Mapping[str, Any], *, clamp_negative: bool = False) -> float | None:
    """Return elapsed seconds from a status mapping's start/end timestamps."""

    start = parse_time(status.get("start_time"))
    end = parse_time(status.get("end_time"))
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    if seconds < 0:
        return 0.0 if clamp_negative else None
    return seconds


def duration_from_status_file(path: Path, *, clamp_negative: bool = False) -> float | None:
    """Return elapsed seconds from ``path/status.json`` or a status file path."""

    status_path = Path(path)
    if status_path.name != "status.json":
        status_path = status_path / "status.json"
    status = load_json_dict_if_present(status_path)
    return duration_from_status(status, clamp_negative=clamp_negative)


def status_of(attempt_dir: Path) -> str:
    """Return an attempt directory's recorded status."""

    status_path = attempt_dir / "status.json"
    if not status_path.is_file():
        return "missing_status"
    return str(load_json_dict_if_present(status_path).get("status", "unknown"))
