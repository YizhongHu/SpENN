"""Collect compact final pair-stability summaries from raw final artifacts.

``final_collect.py`` is the only final-reporting stage that reads raw training
and evaluation artifacts. It consumes ``05_final_grid``, ``06_final_train``,
and ``07_final_eval`` provenance and reduces them into compact, reusable CSV
tables under ``08_final_collect``. Plotting/reporting code should consume these
tables rather than reparsing raw task records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from run_utils import (
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_EVAL,
    attempt_ids,
    new_attempt_id,
    read_json,
    stage_dir,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
EXACT_HOOKE_ENERGY = 2.0
EXPECTED_FINAL_SEEDS = 10
DEFAULT_HISTOGRAM_BINS = 32
PATHOLOGY_ABS_LOCAL_ENERGY = 10.0

COMPACT_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
)

RUN_INDEX_COLUMNS = [
    "final_run_id",
    "source_champion_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "model_seed",
    "sampler_seed",
    "eval_seed",
    "train_status",
    "eval_status",
    "n_eval_tasks_success",
    "n_eval_tasks_failed",
    "train_wall_time_sec",
    "eval_wall_time_sec",
]

ARCHITECTURE_SUMMARY_COLUMNS = [
    "basis_class",
    "normalization",
    "winner_kind",
    "n_success",
    "n_expected",
    "energy_error_median",
    "energy_error_q25",
    "energy_error_q75",
    "local_energy_var_median",
    "pathology_fraction_median",
    "tail_outlier_fraction_median",
    "trace_failure_count_total",
    "major_failure_mode",
]

ENERGY_BY_RUN_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "energy_mean",
    "energy_stderr",
    "energy_error",
    "local_energy_var",
    "finite_fraction",
    "pathology_fraction",
]

LOCAL_ENERGY_HISTOGRAM_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "count",
    "density",
]

CUSP_PROFILE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_id",
    "direction_id",
    "r12",
    "even_slope_median",
    "even_slope_q25",
    "even_slope_q75",
    "c_minus_1_median",
    "c_minus_1_q25",
    "c_minus_1_q75",
    "odd_slant_median",
    "odd_slant_q25",
    "odd_slant_q75",
    "finite_fraction",
]

TAIL_PROFILE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_id",
    "tail_path",
    "radius",
    "local_energy_median",
    "local_energy_q25",
    "local_energy_q75",
    "logabs_median",
    "logabs_q25",
    "logabs_q75",
    "exact_logabs",
    "outlier_fraction",
    "finite_fraction",
]

STRATIFIED_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "stratum",
    "local_energy_median",
    "local_energy_var",
    "abs_local_energy_q95",
    "abs_local_energy_q99",
    "pathology_fraction",
    "finite_fraction",
    "median_abs_energy_error",
]

HOOKE_ORBITAL_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_bin",
    "r12_bin",
    "r12_center",
    "R_norm_center",
    "local_energy_median",
    "local_energy_q25",
    "local_energy_q75",
    "abs_energy_error_median",
    "finite_fraction",
    "pathology_fraction",
]

SYMMETRY_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "symmetry_task",
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
    "finite_fraction",
]

TRACE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "trace_kind",
    "layer",
    "key",
    "rms_q95",
    "rms_q99",
    "max_abs",
    "nonfinite_count",
    "compared_entry_count",
    "comparison_error_count",
    "max_equivariance_error",
]

TRAINING_CURVE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "step",
    "energy_mean",
    "energy_stderr",
    "local_energy_var",
    "acceptance_rate",
    "grad_norm",
    "wall_time_sec",
]

RESOURCE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "peak_memory_mb",
    "device_type",
]

FAILURE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "task",
    "severity",
    "failure_mode",
    "metric_name",
    "metric_value",
    "threshold",
    "geometry_context",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        namespace = str(record.get("namespace", "")).strip("/")
        metrics = record.get("metrics", {})
        step = record.get("step", "")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            rows.append({"step": step, "namespace": namespace, "metric": str(key), "value": _csv_value(value)})
    return rows


def _metric_map(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {f"{row['namespace']}/{row['metric']}": row["value"] for row in rows}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _clean(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def _mean(values: Iterable[float | None]) -> float | None:
    clean = _clean(values)
    return statistics.fmean(clean) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = _clean(values)
    return statistics.median(clean) if clean else None


def _variance(values: Iterable[float | None]) -> float | None:
    clean = _clean(values)
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.variance(clean)


def _quantile(values: Iterable[float | None], q: float) -> float | None:
    clean = sorted(_clean(values))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[int(position)]
    weight = position - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _sum(values: Iterable[float | None]) -> float | None:
    clean = _clean(values)
    return math.fsum(clean) if clean else None


def _max(values: Iterable[float | None]) -> float | None:
    clean = _clean(values)
    return max(clean) if clean else None


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _duration_from_status(status: dict[str, Any]) -> float | None:
    start = _parse_time(status.get("start_time"))
    end = _parse_time(status.get("end_time"))
    if start is None or end is None:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _status_of(attempt_dir: Path) -> str:
    status = attempt_dir / "status.json"
    if not status.is_file():
        return "missing_status"
    return str(read_json(status).get("status", "unknown"))


def _iter_final_eval_attempts(results_root: Path, final_eval_attempt_id: str | None) -> list[tuple[str, str, Path]]:
    eval_stage = stage_dir(results_root, STAGE_FINAL_EVAL)
    if not eval_stage.is_dir():
        return []
    attempts = []
    for run_dir in sorted(child for child in eval_stage.iterdir() if child.is_dir()):
        if run_dir.name in {"slurm_logs", "chunk_status"}:
            continue
        attempt_id = final_eval_attempt_id
        if attempt_id is None:
            ids = attempt_ids(run_dir)
            if not ids:
                continue
            attempt_id = ids[-1]
        attempt_dir = run_dir / attempt_id
        if attempt_dir.is_dir():
            attempts.append((run_dir.name, attempt_id, attempt_dir))
    return attempts


def _winner_kind(job: dict[str, Any]) -> str:
    raw = str(job.get("winner_kind", ""))
    return "energy" if raw == "energy" else "stability"


def _basis_class(job: dict[str, Any]) -> str:
    return str(job.get("basis_envelope", job.get("architecture", "")))


def _seed_index(job: dict[str, Any]) -> str:
    return str(job.get("replicate_index", ""))


def _minor_hparams(job: dict[str, Any]) -> str:
    keys = ("architecture", "lr", "channels")
    return ";".join(f"{key}={job.get(key, '')}" for key in keys if job.get(key, "") != "")


def _run_context(final_run_id: str, attempt_id: str, attempt_dir: Path) -> dict[str, Any]:
    job = _load_json_if_present(attempt_dir / "source_final_job.json")
    checkpoint = _load_json_if_present(attempt_dir / "evaluated_checkpoint.json")
    source_train = _load_json_if_present(attempt_dir / "source_final_train_attempt.json")
    eval_status_json = _load_json_if_present(attempt_dir / "status.json")
    train_attempt_dir = Path(str(source_train.get("final_train_attempt_dir", "")))
    train_status_json = _load_json_if_present(train_attempt_dir / "status.json")
    train_metadata = _load_json_if_present(train_attempt_dir / "metadata.json")
    eval_metrics = _read_metrics_jsonl(attempt_dir / "metrics.jsonl")
    train_metrics = _read_metrics_jsonl(train_attempt_dir / "metrics.jsonl")
    return {
        "final_run_id": final_run_id,
        "attempt_id": attempt_id,
        "attempt_dir": attempt_dir,
        "job": job,
        "checkpoint": checkpoint,
        "source_train": source_train,
        "train_attempt_dir": train_attempt_dir,
        "train_status_json": train_status_json,
        "train_metadata": train_metadata,
        "eval_status": str(eval_status_json.get("status", _status_of(attempt_dir))),
        "eval_status_json": eval_status_json,
        "eval_metrics": eval_metrics,
        "eval_metric_map": _metric_map(eval_metrics),
        "train_metrics": train_metrics,
    }


def _base_row(context: dict[str, Any]) -> dict[str, Any]:
    job = context["job"]
    return {
        "final_run_id": context["final_run_id"],
        "source_champion_id": job.get("source_champion_id", ""),
        "basis_class": _basis_class(job),
        "normalization": job.get("normalization", ""),
        "winner_kind": _winner_kind(job),
        "seed_index": _seed_index(job),
        "model_seed": job.get("final_train_model_seed", ""),
        "sampler_seed": job.get("final_train_sampler_seed", ""),
        "eval_seed": job.get("final_eval_seed", ""),
    }


def _finite_fraction(values: Sequence[Any]) -> float | None:
    if not values:
        return None
    finite = sum(1 for value in values if str(value).lower() != "false")
    return finite / len(values)


def _is_pathological(row: dict[str, Any]) -> bool:
    finite = str(row.get("finite", "True")).lower() == "true"
    local_energy = _as_float(row.get("local_energy"))
    return (not finite) or (local_energy is not None and abs(local_energy) > PATHOLOGY_ABS_LOCAL_ENERGY)


def _task_from_namespace(namespace: str) -> str:
    if namespace.startswith("eval/"):
        return namespace[len("eval/") :]
    return namespace


def _failure_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    base = _base_row(context)
    failures: list[dict[str, Any]] = []
    status = context["eval_status"]
    if status not in {"completed", "success"}:
        failures.append({**base, "task": "run", "severity": "failed", "failure_mode": "run_status", "metric_name": "status", "metric_value": status, "threshold": "completed", "geometry_context": ""})
    for metric_name, value in context["eval_metric_map"].items():
        namespace, _, key = metric_name.rpartition("/")
        task = _task_from_namespace(namespace)
        bool_value = _as_bool(value)
        numeric = _as_float(value)
        if key == "task_failed" and bool_value:
            failures.append({**base, "task": task, "severity": "failed", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "False", "geometry_context": ""})
        if numeric is None:
            continue
        failure_key = any(needle in key for needle in ("failure_count", "nonfinite_count", "pathology_count", "outlier_count", "mismatch_count", "comparison_error_count", "missing_key_count", "extra_key_count", "near_zero_count"))
        if failure_key and numeric > 0:
            failures.append({**base, "task": task, "severity": "warning", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "0", "geometry_context": ""})
        if key.endswith("finite_fraction") and numeric < 1.0:
            failures.append({**base, "task": task, "severity": "warning", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "1.0", "geometry_context": ""})
    return failures


def _major_failure_mode(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    failed = [row for row in rows if row.get("severity") == "failed"]
    row = failed[0] if failed else rows[0]
    return f"{row.get('task', '')}:{row.get('failure_mode', '')}"


def _task_status_counts(metrics: dict[str, Any]) -> tuple[int, int]:
    success = 0
    failed = 0
    for key, value in metrics.items():
        if key.endswith("/status/task_success") and _as_bool(value):
            success += 1
        if key.endswith("/status/task_failed") and _as_bool(value):
            failed += 1
    return success, failed


def _run_index_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    checkpoint = context["checkpoint"]
    source_train = context["source_train"]
    train_status = "checkpoint_selected" if checkpoint.get("resolved_checkpoint_dir") else ("attempt_recorded" if source_train.get("final_train_attempt_id") else "")
    n_success, n_failed = _task_status_counts(context["eval_metric_map"])
    return {
        **base,
        "train_status": train_status,
        "eval_status": context["eval_status"],
        "n_eval_tasks_success": n_success,
        "n_eval_tasks_failed": n_failed,
        "train_wall_time_sec": _format_number(_duration_from_status(context["train_status_json"])),
        "eval_wall_time_sec": _format_number(_duration_from_status(context["eval_status_json"])),
    }


def _energy_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    metrics = context["eval_metric_map"]
    energy = _as_float(metrics.get("eval/energy/local_energy_mean"))
    n_finite = _as_float(metrics.get("eval/energy/local_energy_n_finite"))
    n_total = _as_float(metrics.get("eval/energy/local_energy_n_total"))
    finite_fraction = _as_float(metrics.get("eval/energy/local_energy_finite_fraction"))
    if finite_fraction is None and n_finite is not None and n_total:
        finite_fraction = n_finite / n_total
    pathology_count = _as_float(metrics.get("eval/energy/local_energy_pathology_count"))
    pathology_fraction = None
    if pathology_count is not None and n_total:
        pathology_fraction = pathology_count / n_total
    return {
        **base,
        "energy_mean": _format_number(energy),
        "energy_stderr": _format_number(_as_float(metrics.get("eval/energy/local_energy_stderr"))),
        "energy_error": _format_number(None if energy is None else energy - EXACT_HOOKE_ENERGY),
        "local_energy_var": _format_number(_as_float(metrics.get("eval/energy/local_energy_variance"))),
        "finite_fraction": _format_number(finite_fraction),
        "pathology_fraction": _format_number(pathology_fraction),
    }


def _bin_edges(values: Sequence[float], n_bins: int = DEFAULT_HISTOGRAM_BINS) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        low -= pad
        high += pad
    width = (high - low) / n_bins
    return [low + index * width for index in range(n_bins + 1)]


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[tuple[float, float, int, float]]:
    if not values or len(edges) < 2:
        return []
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        if value < edges[0] or value > edges[-1]:
            continue
        index = len(edges) - 2 if value == edges[-1] else max(0, min(len(edges) - 2, int((value - edges[0]) / (edges[-1] - edges[0]) * (len(edges) - 1))))
        counts[index] += 1
    total = sum(counts)
    rows = []
    for index, count in enumerate(counts):
        left = edges[index]
        right = edges[index + 1]
        width = right - left
        density = 0.0 if total == 0 or width <= 0 else count / (total * width)
        rows.append((left, right, count, density))
    return rows


def _record_context(context: dict[str, Any], task: str, row: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    out = {**base, **row, "task": task}
    if "radius" in out:
        out["R_norm"] = out.get("radius", "")
        out["R_norm_bin"] = _bin_value(out.get("radius"))
    if "r12" in out:
        out["r12_bin"] = _bin_value(out.get("r12"))
    if "center_of_mass_id" in out:
        out["com_id"] = out.get("center_of_mass_id", "")
    return out


def _bin_value(value: Any, *, width: float = 0.5) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    low = math.floor(numeric / width) * width
    high = low + width
    return f"[{low:.2g},{high:.2g})"


def _center_of_bin(label: str) -> str:
    if not label.startswith("[") or "," not in label:
        return ""
    left, right = label.strip(")").strip("[").split(",", 1)
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return ""
    return _format_number((left_f + right_f) / 2.0)


def _task_records(context: dict[str, Any], task: str, filename: str) -> list[dict[str, Any]]:
    return [_record_context(context, task, row) for row in _read_csv(context["attempt_dir"] / task / filename)]


def _local_energy_values(contexts: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    values_by_run: dict[str, list[float]] = {}
    for context in contexts:
        rows = _read_csv(context["attempt_dir"] / "energy" / "mcmc_energy_samples.csv")
        values_by_run[context["final_run_id"]] = [value for value in (_as_float(row.get("local_energy")) for row in rows) if value is not None]
    return values_by_run


def _local_energy_histograms(contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values_by_run = _local_energy_values(contexts)
    all_values = [value for values in values_by_run.values() for value in values]
    edges = _bin_edges(all_values)
    rows = []
    for context in contexts:
        base = _base_row(context)
        for left, right, count, density in _histogram(values_by_run.get(context["final_run_id"], []), edges):
            rows.append({**base, "bin_left": _format_number(left), "bin_right": _format_number(right), "bin_center": _format_number((left + right) / 2.0), "count": count, "density": _format_number(density)})
    return rows


def _cusp_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "cusp", "cusp_profiles.csv")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("com_id", row.get("center_of_mass_id", ""))), str(row.get("direction_id", "")), str(row.get("r12", "")))].append(row)
    out = []
    base = _base_row(context)
    for (com_id, direction_id, r12), group in sorted(groups.items()):
        out.append({
            **base,
            "com_id": com_id,
            "direction_id": direction_id,
            "r12": r12,
            "even_slope_median": _format_number(_median(_as_float(row.get("even_slope")) for row in group)),
            "even_slope_q25": _format_number(_quantile((_as_float(row.get("even_slope")) for row in group), 0.25)),
            "even_slope_q75": _format_number(_quantile((_as_float(row.get("even_slope")) for row in group), 0.75)),
            "c_minus_1_median": _format_number(_median(_as_float(row.get("c_minus_1")) for row in group)),
            "c_minus_1_q25": _format_number(_quantile((_as_float(row.get("c_minus_1")) for row in group), 0.25)),
            "c_minus_1_q75": _format_number(_quantile((_as_float(row.get("c_minus_1")) for row in group), 0.75)),
            "odd_slant_median": _format_number(_median(_as_float(row.get("odd_slant")) for row in group)),
            "odd_slant_q25": _format_number(_quantile((_as_float(row.get("odd_slant")) for row in group), 0.25)),
            "odd_slant_q75": _format_number(_quantile((_as_float(row.get("odd_slant")) for row in group), 0.75)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
        })
    return out


def _tail_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "tail", "tail_profiles.csv")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tail_path = str(row.get("tail_path", f"direction-{row.get('direction_id', '')}/relative-{row.get('relative_direction_id', '')}"))
        groups[(str(row.get("com_id", "")), tail_path, str(row.get("radius", "")))].append(row)
    out = []
    base = _base_row(context)
    for (com_id, tail_path, radius), group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        logabs = [_as_float(row.get("logabs")) for row in group]
        out.append({
            **base,
            "com_id": com_id,
            "tail_path": tail_path,
            "radius": radius,
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_q25": _format_number(_quantile(energies, 0.25)),
            "local_energy_q75": _format_number(_quantile(energies, 0.75)),
            "logabs_median": _format_number(_median(logabs)),
            "logabs_q25": _format_number(_quantile(logabs, 0.25)),
            "logabs_q75": _format_number(_quantile(logabs, 0.75)),
            "exact_logabs": next((row.get("exact_logabs", "") for row in group if row.get("exact_logabs", "") != ""), ""),
            "outlier_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
        })
    return out


def _stratified_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "stratified_geometry", "stratified_metrics.csv")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("stratum", ""))].append(row)
        groups["all"].append(row)
    out = []
    base = _base_row(context)
    for stratum, group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        abs_energies = [None if value is None else abs(value) for value in energies]
        abs_errors = [None if value is None else abs(value - EXACT_HOOKE_ENERGY) for value in energies]
        out.append({
            **base,
            "stratum": stratum,
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_var": _format_number(_variance(energies)),
            "abs_local_energy_q95": _format_number(_quantile(abs_energies, 0.95)),
            "abs_local_energy_q99": _format_number(_quantile(abs_energies, 0.99)),
            "pathology_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
            "median_abs_energy_error": _format_number(_median(abs_errors)),
        })
    return out


def _hooke_orbital_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "hooke_orbital", "hooke_orbital_metrics.csv")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("R_norm_bin", _bin_value(row.get("radius")))), str(row.get("r12_bin", _bin_value(row.get("r12")))))].append(row)
    out = []
    base = _base_row(context)
    for (com_bin, r12_bin), group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        abs_errors = [None if value is None else abs(value - EXACT_HOOKE_ENERGY) for value in energies]
        out.append({
            **base,
            "com_bin": com_bin,
            "r12_bin": r12_bin,
            "r12_center": _center_of_bin(r12_bin),
            "R_norm_center": _center_of_bin(com_bin),
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_q25": _format_number(_quantile(energies, 0.25)),
            "local_energy_q75": _format_number(_quantile(energies, 0.75)),
            "abs_energy_error_median": _format_number(_median(abs_errors)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
            "pathology_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
        })
    return out


def _symmetry_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base = _base_row(context)
    for task in ("full_model_antisymmetry", "spatial_exchange_symmetry", "rotation_consistency"):
        rows = _task_records(context, task, "transform_records.csv")
        if not rows:
            continue
        errors = [_as_float(row.get("logabs_abs_error", row.get("logabs_error", row.get("max_abs_error")))) for row in rows]
        out.append({
            **base,
            "symmetry_task": task,
            "logabs_error_max": _format_number(_max(errors)),
            "logabs_error_median": _format_number(_median(errors)),
            "sign_mismatch_count": sum(1 for row in rows if _as_bool(row.get("sign_mismatch"))),
            "parity_mismatch_count": sum(1 for row in rows if _as_bool(row.get("parity_mismatch"))),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in rows])),
        })
    return out


def _trace_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base = _base_row(context)
    metrics = context["eval_metric_map"]
    for task in ("trace_equivariance", "feature_trace_stability", "readout_trace_stability"):
        rows = _task_records(context, task, "trace_records.csv")
        if not rows and task == "trace_equivariance":
            key = ""
            out.append({**base, "trace_kind": task, "layer": "", "key": key, "rms_q95": "", "rms_q99": "", "max_abs": "", "nonfinite_count": "", "compared_entry_count": metrics.get("eval/trace_equivariance/compared_entry_count", ""), "comparison_error_count": metrics.get("eval/trace_equivariance/comparison_error_count", ""), "max_equivariance_error": metrics.get("eval/trace_equivariance/max_abs_error", "")})
            continue
        for row in rows:
            key = str(row.get("entry_key", row.get("key", "")))
            layer = key.split("/", 1)[0] if key else ""
            out.append({
                **base,
                "trace_kind": task,
                "layer": layer,
                "key": key,
                "rms_q95": row.get("q95_abs", row.get("rms_q95", "")),
                "rms_q99": row.get("q99_abs", row.get("rms_q99", "")),
                "max_abs": row.get("max_abs", row.get("max_abs_error", "")),
                "nonfinite_count": row.get("nonfinite_count", row.get("readout_nonfinite_count", "")),
                "compared_entry_count": row.get("compared_entry_count", metrics.get(f"eval/{task}/compared_entry_count", "")),
                "comparison_error_count": row.get("comparison_error_count", metrics.get(f"eval/{task}/comparison_error_count", "")),
                "max_equivariance_error": row.get("max_abs_error", metrics.get(f"eval/{task}/max_abs_error", "")),
            })
    return out


def _training_curve_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    base = _base_row(context)
    by_step: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in context["train_metrics"]:
        step = str(row.get("step", ""))
        namespace = row.get("namespace", "")
        metric = row.get("metric", "")
        if namespace == "train" and metric == "energy":
            by_step[step]["energy_mean"] = row.get("value", "")
        elif namespace == "train" and metric == "energy_stderr":
            by_step[step]["energy_stderr"] = row.get("value", "")
        elif namespace == "train" and metric == "energy_variance":
            by_step[step]["local_energy_var"] = row.get("value", "")
        elif namespace == "train" and metric == "grad_norm":
            by_step[step]["grad_norm"] = row.get("value", "")
        elif namespace == "train/sampler" and metric == "acceptance_rate":
            by_step[step]["acceptance_rate"] = row.get("value", "")
    return [
        {**base, "step": step, "energy_mean": data.get("energy_mean", ""), "energy_stderr": data.get("energy_stderr", ""), "local_energy_var": data.get("local_energy_var", ""), "acceptance_rate": data.get("acceptance_rate", ""), "grad_norm": data.get("grad_norm", ""), "wall_time_sec": data.get("wall_time_sec", "")}
        for step, data in sorted(by_step.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0)
    ]


def _resource_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    metadata = context["train_metadata"]
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    return {
        **base,
        "train_wall_time_sec": _format_number(_duration_from_status(context["train_status_json"])),
        "eval_wall_time_sec": _format_number(_duration_from_status(context["eval_status_json"])),
        "peak_memory_mb": metadata.get("peak_memory_mb", ""),
        "device_type": runtime.get("device", metadata.get("device", "")),
    }


def _architecture_summary(energy_rows: Sequence[dict[str, Any]], tail_rows: Sequence[dict[str, Any]], trace_rows: Sequence[dict[str, Any]], failure_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in energy_rows:
        groups[(str(row["basis_class"]), str(row["normalization"]), str(row["winner_kind"]))].append(row)
    tail_by_run = defaultdict(list)
    for row in tail_rows:
        tail_by_run[row["final_run_id"]].append(_as_float(row.get("outlier_fraction")))
    trace_by_run = defaultdict(list)
    for row in trace_rows:
        trace_by_run[row["final_run_id"]].append(_as_float(row.get("comparison_error_count")))
        trace_by_run[row["final_run_id"]].append(_as_float(row.get("nonfinite_count")))
    failures_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        failures_by_group[(str(row["basis_class"]), str(row["normalization"]), str(row["winner_kind"]))].append(row)

    out = []
    for key, rows in sorted(groups.items()):
        run_ids = [row["final_run_id"] for row in rows]
        energy_errors = [_as_float(row.get("energy_error")) for row in rows]
        variances = [_as_float(row.get("local_energy_var")) for row in rows]
        pathologies = [_as_float(row.get("pathology_fraction")) for row in rows]
        tail_outliers = [_median(tail_by_run[run_id]) for run_id in run_ids]
        trace_failures = [_sum(trace_by_run[run_id]) for run_id in run_ids]
        successful = [row for row in rows if _as_float(row.get("energy_mean")) is not None]
        out.append({
            "basis_class": key[0],
            "normalization": key[1],
            "winner_kind": key[2],
            "n_success": len(successful),
            "n_expected": EXPECTED_FINAL_SEEDS,
            "energy_error_median": _format_number(_median(energy_errors)),
            "energy_error_q25": _format_number(_quantile(energy_errors, 0.25)),
            "energy_error_q75": _format_number(_quantile(energy_errors, 0.75)),
            "local_energy_var_median": _format_number(_median(variances)),
            "pathology_fraction_median": _format_number(_median(pathologies)),
            "tail_outlier_fraction_median": _format_number(_median(tail_outliers)),
            "trace_failure_count_total": _format_number(_sum(trace_failures)),
            "major_failure_mode": _major_failure_mode(failures_by_group.get(key, [])),
        })
    return out


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")


def collect_final_outputs(
    *,
    results_root: str | Path,
    collect_attempt_id: str | None = None,
    final_eval_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Collect compact final-summary tables from final train/eval artifacts."""

    results_root = Path(results_root)
    collect_attempt_id = collect_attempt_id or new_attempt_id()
    attempt = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    contexts = [_run_context(final_run_id, attempt_id, attempt_dir) for final_run_id, attempt_id, attempt_dir in _iter_final_eval_attempts(results_root, final_eval_attempt_id)]
    run_index_rows = [_run_index_row(context) for context in contexts]
    energy_rows = [_energy_row(context) for context in contexts]
    histogram_rows = _local_energy_histograms(contexts)
    cusp_rows = [row for context in contexts for row in _cusp_summary(context)]
    tail_rows = [row for context in contexts for row in _tail_summary(context)]
    stratified_rows = [row for context in contexts for row in _stratified_summary(context)]
    hooke_rows = [row for context in contexts for row in _hooke_orbital_summary(context)]
    symmetry_rows = [row for context in contexts for row in _symmetry_summary(context)]
    trace_rows = [row for context in contexts for row in _trace_summary(context)]
    training_rows = [row for context in contexts for row in _training_curve_summary(context)]
    resource_rows = [_resource_row(context) for context in contexts]
    failure_rows = [row for context in contexts for row in _failure_rows(context)]
    architecture_rows = _architecture_summary(energy_rows, tail_rows, trace_rows, failure_rows)

    table_specs = {
        "run_index.csv": (run_index_rows, RUN_INDEX_COLUMNS),
        "architecture_summary.csv": (architecture_rows, ARCHITECTURE_SUMMARY_COLUMNS),
        "energy_by_run.csv": (energy_rows, ENERGY_BY_RUN_COLUMNS),
        "local_energy_histograms.csv": (histogram_rows, LOCAL_ENERGY_HISTOGRAM_COLUMNS),
        "cusp_profile_summary.csv": (cusp_rows, CUSP_PROFILE_COLUMNS),
        "tail_profile_summary.csv": (tail_rows, TAIL_PROFILE_COLUMNS),
        "stratified_summary.csv": (stratified_rows, STRATIFIED_COLUMNS),
        "hooke_orbital_summary.csv": (hooke_rows, HOOKE_ORBITAL_COLUMNS),
        "symmetry_summary.csv": (symmetry_rows, SYMMETRY_COLUMNS),
        "trace_summary.csv": (trace_rows, TRACE_COLUMNS),
        "training_curve_summary.csv": (training_rows, TRAINING_CURVE_COLUMNS),
        "resource_summary.csv": (resource_rows, RESOURCE_COLUMNS),
        "failure_modes.csv": (failure_rows, FAILURE_COLUMNS),
    }
    for filename, (rows, columns) in table_specs.items():
        _write_csv(attempt / filename, rows, columns)

    manifest = {
        "study": "pair_stability",
        "stage": STAGE_FINAL_COLLECT,
        "attempt_id": collect_attempt_id,
        "final_eval_attempt_id": final_eval_attempt_id,
        "n_final_eval_attempts": len(contexts),
        "tables": {filename: len(rows) for filename, (rows, _) in table_specs.items()},
        "source_stages": {
            "final_grid": "05_final_grid",
            "final_train": "06_final_train",
            "final_eval": "07_final_eval",
        },
    }
    _write_manifest(attempt / "manifest.yaml", manifest)
    write_latest(stage_dir(results_root, STAGE_FINAL_COLLECT), collect_attempt_id)
    return {"attempt_dir": str(attempt), "manifest": manifest}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-collect arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-eval-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Collect compact final tables."""

    args = parse_args(argv)
    result = collect_final_outputs(
        results_root=args.results_root,
        collect_attempt_id=args.attempt_id,
        final_eval_attempt_id=args.final_eval_attempt_id,
    )
    manifest = result["manifest"]
    print(
        f"[pair_stability] final collect consumed {manifest['n_final_eval_attempts']} "
        f"final-eval attempts -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
