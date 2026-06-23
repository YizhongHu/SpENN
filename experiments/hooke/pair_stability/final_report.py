"""Build final pair-stability report artifacts from ``07_final_eval`` outputs.

This stage is deliberately offline: it consumes existing final-evaluation
artifacts, joins the stored provenance onto metric/record tables, and renders
report figures. It does not import model code, load checkpoints, or rerun
training/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from run_utils import (
    STAGE_FINAL_EVAL,
    STAGE_FINAL_REPORT,
    attempt_ids,
    new_attempt_id,
    read_json,
    stage_dir,
    write_json,
    write_latest,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
EXACT_HOOKE_ENERGY = 2.0

EVAL_RECORD_SOURCES = {
    "energy_samples.csv": [("energy", "mcmc_energy_samples.csv")],
    "cusp_profiles.csv": [("cusp", "cusp_profiles.csv")],
    "tail_profiles.csv": [("tail", "tail_profiles.csv")],
    "stratified_geometry.csv": [("stratified_geometry", "stratified_metrics.csv")],
    "hooke_orbital.csv": [("hooke_orbital", "hooke_orbital_metrics.csv")],
    "symmetry_diagnostics.csv": [
        ("full_model_antisymmetry", "transform_records.csv"),
        ("spatial_exchange_symmetry", "transform_records.csv"),
        ("rotation_consistency", "transform_records.csv"),
    ],
    "trace_diagnostics.csv": [
        ("trace_equivariance", "trace_records.csv"),
        ("feature_trace_stability", "trace_records.csv"),
        ("readout_trace_stability", "trace_records.csv"),
    ],
}

LEGACY_PLOT_ALIASES = {
    "stratified_metrics.csv": "stratified_geometry.csv",
    "hooke_orbital_metrics.csv": "hooke_orbital.csv",
    "mcmc_energy_samples.csv": "energy_samples.csv",
    "symmetry_metrics.csv": "symmetry_diagnostics.csv",
    "spatial_exchange_metrics.csv": "symmetry_diagnostics.csv",
    "rotation_metrics.csv": "symmetry_diagnostics.csv",
    "trace_metrics.csv": "trace_diagnostics.csv",
    "feature_trace_metrics.csv": "trace_diagnostics.csv",
    "readout_trace_metrics.csv": "trace_diagnostics.csv",
}

FINAL_CHAMPION_COLUMNS = [
    "final_run_id",
    "source_champion_id",
    "basis",
    "normalization",
    "minor_hparams",
    "model_seed",
    "sampler_seed",
    "eval_seed",
    "train_status",
    "eval_status",
    "final_energy_mean",
    "final_energy_stderr",
    "energy_error",
    "local_energy_var",
    "major_failure_mode",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "winner_kind",
    "replicate_index",
    "resolved_checkpoint_dir",
]

FINAL_FAMILY_COLUMNS = [
    "basis",
    "normalization",
    "minor_hparams",
    "n_replicates",
    "energy_mean_avg",
    "energy_mean_std",
    "energy_error_avg",
    "local_energy_var_median",
    "cusp_error_median",
    "tail_outlier_count_total",
    "trace_failure_count_total",
    "antisymmetry_error_max",
    "rank_energy",
    "rank_stability",
    "rank_overall",
]

FAILURE_COLUMNS = [
    "final_run_id",
    "task",
    "severity",
    "failure_mode",
    "metric_name",
    "metric_value",
    "threshold",
    "geometry_context",
]

RESOURCE_COLUMNS = [
    "final_run_id",
    "final_eval_attempt_id",
    "train_status",
    "eval_status",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "n_eval_rows",
    "n_eval_tasks",
    "n_plot_record_rows",
]


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


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
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
            rows.append(
                {
                    "step": step,
                    "namespace": namespace,
                    "metric": str(key),
                    "value": _csv_value(value),
                }
            )
    return rows


def _metric_map(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = {}
    for row in rows:
        values[f"{row['namespace']}/{row['metric']}"] = row["value"]
    return values


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


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def _std(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0.0 if clean else None
    return statistics.stdev(clean)


def _sum(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return math.fsum(clean) if clean else None


def _max(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
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


def _minor_hparams(job: dict[str, Any]) -> str:
    keys = ("architecture", "lr", "channels", "winner_kind")
    return ";".join(f"{key}={job.get(key, '')}" for key in keys if job.get(key, "") != "")


def _family_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("basis", "")), str(row.get("normalization", "")), str(row.get("minor_hparams", "")))


def _train_status(checkpoint: dict[str, Any], source_train: dict[str, Any]) -> str:
    if checkpoint.get("resolved_checkpoint_dir"):
        return "checkpoint_selected"
    if source_train.get("final_train_attempt_id"):
        return "attempt_recorded"
    return ""


def _train_wall_time(job: dict[str, Any]) -> str:
    champion = job.get("source_champion", {})
    if not isinstance(champion, dict):
        champion = {}
    for key in (
        "train_wall_time_sec",
        "train_wall_time_sec_seed_median",
        "train/runtime/wall_time_sec_seed_median",
        "metric_seed_median",
    ):
        value = champion.get(key, "")
        if _as_float(value) is not None:
            return str(value)
    metric = champion.get("metric", "")
    if metric == "train/runtime/wall_time_sec_seed_median" and _as_float(champion.get("metric_value")) is not None:
        return str(champion.get("metric_value"))
    return ""


def _task_from_namespace(namespace: str) -> str:
    if namespace.startswith("eval/"):
        return namespace[len("eval/") :]
    return namespace


def _failure_rows(final_run_id: str, status: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if status not in {"completed", "success"}:
        failures.append(
            {
                "final_run_id": final_run_id,
                "task": "run",
                "severity": "failed",
                "failure_mode": "run_status",
                "metric_name": "status",
                "metric_value": status,
                "threshold": "completed",
                "geometry_context": "",
            }
        )
    for metric_name, value in metrics.items():
        namespace, _, key = metric_name.rpartition("/")
        task = _task_from_namespace(namespace)
        bool_value = _as_bool(value)
        numeric = _as_float(value)
        if key == "task_failed" and bool_value:
            failures.append(
                {
                    "final_run_id": final_run_id,
                    "task": task,
                    "severity": "failed",
                    "failure_mode": key,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "threshold": "False",
                    "geometry_context": "",
                }
            )
        if numeric is None:
            continue
        failure_key = any(
            needle in key
            for needle in (
                "failure_count",
                "nonfinite_count",
                "pathology_count",
                "outlier_count",
                "mismatch_count",
                "comparison_error_count",
                "missing_key_count",
                "extra_key_count",
                "near_zero_count",
            )
        )
        if failure_key and numeric > 0:
            failures.append(
                {
                    "final_run_id": final_run_id,
                    "task": task,
                    "severity": "warning",
                    "failure_mode": key,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "threshold": "0",
                    "geometry_context": "",
                }
            )
        if key.endswith("finite_fraction") and numeric < 1.0:
            failures.append(
                {
                    "final_run_id": final_run_id,
                    "task": task,
                    "severity": "warning",
                    "failure_mode": key,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "threshold": "1.0",
                    "geometry_context": "",
                }
            )
    return failures


def _major_failure_mode(failures: Sequence[dict[str, Any]]) -> str:
    if not failures:
        return ""
    failed = [row for row in failures if row.get("severity") == "failed"]
    row = failed[0] if failed else failures[0]
    return f"{row.get('task', '')}:{row.get('failure_mode', '')}"


def _record_context(
    *,
    final_run_id: str,
    attempt_id: str,
    job: dict[str, Any],
    task: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    output = dict(row)
    output["final_run_id"] = final_run_id
    output["final_eval_attempt_id"] = attempt_id
    output["task"] = task
    output["architecture"] = job.get("architecture", "")
    output["basis"] = job.get("basis_envelope", job.get("architecture", ""))
    output["normalization"] = job.get("normalization", "")
    output["winner_kind"] = job.get("winner_kind", "")
    output["replicate"] = job.get("replicate_index", "")
    output["eval_seed"] = job.get("final_eval_seed", "")
    sample_id = row.get("sample_index", row.get("record_index", row.get("orbit_id", row.get("key", ""))))
    output["geometry_id"] = f"{task}:{sample_id}"
    output.setdefault("R_norm", "")
    output.setdefault("R_norm_bin", "")
    output.setdefault("r12_bin", "")
    output.setdefault("com_id", "")
    if "radius" in output:
        output["R_norm"] = output.get("radius", "")
        output["R_norm_bin"] = _bin_value(output.get("radius"))
    if "r12" in output:
        output["r12_bin"] = _bin_value(output.get("r12"))
    if "center_of_mass_id" in output:
        output["com_id"] = output.get("center_of_mass_id", "")
    if "local_energy" in output:
        local_energy = _as_float(output.get("local_energy"))
        output["exact_local_energy"] = EXACT_HOOKE_ENERGY
        output["energy_error"] = "" if local_energy is None else local_energy - EXACT_HOOKE_ENERGY
    output.setdefault("exact_logabs", "")
    finite = str(output.get("finite", "True")).lower()
    local_energy = _as_float(output.get("local_energy"))
    output["pathology_flag"] = finite == "false" or (local_energy is not None and abs(local_energy) > 10.0)
    return output


def _bin_value(value: Any, *, width: float = 0.5) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    low = math.floor(numeric / width) * width
    high = low + width
    return f"[{low:.2g},{high:.2g})"


def _plot_table_rows(
    final_run_id: str,
    attempt_id: str,
    attempt_dir: Path,
    job: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    tables = {name: [] for name in EVAL_RECORD_SOURCES}
    for table_name, sources in EVAL_RECORD_SOURCES.items():
        for task, filename in sources:
            rows = _read_csv(attempt_dir / task / filename)
            tables[table_name].extend(
                _record_context(
                    final_run_id=final_run_id,
                    attempt_id=attempt_id,
                    job=job,
                    task=task,
                    row=row,
                )
                for row in rows
            )
    tables["training_curves.csv"] = []
    return tables


def _attempt_records(results_root: Path, final_eval_attempt_id: str | None) -> list[dict[str, Any]]:
    records = []
    for final_run_id, attempt_id, attempt_dir in _iter_final_eval_attempts(results_root, final_eval_attempt_id):
        job = _load_json_if_present(attempt_dir / "source_final_job.json")
        checkpoint = _load_json_if_present(attempt_dir / "evaluated_checkpoint.json")
        source_train = _load_json_if_present(attempt_dir / "source_final_train_attempt.json")
        status_json = _load_json_if_present(attempt_dir / "status.json")
        status = str(status_json.get("status", _status_of(attempt_dir)))
        metric_rows = _read_metrics_jsonl(attempt_dir / "metrics.jsonl")
        metrics = _metric_map(metric_rows)
        plot_tables = _plot_table_rows(final_run_id, attempt_id, attempt_dir, job)
        records.append(
            {
                "final_run_id": final_run_id,
                "attempt_id": attempt_id,
                "attempt_dir": attempt_dir,
                "job": job,
                "checkpoint": checkpoint,
                "source_train": source_train,
                "status": status,
                "status_json": status_json,
                "metric_rows": metric_rows,
                "metrics": metrics,
                "plot_tables": plot_tables,
            }
        )
    return records


def _champion_row(record: dict[str, Any], failures: Sequence[dict[str, Any]]) -> dict[str, Any]:
    job = record["job"]
    metrics = record["metrics"]
    checkpoint = record["checkpoint"]
    energy_mean = _as_float(metrics.get("eval/energy/local_energy_mean"))
    energy_stderr = _as_float(metrics.get("eval/energy/local_energy_stderr"))
    energy_error = None if energy_mean is None else energy_mean - EXACT_HOOKE_ENERGY
    return {
        "final_run_id": record["final_run_id"],
        "source_champion_id": job.get("source_champion_id", ""),
        "architecture": job.get("architecture", ""),
        "basis": job.get("basis_envelope", job.get("architecture", "")),
        "normalization": job.get("normalization", ""),
        "minor_hparams": _minor_hparams(job),
        "model_seed": job.get("final_train_model_seed", ""),
        "sampler_seed": job.get("final_train_sampler_seed", ""),
        "eval_seed": job.get("final_eval_seed", ""),
        "train_status": _train_status(checkpoint, record["source_train"]),
        "eval_status": record["status"],
        "final_energy_mean": _format_number(energy_mean),
        "final_energy_stderr": _format_number(energy_stderr),
        "energy_error": _format_number(energy_error),
        "local_energy_var": _format_number(_as_float(metrics.get("eval/energy/local_energy_variance"))),
        "major_failure_mode": _major_failure_mode(failures),
        "train_wall_time_sec": _train_wall_time(job),
        "eval_wall_time_sec": _format_number(_duration_from_status(record["status_json"])),
        "winner_kind": job.get("winner_kind", ""),
        "replicate_index": job.get("replicate_index", ""),
        "resolved_checkpoint_dir": checkpoint.get("resolved_checkpoint_dir", ""),
    }


def _metric_rows_by_run(record: dict[str, Any]) -> list[dict[str, Any]]:
    job = record["job"]
    rows = []
    for row in record["metric_rows"]:
        out = dict(row)
        out.update(
            {
                "final_run_id": record["final_run_id"],
                "final_eval_attempt_id": record["attempt_id"],
                "basis": job.get("basis_envelope", job.get("architecture", "")),
                "normalization": job.get("normalization", ""),
                "minor_hparams": _minor_hparams(job),
                "winner_kind": job.get("winner_kind", ""),
                "replicate_index": job.get("replicate_index", ""),
            }
        )
        rows.append(out)
    return rows


def _family_rows(champion_rows: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_by_run = {record["final_run_id"]: record["metrics"] for record in records}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in champion_rows:
        groups[_family_key(row)].append(row)

    family_rows = []
    for key, rows in sorted(groups.items()):
        run_ids = [str(row["final_run_id"]) for row in rows]
        energy_means = [_as_float(row.get("final_energy_mean")) for row in rows]
        energy_errors = [_as_float(row.get("energy_error")) for row in rows]
        variances = [_as_float(row.get("local_energy_var")) for row in rows]
        cusp_errors = [_as_float(metrics_by_run[run_id].get("eval/cusp/cusp_even_slope_abs_error")) for run_id in run_ids]
        tail_counts = [_as_float(metrics_by_run[run_id].get("eval/tail/local_energy_pathology_count")) for run_id in run_ids]
        trace_counts = [
            _sum(
                [
                    _as_float(metrics_by_run[run_id].get("eval/trace_equivariance/failure_count")),
                    _as_float(metrics_by_run[run_id].get("eval/trace_equivariance/comparison_error_count")),
                    _as_float(metrics_by_run[run_id].get("eval/feature_trace_stability/feature_nonfinite_count")),
                    _as_float(metrics_by_run[run_id].get("eval/readout_trace_stability/readout_nonfinite_count")),
                ]
            )
            for run_id in run_ids
        ]
        antisymmetry_errors = [
            _as_float(metrics_by_run[run_id].get("eval/full_model_antisymmetry/logabs_max_abs_error"))
            for run_id in run_ids
        ]
        family_rows.append(
            {
                "basis": key[0],
                "normalization": key[1],
                "minor_hparams": key[2],
                "n_replicates": len(rows),
                "energy_mean_avg": _format_number(_mean(energy_means)),
                "energy_mean_std": _format_number(_std(energy_means)),
                "energy_error_avg": _format_number(_mean(energy_errors)),
                "local_energy_var_median": _format_number(_median(variances)),
                "cusp_error_median": _format_number(_median(cusp_errors)),
                "tail_outlier_count_total": _format_number(_sum(tail_counts)),
                "trace_failure_count_total": _format_number(_sum(trace_counts)),
                "antisymmetry_error_max": _format_number(_max(antisymmetry_errors)),
            }
        )
    _assign_ranks(family_rows)
    return family_rows


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    def rank_by(key: str, *, absolute: bool = False) -> dict[int, int]:
        scored = []
        for index, row in enumerate(rows):
            value = _as_float(row.get(key))
            if value is None:
                value = math.inf
            if absolute:
                value = abs(value)
            scored.append((value, index))
        return {index: rank for rank, (_, index) in enumerate(sorted(scored), start=1)}

    energy_ranks = rank_by("energy_error_avg", absolute=True)
    stability_scores = []
    for index, row in enumerate(rows):
        score = math.fsum(
            value
            for value in (
                _as_float(row.get("local_energy_var_median")),
                _as_float(row.get("cusp_error_median")),
                _as_float(row.get("tail_outlier_count_total")),
                _as_float(row.get("trace_failure_count_total")),
                _as_float(row.get("antisymmetry_error_max")),
            )
            if value is not None
        )
        stability_scores.append((score, index))
    stability_ranks = {index: rank for rank, (_, index) in enumerate(sorted(stability_scores), start=1)}
    for index, row in enumerate(rows):
        row["rank_energy"] = energy_ranks[index]
        row["rank_stability"] = stability_ranks[index]
        row["rank_overall"] = energy_ranks[index] + stability_ranks[index]
    rows.sort(key=lambda row: (int(row["rank_overall"]), int(row["rank_energy"])))


def _resource_row(record: dict[str, Any], plot_row_count: int) -> dict[str, Any]:
    metrics = record["metrics"]
    eval_tasks = sorted(
        _task_from_namespace(key.rpartition("/")[0])
        for key in metrics
        if key.startswith("eval/") and "/status/" not in key
    )
    return {
        "final_run_id": record["final_run_id"],
        "final_eval_attempt_id": record["attempt_id"],
        "train_status": _train_status(record["checkpoint"], record["source_train"]),
        "eval_status": record["status"],
        "train_wall_time_sec": _train_wall_time(record["job"]),
        "eval_wall_time_sec": _format_number(_duration_from_status(record["status_json"])),
        "n_eval_rows": len(record["metric_rows"]),
        "n_eval_tasks": len(set(eval_tasks)),
        "n_plot_record_rows": plot_row_count,
    }


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rhu/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_no_data(path: Path, title: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_labels, x_labels, matrix = _heatmap_matrix(
        rows,
        row_key=row_key,
        col_key=col_key,
        value_key=value_key,
    )
    if not matrix:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    from matplotlib.colors import SymLogNorm

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(x_labels)), max(3.5, 0.8 * len(y_labels))))
    finite_values = [value for row in matrix for value in row if value is not None]
    vmax = max(abs(value) for value in finite_values) if finite_values else 1.0
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    colorbar_label = value_key
    if transform == "signed_log":
        nonzero = [abs(value) for value in finite_values if value != 0.0]
        norm = SymLogNorm(linthresh=min(nonzero), vmin=-vmax, vmax=vmax, base=10) if nonzero else None
        image = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
        colorbar_label = f"{value_key} (symmetric log color; labels are real scale)"
    else:
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value is not None:
                ax.text(x_index, y_index, f"{value:.2g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return real-scale heatmap cell means for plotting and annotations."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get(value_key))
        if value is None:
            continue
        cells[(str(row.get(row_key, "")), str(row.get(col_key, "")))].append(value)
    if not cells:
        return [], [], []
    y_labels = sorted({key[0] for key in cells})
    x_labels = sorted({key[1] for key in cells})
    matrix = []
    for y in y_labels:
        row_values = []
        for x in x_labels:
            row_values.append(_mean(cells.get((y, x), [])))
        matrix.append(row_values)
    return y_labels, x_labels, matrix


def _save_scatter(path: Path, rows: Sequence[dict[str, Any]], *, x_key: str, y_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [
        (x, y, str(row.get("basis", "")), str(row.get("normalization", "")))
        for row in rows
        if (x := _as_float(row.get(x_key))) is not None and (y := _as_float(row.get(y_key))) is not None
    ]
    if not points:
        _save_no_data(path, title)
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    for x, y, basis, norm in points:
        ax.scatter(x, y, label=f"{basis}/{norm}", s=36)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    if len(unique) <= 12:
        ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _energy_variance_points(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return positive log-log points for the 1B energy/stability scatter."""

    points = []
    for row in rows:
        energy_error = _as_float(row.get("energy_error"))
        variance = _as_float(row.get("local_energy_var"))
        if energy_error is None or variance is None:
            continue
        abs_error = abs(energy_error)
        if abs_error <= 0.0 or variance <= 0.0:
            continue
        architecture = str(row.get("architecture", row.get("basis", "")))
        points.append(
            {
                "abs_energy_error": abs_error,
                "local_energy_var": variance,
                "architecture": architecture,
                "normalization": str(row.get("normalization", "")),
            }
        )
    return points


def _save_energy_variance_scatter(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    """Write the 1B scatter with independent architecture/color and norm/shape legends."""

    path.parent.mkdir(parents=True, exist_ok=True)
    points = _energy_variance_points(rows)
    if not points:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    from matplotlib.lines import Line2D

    architectures = sorted({str(point["architecture"]) for point in points})
    normalizations = sorted({str(point["normalization"]) for point in points})
    cmap = plt.get_cmap("tab20" if len(architectures) > 10 else "tab10")
    colors = {architecture: cmap(index % cmap.N) for index, architecture in enumerate(architectures)}
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h", "p"]
    marker_by_norm = {
        normalization: markers[index % len(markers)] for index, normalization in enumerate(normalizations)
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for point in points:
        architecture = str(point["architecture"])
        normalization = str(point["normalization"])
        ax.scatter(
            point["abs_energy_error"],
            point["local_energy_var"],
            color=colors[architecture],
            marker=marker_by_norm[normalization],
            s=58,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("abs energy error |E - 2|")
    ax.set_ylabel("local-energy variance")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[architecture],
            markeredgecolor="black",
            markersize=7,
            label=architecture,
        )
        for architecture in architectures
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_norm[normalization],
            color="black",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            linestyle="none",
            markersize=7,
            label=normalization,
        )
        for normalization in normalizations
    ]
    architecture_legend = ax.legend(
        handles=color_handles,
        title="Architecture",
        fontsize=7,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(architecture_legend)
    ax.legend(
        handles=shape_handles,
        title="Normalization",
        fontsize=7,
        title_fontsize=8,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _local_energy_distribution_groups(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[float]]]:
    """Group local-energy samples by normalization row and architecture column."""

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        local_energy = _as_float(row.get("local_energy"))
        if local_energy is None:
            continue
        normalization = str(row.get("normalization", ""))
        architecture = str(row.get("architecture", row.get("basis", "")))
        groups[(normalization, architecture)].append(local_energy)
    normalizations = sorted({key[0] for key in groups})
    architectures = sorted({key[1] for key in groups})
    return normalizations, architectures, groups


def _histogram_bin_count(values: Sequence[float]) -> int:
    """Return a readable per-panel histogram bin count."""

    if len(values) <= 1:
        return 1
    return min(24, max(4, int(math.sqrt(len(values)))))


def _save_local_energy_distribution_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
) -> None:
    """Write 1C as normalization-by-architecture local-energy histograms."""

    path.parent.mkdir(parents=True, exist_ok=True)
    normalizations, architectures, groups = _local_energy_distribution_groups(rows)
    if not groups:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    n_rows = len(normalizations)
    n_cols = len(architectures)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4.0, 3.2 * n_cols), max(3.0, 2.4 * n_rows)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            values = groups.get((normalization, architecture), [])
            if values:
                ax.hist(values, bins=_histogram_bin_count(values), color="#4C78A8", edgecolor="black", alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\ncount")
            if row_index == n_rows - 1:
                ax.set_xlabel("local_energy")
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line_plot(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        label = "/".join(str(row.get(key, "")) for key in group_keys if row.get(key, "") != "")
        groups[label or "all"].append((x, y))
    if not groups:
        _save_no_data(path, title)
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in sorted(groups.items()):
        values = sorted(values)
        ax.plot([point[0] for point in values], [point[1] for point in values], marker="o", label=label)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    if len(groups) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_bar(path: Path, rows: Sequence[dict[str, Any]], *, label_key: str, value_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [(str(row.get(label_key, "")), _as_float(row.get(value_key))) for row in rows]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        _save_no_data(path, title)
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(values)), 4))
    ax.bar(range(len(values)), [value for _, value in values])
    ax.set_xticks(range(len(values)), [label for label, _ in values], rotation=45, ha="right")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_figures(
    figures_dir: Path,
    *,
    champion_rows: Sequence[dict[str, Any]],
    family_rows: Sequence[dict[str, Any]],
    plot_rows_by_table: dict[str, list[dict[str, Any]]],
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []

    figure_specs = [
        (
            "1A_real_scale_energy_error_heatmap.png",
            lambda path: _save_heatmap(
                path,
                champion_rows,
                row_key="basis",
                col_key="normalization",
                value_key="energy_error",
                title="Signed final energy error",
            ),
        ),
        (
            "1A_log_scale_energy_error_heatmap.png",
            lambda path: _save_heatmap(
                path,
                champion_rows,
                row_key="basis",
                col_key="normalization",
                value_key="energy_error",
                title="Signed log10 final energy error",
                transform="signed_log",
            ),
        ),
        (
            "1B_energy_error_vs_local_energy_variance.png",
            lambda path: _save_energy_variance_scatter(
                path,
                champion_rows,
                title="Absolute energy error vs local-energy variance",
            ),
        ),
        (
            "1C_local_energy_distribution_grid.png",
            lambda path: _save_local_energy_distribution_grid(
                path,
                plot_rows_by_table["energy_samples.csv"],
                title="MCMC local-energy distributions",
            ),
        ),
        (
            "2A_cusp_local_energy_by_com.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["cusp_profiles.csv"],
                x_key="r12",
                y_key="local_energy",
                group_keys=("basis", "normalization", "center_of_mass_id", "direction_id"),
                title="Cusp local energy by CoM path",
            ),
        ),
        (
            "2B_cusp_c_minus_1_placeholder.png",
            lambda path: _save_no_data(path, "C_-1 profile records not emitted by current evaluator"),
        ),
        (
            "2C_cusp_odd_slant_placeholder.png",
            lambda path: _save_no_data(path, "Odd-slant profile records not emitted by current evaluator"),
        ),
        (
            "3A_tail_local_energy_by_path.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["tail_profiles.csv"],
                x_key="radius",
                y_key="local_energy",
                group_keys=("basis", "normalization", "direction_id", "relative_direction_id"),
                title="Tail local energy by path",
            ),
        ),
        (
            "3B_tail_logabs_with_reference.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["tail_profiles.csv"],
                x_key="radius",
                y_key="logabs",
                group_keys=("basis", "normalization", "direction_id"),
                title="Tail logabs by path; exact reference included when records provide it",
            ),
        ),
        (
            "3C_tail_outlier_heatmap.png",
            lambda path: _save_heatmap(
                path,
                family_rows,
                row_key="basis",
                col_key="normalization",
                value_key="tail_outlier_count_total",
                title="Tail pathology count",
            ),
        ),
        (
            "4_stratified_geometry_aggregate_heatmap.png",
            lambda path: _save_heatmap(
                path,
                plot_rows_by_table["stratified_geometry.csv"],
                row_key="basis",
                col_key="normalization",
                value_key="energy_error",
                title="Stratified geometry signed local-energy error",
            ),
        ),
        (
            "5A_hooke_orbital_local_energy_distribution.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["hooke_orbital.csv"],
                x_key="sample_index",
                y_key="local_energy",
                group_keys=("basis", "normalization", "replicate"),
                title="Hooke-orbital local-energy samples",
            ),
        ),
        (
            "5B_hooke_orbital_local_energy_vs_r12.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["hooke_orbital.csv"],
                x_key="r12",
                y_key="local_energy",
                group_keys=("basis", "normalization", "R_norm_bin"),
                title="Hooke-orbital local energy vs r12 by CoM-radius bin",
            ),
        ),
        (
            "5C_hooke_orbital_local_energy_vs_radius.png",
            lambda path: _save_line_plot(
                path,
                plot_rows_by_table["hooke_orbital.csv"],
                x_key="radius",
                y_key="local_energy",
                group_keys=("basis", "normalization", "r12_bin"),
                title="Hooke-orbital local energy vs CoM radius by r12 bin",
            ),
        ),
        (
            "6_symmetry_failure_counts.png",
            lambda path: _save_bar(
                path,
                family_rows,
                label_key="normalization",
                value_key="antisymmetry_error_max",
                title="Full-model antisymmetry max logabs error",
            ),
        ),
        (
            "7_trace_failure_counts.png",
            lambda path: _save_bar(
                path,
                family_rows,
                label_key="normalization",
                value_key="trace_failure_count_total",
                title="Trace failure counts",
            ),
        ),
        (
            "8_training_curves.png",
            lambda path: _save_no_data(path, "Training curves are not present in 07_final_eval artifacts"),
        ),
    ]

    for filename, writer in figure_specs:
        path = figures_dir / filename
        writer(path)
        written.append(filename)
    strata = sorted(
        {
            str(row.get("stratum", ""))
            for row in plot_rows_by_table["stratified_geometry.csv"]
            if row.get("stratum", "") != ""
        }
    )
    for stratum in strata:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stratum)
        filename = f"4_stratified_geometry_{safe}_heatmap.png"
        rows = [row for row in plot_rows_by_table["stratified_geometry.csv"] if str(row.get("stratum", "")) == stratum]
        _save_heatmap(
            figures_dir / filename,
            rows,
            row_key="basis",
            col_key="normalization",
            value_key="energy_error",
            title=f"Stratified geometry signed local-energy error: {stratum}",
        )
        written.append(filename)
    return written


def build_report(
    *,
    results_root: str | Path,
    report_attempt_id: str | None = None,
    final_eval_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Write an ``08_final_report`` attempt from final-eval artifacts."""

    results_root = Path(results_root)
    report_attempt_id = report_attempt_id or new_attempt_id()
    attempt = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    summary_dir = attempt / "summary_tables"
    plot_dir = attempt / "plot_tables"
    figures_dir = attempt / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = _attempt_records(results_root, final_eval_attempt_id)
    champion_rows = []
    metric_rows = []
    failure_rows = []
    resource_rows = []
    plot_rows_by_table = {name: [] for name in [*EVAL_RECORD_SOURCES, "training_curves.csv"]}

    for record in records:
        failures = _failure_rows(record["final_run_id"], record["status"], record["metrics"])
        failure_rows.extend(failures)
        champion_rows.append(_champion_row(record, failures))
        metric_rows.extend(_metric_rows_by_run(record))
        for table_name, rows in record["plot_tables"].items():
            plot_rows_by_table[table_name].extend(rows)
        resource_rows.append(
            _resource_row(
                record,
                sum(len(rows) for rows in record["plot_tables"].values()),
            )
        )

    family_rows = _family_rows(champion_rows, records)
    figure_names = _write_figures(
        figures_dir,
        champion_rows=champion_rows,
        family_rows=family_rows,
        plot_rows_by_table=plot_rows_by_table,
    )

    _write_csv(summary_dir / "final_champions.csv", champion_rows, FINAL_CHAMPION_COLUMNS)
    _write_csv(summary_dir / "final_metrics_by_run.csv", metric_rows)
    _write_csv(summary_dir / "final_metrics_by_family.csv", family_rows, FINAL_FAMILY_COLUMNS)
    _write_csv(summary_dir / "failure_modes.csv", failure_rows, FAILURE_COLUMNS)
    _write_csv(summary_dir / "resource_summary.csv", resource_rows, RESOURCE_COLUMNS)

    # Compatibility aliases for the first PR8.9 report reducer.
    _write_csv(summary_dir / "champion_summary.csv", champion_rows, FINAL_CHAMPION_COLUMNS)
    _write_csv(summary_dir / "metric_summary.csv", metric_rows)
    _write_csv(summary_dir / "seed_replicate_summary.csv", champion_rows, FINAL_CHAMPION_COLUMNS)

    for table_name, rows in plot_rows_by_table.items():
        _write_csv(plot_dir / table_name, rows)
    for alias, source_name in LEGACY_PLOT_ALIASES.items():
        _write_csv(plot_dir / alias, plot_rows_by_table[source_name])

    report = {
        "study": "pair_stability",
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": report_attempt_id,
        "final_eval_attempt_id": final_eval_attempt_id,
        "n_final_eval_attempts": len(records),
        "n_metric_rows": len(metric_rows),
        "n_failure_rows": len(failure_rows),
        "summary_tables": {
            "final_champions.csv": len(champion_rows),
            "final_metrics_by_run.csv": len(metric_rows),
            "final_metrics_by_family.csv": len(family_rows),
            "failure_modes.csv": len(failure_rows),
            "resource_summary.csv": len(resource_rows),
        },
        "plot_tables": {name: len(rows) for name, rows in plot_rows_by_table.items()},
        "figures": figure_names,
        "caveats": [
            "final_report.py consumes 07_final_eval artifacts only and does not rerun models.",
            "Training curves are emitted only if 07_final_eval artifacts contain them; the current evaluator does not.",
            "Exact logabs references are propagated when record tables provide them; current final-eval records do not include exact_logabs.",
        ],
    }
    write_json(attempt / "final_report.json", report)
    (attempt / "report.md").write_text(_report_markdown(report, family_rows, failure_rows))
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), report_attempt_id)
    return {"attempt_dir": str(attempt), "report": report}


def _report_markdown(
    report: dict[str, Any],
    family_rows: Sequence[dict[str, Any]],
    failure_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Hooke Pair-Stability Final Report",
        "",
        "## Scope And Provenance",
        "",
        "This report consumes `07_final_eval` artifacts only. It does not load checkpoints or rerun models.",
        f"Final-eval attempts consumed: {report['n_final_eval_attempts']}.",
        "",
        "## Final Champion Summary",
        "",
        "See `summary_tables/final_champions.csv`.",
        "",
        "## Family-Level Ranking",
        "",
    ]
    if family_rows:
        lines.extend(
            [
                "| rank_overall | basis | normalization | energy_error_avg | local_energy_var_median |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for row in family_rows[:12]:
            lines.append(
                "| {rank_overall} | {basis} | {normalization} | {energy_error_avg} | {local_energy_var_median} |".format(
                    **row
                )
            )
    else:
        lines.append("No completed family rows were found.")
    lines.extend(
        [
            "",
            "## Energy And Local-Energy Results",
            "",
            "Energy figures use signed error relative to exact Hooke energy `E = 2`. Runtime is reported separately.",
            "",
            "## Cusp Diagnostics",
            "",
            "Cusp profile tables preserve center-of-mass and direction columns when present.",
            "",
            "## Tail Diagnostics",
            "",
            "Tail tables preserve path/direction columns. Exact log-amplitude references are included when records provide them.",
            "",
            "## Stratified Geometry Diagnostics",
            "",
            "Stratified geometry tables preserve stratum labels for per-stratum inspection.",
            "",
            "## Hooke-Orbital Diagnostics",
            "",
            "Hooke-orbital tables preserve `r12` and radius columns for relative-coordinate and center-of-mass analysis.",
            "",
            "## Symmetry Diagnostics",
            "",
            "See `plot_tables/symmetry_diagnostics.csv` and the symmetry figures.",
            "",
            "## Trace Diagnostics",
            "",
            "See `plot_tables/trace_diagnostics.csv` and the trace figures.",
            "",
            "## Training And Resource Summary",
            "",
            "See `summary_tables/resource_summary.csv`. Resource metrics are not mixed into model-quality rankings.",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "## Failure Modes",
            "",
            f"Suspicious or failed task rows: {len(failure_rows)}.",
            "",
            "## Next-Scan Implications",
            "",
            "Use family-level energy, local-energy variance, cusp/tail pathologies, and trace/symmetry diagnostics jointly. "
            "Do not select future scan directions from runtime/resource columns.",
            "",
            "## Tables And Figures",
            "",
            "Summary tables:",
        ]
    )
    for name, n_rows in report["summary_tables"].items():
        lines.append(f"- `summary_tables/{name}`: {n_rows} rows")
    lines.append("")
    lines.append("Plot tables:")
    for name, n_rows in report["plot_tables"].items():
        lines.append(f"- `plot_tables/{name}`: {n_rows} rows")
    lines.append("")
    lines.append("Figures:")
    for name in report["figures"]:
        lines.append(f"- `figures/{name}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-report arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-eval-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write final report artifacts."""

    args = parse_args(argv)
    result = build_report(
        results_root=args.results_root,
        report_attempt_id=args.attempt_id,
        final_eval_attempt_id=args.final_eval_attempt_id,
    )
    report = result["report"]
    print(
        f"[pair_stability] final report consumed {report['n_final_eval_attempts']} "
        f"final-eval attempts -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
