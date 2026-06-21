"""Build final pair-stability report artifacts from ``07_final_eval`` outputs.

This stage consumes only final-evaluation artifacts. It does not import model
code and does not rerun training or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

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

PLOT_TABLES = {
    "cusp_profiles.csv": ("cusp", "cusp_profiles.csv"),
    "tail_profiles.csv": ("tail", "tail_profiles.csv"),
    "stratified_metrics.csv": ("stratified_geometry", "stratified_metrics.csv"),
    "hooke_orbital_metrics.csv": ("hooke_orbital", "hooke_orbital_metrics.csv"),
    "mcmc_energy_samples.csv": ("energy", "mcmc_energy_samples.csv"),
    "symmetry_metrics.csv": ("full_model_antisymmetry", "transform_records.csv"),
    "spatial_exchange_metrics.csv": ("spatial_exchange_symmetry", "transform_records.csv"),
    "rotation_metrics.csv": ("rotation_consistency", "transform_records.csv"),
    "trace_metrics.csv": ("trace_equivariance", "trace_records.csv"),
    "feature_trace_metrics.csv": ("feature_trace_stability", "trace_records.csv"),
    "readout_trace_metrics.csv": ("readout_trace_stability", "trace_records.csv"),
}


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
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            rows.append({"namespace": namespace, "metric": str(key), "value": _csv_value(value)})
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return json.dumps(value, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_table_rows(final_run_id: str, attempt_id: str, attempt_dir: Path, table_name: str) -> list[dict[str, Any]]:
    task_dir, filename = PLOT_TABLES[table_name]
    rows = _read_csv(attempt_dir / task_dir / filename)
    for row in rows:
        row.setdefault("final_run_id", final_run_id)
        row.setdefault("final_eval_attempt_id", attempt_id)
        row["final_run_id"] = final_run_id
        row["final_eval_attempt_id"] = attempt_id
        row["task"] = task_dir
    return rows


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
    (attempt / "summary_tables").mkdir(parents=True, exist_ok=True)
    (attempt / "plot_tables").mkdir(parents=True, exist_ok=True)
    (attempt / "figures").mkdir(parents=True, exist_ok=True)

    eval_attempts = _iter_final_eval_attempts(results_root, final_eval_attempt_id)
    champion_rows = []
    seed_rows = []
    metric_rows = []
    plot_rows_by_table = {name: [] for name in PLOT_TABLES}

    for final_run_id, attempt_id, attempt_dir in eval_attempts:
        job = read_json(attempt_dir / "source_final_job.json") if (attempt_dir / "source_final_job.json").is_file() else {}
        checkpoint = read_json(attempt_dir / "evaluated_checkpoint.json") if (attempt_dir / "evaluated_checkpoint.json").is_file() else {}
        status = _status_of(attempt_dir)
        champion_rows.append(
            {
                "final_run_id": final_run_id,
                "final_eval_attempt_id": attempt_id,
                "status": status,
                "winner_kind": job.get("winner_kind", ""),
                "architecture": job.get("architecture", ""),
                "basis_envelope": job.get("basis_envelope", ""),
                "normalization": job.get("normalization", ""),
                "lr": job.get("lr", ""),
                "channels": job.get("channels", ""),
                "replicate_index": job.get("replicate_index", ""),
                "resolved_checkpoint_dir": checkpoint.get("resolved_checkpoint_dir", ""),
            }
        )
        seed_rows.append(
            {
                "final_run_id": final_run_id,
                "replicate_index": job.get("replicate_index", ""),
                "final_train_sampler_seed": job.get("final_train_sampler_seed", ""),
                "final_train_model_seed": job.get("final_train_model_seed", ""),
                "final_eval_seed": job.get("final_eval_seed", ""),
                "status": status,
            }
        )
        for row in _read_metrics_jsonl(attempt_dir / "metrics.jsonl"):
            row.update(final_run_id=final_run_id, final_eval_attempt_id=attempt_id)
            metric_rows.append(row)
        for table_name in PLOT_TABLES:
            plot_rows_by_table[table_name].extend(_plot_table_rows(final_run_id, attempt_id, attempt_dir, table_name))

    _write_csv(attempt / "summary_tables" / "champion_summary.csv", champion_rows)
    _write_csv(attempt / "summary_tables" / "metric_summary.csv", metric_rows)
    _write_csv(attempt / "summary_tables" / "seed_replicate_summary.csv", seed_rows)
    for table_name, rows in plot_rows_by_table.items():
        _write_csv(attempt / "plot_tables" / table_name, rows)

    report = {
        "study": "pair_stability",
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": report_attempt_id,
        "final_eval_attempt_id": final_eval_attempt_id,
        "n_final_eval_attempts": len(eval_attempts),
        "n_metric_rows": len(metric_rows),
        "plot_tables": {name: len(rows) for name, rows in plot_rows_by_table.items()},
    }
    write_json(attempt / "final_report.json", report)
    (attempt / "report.md").write_text(_report_markdown(report))
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), report_attempt_id)
    return {"attempt_dir": str(attempt), "report": report}


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hooke Pair-Stability Final Report",
        "",
        f"- Final eval attempts: {report['n_final_eval_attempts']}",
        f"- Metric rows: {report['n_metric_rows']}",
        "",
        "## Plot Tables",
        "",
    ]
    for name, n_rows in report["plot_tables"].items():
        lines.append(f"- `{name}`: {n_rows} rows")
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
