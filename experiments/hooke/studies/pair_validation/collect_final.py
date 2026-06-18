"""Collect Hooke pair final-train and final-eval artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from . import collect
    from .study_manifest import (
        final_eval_report_dir,
        final_train_report_dir,
        load_jsonl,
        load_yaml,
        phase_run_root,
        phase_study_name,
        phase_study_phase,
        phase_study_version,
        run_kind_for_dir,
        select_report_dir,
        selected_hyperparameters,
    )
except ImportError:  # pragma: no cover - direct script execution
    import collect
    from study_manifest import (
        final_eval_report_dir,
        final_train_report_dir,
        load_jsonl,
        load_yaml,
        phase_run_root,
        phase_study_name,
        phase_study_phase,
        phase_study_version,
        run_kind_for_dir,
        select_report_dir,
        selected_hyperparameters,
    )


FINAL_EVAL_COLUMNS = (
    "run_dir",
    "status",
    "study_name",
    "study_version",
    "study_phase",
    "config_id",
    "training_seed",
    "eval_seed",
    "load.path",
    "checkpoint_exists",
    "eval/energy/local_energy_mean",
    "eval/energy/local_energy_stderr",
    "eval/energy/local_energy_variance",
    "eval/energy/reference_energy",
    "eval/energy/energy_error",
    "eval/energy/energy_abs_error",
    "eval/energy/term/kinetic_mean",
    "eval/energy/term/harmonic_trap_mean",
    "eval/energy/term/electron_electron_mean",
    "eval/energy/local_energy_finite_fraction",
    "eval/energy/local_energy_q001",
    "eval/energy/local_energy_q01",
    "eval/energy/local_energy_q05",
    "eval/energy/local_energy_q50",
    "eval/energy/local_energy_q95",
    "eval/energy/local_energy_q99",
    "eval/energy/local_energy_q999",
    "eval/energy/local_energy_min",
    "eval/energy/local_energy_max",
    "eval/energy/local_energy_n_finite",
    "eval/energy/local_energy_n_total",
    "eval/energy/local_energy_nonfinite_count",
    "eval/virial_residual",
    "eval/virial_relative_residual",
    "eval/cusp/c_minus_1_abs_max",
    "eval/cusp/c_minus_1_abs_q95",
    "eval/cusp/cusp_even_slope_abs_error",
    "eval/cusp/cusp_even_slope_abs_error_max",
    "eval/cusp/nonfinite_local_energy_count",
    "eval/tail/local_energy_q95",
    "eval/tail/local_energy_q99",
    "eval/tail/local_energy_finite_fraction",
    "eval/tail/nonfinite_local_energy_count",
    "eval/tail/tail_outlier_count",
    "eval/exchange/logabs_max_abs_error",
    "eval/exchange/logabs_mean_abs_error",
    "eval/exchange/sign_failure_count",
    "eval/exchange/failure_count",
    "eval/rotation/logabs_max_abs_error",
    "eval/rotation/logabs_mean_abs_error",
    "eval/rotation/local_energy_max_abs_error",
    "eval/rotation/local_energy_mean_abs_error",
    "eval/rotation/sign_failure_count",
    "eval/rotation/failure_count",
    "eval/full_model_equivariance/logabs_max_abs_error",
    "eval/full_model_equivariance/logabs_mean_abs_error",
    "eval/full_model_equivariance/sign_failure_count",
    "eval/full_model_equivariance/failure_count",
    "eval/trace_equivariance/max_abs_error",
    "eval/trace_equivariance/mean_abs_error",
    "eval/trace_equivariance/failure_count",
    "eval/trace_equivariance/missing_key_count",
    "eval/trace_equivariance/extra_key_count",
    "eval/feature_trace/feature_rms_max",
    "eval/feature_trace/feature_rms_q95",
    "eval/feature_trace/feature_max_abs_max",
    "eval/feature_trace/feature_nonfinite_count",
    "eval/readout_trace/pfaffian_near_zero_count",
    "eval/readout_trace/condition_number_q95",
    "eval/readout_trace/condition_number_max",
    "eval/readout_trace/readout_nonfinite_count",
    "artifact/diagnostics_index",
    "artifact/sampled_eval_table",
    "artifact/diagnostics_index_exists",
    "artifact/sampled_eval_table_exists",
    "artifact/diagnostics_index_readable",
    "artifact/sampled_eval_table_readable",
    "artifact/diagnostics_index_enabled",
    "artifact/sampled_eval_table_enabled",
    "artifact/diagnostics_index_expected",
    "artifact/sampled_eval_table_expected",
    "artifact/diagnostics_index_warning",
    "artifact/sampled_eval_table_warning",
    "eval/sampler/acceptance_rate",
    "eval/sampler/n_walkers",
    "eval/sampler/burn_in",
    "eval/sampler/n_steps",
    "eval/sampler/proposal_scale",
    "eval/sampler/radius_mean",
    "eval/sampler/radius_q99",
    "eval/sampler/radius_max",
    "eval/sampler/electron_distance_q01",
    "eval/sampler/electron_distance_min",
    "eval/sampler/position_rms",
    "eval/perf/wall_time_sec",
    "runtime/wall_time_sec",
    "git/sha",
    "wandb/run_id",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run final collector CLI."""

    args = _parse_args(argv)
    collect_final(
        manifest_path=args.manifest,
        final_train_root=args.final_train_root,
        final_eval_root=args.final_eval_root,
        output_dir=args.output_dir,
        final_eval_jobs_path=args.final_eval_jobs,
        selected_config_path=args.selected_config,
        include_smoke=args.include_smoke,
        strict_artifacts=args.strict_artifacts,
    )
    return 0


def collect_final(
    *,
    manifest_path: str | Path,
    final_train_root: str | Path | None = None,
    final_eval_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    final_eval_jobs_path: str | Path | None = None,
    selected_config_path: str | Path | None = None,
    include_smoke: bool = False,
    strict_artifacts: bool = False,
) -> dict[str, Any]:
    """Collect final artifacts and write CSV/JSON/Markdown outputs."""

    manifest = load_yaml(manifest_path)
    train_output = Path(output_dir) if output_dir is not None else Path(final_train_report_dir(manifest))
    eval_output = Path(output_dir) if output_dir is not None else Path(final_eval_report_dir(manifest))
    train_output.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)
    train_root = Path(final_train_root) if final_train_root is not None else Path(phase_run_root(manifest, "final_train"))
    eval_root = Path(final_eval_root) if final_eval_root is not None else Path(phase_run_root(manifest, "final_eval"))

    final_train_rows = _collect_train_rows(manifest, train_root, include_smoke=include_smoke)
    final_eval_rows = _collect_eval_rows(manifest, eval_root, include_smoke=include_smoke)
    if final_eval_jobs_path is not None:
        _verify_job_checkpoints(final_eval_jobs_path)
    if strict_artifacts:
        _raise_for_strict_artifact_failures(final_eval_rows)

    _write_table(final_train_rows, train_output / "final_train_runs.csv", columns=collect.REQUIRED_COLUMNS)
    _write_jsonl(final_train_rows, train_output / "final_train_runs.jsonl")
    _write_table(final_eval_rows, eval_output / "final_eval_runs.csv", columns=FINAL_EVAL_COLUMNS)
    _write_jsonl(final_eval_rows, eval_output / "final_eval_runs.jsonl")

    if selected_config_path is None:
        default_selected_config = Path(select_report_dir(manifest)) / "selected_config.yaml"
        selected_config_path = default_selected_config if default_selected_config.exists() else None
    selected = load_yaml(selected_config_path) if selected_config_path is not None else {}
    summary = _summary(final_train_rows, final_eval_rows, selected)
    _write_table([summary], eval_output / "final_benchmark_summary.csv", columns=tuple(summary))
    with (eval_output / "final_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (eval_output / "final_benchmark_report.md").write_text(
        _report(manifest, selected, final_train_rows, final_eval_rows, summary),
        encoding="utf-8",
    )
    return {"final_train_runs": final_train_rows, "final_eval_runs": final_eval_rows, "summary": summary}


def _collect_train_rows(
    manifest: Mapping[str, Any],
    root: str | Path,
    *,
    include_smoke: bool,
) -> list[dict[str, Any]]:
    study_name = phase_study_name(manifest, "final_train")
    study_version = phase_study_version(manifest, "final_train")
    study_phase = phase_study_phase(manifest, "final_train")
    rows = []
    run_root = Path(root)
    for run_dir in _discover_run_dirs(run_root):
        if not include_smoke and run_kind_for_dir(run_root, run_dir) == "smoke":
            continue
        row = collect.collect_run_dir(run_dir)
        if (
            row.get("study_name") == study_name
            and row.get("study_version") == study_version
            and row.get("study_phase") in (None, "", study_phase)
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("run_dir", "")))


def _collect_eval_rows(
    manifest: Mapping[str, Any],
    root: str | Path,
    *,
    include_smoke: bool,
) -> list[dict[str, Any]]:
    study_name = phase_study_name(manifest, "final_eval")
    study_version = phase_study_version(manifest, "final_eval")
    study_phase = phase_study_phase(manifest, "final_eval")
    rows = []
    run_root = Path(root)
    for run_dir in _discover_run_dirs(run_root):
        if not include_smoke and run_kind_for_dir(run_root, run_dir) == "smoke":
            continue
        row = _collect_eval_run_dir(run_dir, artifact_requirements=_artifact_requirements(manifest))
        if (
            row.get("study_name") == study_name
            and row.get("study_version") == study_version
            and row.get("study_phase") in (None, "", study_phase)
        ):
            rows.append(row)
    return sorted(rows, key=lambda row: str(row.get("run_dir", "")))


def _collect_eval_run_dir(
    run_dir: str | Path,
    *,
    artifact_requirements: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = collect._load_yaml_if_present(run_path / "resolved_config.yaml")
    metrics = _read_metrics(run_path)
    metadata = collect._load_json_if_present(run_path / "metadata.json")
    status_artifact = collect._load_json_if_present(run_path / "status.json")
    run_start = collect._load_json_if_present(run_path / "run_start.json")

    row: dict[str, Any] = {column: None for column in FINAL_EVAL_COLUMNS}
    row["run_dir"] = str(run_path)
    row["study_name"] = _select(cfg, "study.name")
    row["study_version"] = _select(cfg, "study.version")
    row["study_phase"] = _select(cfg, "study.phase")
    row["config_id"] = _select(cfg, "study.config_id")
    row["training_seed"] = _select(cfg, "evaluation.training_seed")
    row["eval_seed"] = _select(cfg, "runtime.seed")
    row["load.path"] = _select(cfg, "load.path")
    row["checkpoint_exists"] = bool(row["load.path"] and Path(str(row["load.path"])).exists())
    row.update({key: metrics.get(key) for key in FINAL_EVAL_COLUMNS if key in metrics})
    row.update(_derive_virial_metrics(row))
    row.update(_collect_artifact_columns(run_path, requirements=artifact_requirements or {}))
    row["status"] = _classify_eval_status(run_path, metrics, metadata, status_artifact)
    row["git/sha"] = _select(run_start, "git.sha") or metadata.get("git_commit")
    row["wandb/run_id"] = _select(metadata, "wandb.run_id") or _select(metadata, "wandb_run_id")
    return row


def _classify_eval_status(
    run_dir: Path,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    status_artifact: Mapping[str, Any],
) -> str:
    status_values = {
        str(status_artifact.get("status", "")).lower(),
        str(metadata.get("status", "")).lower(),
    }
    if not collect._has_metric_file(run_dir):
        return "missing_metrics"
    if metrics.get("eval/energy/local_energy_mean") is None:
        return "missing_eval"
    if "completed" in status_values:
        return "completed"
    if (run_dir / "error.json").exists() or status_values.intersection({"failed", "error", "exception"}):
        return "failed"
    return "incomplete"


def _verify_job_checkpoints(path: str | Path) -> None:
    missing = [str(row.get("checkpoint_path")) for row in load_jsonl(path) if not Path(str(row.get("checkpoint_path"))).exists()]
    if missing:
        raise FileNotFoundError("final_eval jobs point to missing checkpoint paths: " + ", ".join(missing[:5]))


def _derive_virial_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    kinetic = _as_float(row.get("eval/energy/term/kinetic_mean"), default=math.nan)
    harmonic = _as_float(row.get("eval/energy/term/harmonic_trap_mean"), default=math.nan)
    electron_electron = _as_float(row.get("eval/energy/term/electron_electron_mean"), default=math.nan)
    if not all(math.isfinite(value) for value in (kinetic, harmonic, electron_electron)):
        return {"eval/virial_residual": None, "eval/virial_relative_residual": None}
    residual = 2.0 * kinetic - 2.0 * harmonic + electron_electron
    denominator = abs(2.0 * kinetic) + abs(2.0 * harmonic) + abs(electron_electron)
    relative = abs(residual) / denominator if denominator else 0.0
    return {"eval/virial_residual": residual, "eval/virial_relative_residual": relative}


_KNOWN_ARTIFACT_DEFAULTS = {
    "diagnostics_index": "diagnostics/index.json",
    "sampled_eval_table": "diagnostics/eval/energy/sampled_eval_table.csv",
}


def _artifact_requirements(manifest: Mapping[str, Any]) -> dict[str, bool]:
    configured = _select(manifest, "physics_sanity.artifacts") or {}
    requirements = {"diagnostics_index": True}
    if isinstance(configured, Mapping):
        for name, policy in configured.items():
            required = False
            if isinstance(policy, Mapping):
                required = bool(policy.get("required", False))
            requirements[str(name)] = required
    return requirements


def _collect_artifact_columns(run_dir: Path, *, requirements: Mapping[str, bool]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    index_path = run_dir / "diagnostics" / "index.json"
    index_payload = collect._load_json_if_present(index_path)
    index_entries: dict[str, Mapping[str, Any]] = {}
    for entry in index_payload.get("artifacts", []) or []:
        if isinstance(entry, Mapping) and entry.get("name"):
            index_entries[str(entry.get("name"))] = entry
    for task in index_payload.get("tasks", []) or []:
        if not isinstance(task, Mapping):
            continue
        for entry in task.get("artifacts", []) or []:
            if isinstance(entry, Mapping) and entry.get("name"):
                index_entries[str(entry.get("name"))] = entry
    index_entry = {
        "path": "diagnostics/index.json",
        "enabled": True,
        "expected": True,
        "exists": index_path.exists(),
        "readable": index_path.is_file() and os.access(index_path, os.R_OK),
        "warning": "" if index_path.exists() else "missing",
    }
    entries = {"diagnostics_index": index_entry, **index_entries}
    for name, default_path in _KNOWN_ARTIFACT_DEFAULTS.items():
        entry = entries.get(name, {})
        raw_path = entry.get("path") or default_path
        artifact_path = _resolve_artifact_path(run_dir, raw_path)
        enabled = bool(entry.get("enabled", True))
        expected = bool(requirements.get(name, entry.get("expected", False)))
        exists = artifact_path.exists()
        readable = artifact_path.is_file() and os.access(artifact_path, os.R_OK)
        warning = str(entry.get("warning") or "")
        if not warning:
            if not enabled:
                warning = "disabled"
            elif expected and not exists:
                warning = "missing"
            elif exists and not readable:
                warning = "unreadable"
        columns[f"artifact/{name}"] = str(artifact_path)
        columns[f"artifact/{name}_exists"] = exists
        columns[f"artifact/{name}_readable"] = readable
        columns[f"artifact/{name}_enabled"] = enabled
        columns[f"artifact/{name}_expected"] = expected
        columns[f"artifact/{name}_warning"] = warning
    return columns


def _resolve_artifact_path(run_dir: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else run_dir / path


def _raise_for_strict_artifact_failures(rows: Sequence[Mapping[str, Any]]) -> None:
    failures = []
    for row in rows:
        run_dir = row.get("run_dir", "")
        for name in _KNOWN_ARTIFACT_DEFAULTS:
            expected = _as_bool(row.get(f"artifact/{name}_expected"))
            enabled = _as_bool(row.get(f"artifact/{name}_enabled"))
            exists = _as_bool(row.get(f"artifact/{name}_exists"))
            readable = _as_bool(row.get(f"artifact/{name}_readable"))
            if expected and enabled and (not exists or not readable):
                failures.append(f"{run_dir}: {name} {row.get(f'artifact/{name}_warning') or 'missing/unreadable'}")
    if failures:
        raise FileNotFoundError("strict artifact validation failed: " + "; ".join(failures[:10]))


def _summary(
    final_train_rows: Sequence[Mapping[str, Any]],
    final_eval_rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    completed = [row for row in final_eval_rows if row.get("status") == "completed"]
    energies = [_as_float(row.get("eval/energy/local_energy_mean"), default=math.nan) for row in completed]
    finite_energies = [value for value in energies if math.isfinite(value)]
    errors = [_as_float(row.get("eval/energy/energy_abs_error"), default=math.nan) for row in completed]
    finite_errors = [value for value in errors if math.isfinite(value)]
    stderrs = [_as_float(row.get("eval/energy/local_energy_stderr"), default=math.nan) for row in completed]
    finite_stderrs = [value for value in stderrs if math.isfinite(value)]
    return {
        "selected_config_id": _select(selected, "selected.config_id") or "",
        "final_train_runs": len(final_train_rows),
        "final_eval_runs": len(final_eval_rows),
        "final_eval_completed": len(completed),
        "final_eval_failed_or_incomplete": len(final_eval_rows) - len(completed),
        "energy_mean": _mean(finite_energies),
        "energy_median": _median(finite_energies),
        "energy_min": min(finite_energies) if finite_energies else "",
        "energy_max": max(finite_energies) if finite_energies else "",
        "energy_seed_spread": (max(finite_energies) - min(finite_energies)) if len(finite_energies) >= 2 else "",
        "energy_stderr_median": _median(finite_stderrs),
        "energy_abs_error_mean": _mean(finite_errors),
        "energy_abs_error_median": _median(finite_errors),
    }


def _report(
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    final_train_rows: Sequence[Mapping[str, Any]],
    final_eval_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> str:
    selected_hparams = selected_hyperparameters(selected)
    lines = [
        f"# Final benchmark report: {_select(manifest, 'study.name')} {_select(manifest, 'study.version')}",
        "",
        "## Summary",
        "",
        f"- selected config: `{summary.get('selected_config_id', '')}`",
        f"- final train seeds: `{_seeds(final_train_rows, 'runtime.seed')}`",
        f"- final eval seeds: `{_seeds(final_eval_rows, 'eval_seed')}`",
        f"- completion/failure count: `{summary['final_eval_completed']}` / `{summary['final_eval_failed_or_incomplete']}`",
        f"- final energy estimate: `{_format(summary['energy_mean'])}`",
        f"- exact error, if available: `{_format(summary['energy_abs_error_mean'])}`",
        "",
        "## Selected config",
        "",
    ]
    if selected_hparams:
        lines.extend(f"- `{key}`: `{value}`" for key, value in selected_hparams.items())
    else:
        lines.append("- not provided")
    sampler = _select(manifest, "final_evaluation.sampler") or {}
    lines.extend(
        [
            "",
            "## Final protocol",
            "",
            "- final train checkpoint source: `final_train_runs.csv`",
            f"- eval sampler budget: `{sampler}`",
            "- load.mode: `model_only`",
            "",
            "## Results",
            "",
        ]
    )
    lines.extend(_results_table(final_eval_rows[:10]))
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- mean/median energy: `{_format(summary['energy_mean'])}` / `{_format(summary['energy_median'])}`",
            f"- seed-to-seed spread: `{_format(summary['energy_seed_spread'])}`",
            f"- MC stderr median: `{_format(summary['energy_stderr_median'])}`",
            f"- exact-reference error: `{_format(summary['energy_abs_error_mean'])}`",
            "",
            "## Diagnostics",
            "",
            f"- failed/incomplete eval runs: `{summary['final_eval_failed_or_incomplete']}`",
            f"- checkpoint paths missing in collected eval configs: `{sum(1 for row in final_eval_rows if not row.get('checkpoint_exists'))}`",
            "",
            "## Artifacts",
            "",
            "- final_train_runs.csv",
            "- final_eval_runs.csv",
            "- final_benchmark_summary.csv",
            "- final_benchmark_summary.json",
        ]
    )
    return "\n".join(lines) + "\n"


def _results_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    table = [
        "| train_seed | eval_seed | eval/energy/local_energy_mean | stderr | abs_error | acceptance_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row.get("training_seed") or ""),
                    str(row.get("eval_seed") or ""),
                    _format(row.get("eval/energy/local_energy_mean")),
                    _format(row.get("eval/energy/local_energy_stderr")),
                    _format(row.get("eval/energy/energy_abs_error")),
                    _format(row.get("eval/sampler/acceptance_rate")),
                ]
            )
            + " |"
        )
    if not rows:
        table.append("|  |  |  |  |  |  |")
    return table


def _discover_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    run_dirs = []
    for path in run_root.rglob("resolved_config.yaml"):
        if "checkpoints" in path.relative_to(run_root).parts:
            continue
        run_dirs.append(path.parent)
    return sorted(run_dirs)


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for path in (run_dir / "metrics.csv", run_dir / "metrics.jsonl"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    namespace = str(row.get("namespace") or "").strip("/")
                    key = str(row.get("key") or "").strip("/")
                    if namespace and key:
                        metrics[f"{namespace}/{key}"] = collect.parse_scalar(row.get("value"))
        else:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    namespace = str(record.get("namespace") or "").strip("/")
                    values = record.get("metrics") or {}
                    if isinstance(values, Mapping):
                        for key, value in values.items():
                            metrics[f"{namespace}/{key}"] = value
    return metrics


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path, *, columns: Sequence[str]) -> None:
    extra = sorted({key for row in rows for key in row if key not in columns})
    fieldnames = [*columns, *extra]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--final-train-root", type=Path, default=None)
    parser.add_argument("--final-eval-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--final-eval-jobs", type=Path)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--strict-artifacts", action="store_true")
    return parser.parse_args(argv)


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_float(value: Any, *, default: float) -> float:
    parsed = collect.parse_scalar(value)
    if parsed is None or isinstance(parsed, bool):
        return default
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    parsed = collect.parse_scalar(value)
    if isinstance(parsed, bool):
        return parsed
    if parsed is None:
        return False
    if isinstance(parsed, (int, float)):
        return bool(parsed)
    text = str(parsed).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _mean(values: Sequence[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _median(values: Sequence[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _seeds(rows: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    return sorted({row.get(key) for row in rows if row.get(key) not in (None, "")})


def _format(value: Any) -> str:
    if value in (None, ""):
        return ""
    number = _as_float(value, default=math.nan)
    if math.isfinite(number):
        return f"{number:.12g}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = ["FINAL_EVAL_COLUMNS", "collect_final", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
