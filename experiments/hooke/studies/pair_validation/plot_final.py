"""Generate Hooke pair final-eval physics-sanity tables, plots, and report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .study_manifest import final_eval_report_dir, load_yaml
except ImportError:  # pragma: no cover - direct script execution
    from study_manifest import final_eval_report_dir, load_yaml


ARTIFACT_NAMES = (
    "diagnostics_index",
    "sampled_eval_table",
    "pair_distance_probe",
    "center_of_mass_probe",
    "exchange_trace",
    "rotation_trace",
    "trace_equivariance_trace",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run report generation CLI."""

    args = _parse_args(argv)
    plot_final(
        manifest_path=args.manifest,
        final_eval_dir=args.final_eval_dir,
        final_eval_runs_path=args.final_eval_runs,
        summary_path=args.summary,
        output_dir=args.output_dir,
    )
    return 0


def plot_final(
    *,
    manifest_path: str | Path,
    final_eval_dir: str | Path | None = None,
    final_eval_runs_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate final-eval tables, plots, and Markdown report from files only."""

    manifest = load_yaml(manifest_path)
    report_dir = Path(output_dir or final_eval_dir or final_eval_report_dir(manifest))
    runs_path = Path(final_eval_runs_path) if final_eval_runs_path is not None else report_dir / "final_eval_runs.csv"
    summary_csv = Path(summary_path) if summary_path is not None else report_dir / "final_benchmark_summary.csv"
    rows = _read_csv(runs_path)
    summary_rows = _read_csv(summary_csv) if summary_csv.exists() else []
    tables_dir = report_dir / "tables"
    plots_dir = report_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {"tables": [], "plots": [], "warnings": []}
    outputs["tables"].append(_write_energy_reference_table(rows, tables_dir / "energy_reference.csv"))
    outputs["tables"].append(_write_components_table(rows, tables_dir / "energy_components_and_virial.csv"))
    outputs["tables"].append(_write_local_energy_table(rows, tables_dir / "local_energy_summary.csv"))
    outputs["tables"].append(_write_probe_summary(rows, "pair_distance_probe", tables_dir / "pair_distance_probe_summary.csv"))
    outputs["tables"].append(_write_probe_summary(rows, "center_of_mass_probe", tables_dir / "center_of_mass_probe_summary.csv"))
    outputs["tables"].append(_write_exchange_summary(rows, tables_dir / "exchange_summary.csv"))
    outputs["tables"].append(_write_rotation_summary(rows, tables_dir / "rotation_summary.csv"))
    outputs["tables"].append(_write_trace_summary(rows, tables_dir / "trace_equivariance_summary.csv"))
    artifact_rows = _artifact_summary_rows(rows)
    outputs["tables"].append(_write_table(artifact_rows, tables_dir / "artifact_summary.csv"))

    outputs["plots"].extend(_write_energy_plots(rows, plots_dir))
    pair_rows = _read_artifact_csvs(rows, "pair_distance_probe")
    com_rows = _read_artifact_csvs(rows, "center_of_mass_probe")
    sampled_rows = _read_artifact_csvs(rows, "sampled_eval_table")
    outputs["plots"].extend(_write_pair_probe_plots(pair_rows, plots_dir))
    outputs["plots"].extend(_write_center_probe_plots(com_rows, plots_dir))
    if sampled_rows:
        max_points = int(_select(manifest, "physics_sanity.plots.max_scatter_points") or 50000)
        outputs["plots"].extend(_write_sampled_eval_plots(sampled_rows[:max_points], plots_dir))
    else:
        outputs["warnings"].append("sampled-eval table disabled or missing; sampled-eval plots skipped")

    outputs["warnings"].extend(_artifact_warnings(artifact_rows))
    report = _report(
        manifest_path=manifest_path,
        manifest=manifest,
        rows=rows,
        summary=summary_rows[0] if summary_rows else {},
        artifact_rows=artifact_rows,
        warnings=outputs["warnings"],
    )
    report_path = report_dir / "final_benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")
    outputs["report"] = str(report_path)
    return outputs


def _write_energy_reference_table(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    output = []
    for row in rows:
        output.append(
            {
                "train_seed": row.get("training_seed", ""),
                "eval_seed": row.get("eval_seed", ""),
                "energy": row.get("eval/energy", ""),
                "stderr": row.get("eval/energy_stderr", ""),
                "reference_energy": row.get("eval/reference_energy", ""),
                "energy_error": row.get("eval/energy_error", ""),
                "energy_abs_error": row.get("eval/energy_abs_error", ""),
            }
        )
    return _write_table(output, path)


def _write_components_table(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    quantities = {
        "kinetic": "eval/energy_term_kinetic",
        "harmonic_trap": "eval/energy_term_harmonic_trap",
        "electron_electron": "eval/energy_term_electron_electron",
        "total_energy": "eval/energy",
        "virial_residual": "eval/virial_residual",
        "virial_relative_residual": "eval/virial_relative_residual",
    }
    output = []
    for quantity, key in quantities.items():
        values = _finite_values(row.get(key) for row in rows)
        output.append(_aggregate_row(quantity, values))
    return _write_table(output, path)


def _write_local_energy_table(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    keys = {
        "local_energy_q001": "eval/local_energy_q001",
        "local_energy_q01": "eval/local_energy_q01",
        "local_energy_q50": "eval/local_energy_q50",
        "local_energy_q99": "eval/local_energy_q99",
        "local_energy_q999": "eval/local_energy_q999",
        "local_energy_error_q001": "eval/local_energy_error_q001",
        "local_energy_error_q01": "eval/local_energy_error_q01",
        "local_energy_error_q50": "eval/local_energy_error_q50",
        "local_energy_error_q99": "eval/local_energy_error_q99",
        "local_energy_error_q999": "eval/local_energy_error_q999",
        "local_energy_error_mean": "eval/local_energy_error_mean",
        "local_energy_abs_error_mean": "eval/local_energy_abs_error_mean",
        "finite_fraction": "eval/local_energy_finite_fraction",
        "nonfinite_count": "eval/local_energy_nonfinite_count",
    }
    output = []
    for quantity, key in keys.items():
        output.append(_aggregate_row(quantity, _finite_values(row.get(key) for row in rows)))
    return _write_table(output, path)


def _write_probe_summary(rows: Sequence[Mapping[str, Any]], artifact_name: str, path: Path) -> str:
    output = []
    for run in rows:
        probe_rows = _read_csv_if_present(run.get(f"artifact/{artifact_name}"))
        if not probe_rows:
            output.append({"run_dir": run.get("run_dir", ""), "warning": "missing or empty"})
            continue
        errors = _finite_values(row.get("model_local_energy_error") for row in probe_rows)
        aligned = [abs(value) for value in _finite_values(row.get("aligned_logabs_error") for row in probe_rows)]
        rel = [abs(value) for value in _finite_values(row.get("relative_abs_psi_error") for row in probe_rows)]
        output.append(
            {
                "run_dir": run.get("run_dir", ""),
                "max_local_energy_abs_error": max(abs(value) for value in errors) if errors else "",
                "q95_local_energy_abs_error": _quantile([abs(value) for value in errors], 0.95) if errors else "",
                "nonfinite_count": sum(1 for row in probe_rows if not _as_bool(row.get("finite"))),
                "max_aligned_logabs_error": max(aligned) if aligned else "",
                "max_relative_abs_psi_error": max(rel) if rel else "",
                "estimated_cusp_slope": _estimate_cusp_slope(probe_rows) if artifact_name == "pair_distance_probe" else "",
            }
        )
    return _write_table(output, path)


def _write_exchange_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    output = []
    for row in rows:
        output.append(
            {
                "run_dir": row.get("run_dir", ""),
                "contract": "symmetric_spatial_singlet",
                "max_abs_error": row.get("eval/checks/exchange/logabs_max_abs_error", ""),
                "mean_abs_error": row.get("eval/checks/exchange/logabs_mean_abs_error", ""),
                "failure_count": row.get("eval/checks/exchange/sign_failure_count", ""),
                "nonfinite_count": row.get("eval/checks/exchange/nonfinite_count", ""),
            }
        )
    return _write_table(output, path)


def _write_rotation_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    output = []
    for row in rows:
        output.append(
            {
                "run_dir": row.get("run_dir", ""),
                "check_type": "spatial_rotation",
                "max_abs_error": row.get("eval/checks/rotation/logabs_max_abs_error", ""),
                "mean_abs_error": row.get("eval/checks/rotation/logabs_mean_abs_error", ""),
                "local_energy_max_abs_error": row.get("eval/checks/rotation/local_energy_max_abs_error", ""),
                "local_energy_mean_abs_error": row.get("eval/checks/rotation/local_energy_mean_abs_error", ""),
                "failure_count": "",
                "nonfinite_count": row.get("eval/checks/rotation/nonfinite_count", ""),
            }
        )
    return _write_table(output, path)


def _write_trace_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    output = []
    for row in rows:
        output.append(
            {
                "run_dir": row.get("run_dir", ""),
                "check_type": "semantic_trace_equivariance",
                "max_abs_error": row.get("eval/checks/trace_equivariance/max_abs_error", ""),
                "mean_abs_error": row.get("eval/checks/trace_equivariance/mean_abs_error", ""),
                "failure_count": row.get("eval/checks/trace_equivariance/failure_count", ""),
                "nonfinite_count": "",
            }
        )
    return _write_table(output, path)


def _artifact_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for name in ARTIFACT_NAMES:
            path = str(row.get(f"artifact/{name}") or "")
            output.append(
                {
                    "run_dir": row.get("run_dir", ""),
                    "artifact_name": name,
                    "path": path,
                    "expected": row.get(f"artifact/{name}_expected", ""),
                    "enabled": row.get(f"artifact/{name}_enabled", ""),
                    "exists": row.get(f"artifact/{name}_exists", ""),
                    "readable": row.get(f"artifact/{name}_readable", ""),
                    "rows": _count_artifact_rows(path),
                    "warning": row.get(f"artifact/{name}_warning", ""),
                }
            )
    return output


def _write_energy_plots(rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> list[str]:
    plt = _pyplot()
    plots = []
    x = list(range(len(rows)))
    energies = [_as_float(row.get("eval/energy")) for row in rows]
    stderrs = [_as_float(row.get("eval/energy_stderr")) for row in rows]
    refs = _finite_values(row.get("eval/reference_energy") for row in rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, energies, yerr=stderrs, marker="o", linestyle="none")
    if refs:
        ax.axhline(refs[0], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("final eval run")
    ax.set_ylabel("energy")
    path = plots_dir / "energy_by_run.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))

    errors = [_as_float(row.get("eval/energy_error")) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, errors, yerr=stderrs, marker="o", linestyle="none")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("final eval run")
    ax.set_ylabel("energy error")
    path = plots_dir / "energy_error_by_run.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    return plots


def _write_pair_probe_plots(rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> list[str]:
    if not rows:
        return []
    plots = []
    plots.append(_scatter_plot(rows, "pair_distance", "model_local_energy", plots_dir / "probe_pair_distance_local_energy.png", hline=_first_finite(rows, "exact_local_energy")))
    plots.append(_scatter_plot(rows, "pair_distance", "model_logabs", plots_dir / "probe_pair_distance_logabs.png", exact_key="exact_logabs", sign_key="model_sign"))
    plots.append(_scatter_plot(rows, "pair_distance", "model_relative_abs_psi", plots_dir / "probe_pair_distance_relative_abs_psi.png", exact_key="exact_relative_abs_psi", sign_key="model_sign"))
    plots.append(_cusp_plot(rows, plots_dir / "probe_cusp_slope.png"))
    return plots


def _write_center_probe_plots(rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> list[str]:
    if not rows:
        return []
    return [
        _scatter_plot(rows, "center_of_mass_radius", "model_logabs", plots_dir / "probe_center_of_mass_logabs.png", exact_key="exact_logabs", sign_key="model_sign"),
        _scatter_plot(rows, "center_of_mass_radius", "model_relative_abs_psi", plots_dir / "probe_center_of_mass_relative_abs_psi.png", exact_key="exact_relative_abs_psi", sign_key="model_sign"),
    ]


def _write_sampled_eval_plots(rows: Sequence[Mapping[str, Any]], plots_dir: Path) -> list[str]:
    if not rows:
        return []
    plt = _pyplot()
    plots = []
    energies = _finite_values(row.get("local_energy") for row in rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(energies, bins=50)
    ax.set_xlabel("local energy")
    ax.set_ylabel("count")
    path = plots_dir / "local_energy_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots.append(str(path))
    plots.append(_scatter_plot(rows, "electron_distance", "local_energy", plots_dir / "local_energy_vs_electron_distance.png"))
    plots.append(_scatter_plot(rows, "center_of_mass_radius", "local_energy", plots_dir / "local_energy_vs_center_of_mass_radius.png"))
    return plots


def _scatter_plot(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    path: Path,
    *,
    exact_key: str | None = None,
    sign_key: str | None = None,
    hline: float | None = None,
) -> str:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if sign_key is None:
        x, y = _xy(rows, x_key, y_key)
        ax.scatter(x, y, s=10, alpha=0.75, label=y_key)
    else:
        for sign, marker in ((1.0, "o"), (-1.0, "x"), (0.0, "s")):
            signed = [row for row in rows if _as_float(row.get(sign_key)) == sign]
            x, y = _xy(signed, x_key, y_key)
            if x:
                ax.scatter(x, y, s=10, alpha=0.75, marker=marker, label=f"sign {sign:g}")
    if exact_key is not None:
        x, y = _xy(rows, x_key, exact_key)
        if x:
            ax.plot(x, y, color="black", linewidth=1, alpha=0.8, label=exact_key)
    if hline is not None and math.isfinite(hline):
        ax.axhline(hline, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _cusp_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    plt = _pyplot()
    by_group: dict[tuple[Any, Any], list[tuple[float, float]]] = {}
    for row in rows:
        key = (row.get("direction_id"), row.get("center_of_mass_radius"))
        pair = (_as_float(row.get("pair_distance")), _as_float(row.get("model_logabs")))
        if all(math.isfinite(value) for value in pair):
            by_group.setdefault(key, []).append(pair)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, values in by_group.items():
        values = sorted(values)
        slopes = []
        xs = []
        for (x0, y0), (x1, y1) in zip(values, values[1:]):
            if x1 != x0:
                xs.append(0.5 * (x0 + x1))
                slopes.append((y1 - y0) / (x1 - x0))
        if xs:
            ax.plot(xs, slopes, linewidth=1, alpha=0.7, label=str(key))
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("pair_distance")
    ax.set_ylabel("d log|psi| / d r12")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _report(
    *,
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    artifact_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    diagnostic_status = "PASS" if not warnings else "WARN"
    lines = [
        "# Hooke pair final benchmark report",
        "",
        "## Summary",
        "",
        f"- study: `{_select(manifest, 'study.name')} {_select(manifest, 'study.version')}`",
        f"- selected config: `{summary.get('selected_config_id', '')}`",
        f"- exact reference energy: `{_first_existing(rows, 'eval/reference_energy')}`",
        f"- aggregate final energy: `{summary.get('energy_mean', '')}`",
        f"- aggregate energy error: `{summary.get('energy_abs_error_mean', '')}`",
        f"- completed / failed final eval runs: `{len(completed)}` / `{len(rows) - len(completed)}`",
        f"- diagnostic status: `{diagnostic_status}`",
        "",
        "## Final Evaluation Inputs",
        "",
        f"- manifest path: `{manifest_path}`",
        "- selected_config path: `reports/03_select/selected_config.yaml`",
        "- final_eval_jobs path: `reports/05_final_eval/plans/final_eval_jobs.jsonl`",
        "- final_eval_runs path: `final_eval_runs.csv`",
        "- checkpoint source: `load.path`",
        f"- eval sampler budget: `{_select(manifest, 'final_evaluation.sampler')}`",
        "",
        "## Energy Reference Check",
        "",
        _markdown_table(_energy_reference_preview(rows)),
        "",
        "- plots/energy_by_run.png",
        "- plots/energy_error_by_run.png",
        "",
        "## Local-Energy Scalar Diagnostics",
        "",
        _markdown_table(_quantity_preview(rows)),
        "",
    ]
    if any(_as_bool(row.get("artifact/sampled_eval_table_exists")) for row in rows):
        lines.extend(
            [
                "- plots/local_energy_histogram.png",
                "- plots/local_energy_vs_electron_distance.png",
                "- plots/local_energy_vs_center_of_mass_radius.png",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> Sampled-eval table was not requested for this run. Local-energy diagnostics are scalar summaries from EnergyEvaluation.",
                "",
            ]
        )
    lines.extend(
        [
            "## Energy Components And Virial",
            "",
            "- tables/energy_components_and_virial.csv",
            "",
            "## Pair-Distance Probe",
            "",
            "- tables/pair_distance_probe_summary.csv",
            "- plots/probe_pair_distance_local_energy.png",
            "- plots/probe_pair_distance_logabs.png",
            "- plots/probe_pair_distance_relative_abs_psi.png",
            "- plots/probe_cusp_slope.png",
            "",
            "## Center-Of-Mass Probe",
            "",
            "- tables/center_of_mass_probe_summary.csv",
            "- plots/probe_center_of_mass_logabs.png",
            "- plots/probe_center_of_mass_relative_abs_psi.png",
            "",
            "## Position-Exchange Check",
            "",
            "- tables/exchange_summary.csv",
            "",
            "## Rotation Check",
            "",
            "- tables/rotation_summary.csv",
            "",
            "## Trace Equivariance Check",
            "",
            "- tables/trace_equivariance_summary.csv",
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings[:25])
    if not warnings:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- final_eval_runs.csv",
            "- final_benchmark_summary.csv",
            "- final_benchmark_summary.json",
            "- tables/",
            "- plots/",
        ]
    )
    lines.extend(f"- {row.get('path')}" for row in artifact_rows if row.get("artifact_name") == "diagnostics_index" and row.get("exists"))
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--final-eval-dir", type=Path, default=None)
    parser.add_argument("--final-eval-runs", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [{key: _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _read_csv_if_present(path: Any) -> list[dict[str, Any]]:
    if path in (None, ""):
        return []
    candidate = Path(str(path))
    if not candidate.is_file():
        return []
    return _read_csv(candidate)


def _read_artifact_csvs(rows: Sequence[Mapping[str, Any]], artifact_name: str) -> list[dict[str, Any]]:
    combined = []
    for row in rows:
        for item in _read_csv_if_present(row.get(f"artifact/{artifact_name}")):
            combined.append({"run_dir": row.get("run_dir", ""), **item})
    return combined


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> str:
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    return str(path)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if lowered in {"-inf", "-infinity"}:
        return -math.inf
    if lowered == "nan":
        return math.nan
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _as_float(value: Any) -> float:
    parsed = _parse_scalar(value)
    if parsed is None or isinstance(parsed, bool):
        return math.nan
    try:
        return float(parsed)
    except (TypeError, ValueError):
        return math.nan


def _as_bool(value: Any) -> bool:
    parsed = _parse_scalar(value)
    if isinstance(parsed, bool):
        return parsed
    if parsed is None:
        return False
    if isinstance(parsed, (int, float)):
        return bool(parsed)
    return str(parsed).strip().lower() in {"1", "true", "yes", "y"}


def _finite_values(values: Sequence[Any]) -> list[float]:
    output = []
    for value in values:
        number = _as_float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def _aggregate_row(quantity: str, values: Sequence[float]) -> dict[str, Any]:
    return {
        "quantity": quantity,
        "median": _median(values) if values else "",
        "mean": sum(values) / len(values) if values else "",
        "min": min(values) if values else "",
        "max": max(values) if values else "",
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (1.0 - (position - lower)) * ordered[lower] + (position - lower) * ordered[upper]


def _estimate_cusp_slope(rows: Sequence[Mapping[str, Any]]) -> float | str:
    pairs = sorted(
        (x, y)
        for x, y in ((_as_float(row.get("pair_distance")), _as_float(row.get("model_logabs"))) for row in rows)
        if math.isfinite(x) and math.isfinite(y)
    )
    slopes = [(y1 - y0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]) if x1 != x0]
    return _median(slopes[:5]) if slopes else ""


def _count_artifact_rows(path: str) -> int | str:
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_file():
        return ""
    if candidate.suffix == ".csv":
        return len(_read_csv(candidate))
    if candidate.suffix == ".jsonl":
        with candidate.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    return ""


def _artifact_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        warning = str(row.get("warning") or "")
        if warning and warning != "disabled":
            warnings.append(f"{row.get('artifact_name')}: {warning} ({row.get('path')})")
    return warnings


def _xy(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            x_values.append(x)
            y_values.append(y)
    return x_values, y_values


def _first_finite(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for row in rows:
        value = _as_float(row.get(key))
        if math.isfinite(value):
            return value
    return None


def _first_existing(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key) not in (None, ""):
            return row.get(key)
    return ""


def _select(container: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = container
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _energy_reference_preview(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "train_seed": row.get("training_seed", ""),
            "eval_seed": row.get("eval_seed", ""),
            "energy": row.get("eval/energy", ""),
            "stderr": row.get("eval/energy_stderr", ""),
            "reference": row.get("eval/reference_energy", ""),
            "error": row.get("eval/energy_error", ""),
            "abs_error": row.get("eval/energy_abs_error", ""),
        }
        for row in rows[:10]
    ]


def _quantity_preview(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        ("finite fraction", "eval/local_energy_finite_fraction"),
        ("q001", "eval/local_energy_q001"),
        ("q01", "eval/local_energy_q01"),
        ("q50", "eval/local_energy_q50"),
        ("q99", "eval/local_energy_q99"),
        ("q999", "eval/local_energy_q999"),
        ("local-energy error mean", "eval/local_energy_error_mean"),
        ("local-energy absolute error mean", "eval/local_energy_abs_error_mean"),
    ]
    return [{"quantity": name, "value": _median(_finite_values(row.get(key) for row in rows)) if rows else ""} for name, key in keys]


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "|  |\n| --- |\n|  |"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


__all__ = ["main", "plot_final"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
