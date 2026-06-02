"""Create Matplotlib figures from Hooke experiment CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPL_CONFIG_DIR = Path("/tmp/spenn_mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_FIGURE_DIR = ROOT / "experiments/hooke/figures"


def main() -> None:
    """Generate all Hooke debug figures."""

    args = _parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "singlet": args.singlet_run or _latest_run("hooke_singlet"),
        "triplet": args.triplet_run or _latest_run("hooke_triplet"),
        "singlet_spenn": args.singlet_spenn_run or _latest_run("hooke_singlet_spenn"),
        "triplet_spenn": args.triplet_spenn_run or _latest_run("hooke_triplet_spenn"),
    }
    for sector, run_dir in runs.items():
        generate_sector_figures(sector, run_dir=run_dir, figure_dir=args.figure_dir)
    generate_summary_figures(runs, figure_dir=args.figure_dir)
    print(f"wrote figures to {args.figure_dir}")


def generate_sector_figures(sector: str, *, run_dir: Path, figure_dir: Path) -> None:
    """Generate figures for one Hooke run.

    Parameters
    ----------
    sector : str
        Hooke sector label used in output filenames.
    run_dir : pathlib.Path
        Run artifact directory containing ``metrics`` and ``plots`` folders.
    figure_dir : pathlib.Path
        Destination directory for figure files.
    """

    sector_dir = figure_dir / sector
    sector_dir.mkdir(parents=True, exist_ok=True)
    energy_trace_path = run_dir / "metrics/train_metrics.csv" if sector.endswith("_spenn") else run_dir / "metrics/energy_trace.csv"
    if not energy_trace_path.exists():
        energy_trace_path = run_dir / "metrics/energy_trace.csv"
    energy = _read_csv(energy_trace_path)
    energy_x_label = "training step" if energy_trace_path.name == "train_metrics.csv" else "production block"
    comparison_metrics = run_dir / "metrics/comparison_metrics.csv"
    metrics_path = comparison_metrics if comparison_metrics.exists() else run_dir / "metrics/train_metrics.csv"
    metrics = _read_csv(metrics_path)[0]
    local_hist = _read_csv(run_dir / "plots/local_energy_histogram.csv")
    r12_hist = _read_csv(run_dir / "plots/r12_histogram.csv")
    cusp = _read_csv(run_dir / "plots/cusp_diagnostic_plot.csv")
    radial = _read_csv(run_dir / "plots/wavefunction_radial_cut.csv")

    _line_plot(
        sector_dir / f"hooke_{sector}_energy_trace",
        title=f"Hooke {sector}: energy trace",
        x=_column(energy, "step"),
        series=[
            ("energy mean", _column_any(energy, ("energy/mean", "spenn/energy/mean"))),
            ("exact", _column_any(energy, ("energy/exact", "exact/energy"))),
        ],
        x_label=energy_x_label,
        y_label="energy",
    )
    _line_plot(
        sector_dir / f"hooke_{sector}_local_energy_variance",
        title=f"Hooke {sector}: local-energy variance",
        x=_column(energy, "step"),
        series=[("variance", _column_any(energy, ("local_energy/variance", "spenn/local_energy/variance")))],
        x_label=energy_x_label,
        y_label="variance",
        log_y=True,
    )
    _line_plot(
        sector_dir / f"hooke_{sector}_acceptance_rate",
        title=f"Hooke {sector}: acceptance rate",
        x=_column(energy, "step"),
        series=[("acceptance", _column(energy, "sampler/acceptance_rate"))],
        x_label="production block",
        y_label="acceptance rate",
        y_limits=(0.0, 1.0),
    )
    _histogram_plot(
        sector_dir / f"hooke_{sector}_local_energy_histogram",
        title=f"Hooke {sector}: local-energy histogram",
        hist=local_hist,
        x_label="local energy",
    )
    _histogram_plot(
        sector_dir / f"hooke_{sector}_r12_histogram",
        title=f"Hooke {sector}: pair-distance histogram",
        hist=r12_hist,
        x_label="r12",
    )
    cusp_r12 = _column(cusp, "r12")
    cusp_logabs = _column_any(cusp, ("factored_logabs", "spenn_factored_logabs"))
    target = cusp[0]["target_slope"]
    target_line = [cusp_logabs[0] + target * (r12 - cusp_r12[0]) for r12 in cusp_r12]
    _line_plot(
        sector_dir / f"hooke_{sector}_cusp_diagnostic",
        title=f"Hooke {sector}: cusp diagnostic",
        x=cusp_r12,
        series=[
            ("factored logabs", cusp_logabs),
            ("target slope", target_line),
        ],
        x_label="r12",
        y_label="factored logabs",
    )
    _line_plot(
        sector_dir / f"hooke_{sector}_wavefunction_radial_cut",
        title=f"Hooke {sector}: wavefunction radial cut",
        x=_column(radial, "r12"),
        series=_radial_series(radial),
        x_label="r12",
        y_label="centered logabs",
    )
    _symmetry_plot(sector_dir / f"hooke_{sector}_exchange_symmetry_error", sector=sector, metrics=metrics)


def generate_summary_figures(runs: dict[str, Path], *, figure_dir: Path) -> None:
    """Generate compact cross-sector Hooke report figures.

    Parameters
    ----------
    runs : dict
        Mapping from report labels to run artifact directories.
    figure_dir : pathlib.Path
        Destination directory for figure files.
    """

    summary_dir = figure_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    singlet = _summary_metrics(runs["singlet_spenn"])
    triplet = _summary_metrics(runs["triplet_spenn"])
    _bar_plot(
        summary_dir / "hooke_spenn_energy_abs_error",
        title="Hooke SpENN: energy absolute error",
        names=["singlet", "triplet"],
        values=[
            _metric_any(singlet, ("comparison/energy_abs_error",)),
            _metric_any(triplet, ("comparison/energy_abs_error",)),
        ],
        y_label="|E - E_exact|",
        log_y=True,
    )
    _bar_plot(
        summary_dir / "hooke_spenn_exchange_error",
        title="Hooke SpENN: exchange symmetry error",
        names=["singlet antisym max", "triplet antisym max"],
        values=[
            _metric_any(singlet, ("comparison/antisymmetry_error_max",)),
            _metric_any(triplet, ("comparison/antisymmetry_error_max",)),
        ],
        y_label="max error",
        log_y=True,
    )
    _shape_comparison_plot(
        summary_dir / "hooke_spenn_wavefunction_shape_comparison",
        runs={
            "singlet": runs["singlet_spenn"],
            "triplet": runs["triplet_spenn"],
        },
    )


def _line_plot(
    stem: Path,
    *,
    title: str,
    x: list[float],
    series: list[tuple[str, list[float]]],
    x_label: str,
    y_label: str,
    y_limits: tuple[float, float] | None = None,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for label, values in series:
        ax.plot(x, values, marker="o", linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, stem)


def _histogram_plot(stem: Path, *, title: str, hist: list[dict[str, float]], x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    centers = [0.5 * (row["bin_left"] + row["bin_right"]) for row in hist]
    widths = [row["bin_right"] - row["bin_left"] for row in hist]
    counts = _column(hist, "count")
    ax.bar(centers, counts, width=widths, align="center", color="#4c78a8", alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, stem)


def _bar_plot(
    stem: Path,
    *,
    title: str,
    names: list[str],
    values: list[float],
    y_label: str,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    plotted = [max(abs(value), 1.0e-16) for value in values] if log_y else values
    ax.bar(names, plotted, color=["#4c78a8", "#f58518"], alpha=0.86)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, stem)


def _symmetry_plot(stem: Path, *, sector: str, metrics: dict[str, float]) -> None:
    if "comparison/antisymmetry_error_max" in metrics or "symmetry/antisym_error_max" in metrics:
        names = ["antisym mean", "antisym max"]
        values = [
            _metric_any(metrics, ("symmetry/antisym_error_mean", "comparison/antisymmetry_error_mean")),
            _metric_any(metrics, ("symmetry/antisym_error_max", "comparison/antisymmetry_error_max")),
        ]
    else:
        names = ["swap mean", "swap max"]
        values = [
            _metric_any(metrics, ("symmetry/swap_logabs_error_mean", "comparison/swap_logabs_error_mean")),
            _metric_any(metrics, ("symmetry/swap_logabs_error_max", "comparison/swap_logabs_error_max")),
        ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.bar(names, values, color="#7a5195", alpha=0.86)
    ax.set_title(f"Hooke {sector}: exchange symmetry error")
    ax.set_ylabel("error")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, stem)


def _shape_comparison_plot(stem: Path, *, runs: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True, sharey=False)
    for ax, sector in zip(axes, ("singlet", "triplet"), strict=True):
        rows = _read_csv(runs[sector] / "plots" / "wavefunction_radial_cut.csv")
        r12 = _column(rows, "r12")
        ax.plot(r12, _column(rows, "exact_logabs_centered"), linewidth=2.2, label="exact")
        ax.plot(r12, _column(rows, "spenn_logabs_centered"), linestyle="--", linewidth=2.2, label="SpENN")
        ax.set_title(f"{sector}: radial shape")
        ax.set_xlabel("r12")
        ax.set_ylabel("centered logabs")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    _save(fig, stem)


def _read_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def _summary_metrics(run_dir: Path) -> dict[str, float]:
    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {key: float(value) for key, value in summary["metrics"].items() if isinstance(value, int | float)}


def _column(rows: list[dict[str, float]], key: str) -> list[float]:
    return [row[key] for row in rows]


def _column_any(rows: list[dict[str, float]], keys: tuple[str, ...]) -> list[float]:
    for key in keys:
        if rows and key in rows[0]:
            return _column(rows, key)
    raise KeyError(f"None of {keys!r} found in CSV columns")


def _metric_any(metrics: dict[str, float], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key in metrics:
            return metrics[key]
    raise KeyError(f"None of {keys!r} found in metrics")


def _radial_series(rows: list[dict[str, float]]) -> list[tuple[str, list[float]]]:
    if rows and "spenn_logabs_centered" in rows[0]:
        return [
            ("SpENN centered logabs", _column(rows, "spenn_logabs_centered")),
            ("exact centered logabs", _column(rows, "exact_logabs_centered")),
        ]
    return [("centered logabs", _column(rows, "logabs_centered"))]


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singlet-run", type=Path, default=None)
    parser.add_argument("--triplet-run", type=Path, default=None)
    parser.add_argument("--singlet-spenn-run", type=Path, default=None)
    parser.add_argument("--triplet-spenn-run", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def _latest_run(run_name: str) -> Path:
    candidates = sorted(
        (ROOT / "outputs").glob(f"*/{run_name}/*/artifacts/summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No Hooke run artifacts found for {run_name!r}; pass the run directory explicitly.")
    return candidates[-1].parents[1]


if __name__ == "__main__":
    main()
