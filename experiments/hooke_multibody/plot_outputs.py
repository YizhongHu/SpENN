"""Create Hooke multibody figures from saved CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

MPL_CONFIG_DIR = Path("/tmp/spenn_mplconfig")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

FIGURE_ROOT = Path(__file__).resolve().parent / "figures"


def main() -> None:
    """Generate Hooke multibody figures.

    Returns
    -------
    None
        Figure paths are printed to standard output.
    """

    args = _parse_args()
    outputs = plot_run(args.run, figure_root=args.figure_root)
    print("\n".join(str(path) for path in outputs))


def plot_run(run_dir: Path, *, figure_root: Path = FIGURE_ROOT) -> list[Path]:
    """Generate figures for a saved Hooke multibody run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run artifact directory.
    figure_root : pathlib.Path, optional
        Root directory for generated PNG figures.

    Returns
    -------
    list of pathlib.Path
        Written figure paths.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    run_id = str(summary["run_id"])
    output_dir = figure_root / "spenn"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    scan_rows = _read_csv(run_dir / "metrics" / "spin_scan_summary.csv")
    if scan_rows:
        path = output_dir / f"{run_id}_spin_scan_energy.png"
        if _plot_spin_scan(plt, scan_rows, path):
            written.append(path)

    energy_rows = _read_csv(run_dir / "metrics" / "train_metrics.csv") or _read_csv(run_dir / "metrics" / "energy_trace.csv")
    if energy_rows:
        path = output_dir / f"{run_id}_energy_trace.png"
        if _plot_line(plt, energy_rows, "step", "spenn/energy/mean", path, "VMC energy"):
            written.append(path)
        variance_path = output_dir / f"{run_id}_local_energy_variance.png"
        if _plot_line(plt, energy_rows, "step", "spenn/local_energy/variance", variance_path, "Local-energy variance"):
            written.append(variance_path)
        acceptance_path = output_dir / f"{run_id}_acceptance_rate.png"
        if _plot_line(plt, energy_rows, "step", "sampler/acceptance_rate", acceptance_path, "Sampler acceptance"):
            written.append(acceptance_path)

    local_energy_rows = _read_csv(run_dir / "plots" / "local_energy_histogram.csv")
    if local_energy_rows:
        path = output_dir / f"{run_id}_local_energy_histogram.png"
        if _plot_histogram(plt, local_energy_rows, path, "Local-energy histogram", "local energy", "count"):
            written.append(path)

    pair_rows = _read_csv(run_dir / "plots" / "pair_distance_histogram.csv")
    if pair_rows:
        path = output_dir / f"{run_id}_pair_distance_histogram.png"
        if _plot_histogram(plt, pair_rows, path, "Pair-distance histogram", "pair distance", "count"):
            written.append(path)

    radial_rows = _read_csv(run_dir / "plots" / "radial_density.csv")
    if radial_rows:
        path = output_dir / f"{run_id}_radial_density.png"
        if _plot_histogram(plt, radial_rows, path, "One-body radial density", "radius", "probability_density"):
            written.append(path)

    cusp_rows = _read_csv(run_dir / "plots" / "cusp_slope_by_spin.csv")
    if cusp_rows:
        path = output_dir / f"{run_id}_cusp_slope_by_spin.png"
        if _plot_cusp(plt, cusp_rows, path):
            written.append(path)

    antisymmetry_rows = _read_csv(run_dir / "plots" / "particle_antisymmetry.csv")
    if antisymmetry_rows:
        path = output_dir / f"{run_id}_particle_antisymmetry.png"
        if _plot_antisymmetry(plt, antisymmetry_rows, path):
            written.append(path)
    return written


def _plot_line(plt, rows: list[dict[str, str]], x_key: str, y_key: str, path: Path, title: str) -> bool:
    pairs = _numeric_pairs(rows, x_key, y_key)
    if not pairs:
        return False
    x, y = zip(*pairs, strict=True)
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_histogram(
    plt,
    rows: list[dict[str, str]],
    path: Path,
    title: str,
    x_label: str,
    y_key: str,
) -> bool:
    bars = []
    for row in rows:
        left = _to_float(row.get("bin_left", ""))
        right = _to_float(row.get("bin_right", ""))
        value = _to_float(row.get(y_key, ""))
        if left is None or right is None or value is None:
            continue
        bars.append((0.5 * (left + right), right - left, value))
    if not bars:
        return False
    centers, widths, values = zip(*bars, strict=True)
    plt.figure(figsize=(5.0, 3.2))
    plt.bar(centers, values, width=widths, align="center", alpha=0.82)
    plt.xlabel(x_label)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_cusp(plt, rows: list[dict[str, str]], path: Path) -> bool:
    labels: list[str] = []
    measured: list[float] = []
    target: list[float] = []
    colors: list[str] = []
    for row in rows:
        slope = _to_float(row.get("measured_slope", ""))
        target_slope = _to_float(row.get("target_slope", ""))
        if slope is None or target_slope is None:
            continue
        labels.append(f"{row.get('pair_i', '?')}-{row.get('pair_j', '?')}\n{row.get('spin_relation', '')}")
        measured.append(slope)
        target.append(target_slope)
        colors.append("#4c78a8" if row.get("spin_relation") == "same" else "#f58518")
    if not measured:
        return False
    x = list(range(len(measured)))
    plt.figure(figsize=(max(5.0, 0.6 * len(measured)), 3.4))
    plt.bar(x, measured, color=colors, alpha=0.82, label="measured")
    plt.scatter(x, target, color="black", marker="x", label="target", zorder=3)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("slope")
    plt.title("Spin-resolved cusp slopes")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_antisymmetry(plt, rows: list[dict[str, str]], path: Path) -> bool:
    labels: list[str] = []
    errors: list[float] = []
    sign_flip: list[float] = []
    for row in rows:
        error = _to_float(row.get("antisymmetry_error_max", ""))
        flip = _to_float(row.get("sign_flip_accuracy", ""))
        if error is None or flip is None:
            continue
        labels.append(f"{row.get('pair_i', '?')}-{row.get('pair_j', '?')}")
        errors.append(error)
        sign_flip.append(flip)
    if not errors:
        return False
    x = list(range(len(errors)))
    plt.figure(figsize=(max(5.0, 0.55 * len(errors)), 3.4))
    plt.bar(x, errors, color="#7a5195", alpha=0.82, label="max antisymmetry error")
    plt.plot(x, sign_flip, color="black", marker="o", linewidth=1.2, label="sign-flip accuracy")
    plt.xticks(x, labels)
    plt.ylabel("value")
    plt.title("Particle-token antisymmetry")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def _plot_spin_scan(plt, rows: list[dict[str, str]], path: Path) -> bool:
    records = []
    for row in rows:
        energy = _to_float(row.get("energy_mean", ""))
        variance = _to_float(row.get("local_energy_variance", ""))
        acceptance = _to_float(row.get("acceptance_rate", ""))
        if energy is None:
            continue
        label = f"{row.get('n_up', '?')}/{row.get('n_down', '?')}"
        records.append((label, energy, variance, acceptance))
    if not records:
        return False
    best_index = min(range(len(records)), key=lambda index: records[index][1])
    labels = [record[0] for record in records]
    energies = [record[1] for record in records]
    variances = [record[2] for record in records]
    acceptance_rates = [record[3] for record in records]
    colors = ["#4c78a8"] * len(records)
    colors[best_index] = "#f58518"
    x = list(range(len(records)))
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 6.8), sharex=True)
    axes[0].bar(x, energies, color=colors, alpha=0.86)
    axes[0].set_ylabel("energy")
    axes[0].set_title("Fixed spin-sector scan")
    _plot_optional_bar(axes[1], x, variances, colors, "variance")
    _plot_optional_bar(axes[2], x, acceptance_rates, colors, "acceptance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("n_up/n_down")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_optional_bar(axis, x: list[int], values: list[float | None], colors: list[str], ylabel: str) -> None:
    plotted = [float("nan") if value is None else value for value in values]
    axis.bar(x, plotted, color=colors, alpha=0.86)
    axis.set_ylabel(ylabel)


def _numeric_pairs(rows: list[dict[str, str]], x_key: str, y_key: str) -> list[tuple[float, float]]:
    pairs = []
    for row in rows:
        x = _to_float(row.get(x_key, ""))
        y = _to_float(row.get(y_key, ""))
        if x is not None and y is not None:
            pairs.append((x, y))
    return pairs


def _to_float(value: str | object) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
