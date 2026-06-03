"""Create Hooke multibody figures from saved CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

    import matplotlib.pyplot as plt

    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    run_id = str(summary["run_id"])
    output_dir = figure_root / "spenn"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    energy_rows = _read_csv(run_dir / "metrics" / "energy_trace.csv")
    if energy_rows:
        path = output_dir / f"{run_id}_energy_trace.png"
        _plot_line(plt, energy_rows, "step", "spenn/energy/mean", path, "VMC energy")
        written.append(path)

    for table, x_key, y_key, title in [
        ("pair_distance_histogram", "bin_left", "count", "Pair-distance histogram"),
        ("radial_density", "bin_left", "probability_density", "One-body radial density"),
        ("cusp_slope_by_spin", "pair_i", "measured_slope", "Spin-resolved cusp slopes"),
        ("particle_antisymmetry", "pair_i", "antisymmetry_error_max", "Particle antisymmetry error"),
    ]:
        rows = _read_csv(run_dir / "plots" / f"{table}.csv")
        if rows:
            path = output_dir / f"{run_id}_{table}.png"
            _plot_line(plt, rows, x_key, y_key, path, title)
            written.append(path)
    return written


def _plot_line(plt, rows: list[dict[str, str]], x_key: str, y_key: str, path: Path, title: str) -> None:
    x = [float(row[x_key]) for row in rows if row.get(x_key, "") != ""]
    y = [float(row[y_key]) for row in rows if row.get(y_key, "") != ""]
    if not x or not y:
        return
    plt.figure(figsize=(5.0, 3.2))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
