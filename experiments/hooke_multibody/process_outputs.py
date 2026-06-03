"""Process Hooke multibody run artifacts into comparison-ready CSV/JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from spenn.training.artifacts import write_csv, write_json

DATA_EXPORTS = {
    "energy_trace.csv": Path("metrics/energy_trace.csv"),
    "sampler_metrics.csv": Path("metrics/sampler_metrics.csv"),
    "train_metrics.csv": Path("metrics/train_metrics.csv"),
    "local_energy_histogram.csv": Path("plots/local_energy_histogram.csv"),
    "pair_distance_histogram.csv": Path("plots/pair_distance_histogram.csv"),
    "radial_density.csv": Path("plots/radial_density.csv"),
    "cusp_slope_by_spin.csv": Path("plots/cusp_slope_by_spin.csv"),
    "particle_antisymmetry.csv": Path("plots/particle_antisymmetry.csv"),
    "spin_scan_summary.csv": Path("metrics/spin_scan_summary.csv"),
}


def main() -> None:
    """Process saved Hooke multibody artifacts.

    Returns
    -------
    None
        A compact summary is printed to standard output as JSON.
    """

    args = _parse_args()
    summary = process_run(args.spenn_run, reference_run=args.reference_run, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def process_run(
    spenn_run: Path,
    *,
    reference_run: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Write processed CSV/JSON artifacts from saved run summaries.

    Parameters
    ----------
    spenn_run : pathlib.Path
        SpENN run directory containing ``artifacts/summary.json``.
    reference_run : pathlib.Path or None, optional
        Optional reference-placeholder run directory.
    output_dir : pathlib.Path or None, optional
        Output directory. If ``None``, data are written inside `spenn_run`.

    Returns
    -------
    dict
        Processed summary.
    """

    target = output_dir or spenn_run
    spenn_summary = _load_summary(spenn_run)
    metrics = spenn_summary["metrics"]
    row = {
        "run_id": spenn_summary["run_id"],
        "run_time": spenn_summary.get("run_time", ""),
        "energy_mean": metrics.get("spenn/energy/mean", ""),
        "local_energy_variance": metrics.get("spenn/local_energy/variance", ""),
        "acceptance_rate": metrics.get("sampler/acceptance_rate", ""),
        "mean_pair_distance": metrics.get("sampler/mean_pair_distance", ""),
        "reference_available": False,
    }
    processed: dict[str, object] = {
        "spenn_run": str(spenn_run),
        "spenn_observables": row,
        "reference_run": None,
        "reference_available": False,
    }
    if reference_run is not None:
        reference_summary = _load_summary(reference_run)
        processed["reference_run"] = str(reference_run)
        processed["reference_available"] = bool(reference_summary.get("reference_available", False))
        row["reference_available"] = processed["reference_available"]
    write_csv(target / "data" / "spenn_observables.csv", [row])
    processed["data_files"] = _export_data_tables(spenn_run, target / "data")
    write_json(target / "artifacts" / "processed_summary.json", processed)
    return processed


def _export_data_tables(run_dir: Path, data_dir: Path) -> dict[str, str]:
    exported: dict[str, str] = {}
    for name, relative_source in DATA_EXPORTS.items():
        source = run_dir / relative_source
        if not source.exists():
            continue
        destination = data_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        exported[name] = str(destination)
    return exported


def _load_summary(run_dir: Path) -> dict[str, object]:
    with (run_dir / "artifacts" / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spenn-run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
