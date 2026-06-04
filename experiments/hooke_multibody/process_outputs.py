"""Process Hooke multibody run artifacts into comparison-ready CSV/JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from spenn.training.artifacts import write_csv, write_json

DATA_EXPORTS = {
    "energy_trace.csv": Path("metrics/energy_trace.csv"),
    "eval_metrics.csv": Path("metrics/eval_metrics.csv"),
    "sampler_metrics.csv": Path("metrics/sampler_metrics.csv"),
    "train_metrics.csv": Path("metrics/train_metrics.csv"),
    "local_energy_samples.csv": Path("plots/local_energy_samples.csv"),
    "pair_distance_samples.csv": Path("plots/pair_distance_samples.csv"),
    "local_energy_histogram.csv": Path("plots/local_energy_histogram.csv"),
    "pair_distance_histogram.csv": Path("plots/pair_distance_histogram.csv"),
    "radial_density.csv": Path("plots/radial_density.csv"),
    "cusp_slope_by_spin.csv": Path("plots/cusp_slope_by_spin.csv"),
    "particle_antisymmetry.csv": Path("plots/particle_antisymmetry.csv"),
    "spin_scan_summary.csv": Path("metrics/spin_scan_summary.csv"),
}
REFERENCE_DATA_EXPORTS = {
    "reference_observables.csv": Path("data/reference_observables.csv"),
    "reference_radial_density.csv": Path("data/reference_radial_density.csv"),
    "reference_pair_distance_density.csv": Path("data/reference_pair_distance_density.csv"),
}
ENERGY_PLAUSIBILITY_COLUMNS = [
    "run_id",
    "run_time",
    "n_electrons",
    "harmonic_omega",
    "spatial_dim",
    "specht_M",
    "n_up",
    "n_down",
    "energy_mean",
    "energy_sem",
    "local_energy_variance",
    "acceptance_rate",
    "reference_available",
    "reference_method",
    "reference_energy",
    "energy_delta",
    "energy_abs_delta",
    "baseline_available",
    "baseline_method",
    "baseline_energy",
    "energy_minus_baseline",
    "energy_abs_minus_baseline",
]


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
    reference_summary = _load_summary(reference_run) if reference_run is not None else None
    if spenn_summary.get("mode") == "spin_scan":
        return _process_spin_scan(
            spenn_run,
            target,
            spenn_summary,
            reference_run=reference_run,
            reference_summary=reference_summary,
        )
    metrics = spenn_summary["metrics"]
    system = _summary_system(spenn_summary)
    baseline = _baseline_comparison(metrics.get("spenn/energy/mean", ""), reference_summary, system=system)
    row = {
        "run_id": spenn_summary["run_id"],
        "run_time": spenn_summary.get("run_time", ""),
        "energy_mean": metrics.get("spenn/energy/mean", ""),
        "local_energy_variance": metrics.get("spenn/local_energy/variance", ""),
        "acceptance_rate": metrics.get("sampler/acceptance_rate", ""),
        "mean_pair_distance": metrics.get("sampler/mean_pair_distance", ""),
        "cusp_only_same_mean_error": metrics.get("cusp/cusp_only_same_mean_error", ""),
        "cusp_only_opposite_mean_error": metrics.get("cusp/cusp_only_opposite_mean_error", ""),
        "full_same_mean_error": metrics.get("cusp/same_mean_error", ""),
        "full_opposite_mean_error": metrics.get("cusp/opposite_mean_error", ""),
        "smooth_residual_same_mean_slope": metrics.get("cusp/smooth_residual_same_mean_slope", ""),
        "smooth_residual_opposite_mean_slope": metrics.get("cusp/smooth_residual_opposite_mean_slope", ""),
        "reference_available": False,
        **baseline,
    }
    processed: dict[str, object] = {
        "spenn_run": str(spenn_run),
        "spenn_observables": row,
        "reference_run": None,
        "reference_available": False,
        "baseline_available": baseline["baseline_available"],
    }
    if reference_run is not None and reference_summary is not None:
        processed["reference_run"] = str(reference_run)
        processed["reference_available"] = bool(reference_summary.get("reference_available", False))
        row["reference_available"] = processed["reference_available"]
        if bool(baseline["baseline_available"]):
            processed["reference_data_files"] = _export_reference_tables(reference_run, target / "data")
        else:
            _clear_reference_tables(target / "data")
    else:
        _clear_reference_tables(target / "data")
    write_csv(target / "data" / "spenn_observables.csv", [row])
    plausibility_rows = [_plausibility_row_from_run_summary(spenn_summary, reference_summary=reference_summary)]
    write_csv(target / "data" / "energy_plausibility.csv", plausibility_rows)
    processed["data_files"] = _export_data_tables(spenn_run, target / "data")
    processed["data_files"]["energy_plausibility.csv"] = str(target / "data" / "energy_plausibility.csv")
    write_json(target / "artifacts" / "processed_summary.json", processed)
    return processed


def _process_spin_scan(
    scan_run: Path,
    target: Path,
    summary: dict[str, object],
    *,
    reference_run: Path | None = None,
    reference_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    best_run = summary.get("best_run", {})
    baseline = _baseline_comparison(
        best_run.get("energy_mean", "") if isinstance(best_run, dict) else "",
        reference_summary,
        system=_scan_run_system(best_run, summary) if isinstance(best_run, dict) else None,
    )
    processed: dict[str, object] = {
        "spenn_run": str(scan_run),
        "mode": "spin_scan",
        "run_id": summary["run_id"],
        "run_time": summary.get("run_time", ""),
        "best_run": best_run,
        "reference_run": None,
        "reference_available": False,
        "baseline_available": baseline["baseline_available"],
    }
    if reference_run is not None and reference_summary is not None:
        processed["reference_run"] = str(reference_run)
        processed["reference_available"] = bool(reference_summary.get("reference_available", False))
        if bool(baseline["baseline_available"]):
            processed["reference_data_files"] = _export_reference_tables(reference_run, target / "data")
        else:
            _clear_reference_tables(target / "data")
    else:
        _clear_reference_tables(target / "data")
    plausibility_rows = [
        _plausibility_row_from_scan_run(run, summary, reference_summary=reference_summary)
        for run in summary.get("runs", [])
    ]
    write_csv(target / "data" / "energy_plausibility.csv", plausibility_rows)
    processed["data_files"] = _export_data_tables(scan_run, target / "data")
    processed["data_files"]["energy_plausibility.csv"] = str(target / "data" / "energy_plausibility.csv")
    write_json(target / "artifacts" / "processed_summary.json", processed)
    return processed


def _plausibility_row_from_run_summary(
    summary: dict[str, object],
    *,
    reference_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg = summary["config"]
    metrics = summary["metrics"]
    system = cfg["system"]
    specht = cfg.get("specht", {})
    reference_available = bool(cfg.get("validation", {}).get("reference_available", False))
    reference_energy = metrics.get("exact/energy", "")
    energy_mean = metrics.get("spenn/energy/mean", "")
    energy_delta = _energy_delta(energy_mean, reference_energy)
    baseline = _baseline_comparison(energy_mean, reference_summary, system=system)
    return _ordered_plausibility_row(
        {
            "run_id": summary["run_id"],
            "run_time": summary.get("run_time", ""),
            "n_electrons": system.get("n_electrons", ""),
            "harmonic_omega": system.get("harmonic_omega", ""),
            "spatial_dim": system.get("spatial_dim", ""),
            "specht_M": specht.get("M", ""),
            "n_up": system.get("n_up", ""),
            "n_down": system.get("n_down", ""),
            "energy_mean": energy_mean,
            "energy_sem": metrics.get("spenn/energy/sem", ""),
            "local_energy_variance": metrics.get("spenn/local_energy/variance", ""),
            "acceptance_rate": metrics.get("sampler/acceptance_rate", ""),
            "reference_available": reference_available,
            "reference_method": "configured_exact" if reference_available else "none",
            "reference_energy": reference_energy,
            "energy_delta": energy_delta,
            "energy_abs_delta": "" if energy_delta == "" else abs(float(energy_delta)),
            **baseline,
        }
    )


def _plausibility_row_from_scan_run(
    run: object,
    summary: dict[str, object],
    *,
    reference_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(run, dict):
        raise TypeError(f"scan run summary must be a mapping, got {type(run).__name__}")
    cfg = summary.get("config", {})
    specht = cfg.get("specht", {}) if isinstance(cfg, dict) else {}
    energy_mean = run.get("energy_mean", "")
    system = _scan_run_system(run, summary)
    baseline = _baseline_comparison(energy_mean, reference_summary, system=system)
    return _ordered_plausibility_row(
        {
            "run_id": run.get("run_id", ""),
            "run_time": run.get("run_time", summary.get("run_time", "")),
            "n_electrons": run.get("n_electrons", ""),
            "harmonic_omega": system.get("harmonic_omega", ""),
            "spatial_dim": system.get("spatial_dim", ""),
            "specht_M": specht.get("M", ""),
            "n_up": run.get("n_up", ""),
            "n_down": run.get("n_down", ""),
            "energy_mean": energy_mean,
            "energy_sem": run.get("energy_sem", ""),
            "local_energy_variance": run.get("local_energy_variance", ""),
            "acceptance_rate": run.get("acceptance_rate", ""),
            "reference_available": False,
            "reference_method": "none",
            "reference_energy": "",
            "energy_delta": "",
            "energy_abs_delta": "",
            **baseline,
        }
    )


def _ordered_plausibility_row(row: dict[str, object]) -> dict[str, object]:
    return {column: row.get(column, "") for column in ENERGY_PLAUSIBILITY_COLUMNS}


def _energy_delta(energy_mean: object, reference_energy: object) -> float | str:
    if energy_mean == "" or reference_energy == "":
        return ""
    return float(energy_mean) - float(reference_energy)


def _baseline_comparison(
    energy_mean: object,
    reference_summary: dict[str, object] | None,
    *,
    system: Mapping[str, object] | None = None,
    n_electrons: object | None = None,
) -> dict[str, object]:
    if reference_summary is None or not bool(reference_summary.get("baseline_available", False)):
        return _blank_baseline_comparison()
    comparison_system = dict(system or {})
    if n_electrons is not None:
        comparison_system.setdefault("n_electrons", n_electrons)
    if comparison_system and not _reference_matches_system(reference_summary, comparison_system):
        return _blank_baseline_comparison()
    baseline_energy = reference_summary.get("baseline_energy", "")
    delta = "" if energy_mean == "" or baseline_energy == "" else float(energy_mean) - float(baseline_energy)
    return {
        "baseline_available": True,
        "baseline_method": reference_summary.get("baseline_method", ""),
        "baseline_energy": baseline_energy,
        "energy_minus_baseline": delta,
        "energy_abs_minus_baseline": "" if delta == "" else abs(float(delta)),
    }


def _blank_baseline_comparison() -> dict[str, object]:
    return {
        "baseline_available": False,
        "baseline_method": "",
        "baseline_energy": "",
        "energy_minus_baseline": "",
        "energy_abs_minus_baseline": "",
    }


def _reference_matches_system(reference_summary: dict[str, object], system: Mapping[str, object]) -> bool:
    reference_system = _summary_system(reference_summary)
    for field in ("n_electrons", "harmonic_omega", "spatial_dim"):
        reference_value = reference_system.get(field, None)
        system_value = system.get(field, None)
        if not _system_value_available(reference_value) or not _system_value_available(system_value):
            continue
        if field == "harmonic_omega":
            if abs(float(reference_value) - float(system_value)) > 1.0e-12:
                return False
        elif int(reference_value) != int(system_value):
            return False
    return True


def _summary_system(summary: Mapping[str, object]) -> dict[str, object]:
    cfg = summary.get("config", {})
    system = cfg.get("system", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(system, Mapping):
        return dict(system)
    return {}


def _scan_run_system(run: Mapping[str, object], summary: Mapping[str, object]) -> dict[str, object]:
    system = _summary_system(summary)
    for key in ("n_electrons", "n_up", "n_down", "spatial_dim", "harmonic_omega"):
        value = run.get(key, None)
        if _system_value_available(value):
            system[key] = value
    omega = run.get("omega", None)
    if _system_value_available(omega):
        system["harmonic_omega"] = omega
    return system


def _system_value_available(value: object) -> bool:
    return value is not None and value != ""


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


def _export_reference_tables(run_dir: Path, data_dir: Path) -> dict[str, str]:
    exported: dict[str, str] = {}
    for name, relative_source in REFERENCE_DATA_EXPORTS.items():
        source = run_dir / relative_source
        if not source.exists():
            continue
        destination = data_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        exported[name] = str(destination)
    return exported


def _clear_reference_tables(data_dir: Path) -> None:
    for name in REFERENCE_DATA_EXPORTS:
        path = data_dir / name
        if path.exists():
            path.unlink()


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
