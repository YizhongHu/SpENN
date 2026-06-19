"""Plot exact Hooke singlet local energy against pair distance."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_PROBE = Path("diagnostics") / "pair_distance_probe" / "probe.csv"
DEFAULT_PLOT_NAME = "model_local_energy_vs_pair_distance.png"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pair-distance plotting CLI."""

    args = _parse_args(argv)
    output = plot_pair_distance(
        run_dir=args.run_dir,
        probe_csv=args.probe_csv,
        output=args.output,
    )
    print(output)
    return 0


def plot_pair_distance(
    *,
    run_dir: str | Path,
    probe_csv: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    """Plot ``model_local_energy`` over ``pair_distance`` from a probe CSV.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        Run directory containing the diagnostics artifacts.
    probe_csv : str or pathlib.Path or None, optional
        Probe CSV path. If relative, it is resolved under ``run_dir``.
    output : str or pathlib.Path or None, optional
        Output image path. If relative, it is resolved under ``run_dir``.

    Returns
    -------
    pathlib.Path
        Path to the written PNG.
    """

    run_path = Path(run_dir)
    probe_path = _resolve_under_run(run_path, probe_csv, DEFAULT_PROBE)
    output_path = _resolve_under_run(
        run_path,
        output,
        DEFAULT_PROBE.parent / DEFAULT_PLOT_NAME,
    )
    rows = _read_csv(probe_path)
    series = _series(rows, x_key="pair_distance", y_key="model_local_energy")
    if not series:
        raise ValueError(f"{probe_path} has no finite pair_distance/model_local_energy rows")

    exact_series = _series(rows, x_key="pair_distance", y_key="exact_local_energy")
    reference_energy = _first_finite(row.get("exact_local_energy") for row in rows)
    if reference_energy is None:
        reference_energy = _first_finite(row.get("model_local_energy") for row in rows)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for path_key, values in _path_series(rows):
        x_values = [x for x, _ in values]
        y_values = [y for _, y in values]
        label = _path_label(path_key)
        ax.plot(
            x_values,
            y_values,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            alpha=0.8,
            label=label,
        )
    if exact_series:
        x_values = [x for x, _ in exact_series]
        y_values = [y for _, y in exact_series]
        ax.plot(
            x_values,
            y_values,
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="exact_local_energy",
        )
    elif reference_energy is not None and math.isfinite(reference_energy):
        ax.axhline(reference_energy, color="black", linewidth=1.2, linestyle="--", label="reference")

    ax.set_xscale("log")
    ax.set_xlabel("pair_distance")
    ax.set_ylabel("model_local_energy")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _path_series(rows: Sequence[Mapping[str, Any]]) -> list[tuple[tuple[Any, ...], list[tuple[float, float]]]]:
    grouped: dict[tuple[Any, ...], list[tuple[float, float]]] = {}
    for row in rows:
        x_value = _as_float(row.get("pair_distance"))
        y_value = _as_float(row.get("model_local_energy"))
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        key = (row.get("center_of_mass_radius"), row.get("direction_id"))
        grouped.setdefault(key, []).append((x_value, y_value))
    return [
        (key, sorted(values, key=lambda item: item[0]))
        for key, values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0]))
    ]


def _path_label(path_key: tuple[Any, ...]) -> str:
    center, direction = path_key
    if center in (None, "") and direction in (None, ""):
        return "model"
    return f"model COM={center}, dir={direction}"


def _series(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> list[tuple[float, float]]:
    values = []
    for row in rows:
        x_value = _as_float(row.get(x_key))
        y_value = _as_float(row.get(y_key))
        if math.isfinite(x_value) and math.isfinite(y_value):
            values.append((x_value, y_value))
    return sorted(values, key=lambda item: item[0])


def _resolve_under_run(run_dir: Path, configured: str | Path | None, default_relative: Path) -> Path:
    path = default_relative if configured is None else Path(configured)
    if path.is_absolute():
        return path
    return run_dir / path


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _parse_scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _first_finite(values: Sequence[Any]) -> float | None:
    for value in values:
        number = _as_float(value)
        if math.isfinite(number):
            return number
    return None


def _parse_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
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


def _as_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


__all__ = ["main", "plot_pair_distance"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
