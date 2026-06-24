"""Render final reports from compact ``08_final_collect`` tables.

This stage is intentionally report-oriented and fast. It consumes only compact
tables written by ``final_collect.py``; it does not read raw final-eval task
records, final-train metrics, or checkpoints.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from artifacts import read_csv as _read_csv, write_csv as _write_csv_columns
import plot
from run_utils import (
    STAGE_FINAL_COLLECT,
    STAGE_FINAL_REPORT,
    latest_attempt_id,
    log_prefix,
    new_attempt_id,
    stage_dir,
    study_name_from_manifest,
    write_json,
    write_latest,
)
from stats import (
    as_float as _as_float,
    crop_bar_series_to_weighted_quantiles as _crop_bar_series_to_weighted_quantiles,
    finite_values as _finite_values,
    format_number as _format_number,
    mean as _mean,
    median as _median,
    quantile as _quantile,
    variance as _variance,
)

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
EXACT_HOOKE_ENERGY = 2.0
WINNER_KINDS = ("energy", "stability")
NARROW_WINNER_HEATMAP_WIDTH_SCALE = 0.75
SYMMETRY_METRICS = (
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
    "finite_fraction",
)
FEATURE_TRACE_METRICS = (
    "rms_q95",
    "max_abs",
    "nonfinite_count",
)
FEATURE_TRACE_EXCLUDED_LAYERS = {
    "feature_normalization.norm",
    "feature_normalize.norm",
    "layers.0.update_norm",
    "update_norm",
}
ENERGY_COMPONENT_QUANTITIES = (
    ("kinetic", "kinetic_mean"),
    ("harmonic_trap", "harmonic_trap_mean"),
    ("electron_electron", "electron_electron_mean"),
    ("total_energy", "energy_mean"),
    ("virial_residual", "virial_residual"),
    ("virial_relative_residual", "virial_relative_residual"),
)
ENERGY_COMPONENT_COLUMNS = (
    "winner_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "quantity",
    "n",
    "mean",
    "median",
    "min",
    "max",
)
VIRIAL_RESIDUAL_STATS = ("mean", "median", "min", "max")

COMPACT_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
)


def _derive_virial_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    kinetic = _as_float(row.get("kinetic_mean"))
    harmonic = _as_float(row.get("harmonic_trap_mean"))
    electron_electron = _as_float(row.get("electron_electron_mean"))
    if kinetic is None or harmonic is None or electron_electron is None:
        return {"virial_residual": None, "virial_relative_residual": None}
    residual = 2.0 * kinetic - 2.0 * harmonic + electron_electron
    denominator = abs(2.0 * kinetic) + abs(2.0 * harmonic) + abs(electron_electron)
    relative = abs(residual) / denominator if denominator else 0.0
    return {"virial_residual": residual, "virial_relative_residual": relative}


def _energy_component_value(row: dict[str, Any], key: str) -> float | None:
    value = _as_float(row.get(key))
    if value is not None:
        return value
    if key in {"virial_residual", "virial_relative_residual"}:
        return _derive_virial_metrics(row)[key]
    return None


def _winner_id(basis_class: str, normalization: str, winner_kind: str) -> str:
    return _safe_label(f"{basis_class}_{normalization}_{winner_kind}")


def _aggregate_component_row(
    *,
    basis_class: str,
    normalization: str,
    winner_kind: str,
    quantity: str,
    values: Sequence[float],
) -> dict[str, Any]:
    return {
        "winner_id": _winner_id(basis_class, normalization, winner_kind),
        "basis_class": basis_class,
        "normalization": normalization,
        "winner_kind": winner_kind,
        "quantity": quantity,
        "n": len(values),
        "mean": _format_number(_mean(values)),
        "median": _format_number(_median(values)),
        "min": _format_number(min(values) if values else None),
        "max": _format_number(max(values) if values else None),
    }


def _energy_component_rows_for_group(
    rows: Sequence[dict[str, Any]],
    *,
    basis_class: str,
    normalization: str,
    winner_kind: str,
) -> list[dict[str, Any]]:
    output = []
    for quantity, key in ENERGY_COMPONENT_QUANTITIES:
        values = _finite_values(_energy_component_value(row, key) for row in rows)
        output.append(
            _aggregate_component_row(
                basis_class=basis_class,
                normalization=normalization,
                winner_kind=winner_kind,
                quantity=quantity,
                values=values,
            )
        )
    return output


def _energy_component_tables_by_winner(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return validation-style energy-component tables for each winner family."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        basis_class = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        winner_kind = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        groups[(basis_class, normalization, winner_kind)].append(row)

    tables = {}
    for (basis_class, normalization, winner_kind), group_rows in sorted(groups.items()):
        tables[_winner_id(basis_class, normalization, winner_kind)] = _energy_component_rows_for_group(
            group_rows,
            basis_class=basis_class,
            normalization=normalization,
            winner_kind=winner_kind,
        )
    return tables


def _combined_energy_component_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return combined energy-component rows in stable winner-id order."""

    by_winner = _energy_component_tables_by_winner(rows)
    return [row for winner_id in sorted(by_winner) for row in by_winner[winner_id]]


def _virial_residual_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return virial-residual summary rows for heatmaps."""

    return [row for row in rows if row.get("quantity") == "virial_residual"]


def _resolve_collect_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    collect_stage = stage_dir(results_root, STAGE_FINAL_COLLECT)
    attempt_id = latest_attempt_id(collect_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no final-collect attempts under {stage_dir(results_root, STAGE_FINAL_COLLECT)}")
    return attempt_id


def _load_collect_manifest(collect_dir: Path) -> dict[str, Any]:
    """Read top-level scalar fields from final_collect's manifest.yaml."""

    manifest_path = collect_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    manifest: dict[str, Any] = {}
    for line in manifest_path.read_text().splitlines():
        if line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value:
            manifest[key.strip()] = value
    return manifest


def _load_collect_tables(
    results_root: Path,
    collect_attempt_id: str,
) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"missing final-collect attempt: {collect_dir}")
    return collect_dir, _load_collect_manifest(collect_dir), {name: _read_csv(collect_dir / name) for name in COMPACT_TABLES}


def _architecture_label(row: dict[str, Any]) -> str:
    return str(row.get("basis_class", row.get("architecture", row.get("basis", "")))) or "all"


def _winner_rows(rows: Sequence[dict[str, Any]], winner_kind: str) -> list[dict[str, Any]]:
    if winner_kind == "energy":
        return [row for row in rows if str(row.get("winner_kind", "")).strip() == "energy"]
    return [row for row in rows if str(row.get("winner_kind", "")).strip() not in {"", "energy"}]


def _winner_title(winner_kind: str) -> str:
    return f"{winner_kind} winners"


def _winner_filename(prefix: str, winner_kind: str, suffix: str) -> str:
    return f"{prefix}_{winner_kind}_winner_{suffix}"


def _save_winner_pair_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
    width_scale: float = 1.0,
) -> None:
    """Save energy/stability winner heatmaps side by side with one scale."""

    plot.save_winner_pair_heatmap(
        path,
        {winner: _winner_rows(rows, winner) for winner in WINNER_KINDS},
        row_key=row_key,
        col_key=col_key,
        value_key=value_key,
        title=title,
        panel_titles={winner: _winner_title(winner) for winner in WINNER_KINDS},
        transform=transform,
        width_scale=width_scale,
    )


def _figure_label(section: str, index: int) -> str:
    return f"{section}{chr(ord('A') + index)}"


def _safe_label(value: Any) -> str:
    label = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label) or "all"


def _unique_in_order(values: Sequence[Any]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        label = str(value)
        if label == "" or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


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
        points.append(
            {
                "abs_energy_error": abs_error,
                "local_energy_var": variance,
                "architecture": str(row.get("basis_class", row.get("architecture", ""))),
                "normalization": str(row.get("normalization", "")),
                "winner_kind": "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability",
            }
        )
    return points


def _save_energy_variance_scatter(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    points = _energy_variance_points(rows)
    if not points:
        plot.save_no_data(path, title)
        return
    plot.save_loglog_scatter_grid(
        path,
        points,
        panel_key="winner_kind",
        panel_keys=WINNER_KINDS,
        panel_titles={winner: _winner_title(winner) for winner in WINNER_KINDS},
        x_key="abs_energy_error",
        y_key="local_energy_var",
        color_key="architecture",
        marker_key="normalization",
        x_label="abs energy error |E - 2|",
        y_label="local-energy variance",
        title=f"{title}\nWinner type is separated by panel; color is architecture and marker shape is normalization.",
        color_title="Architecture",
        marker_title="Normalization",
    )


def _local_energy_distribution_groups(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Group compact local-energy histograms by normalization and architecture."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalization = str(row.get("normalization", ""))
        architecture = _architecture_label(row)
        groups[(normalization, architecture)].append(row)
    normalizations = sorted({key[0] for key in groups})
    architectures = sorted({key[1] for key in groups})
    return normalizations, architectures, groups


def _local_energy_bar_series(rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    """Aggregate run-level histogram bins for one displayed distribution."""

    counts_by_center: dict[float, float] = defaultdict(float)
    widths_by_center: dict[float, float] = {}
    for row in rows:
        center = _as_float(row.get("bin_center"))
        count = _as_float(row.get("count")) or 0.0
        if center is None or count <= 0.0:
            continue
        left = _as_float(row.get("bin_left"))
        right = _as_float(row.get("bin_right"))
        counts_by_center[center] += count
        widths_by_center[center] = (right - left) if left is not None and right is not None else 1.0
    centers = sorted(counts_by_center)
    counts = [counts_by_center[center] for center in centers]
    widths = [widths_by_center[center] for center in centers]
    return _crop_bar_series_to_weighted_quantiles(centers, counts, widths)


def _save_local_energy_distribution_grid(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    normalizations, architectures, groups = _local_energy_distribution_groups(rows)
    if not groups:
        plot.save_no_data(path, title)
        return

    bars = []
    for (normalization, architecture), values in groups.items():
        centers, counts, widths = _local_energy_bar_series(values)
        for center, count, width in zip(centers, counts, widths, strict=True):
            bars.append(
                {
                    "panel_key": (normalization, architecture),
                    "x": center,
                    "height": count,
                    "width": width,
                    "color": "#4C78A8",
                }
            )
    if not bars:
        plot.save_no_data(path, title)
        return
    plot.save_grouped_bar_grid(
        path,
        bars,
        row_keys=normalizations,
        col_keys=architectures,
        x_label="local_energy",
        y_label="count",
        title=title,
        figsize=(max(4.0, 3.2 * len(architectures)), max(3.0, 2.4 * len(normalizations))),
        rect=(0.0, 0.0, 1.0, 0.94),
        suptitle_y=0.98,
        bbox_inches=None,
    )


def _com_label(raw: Any) -> str:
    value = str(raw).strip()
    return f"CoM {value}" if value else "CoM all"


def _cusp_profile_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return cusp profile points with one line per center of mass.

    Cusp diagnostics evaluate multiple directions for each center of mass. The
    local energy and log amplitude should agree across directions at fixed CoM,
    so Figure 2 collapses directions and seeds into one CoM line. Values are
    means over all compact direction/seed records; error bars show their
    variance.
    """

    cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        r12 = _as_float(row.get("r12"))
        value = _as_float(row.get(value_key))
        if r12 is None or value is None:
            continue
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        cells[(basis, normalization, com, r12)].append(value)

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, r12), values in sorted(cells.items()):
        mean = _mean(values)
        variance = _variance(values)
        if mean is None or variance is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "r12": r12,
                "mean": mean,
                "variance": variance,
                "n_records": len(values),
            }
        )
    return out


def _save_cusp_winner_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
    y_label: str,
    title: str,
) -> None:
    """Save one cusp profile metric as a winner-specific architecture grid."""

    profiles = _cusp_profile_points(rows, winner_kind=winner_kind, value_key=value_key)
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    line_labels = sorted({cell[2] for cell in profiles})
    series = [
        {
            "panel_key": (normalization, architecture),
            "line_key": com,
            "x": point["r12"],
            "y": point["mean"],
            "yerr": point["variance"],
            "linewidth": 1.0,
            "marker": "o",
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=line_labels,
        x_label="r12",
        y_label=y_label,
        title=f"{title}\nLines are CoM groups; means and variances pool all direction/seed records.",
        legend_title="CoM",
        rect=(0.0, 0.0, 0.86, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.1 * len(normalizations))),
    )


def _cusp_derivative_profiles(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, float | int]]],
    dict[tuple[str, str, str], list[dict[str, float | int]]],
]:
    """Return seed-aggregated model and target cusp derivative profiles."""

    model_cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    target_cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        row_winner = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        if row_winner != winner_kind:
            continue
        r12 = _as_float(row.get("r12"))
        value = _as_float(row.get("d_logabs_dr_median"))
        if r12 is None or value is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        model_cells[(architecture, normalization, com, r12)].append(value)
        target = _as_float(row.get("target_d_logabs_dr"))
        if target is not None:
            target_cells[(architecture, normalization, com, r12)].append(target)

    def profiles(cells: dict[tuple[str, str, str, float], list[float]]) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
        out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
        for (architecture, normalization, com, r12), values in sorted(cells.items()):
            median = _median(values)
            if median is None:
                continue
            out[(architecture, normalization, com)].append({"r12": r12, "median": median, "n_records": len(values)})
        return out

    return profiles(model_cells), profiles(target_cells)


def _save_cusp_derivative_winner_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
) -> None:
    """Save Figure 2D for one winner type."""

    path.parent.mkdir(parents=True, exist_ok=True)
    model_profiles, target_profiles = _cusp_derivative_profiles(rows, winner_kind=winner_kind)
    if not model_profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in model_profiles})
    normalizations = sorted({key[1] for key in model_profiles})
    com_labels = sorted({key[2] for key in model_profiles})
    plt = plot.pyplot()
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(com_labels)}
    fig, axes = plt.subplots(
        len(normalizations),
        len(architectures),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    target_handle = None
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            plotted = False
            for com in com_labels:
                points = sorted(model_profiles.get((architecture, normalization, com), []), key=lambda item: float(item["r12"]))
                if not points:
                    continue
                (line,) = ax.plot(
                    [float(point["r12"]) for point in points],
                    [float(point["median"]) for point in points],
                    linewidth=1.1,
                    marker="o",
                    markersize=2.8,
                    color=colors[com],
                    label=com,
                )
                legend_handles.setdefault(com, line)
                target_points = sorted(target_profiles.get((architecture, normalization, com), []), key=lambda item: float(item["r12"]))
                if target_points:
                    (target_line,) = ax.plot(
                        [float(point["r12"]) for point in target_points],
                        [float(point["median"]) for point in target_points],
                        linewidth=1.0,
                        linestyle="--",
                        color=colors[com],
                        alpha=0.7,
                    )
                    target_handle = target_handle or target_line
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\nd logabs / dr")
            if row_index == len(normalizations) - 1:
                ax.set_xlabel("r12")
            ax.grid(True, linewidth=0.35, alpha=0.35)
    if legend_handles:
        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        if target_handle is not None:
            handles.append(target_handle)
            labels.append("target")
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=7,
            title="CoM",
            title_fontsize=8,
            borderaxespad=0.0,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86 if legend_handles else 1.0, 0.92))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _tail_seed_profile_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return tail profile points with seed variance for each CoM line.

    Tail records may contain multiple radial paths for the same CoM, radius, and
    seed. Those paths are averaged inside a seed first; the plotted uncertainty
    is then the variance of those seed-level values.
    """

    per_seed: dict[tuple[str, str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        radius = _as_float(row.get("radius"))
        value = _as_float(row.get(value_key))
        if radius is None or value is None:
            continue
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        seed = str(row.get("seed_index", row.get("final_run_id", "")))
        per_seed[(basis, normalization, com, radius, seed)].append(value)

    by_radius: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for (basis, normalization, com, radius, _seed), values in per_seed.items():
        mean = _mean(values)
        if mean is not None:
            by_radius[(basis, normalization, com, radius)].append(mean)

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, radius), seed_values in sorted(by_radius.items()):
        mean = _mean(seed_values)
        variance = _variance(seed_values)
        if mean is None or variance is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "radius": radius,
                "mean": mean,
                "variance": variance,
                "n_seeds": len(seed_values),
            }
        )
    return out


def _tail_local_energy_bar_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return tail local-energy bar points with q5-q85 ranges."""

    cells: dict[tuple[str, str, str, float], list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        radius = _as_float(row.get("radius"))
        median = _as_float(row.get("local_energy_median"))
        if radius is None or median is None:
            continue
        low = _as_float(row.get("local_energy_q05"))
        if low is None:
            low = _as_float(row.get("local_energy_q25"))
        high = _as_float(row.get("local_energy_q85"))
        if high is None:
            high = _as_float(row.get("local_energy_q75"))
        low = median if low is None else low
        high = median if high is None else high
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        cells[(basis, normalization, com, radius)].append((median, low, high))

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, radius), values in sorted(cells.items()):
        medians = [value[0] for value in values]
        lows = [value[1] for value in values]
        highs = [value[2] for value in values]
        median = _median(medians)
        low = _median(lows)
        high = _median(highs)
        if median is None or low is None or high is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "radius": radius,
                "median": median,
                "low": min(low, median),
                "high": max(high, median),
                "n_records": len(values),
            }
        )
    return out


def _tail_bar_width(radii: Sequence[float], n_groups: int) -> float:
    unique = sorted({radius for radius in radii if math.isfinite(radius)})
    gaps = [right - left for left, right in zip(unique, unique[1:], strict=False) if right > left]
    base = min(gaps) * 0.75 if gaps else max(0.08, abs(unique[0]) * 0.08 if unique else 0.08)
    return base / max(1, n_groups)


def _save_tail_local_energy_bar_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
) -> None:
    """Save tail local-energy medians as bars with q5-q85 ranges."""

    profiles = _tail_local_energy_bar_points(rows, winner_kind=winner_kind)
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    com_labels = sorted({cell[2] for cell in profiles})

    all_radii = [float(point["radius"]) for points in profiles.values() for point in points]
    width = _tail_bar_width(all_radii, len(com_labels))
    bars = [
        {
            "panel_key": (normalization, architecture),
            "bar_key": com,
            "x": point["radius"],
            "height": point["median"],
            "yerr_low": max(0.0, float(point["median"]) - float(point["low"])),
            "yerr_high": max(0.0, float(point["high"]) - float(point["median"])),
            "width": width,
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_bar_grid(
        path,
        bars,
        row_keys=normalizations,
        col_keys=architectures,
        bar_keys=com_labels,
        x_label="radius",
        y_label="local energy",
        title=f"{title}\nBars show median local energy; error bars show q5-q85.",
        legend_title="CoM",
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        rect=(0.0, 0.0, 0.88, 0.94),
    )


def _save_tail_logabs_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
) -> None:
    """Save tail logabs profiles as lines."""

    profiles = _tail_seed_profile_points(rows, winner_kind=winner_kind, value_key="logabs_median")
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    com_labels = sorted({cell[2] for cell in profiles})
    series = [
        {
            "panel_key": (normalization, architecture),
            "line_key": com,
            "x": point["radius"],
            "y": point["mean"],
            "yerr": point["variance"],
            "marker": "o",
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=com_labels,
        x_label="radius",
        y_label="logabs",
        title=f"{title}\nLines are CoM groups; error bars are seed variance over final replicates.",
        legend_title="CoM",
        rect=(0.0, 0.0, 0.88, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
    )


def _group_label(row: dict[str, Any], group_keys: Sequence[str]) -> str:
    parts = [str(row.get(key, "")) for key in group_keys if row.get(key, "") != ""]
    return "/".join(parts) if parts else "all"


def _save_line_plot(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend: str = "auto",
    legend_title: str | None = None,
) -> None:
    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        series.append({"line_key": _group_label(row, group_keys), "x": x, "y": y})
    if not series:
        plot.save_no_data(path, title)
        return
    plot.save_grouped_line_plot(
        path,
        series,
        x_label=x_key,
        y_label=y_key,
        title=title,
        legend=legend,
        legend_title=legend_title,
    )


def _save_architecture_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend_title: str,
) -> None:
    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        series.append(
            {
                "panel_key": architecture,
                "line_key": _group_label(row, group_keys),
                "x": x,
                "y": y,
            }
        )
    if not series:
        plot.save_no_data(path, title)
        return

    architectures = sorted({str(row["panel_key"]) for row in series})
    labels = sorted({str(row["line_key"]) for row in series})
    plot.save_grouped_line_grid(
        path,
        series,
        panel_keys=architectures,
        panel_title=lambda key: str(key),
        line_keys=labels,
        x_label=x_key,
        y_label=y_key,
        title=title,
        legend_title=legend_title,
        rect=(0.0, 0.0, 0.83, 0.94),
    )


def _save_architecture_normalization_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend_title: str,
) -> None:
    """Save a line grid with normalization rows and architecture columns."""

    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        series.append(
            {
                "panel_key": (normalization, architecture),
                "line_key": _group_label(row, group_keys),
                "x": x,
                "y": y,
            }
        )
    if not series:
        plot.save_no_data(path, title)
        return

    x_label = x_key.replace("_", " ")
    y_label = y_key.replace("_", " ")
    architectures = sorted({str(row["panel_key"][1]) for row in series})
    normalizations = sorted({str(row["panel_key"][0]) for row in series})
    labels = sorted({str(row["line_key"]) for row in series})
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=labels,
        x_label=x_label,
        y_label=y_label,
        title=title,
        legend_title=legend_title,
        rect=(0.0, 0.0, 0.84, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
    )


def _training_curve_value(row: dict[str, Any], value_mode: str) -> float | None:
    energy = _as_float(row.get("energy_mean"))
    if energy is None:
        return None
    if value_mode == "abs_energy_error":
        return abs(energy - EXACT_HOOKE_ENERGY)
    if value_mode == "energy_mean":
        return energy
    raise ValueError(f"unknown training curve value mode: {value_mode}")


def _smooth_training_points(points: Sequence[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
    """Return a centered rolling mean for one training run."""

    sorted_points = sorted(points, key=lambda point: float(point["step"]))
    if window <= 1 or len(sorted_points) <= 1:
        return [dict(point) for point in sorted_points]
    radius = max(1, window // 2)
    smoothed = []
    for index, point in enumerate(sorted_points):
        low = max(0, index - radius)
        high = min(len(sorted_points), index + radius + 1)
        value = _mean([float(item["value"]) for item in sorted_points[low:high]])
        if value is not None:
            smoothed.append({**point, "value": value})
    return smoothed


def _training_run_curves(
    rows: Sequence[dict[str, Any]],
    *,
    value_mode: str = "energy_mean",
    smooth_window: int = 5,
) -> dict[tuple[str, str, str, str], list[dict[str, float | str]]]:
    """Return one smoothed curve per final-training run."""

    per_run_step: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        step = _as_float(row.get("step"))
        value = _training_curve_value(row, value_mode)
        if step is None or value is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        winner = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        run_id = str(row.get("final_run_id", "")) or f"seed-{row.get('seed_index', '')}"
        per_run_step[(architecture, normalization, winner, run_id, step)].append(value)

    raw_curves: dict[tuple[str, str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for (architecture, normalization, winner, run_id, step), values in sorted(per_run_step.items()):
        value = _mean(values)
        if value is not None:
            raw_curves[(architecture, normalization, winner, run_id)].append({"step": step, "value": value, "run_id": run_id})
    return {
        key: _smooth_training_points(points, window=smooth_window)
        for key, points in raw_curves.items()
    }


def _save_training_curve_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_mode: str,
    y_label: str,
    title: str,
    semilogy: bool = False,
    smooth_window: int = 5,
) -> None:
    """Save one winner family's final-training curves."""

    curves = _training_run_curves(_winner_rows(rows, winner_kind), value_mode=value_mode, smooth_window=smooth_window)
    curves = {key: points for key, points in curves.items() if key[2] == winner_kind}
    if not curves:
        plot.save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in curves})
    normalizations = sorted({key[1] for key in curves})
    runs_by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    series = []
    for (architecture, normalization, _winner, run_id), points in curves.items():
        runs_by_cell[(normalization, architecture)].append(run_id)
        for point in points:
            series.append(
                {
                    "panel_key": (normalization, architecture),
                    "line_key": run_id,
                    "x": point["step"],
                    "y": point["value"],
                    "linewidth": 0.9,
                    "alpha": 0.45,
                    "marker": "",
                }
            )
    panel_notes = {
        key: f"n={len(run_ids)}"
        for key, run_ids in runs_by_cell.items()
    }
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        x_label="step",
        y_label=y_label,
        title=f"{title}\nEach line is one final-training run; curves use a {smooth_window}-point centered rolling mean.",
        legend_title=None,
        show_legend=False,
        sharey=semilogy,
        yscale="log" if semilogy else None,
        panel_notes=panel_notes,
        rect=(0.0, 0.0, 1.0, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
    )


def _save_symmetry_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
) -> None:
    """Save one symmetry metric as architecture-by-normalization heatmaps."""

    symmetries = _unique_in_order(row.get("symmetry_task", "") for row in rows)
    if not symmetries:
        plot.save_no_data(path, title)
        return
    panel_rows = {}
    for symmetry in symmetries:
        symmetry_rows = [row for row in rows if str(row.get("symmetry_task", "")) == symmetry]
        for winner in WINNER_KINDS:
            panel_rows[(symmetry, winner)] = _winner_rows(symmetry_rows, winner)
    plot.save_row_scoped_heatmap_grid(
        path,
        panel_rows,
        row_labels=symmetries,
        col_labels=WINNER_KINDS,
        row_key="basis_class",
        col_key="normalization",
        value_key=metric_key,
        title=title,
        panel_title=lambda symmetry, winner: f"{symmetry}\n{_winner_title(winner)}",
        colorbar_ticks="none",
        figsize=(max(14.0, 7.0 * len(WINNER_KINDS)), max(3.2, 3.0 * len(symmetries))),
        subplot_adjust={"left": 0.07, "right": 0.89, "bottom": 0.08, "top": 0.90, "wspace": 0.55, "hspace": 0.65},
    )


def _save_feature_trace_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
) -> None:
    """Save one feature-trace metric as layer-by-winner heatmaps."""

    trace_rows = [
        row
        for row in rows
        if str(row.get("trace_kind", "")) == "feature_trace_stability"
        and str(row.get("layer", "")) not in FEATURE_TRACE_EXCLUDED_LAYERS
    ]
    layers = _unique_in_order(row.get("layer", "") for row in trace_rows)
    if not layers:
        plot.save_no_data(path, title)
        return

    panel_rows = {}
    for layer in layers:
        layer_rows = [row for row in trace_rows if str(row.get("layer", "")) == layer]
        for winner in WINNER_KINDS:
            panel_rows[(layer, winner)] = _winner_rows(layer_rows, winner)
    plot.save_row_scoped_heatmap_grid(
        path,
        panel_rows,
        row_labels=layers,
        col_labels=WINNER_KINDS,
        row_key="basis_class",
        col_key="normalization",
        value_key=metric_key,
        title=title,
        panel_title=lambda layer, winner: f"{layer}\n{_winner_title(winner)}",
        figsize=(max(11.0, 5.5 * len(WINNER_KINDS)), max(3.2, 2.55 * len(layers))),
        subplot_adjust={"left": 0.08, "right": 0.89, "bottom": 0.04, "top": 0.94, "wspace": 0.65, "hspace": 0.75},
    )


def _save_virial_residual_heatmap(path: Path, rows: Sequence[dict[str, Any]], *, stat: str) -> None:
    """Save one signed-log virial-residual winner-pair heatmap."""

    _save_winner_pair_heatmap(
        path,
        rows,
        row_key="basis_class",
        col_key="normalization",
        value_key=stat,
        title=f"Virial residual {stat}",
        transform="signed_log",
        width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE,
    )


def _write_figures(figures_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    architecture_rows = tables["architecture_summary.csv"]
    energy = tables["energy_by_run.csv"]
    energy_components = _combined_energy_component_rows(energy)
    virial_residual = _virial_residual_rows(energy_components)
    histograms = tables["local_energy_histograms.csv"]
    stratified = tables["stratified_summary.csv"]

    def add(filename: str, writer: Any) -> None:
        writer(figures_dir / filename)
        written.append(filename)

    add(
        "1A_real_scale_energy_error_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key="basis_class", col_key="normalization", value_key="energy_error_median", title="Median signed final energy error", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    add(
        "1A_log_scale_energy_error_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key="basis_class", col_key="normalization", value_key="energy_error_median", title="Median signed final energy error", transform="signed_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    for winner in WINNER_KINDS:
        add(
            _winner_filename("1C", winner, "local_energy_distribution_grid.png"),
            lambda path, winner=winner: _save_local_energy_distribution_grid(path, _winner_rows(histograms, winner), title=f"MCMC local-energy histograms: {_winner_title(winner)}"),
        )

    add("1B_energy_error_vs_local_energy_variance.png", lambda path: _save_energy_variance_scatter(path, energy, title="Absolute energy error vs local-energy variance"))
    cusp_rows = tables["cusp_profile_summary.csv"]
    for winner in WINNER_KINDS:
        add(
            _winner_filename("2A", winner, "cusp_local_energy_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="local_energy_median", y_label="local energy", title=f"Cusp local energy profiles: {_winner_title(winner)}"),
        )
        add(
            _winner_filename("2B", winner, "cusp_logabs_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="logabs_median", y_label="logabs", title=f"Cusp logabs profiles: {_winner_title(winner)}"),
        )
        add(
            _winner_filename("2C", winner, "cusp_finite_fraction_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="finite_fraction", y_label="finite fraction", title=f"Cusp finite fraction profiles: {_winner_title(winner)}"),
        )
    for winner in WINNER_KINDS:
        add(
            _winner_filename("2D", winner, "cusp_dlogabs_dr_grid.png"),
            lambda path, winner=winner: _save_cusp_derivative_winner_grid(path, cusp_rows, winner_kind=winner, title=f"Cusp derivative profiles: {_winner_title(winner)}"),
        )
    add("3A_tail_energy_winner_local_energy_bars.png", lambda path: _save_tail_local_energy_bar_grid(path, tables["tail_profile_summary.csv"], winner_kind="energy", title="Tail local energy: energy winners"))
    add("3B_tail_stability_winner_local_energy_bars.png", lambda path: _save_tail_local_energy_bar_grid(path, tables["tail_profile_summary.csv"], winner_kind="stability", title="Tail local energy: stability winners"))
    add("3C_tail_energy_winner_logabs_grid.png", lambda path: _save_tail_logabs_line_grid(path, tables["tail_profile_summary.csv"], winner_kind="energy", title="Tail logabs: energy winners"))
    add("3D_tail_stability_winner_logabs_grid.png", lambda path: _save_tail_logabs_line_grid(path, tables["tail_profile_summary.csv"], winner_kind="stability", title="Tail logabs: stability winners"))

    add(
        "3E_tail_outlier_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key="basis_class", col_key="normalization", value_key="tail_outlier_fraction_median", title="Tail outlier fraction", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )

    aggregate = [row for row in stratified if row.get("stratum") == "all"]
    add(
        f"{_figure_label('4', 0)}_stratified_geometry_aggregate_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, aggregate, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title="Stratified median absolute energy error", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    add(
        f"{_figure_label('4', 0)}_stratified_geometry_aggregate_log_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, aggregate, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title="Stratified median absolute energy error", transform="positive_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )

    hooke_rows = tables["hooke_orbital_summary.csv"]
    for winner in WINNER_KINDS:
        rows = _winner_rows(hooke_rows, winner)
        add(
            _winner_filename("5A", winner, "hooke_orbital_local_energy_distribution.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("com_bin",), title=f"Hooke-orbital local-energy medians: {_winner_title(winner)}", legend_title="CoM bin"),
        )
        add(
            _winner_filename("5B", winner, "hooke_orbital_local_energy_vs_r12.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("com_bin",), title=f"Hooke-orbital local energy vs r12: {_winner_title(winner)}", legend_title="CoM bin"),
        )
        add(
            _winner_filename("5C", winner, "hooke_orbital_local_energy_vs_radius.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="R_norm_center", y_key="local_energy_median", group_keys=("r12_bin",), title=f"Hooke-orbital local energy vs CoM radius: {_winner_title(winner)}", legend_title="r12 bin"),
        )

    for metric_index, metric in enumerate(SYMMETRY_METRICS):
        add(
            f"{_figure_label('6', metric_index)}_symmetry_{metric}_heatmap_grid.png",
            lambda path, metric=metric: _save_symmetry_metric_grid(path, tables["symmetry_summary.csv"], metric_key=metric, title=f"Symmetry diagnostic: {metric}"),
        )

    for metric_index, metric in enumerate(FEATURE_TRACE_METRICS):
        add(
            f"{_figure_label('7', metric_index)}_feature_trace_{metric}_heatmap_grid.png",
            lambda path, metric=metric: _save_feature_trace_metric_grid(path, tables["trace_summary.csv"], metric_key=metric, title=f"Feature-trace stability diagnostic: {metric}"),
        )

    add(
        "8A_energy_winner_training_energy.png",
        lambda path: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind="energy", value_mode="energy_mean", y_label="energy mean", title="Final-train energy curves: energy winners"),
    )
    add(
        "8B_energy_winner_abs_energy_error_semilogy.png",
        lambda path: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind="energy", value_mode="abs_energy_error", y_label="abs energy error |E - 2|", title="Final-train absolute energy error: energy winners", semilogy=True),
    )
    add(
        "8C_stability_winner_training_energy.png",
        lambda path: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind="stability", value_mode="energy_mean", y_label="energy mean", title="Final-train energy curves: stability winners"),
    )
    add(
        "8D_stability_winner_abs_energy_error_semilogy.png",
        lambda path: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind="stability", value_mode="abs_energy_error", y_label="abs energy error |E - 2|", title="Final-train absolute energy error: stability winners", semilogy=True),
    )

    for stat_index, stat in enumerate(VIRIAL_RESIDUAL_STATS):
        add(
            f"{_figure_label('9', stat_index)}_virial_residual_{stat}_log_heatmap.png",
            lambda path, stat=stat: _save_virial_residual_heatmap(path, virial_residual, stat=stat),
        )

    strata = sorted({str(row.get("stratum", "")) for row in stratified if row.get("stratum", "") not in {"", "all"}})
    for stratum_index, stratum in enumerate(strata, start=1):
        label = _figure_label("4", stratum_index)
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stratum)
        rows = [row for row in stratified if str(row.get("stratum", "")) == stratum]
        add(
            f"{label}_stratified_geometry_{safe}_heatmap.png",
            lambda path, rows=rows, stratum=stratum: _save_winner_pair_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
        )
        add(
            f"{label}_stratified_geometry_{safe}_log_heatmap.png",
            lambda path, rows=rows, stratum=stratum: _save_winner_pair_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}", transform="positive_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
        )
    return written


def _copy_tables(collect_dir: Path, tables_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        shutil.copyfile(collect_dir / name, tables_dir / name)
        counts[name] = len(rows)
    return counts


def _write_energy_component_tables(tables_dir: Path, energy_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Write combined and per-winner virial tables from final MCMC energy rows."""

    by_winner = _energy_component_tables_by_winner(energy_rows)
    combined = [row for winner_id in sorted(by_winner) for row in by_winner[winner_id]]
    counts = {"energy_components_and_virial_by_winner.csv": len(combined)}
    _write_csv_columns(
        tables_dir / "energy_components_and_virial_by_winner.csv",
        combined,
        ENERGY_COMPONENT_COLUMNS,
    )
    for winner_id, rows in by_winner.items():
        relative_path = f"energy_components_and_virial/{winner_id}.csv"
        _write_csv_columns(tables_dir / relative_path, rows, ENERGY_COMPONENT_COLUMNS)
        counts[relative_path] = len(rows)
    return counts


def build_report(
    *,
    results_root: str | Path,
    report_attempt_id: str | None = None,
    final_collect_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Write ``09_final_report`` artifacts from compact collect outputs."""

    results_root = Path(results_root)
    final_collect_attempt_id = _resolve_collect_attempt_id(results_root, final_collect_attempt_id)
    report_attempt_id = report_attempt_id or final_collect_attempt_id or new_attempt_id()
    report_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    collect_dir, collect_manifest, tables = _load_collect_tables(results_root, final_collect_attempt_id)
    study = study_name_from_manifest(collect_manifest)
    table_counts = _copy_tables(collect_dir, tables_dir, tables)
    table_counts.update(_write_energy_component_tables(tables_dir, tables["energy_by_run.csv"]))
    figures = _write_figures(figures_dir, tables)

    report = {
        "study": study,
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": report_attempt_id,
        "final_collect_attempt_id": final_collect_attempt_id,
        "final_collect_dir": str(collect_dir),
        "tables": table_counts,
        "figures": figures,
        "caveats": [
            "final_report.py consumes 08_final_collect compact tables only.",
            "Raw final-eval and final-train artifacts are reduced by final_collect.py.",
            "Runtime/resource summaries are reported separately from model-quality ranking.",
        ],
    }
    write_json(report_dir / "final_report.json", report)
    (report_dir / "report.md").write_text(_report_markdown(report, tables))
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), report_attempt_id)
    return {"attempt_dir": str(report_dir), "report": report}


def _report_markdown(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    architecture = sorted(tables["architecture_summary.csv"], key=lambda row: (row.get("basis_class", ""), row.get("normalization", ""), row.get("winner_kind", "")))
    report_title = str(report.get("study") or "Study").replace("_", " ").title()
    lines = [
        f"# {report_title} Final Report",
        "",
        "## Scope And Provenance",
        "",
        "This report consumes `08_final_collect` compact tables only. It does not parse raw model, train, or eval records.",
        f"Final collect attempt: `{report['final_collect_attempt_id']}`.",
        "",
        "## Final Champion Summary",
        "",
        "See `tables/run_index.csv` and `tables/energy_by_run.csv`.",
        "",
        "## Family-Level Ranking",
        "",
    ]
    if architecture:
        lines.extend(["| basis_class | normalization | winner_kind | n_success/n_expected | energy_error_median | local_energy_var_median |", "|---|---|---|---:|---:|---:|"])
        for row in architecture[:20]:
            lines.append(
                f"| {row.get('basis_class', '')} | {row.get('normalization', '')} | {row.get('winner_kind', '')} | "
                f"{row.get('n_success', '')}/{row.get('n_expected', '')} | {row.get('energy_error_median', '')} | {row.get('local_energy_var_median', '')} |"
            )
    else:
        lines.append("No architecture summary rows were found.")
    lines.extend(
        [
            "",
            "## Energy And Local-Energy Results",
            "",
            "Energy figures use signed error relative to exact Hooke energy `E = 2`; heatmaps place energy and stability winners side by side with a shared color scale. Figure 1B separates energy and stability winners into adjacent panels while keeping architecture color and normalization marker encodings fixed. Signed-log heatmap variants use real-scale cell labels.",
            "",
            "Energy component and virial tables are written to `tables/energy_components_and_virial_by_winner.csv` and to one validation-style table per winner family under `tables/energy_components_and_virial/`. The virial residual is `2 * kinetic - 2 * harmonic_trap + electron_electron`; the relative residual divides its absolute value by the absolute component scale.",
            "",
            "## Cusp Diagnostics",
            "",
            "Cusp tables preserve center-of-mass and direction columns when present. Figures 2A/2B/2C emit separate energy/stability grids for sampled local-energy, log-amplitude, and finite-fraction profiles against `r12`; directions and seeds are pooled so each subplot has one line per CoM and variance error bars aggregate all compact direction/seed records. Figure 2D emits separate energy/stability grids for `d_logabs_dr_median` against `r12`, with normalization rows, architecture columns, solid CoM model lines, and dashed target derivative references.",
            "",
            "## Tail Diagnostics",
            "",
            "Tail tables preserve path columns. Figures 3A/3B show local-energy bar grids for energy and stability winners; bars are grouped by CoM and their error bars show q5-q85. Figures 3C/3D show logabs line grids with seed-variance error bars. Figure 3E summarizes tail outlier fraction.",
            "",
            "## Stratified Geometry Diagnostics",
            "",
            "Stratified summaries include per-stratum rows and `stratum=all` aggregate rows. Figures 4A, 4B, ... group the aggregate and per-stratum heatmaps; each group has a real-scale and positive-log-color version with energy and stability winners side by side on a shared color scale.",
            "",
            "## Hooke-Orbital Diagnostics",
            "",
            "Hooke-orbital summaries are binned by CoM-radius and `r12` bins. Figure 5 line plots are emitted separately for energy and stability winners, with normalization rows, architecture columns, and the remaining bin dimension in the external legend.",
            "",
            "## Symmetry Diagnostics",
            "",
            "See `tables/symmetry_summary.csv` and the symmetry figures. Figures 6A, 6B, ... emit one heatmap-grid figure per scalar symmetry metric; each grid uses symmetry tasks as rows, energy/stability winners as columns, architecture by normalization inside each subplot, and one shared color scale per symmetry-task row. Positive-only metric heatmaps use a monochrome red bar, with log color selected by default when a row's shared values span orders of magnitude.",
            "",
            "## Trace Diagnostics",
            "",
            "See `tables/trace_summary.csv` and the trace figures. Figures 7A, 7B, and 7C focus on feature-trace stability and emit one heatmap-grid figure each for `rms_q95`, `max_abs`, and `nonfinite_count`; each grid uses layer rows, energy/stability winner columns, architecture by normalization inside each subplot, and one tick-only colorbar with its own scale for every layer row.",
            "",
            "## Training And Resource Summary",
            "",
            "See `tables/training_curve_summary.csv` and `tables/resource_summary.csv`. Runtime is not mixed into quality ranking. Figures 8A and 8C show one smoothed training-energy curve per final-training run for energy and stability winners; Figures 8B and 8D show the corresponding semilogy absolute energy error curves with one shared vertical axis per grid.",
            "",
            "## Virial Diagnostics",
            "",
            "See `tables/energy_components_and_virial_by_winner.csv`. Figures 9A, 9B, 9C, and 9D show virial-residual mean, median, minimum, and maximum. Heatmaps place energy and stability winners side by side with signed-log color scales; cell labels remain on the real signed scale.",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(["", "## Next-Scan Implications", "", "Use energy, local-energy variance, pathology, cusp/tail, symmetry, and trace summaries jointly. Keep energy and stability winners separate.", "", "## Tables And Figures", "", "Tables:"])
    for name, n_rows in report["tables"].items():
        lines.append(f"- `tables/{name}`: {n_rows} rows")
    lines.append("")
    lines.append("Figures:")
    for name in report["figures"]:
        lines.append(f"- `figures/{name}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-report arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-collect-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write final report artifacts."""

    args = parse_args(argv)
    prefix = log_prefix()
    print(f"{prefix} final report results_root={args.results_root}")
    if args.final_collect_attempt_id:
        print(f"{prefix} final report using final_collect_attempt_id={args.final_collect_attempt_id}")
    else:
        print(f"{prefix} final report using latest final-collect attempt")
    result = build_report(
        results_root=args.results_root,
        report_attempt_id=args.attempt_id,
        final_collect_attempt_id=args.final_collect_attempt_id,
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} final report consumed 08_final_collect/{report['final_collect_attempt_id']} "
        f"-> {result['attempt_dir']}"
    )
    print(f"{prefix} final report copied table rows:")
    for filename, count in report["tables"].items():
        print(f"{prefix}   {filename}: {count}")
    print(f"{prefix} final report wrote {len(report['figures'])} figures")
    figure_counts: dict[str, int] = {}
    for figure in report["figures"]:
        section = figure.split("_", 1)[0]
        figure_counts[section] = figure_counts.get(section, 0) + 1
    for section, count in sorted(figure_counts.items()):
        print(f"{prefix}   figures {section}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
