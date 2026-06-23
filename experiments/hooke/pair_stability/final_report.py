"""Render final pair-stability reports from compact ``08_final_collect`` tables.

This stage is intentionally report-oriented and fast. It consumes only compact
tables written by ``final_collect.py``; it does not read raw final-eval task
records, final-train metrics, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from run_utils import (
    STAGE_FINAL_COLLECT,
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
WINNER_KINDS = ("energy", "stability")
SYMMETRY_METRICS = (
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
    "finite_fraction",
)

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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return math.fsum(clean) / len(clean) if clean else None


def _variance(values: Sequence[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    mean = math.fsum(clean) / len(clean)
    return math.fsum((value - mean) ** 2 for value in clean) / (len(clean) - 1)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def _resolve_collect_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    latest = stage_dir(results_root, STAGE_FINAL_COLLECT) / "latest.json"
    if latest.is_file():
        payload = read_json(latest)
        if payload.get("attempt_id"):
            return str(payload["attempt_id"])
    attempts = attempt_ids(stage_dir(results_root, STAGE_FINAL_COLLECT))
    if not attempts:
        raise FileNotFoundError(f"no final-collect attempts under {stage_dir(results_root, STAGE_FINAL_COLLECT)}")
    return attempts[-1]


def _load_collect_tables(results_root: Path, collect_attempt_id: str) -> tuple[Path, dict[str, list[dict[str, Any]]]]:
    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"missing final-collect attempt: {collect_dir}")
    return collect_dir, {name: _read_csv(collect_dir / name) for name in COMPACT_TABLES}


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rhu/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_no_data(path: Path, title: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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


def _heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return real-scale heatmap cell means for plotting and annotations."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get(value_key))
        if value is None:
            continue
        cells[(str(row.get(row_key, "")), str(row.get(col_key, "")))].append(value)
    if not cells:
        return [], [], []
    y_labels = sorted({key[0] for key in cells})
    x_labels = sorted({key[1] for key in cells})
    matrix = []
    for y in y_labels:
        row_values = []
        for x in x_labels:
            row_values.append(_mean(cells.get((y, x), [])))
        matrix.append(row_values)
    return y_labels, x_labels, matrix


def _draw_heatmap_axis(
    fig: Any,
    ax: Any,
    *,
    y_labels: Sequence[str],
    x_labels: Sequence[str],
    matrix: Sequence[Sequence[float | None]],
    value_key: str,
    title: str,
    transform: str | None,
) -> None:
    """Draw one heatmap axis with real-scale annotations."""

    from matplotlib.colors import SymLogNorm

    finite_values = [value for row in matrix for value in row if value is not None]
    if not finite_values:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        return

    vmax = max(abs(value) for value in finite_values)
    vmax = vmax if vmax > 0.0 else 1.0
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    colorbar_label = value_key
    if transform == "signed_log":
        nonzero = [abs(value) for value in finite_values if value != 0.0]
        norm = SymLogNorm(linthresh=min(nonzero), vmin=-vmax, vmax=vmax, base=10) if nonzero else None
        image = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
        colorbar_label = f"{value_key} (symmetric log color; labels are real scale)"
    else:
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value is not None:
                ax.text(x_index, y_index, f"{value:.2g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)


def _save_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_labels, x_labels, matrix = _heatmap_matrix(rows, row_key=row_key, col_key=col_key, value_key=value_key)
    if not matrix:
        _save_no_data(path, title)
        return

    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(x_labels)), max(3.5, 0.8 * len(y_labels))))
    _draw_heatmap_axis(
        fig,
        ax,
        y_labels=y_labels,
        x_labels=x_labels,
        matrix=matrix,
        value_key=value_key,
        title=title,
        transform=transform,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
                "winner_kind": str(row.get("winner_kind", "")),
            }
        )
    return points


def _save_energy_variance_scatter(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = _energy_variance_points(rows)
    if not points:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    from matplotlib.lines import Line2D

    architectures = sorted({str(point["architecture"]) for point in points})
    normalizations = sorted({str(point["normalization"]) for point in points})
    cmap = plt.get_cmap("tab20" if len(architectures) > 10 else "tab10")
    colors = {architecture: cmap(index % cmap.N) for index, architecture in enumerate(architectures)}
    markers = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">", "h", "p"]
    marker_by_norm = {normalization: markers[index % len(markers)] for index, normalization in enumerate(normalizations)}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for point in points:
        ax.scatter(
            point["abs_energy_error"],
            point["local_energy_var"],
            color=colors[str(point["architecture"])],
            marker=marker_by_norm[str(point["normalization"])],
            s=58,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("abs energy error |E - 2|")
    ax.set_ylabel("local-energy variance")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[architecture], markeredgecolor="black", markersize=7, label=architecture)
        for architecture in architectures
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker_by_norm[normalization], color="black", markerfacecolor="lightgray", markeredgecolor="black", linestyle="none", markersize=7, label=normalization)
        for normalization in normalizations
    ]
    architecture_legend = ax.legend(handles=color_handles, title="Architecture", fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.add_artist(architecture_legend)
    ax.legend(handles=shape_handles, title="Normalization", fontsize=7, title_fontsize=8, loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    return centers, [counts_by_center[center] for center in centers], [widths_by_center[center] for center in centers]


def _save_local_energy_distribution_grid(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalizations, architectures, groups = _local_energy_distribution_groups(rows)
    if not groups:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    n_rows = len(normalizations)
    n_cols = len(architectures)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.0, 3.2 * n_cols), max(3.0, 2.4 * n_rows)), squeeze=False, sharex=False, sharey=False)
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            values = groups.get((normalization, architecture), [])
            if values:
                centers, counts, widths = _local_energy_bar_series(values)
                if centers:
                    ax.bar(centers, counts, width=widths, align="center", color="#4C78A8", edgecolor="black", alpha=0.85)
                else:
                    ax.text(0.5, 0.5, "No finite samples", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\ncount")
            if row_index == n_rows - 1:
                ax.set_xlabel("local_energy")
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _com_label(raw: Any) -> str:
    value = str(raw).strip()
    return f"CoM {value}" if value else "CoM all"


def _cusp_line_label(row: dict[str, Any]) -> str:
    com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
    direction = str(row.get("direction_id", "")).strip()
    return f"{com} / dir {direction}" if direction else com


def _cusp_seed_profile_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return cusp profile points with seed variance for each CoM/direction line."""

    per_seed: dict[tuple[str, str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        r12 = _as_float(row.get("r12"))
        value = _as_float(row.get(value_key))
        if r12 is None or value is None:
            continue
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        line = _cusp_line_label(row)
        seed = str(row.get("seed_index", row.get("final_run_id", "")))
        per_seed[(basis, normalization, line, r12, seed)].append(value)

    by_r12: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for (basis, normalization, line, r12, _seed), values in per_seed.items():
        mean = _mean(values)
        if mean is not None:
            by_r12[(basis, normalization, line, r12)].append(mean)

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, line, r12), seed_values in sorted(by_r12.items()):
        mean = _mean(seed_values)
        variance = _variance(seed_values)
        if mean is None or variance is None:
            continue
        out[(basis, normalization, line)].append(
            {
                "r12": r12,
                "mean": mean,
                "variance": variance,
                "n_seeds": len(seed_values),
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

    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = _cusp_seed_profile_points(rows, winner_kind=winner_kind, value_key=value_key)
    if not profiles:
        _save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    line_labels = sorted({cell[2] for cell in profiles})

    plt = _pyplot()
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(line_labels)}
    fig, axes = plt.subplots(
        len(normalizations),
        len(architectures),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.1 * len(normalizations))),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            plotted = False
            for label in line_labels:
                points = sorted(profiles.get((architecture, normalization, label), []), key=lambda item: float(item["r12"]))
                if not points:
                    continue
                container = ax.errorbar(
                    [float(point["r12"]) for point in points],
                    [float(point["mean"]) for point in points],
                    yerr=[float(point["variance"]) for point in points],
                    marker="o",
                    linewidth=1.0,
                    markersize=2.8,
                    capsize=2.0,
                    color=colors[label],
                    label=label,
                )
                legend_handles.setdefault(label, container)
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\n{y_label}")
            if row_index == len(normalizations) - 1:
                ax.set_xlabel("r12")
            ax.grid(True, linewidth=0.35, alpha=0.35)

    fig.suptitle(f"{title}\nLines are CoM/direction groups; error bars are seed variance over final replicates.", y=0.995)
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=7,
            title="CoM / direction",
            title_fontsize=8,
            borderaxespad=0.0,
            ncol=max(1, math.ceil(len(legend_handles) / 28)),
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.86 if legend_handles else 1.0, 0.94))
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


def _save_tail_winner_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("local energy", "local_energy_median"),
        ("logabs", "logabs_median"),
    )
    profile_by_metric = {
        label: _tail_seed_profile_points(rows, winner_kind=winner_kind, value_key=value_key)
        for label, value_key in metrics
    }
    cells = {cell for profiles in profile_by_metric.values() for cell in profiles}
    if not cells:
        _save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in cells})
    normalizations = sorted({cell[1] for cell in cells})
    com_labels = sorted({cell[2] for cell in cells})

    plt = _pyplot()
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(com_labels)}
    n_rows = len(metrics) * len(normalizations)
    n_cols = len(architectures)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5.0, 3.1 * n_cols), max(3.6, 2.2 * n_rows)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    for metric_index, (metric_label, _value_key) in enumerate(metrics):
        profiles = profile_by_metric[metric_label]
        for norm_index, normalization in enumerate(normalizations):
            row_index = metric_index * len(normalizations) + norm_index
            for col_index, architecture in enumerate(architectures):
                ax = axes[row_index][col_index]
                plotted = False
                for com in com_labels:
                    points = sorted(profiles.get((architecture, normalization, com), []), key=lambda item: float(item["radius"]))
                    if not points:
                        continue
                    container = ax.errorbar(
                        [float(point["radius"]) for point in points],
                        [float(point["mean"]) for point in points],
                        yerr=[float(point["variance"]) for point in points],
                        marker="o",
                        linewidth=1.1,
                        markersize=3.0,
                        capsize=2.0,
                        color=colors[com],
                        label=com,
                    )
                    legend_handles.setdefault(com, container)
                    plotted = True
                if not plotted:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                if row_index == 0:
                    ax.set_title(architecture, fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"{normalization}\n{metric_label}\nseed mean")
                if row_index == n_rows - 1:
                    ax.set_xlabel("radius")
                ax.grid(True, linewidth=0.35, alpha=0.35)

    fig.suptitle(f"{title}\nLines are CoM groups; error bars are seed variance over final replicates.", y=0.995)
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            title="CoM",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(6, len(legend_handles)),
            fontsize=7,
            title_fontsize=8,
        )
    top = 0.91 if legend_handles else 0.94
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        groups[_group_label(row, group_keys)].append((x, y))
    if not groups:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in sorted(groups.items()):
        values = sorted(values)
        ax.plot([point[0] for point in values], [point[1] for point in values], marker="o", label=label)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    if legend == "outside":
        ax.legend(
            fontsize=6,
            title=legend_title,
            title_fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            ncol=max(1, math.ceil(len(groups) / 24)),
        )
    elif legend == "auto" and len(groups) <= 12:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        groups[(architecture, _group_label(row, group_keys))].append((x, y))
    if not groups:
        _save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in groups})
    labels = sorted({key[1] for key in groups})
    plt = _pyplot()
    from matplotlib.lines import Line2D

    n_cols = min(3, max(1, math.ceil(math.sqrt(len(architectures)))))
    n_rows = math.ceil(len(architectures) / n_cols)
    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5.0, 3.4 * n_cols), max(3.0, 2.6 * n_rows)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    flat_axes = [axis for axis_row in axes for axis in axis_row]
    for index, architecture in enumerate(architectures):
        ax = flat_axes[index]
        plotted = False
        for label in labels:
            values = sorted(groups.get((architecture, label), []))
            if not values:
                continue
            ax.plot(
                [point[0] for point in values],
                [point[1] for point in values],
                marker="o",
                linewidth=1.1,
                markersize=3.0,
                color=colors[label],
                label=label,
            )
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(architecture, fontsize=9)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, linewidth=0.35, alpha=0.35)
    for ax in flat_axes[len(architectures):]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color=colors[label], linewidth=1.1, markersize=3.0, label=label)
        for label in labels
    ]
    if handles:
        fig.legend(
            handles,
            labels,
            title=legend_title,
            fontsize=6,
            title_fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0.0,
            ncol=max(1, math.ceil(len(labels) / 28)),
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.83 if handles else 1.0, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _scalar_metric_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    row_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return a matrix of seed/record means for scalar diagnostic metrics."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        label = _group_label(row, row_keys)
        for metric in metric_keys:
            value = _as_float(row.get(metric))
            if value is not None:
                cells[(label, metric)].append(value)
    if not cells:
        return [], [], []
    y_labels = sorted({key[0] for key in cells})
    x_labels = [metric for metric in metric_keys if any((label, metric) in cells for label in y_labels)]
    matrix = []
    for label in y_labels:
        matrix.append([_mean(cells.get((label, metric), [])) for metric in x_labels])
    return y_labels, x_labels, matrix


def _save_scalar_metric_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_keys: Sequence[str],
    metric_keys: Sequence[str],
    title: str,
) -> None:
    """Save a heatmap for tables with multiple scalar metrics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    y_labels, x_labels, matrix = _scalar_metric_matrix(rows, row_keys=row_keys, metric_keys=metric_keys)
    if not matrix:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(x_labels)), max(3.8, 0.38 * len(y_labels))))
    _draw_heatmap_axis(
        fig,
        ax,
        y_labels=y_labels,
        x_labels=x_labels,
        matrix=matrix,
        value_key="mean scalar value",
        title=title,
        transform=None,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_symmetry_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
) -> None:
    """Save one symmetry metric as architecture-by-normalization heatmaps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    symmetries = _unique_in_order(row.get("symmetry_task", "") for row in rows)
    if not symmetries:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(symmetries),
        len(WINNER_KINDS),
        figsize=(max(7.5, 4.0 * len(WINNER_KINDS)), max(3.2, 3.0 * len(symmetries))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, symmetry in enumerate(symmetries):
        symmetry_rows = [row for row in rows if str(row.get("symmetry_task", "")) == symmetry]
        for col_index, winner in enumerate(WINNER_KINDS):
            ax = axes[row_index][col_index]
            panel_rows = _winner_rows(symmetry_rows, winner)
            y_labels, x_labels, matrix = _heatmap_matrix(
                panel_rows,
                row_key="basis_class",
                col_key="normalization",
                value_key=metric_key,
            )
            _draw_heatmap_axis(
                fig,
                ax,
                y_labels=y_labels,
                x_labels=x_labels,
                matrix=matrix,
                value_key=metric_key,
                title=f"{symmetry}: {_winner_title(winner)}",
                transform=None,
            )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_figures(figures_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    architecture = tables["architecture_summary.csv"]
    energy = tables["energy_by_run.csv"]
    histograms = tables["local_energy_histograms.csv"]
    stratified = tables["stratified_summary.csv"]

    def add(filename: str, writer: Any) -> None:
        writer(figures_dir / filename)
        written.append(filename)

    for winner in WINNER_KINDS:
        rows = _winner_rows(architecture, winner)
        add(
            _winner_filename("1A", winner, "real_scale_energy_error_heatmap.png"),
            lambda path, rows=rows, winner=winner: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="energy_error_median", title=f"Median signed final energy error: {_winner_title(winner)}"),
        )
        add(
            _winner_filename("1A", winner, "log_scale_energy_error_heatmap.png"),
            lambda path, rows=rows, winner=winner: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="energy_error_median", title=f"Median signed final energy error: {_winner_title(winner)}", transform="signed_log"),
        )
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
    add("3A_tail_energy_winner_grid.png", lambda path: _save_tail_winner_grid(path, tables["tail_profile_summary.csv"], winner_kind="energy", title="Tail profiles: energy winners"))
    add("3B_tail_stability_winner_grid.png", lambda path: _save_tail_winner_grid(path, tables["tail_profile_summary.csv"], winner_kind="stability", title="Tail profiles: stability winners"))

    for winner in WINNER_KINDS:
        add(
            _winner_filename("3C", winner, "tail_outlier_heatmap.png"),
            lambda path, winner=winner: _save_heatmap(path, _winner_rows(architecture, winner), row_key="basis_class", col_key="normalization", value_key="tail_outlier_fraction_median", title=f"Tail outlier fraction: {_winner_title(winner)}"),
        )

    aggregate = [row for row in stratified if row.get("stratum") == "all"]
    for winner in WINNER_KINDS:
        rows = _winner_rows(aggregate, winner)
        add(
            _winner_filename("4", winner, "stratified_geometry_aggregate_heatmap.png"),
            lambda path, rows=rows, winner=winner: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {_winner_title(winner)}"),
        )
        add(
            _winner_filename("4", winner, "stratified_geometry_aggregate_log_heatmap.png"),
            lambda path, rows=rows, winner=winner: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {_winner_title(winner)}", transform="signed_log"),
        )

    hooke_rows = tables["hooke_orbital_summary.csv"]
    for winner in WINNER_KINDS:
        rows = _winner_rows(hooke_rows, winner)
        add(
            _winner_filename("5A", winner, "hooke_orbital_local_energy_distribution.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("normalization", "com_bin"), title=f"Hooke-orbital local-energy medians: {_winner_title(winner)}", legend_title="normalization / CoM bin"),
        )
        add(
            _winner_filename("5B", winner, "hooke_orbital_local_energy_vs_r12.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("normalization", "com_bin"), title=f"Hooke-orbital local energy vs r12: {_winner_title(winner)}", legend_title="normalization / CoM bin"),
        )
        add(
            _winner_filename("5C", winner, "hooke_orbital_local_energy_vs_radius.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_line_grid(path, rows, x_key="R_norm_center", y_key="local_energy_median", group_keys=("normalization", "r12_bin"), title=f"Hooke-orbital local energy vs CoM radius: {_winner_title(winner)}", legend_title="normalization / r12 bin"),
        )

    for metric in SYMMETRY_METRICS:
        add(
            f"6_symmetry_{metric}_heatmap_grid.png",
            lambda path, metric=metric: _save_symmetry_metric_grid(path, tables["symmetry_summary.csv"], metric_key=metric, title=f"Symmetry diagnostic: {metric}"),
        )

    for winner in WINNER_KINDS:
        add(
            _winner_filename("7", winner, "trace_scalar_heatmap.png"),
            lambda path, winner=winner: _save_scalar_metric_heatmap(path, _winner_rows(tables["trace_summary.csv"], winner), row_keys=("basis_class", "normalization", "trace_kind", "layer"), metric_keys=("rms_q95", "rms_q99", "max_abs", "nonfinite_count", "compared_entry_count", "comparison_error_count", "max_equivariance_error"), title=f"Trace scalar diagnostics: {_winner_title(winner)}"),
        )

    add("8_training_curves.png", lambda path: _save_line_plot(path, tables["training_curve_summary.csv"], x_key="step", y_key="energy_mean", group_keys=("basis_class", "normalization", "winner_kind"), title="Final train energy curves", legend="outside", legend_title="architecture / normalization / winner"))

    strata = sorted({str(row.get("stratum", "")) for row in stratified if row.get("stratum", "") not in {"", "all"}})
    for stratum in strata:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stratum)
        rows = [row for row in stratified if str(row.get("stratum", "")) == stratum]
        for winner in WINNER_KINDS:
            winner_rows = _winner_rows(rows, winner)
            add(
                _winner_filename("4", winner, f"stratified_geometry_{safe}_heatmap.png"),
                lambda path, rows=winner_rows, winner=winner, stratum=stratum: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}: {_winner_title(winner)}"),
            )
            add(
                _winner_filename("4", winner, f"stratified_geometry_{safe}_log_heatmap.png"),
                lambda path, rows=winner_rows, winner=winner, stratum=stratum: _save_heatmap(path, rows, row_key="basis_class", col_key="normalization", value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}: {_winner_title(winner)}", transform="signed_log"),
            )
    return written


def _copy_tables(collect_dir: Path, tables_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        shutil.copyfile(collect_dir / name, tables_dir / name)
        counts[name] = len(rows)
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
    collect_dir, tables = _load_collect_tables(results_root, final_collect_attempt_id)
    table_counts = _copy_tables(collect_dir, tables_dir, tables)
    figures = _write_figures(figures_dir, tables)

    report = {
        "study": "pair_stability",
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
    lines = [
        "# Hooke Pair-Stability Final Report",
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
            "Energy figures use signed error relative to exact Hooke energy `E = 2`; grid figures are emitted separately for energy and stability winners. Signed-log heatmap variants use real-scale cell labels.",
            "",
            "## Cusp Diagnostics",
            "",
            "Cusp tables preserve center-of-mass and direction columns when present. Figures 2A/2B/2C emit separate energy/stability grids for sampled local-energy, log-amplitude, and finite-fraction profiles against `r12`, with shared CoM/direction legends. Aggregate cusp-slope diagnostics remain in the metrics tables.",
            "",
            "## Tail Diagnostics",
            "",
            "Tail tables preserve path columns. Figures 3A/3B split energy and stability winners into subplot grids by architecture and normalization; each subplot draws CoM lines with seed-variance error bars for local energy and logabs, with a shared CoM legend. Exact log-amplitude references are included when collect inputs provide them.",
            "",
            "## Stratified Geometry Diagnostics",
            "",
            "Stratified summaries include per-stratum rows and `stratum=all` aggregate rows. Each Figure 4 heatmap has a real-scale and signed-log-color version, emitted separately for energy and stability winners.",
            "",
            "## Hooke-Orbital Diagnostics",
            "",
            "Hooke-orbital summaries are binned by CoM-radius and `r12` bins. Figure 5 line plots are emitted separately for energy and stability winners, split architectures into separate subplots, and place the remaining group legend outside the plotting area.",
            "",
            "## Symmetry Diagnostics",
            "",
            "See `tables/symmetry_summary.csv` and the symmetry figures. Figure 6 emits one heatmap-grid figure per scalar symmetry metric; each grid uses symmetry tasks as rows, energy/stability winners as columns, and architecture by normalization inside each subplot.",
            "",
            "## Trace Diagnostics",
            "",
            "See `tables/trace_summary.csv` and the trace figures. Figure 7 emits separate energy/stability heatmaps for scalar trace diagnostics.",
            "",
            "## Training And Resource Summary",
            "",
            "See `tables/training_curve_summary.csv` and `tables/resource_summary.csv`. Runtime is not mixed into quality ranking. Figure 8 places the architecture/normalization/winner legend outside the plotting area.",
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
    result = build_report(
        results_root=args.results_root,
        report_attempt_id=args.attempt_id,
        final_collect_attempt_id=args.final_collect_attempt_id,
    )
    report = result["report"]
    print(
        f"[pair_stability] final report consumed 08_final_collect/{report['final_collect_attempt_id']} "
        f"-> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
