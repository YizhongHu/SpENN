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
EXACT_HOOKE_ENERGY = 2.0
WINNER_KINDS = ("energy", "stability")
NARROW_WINNER_HEATMAP_WIDTH_SCALE = 0.75
POSITIVE_HEATMAP_CMAP = "Reds"
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


def _median(values: Sequence[float]) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


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


def _matrix_values(matrix: Sequence[Sequence[float | None]]) -> list[float]:
    return [value for row in matrix for value in row if value is not None]


def _heatmap_colorbar_label(value_key: str, transform: str | None) -> str:
    if transform == "signed_log":
        return f"{value_key} (symmetric log color; labels are real scale)"
    if transform == "positive_log":
        return f"{value_key} (monochrome log color; labels are real scale)"
    if transform == "positive_linear":
        return f"{value_key} (monochrome color; labels are real scale)"
    return value_key


def _resolve_heatmap_transform(values: Sequence[float], requested: str | None) -> str:
    """Choose a heatmap color scale from the finite shared values."""

    if requested is not None:
        return requested
    finite = [value for value in values if math.isfinite(value)]
    if finite and min(finite) >= 0.0:
        positive = [value for value in finite if value > 0.0]
        if positive and max(positive) / min(positive) >= 10.0:
            return "positive_log"
        return "positive_linear"
    return "signed_linear"


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
    scale_values: Sequence[float] | None = None,
    add_colorbar: bool = True,
) -> Any | None:
    """Draw one heatmap axis with real-scale annotations."""

    from matplotlib.colors import LogNorm, SymLogNorm

    finite_values = _matrix_values(matrix)
    if not finite_values:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        return None

    scale = [value for value in (scale_values or finite_values) if math.isfinite(value)]
    if not scale:
        scale = finite_values
    vmax = max(abs(value) for value in scale)
    vmax = vmax if vmax > 0.0 else 1.0
    resolved_transform = _resolve_heatmap_transform(scale, transform)
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    if resolved_transform == "signed_log":
        nonzero = [abs(value) for value in scale if value != 0.0]
        norm = SymLogNorm(linthresh=min(nonzero), vmin=-vmax, vmax=vmax, base=10) if nonzero else None
        image = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
    elif resolved_transform == "positive_log":
        positive = [value for value in scale if value > 0.0]
        if positive:
            vmin = min(positive)
            positive_data = [[math.nan if value is None else max(value, vmin) for value in row] for row in matrix]
            positive_vmax = max(positive)
            if positive_vmax <= vmin:
                positive_vmax = vmin * 1.000001
            image = ax.imshow(positive_data, cmap=POSITIVE_HEATMAP_CMAP, norm=LogNorm(vmin=vmin, vmax=positive_vmax, clip=True), aspect="auto")
        else:
            image = ax.imshow(data, cmap=POSITIVE_HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    elif resolved_transform == "positive_linear":
        image = ax.imshow(data, cmap=POSITIVE_HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    else:
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value is not None:
                ax.text(x_index, y_index, f"{value:.2g}", ha="center", va="center", fontsize=8)
    if add_colorbar:
        fig.colorbar(image, ax=ax, label=_heatmap_colorbar_label(value_key, resolved_transform), fraction=0.046, pad=0.04)
    return image


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

    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = {}
    scale_values = []
    max_x = 0
    max_y = 0
    for winner in WINNER_KINDS:
        y_labels, x_labels, matrix = _heatmap_matrix(
            _winner_rows(rows, winner),
            row_key=row_key,
            col_key=col_key,
            value_key=value_key,
        )
        matrices[winner] = (y_labels, x_labels, matrix)
        scale_values.extend(_matrix_values(matrix))
        max_x = max(max_x, len(x_labels))
        max_y = max(max_y, len(y_labels))
    if not scale_values:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    width = max(7.0, 2.6 * max_x * len(WINNER_KINDS) * width_scale)
    fig, axes = plt.subplots(1, len(WINNER_KINDS), figsize=(width, max(3.5, 0.8 * max_y)), squeeze=False)
    images = []
    for col_index, winner in enumerate(WINNER_KINDS):
        y_labels, x_labels, matrix = matrices[winner]
        image = _draw_heatmap_axis(
            fig,
            axes[0][col_index],
            y_labels=y_labels,
            x_labels=x_labels,
            matrix=matrix,
            value_key=value_key,
            title=_winner_title(winner),
            transform=transform,
            scale_values=scale_values,
            add_colorbar=False,
        )
        if image is not None:
            images.append(image)
        if col_index > 0:
            axes[0][col_index].set_yticklabels([])
    if images:
        fig.colorbar(images[0], ax=list(axes.ravel()), label=_heatmap_colorbar_label(value_key, transform), fraction=0.046, pad=0.04)
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.84, wspace=0.45)
    fig.savefig(path, dpi=160, bbox_inches="tight")
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
                "winner_kind": "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability",
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

    fig, axes = plt.subplots(1, len(WINNER_KINDS), figsize=(11.0, 4.8), sharex=True, sharey=True)
    for ax, winner_kind in zip(axes, WINNER_KINDS, strict=True):
        winner_points = [point for point in points if str(point["winner_kind"]) == winner_kind]
        for point in winner_points:
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
        if not winner_points:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("abs energy error |E - 2|")
        ax.set_title(_winner_title(winner_kind))
        ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("local-energy variance")

    color_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[architecture], markeredgecolor="black", markersize=7, label=architecture)
        for architecture in architectures
    ]
    shape_handles = [
        Line2D([0], [0], marker=marker_by_norm[normalization], color="black", markerfacecolor="lightgray", markeredgecolor="black", linestyle="none", markersize=7, label=normalization)
        for normalization in normalizations
    ]
    architecture_legend = axes[-1].legend(handles=color_handles, title="Architecture", fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axes[-1].add_artist(architecture_legend)
    axes[-1].legend(handles=shape_handles, title="Normalization", fontsize=7, title_fontsize=8, loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.suptitle(f"{title}\nWinner type is separated by panel; color is architecture and marker shape is normalization.", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.90))
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

    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = _cusp_profile_points(rows, winner_kind=winner_kind, value_key=value_key)
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

    fig.suptitle(f"{title}\nLines are CoM groups; means and variances pool all direction/seed records.", y=0.995)
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=7,
            title="CoM",
            title_fontsize=8,
            borderaxespad=0.0,
            ncol=max(1, math.ceil(len(legend_handles) / 28)),
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.86 if legend_handles else 1.0, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        _save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in model_profiles})
    normalizations = sorted({key[1] for key in model_profiles})
    com_labels = sorted({key[2] for key in model_profiles})
    plt = _pyplot()
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
    n_rows = len(normalizations)
    n_cols = len(metrics) * len(architectures)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(8.0, 2.7 * n_cols), max(3.6, 2.2 * n_rows)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    for metric_index, (metric_label, _value_key) in enumerate(metrics):
        profiles = profile_by_metric[metric_label]
        for norm_index, normalization in enumerate(normalizations):
            row_index = norm_index
            for col_index, architecture in enumerate(architectures):
                axis_col = metric_index * len(architectures) + col_index
                ax = axes[row_index][axis_col]
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
                    ax.set_title(f"{metric_label}\n{architecture}", fontsize=9)
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
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            ncol=max(1, math.ceil(len(legend_handles) / 28)),
            fontsize=7,
            title_fontsize=8,
            borderaxespad=0.0,
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.88 if legend_handles else 1.0, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight")
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

    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        groups[(architecture, normalization, _group_label(row, group_keys))].append((x, y))
    if not groups:
        _save_no_data(path, title)
        return

    x_label = x_key.replace("_", " ")
    y_label = y_key.replace("_", " ")
    architectures = sorted({key[0] for key in groups})
    normalizations = sorted({key[1] for key in groups})
    labels = sorted({key[2] for key in groups})
    plt = _pyplot()
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(
        len(normalizations),
        len(architectures),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            plotted = False
            for label in labels:
                values = sorted(groups.get((architecture, normalization, label), []))
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
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\n{y_label}")
            if row_index == len(normalizations) - 1:
                ax.set_xlabel(x_label)
            ax.grid(True, linewidth=0.35, alpha=0.35)

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
    fig.tight_layout(rect=(0.0, 0.0, 0.84 if handles else 1.0, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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

    path.parent.mkdir(parents=True, exist_ok=True)
    curves = _training_run_curves(_winner_rows(rows, winner_kind), value_mode=value_mode, smooth_window=smooth_window)
    curves = {key: points for key, points in curves.items() if key[2] == winner_kind}
    if not curves:
        _save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in curves})
    normalizations = sorted({key[1] for key in curves})
    runs_by_cell: dict[tuple[str, str], list[tuple[str, list[dict[str, float | str]]]]] = defaultdict(list)
    for (architecture, normalization, _winner, run_id), points in curves.items():
        runs_by_cell[(architecture, normalization)].append((run_id, points))

    plt = _pyplot()
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(
        len(normalizations),
        len(architectures),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        squeeze=False,
        sharex=True,
        sharey=semilogy,
    )
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            if semilogy:
                ax.set_yscale("log")
            plotted = False
            run_curves = sorted(runs_by_cell.get((architecture, normalization), []), key=lambda item: item[0])
            for run_index, (_run_id, points) in enumerate(run_curves):
                points = sorted(points, key=lambda point: float(point["step"]))
                if semilogy:
                    points = [point for point in points if float(point["value"]) > 0.0]
                if not points:
                    continue
                ax.plot(
                    [float(point["step"]) for point in points],
                    [float(point["value"]) for point in points],
                    linewidth=0.9,
                    alpha=0.45,
                    color=cmap(run_index % cmap.N),
                )
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.97, 0.94, f"n={len(run_curves)}", ha="right", va="top", transform=ax.transAxes, fontsize=7)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\n{y_label}")
            if row_index == len(normalizations) - 1:
                ax.set_xlabel("step")
            ax.grid(True, linewidth=0.35, alpha=0.35)

    fig.suptitle(f"{title}\nEach line is one final-training run; curves use a {smooth_window}-point centered rolling mean.", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=160, bbox_inches="tight")
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
    matrices = {}
    scale_values_by_symmetry: dict[str, list[float]] = defaultdict(list)
    for symmetry in symmetries:
        symmetry_rows = [row for row in rows if str(row.get("symmetry_task", "")) == symmetry]
        for winner in WINNER_KINDS:
            y_labels, x_labels, matrix = _heatmap_matrix(
                _winner_rows(symmetry_rows, winner),
                row_key="basis_class",
                col_key="normalization",
                value_key=metric_key,
            )
            matrices[(symmetry, winner)] = (y_labels, x_labels, matrix)
            scale_values_by_symmetry[symmetry].extend(_matrix_values(matrix))
    if not any(scale_values_by_symmetry.values()):
        _save_no_data(path, title)
        return

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(symmetries),
        len(WINNER_KINDS),
        figsize=(max(14.0, 7.0 * len(WINNER_KINDS)), max(3.2, 3.0 * len(symmetries))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, symmetry in enumerate(symmetries):
        row_scale_values = scale_values_by_symmetry[symmetry]
        row_images = []
        for col_index, winner in enumerate(WINNER_KINDS):
            ax = axes[row_index][col_index]
            y_labels, x_labels, matrix = matrices[(symmetry, winner)]
            image = _draw_heatmap_axis(
                fig,
                ax,
                y_labels=y_labels,
                x_labels=x_labels,
                matrix=matrix,
                value_key=metric_key,
                title=f"{symmetry}\n{_winner_title(winner)}",
                transform=None,
                scale_values=row_scale_values,
                add_colorbar=False,
            )
            if image is not None:
                row_images.append(image)
            if col_index > 0:
                ax.set_yticklabels([])
        if row_images:
            colorbar = fig.colorbar(
                row_images[0],
                ax=list(axes[row_index]),
                fraction=0.035,
                pad=0.035,
            )
            colorbar.set_ticks([])
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.08, top=0.90, wspace=0.55, hspace=0.65)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_feature_trace_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
) -> None:
    """Save one feature-trace metric as layer-by-winner heatmaps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = [
        row
        for row in rows
        if str(row.get("trace_kind", "")) == "feature_trace_stability"
        and str(row.get("layer", "")) not in FEATURE_TRACE_EXCLUDED_LAYERS
    ]
    layers = _unique_in_order(row.get("layer", "") for row in trace_rows)
    if not layers:
        _save_no_data(path, title)
        return

    matrices = {}
    scale_values = []
    for layer in layers:
        layer_rows = [row for row in trace_rows if str(row.get("layer", "")) == layer]
        for winner in WINNER_KINDS:
            y_labels, x_labels, matrix = _heatmap_matrix(
                _winner_rows(layer_rows, winner),
                row_key="basis_class",
                col_key="normalization",
                value_key=metric_key,
            )
            matrices[(layer, winner)] = (y_labels, x_labels, matrix)
            scale_values.extend(_matrix_values(matrix))
    if not scale_values:
        _save_no_data(path, title)
        return

    plt = _pyplot()
    fig, axes = plt.subplots(
        len(layers),
        len(WINNER_KINDS),
        figsize=(max(11.0, 5.5 * len(WINNER_KINDS)), max(3.2, 2.55 * len(layers))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    images = []
    for row_index, layer in enumerate(layers):
        for col_index, winner in enumerate(WINNER_KINDS):
            ax = axes[row_index][col_index]
            y_labels, x_labels, matrix = matrices[(layer, winner)]
            image = _draw_heatmap_axis(
                fig,
                ax,
                y_labels=y_labels,
                x_labels=x_labels,
                matrix=matrix,
                value_key=metric_key,
                title=f"{layer}\n{_winner_title(winner)}",
                transform=None,
                scale_values=scale_values,
                add_colorbar=False,
            )
            if image is not None:
                images.append(image)
            if col_index > 0:
                ax.set_yticklabels([])
    if images:
        fig.colorbar(images[0], ax=list(axes.ravel()), label=metric_key, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.04, top=0.94, wspace=0.65, hspace=0.75)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_figures(figures_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    architecture_rows = tables["architecture_summary.csv"]
    energy = tables["energy_by_run.csv"]
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
    add("3A_tail_energy_winner_grid.png", lambda path: _save_tail_winner_grid(path, tables["tail_profile_summary.csv"], winner_kind="energy", title="Tail profiles: energy winners"))
    add("3B_tail_stability_winner_grid.png", lambda path: _save_tail_winner_grid(path, tables["tail_profile_summary.csv"], winner_kind="stability", title="Tail profiles: stability winners"))

    add(
        "3C_tail_outlier_heatmap.png",
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
            "Energy figures use signed error relative to exact Hooke energy `E = 2`; heatmaps place energy and stability winners side by side with a shared color scale. Figure 1B separates energy and stability winners into adjacent panels while keeping architecture color and normalization marker encodings fixed. Signed-log heatmap variants use real-scale cell labels.",
            "",
            "## Cusp Diagnostics",
            "",
            "Cusp tables preserve center-of-mass and direction columns when present. Figures 2A/2B/2C emit separate energy/stability grids for sampled local-energy, log-amplitude, and finite-fraction profiles against `r12`; directions and seeds are pooled so each subplot has one line per CoM and variance error bars aggregate all compact direction/seed records. Figure 2D emits separate energy/stability grids for `d_logabs_dr_median` against `r12`, with normalization rows, architecture columns, solid CoM model lines, and dashed target derivative references.",
            "",
            "## Tail Diagnostics",
            "",
            "Tail tables preserve path columns. Figures 3A/3B split energy and stability winners into subplot grids by architecture and normalization; each subplot draws CoM lines with seed-variance error bars for local energy and logabs, with a shared CoM legend. Exact log-amplitude references are included when collect inputs provide them.",
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
            "See `tables/trace_summary.csv` and the trace figures. Figures 7A, 7B, and 7C focus on feature-trace stability and emit one heatmap-grid figure each for `rms_q95`, `max_abs`, and `nonfinite_count`; each grid uses layer rows, energy/stability winner columns, architecture by normalization inside each subplot, and one shared color scale per metric.",
            "",
            "## Training And Resource Summary",
            "",
            "See `tables/training_curve_summary.csv` and `tables/resource_summary.csv`. Runtime is not mixed into quality ranking. Figures 8A and 8C show one smoothed training-energy curve per final-training run for energy and stability winners; Figures 8B and 8D show the corresponding semilogy absolute energy error curves with one shared vertical axis per grid.",
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
    print(f"[pair_stability] final report results_root={args.results_root}")
    if args.final_collect_attempt_id:
        print(f"[pair_stability] final report using final_collect_attempt_id={args.final_collect_attempt_id}")
    else:
        print("[pair_stability] final report using latest final-collect attempt")
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
    print("[pair_stability] final report copied table rows:")
    for filename, count in report["tables"].items():
        print(f"[pair_stability]   {filename}: {count}")
    print(f"[pair_stability] final report wrote {len(report['figures'])} figures")
    figure_counts: dict[str, int] = {}
    for figure in report["figures"]:
        section = figure.split("_", 1)[0]
        figure_counts[section] = figure_counts.get(section, 0) + 1
    for section, count in sorted(figure_counts.items()):
        print(f"[pair_stability]   figures {section}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
