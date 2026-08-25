"""Deterministic, headless Matplotlib figures for He-v1 diagnostics.

The reporting stage prepares small, provenance-bearing row dictionaries.  This
module owns presentation only: an accessible palette, redundant markers and
line styles, stable metadata, units, and the three publication formats.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

COLOR = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "grey": "#777777",
}
CHECKPOINT_STYLE = {
    "step_025000": (COLOR["blue"], "o", "-"),
    "step_050000": (COLOR["orange"], "s", "--"),
}
REFERENCE_ENERGY_HA = -2.903724377034119598
FIGURE_FORMATS = ("svg", "pdf", "png")


def pyplot() -> Any:
    """Return pyplot with a deterministic headless configuration."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "tpen-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    matplotlib.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "figure.dpi": 120,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
            "savefig.bbox": "tight",
            "svg.hashsalt": "tpen-he-v1-diagnostic-report-v1",
        }
    )
    import matplotlib.pyplot as plt

    return plt


def save_triplet(fig: Any, output_dir: Path, stem: str) -> tuple[Path, ...]:
    """Save one figure as SVG, PDF, and 300-DPI PNG with stable metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_dir / f"{stem}.{suffix}" for suffix in FIGURE_FORMATS)
    fig.savefig(
        paths[0],
        format="svg",
        metadata={"Creator": "TPEN he-v1 diagnostic reporting", "Date": None},
    )
    fig.savefig(
        paths[1],
        format="pdf",
        metadata={
            "Creator": "TPEN he-v1 diagnostic reporting",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        paths[2],
        format="png",
        dpi=300,
        metadata={"Software": "TPEN he-v1 diagnostic reporting"},
    )
    return paths


def render_all(
    output_dir: Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[Path, ...]]:
    """Render every required diagnostic view in deterministic order."""

    renderers: tuple[
        tuple[str, Callable[[Sequence[Mapping[str, Any]]], Any]], ...
    ] = (
        ("energy_mcse", energy_mcse_figure),
        ("distribution_ccdf", distribution_ccdf_figure),
        ("conditioned_variance", conditioned_variance_figure),
        ("cusp_curvature", cusp_curvature_figure),
        ("singular_cancellation", singular_cancellation_figure),
        ("tails", tails_figure),
        ("symmetry_equivariance", symmetry_equivariance_figure),
        ("sampler_health_timing", sampler_health_timing_figure),
        ("factor_response", factor_response_figure),
    )
    plt = pyplot()
    rendered: dict[str, tuple[Path, ...]] = {}
    for stem, renderer in renderers:
        fig = renderer(tables.get(stem, ()))
        rendered[stem] = save_triplet(fig, output_dir, stem)
        plt.close(fig)
    return rendered


def energy_mcse_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot primary and diagnostic trajectory estimators without IID substitution."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    available = [row for row in rows if row.get("status") == "available"]
    primary = [row for row in available if row.get("estimator_role") == "primary"]
    diagnostic = [row for row in available if row.get("estimator_role") == "diagnostic"]
    _energy_panel(axes[0], primary, title="Primary 256×4096 estimates")
    _energy_panel(axes[1], diagnostic, title="Diagnostic protocol estimates")
    axes[0].axhline(
        REFERENCE_ENERGY_HA,
        color=COLOR["black"],
        linestyle=":",
        label="nonrelativistic reference",
    )
    axes[1].axhline(REFERENCE_ENERGY_HA, color=COLOR["black"], linestyle=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Trajectory energy estimator and correlation-aware MCSE")
    return fig


def _energy_panel(ax: Any, rows: Sequence[Mapping[str, Any]], *, title: str) -> None:
    if not rows:
        _no_data(ax, "estimator unavailable")
        ax.set_title(title)
        return
    labels = sorted({str(row["protocol"]) for row in rows})
    for checkpoint in sorted({str(row["checkpoint_label"]) for row in rows}):
        selected = [row for row in rows if row["checkpoint_label"] == checkpoint]
        color, marker, linestyle = _checkpoint_style(checkpoint)
        for index, protocol in enumerate(labels):
            values = [row for row in selected if row["protocol"] == protocol]
            for offset, row in enumerate(values):
                x = index + (offset - (len(values) - 1) / 2) * 0.06
                ax.errorbar(
                    x,
                    _float(row["trajectory_mean_ha"]),
                    yerr=_float(row["mcse_ha"]),
                    fmt=marker,
                    color=color,
                    linestyle="none",
                    capsize=2.5,
                    markerfacecolor="white" if title.startswith("Diagnostic") else color,
                    label=_checkpoint_label(checkpoint) if index == 0 and offset == 0 else None,
                )
        if len(labels) > 1:
            means = [
                np.mean(
                    [
                        _float(row["trajectory_mean_ha"])
                        for row in selected
                        if row["protocol"] == protocol
                    ]
                )
                for protocol in labels
                if any(row["protocol"] == protocol for row in selected)
            ]
            if len(means) == len(labels):
                ax.plot(range(len(labels)), means, color=color, linestyle=linestyle, alpha=0.55)
    ax.set_title(title)
    ax.set_ylabel("Energy (Ha)")
    ax.set_xticks(range(len(labels)), [_short_protocol(label) for label in labels], rotation=25, ha="right")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def distribution_ccdf_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot memory-mapped local-energy distributions and diagnostic CCDFs."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    histogram = [row for row in rows if row.get("view") == "histogram"]
    ccdf = [row for row in rows if row.get("view") == "ccdf"]
    for checkpoint in sorted({str(row["checkpoint_label"]) for row in histogram}):
        chosen = [row for row in histogram if row["checkpoint_label"] == checkpoint]
        chosen.sort(key=lambda row: _float(row["bin_left_transformed"]))
        color, marker, linestyle = _checkpoint_style(checkpoint)
        x = [0.5 * (_float(row["bin_left_transformed"]) + _float(row["bin_right_transformed"])) for row in chosen]
        y = [_float(row["probability"]) for row in chosen]
        axes[0].plot(x, y, color=color, linestyle=linestyle, marker=marker, markevery=max(1, len(x) // 10), label=_checkpoint_label(checkpoint))
    for checkpoint in sorted({str(row["checkpoint_label"]) for row in ccdf}):
        chosen = [row for row in ccdf if row["checkpoint_label"] == checkpoint]
        chosen.sort(key=lambda row: _float(row["threshold_ha"]))
        color, marker, linestyle = _checkpoint_style(checkpoint)
        axes[1].plot(
            [_float(row["threshold_ha"]) for row in chosen],
            [_float(row["probability"]) for row in chosen],
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=_checkpoint_label(checkpoint),
        )
    axes[0].set_title("Complete retained-record distribution")
    axes[0].set_xlabel(r"signed $\log_{10}(1+|E_L-E_{ref}|/\mathrm{Ha})$")
    axes[0].set_ylabel("Probability per bin")
    axes[1].set_title("Absolute-deviation CCDF (diagnostic)")
    axes[1].set_xlabel(r"$|E_L-\bar E_L|$ threshold (Ha)")
    axes[1].set_ylabel("Probability")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    if histogram:
        axes[0].legend(fontsize=8)
    if not histogram:
        _no_data(axes[0], "record distribution unavailable")
    if not ccdf:
        _no_data(axes[1], "CCDF unavailable")
    fig.suptitle("Local-energy distribution and tail probability")
    return fig


def conditioned_variance_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot diagnostic variance attribution by predeclared geometry bins."""

    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), constrained_layout=True)
    quantities = (
        "minimum_electron_nuclear_radius",
        "electron_electron_distance",
        "maximum_electron_nuclear_radius",
        "hyperradius",
        "cos_theta12",
        "logabs",
    )
    titles = (
        r"minimum $r_{iA}$",
        r"electron distance $r_{12}$",
        r"maximum $r_{iA}$",
        "hyperradius",
        r"angular coordinate $\cos\theta_{12}$",
        r"$\log|\Psi|$",
    )
    for ax, quantity, title in zip(axes.flat, quantities, titles, strict=True):
        subset = [row for row in rows if row.get("quantity") == quantity and row.get("bin_kind") == "finite"]
        labels = list(dict.fromkeys(str(row["bin_label"]) for row in subset))
        for checkpoint in sorted({str(row["checkpoint_label"]) for row in subset}):
            chosen = [row for row in subset if row["checkpoint_label"] == checkpoint]
            by_label = {str(row["bin_label"]): row for row in chosen}
            color, marker, linestyle = _checkpoint_style(checkpoint)
            ax.plot(
                range(len(labels)),
                [_float(by_label[label]["second_moment_contribution_ha2"]) for label in labels],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=_checkpoint_label(checkpoint),
            )
        ax.set_title(title)
        ax.set_ylabel(r"Contribution to Var($E_L$) (Ha$^2$)")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        if not subset:
            _no_data(ax, "conditioned metric unavailable")
    if rows:
        axes.flat[0].legend(fontsize=8)
    fig.suptitle("Conditioned variance attribution (diagnostic, not headline energy)")
    return fig


def cusp_curvature_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot executed cusp derivatives separately from ideal laws and curvature."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    panels = (
        ("electron_nucleus", "first_derivative", "Electron–nucleus cusp"),
        ("electron_electron", "first_derivative", "Electron–electron cusp"),
        ("curvature", "second_derivative", "Direct executed curvature"),
    )
    for ax, (view, value_key, title) in zip(axes, panels, strict=True):
        subset = [
            row
            for row in rows
            if row.get("view") == view
            and str(row.get("series", "")).startswith("executed_")
            and row.get("available") is True
        ]
        for checkpoint in sorted({str(row["checkpoint_label"]) for row in subset}):
            chosen = [row for row in subset if row["checkpoint_label"] == checkpoint]
            chosen.sort(key=lambda row: _float(row["radius_bohr"]))
            color, marker, linestyle = _checkpoint_style(checkpoint)
            ax.plot(
                [_float(row["radius_bohr"]) for row in chosen],
                [_float(row[value_key]) for row in chosen],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markevery=max(1, len(chosen) // 8),
                label=f"executed {_checkpoint_label(checkpoint)}",
            )
        analytic_values = {
            _float(row[value_key])
            for row in rows
            if row.get("view") == view
            and row.get("series") == "analytic_ideal_cusp_law"
        }
        if analytic_values:
            ideal = next(iter(analytic_values))
            ax.axhline(
                ideal,
                color=COLOR["black"],
                linestyle=":",
                label=f"ideal cusp law ({ideal:g})",
            )
        elif view == "curvature":
            ax.text(
                0.04,
                0.04,
                "No universal Kato curvature target",
                transform=ax.transAxes,
                fontsize=7.5,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Realized radius (bohr)")
        ax.set_ylabel("First derivative (bohr⁻¹)" if value_key == "first_derivative" else "Second derivative (bohr⁻²)")
        ax.set_title(title)
        if not subset:
            _no_data(ax, "executed metric unavailable")
        if subset or analytic_values:
            ax.legend(fontsize=7)
    fig.suptitle("Executed cusp response, ideal law, and curvature are distinct")
    return fig


def singular_cancellation_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot singular-term cancellation magnitude and residual."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for checkpoint in sorted({str(row["checkpoint_label"]) for row in rows}):
        chosen = [row for row in rows if row["checkpoint_label"] == checkpoint and row.get("available") is True]
        chosen.sort(key=lambda row: _float(row["radius_bohr"]))
        color, marker, linestyle = _checkpoint_style(checkpoint)
        x = [_float(row["radius_bohr"]) for row in chosen]
        axes[0].plot(x, [_float(row["cancellation_abs_sum_ha"]) for row in chosen], color=color, marker=marker, linestyle=linestyle, markevery=max(1, len(chosen) // 8), label=_checkpoint_label(checkpoint))
        axes[1].plot(x, [abs(_float(row["cancellation_residual_ha"])) for row in chosen], color=color, marker=marker, linestyle=linestyle, markevery=max(1, len(chosen) // 8), label=_checkpoint_label(checkpoint))
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Realized coalescence radius (bohr)")
    axes[0].set_title("Absolute Hamiltonian term sum")
    axes[0].set_ylabel("Absolute term sum (Ha)")
    axes[1].set_title("Cancellation residual")
    axes[1].set_ylabel("|residual| (Ha)")
    if rows:
        axes[0].legend(fontsize=8)
    else:
        _no_data(axes[0], "cancellation unavailable")
        _no_data(axes[1], "cancellation unavailable")
    fig.suptitle("Singular cancellation in the executed Hamiltonian")
    return fig


def tails_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot one-electron and centre-of-mass executed log-amplitude tails."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    panels = (("one_electron", "One-electron escape"), ("center_of_mass", "Centre-of-mass escape"))
    for ax, (view, title) in zip(axes, panels, strict=True):
        subset = [row for row in rows if row.get("view") == view and row.get("available") is True]
        for checkpoint in sorted({str(row["checkpoint_label"]) for row in subset}):
            chosen = [row for row in subset if row["checkpoint_label"] == checkpoint]
            chosen.sort(key=lambda row: _float(row["radius_bohr"]))
            color, marker, linestyle = _checkpoint_style(checkpoint)
            ax.plot(
                [_float(row["radius_bohr"]) for row in chosen],
                [_float(row["executed_logabs"]) for row in chosen],
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=f"{_checkpoint_label(checkpoint)}; outer slope={row_value(chosen, 'outer_slope_bohr_inv')}",
            )
        ax.set_title(title)
        ax.set_xlabel("Escape radius (bohr)")
        ax.set_ylabel(r"executed $\log|\Psi|$")
        if not subset:
            _no_data(ax, "tail metric unavailable")
    if rows:
        axes[0].legend(fontsize=7)
    fig.suptitle("Executed outer-tail response")
    return fig


def symmetry_equivariance_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot transform and trace invariant errors using worst-case aggregation."""

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9.2, 4.0), constrained_layout=True)
    labels = list(dict.fromkeys(str(row["metric_label"]) for row in rows))
    for checkpoint in sorted({str(row["checkpoint_label"]) for row in rows}):
        chosen = {str(row["metric_label"]): row for row in rows if row["checkpoint_label"] == checkpoint}
        color, marker, _ = _checkpoint_style(checkpoint)
        values = [max(_float(chosen[label]["value"]), 1.0e-18) if label in chosen and chosen[label].get("available") is True else np.nan for label in labels]
        offsets = -0.12 if checkpoint == "step_025000" else 0.12
        ax.scatter(np.arange(len(labels)) + offsets, values, color=color, marker=marker, label=_checkpoint_label(checkpoint), zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Worst-case error or count (log scale)")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_title("Antisymmetry, spatial exchange, rotation, and trace equivariance")
    if rows:
        ax.legend(fontsize=8)
    else:
        _no_data(ax, "invariant metrics unavailable")
    return fig


def sampler_health_timing_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot sampler health separately from evaluation time and ESS throughput."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    primary = [row for row in rows if row.get("estimator_role") == "primary"]
    panels = (
        (axes[0], "acceptance_rate", "Retained acceptance", "fraction"),
        (axes[1], "wall_time_sec", "Evaluation wall time", "seconds"),
        (axes[2], "ess_per_second", "ESS throughput", "ESS s⁻¹"),
    )
    for ax, key, title, ylabel in panels:
        labels = sorted({str(row["row_id"]) for row in primary})
        for checkpoint in sorted({str(row["checkpoint_label"]) for row in primary}):
            chosen = [row for row in primary if row["checkpoint_label"] == checkpoint and row.get(f"{key}_available") is True]
            color, marker, _ = _checkpoint_style(checkpoint)
            ax.scatter(
                [labels.index(str(row["row_id"])) for row in chosen],
                [_float(row[key]) for row in chosen],
                color=color,
                marker=marker,
                label=_checkpoint_label(checkpoint),
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        if not primary:
            _no_data(ax, "metric unavailable")
    if primary:
        axes[0].legend(fontsize=8)
    fig.suptitle("Sampler health and cost (availability preserved)")
    return fig


def factor_response_figure(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Plot paired fixed-configuration and independent re-equilibrated response."""

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    panels = (
        ("fixed_configuration_paired", "Fixed configurations (paired response)"),
        ("re_equilibrated_independent", "Re-equilibrated chains (independent estimates)"),
    )
    for ax, (basis, title) in zip(axes, panels, strict=True):
        subset = [row for row in rows if row.get("comparison_basis") == basis and row.get("status") == "available"]
        labels = list(dict.fromkeys(str(row["arm_label"]) for row in subset))
        for checkpoint in sorted({str(row["checkpoint_label"]) for row in subset}):
            chosen = {str(row["arm_label"]): row for row in subset if row["checkpoint_label"] == checkpoint}
            color, marker, linestyle = _checkpoint_style(checkpoint)
            values = [_float(chosen[label]["delta_energy_ha"]) for label in labels]
            ax.errorbar(
                range(len(labels)),
                values,
                yerr=[_float(chosen[label]["delta_uncertainty_ha"]) for label in labels],
                color=color,
                marker=marker,
                linestyle=linestyle,
                capsize=2.5,
                label=_checkpoint_label(checkpoint),
            )
        ax.axhline(0.0, color=COLOR["black"], linestyle=":")
        ax.set_title(title)
        ax.set_ylabel(r"$\Delta E_L$ from baseline (Ha)")
        ax.set_xticks(range(len(labels)), [_short_factor(label) for label in labels], rotation=35, ha="right")
        if not subset:
            _no_data(ax, "factor metric unavailable")
    if rows:
        axes[0].legend(fontsize=8)
    fig.suptitle("Factor response: comparison semantics are not interchangeable")
    return fig


def _checkpoint_style(checkpoint: str) -> tuple[str, str, str]:
    return CHECKPOINT_STYLE.get(checkpoint, (COLOR["grey"], "D", "-."))


def _checkpoint_label(checkpoint: str) -> str:
    return checkpoint.replace("step_", "") + " updates"


def _short_protocol(label: str) -> str:
    return label.replace("primary_", "").replace("long_", "long ").replace("_", " ")


def _short_factor(label: str) -> str:
    return label.replace("_minus_10pct", " −10%").replace("_plus_10pct", " +10%").replace("_", " ")


def _float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"plot value must be finite, got {value!r}")
    return number


def row_value(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    """Return one compact value label from a non-empty plotted row set."""

    values = [row.get(key) for row in rows if row.get(key) not in (None, "")]
    return "unavailable" if not values else f"{_float(values[0]):.3g}"


def _no_data(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=COLOR["grey"])


__all__ = [
    "FIGURE_FORMATS",
    "REFERENCE_ENERGY_HA",
    "render_all",
    "save_triplet",
]
