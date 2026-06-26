"""Localize the local-energy explosion across final-train seeds.

The Hooke pair singlet has a flat exact local energy E_L(R) = 2.0 everywhere.
Each final-eval run ships two deterministic geometry sweeps:

* ``pair_distance_probe`` -- vary the inter-electron distance r12 (1e-4 -> 8)
  with the pair centred, three orientation directions.
* ``center_of_mass_probe`` -- vary the centre-of-mass radius (0 -> 8) at fixed
  r12 = 1, three directions.

For a correct wavefunction the model local energy should hug 2.0 along both
sweeps. Where it departs tells us *what* explodes: a residual near r12 -> 0 is a
cusp violation; a runaway at large r12 or large COM radius is a tail-curvature
failure (kinetic term no longer cancels the harmonic trap).

This script reports, per seed, where and how large the worst departures are and
splits the error into a cusp band and a tail band so the three suspect seeds can
be compared against the healthy ones. It also writes comparison plots.
"""

from __future__ import annotations

import csv
import math
import pathlib

import numpy as np

STUDY_DIR = pathlib.Path(__file__).resolve().parent
EVAL_ROOT = STUDY_DIR.parents[0] / (
    "pair_validation/reports/05_final_eval/outputs/full/"
    "config_lr0-003_channels32_layers1_gate_activationsigmoid"
)
PLOT_DIR = STUDY_DIR / "plots"

SEEDS = list(range(100, 110))
PAIRS = {ts: 100000 + (ts - 100) for ts in SEEDS}
SUSPECT = {105, 106, 107}

EXACT_E_L = 2.0


def load_probe(train_seed: int, probe: str) -> dict[str, np.ndarray]:
    path = EVAL_ROOT / f"train_seed={train_seed}_eval_seed={PAIRS[train_seed]}" / "diagnostics" / probe / "probe.csv"
    cols: dict[str, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        for name in names:
            cols[name] = []
        for row in reader:
            for name in names:
                val = row[name]
                try:
                    cols[name].append(float(val))
                except (TypeError, ValueError):
                    cols[name].append(math.nan)  # e.g. boolean/str -> nan
    return {name: np.asarray(values, dtype=float) for name, values in cols.items()}


def summarize_pair(train_seed: int) -> dict:
    p = load_probe(train_seed, "pair_distance_probe")
    r = p["pair_distance"]
    e_err = p["model_local_energy_error"]  # model_local_energy - 2.0
    e_loc = p["model_local_energy"]
    # Average the three directions onto the shared r grid for banding stats.
    cusp = r < 0.5
    mid = (r >= 0.5) & (r <= 3.0)
    tail = r > 3.0
    return {
        "seed": train_seed,
        "pair_abs_err_max": float(np.nanmax(np.abs(e_err))),
        "pair_err_at_rmax": float(e_err[np.argmax(r)]),
        "pair_cusp_abs_err_max": float(np.nanmax(np.abs(e_err[cusp]))) if cusp.any() else math.nan,
        "pair_mid_abs_err_max": float(np.nanmax(np.abs(e_err[mid]))) if mid.any() else math.nan,
        "pair_tail_abs_err_max": float(np.nanmax(np.abs(e_err[tail]))) if tail.any() else math.nan,
        "pair_EL_min": float(np.nanmin(e_loc)),
        "pair_EL_max": float(np.nanmax(e_loc)),
        "pair_r_at_worst": float(r[np.nanargmax(np.abs(e_err))]),
    }


def summarize_com(train_seed: int) -> dict:
    p = load_probe(train_seed, "center_of_mass_probe")
    rc = p["center_of_mass_radius"]
    e_err = p["model_local_energy_error"]
    e_loc = p["model_local_energy"]
    tail = rc > 3.0
    return {
        "com_abs_err_max": float(np.nanmax(np.abs(e_err))),
        "com_err_at_rmax": float(e_err[np.argmax(rc)]),
        "com_tail_abs_err_max": float(np.nanmax(np.abs(e_err[tail]))) if tail.any() else math.nan,
        "com_EL_min": float(np.nanmin(e_loc)),
        "com_EL_max": float(np.nanmax(e_loc)),
        "com_rc_at_worst": float(rc[np.nanargmax(np.abs(e_err))]),
    }


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.4f}"


def print_table(rows: list[dict], cols: list[str]) -> None:
    def cell(r, c):
        return _fmt(r.get(c))
    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        mark = " <-" if r["seed"] in SUSPECT else ""
        print("  ".join(cell(r, c).rjust(widths[c]) for c in cols) + mark)


def _grid(p: dict[str, np.ndarray], sweep: str) -> np.ndarray:
    """Sorted unique sweep values (the shared radius grid)."""

    return np.unique(p[sweep])


def _envelope(p: dict[str, np.ndarray], sweep: str) -> tuple[np.ndarray, np.ndarray]:
    """Max |E_L - 2| over all directions/offsets at each sweep radius."""

    xs = _grid(p, sweep)
    err = np.abs(p["model_local_energy_error"])
    worst = np.array([np.nanmax(err[np.isclose(p[sweep], x)]) for x in xs])
    return xs, worst


def _worst_slice(p: dict[str, np.ndarray], sweep: str, secondary: str) -> tuple[np.ndarray, np.ndarray]:
    """1D slice (fixed direction + secondary value) passing through the worst point."""

    j = int(np.nanargmax(np.abs(p["model_local_energy_error"])))
    mask = np.isclose(p["direction_id"], p["direction_id"][j]) & np.isclose(p[secondary], p[secondary][j])
    order = np.argsort(p[sweep][mask])
    return mask, order


def make_plots(rows_pair: dict[int, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: explosion envelope -- worst |E_L - 2| over all dirs/offsets.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for train_seed in SEEDS:
        p = load_probe(train_seed, "pair_distance_probe")
        xs, env = _envelope(p, "pair_distance")
        style = dict(lw=2.6, zorder=3) if train_seed in SUSPECT else dict(lw=1.0, alpha=0.55)
        axes[0].semilogy(xs, env + 1e-3, label=f"seed {train_seed}", **style)
    axes[0].set(xlabel="pair distance r12", ylabel="max |E_L - 2|  (log)",
                title="Explosion envelope vs pair distance", xlim=(0, 8))
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3, which="both")

    for train_seed in SEEDS:
        p = load_probe(train_seed, "center_of_mass_probe")
        xs, env = _envelope(p, "center_of_mass_radius")
        style = dict(lw=2.6, zorder=3) if train_seed in SUSPECT else dict(lw=1.0, alpha=0.55)
        axes[1].semilogy(xs, env + 1e-3, label=f"seed {train_seed}", **style)
    axes[1].set(xlabel="center-of-mass radius", ylabel="max |E_L - 2|  (log)",
                title="Explosion envelope vs COM radius", xlim=(0, 8))
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Hooke pair singlet: worst local-energy departure from exact 2.0 (suspect seeds bold)")
    fig.tight_layout()
    out = PLOT_DIR / "local_energy_sweeps.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")

    # --- Figure 2: per-seed defect signature (E_L, log|psi|, sign) on its bad sweep.
    panels = [
        (105, "center_of_mass_probe", "center_of_mass_radius", "COM-coordinate node (sign flip)"),
        (106, "pair_distance_probe", "pair_distance", "relative-coord near-nodes (tail)"),
        (107, "center_of_mass_probe", "center_of_mass_radius", "smooth COM tail curvature"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for col, (seed, probe, sweep, caption) in enumerate(panels):
        p = load_probe(seed, probe)
        secondary = "center_of_mass_radius" if "pair" in probe else "pair_distance"
        mask, order = _worst_slice(p, sweep, secondary)
        x = p[sweep][mask][order]
        el = p["model_local_energy"][mask][order]
        mlog = p["model_logabs"][mask][order]
        elog = p["exact_logabs"][mask][order]
        sign = p["model_sign"][mask][order]

        ax = axes[0, col]
        ax.plot(x, el, "k", lw=2)
        ax.axhline(2.0, color="gray", ls="--", lw=1)
        ax.set(title=f"seed {seed}: {caption}", xlabel=sweep, ylabel="model E_L")
        ax.grid(alpha=0.3)

        ax2 = axes[1, col]
        ax2.plot(x, mlog, "k", lw=2, label="model log|psi|")
        ax2.plot(x, elog, "m--", lw=1.3, label="exact log|psi|")
        flips = x[1:][np.abs(np.diff(np.sign(sign))) > 0]
        for xf in flips:
            ax2.axvline(xf, color="r", ls=":", lw=1)
        ax2.set(xlabel=sweep, ylabel="log|psi|")
        ax2.legend(fontsize=7, title="red dotted = sign flip (node)")
        ax2.grid(alpha=0.3)

    fig.suptitle("Per-seed wavefunction defect signatures (clean dir-0 slice)")
    fig.tight_layout()
    out2 = PLOT_DIR / "defect_signatures.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"wrote {out2}")


def main() -> None:
    rows = []
    for s in SEEDS:
        row = summarize_pair(s)
        row.update(summarize_com(s))
        rows.append(row)

    print("PAIR-DISTANCE PROBE (exact E_L = 2.0 everywhere):")
    print_table(
        rows,
        ["seed", "pair_abs_err_max", "pair_r_at_worst", "pair_cusp_abs_err_max",
         "pair_mid_abs_err_max", "pair_tail_abs_err_max", "pair_EL_min", "pair_EL_max"],
    )
    print("\nCENTER-OF-MASS PROBE:")
    print_table(
        rows,
        ["seed", "com_abs_err_max", "com_rc_at_worst", "com_tail_abs_err_max",
         "com_EL_min", "com_EL_max"],
    )

    make_plots({r["seed"]: r for r in rows})


if __name__ == "__main__":
    main()
