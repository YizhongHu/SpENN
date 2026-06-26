"""Quick probe: load final_train metrics for all seeds and summarize trajectories.

Reads the long-format ``metrics.csv`` (columns: step, namespace, key, value) for
each seed of the selected final-train config and reports where ``train|energy``,
``train|loss``, ``train|grad_norm``, and ``train|param_norm`` blow up.

Uses only stdlib ``csv`` + ``numpy`` (the repo does not depend on pandas).
"""

from __future__ import annotations

import csv
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / (
    "pair_validation/reports/04_final_train/outputs/full/"
    "config_lr0-003_channels32_layers1_gate_activationsigmoid"
)

SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
KEYS = [
    ("train", "energy"),
    ("train", "loss"),
    ("train", "grad_norm"),
    ("train", "param_norm"),
    ("train", "energy_variance"),
    ("train", "local_energy_nonfinite_count"),
]


def _to_float(text: str) -> float:
    """Parse a metric value, mapping inf/nan strings to numpy floats."""

    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def load_series(seed: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Return ``(namespace, key) -> (steps, values)`` for the requested KEYS."""

    wanted = set(KEYS)
    buckets: dict[tuple[str, str], list[tuple[int, float]]] = {k: [] for k in KEYS}
    with (ROOT / f"seed={seed}" / "metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["namespace"], row["key"])
            if key not in wanted:
                continue
            buckets[key].append((int(row["step"]), _to_float(row["value"])))
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, pairs in buckets.items():
        pairs.sort(key=lambda p: p[0])
        steps = np.array([p[0] for p in pairs], dtype=int)
        values = np.array([p[1] for p in pairs], dtype=float)
        out[key] = (steps, values)
    return out


def summarize(seed: int) -> dict:
    series = load_series(seed)
    steps, energy = series[("train", "energy")]
    _, loss = series[("train", "loss")]
    _, grad = series[("train", "grad_norm")]
    _, pnorm = series[("train", "param_norm")]
    _, evar = series[("train", "energy_variance")]

    nonfinite_mask = ~np.isfinite(energy)
    first_nonfinite = int(steps[nonfinite_mask][0]) if nonfinite_mask.any() else None

    # First step where |energy| exceeds a wild threshold (pre-NaN blowup).
    blow_thresh = 1e3
    blow_mask = np.isfinite(energy) & (np.abs(energy) > blow_thresh)
    first_blow = int(steps[blow_mask][0]) if blow_mask.any() else None

    finite_energy = energy[np.isfinite(energy)]
    return {
        "seed": seed,
        "n_steps": len(steps),
        "E_first": energy[0] if len(energy) else math.nan,
        "E_last": energy[-1] if len(energy) else math.nan,
        "E_min": float(np.min(finite_energy)) if finite_energy.size else math.nan,
        "E_max": float(np.max(finite_energy)) if finite_energy.size else math.nan,
        "E_nonfinite": int(nonfinite_mask.sum()),
        "first_nonfin_step": first_nonfinite,
        f"first_|E|>{blow_thresh:g}_step": first_blow,
        "loss_absmax": float(np.nanmax(np.abs(loss))) if loss.size else math.nan,
        "grad_max": float(np.nanmax(grad)) if grad.size else math.nan,
        "evar_max": float(np.nanmax(evar)) if evar.size else math.nan,
        "pnorm_first": pnorm[0] if pnorm.size else math.nan,
        "pnorm_last": pnorm[-1] if pnorm.size else math.nan,
        "pnorm_max": float(np.nanmax(pnorm)) if pnorm.size else math.nan,
    }


def main() -> None:
    rows = [summarize(s) for s in SEEDS]
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r[c]).rjust(widths[c]) for c in cols))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
