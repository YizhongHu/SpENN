"""Probe final-eval metrics across train seeds to locate the explosion.

For each ``train_seed`` of the selected config, reads the eval ``metrics.csv``
(long format) and reports the headline energy, its error vs the reference, the
variance, kinetic-term magnitude, and local-energy tail/finite statistics.
"""

from __future__ import annotations

import csv
import math
import pathlib

EVAL_ROOT = pathlib.Path(__file__).resolve().parents[1] / (
    "pair_validation/reports/05_final_eval/outputs/full/"
    "config_lr0-003_channels32_layers1_gate_activationsigmoid"
)

# train_seed -> eval_seed pairing from the manifest rows.
PAIRS = [(100 + i, 100000 + i) for i in range(10)]

WANTED = [
    ("eval", "energy"),
    ("eval", "energy_error"),
    ("eval", "energy_std"),
    ("eval", "energy_variance"),
    ("eval", "reference_energy"),
    ("eval", "energy_term_kinetic"),
    ("eval", "energy_term_kinetic_variance"),
    ("eval", "energy_term_electron_electron"),
    ("eval", "energy_term_harmonic_trap"),
    ("eval", "local_energy_finite_fraction"),
    ("eval", "local_energy_nonfinite_count"),
    ("eval", "local_energy_min"),
    ("eval", "local_energy_max"),
    ("eval", "local_energy_q001"),
    ("eval", "local_energy_q999"),
    ("eval/sampler", "acceptance_rate"),
    ("eval/sampler", "radius_q99"),
    ("eval/sampler", "radius_max"),
    ("eval/sampler", "electron_distance_q01"),
    ("eval/sampler", "electron_distance_min"),
]


def load_eval(train_seed: int, eval_seed: int) -> dict[tuple[str, str], float]:
    path = EVAL_ROOT / f"train_seed={train_seed}_eval_seed={eval_seed}" / "metrics.csv"
    wanted = set(WANTED)
    out: dict[tuple[str, str], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["namespace"], row["key"])
            if key not in wanted:
                continue
            try:
                out[key] = float(row["value"])
            except (TypeError, ValueError):
                out[key] = math.nan
    return out


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.5f}"


def main() -> None:
    short = {
        ("eval", "energy"): "E",
        ("eval", "energy_error"): "E_err",
        ("eval", "energy_std"): "E_std",
        ("eval", "energy_variance"): "E_var",
        ("eval", "reference_energy"): "E_ref",
        ("eval", "energy_term_kinetic"): "kin",
        ("eval", "energy_term_kinetic_variance"): "kin_var",
        ("eval", "energy_term_electron_electron"): "ee",
        ("eval", "energy_term_harmonic_trap"): "trap",
        ("eval", "local_energy_finite_fraction"): "Lfin_frac",
        ("eval", "local_energy_nonfinite_count"): "Lnonfin",
        ("eval", "local_energy_min"): "Lmin",
        ("eval", "local_energy_max"): "Lmax",
        ("eval", "local_energy_q001"): "Lq001",
        ("eval", "local_energy_q999"): "Lq999",
        ("eval/sampler", "acceptance_rate"): "accept",
        ("eval/sampler", "radius_q99"): "r_q99",
        ("eval/sampler", "radius_max"): "r_max",
        ("eval/sampler", "electron_distance_q01"): "d_q01",
        ("eval/sampler", "electron_distance_min"): "d_min",
    }
    rows = []
    for ts, es in PAIRS:
        data = load_eval(ts, es)
        row = {"train_seed": ts}
        for key in WANTED:
            row[short[key]] = data.get(key, math.nan)
        rows.append(row)

    cols = ["train_seed"] + [short[k] for k in WANTED]
    cell = lambda r, c: str(r[c]) if c == "train_seed" else _fmt(r[c])
    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(cell(r, c).rjust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
