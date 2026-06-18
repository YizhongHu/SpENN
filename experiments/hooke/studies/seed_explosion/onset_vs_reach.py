"""Bridge the latent tail defect to the observed Monte-Carlo variance.

The deterministic probe scans out to radius 8, far past where the Metropolis
sampler actually goes. To show the probe defect is what inflates the *sampled*
energy variance, we compute, per seed, the smallest sweep radius at which the
local-energy error first exceeds 0.5 (the "onset" of the tail explosion) and
compare it to the sampler's q99 / max electron radius recorded at eval time.
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

SEEDS = list(range(100, 110))
PAIRS = {ts: 100000 + (ts - 100) for ts in SEEDS}
SUSPECT = {105, 106, 107}
ONSET = 0.5  # |E_L - 2| threshold marking the start of the tail blowup


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
                try:
                    cols[name].append(float(row[name]))
                except (TypeError, ValueError):
                    cols[name].append(math.nan)
    return {name: np.asarray(vals, dtype=float) for name, vals in cols.items()}


def load_sampler_stats(train_seed: int) -> dict[str, float]:
    path = EVAL_ROOT / f"train_seed={train_seed}_eval_seed={PAIRS[train_seed]}" / "metrics.csv"
    wanted = {
        ("eval/sampler", "radius_q99"),
        ("eval/sampler", "radius_max"),
        ("eval/sampler", "center_of_mass_rms"),
        ("eval", "energy_variance"),
        ("eval", "local_energy_min"),
    }
    out: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["namespace"], row["key"])
            if key in wanted:
                try:
                    out[row["key"]] = float(row["value"])
                except (TypeError, ValueError):
                    out[row["key"]] = math.nan
    return out


def onset_radius(seed: int, probe: str, sweep: str) -> float:
    p = load_probe(seed, probe)
    err = np.abs(p["model_local_energy_error"])
    x = p[sweep]
    mask = err > ONSET
    return float(np.min(x[mask])) if mask.any() else math.inf


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "nan"
    if math.isinf(v):
        return "none"
    if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.4f}"


def main() -> None:
    cols = [
        "seed", "pair_onset_r12", "com_onset_rcom", "samp_radius_q99",
        "samp_radius_max", "samp_com_rms", "E_var", "L_min",
    ]
    rows = []
    for s in SEEDS:
        stats = load_sampler_stats(s)
        rows.append({
            "seed": s,
            "pair_onset_r12": onset_radius(s, "pair_distance_probe", "pair_distance"),
            "com_onset_rcom": onset_radius(s, "center_of_mass_probe", "center_of_mass_radius"),
            "samp_radius_q99": stats.get("radius_q99", math.nan),
            "samp_radius_max": stats.get("radius_max", math.nan),
            "samp_com_rms": stats.get("center_of_mass_rms", math.nan),
            "E_var": stats.get("energy_variance", math.nan),
            "L_min": stats.get("local_energy_min", math.nan),
        })

    def cell(r, c):
        return str(r[c]) if c == "seed" else _fmt(r[c])

    widths = {c: max(len(c), *(len(cell(r, c)) for r in rows)) for c in cols}
    print("onset = smallest probe radius where |E_L - 2| > 0.5; sampler reach for comparison")
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        mark = " <-" if r["seed"] in SUSPECT else ""
        print("  ".join(cell(r, c).rjust(widths[c]) for c in cols) + mark)


if __name__ == "__main__":
    main()
