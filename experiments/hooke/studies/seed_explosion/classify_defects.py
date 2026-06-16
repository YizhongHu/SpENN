"""Classify each suspect seed's defect: spurious node vs smooth tail-curvature.

A spurious node shows ``model_sign`` flipping and ``model_relative_abs_psi``
collapsing toward 0 along the sweep (the exact nodeless singlet never does this).
A smooth tail error keeps a constant sign and a monotone |psi| decay but a
diverging kinetic term. We scan all three probe directions of both sweeps for
each seed and report sign flips, the |psi| minimum, and the worst local energy.
"""

from __future__ import annotations

import csv
import math
import pathlib

import numpy as np

EVAL_ROOT = pathlib.Path(__file__).resolve().parents[1] / (
    "pair_validation/reports/05_final_eval/outputs/full/"
    "config_lr0-003_channels32_layers1_gate_activationsigmoid"
)
PAIRS = {ts: 100000 + (ts - 100) for ts in range(100, 110)}
SUSPECT = [105, 106, 107]


def load(train_seed: int, probe: str) -> dict[str, np.ndarray]:
    path = EVAL_ROOT / f"train_seed={train_seed}_eval_seed={PAIRS[train_seed]}" / "diagnostics" / probe / "probe.csv"
    cols: dict[str, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for name in reader.fieldnames or []:
            cols[name] = []
        for row in reader:
            for name in cols:
                try:
                    cols[name].append(float(row[name]))
                except (TypeError, ValueError):
                    cols[name].append(math.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def analyse(train_seed: int, probe: str, sweep: str) -> dict:
    p = load(train_seed, probe)
    n_sign_flips = 0
    min_abs_psi = math.inf
    worst_err = 0.0
    worst_at = math.nan
    for d in (0, 1, 2):
        m = p["direction_id"] == d
        order = np.argsort(p[sweep][m])
        sign = p["model_sign"][m][order]
        rel = p["model_relative_abs_psi"][m][order]
        err = p["model_local_energy_error"][m][order]
        x = p[sweep][m][order]
        n_sign_flips += int(np.sum(np.abs(np.diff(np.sign(sign))) > 0))
        min_abs_psi = min(min_abs_psi, float(np.nanmin(rel)))
        j = int(np.nanargmax(np.abs(err)))
        if abs(err[j]) > abs(worst_err):
            worst_err = float(err[j])
            worst_at = float(x[j])
    return {
        "probe": probe.replace("_probe", ""),
        "sign_flips": n_sign_flips,
        "min_rel_abs_psi": min_abs_psi,
        "worst_EL_err": worst_err,
        "worst_at": worst_at,
    }


def main() -> None:
    print("defect classification (sign_flips>0 + |psi|->0  =>  spurious node)\n")
    header = f"{'seed':>5}  {'probe':>14}  {'sign_flips':>10}  {'min|psi|/psi0':>14}  {'worst_EL_err':>13}  {'worst_at':>9}"
    print(header)
    print("-" * len(header))
    for seed in SUSPECT:
        for probe, sweep in (("pair_distance_probe", "pair_distance"),
                             ("center_of_mass_probe", "center_of_mass_radius")):
            r = analyse(seed, probe, sweep)
            print(f"{seed:>5}  {r['probe']:>14}  {r['sign_flips']:>10}  "
                  f"{r['min_rel_abs_psi']:>14.3e}  {r['worst_EL_err']:>+13.3f}  {r['worst_at']:>9.3f}")
        print()


if __name__ == "__main__":
    main()
