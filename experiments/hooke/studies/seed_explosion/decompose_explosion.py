"""Decompose the local-energy explosion into Hamiltonian terms.

For the Hooke pair singlet,  E_L = T + V_trap + V_ee  where the trap and
electron-electron terms are fixed analytic functions of geometry (the probe
records them exactly). Any departure of E_L from 2.0 must therefore live in the
kinetic term T = -1/2 (lap logpsi + |grad logpsi|^2). This script confirms that
and prints the term breakdown at each seed's worst probe point. The visual
counterpart (E_L and log|psi| shape) is rendered by ``analyze_probes.py``.
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
SUSPECT = [105, 106, 107]
PROBE_OF = {105: "center_of_mass_probe", 106: "pair_distance_probe", 107: "center_of_mass_probe"}
SWEEPVAR = {"center_of_mass_probe": "center_of_mass_radius", "pair_distance_probe": "pair_distance"}


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


def worst_point(seed: int) -> None:
    probe = PROBE_OF[seed]
    p = load_probe(seed, probe)
    e_err = p["model_local_energy_error"]
    i = int(np.nanargmax(np.abs(e_err)))
    sweep = SWEEPVAR[probe]
    el = p["model_local_energy"][i]
    kin = p["kinetic_energy"][i]
    trap = p["harmonic_trap_energy"][i]
    ee = p["electron_electron_energy"][i]
    recon = kin + trap + ee
    print(f"seed {seed}  [{probe}]  worst at {sweep}={p[sweep][i]:.4f} dir={int(p['direction_id'][i])}")
    print(f"    model E_L      = {el:+.4f}   (exact 2.0, error {e_err[i]:+.4f})")
    print(f"    kinetic T      = {kin:+.4f}")
    print(f"    harmonic trap  = {trap:+.4f}   (analytic, geometry-fixed)")
    print(f"    e-e Coulomb    = {ee:+.4f}    (analytic, geometry-fixed)")
    print(f"    T+trap+ee      = {recon:+.4f}   (=> deviation lives in T: {kin - (2.0 - trap - ee):+.4f})")
    print(f"    model_logabs   = {p['model_logabs'][i]:+.4f}   exact_logabs = {p['exact_logabs'][i]:+.4f}")


def main() -> None:
    for seed in SUSPECT:
        worst_point(seed)
        print()


if __name__ == "__main__":
    main()
