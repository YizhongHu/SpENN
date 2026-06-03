# Hooke Multibody Report

## Purpose

Test whether the SpENN-QMC stack can run a genuinely multielectron Hooke trap
with VMC energy minimization only. The current scaffold targets `N=3`; it is a
VMC-only SpENN launch path, not a supervised exact-reference benchmark.

## Hamiltonian

The system is an `N`-electron harmonic Coulomb trap:

```text
H = sum_i (-1/2 nabla_i^2 + 1/2 omega^2 |r_i|^2) + sum_{i<j} 1 / |r_i-r_j|
```

## Model

The trial form is `psi_theta(R) = exp(J_ee(R)) psi_SpENN(R)`. The configured
cusp slopes are `1/4` for same-spin pairs and `1/2` for opposite-spin pairs.
The initial readout is Pfaffian-based and uses particle-token antisymmetry.
Here a particle token contains both position and spin label, so antisymmetry
checks permute the two together.

## Reference

No numeric multibody reference is configured yet. SpENN is not trained against
any exact or reference wavefunction. Energy tables should be interpreted as VMC
estimates until an independent reference pipeline is added.

## Spin Scan

`run_spenn.py --config benchmark --scan-spins` runs fixed spin sectors from
`configs/benchmark.yaml`, currently `(3, 0)`, `(2, 1)`, `(1, 2)`, and `(0, 3)`.
It trains each sector separately with VMC and records the lowest sampled VMC
energy as the scan best candidate. The scan does not optimize or mix spin
sectors during one run.

## Data and Plots

Run directories are written under `outputs/YYYY-MM-DD/`. Generated configs and
summaries record `run.time` in `HH-MM-SS` format, and generated run ids include
that timestamp. `process_outputs.py` writes processed CSV/JSON under the saved
run directory, while `plot_outputs.py` reads saved metrics and plot CSVs and
writes PNGs under `experiments/hooke_multibody/figures/spenn/`.

## Slurm

`slurm/cpu_smoke.job` runs the multibody integration smoke test with the `cpu`
uv extra. `slurm/gpu_smoke.job` uses `.venv-gpu`, the `cu126` uv extra, checks
CUDA, and runs the smoke config on `device=cuda`.

As of 2026-06-03, `sbatch --test-only` for both Slurm scripts failed with
`allocation failure: Unable to contact slurm controller (connect failure)`.
The scripts are present, but no controller-backed Slurm smoke job was accepted
from this checkout.

## Known Limitations

The scaffold is intentionally small. Pfaffian evaluation, local-energy second
derivatives, and Specht tuple tensors become expensive quickly as `N`, channel
counts, or Specht order increase.
