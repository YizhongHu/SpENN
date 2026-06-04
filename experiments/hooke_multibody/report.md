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

No numeric high-accuracy multibody reference is configured yet. SpENN is not
trained against any exact or reference wavefunction. Energy tables should be
interpreted as VMC estimates until an independent high-accuracy reference
pipeline is added.

`configs/reference.yaml` and `run_reference.py` currently write a deterministic
Gaussian Hartree variational baseline with closed-form energy and density
tables. This keeps the run/comparison interface reproducible without claiming a
high-accuracy `N=3` reference that is not yet implemented. The baseline leaves
`reference_available=false` and records comparison quantities in separate
baseline columns.

## Configuration Snapshot

The smoke and benchmark runs inherit from `configs/spenn.yaml`. The default
physical system is `N=3`, `omega=0.5`, and the fixed spin sector
`n_up=2, n_down=1`. The model is
`SpENNWavefunction(exp(J_ee) * PfaffianReadout(SpechtMP(ElectronPairEncoder)))`.
The encoder includes spin labels as particle-token features, SpechtMP uses
explicit gate-based activations, and the readout is Pfaffian-based with odd
electron bordering enabled for `N=3`.

Training uses `VMCLoss` with Adam and Metropolis walkers only. There is no
supervised exact/reference loss in the config, and `system.exact_energy` is
`null`. The smoke template reduces channels, walkers, production blocks, and
VMC steps to keep CI/runtime checks cheap; the benchmark template increases
those values slightly and enables the fixed-sector spin scan.

## Spin Scan

`run_spenn.py --config benchmark --scan-spins` runs fixed spin sectors from
`configs/benchmark.yaml`, currently `(3, 0)`, `(2, 1)`, `(1, 2)`, and `(0, 3)`.
It trains each sector separately with VMC and records the lowest sampled value
in this smoke scan as the scan best candidate. The scan does not optimize or
mix spin sectors during one run. Scan parents have their own processed CSV/JSON
output and fixed-sector energy/variance/acceptance figure.

## Data and Plots

Run directories are written under `outputs/YYYY-MM-DD/`. Generated configs and
summaries record `run.time` in `HH-MM-SS` format, and generated run ids include
that timestamp. `process_outputs.py` writes processed CSV/JSON under the saved
run directory, while `plot_outputs.py` reads saved metrics and plot CSVs and
writes PNGs under `experiments/hooke_multibody/figures/spenn/`.
Sampler-health outputs include acceptance, proposal scale, pair-distance
summaries, local-energy sample count, autocorrelation time, and effective
sample size when enough sequential production blocks are present. Final
production/evaluation metrics are written to `metrics/eval_metrics.csv` and
promoted into processed `data/eval_metrics.csv`.
The wrapper success gate requires the final acceptance rate to stay in the
configured sampler-health range, currently `0.3 <= acceptance_rate <= 0.7`.
`energy_plausibility.csv` is the canonical energy table for now. Because no
high-accuracy reference is available, its exact-reference and delta columns are
intentionally blank. If a Gaussian Hartree baseline run is supplied during
processing, separate baseline columns record the baseline energy and SpENN
offset from that baseline only when `system.n_electrons`,
`system.harmonic_omega`, and `system.spatial_dim` match. Cusp plots report a
two-sided direction-averaged full-wavefunction slope, the analytic
cusp-module-only slope, and the residual smooth-factor slope.

## Reproduction

Run from the repository root with the CPU uv extra:

```bash
uv sync --extra cpu
uv run --extra cpu python experiments/hooke_multibody/run_reference.py --config reference
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config smoke
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config benchmark --scan-spins
```

Process and plot a saved SpENN run or scan parent without a baseline with:

```bash
uv run --extra cpu python experiments/hooke_multibody/process_outputs.py --spenn-run outputs/YYYY-MM-DD/<run-name>/<run-id>
uv run --extra cpu python experiments/hooke_multibody/plot_outputs.py --run outputs/YYYY-MM-DD/<run-name>/<run-id>
```

For the baseline-aware flow, run the reference wrapper first, process the saved
SpENN run or scan parent with that reference run, then regenerate plots:

```bash
uv run --extra cpu python experiments/hooke_multibody/run_reference.py --config reference
uv run --extra cpu python experiments/hooke_multibody/process_outputs.py --spenn-run outputs/YYYY-MM-DD/<spenn-run-name>/<spenn-run-id> --reference-run outputs/YYYY-MM-DD/hooke_multibody_reference/<reference-run-id>
uv run --extra cpu python experiments/hooke_multibody/plot_outputs.py --run outputs/YYYY-MM-DD/<spenn-run-name>/<spenn-run-id>
```

When metadata-compatible baseline CSVs are present, `plot_outputs.py` overlays
the baseline energy and density curves on the relevant SpENN figures. Spin-scan
parent figures also include the baseline energy line after baseline-aware
processing.

All `run_*.py` files are wrappers around reusable training/artifact utilities;
they do not instantiate core Hamiltonian, sampler, model, optimizer, loss, or
trainer objects directly.
Local CSV/JSON/checkpoint artifacts are always written. W&B tracking is
available through `tracking.wandb.*` config fields and is disabled by default
for smoke and CI runs. To enable it, include the optional extra, for example
`uv run --extra cpu --extra wandb python experiments/hooke_multibody/run_spenn.py --config smoke tracking.wandb.enabled=true`.

## Local Sanity Snapshot

The following local CPU artifacts were generated on 2026-06-03 under
`outputs/codex_hooke_multibody/`. They are smoke-scale checks of the workflow,
not converged VMC evidence.

The Gaussian Hartree baseline for `N=3`, `omega=0.5`, `spatial_dim=3` has
`alpha=0.1822844261592677` and energy `3.80847536`. The `delta GH` column is
`energy - baseline_energy`; it is a baseline offset, not an exact-reference
error.

| run | sector | energy | delta GH | sem | variance | acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hooke_multibody_spenn_15-23-38_b88e8f03` | `N=3, up=2, down=1` | 4.72118557 | 0.91271021 | 0.08647394 | 0.05982194 | 0.625 |
| `hooke_multibody_spin_scan_10-03-43_514cbd97_up3_down0` | `N=3, up=3, down=0` | 4.59707710 | 0.78860173 | 0.08134876 | 0.02647048 | 0.625 |
| `hooke_multibody_spin_scan_10-03-43_514cbd97_up2_down1` | `N=3, up=2, down=1` | 4.03680492 | 0.22832956 | 0.16581581 | 0.10997953 | 0.625 |
| `hooke_multibody_spin_scan_10-03-43_514cbd97_up1_down2` | `N=3, up=1, down=2` | 4.45998269 | 0.65150733 | 0.12493563 | 0.06243565 | 0.625 |
| `hooke_multibody_spin_scan_10-03-43_514cbd97_up0_down3` | `N=3, up=0, down=3` | 4.71338838 | 0.90491302 | 0.12463252 | 0.06213306 | 0.625 |

The smoke run had particle-token antisymmetry error below `6e-16` and sign-flip
accuracy `1.0`. The analytic cusp module itself had small short-range slope
errors (`cusp_only_same_mean_error=-9.49e-4`,
`cusp_only_opposite_mean_error=-5.05e-3`). With the float64 Pfaffian readout
floor lowered to `1e-30`, the two-sided full-wavefunction slope errors were
diagnostic-scale for this smoke run (`same_mean_error=-4.26e-3`,
`opposite_mean_error=-9.94e-2`). The remaining smooth-factor residual slopes
were `-3.31e-3` for the same-spin pair and `-9.43e-2` averaged over
opposite-spin pairs. These are smoke diagnostics, not convergence claims.

## Figure Gallery

The energy, pair-distance, radial-density, and fixed-sector scan figures below
were regenerated after baseline-aware processing. They include Gaussian Hartree
overlays where the corresponding baseline CSV is available.

Smoke energy trace:

![Smoke energy trace](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_energy_trace.png)

Local-energy variance:

![Local-energy variance](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_local_energy_variance.png)

Sampler acceptance:

![Sampler acceptance](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_acceptance_rate.png)

Local-energy histogram:

![Local-energy histogram](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_local_energy_histogram.png)

Pair-distance histogram:

![Pair-distance histogram](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_pair_distance_histogram.png)

Radial density:

![Radial density](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_radial_density.png)

Spin-resolved cusp slopes:

![Spin-resolved cusp slopes](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_cusp_slope_by_spin.png)

Particle-token antisymmetry:

![Particle-token antisymmetry](figures/spenn/hooke_multibody_spenn_15-23-38_b88e8f03_particle_antisymmetry.png)

Fixed-sector spin scan:

![Fixed-sector spin scan](figures/spenn/hooke_multibody_spin_scan_10-03-43_514cbd97_spin_scan_energy.png)

## Slurm

`slurm/cpu_smoke.job` runs the multibody integration smoke test with the `cpu`
uv extra. `slurm/gpu_smoke.job` uses `.venv-gpu`, the `cu126` uv extra, checks
CUDA, and runs the smoke config on `device=cuda`.

As of 2026-06-03, both `sbatch --test-only` and real `sbatch --parsable`
submission attempts failed for the CPU and GPU smoke scripts because the login
node could not contact the Slurm controller. The real submission error was
`sbatch: error: Batch job submission failed: Unable to contact slurm controller
(connect failure)`. A later retry also printed `sbatch: error: Failed to
lookup user homedir to load slurm defaults.` before the same controller-contact
failure. The latest bounded retry with `timeout 90s sbatch --test-only ...`
produced no controller response before timing out. The scripts are present, but
no controller-backed Slurm smoke job was accepted from this checkout.

## Known Limitations

The scaffold is intentionally small. Pfaffian evaluation, local-energy second
derivatives, and Specht tuple tensors become expensive quickly as `N`, channel
counts, or Specht order increase.
