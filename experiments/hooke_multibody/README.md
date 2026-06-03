# Hooke Multibody

This experiment extends the two-electron Hooke benchmark to an `N`-electron
harmonic Coulomb trap. The current scaffold is intentionally small: `N=3`,
SpENN training only, and VMC energy minimization only. It is not a supervised
fit to an exact wavefunction.

No analytic or independent high-accuracy multibody reference is configured yet.
The reported energy is therefore a VMC estimate, not an exact-error benchmark.
Reference metadata lives in `configs/reference.yaml` until an external
reference pipeline is added.

Antisymmetry is tested with the particle-token convention: an electron exchange
permutes position and spin label together. It does not swap coordinates while
holding fixed spin labels in place.

## Reproduce

Run from the repository root:

```bash
uv sync --extra cpu
uv run --extra cpu python experiments/hooke_multibody/run_reference.py --config reference
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config smoke
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config benchmark --scan-spins
```

The benchmark spin scan is a fixed-sector scan over `configs/benchmark.yaml`
partitions, currently `(n_up, n_down) = (3, 0)`, `(2, 1)`, `(1, 2)`, and
`(0, 3)`. Each sector is a separate VMC run; the scan summary reports the
lowest sampled VMC energy. Scan parents can also be processed and plotted;
`plot_outputs.py` writes a fixed-sector energy/variance/acceptance figure.

To process a saved run into comparison-ready CSV/JSON:

```bash
uv run --extra cpu python experiments/hooke_multibody/process_outputs.py --spenn-run outputs/YYYY-MM-DD/<run-name>/<run-id>
```

To plot a saved run:

```bash
uv run --extra cpu python experiments/hooke_multibody/plot_outputs.py --run outputs/YYYY-MM-DD/<run-name>/<run-id>
```

Run artifacts are written under `outputs/YYYY-MM-DD/`. Each generated config
records `run.time` in `HH-MM-SS` format, and auto-generated run ids include the
same time stamp.

## Outputs

The generic training stack writes config, summary, checkpoint, metrics CSVs,
and plot-data CSVs. Multibody diagnostics currently include all-pair distance
histograms, one-body radial density, spin-resolved cusp slope estimates, and
particle-token antisymmetry checks. Production sampler health includes
acceptance, proposal scale, pair-distance summaries, local-energy sample count,
autocorrelation time, and effective sample size when enough sequential blocks
are available. `process_outputs.py` also promotes local-energy and pair-distance
sample tables into `data/`. Cusp diagnostics report both full-wavefunction
short-range slopes and analytic cusp-module-only slopes when the model exposes a
`cusp` module. The full-wavefunction fit uses a two-sided pair-separation
average to cancel smooth odd terms before estimating the radial cusp slope.
Figures generated from saved CSVs are written under
`experiments/hooke_multibody/figures/spenn/`.
`process_outputs.py` writes `data/energy_plausibility.csv` for both single runs
and scan parents. Until a reference is added, that table records
`reference_available=false` and leaves reference/delta columns blank.

## Version Notes

Scaffold provenance recorded on 2026-06-03:

```text
base git commit: 64fb489
python: 3.14.5
torch: 2.12.0+cpu
local cuda available: false
```

Local verification used the CPU uv environment through `.venv/bin/python`:

```bash
.venv/bin/python -m pytest \
  tests/unit/diagnostics/test_multibody_wavefunction.py \
  tests/unit/training/test_run_metadata.py \
  tests/integration/test_hooke.py \
  tests/integration/test_hooke_spenn.py \
  tests/integration/test_hooke_multibody.py \
  -q --typeguard-packages=spenn
```

Current focused verification:

```bash
.venv/bin/python -m pytest \
  tests/unit/diagnostics/test_statistics.py \
  tests/unit/diagnostics/test_multibody_wavefunction.py \
  tests/unit/training/test_run_metadata.py \
  tests/integration/test_hooke.py \
  tests/integration/test_hooke_spenn.py \
  tests/integration/test_hooke_multibody.py \
  -q --typeguard-packages=spenn
```

Result: `28 passed in 40.36s`.

The Pfaffian readout floor is set explicitly in `configs/spenn.yaml` through
`readout_hparams.eps`. The multibody templates use a small float64 floor
(`1e-30`) so same-spin cusp diagnostics near antisymmetric nodes are not
dominated by log-amplitude clamping.

## Slurm

Cluster smoke checks live in `experiments/hooke_multibody/slurm/`.

```bash
sbatch experiments/hooke_multibody/slurm/cpu_smoke.job
sbatch experiments/hooke_multibody/slurm/gpu_smoke.job
```

The CPU script runs the multibody integration smoke test with the `cpu` uv
extra. The GPU script uses `.venv-gpu`, the `cu126` uv extra, checks CUDA, and
runs the smoke SpENN config on `device=cuda`.

Current controller status on 2026-06-03: `sbatch --test-only` for both scripts
failed with:

```text
allocation failure: Unable to contact slurm controller (connect failure)
```

No controller-backed Slurm smoke job was accepted from this checkout.
