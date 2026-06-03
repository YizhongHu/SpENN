# Hooke Multibody

This experiment extends the two-electron Hooke benchmark to an `N`-electron
harmonic Coulomb trap. The default scaffold is a small `N=3` VMC-only SpENN run
with particle-token antisymmetry: positions and spin labels permute together.

No analytic or independent high-accuracy multibody reference is configured yet.
The reported energy is therefore a VMC estimate, not an exact-error benchmark.
Reference metadata lives in `configs/reference.yaml` until an external
reference pipeline is added.

## Reproduce

Run from the repository root:

```bash
uv sync --extra cpu
uv run --extra cpu python experiments/hooke_multibody/run_reference.py --config reference
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config smoke
uv run --extra cpu python experiments/hooke_multibody/run_spenn.py --config benchmark --scan-spins
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
particle-token antisymmetry checks. Figures generated from saved CSVs are
written under `experiments/hooke_multibody/figures/`.

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

Result: `16 passed in 84.59s`.

## Slurm

Cluster smoke checks live in `experiments/hooke_multibody/slurm/`.

```bash
sbatch experiments/hooke_multibody/slurm/cpu_smoke.job
sbatch experiments/hooke_multibody/slurm/gpu_smoke.job
```
