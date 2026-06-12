# Hooke pair validation study (`hooke_pair_validation_v1`)

A reproducible validation scan for selecting the Hooke pair model/protocol
before the final benchmark. The protocol is declared in [manifest.yaml](manifest.yaml);
once the scan starts, changing the manifest means a new study version.

The workflow:

```text
1. The user launches one training run per grid point (loop or SLURM array).
2. Each run executes train-end validation: an independent validation sampler
   draws fresh samples from the trained model and validation/* metrics are
   logged into the run directory.
3. collect.py normalizes the run directories into runs.csv / runs.jsonl.
4. select.py applies the manifest-declared rule and writes selection.csv,
   selected_config.yaml, and selection_report.md.
5. Final benchmark evaluation of the selected config happens separately
   (Evaluate runner, eval/* metrics, exact reference energy).
```

Local run outputs are authoritative end to end. W&B may visualize the runs,
but neither the collector nor the selector reads W&B.

## Quick test run (start here)

Before launching the real scan, confirm the whole pipeline
(train → validation → collect → select) works:

```bash
# local workstation / WSL (auto-detects GPU):
bash experiments/hooke/studies/pair_validation/test_run.sh

# FASRC SLURM, GPU:
sbatch -p kozinsky_gpu --gres=gpu:1 -c 4 -t 00:30:00 \
  experiments/hooke/studies/pair_validation/test_run.sh
```

It trains a tiny 2×2 grid at smoke scale into a scratch directory, runs the
collector and selector against it, checks every expected artifact and column,
and prints `TEST RUN PASSED` when everything works as expected.

## Launching the scan (user's duty)

Each grid point in the manifest is one training run of
[experiments/hooke/configs/benchmark/pair_train.yaml](../../configs/benchmark/pair_train.yaml)
with dotlist overrides. `run.py` takes one config per invocation (there is no
Hydra multirun in this repo), so expand the grid with a SLURM array (preferred
on FASRC) or a shell loop.

### SLURM array (preferred)

[launch_array.sh](launch_array.sh) maps `SLURM_ARRAY_TASK_ID` (0–53) onto the
54 grid points by mixed-radix decoding of the manifest axes, then launches one
training run per task. From the repo root:

```bash
mkdir -p slurm_logs   # keep these logs for reproducibility
sbatch experiments/hooke/studies/pair_validation/launch_array.sh
```

Defaults target `kozinsky_gpu` with one GPU per task; override partition or
device as needed, e.g. CPU on sapphire:

```bash
DEVICE=cpu sbatch -p sapphire --gres="" \
  experiments/hooke/studies/pair_validation/launch_array.sh
```

If the grid in the manifest changes, `launch_array.sh`'s axis arrays and
`--array` range must change with it — that is a new study version.

### Shell loop (small machines)

```bash
export UV_PROJECT_ENVIRONMENT=.venv-gpu  # GPU runs; omit for CPU (.venv)

ROOT=outputs/hooke_pair_validation_v1
for seed in 3 9 11; do
  for lr in 3e-4 1e-3 3e-3; do
    for channels in 8 32 128; do
      for gate in silu sigmoid; do
        uv run --extra cu126 python run.py \
          --config experiments/hooke/configs/benchmark/pair_train.yaml \
          run.root=$ROOT \
          study.name=hooke_pair_validation_v1 \
          runtime.seed=$seed \
          optimizer_params.lr=$lr \
          model_params.channels=$channels \
          model_params.gate_activation=$gate
      done
    done
  done
done
```

On the FASRC cluster, submit the same command shape as a SLURM array over the
54 grid points instead of a serial loop.

### W&B policy

Enable the W&B logger by uncommenting it in the benchmark config. Project
policy:

- test runs go to project `SpENN-QMC-test`;
- the real scan goes to project `SpENN-QMC` with tags `hooke-pair`, the
  accelerator (`cpu` or `cuda`), and `study-v1` (the benchmark config's
  `wandb:` block already sets these).

## Collecting results

```bash
uv run python experiments/hooke/studies/pair_validation/collect.py \
  --run-root outputs/hooke_pair_validation_v1 \
  --output-dir experiments/hooke/studies/pair_validation/results
```

The collector scans for run directories (any directory holding
`metadata.json`), reads `resolved_config.yaml`, `status.json`,
`metadata.json`, and `metrics.jsonl`, and writes one normalized row per run to
`runs.csv` and `runs.jsonl`. Failed and incomplete runs appear explicitly with
`status=failed` / `status=incomplete`. The collector never picks a winner.

Each row records the deterministic `config_id`
(e.g. `lr=0.001_channels=32_layers=1_gate_activation=silu`), derived from the
non-seed grid fields of the resolved config (or taken from `study.config_id`
when the launcher recorded one), plus the git sha and run directory.

## Selecting the winner

```bash
uv run python experiments/hooke/studies/pair_validation/select.py \
  --runs experiments/hooke/studies/pair_validation/results/runs.csv \
  --output-dir experiments/hooke/studies/pair_validation/results
```

The selection rule (declared in the manifest, applied by `select.py`):

```text
group runs by the non-seed hyperparameters
failed / ineligible / missing seeds count as +inf validation energy
rank by median validation/energy
tie-break by validation/energy_variance, seed-to-seed spread,
  smaller model (channels), lower wall time, then config_id
```

Eligibility requires `checks/data_integrity/passed`,
`checks/gradient/passed`, `checks/equivariance/full_model/passed`, and
`validation/local_energy_finite_fraction = 1.0`.

Outputs in `results/`: `selection.csv` (full ranking),
`selected_config.yaml` (frozen winner + reproduction overrides), and
`selection_report.md` (ranking, winner geometry table, and flags).

## Why validation never uses the exact reference energy

Validation exists to *choose* a model/protocol from independent samples.
Comparing against the exact Hooke energy during selection would leak the
benchmark answer into the choice and turn the final evaluation into a
formality. Validation therefore logs only estimator quality
(`validation/energy`, `validation/energy_variance`, `validation/energy_stderr`,
finite fractions, sampler metadata); `eval/energy_error`,
`eval/energy_abs_error`, and `eval/reference_energy` belong exclusively to
final evaluation, after the protocol is frozen. The `Validation` callback
rejects diagnostics configured with `reference_energy`.

## How final evaluation differs from validation

```text
validation  - runs at train_end inside each training job, fresh validation
              sampler, validation/* metrics, used for selection
final eval  - separate Evaluate run on the frozen selected protocol, fresh
              evaluation sampler, eval/* metrics, compares against the exact
              Hooke reference energy
```

## Sampler geometry diagnostics

`validation/sampler/*` includes walker-geometry summaries computed from the
validation walkers (`radius_*`, `electron_distance_*`, `position_*`,
`center_of_mass_rms`). For Hooke the sampler must remain confined by the
wavefunction envelope: growing `radius_q99`/`radius_max` means walkers are
escaping into large-radius tails, and a collapsing
`electron_distance_q01`/`electron_distance_min` means near-coalescence
sampling — both make energy estimates untrustworthy. `select.py` flags
missing/nonfinite geometry and `electron_distance_q01` below the manifest's
`geometry_flags.electron_distance_q01_min`, but geometry never decides the
winner unless the manifest lists it under eligibility or tie-breakers.

## Why DataIntegrity is separate from Validation

```text
DataIntegrity  - runtime soundness checks during training (finite local
                 energies, valid signs, well-formed batches), logged under
                 checks/data_integrity/*; a guardrail, not an estimator
Validation     - independent estimator on fresh samples used for
                 model/protocol selection, logged under validation/*
```

A run can be numerically sound yet a poor model (passes DataIntegrity, bad
validation energy), or a good model with a corrupted step (fails
DataIntegrity, which makes it ineligible for selection regardless of its
validation energy).

## Tests

Study-script tests live next to the scripts (experiments code is independent
of `spenn/`):

```bash
uv run pytest experiments/hooke/studies/pair_validation/tests -q
```
