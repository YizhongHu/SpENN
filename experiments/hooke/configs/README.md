# Legacy Hooke Test Configs

This directory contains legacy test/reference configs for quick Hooke checks.
They are self-contained experiment cards, but they are not canonical study
configs.

## Layout

```text
experiments/hooke/configs/
  smoke/
    pair_train.yaml   # legacy cheap VMC training test config
    pair_eval.yaml    # legacy cheap sampled-evaluation test config
  benchmark/
    pair_train.yaml   # legacy benchmark-shaped test/reference config
    pair_eval.yaml    # legacy benchmark-shaped test/reference config
  preflight/
    pair_train_cpu.yaml   # legacy CPU preflight test config
    pair_train_gpu.yaml   # legacy GPU preflight test config
```

The old Hooke pair-validation study configs were removed because they depended
on the retired diagnostics/probe stack. Replacement study-level preflight
configs should use the new evaluator task stack rather than copying these
legacy test configs.

SLURM submission scripts for these configs live in
[`experiments/hooke/slurm/`](../slurm/).

## How to run

From the repository root:

```bash
# Train with a legacy CPU smoke test config:
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_train.yaml

# Evaluate with a legacy CPU smoke test config:
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_eval.yaml
```

Any config value can be overridden with OmegaConf dotlist arguments:

```bash
# Longer local run on GPU with a custom output root:
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_train.yaml \
    runtime.device=cuda training.max_steps=200 run.root=/scratch/$USER/spenn
```

## Config anatomy

Every legacy test config has two halves.

### Parameter blocks

A *parameter block* is a config-only namespace used for user-facing knobs and
interpolation. Nothing in a parameter block is instantiated directly; component
specs read it through `${...}` interpolation.

### Component specs

A *component spec* is a `_target_` config node describing a Python object that
Hydra instantiates. Component specs should reference parameter blocks rather
than repeating literal values.

Minimal example of the pattern:

```yaml
parameter: <value>

component:
  _target_: some.package.Component
  need_parameter: ${parameter}
```

This split keeps the experiment knobs in one readable place at the top of the
file while the bottom half pins down exactly which objects get built.

## Which fields are safe to edit

Safe to edit (parameter blocks — these are the intended knobs):

```text
experiment.*        run identity and run_name
run.root            where outputs land
run.timezone        IANA timezone for run IDs and metadata
runtime.*           seed, device (cpu/cuda), dtype
system.omega        trap strength (changes the exact reference energy!)
model_params.*      model scale
sampler_params.*    walker count and MCMC schedule
training.*          step count and logging cadence (train config)
evaluation.*        term decomposition toggle (eval config)
optimizer_params.*  learning rate (train config)
checks.*            runtime validation cadence and tolerances
load.*              restore intent: path, mode, strictness
checkpoint.*        checkpoint writing cadence and retention (train config)
timing.*            timing instrumentation knobs
status.*            terminal status cadence and included metrics
wandb.*             optional dashboard settings
references.*        reference energies (eval config)
```

Edit with care (component specs — changing these changes *what* is built, not
just how big it is):

```text
runner, model, hamiltonian_terms, sampler, optimizer, trainer,
callbacks, loggers
```

Reference energies belong in the `references` block of evaluation configs,
never in `system`.

## Where outputs go

Each run creates a directory:

```text
<run.root>/<experiment.name>/<experiment.sector>/<run_id>/
  config.yaml           # authored config (rerunnable: run_id/dir reset to null)
  run_start.json        # early run breadcrumb before long work begins
  events.jsonl          # durable lifecycle event stream, including load failures
  error.json            # failure details and traceback, written only on failure
  resolved_config.yaml  # fully interpolated config actually used
  metadata.json         # git/hardware/SLURM/runtime provenance
  status.json           # compact lifecycle status (running/completed/failed)
  metrics.csv           # long-form scalar records: step, namespace, key, value
  metrics.jsonl         # structured metric records
  checkpoints/          # complete step directories plus latest.json
  checks/               # runtime-check artifacts, e.g. equivariance reports
  diagnostics/          # evaluation diagnostic artifacts, when written
```

The local run directory is the authoritative record. W&B, when enabled, is only
a dashboard projection of scalar metrics; it never replaces the local CSV/JSONL
logs and does not receive checkpoints, traces, or raw batches by default.

## Why These Legacy Configs Are Self-Contained

Legacy smoke, preflight, and benchmark-shaped configs deliberately avoid Hydra defaults composition:

- One file answers "what exactly did this run do?" without chasing overrides
  across a config tree.
- The `config.yaml` snapshot in each run directory relaunches an identical run.
- Diffs between two experiment cards show the full experimental delta.

The cost is some duplication between configs; for benchmark provenance that
trade is intentional.
