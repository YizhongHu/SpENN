# Hooke experiment configs

Canonical configs for the Hooke pair workflow. Each config is a self-contained,
readable experiment card: everything one run needs lives in one file, with no
Hydra defaults composition.

## Layout

```text
experiments/hooke/configs/
  smoke/
    pair_train.yaml   # cheap end-to-end VMC training sanity check
    pair_eval.yaml    # cheap sampled-evaluation sanity check
```

SLURM submission scripts for these configs live in
[`experiments/hooke/slurm/`](../slurm/).

## How to run

From the repository root:

```bash
# Train (CPU smoke):
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_train.yaml

# Evaluate (CPU smoke):
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_eval.yaml
```

Any config value can be overridden with OmegaConf dotlist arguments:

```bash
# Longer local run on GPU with a custom output root:
uv run python -u run.py --config experiments/hooke/configs/smoke/pair_train.yaml \
    runtime.device=cuda training.max_steps=200 run.root=/scratch/$USER/spenn
```

## Config anatomy

Every canonical config has two halves.

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

## Why canonical configs are self-contained

Smoke and benchmark configs deliberately avoid Hydra defaults composition:

- One file answers "what exactly did this run do?" without chasing overrides
  across a config tree.
- The `config.yaml` snapshot in each run directory relaunches an identical run.
- Diffs between two experiment cards show the full experimental delta.

The cost is some duplication between configs; for benchmark provenance that
trade is intentional.
