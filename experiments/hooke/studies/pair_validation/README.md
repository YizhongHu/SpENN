# Hooke pair validation study (`hooke_pair_validation_v1`)

This directory records the PR8 validation-scan protocol for the Hooke pair
benchmark. In PR8.1, the canonical deliverable is durable training/run
artifacts and runner-owned checkpoint restore. Benchmark post-processing
scripts are intentionally deferred to PR8.2.

Local run directories are authoritative. W&B may be used for visualization,
but W&B clicks are not a source of protocol, checkpoint, or selection truth.

## Scope

PR8.1 keeps this study boring:

```text
launch validation training runs
run train-end validation inside each training job
write durable checkpoint directories
write early run_start.json breadcrumbs
keep SLURM logs and local run outputs for reproducibility
```

Deferred to PR8.2:

```text
collect.py
select.py
evaluate_selected.py
selection reports
final evaluation command generation
final benchmark summaries
```

Validation is used for later model/protocol selection. Validation does not use
the exact Hooke reference energy. Final evaluation is separate and may compare
against the exact Hooke reference only after selection is frozen.

## Launching The Scan

[launch_array.sh](launch_array.sh) maps `SLURM_ARRAY_TASK_ID` onto the manifest
grid and launches one training run per task. From the repository root:

```bash
mkdir -p slurm_logs
sbatch experiments/hooke/studies/pair_validation/launch_array.sh
```

CPU example:

```bash
DEVICE=cpu sbatch -p sapphire --gres="" \
  experiments/hooke/studies/pair_validation/launch_array.sh
```

The arrays in `launch_array.sh` must mirror [manifest.yaml](manifest.yaml).
Changing the grid means creating a new study version.

## Checkpoint Artifacts

Training runs use `spenn.callback.Checkpoint`, which writes package-owned
directory checkpoints through `spenn.checkpoint`:

```text
checkpoints/
  step_000100/
    manifest.json
    resolved_config.yaml
    model.pt
    optimizer.pt
    trainer.json
    sampler.pt
    rng.pt
    COMPLETE
  latest.json
```

A checkpoint directory is valid only when `COMPLETE` is present. `latest.json`
replaces symlink or `latest.pt` tracking. Checkpoints store state dicts and
metadata; they do not pickle full model objects and do not instantiate models.

Restore intent is runner-owned:

```yaml
load:
  path: null
  mode: none   # none | model_only | train_resume
  strict: true
  allow_protocol_mismatch: false
```

`model_only` is for `Evaluate`. `train_resume` is for `Train`. Config
instantiates objects first; checkpoint restore loads state into those objects.

Checkpoint writing policy stays under `checkpoint` in train configs:

```yaml
checkpoint:
  every_n_steps: 500
  keep_last: 3
```

## Run Durability

Each run writes `run_start.json` early, before long training begins. It records
the run id, run directory, study identity, command, git sha/branch/dirty flag,
hostname, cwd, selected SLURM metadata, and a small allowlist of environment
fields. It intentionally does not dump full `os.environ` or secrets.

Keep these local files with the SLURM logs:

```text
run_start.json
resolved_config.yaml
metadata.json
status.json
metrics.csv
metrics.jsonl
checkpoints/*/manifest.json
checkpoints/*/COMPLETE
checkpoints/latest.json
```

## PR8.2 Handoff

PR8.2 should consume the durable checkpoint interface introduced here. Its
post-processing scripts should read local run directories and checkpoint
manifests, never infer model architecture from checkpoint keys, and never use
W&B as the source of selection or benchmark decisions.
