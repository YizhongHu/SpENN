# Metrics Naming and Logging Conventions

This document defines the canonical metric naming scheme for SpENN logs.

The same logical metric identity should be preserved across CSV, JSONL, and optional W&B logging.

## Goal

Metrics should be:

```text
machine-readable
human-scannable
stable across logger backends
easy to group by phase / subsystem
easy to aggregate across runs
easy to visualize in dashboard tools
```

The local run directory remains the authoritative experiment record. External dashboards such as W&B are projections of local records, not replacements for them.

---

## Canonical metric identity

A scalar metric is identified by:

```text
namespace + key
```

The canonical logical path is:

```text
<namespace>/<key>
```

Examples:

```text
train/energy
train/sampler/acceptance_rate
train/perf/step_time_sec
checks/equivariance/full_model/max_abs_error
runtime/wall_time_sec
diagnostics/energy/time_sec
```

Use:

```text
namespace:
  slash-separated ownership / phase path

key:
  underscore-separated leaf metric name
```

Examples:

| Namespace                        | Key               | Logical path                                   |
| -------------------------------- | ----------------- | ---------------------------------------------- |
| `train`                          | `energy`          | `train/energy`                                 |
| `train`                          | `energy_stderr`   | `train/energy_stderr`                          |
| `train/sampler`                  | `acceptance_rate` | `train/sampler/acceptance_rate`                |
| `train/perf`                     | `step_time_sec`   | `train/perf/step_time_sec`                     |
| `checks/data_validity`           | `passed`          | `checks/data_validity/passed`                  |
| `checks/equivariance/full_model` | `max_abs_error`   | `checks/equivariance/full_model/max_abs_error` |
| `runtime`                        | `wall_time_sec`   | `runtime/wall_time_sec`                        |
| `diagnostics/energy`             | `time_sec`        | `diagnostics/energy/time_sec`                  |

Do not treat `key` alone as globally unique.

---

## Namespace conventions

Namespaces describe **where the metric comes from** or **which subsystem owns it**.

Preferred namespaces:

```text
train
train/sampler
train/perf

eval
eval/sampler
eval/perf

runtime

diagnostics/energy

checks/data_validity
checks/gradient
checks/sampler
checks/equivariance/full_model
checks/equivariance/trace
```

Avoid putting hierarchy inside the key:

```text
Bad:
  namespace = train
  key = sampler.acceptance_rate

Good:
  namespace = train/sampler
  key = acceptance_rate
```

Do not mix separators unless there is a deliberate compatibility reason.

Preferred:

```text
train/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
```

Avoid:

```text
train/sampler.acceptance_rate
checks.equivariance.full_model.max_abs_error
checks_equivariance_full_model_max_abs_error
```

---

## Key conventions

Keys describe the **leaf metric**.

Preferred key style:

```text
loss
energy
energy_variance
energy_std
energy_stderr
local_energy_n_finite
local_energy_n_total
local_energy_finite_fraction
local_energy_nonfinite_count

acceptance_rate
n_walkers
burn_in
n_steps

passed
max_abs_error
n_failed_entries

wall_time_sec
step_time_sec
```

Use underscores inside keys.

Avoid dots in keys.

---

## CSV logging

CSV uses long-form scalar records:

```csv
step,namespace,key,value
1,train,loss,0.6580171494985763
1,train,energy,3.658536762108973
1,train/sampler,acceptance_rate,0.8625
1,checks/data_validity,passed,true
1,checks/equivariance/full_model,max_abs_error,0.0
```

The metric identity is:

```text
(namespace, key)
```

This format avoids sparse, very wide CSV files and supports records from trainers, callbacks, checks, diagnostics, and runtime lifecycle events.

### Required CSV columns

```text
step
namespace
key
value
```

### Optional future columns

If useful later:

```text
run_id
timestamp_unix
```

### Values

Values should be JSON-compatible scalars:

```text
int
float
bool
str
null / empty where appropriate
```

Do not duplicate the same metric under multiple spellings.

Bad:

```csv
step,namespace,key,value
1,train,sampler.acceptance_rate,0.8625
1,train/sampler,acceptance_rate,0.8625
```

Good:

```csv
step,namespace,key,value
1,train/sampler,acceptance_rate,0.8625
```

---

## JSONL logging

JSONL should preserve the same logical metric identity.

Recommended event-oriented shape:

```json
{
  "step": 1,
  "namespace": "train",
  "metrics": {
    "loss": 0.6580171494985763,
    "energy": 3.658536762108973,
    "energy_variance": 1.463636129155399,
    "energy_stderr": 0.3024520756619343
  }
}
```

Subsystem record:

```json
{
  "step": 1,
  "namespace": "train/sampler",
  "metrics": {
    "acceptance_rate": 0.8625,
    "n_walkers": 16,
    "burn_in": 10,
    "n_steps": 5
  }
}
```

Runtime-check record:

```json
{
  "step": 1,
  "namespace": "checks/equivariance/full_model",
  "metrics": {
    "n_particles": 2,
    "n_permutations_tested": 1,
    "n_failed_permutations": 0,
    "max_abs_error": 0.0,
    "passed": true,
    "checker_class": "FullModelEquivarianceChecker"
  }
}
```

A scalar-row JSONL format is also acceptable:

```json
{"step": 1, "namespace": "train", "key": "energy", "value": 3.658536762108973}
{"step": 1, "namespace": "train/sampler", "key": "acceptance_rate", "value": 0.8625}
```

Both forms are valid if the canonical identity remains:

```text
namespace + key
```

---

## W&B projection

W&B is a dashboard / monitoring projection of local records.

The W&B logger should consume backend-neutral records with:

```text
step
namespace
key
value
```

or:

```text
step
namespace
metrics
```

For scalar metrics, the raw W&B metric name is:

```text
<namespace>/<key>
```

Examples:

| Local namespace                  | Local key         | W&B raw metric                                 |
| -------------------------------- | ----------------- | ---------------------------------------------- |
| `train`                          | `loss`            | `train/loss`                                   |
| `train`                          | `energy`          | `train/energy`                                 |
| `train`                          | `energy_stderr`   | `train/energy_stderr`                          |
| `train/sampler`                  | `acceptance_rate` | `train/sampler/acceptance_rate`                |
| `train/perf`                     | `step_time_sec`   | `train/perf/step_time_sec`                     |
| `checks/data_validity`           | `passed`          | `checks/data_validity/passed`                  |
| `checks/equivariance/full_model` | `max_abs_error`   | `checks/equivariance/full_model/max_abs_error` |
| `runtime`                        | `wall_time_sec`   | `runtime/wall_time_sec`                        |
| `diagnostics/energy`             | `time_sec`        | `diagnostics/energy/time_sec`                  |

W&B should not change core metric names.

W&B may add a small, documented set of dashboard aliases and health flags. These are convenience projections, not replacements for canonical metrics.

---

## Dashboard aliases

Dashboard aliases are optional W&B-only duplicates intended to make dashboards easier to build.

If enabled, keep the alias set small.

Recommended aliases:

```text
dashboard/loss
dashboard/energy
dashboard/energy_variance
dashboard/energy_stderr
dashboard/acceptance_rate
dashboard/grad_norm
dashboard/local_energy_finite_fraction
dashboard/step_time_sec
```

Aliases should be derived from canonical metrics.

Examples:

| Canonical metric                     | Dashboard alias                          |
| ------------------------------------ | ---------------------------------------- |
| `train/loss`                         | `dashboard/loss`                         |
| `train/energy`                       | `dashboard/energy`                       |
| `train/energy_variance`              | `dashboard/energy_variance`              |
| `train/energy_stderr`                | `dashboard/energy_stderr`                |
| `train/sampler/acceptance_rate`      | `dashboard/acceptance_rate`              |
| `train/grad_norm`                    | `dashboard/grad_norm`                    |
| `train/local_energy_finite_fraction` | `dashboard/local_energy_finite_fraction` |
| `train/perf/step_time_sec`           | `dashboard/step_time_sec`                |

Do not add dozens of aliases. The canonical namespaces remain the complete metric surface.

---

## Health flags

W&B dashboards benefit from compact numeric status indicators.

If enabled, health flags should use:

```text
1.0 = OK
0.0 = failed / warning condition
```

Recommended health flags:

```text
health/numerics_ok
health/sampler_ok
health/equivariance_ok
health/run_ok
```

Examples:

```text
health/numerics_ok:
  1.0 if local-energy/logabs/batch validity checks pass
  0.0 otherwise

health/sampler_ok:
  1.0 if sampler checks pass
  0.0 otherwise

health/equivariance_ok:
  1.0 if runtime equivariance checks pass
  0.0 otherwise

health/run_ok:
  1.0 if all required health flags are OK
  0.0 otherwise
```

If a flag cannot be derived from the current record, omit it rather than guessing.

Health flags do not replace detailed metrics under `checks/...`.

---

## Step conventions

Steps are 0-indexed: the first training step is `step = 0`. This means the
first step always satisfies `step % every_n_steps == 0` cadence gates, so
periodic callbacks and loggers report at the start of every run.

Training metrics use the training step:

```text
step = trainer step
namespace = train
```

Training runtime checks use the same training step:

```text
step = trainer step
namespace = checks/...
```

Evaluation metrics use an evaluation step or `0` if there is only one evaluation event:

```text
step = eval step or 0
namespace = eval
```

Run-level metadata may use `step = 0`:

```csv
step,namespace,key,value
0,runtime,start_time_unix,1730000000.123
0,runtime,end_time_unix,1730000420.456
0,runtime,wall_time_sec,420.333
```

If a logger supports nullable steps, run-level metadata may use `step = null`. If not, use `0`.

---

## W&B step axes

W&B should use explicit step metrics. Do not rely only on W&B’s implicit step counter.

Suggested definitions:

```python
run.define_metric("train/*", step_metric="train/step")
run.define_metric("train/sampler/*", step_metric="train/step")
run.define_metric("train/perf/*", step_metric="train/step")

run.define_metric("eval/*", step_metric="eval/step")
run.define_metric("eval/sampler/*", step_metric="eval/step")
run.define_metric("eval/perf/*", step_metric="eval/step")

run.define_metric("checks/*", step_metric="checks/train_step")
run.define_metric("diagnostics/*", step_metric="eval/step")

run.define_metric("dashboard/*", step_metric="train/step")
run.define_metric("health/*", step_metric="train/step")
```

The W&B logger is responsible for adding appropriate step fields, for example:

```text
train/step
eval/step
checks/train_step
```

when logging records in those namespaces.

Runtime metrics such as `runtime/wall_time_sec` may be logged once and also written to the W&B run summary.

---

## Timing metrics

Timing metrics should use seconds.

Use suffixes:

```text
_time_sec
_wall_time_sec
```

Recommended run-level timing:

```text
runtime/start_time_unix
runtime/end_time_unix
runtime/wall_time_sec
```

Recommended training timing:

```text
train/perf/step_time_sec
train/perf/sampling_time_sec
train/perf/forward_time_sec
train/perf/local_energy_time_sec
train/perf/backward_time_sec
train/perf/optimizer_step_time_sec
```

Recommended evaluation timing:

```text
eval/perf/wall_time_sec
diagnostics/energy/time_sec
```

CSV examples:

```csv
step,namespace,key,value
0,runtime,start_time_unix,1730000000.123
0,runtime,end_time_unix,1730000420.456
0,runtime,wall_time_sec,420.333
1,train/perf,step_time_sec,0.842
1,train/perf,sampling_time_sec,0.301
1,train/perf,local_energy_time_sec,0.412
0,eval/perf,wall_time_sec,12.4
0,diagnostics/energy,time_sec,11.8
```

Timing is runtime metadata. Do not put timing inside physics/statistics helpers such as local-energy summary functions.

---

## Hamiltonian term metrics

For dict-shaped Hamiltonians, configured keys are authoritative term names.

Example:

```yaml
hamiltonian_terms:
  kinetic:
    _target_: spenn.physics.kinetic.KineticEnergy

  harmonic_trap:
    _target_: spenn.physics.potential.HarmonicTrap
    omega: ${system.omega}

  electron_electron:
    _target_: spenn.physics.potential.ElectronElectronInteraction
```

Metric names should use these configured names:

```text
train/energy_term_kinetic
train/energy_term_harmonic_trap
train/energy_term_electron_electron
```

CSV examples:

```csv
step,namespace,key,value
1,train,energy_term_kinetic,0.04717715942440409
1,train,energy_term_kinetic_variance,0.0025556148614036237
1,train,energy_term_harmonic_trap,3.4269269388366514
1,train,energy_term_electron_electron,0.18443266384791782
```

Do not generate class-name/index metric names such as:

```text
energy_term_HarmonicTrap_0
energy_term_2
```

for named Hamiltonian terms.

---

## Runtime checks

Runtime checks should live under `checks/...`.

Examples:

```text
checks/data_validity/passed
checks/gradient/passed
checks/sampler/passed
checks/equivariance/full_model/passed
checks/equivariance/trace/passed
```

Detailed check metrics stay in the same namespace:

```text
checks/data_validity/local_energy_nonfinite_fraction
checks/gradient/global_grad_norm
checks/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
checks/equivariance/trace/n_failed_entries
```

---

## Canonical examples

Preferred:

```text
train/loss
train/energy
train/energy_variance
train/energy_stderr
train/local_energy_finite_fraction
train/logabs_mean

train/sampler/acceptance_rate
train/sampler/n_walkers

train/perf/step_time_sec
train/perf/sampling_time_sec

checks/data_validity/passed
checks/gradient/global_grad_norm
checks/sampler/acceptance_rate
checks/equivariance/full_model/max_abs_error
checks/equivariance/trace/n_failed_entries

runtime/wall_time_sec

eval/energy
eval/energy_error
eval/perf/wall_time_sec

diagnostics/energy/time_sec
```

Avoid:

```text
train/sampler.acceptance_rate
sampler.acceptance_rate
energy
energy_mean
full_model_equivariance_max_abs_error
checks.equivariance.full_model.max_abs_error
runtime_wall_time_sec
```

---

## Summary

Canonical metric identity:

```text
namespace + key
```

Use:

```text
slash-separated namespace
underscore-separated key
```

CSV stores these separately:

```csv
step,namespace,key,value
1,train,energy,3.65
```

JSONL preserves the same structure:

```json
{"step": 1, "namespace": "train", "metrics": {"energy": 3.65}}
```

W&B joins them:

```text
train/energy
```

Do not maintain multiple primary names for the same metric. If a downstream tool needs flattened names, flatten only at export time.
