# Experiment Timing and Profiling Spec

This note specifies low-overhead timing and resource instrumentation for SpENN
experiments. It is intentionally narrower than the planning/execution split:
it covers what runtime metrics and compact profiling artifacts should exist so
experiment studies can estimate compute cost by hyperparameter without parsing
large raw outputs.

## Motivation

Current experiment logs are good enough to report total run time and whole
training-step time, but not good enough to explain where cost comes from. In
particular, the training loop has distinct phases:

- sampling
- local-energy evaluation
- model forward
- objective construction
- backward pass
- optimizer step
- callbacks/checks/checkpoint overhead

Only whole-step timing is currently emitted. This makes it hard to tell whether
a hyperparameter increases cost through sampling, kinetic-energy/autograd work,
trace-heavy model forward work, callbacks, or checkpoint/report overhead.

The desired result is a compact cost model:

```text
cost ~= f(basis, mechanism, channels, layers, max_order, n_walkers,
          sampler_steps, evaluation_task, device)
```

This should be possible from metrics and compact collect tables alone.

## Scope

In scope:

- Scalar timing metrics in `metrics.jsonl` / `metrics.csv`.
- Small per-run profiling summaries.
- Compact collect/report tables derived from metrics.
- Optional deep PyTorch profiler traces for explicit profiling attempts.

Out of scope:

- Always-on operator traces.
- Collecting raw profiler traces into stage summaries.
- Rewriting the experiment workflow or Hydra layout.
- Ranking model quality by runtime. Runtime is reported separately.

## Design Principles

The default instrumentation must be cheap enough for production runs.

- Use scalar metrics by default.
- Keep profiler traces opt-in.
- Do not parse checkpoints or large diagnostic record CSVs for cost summaries.
- Do not make collect scan unbounded artifacts.
- Keep timing under `*/perf` or `runtime` namespaces.
- Use seconds for timing and MiB for memory.
- Treat local run directories as authoritative; dashboards are projections.

## Metric Contract

### Run-Level Metrics

Namespace: `runtime`

Required:

```text
runtime/start_time_unix
runtime/end_time_unix
runtime/wall_time_sec
runtime/device_type
runtime/host
runtime/process_id
```

Recommended:

```text
runtime/peak_memory_mb
runtime/cuda_max_memory_allocated_mb
runtime/cuda_max_memory_reserved_mb
runtime/cuda_device_name
runtime/cuda_device_count
```

`peak_memory_mb` should be best-effort. On CPU-only runs it may come from
`resource.getrusage(...).ru_maxrss` or Slurm accounting if available. On CUDA
runs, CUDA memory metrics should come from PyTorch.

### Training Phase Metrics

Namespace: `train/perf`

Required when training:

```text
train/perf/step_time_sec
train/perf/step_time_sec_rolling_mean
train/perf/sampling_time_sec
train/perf/batch_build_time_sec
train/perf/local_energy_time_sec
train/perf/forward_time_sec
train/perf/objective_time_sec
train/perf/backward_time_sec
train/perf/optimizer_step_time_sec
train/perf/post_step_metrics_time_sec
```

Recommended:

```text
train/perf/callback_time_sec
train/perf/checkpoint_time_sec
train/perf/samples_per_sec
train/perf/effective_walkers_per_sec
```

Notes:

- `step_time_sec` should remain the total enclosing step wall time.
- Phase metrics should approximately sum to less than or equal to
  `step_time_sec`. Any difference is unclassified overhead.
- `samples_per_sec` should use the number of configurations in the batch,
  not the number of electron coordinates.
- CUDA timing should optionally synchronize before timing boundaries when
  `timing.cuda_synchronize: true`. The default may remain `false`.

### Evaluation Metrics

Namespace: `eval/perf`

Required when evaluating:

```text
eval/perf/wall_time_sec
```

Namespace: `diagnostics/<task_name>` or `eval/perf/<task_name>` depending on
the existing logger convention.

Required per evaluation task:

```text
time_sec
```

Recommended per evaluation component:

```text
generator_time_sec
calculator/<calculator_name>_time_sec
summary/<summary_name>_time_sec
artifact_write_time_sec
```

Component-level evaluation timing is useful because final-eval cost can be
dominated by one task or one calculator. It should still be scalar and compact.

## Implementation Sketch

### Training

Add internal timing boundaries around the existing training-loop phases. The
lowest-risk path is to keep `TrainStepTiming` as the owner of total step time
and add phase timing inside `VMCTrainer.fit`, because the trainer owns the
phase boundaries.

Possible implementation shape:

```python
phase_times = {}

with timer("sampling", phase_times):
    walkers, sampler_stats = sampler.collect_samples(...)
with timer("batch_build", phase_times):
    batch = walkers.make_batch()
with timer("local_energy", phase_times):
    result = local_energy(...)
with timer("forward", phase_times):
    output = model(batch)
with timer("objective", phase_times):
    objective = compute_vmc_objective(...)
with timer("backward", phase_times):
    loss.backward()
with timer("optimizer_step", phase_times):
    optimizer.step()
```

Then log `phase_times` under `train/perf` on the same cadence as existing
training metrics. The timer helper should optionally call `torch.cuda.synchronize`
at boundaries when configured.

### Run Memory

Add a lightweight callback, for example `ResourceUsage`, with triggers:

```text
run_start
step_end
evaluate_end
run_end
exception
```

It should track peaks in memory and log final scalar values under `runtime`.
It may also log sparse step-level memory under `runtime/resources` if useful,
but the required artifact is the final peak.

### Evaluation

`Evaluator._evaluate_task` already has clear generator/calculator/summary
boundaries. Add optional scalar component timing there, or add events such as:

```text
generator_start / generator_end
calculator_start / calculator_end
summary_start / summary_end
```

The callback/event route is cleaner if we want timing behavior to remain
configurable. Direct scalar timing in the evaluator is simpler and probably
acceptable if the metrics are purely runtime metadata.

## Collection Contract

Collect stages should never flatten all timing/profiling records blindly.
Instead they should project configured keys into compact tables.

Recommended compact tables:

```text
resource_summary.csv
cost_by_run.csv
cost_by_axis.csv
cost_by_task.csv
```

### `cost_by_run.csv`

One row per run attempt:

```text
run_id
attempt_id
stage
device_type
status
wall_time_sec
peak_memory_mb
mean_step_time_sec
median_step_time_sec
p95_step_time_sec
mean_sampling_time_sec
mean_local_energy_time_sec
mean_forward_time_sec
mean_backward_time_sec
mean_optimizer_step_time_sec
samples_per_sec
```

Include configured axes as additional columns when available.

### `cost_by_axis.csv`

One row per group, where grouping axes are configured by the study:

```text
stage
axis_name
axis_value
n_runs
wall_time_sec_median
wall_time_sec_q25
wall_time_sec_q75
step_time_sec_median
local_energy_time_sec_median
forward_time_sec_median
backward_time_sec_median
peak_memory_mb_median
```

This table is the main answer to “which hyperparameter contributes to cost?”
It should be derived from compact metrics, not raw artifacts.

### `cost_by_task.csv`

One row per evaluation task or diagnostic:

```text
run_id
attempt_id
task_name
device_type
time_sec
generator_time_sec
calculator_time_sec
summary_time_sec
n_records
artifact_level
```

## Optional Deep Profiling

Use `torch.profiler` only for explicit profiling attempts. The default
experiment path should not emit profiler traces.

Recommended activation:

```yaml
profiling:
  enabled: false
  tool: torch_profiler
  schedule:
    wait: 1
    warmup: 1
    active: 3
  record_shapes: true
  profile_memory: true
  with_stack: false
  output_dir: ${run.dir}/profiles/torch
```

Profiler artifacts may include:

```text
profiles/torch/trace.json
profiles/torch/key_averages.csv
profiles/torch/summary.json
```

Only `summary.json` or selected scalar metrics should be collected by study
collectors. Full traces should remain in the individual run directory.

## Hydra Configuration

This spec does not require making experiment orchestration Hydra-based.

For SpENN run configs, timing controls can stay in the existing run config:

```yaml
timing:
  rolling_window: 20
  cuda_synchronize: false
  phase_timing: true

profiling:
  enabled: false
```

If experiment-level Hydra configs are introduced later, they should own:

```yaml
metric_projection:
  cost:
    keys:
      - runtime/wall_time_sec
      - runtime/peak_memory_mb
      - train/perf/step_time_sec
      - train/perf/local_energy_time_sec
      - train/perf/forward_time_sec
      - train/perf/backward_time_sec
```

The experiment Hydra config should remain separate from SpENN run configs unless
a shared schema becomes clearly useful.

## Cost and Risk

Low-cost pieces:

- Add scalar phase timers to the training loop.
- Add best-effort resource/memory callback.
- Add component timing around evaluation tasks.
- Add compact cost projection tables.

Higher-cost pieces:

- Accurate asynchronous CUDA timing without synchronization overhead.
- Slurm accounting integration across clusters.
- Operator-level PyTorch profiler analysis.
- Automated regression models for cost attribution.

Recommended first PR:

1. Add phase timing metrics for training.
2. Add runtime peak-memory metrics.
3. Extend final collect/resource summary to include timing phase medians.
4. Add tests using fake clocks and fake memory readers.

Recommended second PR:

1. Add evaluation component timing.
2. Add compact `cost_by_run.csv` and `cost_by_axis.csv`.
3. Add report plots or tables for cost, kept separate from quality ranking.

## Acceptance Criteria

- Existing metrics continue to be emitted unchanged.
- New metrics are scalar, JSON-safe, and namespaced.
- Training timing tests are deterministic with a fake clock.
- CUDA synchronization remains configurable and defaults to the current behavior.
- Collect reads only declared cost metrics, not arbitrary raw artifacts.
- Final reports can summarize runtime without parsing checkpoints or large
  diagnostic record CSVs.
