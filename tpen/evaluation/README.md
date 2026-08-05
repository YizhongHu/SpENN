# Evaluation Architecture Notes

Evaluation is organized as:

```text
Generator -> Calculator -> Summary
```

- Generators produce configurations and bookkeeping metadata.
- Calculators compute typed raw primitive quantities.
- Summaries reduce or serialize existing bundle contents into metrics/artifacts.
- Record writers are implemented as summaries for now.
- Summaries may emit artifacts, but they should not compute new scientific primitives.
- Task geometry lives in the generator, not in summaries.
- Task output directories are explicit in `EvaluationTask.output_dir`.
- Experiment configs should route task outputs under `${run.dir}/...`; relative
  task output dirs are resolved against `run_dir` by the evaluator.
- `EvaluationBundle` is not a generic dict and should not become a catch-all diagnostic container.
- Trace equivariance compares typed trace values through explicit `.permute(...)`
  and `.compare(...)` contracts; raw tensors are not permuted by rank guessing.
