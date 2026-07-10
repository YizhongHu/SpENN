# Release Notes

## v0.2.2 - Pair Stability v3 operations and checkpointing

### Added

- Final checkpoints are written at completed training steps.
- Pair Stability v3 stack runner (`submit_stack.sh`).
- Pair Stability v3 smoke gate reports submitted Slurm job IDs and uses
  estimated run time for smoke and full commands.

### Changed

- Pair Stability v3 launchers use per-CPU Slurm memory, and Submitit memory
  handling is corrected.
- Pair Stability collectors scope results to latest planned grids.
- Pair Stability 1B plotting and final reports are refined.
- Pair Stability documentation, smoke configuration, metric documentation, and
  test coverage align with the updated workflow.
- Local environment files are ignored.
- Contributor branch ownership and pull-request guidance are updated.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.2.2`.
- `uv run --extra cpu python -m compileall spenn run.py typechecked.py`
- Focused Pair Stability v3 and checkpoint callback tests: `66 passed`.

## v0.1.0 - Hooke benchmark integration

### Added

- Hooke two-electron benchmark physics, exact-solution fixtures, and smoke
  training examples.
- VMC runner infrastructure with config-root callbacks, logging, checkpoints,
  status files, health checks, and runtime equivariance diagnostics.
- Evaluation task infrastructure for Hooke, local-energy, trace, orbit, and
  sampler-based diagnostics.
- Hooke pair-validation and pair-stability study machinery, including
  planning, selection, final train/eval/collect/report stages, and Slurm/local
  launch support.
- Experiment documentation for the Hooke study workflows and reusable execution
  planning notes under `experiments/`.

### Changed

- The top-level run path now goes through `run.py` and
  `spenn.run.run_from_config`, with runner-owned training/evaluation logic.
- Hydra config compatibility is not guaranteed with pre-`v0.1.0` configs. The
  active contract is the `v0.1.0` config-root ownership model for callbacks and
  loggers.
- Package metadata now reports `spenn.__version__ == "0.1.0"`.

### Validation

- `uv run --extra cpu python -m compileall spenn run.py typechecked.py`
- `uv run --extra cpu pytest -q`

### Deferred

- A reusable experiment toolkit that fully separates planning, execution, and
  study-specific analysis remains future work.
- Dynamic heterogeneous job assignment is intentionally left out of this
  release; current pair-stability runs use static task tables plus claim-based
  execution where needed.
- Further modular scale-control generalization should stay in focused follow-up
  PRs rather than this integration PR.
