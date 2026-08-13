# Release Notes

## v0.3.0 - TPEN public API, typed event clock, and four-facility verification

### Added

- Typed occurrence foundation: a typed callback cadence, typed occurrence
  records, and delivery of a domain's own state object to typed callbacks, with
  the four health observers moved onto typed state delivery.
- Typed training phases with explicit update outcomes, and a typed
  `SamplerStats` record in place of dictionary probing.
- Durable checkpoint update state on `tpen.checkpoint` schema v2, with an
  update-driven checkpoint cadence.
- Portable pair-v1 allocation launcher
  (`experiments/hooke/tpen-pair-v1/launch.py`) that runs one task per explicitly
  bound accelerator through `AllocationPoolExecutor`, using the scheduler job ID
  as the allocation identifier and pass-scoped row claims that are never
  released.
- Pinned `rocm71` Torch extra so OLCF Frontier resolves an AMD ROCm stack.
- Authoritative-edit guard (`tools/check_authoritative_edit.py`) and a native
  unbounded linear pull-request stack workflow.

### Changed

- **Breaking:** the public runtime surface completes the SpENN to TPEN rename.
  Importers must use `tpen`, and the default run name no longer falls back to
  the former branding.
- The accelerator backend is resolved from the runtime instead of naming CUDA
  directly, so CUDA, Intel XPU, and ROCm share one code path.
- A whole training record is attributed to a single model version.
- The shell stack launcher is removed in favour of the tracked Python launcher.

### Fixed

- Launcher and receipt truthfulness: allocation receipts are required, task
  failures propagate to the process exit code, malformed execution receipts are
  rejected instead of silently accepted, evaluation failures are reported
  truthfully, completion truth is enforced, and any nested dataclass field is
  serialized rather than only `Event` and `Operation`.
- Durable run identifiers are normalized and required to be non-empty before
  they reach artifact paths.
- Test-time logger isolation: `tpen.bootstrap` no longer leaks its private
  terminal handler or `propagate=False` between tests, so the bootstrap-channel
  assertion no longer depends on test order. Runtime logging is unchanged.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.3.0`.
- Four-facility same-source hardware gate at tree
  `585df3187b14a017e23b6cf0d1cdb23970218125`, one fresh detached checkout and
  exactly one scheduler job per facility: Cannon CPU `38807293` (`1004 passed`,
  `2 skipped`) and GPU `38807314` on one A100 MIG; Polaris PBS `7438353`
  (A100-SXM4-40GB); Aurora PBS `8751664` (Intel Max 1550 tile, flat device
  hierarchy); Frontier Slurm `5250416` (AMD Instinct MI250X). Every accelerator
  lane produced identical science: `95` metric records, `2310` recursively
  finite numeric leaves with zero non-finite, `25/25` data-integrity checks,
  five COMPLETE `tpen.checkpoint` v2 manifests, `505` typed `tpen.*`
  occurrences, exactly one `run_start` and one `run_end`, and no `spenn`
  leakage.
- Merged integration tree `41004485cb61d5d60584134bc8650314ebcd34fa`
  re-verified on Cannon as Slurm `38838638` (`COMPLETED 0:0`): the accepted
  baseline invocation reproduced `1004 passed`, `2 skipped`; the full `tests`
  tree gave `1014 passed`, `2 skipped`, `6 subtests passed`; and the
  authoritative-edit guard alone gave `10 passed`, `6 subtests passed`.
- This release commit adds only version metadata and these notes on top of the
  verified tree above.

## v0.2.3 - Pair Stability v3 diagnostics and durable execution

### Added

- Sampled evaluation records can retain named local energies for individual
  Hamiltonian terms.
- Pair Stability v3 can archive a bounded, checkpoint-free study snapshot from
  final-report provenance.

### Changed

- Hooke singlet opposite-spin cusp range is fixed at `b=0.25` in the training
  and validation model configurations, with an explicit named override for
  controlled ablations.
- Stack workers retain their source checkout paths, and concurrent Submitit
  chunks serialize `uv sync` outside the checkout to preserve clean run
  provenance.

### Validation

- `uv lock --check`
- Package and runtime metadata agree on `0.2.3`.
- Focused local-energy record tests: `10 passed`.
- Pair Stability v3 GPU smoke stack controller `31658932` completed scan train
  and validation (`64/64` each) plus final train and evaluation (`8/8` each).


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
