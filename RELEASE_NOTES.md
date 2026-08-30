# Release Notes

## v0.4.0 - Analytic electron-nucleus cusp local energy

16 commits since v0.3.4 (`ed09f85`). The headline is an analytic evaluator for
the electron-nucleus cusp; the rest is the typed operator machinery it needed
and two compatibility changes.

### Breaking

- **Minimum Python is now 3.12** (#404). The declaration was the only thing
  lagging: `.python-version`, the docs workflow, and the `h5py` pin comment
  were already 3.12. `uv.lock` loses its sub-3.12 resolution forks, which is
  not a dependency upgrade - every dropped entry carried a
  `python_full_version < '3.11'` or `== '3.11.*'` marker and was already
  unselectable.
- **The Coulomb distance floor now defaults to `0.0`** (#403). The `eps`
  keyword is retained for backwards compatibility and documented as unproven.
  A non-zero floor removes the potential divergence and introduces one into
  the total; measurement found `E(r; eps) = Z/r - Z/eps - Z^2/2` inside the
  floor. Runs that relied on a non-zero default will change.

### Added

- **Analytic electron-nucleus cusp local-energy evaluator** (#410). `T` and
  `V_en` each diverge as `1/r` at coalescence and only their sum is finite, so
  they are evaluated together as
  `T + V_en = -1/2 sum_i [Lap_i f + |grad_i f + grad_i U|^2] + sum_iA g_A(r_iA)`.
  `g_A` is taken from the cusp provider's pre-cancelled `slope_residual`;
  nothing reconstructs `(u' + Z)/r` by subtraction, which would reintroduce
  the genuine `0/0`. Selected by one explicit config override. A slow
  reference implementation ships alongside as an independent executable
  oracle, matching the vectorised kernel in values and parameter gradients at
  `rtol = atol = 1e-12`.
- Typed analytic electron-nucleus cusp capability (#389) and model-owned
  regular/cusp factorization input (#391), which the evaluator consumes.
- Operators declare which physics they compute via a typed `OperatorId`
  (#405); typed context keys, providers, and a read-only context (#406); an
  operator lowering registry and local-energy planner (#407). The planner
  enforces exactly-once operator consumption structurally rather than by a
  downstream check.
- Physical-term and virial-residual metrics owned in `tpen/` (#386) and an
  optional Numdifftools finite-difference kinetic oracle (#388).
- Baseline matrix plan builder (#387), `StagePlanV2` dispatch via
  `ParslAttachExecutor`, DeepQMC environment ported to Polaris (#385), and a
  reusable multi-GPU scaling probe (#392).

### Known limitations

- Cross-device geometry comparison is untested; `CUDA_AVAILABLE` was false on
  every verification node.
- `torch.equal` raises for `chalf`, `float8_e5m2` and `float8_e8m0fnu` even at
  matching dtype, so the geometry comparator can raise from a path annotated
  `-> bool`. These dtypes have no plausible use for nuclear geometry.

## v0.3.4

### Added

- OneQMC/Orbformer results adapter, emitting baseline records from `result.h5` (#290)

### Changed

- Baseline records reported against published values (#335)
- OneQMC adapter: `--allow-short-tail` help no longer asserts a resolution rule that was removed (#294)
- Timing callback stack (#311, #312) was reverted before release after its cadence test conflicted with the configured callback cadence.

### Fixed

- **Config identity is semantic rather than byte-based** (#379). A YAML reorder whose parsed form is identical previously changed the config hash, which failed all 42 identity joins of a completed, archived study and took it to `n_pass=0`. Identity is now computed over the canonicalised document, so an inert reformat leaves it unchanged while a genuine semantic edit still changes it. Receipts written under the old byte-based scheme stay joinable through an explicitly classified legacy path that refuses an identity it cannot account for, and a missing historical revision now fails one row instead of aborting the whole collection. `PyYAML` is declared directly rather than relied on transitively.

## v0.3.3 - Trajectory energy estimand correction

34 commits since v0.3.2 (`dd49cc1`). Two are `dev`/`main` merges; the rest
fall into three areas: the He-v1 42-row evaluation study, the He-cutover
multi-node Polaris port, and agent/cluster infrastructure.

### Added

**He-v1 evaluation study**

- Factor-response evaluation: scale the trained envelope parameters (`b_ee`,
  `c_en`, `d_en`) in a bounded context with byte-verified restore (#337)
- Evaluation rows declare their own task graph; the 42-row evaluation grid is
  frozen (#339)
- The frozen grid's metric-namespace bindings are carried into the plan
  manifest, so a tolerance cannot land on a different task that happens to
  emit the same metric name (#348)
- Production evaluation rows are placed on `kozinsky_gpu` and `seas_gpu`
  together in one job (#349)

**He-cutover: multi-node execution on Polaris**

- Parsl attach executor for allocation-attached batches, with immutable
  dispatch receipts and completion verification (#354)
- He cutover study: two-facility smoke plan, runtime admission,
  allocation-local Parsl pipeline, facility templates, and contract tests
  (#358)
- Attempt-free dispatch seam (#352)
- Full He cutover plan, production grid, and Polaris production template
  (`4410890`; this commit has no PR number in its subject or body)
- Optional `nodes_per_block` on the dispatch context (#367)
- Placement and topology evidence for multi-node runs: which nodes and which
  physical GPUs actually ran (#369)
- Proof manifest of 1 train + 40 eval rows (#371)

### Changed

- **The He-v1 study's canonical energy is now the whole-trajectory estimate
  with its MCSE.** The final-draw snapshot is retained beside it and explicitly
  labelled as a snapshot; `plateau_reached` travels with the MCSE, because a
  Geyer sequence truncated at the window edge understates it. A row whose
  trajectory statistics never resolved renders `absent` and never falls back
  to the snapshot (#373)
- Duplicate row submission is refused before the allocation rather than
  inside it (#357)
- Legacy Hooke studies marked stale with provenance banners; legacy toolkit
  deprecation recorded (#351)
- Running the authoritative-edit guard outside the agent sandbox is sanctioned
  in AGENTS.md (#338)

### Fixed

**He-v1 collector and planner**

- Collection completeness is derived from the plan, not a row-count literal
  (#342)
- Term energies are included in re-equilibrated records, and the planner no
  longer claims two rows (#343)
- Metrics-only evaluation tasks are no longer required to emit an artifact
  (#345)
- The summary-field oracle unwraps delegating calculators (#346)
- The collector's numeric oracles are boundary-sensitive (#341)
- Evaluation task order restored (#375)
- Test files excluded from the import-boundary check (#376)

**He-cutover and Parsl**

- `AllocationContext` accepts empty `visibility_values` in inherit mode (#355)
- he-cutover entrypoints are importable (#359)
- Polaris Torch bootstrap paths repaired (#360)
- Polaris operator overlay protected (#361)
- Parsl dataflow kernel reused across batches (#362)
- Virtualenv interpreter preserved in dispatches (#363)
- The plan owns `result_dir`, so completion probes the path the runner
  actually writes (#364)
- A bad `PBS_NODEFILE` is rejected before Parsl loads, removing a localhost
  false-green (#368)
- Multi-node `LocalProvider` branch; GPUs are no longer assigned by dispatch
  index (#370)

**Cluster tooling**

- Cannon `uv` resolved outside `PATH` (#374)

## v0.3.2 - Current-dev baseline release

Covers the exact 29-commit `origin/main..origin/dev` range ending at
`d04a96fbd882d5c101b1bcadf28e57fc51ec3e11`.

### Added

- Baseline-comparison coverage now includes DeepQMC and Neural Pfaffian
  adapters, shared estimator statistics, an explicit B/N/N2 system registry,
  and the declared `h5py` dependency needed to execute the DeepQMC adapter
  tests.
- Artifact lifecycle schema and lifecycle-gated run-data-cleanup guidance,
  together with the Cannon archive-experiment skill.
- Row-aligned trajectory records; conditioned-energy, sampler-health, and
  cost diagnostics; and expanded symmetry trace records with repaired trace
  identity contracts.
- He-v1 checkpoint replay semantics, reconciled helium atlases, and a minimal
  real-checkpoint evaluation canary with its collection-contract repair.

### Changed and fixed

- Baseline emission now rejects unmeasured, zero, or otherwise unassessable
  error bars; protects short-tail handling; reports measured tail windows; and
  supports an explicit operator caveat for DeepQMC records.
- The below-floor fallback decision and FermiNet attribution guidance are
  documented, and baseline fixtures exercise executable DeepQMC coverage and
  measurable Neural Pfaffian tail variation.

### Excluded

- Closed, unmerged diagnostics I1/#325 and I2/#326 are not included in this
  release, nor is their V1 canary/gate. This section inventories only the
  current `origin/main..origin/dev` baseline above; it makes no claim about
  those excluded diagnostics, generated results, tags, or merges.

## v0.3.1 - Generic atom API, helium study infrastructure, and correlation-aware statistics

Covers the full `v0.3.0..a9f9f7fa` delta of 55 commits.

### Added

- Generic atom API and nuclear geometry, landed as the eleven-PR atom stack
  (#258-#274): an immutable `AtomicConfiguration`, its transport through the
  sampler and checkpoints, batch-owned electron-nucleus data, an explicit
  nuclear factorization seam, generic Coulomb potentials, nucleus-nucleus
  repulsion, and the cusp relocation into `tpen/nn/cusp.py`.
- Helium study infrastructure: the He v1 config (#247), radial diagnostics
  (#249), tolerance gates over emitted cusp and tail metrics (#263), the study
  driver (#264), the trajectory consumer (#265), and the log-amplitude oracle
  with a singlet-purity diagnostic (#262).
- Fixed-model statistics: the trajectory IAT/ESS/MCSE producer (#257), which is
  the correlation-aware uncertainty machinery the He study's headline bar depends
  on, with multi-timescale and heterogeneous-chain coverage for the estimator
  (#278).
- Timing and cost accounting: emitted boundary stamps (#240), the device event
  timer (#243), cost provenance (#246), delivered-allocation GPU-seconds (#250),
  warmup exclusion from throughput projection (#254), and per-axis GPU-second
  rollups (#256).
- NN-QMC baseline comparison surface: the survey (#216), the comparison-system
  registry with provenance (#217), the common results record and run collector
  (#218), the transcribed He reference energy (#220), the reference-energy
  correction with H2 (#232), and the FermiNet run-directory adapter (#233).
- Hooke scan work: the four-level basis choice library (#225), the scan
  train/eval slot surface (#226), the `tpen-pair-scan-v1` fork with an enforced
  choice-library merge (#227), and checked-in scan grids with split-sample
  champion selection (#228).
- The frozen He-v1 production arm is predeclared and checkable rather than
  described (#279).
- `tests/unit/test_version_agreement.py` asserts that all four version sources
  agree, reading each from its own file at runtime.

### Changed

- The cusp law is named by its functional form rather than by its trainability
  (#280).
- The unused `AsymptoticDecay` interface is removed (#281).

### Fixed

- Batched Pfaffian evaluation over walkers and channels (#234).
- Antisymmetry is reported from summary metrics at `artifact_level: summaries`
  (#230).
- The inverse softplus is stable above the float64 overflow boundary (#277).
- Shipped he-v1 artifacts no longer claim a byte identity the check never
  established (#282).

### Validation

- Commit delta measured at the release-prep SHA rather than reused: `v0.3.0..a9f9f7fa`
  = 55 commits, which is 50 at the earlier `dev` tip `515231a1` plus exactly the
  five merged layers #278-#282. `origin/main..dev` = 37 was NOT used as the
  release figure: `main` already carries two post-tag merges, so 37 would
  understate the release by 18 commits. `dev..origin/main` = 0, so `main` remains
  a strict ancestor and the promotion stays fast-forwardable.
- Package metadata, runtime metadata, the `uv.lock` root `tpen` entry, and the
  README current-release text all agree on `0.3.1`, asserted executably rather
  than by inspection. `uv.lock` is load-bearing here, not generated noise: the
  cluster environment is built with `uv sync --extra cpu --locked`, which fails
  closed, so a stale lock entry stops every scheduler job from starting.
- `uv.lock` was regenerated with `uv lock`; the resulting diff touches only the
  root `tpen` version line and no dependency, hash, or resolution marker.
- Five-stage he-v1 smoke on merged `dev`: Cannon Slurm `39726909`, partition
  `test`, `COMPLETED 0:0`, `00:05:48`, pinned in-job to `a9f9f7fa` and tree
  `a246b88f`. Composed `2283 passed`, `8 skipped`, `6 subtests passed`; the
  study-local suite gave `227 passed`. Restore was exercised across both the
  renamed Hydra `_target_` of #280 and the added `readout.channel_weights` key
  of #279, the cross-check mutant went RED and restored clean, and
  `assess_convergence` refused with verdict `indeterminate` without narrowing
  windows.
- That smoke installed `tpen` as `0.3.0` in-job, so it is a PRE-BUMP
  measurement. It establishes that the tree being bumped was releasable; it does
  not validate the version change itself, and it predates the version-agreement
  test.

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
