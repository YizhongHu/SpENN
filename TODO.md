# SpENN Core Restructure TODO

Source of truth for this branch is the PR brief for `codex/restructure` and the
new scaffold documented in package docstrings.

## Completed In This Checkpoint

- [x] Removed the legacy SpechtMP, branch/fusion, and real-space compatibility
      modules.
- [x] Removed Hooke experiments, generated figures, and legacy root configs
      from this restructuring branch.
- [x] Added direct data state names:
      `RealFeature`, `RealInteraction`, `IrrepInteraction`, `IrrepFeature`,
      `RealUpdate`, `EquivariantState`, `ElectronBatch`, and
      `WavefunctionOutput`.
- [x] Added `spenn/data/indices.py` as the home for tuple-index bookkeeping.
- [x] Added runtime-checking `spenn.data.EquivariantMap` with runtime
      equivariance assertion helpers in `spenn.testing.equivariance`.
- [x] Added virtual-support path enumeration in `spenn/reps/paths.py`.
- [x] Added scaffold modules for `EquivariantMixing`, `PathAggregation`,
      `Update`, `SpENNLayer`, `SpENNWaveFunction`, Fourier transforms, and
      readouts.
- [x] Replaced legacy equivariance tests with runtime-check-oriented tests.
- [x] Migrated real tensor states to dense order-indexed block lists with
      zero-channel order-0 sentinels.
- [x] Migrated irrep tensor states to direct `Partition -> tensor` mappings and
      added tensor `validate()` methods.
- [x] Added optional `EquivariantMap` runtime tensor validation via
      `tensor_validation_check`.
- [x] Added `spenn.data.real.zero_block` for reserved zero-order real tensor
      blocks.
- [x] Recorded the orthogonal-basis convention on `spenn.reps.SpechtIrrep`.
- [x] Added deterministic `PathMetadata` and saved
      `spenn/cache/paths_canonical.json` plus
      `spenn/cache/paths_full.json`.
- [x] Added `IrrepMetadata` JSON/cache reader support and saved
      `spenn/cache/irreps.json`.
- [x] Added small-order orthogonal Specht representation fixtures and saved
      `spenn/cache/irreps_m3.pt`.
- [x] Implemented slow reference `EquivariantMixing` with `sum` and
      `completion_mean` aggregation.
- [x] Added a path-by-path vectorized `EquivariantMixing` implementation and
      tests against the slow oracle.
- [x] Made default `EquivariantMixing` load saved path metadata cache files when
      they cover the requested order bounds.
- [x] Added executable Fourier/inverse Fourier scaffold transforms with
      shape-correct orthogonal-basis irrep tails.
- [x] Split irrep activation into the `Activation` template,
      `ActivationByType`, and learned `PathAggregation` so activation keeps
      the path axis before aggregation.
- [x] Added direct runtime equivariance coverage for `ActivationByType` and
      `PathAggregation`.
- [x] Added direct orthogonal-coordinate action coverage for
      higher-dimensional activation and path aggregation.
- [x] Moved runtime equivariance assertion helpers into
      `spenn.testing.equivariance`.
- [x] Consolidated particle-axis permutation, tuple-axis permutation, and tuple
      particle-count checks under `spenn.data`; runtime test schedules live in
      `spenn.testing.equivariance`.
- [x] Added real-space update strategy scaffolds:
      `ReplaceUpdate`, `ResidualUpdate`, and `NormGatedUpdate`.
- [x] Added default-off readout trainability flags for readout weights.
- [x] Moved real update shape compatibility validation into `spenn.data`.
- [x] Moved common real tensor batch, dtype, and particle-count checks into
      `spenn.data`.
- [x] Made `Update` the shared equivariant-map base for update strategies and
      kept shared per-order channel maps in `ChannelMappedUpdate`.
- [x] Added a trainable per-order MLP `Embedding` from non-repeating
      `ElectronBatch` particle-vector tuples to `RealFeature`.
- [x] Moved reusable neural modules to top-level `spenn.nn` modules and
      removed the old utility package from the tracked API.
- [x] Split batch state containers and geometry helpers under
      `spenn.data.batch`.
- [x] Split real and irrep tensor states into `spenn.data.real` and
      `spenn.data.irrep` owner packages, with no legacy tensor shims.
- [x] Replaced the diagonal Fourier scaffold with cache-backed slot Fourier
      transforms over ordered tuple orbit coordinates.
- [x] Added a SageMath-only irrep tensor-cache generator path for Specht
      representation fixtures.

## Next Core Work

- [x] Expand `PfaffianReadout` tests for antisymmetry and odd-electron bordered
      kernels.
- [x] Regenerate checked-in irrep cache files with SageMath on a Sage-enabled
      node before experiment branches.
- [x] Clean up all stale helpers and remove them. Backwards compatibility is
      not required on this branch; prefer deleting unused aliases, shims, and
      legacy helper paths aggressively.
- [ ] Reintroduce experiments only after the new core API passes component-level
      equivariance tests, in a separate experiment branch.

## Validation Expectations

- Use `uv` for test commands.
- Force runtime equivariance checks in tests with `check_probability=1.0`.
- Keep runtime checks opt-in for expensive training paths, but make them easy to
  enable for debugging.
- Do not add compatibility exports for deleted names such as `SpechtMP`,
  `FeatureDict`, `MessageDict`, `FusionMap`, or `BranchMap`.
