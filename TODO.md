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
- [x] Added runtime-checking `spenn.nn.EquivariantMap` and reusable
      `spenn.testing.equivariance` helpers.
- [x] Added virtual-support path enumeration in `spenn/reps/paths.py`.
- [x] Added scaffold modules for `EquivariantMixing`, `SpechtActivation`,
      `Update`, `SpENNLayer`, `SpENNWaveFunction`, Fourier transforms, and
      `PfaffianReadout`.
- [x] Replaced legacy equivariance tests with runtime-check-oriented tests.
- [x] Migrated real tensor states to dense order-indexed block lists with
      zero-channel order-0 sentinels.
- [x] Migrated irrep tensor states to direct `Partition -> tensor` mappings and
      added tensor `validate()` methods.
- [x] Added optional `EquivariantMap` runtime tensor validation via
      `tensor_validation_check`.
- [x] Added `spenn.data.zero_block` for reserved zero-order real tensor blocks.
- [x] Recorded the orthogonal-basis convention on `spenn.reps.SpechtIrrep`.

## Next Core Work

- [ ] Implement the heavy tensor contraction in `EquivariantMixing`.
- [ ] Implement Fourier and inverse Fourier transforms over ordered tuple orbit
      coordinates.
- [ ] Extend `SpechtActivation` with scalar/sign and higher-dimensional gated
      activations that do not mix transforming irrep coordinates incorrectly.
- [ ] Replace the simple identity-channel `Update` with shared channel maps
      while preserving runtime equivariance checks.
- [ ] Add a production embedding from `ElectronBatch` to `RealFeature`.
- [ ] Expand `PfaffianReadout` tests for antisymmetry and odd-electron bordered
      kernels.
- [ ] Reintroduce experiments only after the new core API passes component-level
      equivariance tests.

## Validation Expectations

- Use `uv` for test commands.
- Force runtime equivariance checks in tests with `check_probability=1.0`.
- Keep runtime checks opt-in for expensive training paths, but make them easy to
  enable for debugging.
- Do not add compatibility exports for deleted names such as `SpechtMP`,
  `FeatureDict`, `MessageDict`, `FusionMap`, or `BranchMap`.
