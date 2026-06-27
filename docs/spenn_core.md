# SpENN Core Scaffold

This restructuring branch makes the new SpENN pipeline the primary API:

```text
RealFeature
  -> EquivariantMixing
  -> RealInteraction
  -> FourierTransform
  -> IrrepInteraction
  -> ActivationByType
  -> IrrepInteraction
  -> PathAggregation
  -> IrrepFeature
  -> InverseFourierTransform
  -> RealUpdate
  -> Update
  -> RealFeature
```

Data/state objects live in owner modules under `spenn.data`, with real tuple
containers owned by `spenn.data.real`, irrep containers owned by
`spenn.data.irrep`, tuple helpers owned by `spenn.data.indices`, partition
metadata owned by `spenn.data.partition`, and permutation algebra and
non-identity subset selection owned by `spenn.data.permutation`. The traceable
`EquivariantMap`, passive trace recording, and runtime equivariance checkers
live in `spenn.equivariance`. Trainable or callable neural modules live in
`spenn.nn`. Representation metadata, virtual paths, and Fourier transforms live
in `spenn.reps`.

## Initialization RNGs

SpENN-owned randomized modules should use explicit initializer objects rather
than process-global RNG seeding. New configs should wire
`spenn.nn.TorchInitializer` into randomized model components, for example into
generated `Embedding` MLPs and `PathAggregation` weights. These initializers
materialize local `torch.Generator` instances and do not call
`torch.manual_seed`, `numpy.random.seed`, or `random.seed`.

`model.seed` is a legacy OmegaConf interpolation shim only. It may remain in old
configs so values like `${model.seed}` resolve into explicit initializer specs,
but `SpENNWaveFunction(seed=...)` does not seed or initialize anything. New
configs should prefer a separate initialization seed field plus explicit
initializer wiring.

Runtime equivariance checks are checker-driven:
`spenn.equivariance.checks.FullModelEquivarianceChecker` and
`TraceEquivarianceChecker`, scheduled by `spenn.callback.RuntimeEquivariance`.
They call the normal model `forward`, select permutations via
`spenn.data.permutation.select_nonidentity_permutations`, permute values with
`apply_particle_permutation`, and compare via each value's typed `.compare(...)`.
`EquivariantMap` itself only computes and passively records traces; it does not
check equivariance.

Deleted legacy names should stay deleted on this branch:

- `SpechtMP`
- `FeatureDict`
- `MessageDict`
- `FusionMap`
- `BranchMap`
- `spenn/nn/real_space`
- `spenn/nn/spechtmp`
