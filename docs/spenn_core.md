# TPEN Core Scaffold

The TPEN pipeline is the primary API (MIG-TPEN-000): all stages operate in
real space, and the two compute stages own their activations.

```text
Feature
  -> EquivariantMixing (owned pointwise Gamma)
  -> Interaction
  -> PathAggregation (owned Gamma_c)
  -> Update
  -> Updater
  -> Feature
```

Data/state objects live in owner modules under `tpen.data`, with real tuple
containers owned by `tpen.data.real`, tuple helpers owned by
`tpen.data.indices`, partition metadata owned by `tpen.data.partition`, and
permutation algebra and non-identity subset selection owned by
`tpen.data.permutation`. The traceable `EquivariantMap`, passive trace
recording, and runtime equivariance checkers live in `tpen.equivariance`.
Trainable or callable neural modules live in `tpen.nn`. Virtual-path
metadata lives in `tpen.data.paths`.

## Initialization RNGs

TPEN-owned randomized modules should use explicit initializer objects rather
than process-global RNG seeding. New configs should wire
`tpen.nn.TorchInitializer` into randomized model components, for example into
generated `Embedding` MLPs and `PathAggregation` weights. These initializers
materialize local `torch.Generator` instances and do not call
`torch.manual_seed`, `numpy.random.seed`, or `random.seed`.

`model.seed` is a legacy OmegaConf interpolation shim only. It may remain in old
configs so values like `${model.seed}` resolve into explicit initializer specs,
but `TPENWaveFunction(seed=...)` does not seed or initialize anything. New
configs should prefer a separate initialization seed field plus explicit
initializer wiring.

Runtime equivariance checks are checker-driven:
`tpen.equivariance.checks.FullModelEquivarianceChecker` and
`TraceEquivarianceChecker`, scheduled by `tpen.callback.RuntimeEquivariance`.
They call the normal model `forward`, select permutations via
`tpen.data.permutation.select_nonidentity_permutations`, permute values with
`apply_particle_permutation`, and compare via each value's typed `.compare(...)`.
`EquivariantMap` itself only computes and passively records traces; it does not
check equivariance.

Deleted legacy names should stay deleted on this branch:

- `SpechtMP`
- `FeatureDict`
- `MessageDict`
- `FusionMap`
- `BranchMap`
- `tpen/nn/real_space`
- `tpen/nn/spechtmp`
