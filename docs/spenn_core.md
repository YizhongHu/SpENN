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
metadata owned by `spenn.data.partition`, and permutation algebra owned by
`spenn.data.permutation`. Runtime equivariance assertion helpers and their
test schedule live in `spenn.testing.equivariance`. Trainable or callable
neural modules live in `spenn.nn`. Representation metadata, virtual paths, and
Fourier transforms live in `spenn.reps`.

Runtime equivariance checks are centered on `spenn.data.EquivariantMap`. When
enabled, small systems check every particle permutation. Larger systems check
adjacent transpositions and reversal. Tests should force checks with
`check_probability=1.0`.

The loop over all runtime permutations is owned by
`spenn.testing.equivariance.assert_equivariant_all`; `EquivariantMap` only
decides whether a forward pass should run the check.

Deleted legacy names should stay deleted on this branch:

- `SpechtMP`
- `FeatureDict`
- `MessageDict`
- `FusionMap`
- `BranchMap`
- `spenn/nn/real_space`
- `spenn/nn/spechtmp`
