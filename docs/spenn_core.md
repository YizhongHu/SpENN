# SpENN Core Scaffold

This restructuring branch makes the new SpENN pipeline the primary API:

```text
RealFeature
  -> EquivariantMixing
  -> RealInteraction
  -> FourierTransform
  -> IrrepInteraction
  -> SpechtActivation
  -> IrrepFeature
  -> InverseFourierTransform
  -> RealUpdate
  -> Update
  -> RealFeature
```

Data/state objects live in `spenn.data`, with real tuple containers owned by
`spenn/data/real_tensors.py` and irrep containers owned by
`spenn/data/irrep_tensors.py`. Trainable or callable neural modules live in
`spenn.nn`. Representation-theory helpers and fixed maps live in `spenn.reps`.
Runtime equivariance assertions live in `spenn.testing`.

Runtime equivariance checks are centered on `spenn.nn.EquivariantMap`. When
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
