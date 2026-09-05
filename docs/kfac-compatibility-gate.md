# P2: KFAC compatibility gate

P2 evaluates admission of `gpauloski/kfac-pytorch` at immutable commit
`5987766a43739de7eb950f564da54559f2504579` (version 0.4.2). A negative verdict
completes this gate successfully and keeps P3, the adapter, closed. This slice
does not implement an adapter or authorize native KFAC or linear-method work.

## Criteria declared before implementation

All criteria are conjunctive. An unsupported parameter, convention mismatch,
or unproved requirement prevents admission. Partial support is not a pass.

| Criterion | Required evidence |
| --- | --- |
| C1: semantic coverage | Every trainable scalar in representative embedding (ordinary and seeded), tensor, unary and composite mixing, aggregation, updater, readout, cusp/Jastrow and envelope families has exactly one named Kronecker or exact scalar block. Missing, overlapping or changed ownership fails. |
| C2: VMC convention | Explicitly frozen log-score normalization and centered/uncentered curvature choice; per-family factors agree with the corresponding dense per-sample reference. Energy-weighted objective gradients cannot substitute for score sensitivities. |
| C3: counts | Reduce unnormalized second-moment sums and represented counts, then normalize once. Match the concatenated-sample oracle for equal, unequal and empty local shards; an empty global sample set cannot update. |
| C4: public APIs | All required registration, factor and state behavior uses public upstream extension points. Non-public dependencies are grounds for rejection. |
| C5: restart | Inventory and preserve factors, exact scalar blocks, damping, EMA, schedules, refresh counters, inverse/eigen caches and their age, plus semantic/convention identity. Checkpoints on both sides of factor/inverse refresh boundaries reproduce uninterrupted next updates exactly on the accepted topology. |

The investigation may use an uncentered toy linear model to expose upstream
behavior; this does not choose TPEN's scientific curvature convention. Centered
whole-parameter scores and independently centered Kronecker factors are different
objects. No pass can be inferred from a toy linear control alone.

P2's acceptance contract excludes training and cluster work. The governing review
workflow assigns independent cluster test design and execution to the durable
reviewer. DDP admission additionally requires DG0 and DP1–DP3; local arithmetic
over shards is not a distributed-runtime receipt.

## Provenance

- TPEN baseline: `7ee49320443a5ddd8315587e706081805186d1e0` (`origin/dev`
  when this layer entered work).
- Design: `2026-09-04-helium-vmc-optimizer-adoption-analysis.md`, sections 5
  and 13.5 (operator's source analysis).
- Task Orchestrator: P2 `ec1033d7-0bf2-4373-81ae-4842dd8a3677`;
  predecessor F5 `2171fb45-26d6-42a6-93ed-03984986997c` is terminal.
- [Pinned upstream source](https://github.com/gpauloski/kfac-pytorch/tree/5987766a43739de7eb950f564da54559f2504579).

The decision and reproducible probe are recorded below after inspection; the
criteria above remain fixed.
