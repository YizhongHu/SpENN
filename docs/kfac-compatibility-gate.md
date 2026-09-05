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

## Decision: NOT ADMITTED

**2026-09-05, source-based negative.** P3 must remain closed. The pinned package
does not satisfy C1, C3 or C5 through its stock path, and a complete public-API
extension satisfying all five criteria has not been established. C2's TPEN
curvature choice is also not frozen by this slice's authorization. There is no
partial compatibility pass, automatic optimizer substitution, dependency addition
or runtime enablement in this change.

This is an admission decision at the declared gate, not a proof that every
possible adaptation is impossible. Public extension points exist and deserve
accurate credit. A later proposal would need a separately authorized compatibility
proof before P3 could open; implementing an adapter in order to claim this gate
passed would invert the required staging.

The table below evaluates the **full TPEN compatibility criteria**, including
the proposed public extension, rather than only the stock implementation.
Each criterion uses exactly one of `PASSED`, `FAILED`, or
`UNESTABLISHED-pending-reviewer-cluster-evidence`. **None is PASSED.** The third
state is non-passing and cannot authorize P3. Pending reviewer evidence also
includes any missing convention decision or extension proof; it does not promise
that merely executing this diagnostic can satisfy the criterion.

| Criterion | State | Evidence available and what remains unestablished |
| --- | --- | --- |
| C1: semantic coverage | UNESTABLISHED-pending-reviewer-cluster-evidence | Source refutes stock coverage: required TPEN families are skipped. The candidate census has not run, and complete custom Kronecker/exact-scalar coverage has not been established. |
| C2: VMC convention | UNESTABLISHED-pending-reviewer-cluster-evidence | No operator-frozen TPEN centered/uncentered choice or complete per-family score oracle. The analytic toy example separates conventions; it establishes no TPEN factor compatibility. |
| C3: counts | UNESTABLISHED-pending-reviewer-cluster-evidence | Algebra refutes stock rank/microbatch averaging. Sum-plus-count factors under unequal and empty shards remain unestablished under real multi-rank execution; no count-aware extension or global-empty behavior was exercised. |
| C4: public APIs | UNESTABLISHED-pending-reviewer-cluster-evidence | Public mappings, helpers and state properties exist in source. Every required TPEN registration, factor and state path has not been qualified through those APIs. |
| C5: restart | UNESTABLISHED-pending-reviewer-cluster-evidence | The source-derived cache-age example refutes stock reload semantics. Exact complete-state restart across factor and inverse refresh remains unestablished under real multi-rank execution; neither the toy diagnostic nor whole-TPEN restart was run. |

The narrower claims about **stock** C1 coverage, C3 averaging and C5 reload are
`FAILED` by the source counterexamples below. They do not refute every possible
public extension, so they are not relabeled as failures of all possible
adaptations. Source inspection and local structural/arithmetic controls, even if
executed successfully later, cannot promote the full C3 or C5 criterion to
`PASSED` without the required reviewer cluster evidence.

### Semantic ownership census

The probe builds ordinary-embedding and seeded-embedding variants of a whole
`TPENWaveFunction`, with order-1/order-2 composite unary and tensor producers,
path aggregation, channel-mapped updater, trainable Pfaffian readout, trainable
electron-electron ranges, trainable electron-nucleus curvature law and trainable
Gaussian confinement. It enumerates actual named parameters and aliases and
compares them with upstream `register_modules()` results.

| Family / TPEN source | Declared candidate partition | Stock outcome from source |
| --- | --- | --- |
| Ordinary embedding, `tpen/nn/mlp.py` | One augmented Kronecker block per affine map, weight and bias together | `nn.Linear` leaves are eligible; coverage alone says nothing about VMC factors. |
| Seeded embedding, `tpen/nn/initialization.py` | Same affine partition | `SeededLinear` inherits `nn.Module`, so automatic registration skips it. |
| Unary / tensor mixing, `tpen/nn/linear_equivariant_mixing.py`, `equivariant_mixing.py`, `mixing_kernel.py` | One candidate Kronecker block per named path tensor; flattening, parameter reuse and represented tuple counts need proof | Raw `ParameterList` / `ParameterDict` weights are skipped. |
| Composite mixing, `tpen/nn/composite_mixing.py` | Preserve the distinct named producer blocks; no duplicate composite owner | Composition calls producer `forward_pre_activation()` directly; a producer `Module.__call__` hook cannot be assumed to run. |
| Aggregation / updater, `tpen/nn/path_aggregation.py`, `update.py` | Candidate Kronecker partitions per named order tensor; channel/path semantics need proof | Raw parameter dictionaries are skipped. |
| Pfaffian readout, `tpen/nn/readout/pfaffian.py` | One exact scalar block per indexed `channel_weights` element | Direct vector parameter is skipped. |
| Cusp/Jastrow and confinement, `tpen/nn/cusp.py`, `envelope.py` | One exact scalar block per trainable raw coefficient/range | Direct parameters are skipped; fixed buffers are outside the trainable census. |

These are explicit **candidate obligations**, not implemented curvature blocks.
The JSON census never labels a skipped parameter covered merely because a block
name can be invented. Unknown families and undeclared aliases are rejected. This
representative census cannot qualify arbitrary custom user modules or a changed
parameter layout; those would require a fresh ownership audit.

Upstream evidence: [`get_flattened_modules`, `get_module_helper`,
`register_modules`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/register.py).
The recognized type tuples contain `nn.Linear` and `nn.Conv2d`; a missing helper
causes the registration loop to continue.

### Factor and convention counterexamples

Let the activation rows be `1, 3, 5, 7`. Their unnormalized second-moment sum is
84 and their represented count is 4. The concatenated A factor is therefore 21.
Split them into shards `[1]` and `[3, 5, 7]`: the local factors are 1 and `83/3`,
whose unweighted mean is `43/3`, not 21. Adding an empty shard with zero factor
makes the rank mean `86/9`. The correct reduction remains `(1 + 0 + 83)/(1+0+3)`.
Both A and G must carry their own represented sums/counts, with their relationship
to physical VMC samples established for reused tuple/path parameters.

The same issue occurs before distribution: `save_layer_input()` accumulates
already normalized helper factors; `update_a_factor()` divides by the number
of calls, not represented rows. At EMA decay 1/2 with the upstream identity
initialization, the unequal-microbatch A factor becomes `23/3` instead of 11.
The probe exercises these public methods and also includes a one-shard control
and equal-count control. It performs no collectives; averaging local matrices is
a witness for the reduction formula, not a world-size-2/3 integration test.

For score-convention separation, choose `a=(1,3)` and `g=(2,4)`, so the scalar
parameter scores are `s=a*g=(2,12)`. Then:

| Quantity | Value |
| --- | ---: |
| Raw score second moment `E[s²]` | 74 |
| Uncentered KFAC approximation `E[a²] E[g²]` | 50 |
| Centered score covariance `E[(s-E[s])²]` | 25 |
| Independently centered factors `Var(a) Var(g)` | 1 |

KFAC's factorization is an approximation even before centering. C2 requires the
dense oracle for the *declared approximation*, not arbitrary equality between
KFAC and the full Fisher. No geometric convention is selected by these numbers.

Sources: [`get_cov`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/utils.py),
[`LinearModuleHelper`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/modules.py),
and [`save_layer_input`, `update_a_factor`, `reduce_a_factor` / G counterparts](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/base.py).
Factor reduction requests `average=True` after local normalization.

### Public extension and state inventory

`BaseKFACPreconditioner(layers=..., assignment=..., tdc=...)` accepts an explicit
module-to-layer mapping. `ModuleHelper` and the layer classes expose public
factor, gradient and state methods; `KFACEigenLayer` exposes public `qa`, `qg`,
`da`, `dg`, `dgda` properties. The probe uses these public APIs only. In
particular, `LinearModuleHelper(SeededLinear(...))` is a useful bounded control
for the affine formula even though automatic registration skips that module.
It is not a TPEN score-hook or whole-model compatibility proof.

| State | Stock checkpoint behavior | Required before admission |
| --- | --- | --- |
| A/G EMA factors | Layer state contains A/G; initialization uses identity and the configured decay | Named factors, dtype, ownership and estimator convention |
| Step; factor/inverse cadence | Step plus non-callable cadence values saved | Exact schedule position and refresh age |
| Damping, decay, learning rate, KL cap | Non-callable values saved; callable schedules omitted | Frozen schedules and complete method policy |
| Inverse/eigen caches | Not in layer state; default preconditioner load recomputes | Preserve cache and age, or prove exact equivalent reconstruction |
| Pending microbatch statistics/counts | Not in stock layer state | Restrict checkpoints to quiescent update boundaries or serialize accumulations |
| Scalar-block state; semantic/convention fingerprint | Not supplied for TPEN | Method-owned codec with validation before mutation |
| Base optimizer, sampler/RNG, runtime topology | Outside the preconditioner state | Complete TPEN restart envelope; later DDP-owned receipts |

The restart counterexample needs only a scalar linear map. Use factor cadence 1,
inverse cadence 4, EMA decay 1/2, damping 0.1, and constant output sensitivity 1.
Step 0 with activation 1 gives A=1 and an inverse for A=1. Step 1 with activation
3 updates A to 5 but retains the old inverse. Save after those two completed
steps. The stock checkpoint contains A=5, G=1 and step=2. Default reload
recomputes the inverse for A=5. Step 2 with activation 5 updates the factors
again but does not refresh either inverse. The live update uses denominator
1.1 while the resumed update uses 5.1. This is a cache-age mismatch, not a
floating-point nondeterminism claim.

The probe uses fresh model/preconditioner/SGD objects, deep-copies public state,
and compares the next two parameter and serialized-state updates for checkpoint
positions 1, 2, 3, 4 and 5. Positions surround the inverse refresh at zero-based
step 4; factors refresh each step. It uses an uncentered linear score control,
not a physical VMC training loop. Stock state equality can coexist with different
parameter updates because the omitted caches differ. A future scalar-block codec,
nonconstant schedules, other cadences/devices and whole-TPEN restart are unproved.

Sources: [`BaseKFACPreconditioner` constructor, `state_dict`, `load_state_dict`,
`step`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/base_preconditioner.py),
[`KFACBaseLayer.state_dict`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/base.py),
[`KFACEigenLayer` public cache properties](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/eigen.py).

## Reproduction and validation boundary

`tools/kfac_compatibility_probe.py` is a diagnostic artifact, separate from the
training package. It requires an **already provisioned**, authorized environment
containing TPEN and the pinned KFAC package, plus a clean external source checkout
at the exact pin. It does not install anything. It verifies that every imported
KFAC Python source matches the checkout before evaluating the candidate. Use the
cluster-access skill and current Cannon notes before preparing or executing a
review run; environment changes require the operator's intervention.

Inside the approved scheduler allocation, with its prescribed environment:

```bash
python tools/kfac_compatibility_probe.py --kfac-source "$P2_KFAC_SOURCE" > "$P2_RECEIPT"
```

`P2_KFAC_SOURCE` and `P2_RECEIPT` are explicit operator-supplied paths; preserve
the receipt and stderr. Exit **1** with `probe_status=negative_witnesses_observed`
means the probe completed and found incompatibilities. Exit **2** / `blocked`
or `inconclusive` is not a successful negative execution. There is no exit-0
admission route: this bounded diagnostic cannot manufacture the missing C1–C5
extension proofs. Its `criterion_verdicts` always retain the full criteria as
`UNESTABLISHED-pending-reviewer-cluster-evidence`; completed diagnostics are
reported separately as `observed_stock_failures`. Inspect all criteria and
controls, not only the exit code. A completed diagnostic is not a criterion pass.

At implementation handoff, validation comprises source inspection at the exact
upstream pin, the explicit algebra above, Python AST syntax validation and
`git diff --check`. **The runtime probe has not been executed; no cluster test,
factor measurement or restart measurement is claimed.** The negative admission
decision is based on the inspected stock behavior and absent full compatibility
proof. The durable reviewer owns independent cluster tests and their receipts;
the PR remains work/awaiting review and human merge.
