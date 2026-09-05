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

**2026-09-05, stock failures corroborated on Cannon; full compatibility
unestablished.** P3 must remain closed. The pinned package
does not satisfy C1, C3 or C5 through its stock path, and a complete public-API
extension satisfying all five criteria has not been established. C2's TPEN
curvature choice is also not frozen by this slice's authorization. There is no
partial compatibility pass, automatic optimizer substitution, dependency addition
or runtime enablement in this change.

This is an admission decision at the declared gate, not a proof that every
possible adaptation is impossible. Public extension points exist and deserve
accurate credit: the reviewer established three extension levers at toy scope.
A later proposal would need a separately authorized compatibility
proof before P3 could open; implementing an adapter in order to claim this gate
passed would invert the required staging.

The table below evaluates the **full TPEN compatibility criteria**, including
the proposed public extension, rather than only the stock implementation.
Each criterion uses exactly one of `PASSED`, `FAILED`, or
`UNESTABLISHED-pending-reviewer-cluster-evidence`. **None is PASSED.** The third
state is non-passing and cannot authorize P3. This mandated enum is retained;
the separate blocker column identifies who can actually move each criterion.
A cluster run cannot supply an operator decision or an absent implementation.

| Full criterion | State | Remaining proof and blocking party |
| --- | --- | --- |
| C1: semantic coverage | UNESTABLISHED-pending-reviewer-cluster-evidence | **Pre-adapter qualification:** complete named Kronecker/exact-scalar coverage and trace extraction for raw, composite and scalar families. **Reviewer:** independent cluster verification of that proof. |
| C2: VMC convention | UNESTABLISHED-pending-reviewer-cluster-evidence | **Operator decision:** freeze geometry, score normalization and numerical tolerances. **Pre-adapter qualification:** all-family dense oracles and precision policy addressing the upstream float32 eigensolve. **Reviewer:** execute those oracles. |
| C3: counts | UNESTABLISHED-pending-reviewer-cluster-evidence | **Pre-adapter qualification:** both A/G estimators, physical-sample counts and count-aware transport. **Reviewer:** unequal/empty-shard and global-empty behavior under real multi-rank execution. |
| C4: public APIs | UNESTABLISHED-pending-reviewer-cluster-evidence | **Pre-adapter qualification:** every required TPEN registration, factor, precision and state path through public APIs. **Reviewer:** verify the completed qualification; toy levers cover only a subset. |
| C5: restart | UNESTABLISHED-pending-reviewer-cluster-evidence | **Pre-adapter qualification:** complete method/runtime state, schedules and cache age. **Reviewer:** exact complete-state TPEN restart across factor and inverse refresh under real multi-rank execution. |

“Pre-adapter qualification” names missing proof, not authorization to start P3.
Any work exceeding P2's gate scope requires separate authorization; P3 remains
closed until all criteria pass. Probe JSON uses corresponding `blocking_parties`
metadata (`operator_decision`, `pre_adapter_qualification`,
`reviewer_cluster_evidence`).

The following **scoped results** make the evidence visible without promoting
any full criterion. Jobs and immutable receipts are recorded below.

| Criterion | Stock result | Public extension result established at toy scope |
| --- | --- | --- |
| C1 | **FAILED coverage:** census finds 96 unsupported named tensors across two model variants; automatic seeded registration finds zero modules. | Manual seeded-affine registration matches an `nn.Linear` control exactly for six steps, crossing inverse refresh. Raw/composite/scalar families remain open. |
| C2 | **Precision obstruction observed:** unconditional float32 eigensolve loses float64 information; synthetic preconditioned-gradient errors are `2.15e-5`–`1.00e-3`. No TPEN convention/tolerance pass. | Identical-algorithm float64 reference agrees with an independent float64 solve to `2.52e-15`–`2.03e-13`. An upstream-bypassing public precision path is not qualified. |
| C3 | **FAILED counts and empty-data semantics:** A-EMA `23/3` versus 11; all-empty input changes A-EMA to 0.5; empty plus full gives 5.75 versus 11. | Corrected public subclass gives A-EMA 11 exactly for `(4,)`, `(2,2)`, `(1,3)`, `(1,0,3)` and rejects global-empty A update. Single process; G and multi-rank transport unexercised. |
| C4 | Automatic registration is insufficient; no general impossibility of public adaptation is established. | Manual seeded registration, subclass-owned counted-A state and public eigen-cache codec all exercised successfully. This is not every TPEN path. |
| C5 | **FAILED stock reload:** splits 2/3 diverge; splits 1/4/5 reproduce the next two weights exactly. | Public eigen-cache codec reproduces the next two weight updates exactly at all five splits. Constant schedules, single-process `KFACEigenLayer`; full-state TPEN restart unestablished. |

These stock counterexamples refute the stock path, not every possible public
extension. Sum-plus-count factors under unequal and empty shards, and exact
complete-state restart across factor and inverse refresh, remain unestablished
under real multi-rank execution. Single-process cluster evidence cannot promote
either full criterion to `PASSED`.

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

The executed census reports 44 unsupported named tensors (335 scalar elements)
for ordinary embedding and 52 (375 scalar elements) for seeded embedding.
The total 96 counts named tensors across two variants, not 96 scalar parameters.
The reviewer's manual public mapping, `LinearModuleHelper` and `KAISAAssignment`
cover the seeded affine family with `torch.equal` weights for six steps. This
does not resolve the raw parameter families or composite hook bypass above.

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
Jobs 44568642 and 44571614 measured this result exactly. They also establish
the empty-data failure at the save/update path: `get_a_factor()` on zero rows
returns `[[0.0]]` without error, and an all-empty accumulation followed by
`update_a_factor(alpha=0.5)` moves the identity-initialized A factor to `[[0.5]]`.
An empty microbatch plus all four full rows produces `[[5.75]]`, versus the
count-correct `[[11.0]]`. This is an observed **factor-state update** from no
data, not a measured empty-data parameter update.

The corrected reviewer subclass (job 44571614) owns its `_counted_*`
accumulators and uses public helper methods and `a_factor`/`g_factor` properties;
it does not touch inherited private batch buffers. It undoes each helper's row
normalization, accumulates sums and counts, and reproduces A-EMA 11 exactly
for `(4,)`, `(2,2)`, `(1,3)`, `(1,0,3)`, rejecting global-empty A input.
This establishes a public **toy A-factor** lever. Its G methods, accumulator
restart/reset lifecycle, reused TPEN parameters and count-aware distributed
transport were not exercised. The original private-buffer implementation's
numerical result is retained as historical evidence, not a public-API proof.

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

### Float64 precision obstruction

At the pin, [`compute_a_inv` and `compute_g_inv`](https://github.com/gpauloski/kfac-pytorch/blob/5987766a43739de7eb950f564da54559f2504579/kfac/layers/eigen.py)
cast factors to float32 before `torch.linalg.eigh`, even when both
`factor_dtype` and `inv_dtype` are float64. Converting the resulting caches back
to float64 does not restore lost information. The dtype witness resolves
`1 + 1e-9` to exactly 1 without an exception.

The magnitude witness uses a **synthetic** 32-dimensional A spectrum
`logspace(0,-8)`, G=I, and a gradient in the three smallest eigendirections.
It is not a measured TPEN Fisher/QGT spectrum. Job 44571614 compares upstream
with the same eigensolve/clamping/outer-product-damping algorithm in float64,
and separately checks that reference against a float64 linear solve:

| Damping | Upstream vs float64 reference, relative gradient error | Float64 reference vs independent solve |
| ---: | ---: | ---: |
| `1e-4` | `1.0019447e-3` | `2.0323e-13` |
| `1e-3` | `1.8061195e-4` | `2.2912e-14` |
| `1e-2` | `2.1486727e-5` | `2.5223e-15` |

Upstream-versus-solve errors agree with upstream-versus-reference to the shown
precision. The ascending eigenvalue pairs are `1e-8 → 1.5560710e-8` (55.6%),
`1.8116092e-8 → 2.3168521e-8` (27.9%) and
`3.2819279e-8 → 3.6521367e-8` (11.3%). The review's initial 265% claim paired
oppositely ordered lists and is withdrawn; the corrected receipt supersedes it.

The receipt's `scientifically_material=true` uses the reviewer's pre-declared
`1e-6` diagnostic threshold, not an operator-frozen TPEN tolerance. This is a
C2 obstruction for any proposed adapter retaining the upstream eigen path:
setting float64 dtypes alone cannot support a float64-accuracy contract on
this witness. A qualification proposal must address the precision loss, or
bypass this eigensolve through a qualified public path. It must not relax C2
after observing this result merely to obtain admission.

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

The reviewer measured the predicted stock split pattern (jobs 44568642 and
44571614): splits 1, 4 and 5 give zero next-two-weight differences; split 2 gives
`0.0356506` then `0.0855615`, and split 3 gives `0.0590006` for both comparisons.
The selective divergence is consistent with cache age, rather than generic
nondeterminism. A public codec using `state_dict(include_factors=True)`, cloned
`qa/qg/da/dg/dgda` properties, `load_state_dict(compute_inverses=False)` and
the public cache setters gives **zero weight difference for both next updates
at every split**. This refutes any claim that the toy cache-preservation fix
requires private access. The codec does not establish equality of all complete
method/runtime state, nonconstant schedules, arbitrary refresh cadences,
whole-TPEN blocks or multi-rank restart.

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

The runtime probe **has been executed verbatim** by the independent reviewer
at TPEN `f5659e053b607d13108736a19e43e4951cdcf1c7`, in jobs 44568642 and
44571614. Both report exit 1, `negative_witnesses_observed`, exactly the three
stock failures (C1 coverage, C3 microbatch means, C5 reload), and all full
criterion verdicts unestablished. The upstream imported-source byte check passed.
This supersedes the initial handoff's unexecuted-probe statement. The reviewer
also checked upstream HEAD and newest tag on 2026-09-05: the inspected pin was
HEAD and v0.4.2 the newest tag; a newer-version escape was not found.

Accepted reviewer tests are committed as
[`p2_reviewer_battery.py`](../tools/kfac_review/p2_reviewer_battery.py) (all eight
arms, green controls included) and
[`p2_c2_precision_witness.py`](../tools/kfac_review/p2_c2_precision_witness.py).
They preserve the corrected tested numerical fixtures. Adoption adds actual
checkout-tip and clean-tree checks, upstream byte verification, and explicit
scope comments/metadata. Those changes need independent review at the new PR
head; the historical jobs below verify their stated old tip only.

In the already provisioned, approved allocation, invoke that checkout's venv
interpreter directly (with `P2_KFAC_SOURCE` set):

```bash
.venv/bin/python tools/kfac_review/p2_reviewer_battery.py > "$P2_BATTERY_RECEIPT"
.venv/bin/python tools/kfac_review/p2_c2_precision_witness.py > "$P2_PRECISION_RECEIPT"
```

The battery's exit 0 means completion, not eight passes: inspect each arm's
`status` and observed values, including the embedded probe's expected exit 1.
An arm error, environment error, or nonfinite/incomplete numerical result cannot
establish a claim. These scripts are single-process test fixtures; they are not
an adapter or a distributed proof.

### Reviewer execution receipts

All jobs used Cannon partition `test` and completed `0:0` in a fresh checkout
of `f5659e053b607d13108736a19e43e4951cdcf1c7`. The reviewer used per-checkout
`uv sync --locked --extra cpu` (torch `2.12.0+cpu`), then installed upstream
`0.4.2` with `--no-deps -e` at the exact pin into the **disposable review venv
only**. The repository lockfile was untouched. The interpreter was asserted;
subsequent invocations used that venv's Python directly, because `uv run`
resynchronization would prune the editable KFAC install. This records historical
provisioning, not authority for automatic future environment changes.

| Job | Node | Evidence |
| --- | --- | --- |
| 44568642 | holy8a24102 | All eight initial battery arms completed, including the verbatim probe. Original counted arm's private-buffer characterization is superseded below. |
| 44569422 | holy8a24101 | Initial precision witness. Gradient errors retained; eigenvalue pairing and unmeasured reference-error claims corrected below. |
| 44571614 | holy8a24102 | All eight arms completed with public counted accumulators; corrected ascending eigenvalue pairing and measured independent float64 solve control. |

Local originals, JSON, stdout and stderr are retained under
`/Users/yizhonghu/tpen-p2-review/`; Cannon copies under
`/n/netscratch/kozinsky_lab/Lab/rhu/tpen-p2-kfac-review-f5659e0/`.
Task Orchestrator receipt notes on P2 record transfer hashes, interpreter and
scheduler terminal states. Historical `.sbatch` scripts are accepted as execution
evidence, not committed as reusable launchers: their fixed facility paths,
force checkout and one-off provisioning are not defaults for this gate.

SHA-256 ledger (filenames relative to the retained receipt directory):

| Artifact | SHA-256 |
| --- | --- |
| `battery-receipt-44568642.json` | `d8a6e4487636eef1577aeabc370963137d77aa3f29ec9e4e1293f866fe847485` |
| `c2-precision-receipt-44569422.json` | `47c240a0af51f518dcf5fc89f10e0ba0fcec1b1d75a57b95057018efce8873f0` |
| `battery-receipt-r2-44571614.json` | `cf217c3e2691850b093c7ed1df51df1d48d28f16c0ab2d2ebc8ae101ab681a85` |
| `c2-precision-receipt-r2-44571614.json` | `2f25a63c99e3d7dd67eeca785718a410b2346ce9c2a536950ec18c8093ebb82e` |
| `p2_reviewer_battery.py` as rerun in 44571614, before adoption metadata | `b8d19fc8ed0e2c0f4e7dcee7c3733765d34aab47729b83c0544e0662df3ede96` |
| `p2_c2_precision_witness.py` as rerun in 44571614, before adoption metadata | `df1c29dda92833261e0fd6f1d6dfe8a8f509d64416d60f8b035691290ec426ee` |
| `cannon_job.sbatch` | `e006082369c31e267fe261dc878751e182f122e21dc6eaa02f45e26fcc742ba8` |
| `cannon_job_c2.sbatch` | `a27675cb995b23bfaeff7c3dea12dff768f26f1d4943dc44d93a58850e197048` |
| `cannon_job_r1fix.sbatch` | `5dfaa208d64627416d830f79ee58252dd724ab8d59cc60c3699875afcc3bfa4b` |

All I1–I9 findings and all test arms are accepted with the stated scope and
corrections. The original I5 private-field public-API claim, I7's mismatched
265% eigenvalue claim, and replacing the mandated state enum are disposed;
the corrected evidence and separate blocker metadata are adopted. No spike
disposal rule applies. The PR remains work/awaiting independent review and human
merge; the fixing side does not author its own tip-verification receipt.
