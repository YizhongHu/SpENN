Plan a structural change with rplan. Do not implement code.

## Goal

Current problem:
- The current `HookeOrbitalBasis` evaluates one-dimensional Hermite/oscillator
  functions independently on each Cartesian coordinate and concatenates the
  results:
  $$ \phi_0(x), \cdots , \phi_S(x), \phi_0(y), \cdots , \phi_S(y), \cdoex$$
- It therefore produces coordinatewise one-dimensional features rather than
  genuine multidimensional oscillator basis functions on the spatial vector
  $r = (x_1, ..., x_d)$.
- In three dimensions, the current feature count is
  `d (S + 1)`, whereas a multidimensional tensor-product or total-shell basis
  has different basis elements, semantics, and dimensionality.
- The name `HookeOrbitalBasis` currently suggests multidimensional orbitals,
  but the implementation does not construct products across spatial
  coordinates.  [oai_citation:0‡GitHub](https://raw.githubusercontent.com/YizhongHu/SpENN/dev/spenn/nn/basis.py)

Desired end state:
- `HookeOrbitalBasis` represents each electron using genuine
  d`-dimensional harmonic-oscillator basis functions.
- Each spatial basis channel corresponds to one explicit multi-index
  $n = (n_1, ..., n_d)$ and evaluates

  $$ \Phi_n(r) = \prod_(k = 1)^d phi_(n_k)(x_k) $$

  where $\phi_n$ is the selected one-dimensional Hermite or oscillator
  function.
- The basis exposes an unambiguous truncation convention, deterministic
  multi-index ordering, and correct output width.
- The representation remains a one-body, per-electron feature tensor consumed
  through the existing `ElectronBasisFeatures.one_body` interface.
- The change remains local to the basis/configuration surface and does not
  redesign embedding, tensor-product layers, envelopes, or TPEN.

Non-goals:
- Do not implement the TPEN architecture or rename the project.
- Do not redesign `ElectronBasisFeatures`, embedding, normalization, readout,
  cusp, or envelope ownership.
- Do not introduce pairwise or higher-body basis features.
- Do not create a general-purpose basis framework for hypothetical future
  systems.
- Do not optimize training performance as part of this change.
- Do not alter sampler, optimizer, training schedule, or experiment selection.
- Do not preserve the current coordinatewise behavior under the misleading
  `HookeOrbitalBasis` name merely for compatibility.
- Do not decide whether multidimensional Hooke features improve QMC results;
  this change should first establish mathematically correct basis semantics.

Reason for change:
- Correctness: the current class does not implement the multidimensional
  orbital basis implied by its name.
- Interpretability: each output channel should correspond to a well-defined
  multidimensional oscillator basis element.
- Experiment design: comparisons between raw coordinates and a “Hooke orbital
  basis” should test the intended mathematical object rather than an
  axiswise feature map.
- Maintainability and future TPEN work: the basis contract should be explicit
  before the larger architecture migration.

## Implementation intent

Build a mathematically faithful, explicit, and easily testable multidimensional
Hooke basis as a narrow replacement of the current misleading implementation.

Prioritize:
- unambiguous basis semantics;
- deterministic and inspectable channel ordering;
- exact small reference cases;
- minimal changes outside the basis/configuration seam;
- compatibility decisions made explicitly rather than accidentally.

Avoid:
- broad abstraction;
- speculative support for unrelated basis families;
- hidden index generation;
- silent preservation of old semantics;
- coupling this change to TPEN, performance work, or experiment restructuring.

## Semantic contract [required]

Terms and definitions:
- **Spatial dimension `d`**: the number of coordinate components of each
  electron position.
- **One-dimensional order `n_k`**: the Hermite/oscillator index associated with
  coordinate `x_k`.
- **Multi-index `n`**: the tuple `(n_1, ..., n_d)` identifying one
  multidimensional basis function.
- **Total shell `s`**: the sum `|n| = sum_k n_k`.
- **Maximum shell `S`**: under total-shell truncation, include exactly the
  multi-indices satisfying `|n| <= S`.
- **Box size/order `K`**: under tensor-product box truncation, include exactly
  the multi-indices satisfying `0 <= n_k < K` for every coordinate.
- **Basis channel**: one multidimensional function `Phi_n`; not one coordinate
  component of one-dimensional `phi_n`.
- **Body order**: number of electrons jointly represented by a feature. This
  basis remains body order one and is independent of polynomial degree or
  oscillator shell.
- **Gaussian factor**: the factor `exp(-omega |r|^2 / 2)` obtained by multiplying
  the one-dimensional Gaussian factors across coordinates.
- **Spin channel**: an optional final per-electron channel appended after all
  spatial basis channels; it is not part of the spatial multi-index basis.

Mathematical or behavioral rule:
- Let an electron position be

  `r = (x_1, ..., x_d)`.

- Let

  `xi_k = sqrt(omega) x_k`

  and let the physicists' Hermite polynomials satisfy

  `H_0(xi) = 1`,

  `H_1(xi) = 2 xi`,

  `H_(n + 1)(xi) = 2 xi H_n(xi) - 2 n H_(n - 1)(xi)`.

- Define the one-dimensional factor as either

  `phi_n(x) = H_n(sqrt(omega) x)`

  when the Gaussian factor is excluded, or

  `phi_n(x) = H_n(sqrt(omega) x)
              exp(-omega x^2 / 2)`

  when it is included.

- For each admitted multi-index `n = (n_1, ..., n_d)`, define

  `Phi_n(r) = product_(k = 1)^d phi_(n_k)(x_k)`.

- The spatial feature vector for one electron is the ordered sequence

  `[Phi_n(r)]_(n in I)`,

  where `I` is the admitted, canonically ordered multi-index set.
- If spin is enabled, append the electron spin after all spatial channels.
- Applying an electron permutation must permute only the electron axis and
  leave the basis-channel ordering unchanged.

Inputs and outputs:
- Input:
  - `ElectronBatch`.
  - `positions` with shape
    `[*sample_shape, n_electrons, spatial_dim]`.
  - Optional `spins` compatible with
    `[*sample_shape, n_electrons]`.
  - Basis parameters including:
    - `omega > 0`;
    - `spatial_dim > 0`;
    - explicit truncation convention and its bound;
    - `include_gaussian_factor`;
    - `include_spin`.
- Output:
  - `ElectronBasisFeatures`.
  - `one_body` with shape
    `[*sample_shape, n_electrons, n_spatial_basis + spin_width]`.
  - `pair = None`.
  - Metadata sufficient to identify the basis semantics, including at least
    basis type, spatial dimension, truncation convention, truncation bound,
    Gaussian-factor choice, spin choice, and output width.
- Canonical ordering:
  - Basis channels must follow a deterministic ordering of multi-indices.
  - `rplan` should choose or recommend an ordering that is simple to state,
    test, serialize, and reproduce.
  - A likely default is increasing total degree followed by lexicographic
    ordering within each shell, but this is a decision to evaluate rather than
    assume.
  - Spin, when included, is the final channel.
  - The ordering must not depend on tensor values, runtime device, batch shape,
    or iteration over an unordered container.

Reference examples:
- Three dimensions, total-shell truncation with `S = 0`:
  - Multi-indices: `(0, 0, 0)`.
  - Spatial width: `1`.
  - With Gaussian included:

    `Phi_(0,0,0)(x, y, z)
     = exp(-omega (x^2 + y^2 + z^2) / 2)`.

- Three dimensions, total-shell truncation with `S = 1`:
  - Multi-indices:
    `(0,0,0)`, `(1,0,0)`, `(0,1,0)`, `(0,0,1)`,
    subject to the accepted canonical ordering.
  - Spatial width: `4`.
  - The nonconstant channels are proportional to
    `x g(r)`, `y g(r)`, and `z g(r)`, where

    `g(r) = exp(-omega |r|^2 / 2)`

    when the Gaussian is included.

- Three dimensions, total-shell truncation with `S = 2`:
  - Spatial width:

    `binom(S + d, d) = binom(5, 3) = 10`.

  - Channels include the constant, three first-order functions, three pure
    second-order functions, and three mixed products.

- Three dimensions, box truncation with `K = 2`:
  - Multi-indices satisfy `n_k in {0, 1}`.
  - Spatial width:

    `K^d = 2^3 = 8`.

  - This is not equivalent to total-shell truncation with `S = 1` or `S = 2`.

- At the origin:
  - Every channel with any odd `n_k` is zero.
  - The constant channel is one without the Gaussian factor and also one with
    the Gaussian factor.
  - Even-order channels take the values implied by the physicists' Hermite
    convention.

- In one dimension:
  - The multidimensional implementation must reduce to the existing
    one-dimensional Hermite/oscillator formula, modulo accepted naming,
    ordering, and migration policy.

## Structural surfaces [required]

Current owner and implementation seam:
- `spenn/nn/basis.py`.
- `ElectronBasis`.
- `ElectronBasisFeatures`.
- `HookeHermiteBasis`.
- `HookeOrbitalBasis`.
- `_hermite_features`.
- Configuration construction and resolver surfaces that derive embedding input
  width from `basis.out_features`.
- Existing basis unit and equivariance tests.
- Current experiment configurations selecting `HookeOrbitalBasis`.

The current class reports
`spatial_dim * (max_shell + 1)` spatial channels and flattens over coordinate
component and one-dimensional order, confirming that the current output is
axiswise rather than a multidimensional product basis.  [oai_citation:1‡GitHub](https://raw.githubusercontent.com/YizhongHu/SpENN/dev/spenn/nn/basis.py)

Target owner and public surface:
- The multidimensional oscillator basis should remain owned by
  `spenn.nn.basis`.
- `HookeOrbitalBasis` should expose an explicit multidimensional truncation
  contract.
- `ElectronBasisFeatures.one_body` should remain the public output consumed by
  the embedding.
- `out_features` should be derived exactly from the admitted multi-index set.
- Multi-index enumeration should have one explicit owner within the basis
  implementation and should not be reconstructed by consumers.
- `rplan` should evaluate whether the current axiswise implementation should:
  - be removed;
  - be renamed and retained temporarily as an experimental comparison;
  - or be retained only as private/reference test logic.

Affected consumers and producers:
- Basis instantiation from configuration.
- Config resolvers deriving embedding input dimensions from
  `basis.out_features`.
- Embedding modules consuming `ElectronBasisFeatures.one_body`.
- Basis metadata and any trace/equivariance diagnostics.
- Existing pair-stability experiment configurations using the old
  `HookeOrbitalBasis` semantics.
- Unit, configuration, integration, and smoke tests.
- Checkpoints whose first embedding layer width depends on the old basis
  output width.
- Existing experiment lineage that records configurations using the old class
  name.

Representation decision:
- Preserve a flat per-electron feature tensor at the public embedding seam.
- Internally, the implementation may construct a tensor indexed by
  `[*sample_shape, n_electrons, n_1, ..., n_d]` or by an explicit multi-index
  list, but it must flatten to the canonical basis-channel axis before
  returning `ElectronBasisFeatures`.
- Prefer an explicit, inspectable multi-index enumeration over implicit tensor
  flattening whose order must be inferred from reshape behavior.
- Do not introduce a new public typed structure unless repository inspection
  demonstrates that the flat `one_body` surface cannot represent the semantic
  contract safely.
- The basis owns multi-index generation because basis truncation and channel
  interpretation are intrinsic to the basis, not to embedding or config code.

State and lifecycle:
- The admitted multi-index set and output width are determined from immutable
  configuration at initialization.
- Multi-index metadata should be generated once at initialization or exposed
  through deterministic immutable state.
- The basis must not rediscover channel ordering from runtime tensor values.
- Forward execution should evaluate the configured basis on the input batch
  without mutating semantic metadata.
- The implementation should avoid repeated Python-level combinatorial
  enumeration or device allocations inside each forward pass when they can be
  prepared once.
- Any stored index tensor must move correctly with module device/state behavior
  and must not accidentally become a trainable parameter.
- Persistence and serialization behavior should follow existing module/config
  conventions rather than introduce a separate basis-state format.

## Compatibility and migration [required]

Change policy:
- Treat this as a semantic replacement of the current
  `HookeOrbitalBasis`, with an explicit migration decision for the old
  axiswise feature map.
- It may be a breaking change for configurations and checkpoints even if the
  Python class name remains unchanged.

Compatibility targets:
- Configuration files selecting the current basis.
- Config resolvers and embedding-width derivation.
- Saved checkpoints whose embedding input width was based on
  `spatial_dim * (max_shell + 1)`.
- Result artifacts and experiment manifests that identify the old basis by
  class or metadata name.
- Pair-stability experiment lineage and interpretation.
- Public imports of `HookeOrbitalBasis`.
- `HookeHermiteBasis`, which may share the same axiswise semantic issue and
  should be assessed, but should not automatically be redesigned unless the
  accepted scope includes it.

Migration behavior:
- Existing checkpoints using the old feature width should normally be treated
  as incompatible and require retraining; silently reshaping or partially
  loading the first embedding layer would change model meaning.
- Existing completed experiment artifacts should remain historical records and
  should not be rewritten.
- Existing configs must not silently acquire new basis semantics without a
  version, rename, or clearly documented migration.
- `rplan` should decide whether the smallest sound migration is:
  1. replace `HookeOrbitalBasis` and rename the old class to an explicitly
     axiswise name for temporary comparison;
  2. introduce a new multidimensional class and deprecate the old class;
  3. make a clean breaking replacement and invalidate old configs.
- The recommendation should minimize compatibility machinery while preserving
  enough clarity to interpret prior experiments.

## Sources of truth [required]

Repository and revision:
- Repository: `https://github.com/YizhongHu/SpENN`.
- Planning target: current `dev` branch.
- Before spawning subagents, record the exact inspected commit SHA.
- Do not assume that remembered code from earlier discussions still matches
  `dev`.

Relevant sources, in priority order:
1. User instructions in this planning request.
2. Attached TPEN design document, especially sections defining the basis,
   feature representation, model pipeline, and intended transition from SpENN
   to TPEN.
3. Any GitHub issue or accepted decision specifically covering the Hooke basis
   reimplementation.
4. Current `dev` implementation:
   - `spenn/nn/basis.py`;
   - basis construction/config registration;
   - embedding input-width resolution;
   - tests for bases and equivariance.
5. Current experiment configs using `HookeOrbitalBasis`, especially
   pair-stability configurations.
6. Historical PRs or experiment documentation that explain why the existing
   axiswise basis and Gaussian-factor behavior were introduced.
7. Relevant exact Hooke-system formulas or reference data already adopted by
   the project.

Conflicts or ambiguities among sources:
- The current class name and documentation call the features “orbital-shaped,”
  while the implementation evaluates one-dimensional functions independently
  per coordinate rather than constructing multidimensional orbitals.
- `max_shell` currently means the highest one-dimensional index on each
  coordinate, not necessarily a maximum total shell.
- The intended truncation convention—total shell or Cartesian box—is not yet
  accepted.
- It is unresolved whether the Gaussian factor should be part of the input
  basis when the model already applies an output Gaussian envelope.
- It is unresolved whether `HookeHermiteBasis` should be migrated in the same
  change or left as an explicitly axiswise polynomial basis.
- The TPEN design document may describe the intended basis at a higher level
  than the current implementation; rplan must distinguish accepted design from
  tentative direction.

## Constraints [required]

Correctness and design invariants:
- The basis remains particle-permutation equivariant: permuting electrons
  permutes the electron axis and does not alter basis-channel semantics.
- Every channel corresponds to exactly one documented multi-index.
- Output width agrees exactly with the selected truncation:
  - total-shell: `binom(S + d, d)`;
  - box: `K^d`.
- Channel ordering is deterministic and tested.
- Batch/sample dimensions, dtype, and device are preserved.
- Optional spin remains the final one-body channel.
- `pair` remains `None`.
- `out_features`, actual output width, configuration resolution, and metadata
  must agree.
- Multidimensional products must be constructed explicitly or by a provably
  equivalent tensor operation; an MLP learning products later is not an
  equivalent basis implementation.
- No numerical normalization convention should be implied by the word
  “orbital” unless it is explicitly implemented and tested.
- The implementation must preserve coordinate differentiability needed for
  local-energy calculations.
- Tests must establish semantics independently of the production
  implementation.

Operational constraints:
- Forward execution must remain vectorized over sample and electron axes.
- Avoid material Python loops over batch configurations or electrons.
- Small shell enumeration may occur at initialization, not repeatedly per
  sample.
- The feature width must remain practical for intended small-shell
  experiments.
- Configs and experiment manifests must identify the truncation and Gaussian
  conventions reproducibly.
- The change should be independently releasable as a minor development
  milestone before analytic-envelope, profiling, and TPEN work.
- Cluster experiments are not required to establish the mathematical basis
  contract; only a bounded smoke or parity run should be considered after
  deterministic tests pass.

Forbidden approaches:
- Reintroducing or using `permute_tree`.
- Treating coordinatewise concatenation as a multidimensional orbital basis.
- Relying on reshape order without a documented and tested multi-index map.
- Inferring output width by running a probe batch through the basis.
- Generating semantic metadata lazily from runtime data.
- Hiding old semantics behind the same config without a migration decision.
- Loading old checkpoints by silently dropping, repeating, or remapping input
  weights.
- Redesigning embedding, TPEN layers, experiment infrastructure, or sampler as
  part of this change.
- Introducing a generic abstraction for arbitrary orthogonal-polynomial or
  many-body bases without a demonstrated current requirement.

## Acceptance evidence [required]

Required tests:
- Multi-index enumeration:
  - exact admitted sets for small `d`, `S`, and `K`;
  - exact channel counts;
  - deterministic canonical ordering;
  - no duplicates or omitted indices.
- Reference-value tests:
  - exact one-dimensional reduction;
  - exact values in `d = 2` and `d = 3` for shells `0`, `1`, and at least one
    second-order case;
  - origin values and parity behavior;
  - mixed-product channels such as `(1,1,0)`.
- Shape and contract tests:
  - arbitrary sample shapes;
  - multiple electron counts;
  - spin included/excluded;
  - `out_features` equals returned width;
  - metadata matches configured semantics;
  - dtype/device preservation.
- Equivariance tests:
  - electron permutation acts only on the electron axis;
  - basis-channel values and ordering remain unchanged under electron
    relabeling.
- Differentiability tests:
  - coordinate gradients exist and agree with analytic or independently
    constructed reference values for small cases;
  - no unintended detach or nondifferentiable index-dependent path.
- Gaussian convention tests:
  - enabled and disabled forms agree with the stated formula;
  - multidimensional Gaussian equals the product of coordinate Gaussians.
- Configuration tests:
  - construction of each accepted truncation mode;
  - embedding width is resolved correctly;
  - invalid or ambiguous settings fail clearly.
- Migration tests:
  - old configuration behavior follows the accepted preserve/rename/invalidate
    policy;
  - incompatible checkpoints fail explicitly rather than partially loading.
- Integration test:
  - basis output passes through the existing embedding/model construction path.
- Smoke evidence:
  - a minimal model forward and local-energy evaluation remains finite;
  - a small bounded training smoke run may be used to detect integration
    regressions, but is not evidence that the new basis improves physics.

Success metrics or comparison baseline:
- Exact equality for integer multi-index sets and output counts.
- Numerical reference values within dtype-appropriate tolerance, preferably
  checked in float64.
- Gradient agreement within a declared float64 tolerance.
- Exact agreement between `out_features` and runtime channel width.
- Existing raw-coordinate basis behavior remains unchanged.
- Previous axiswise behavior is either:
  - reproduced under an explicitly renamed legacy/comparison class;
  - or deliberately invalidated according to the accepted migration decision.
- No claim of improved variational energy is required for acceptance.

## Decisions requested from rplan

- Recommend total-shell truncation, box truncation, or support for both.
  - Determine which should be the default, if both are supported.
  - Prefer the smallest API that supports current experimental needs.
- Recommend the canonical multi-index ordering.
- Decide whether the Gaussian factor belongs in the multidimensional input
  basis, should be optional, or should be excluded because the output envelope
  already supplies asymptotic decay.
- Decide the fate of the current axiswise `HookeOrbitalBasis`:
  - rename and retain temporarily as an experimental control;
  - deprecate under a new class;
  - or remove through a clean breaking migration.
- Decide whether `HookeHermiteBasis` is in scope:
  - migrate it to multidimensional products in the same release;
  - rename it explicitly as axiswise;
  - or defer it.
- Decide how configuration expresses truncation without overloading ambiguous
  terms such as `order` or `max_shell`.
- Decide whether the accepted change requires a config-version increment.
- Decide what metadata is necessary for reproducible interpretation without
  introducing a heavy serialization schema.
- Decide whether checkpoint incompatibility is simply documented or enforced
  through an explicit compatibility/version check.
- Evaluate whether supporting both truncation conventions and both Gaussian
  choices is justified by current experiments or is premature generalization.

## Requested planning output

- Recommend the smallest sound direction.
- State the implementation intent that should govern unspecified local choices.
- Explain rationale and material tradeoffs.
- Identify evidence required before implementation.
- Separate requirements, recommendations, optional improvements, and unresolved
  user decisions.
- State any deliberate incompatibility clearly.
- Do not treat prior v3 performance as proof that a true multidimensional
  basis will perform better; distinguish mathematical correction from empirical
  benefit.

Before creating rplan subagents, inspect listed sources. If a required planning
surface remains unspecified, ask only questions needed to fill it and stop.
After all required surfaces are specified, run normal rplan workflow with
planner, implementation representative, and critic. Do not implement code.
