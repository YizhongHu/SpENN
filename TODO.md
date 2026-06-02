# SpENN Master TODO Roadmap

Source of truth: `spenn_project_instructions.md`. This file is a documentation-only roadmap for future implementation work. It intentionally does not define code, change APIs, or override the project instructions.

## Guiding Principle

- Keep SpENN separated into four responsibility layers: representation theory, neural model layers, QMC physics, and sampling/training.
- Keep PyTorch as the primary runtime and autograd backend for model evaluation, local energy, and training.
- Treat SpechtMP as an equivariant encoder that produces features for determinant/Pfaffian readouts.
- Treat the default practical ansatz as a neural Pfaffian whose skew kernel is produced by a Specht-module encoder.
- Do not describe triple-to-pair branching plus Pfaffian readout as a true order-3 hyperpfaffian.
- Preserve intended SpechtMP public interfaces during the prototype so Phase 2 generalization is internal rather than a user-facing rewrite.
- Prioritize mathematical correctness before performance, especially for representation maps, equivariance, antisymmetry, and local energy.

## Global Constraints

- Do not modify existing documentation or source files when working from this roadmap unless an implementation task explicitly asks for it.
- Keep learned parameters from arbitrarily mixing transforming irrep coordinates; learned maps may mix channels, paths, and multiplicity indices.
- Keep fixed representation-theoretic maps `C` and `B` non-learned.
- Enforce exact terminal skew-kernel symmetry for Pfaffian readout: `K_ji = -K_ij`.
- Use `float64` for early VMC correctness work; defer mixed precision until correctness is established.
- Support device, dtype, batch dimension, and configurable electron count through system/config objects.
- Warn or error on dense high-order requests that violate the staged implementation plan.
- First implementation must clearly error for `M > 3` or `M_virtual > 3`.
- First implementation must also clearly error for `M_virtual = 4`; low-rank virtual-4 behavior belongs to Phase 3 and must be explicit opt-in through config.
- Keep dense `M_virtual > 3` out of early implementations unless a later task explicitly implements an approved approximation.

## Phase 0: Package And Codebase Survey

- [x] Survey existing packages before implementing representation theory, Pfaffians, samplers, Hamiltonians, or VMC loops.
- [x] Record exact package versions, licenses, APIs, tensor backends, differentiability support, GPU support, and maintenance status.
- [x] Decide whether each package is adopted as a dependency, optional integration, reference implementation, validation backend, or avoided.
- [x] Use only these recommendation statuses in the survey table: `adopt`, `optional`, `reference only`, and `avoid`.
- [x] Prefer direct PyTorch implementations for differentiable training paths unless a package cleanly supports PyTorch tensors and autograd.
- [x] Treat JAX and C++ projects as references unless concrete integration is justified by measurable value.
- [x] Keep optional package integrations out of core runtime imports until their owning packages or tests implement them.
- [x] Keep optional integrations out of core runtime imports; all optional packages must enter through explicit wrappers/adapters.
- [x] Plan a later PySCF wrapper for molecule/system conversion and reference data, while keeping the first harmonic-trap milestone independent of PySCF.
- [x] Use SciPy/SymPy only as optional offline/dev/test utilities for linear algebra, exact combinatorics, permutations, partitions, and sanity checks.
- [x] Plan Sage/passagemath fixture generation for partitions, tableaux, Specht dimensions, generator matrices, branching multiplicities, and small fusion/intertwiner checks.
- [x] Store generated Sage/passagemath fixtures as plain data with generator package/version, command, date, and basis-convention provenance.
- [x] Keep `pfapack` as a validation/reference backend for Pfaffian values, not the default differentiable training path.
- [x] Add optional `pfapack` comparison tests for small Pfaffians only.
- [x] Spike NetKet only as a tiny JAX SpENN-like prototype on the two-electron harmonic trap.
- [x] Spike a DeepQMC adapter only if choosing a JAX/Haiku backend experiment.
- [x] Check whether any package already provides reusable data containers, electron-system definitions, local energy estimators, samplers, or wavefunction training loops.
- [x] Check whether any package provides Specht modules, Young symmetrizers, branching rules, Littlewood-Richardson coefficients, or symmetric-group irreps useful for generating fixed maps.
- [x] Mark packages as `avoid` when license, maintenance, runtime, differentiability, or integration constraints make them unsuitable, even if no named baseline package is currently in that category.

| Package/codebase | Status | Phase 0 recommendation | Main reason |
| --- | --- | --- | --- |
| PyTorch | adopt | Use as the primary tensor, autograd, module, optimizer, and training runtime backend. | Required by the roadmap and best aligned with differentiable VMC, local energy, and GPU batching. |
| Hydra/OmegaConf | adopt | Use for configuration composition and object instantiation across model, sampler, Hamiltonian, trainer, logging, and hardware groups. | Matches the planned config-driven project structure. |
| WandB | optional | Use for experiment tracking when enabled by config and restricted to rank 0 under distributed training; expect this as a common working mode. | Useful for metrics and run history, but should not be required for core training or tests. |
| DeepQMC | reference only | Study project structure, VMC training patterns, molecule/system abstractions, and sampler ergonomics; attempt an optional backend experiment only if adopting a JAX/Haiku interface and training framework. | Useful neural QMC reference, but direct reuse implies a different backend stack. |
| PyQMC | reference only | Review Hamiltonian/local-energy conventions, walkers, observables, and validation cases. | Useful QMC design reference; likely not a direct SpechtMP runtime dependency. |
| QMCPACK | reference only | Borrow established sampler/local-energy diagnostics and benchmark conventions. | C++ HPC codebase; integration would add heavy complexity. |
| NetKet | optional | Use only as a JAX prototype candidate for a tiny continuous-particle VMC spike, not as a PyTorch dependency. | Has relevant continuous-particle VMC machinery, but belongs behind an optional adapter. |
| jVMC | reference only | Review JAX VMC abstractions and sampler/training organization without adopting it as a PyTorch dependency. | Useful reference, but not the planned runtime stack. |
| FermiNet and other NN-QMC implementations | reference only | Review neural fermionic ansatz ideas, envelope/cusp handling, and training recipes. | Specific implementations are design references, not direct code dependencies. |
| PySCF | optional | Plan a later wrapper for molecule/system conversion and reference data; keep the first custom two-electron harmonic trap independent. | Valuable scientific backend, but core Hamiltonian/System conventions must remain usable without it. |
| pfapack | optional | Use only as a validation/reference backend for small Pfaffian tests and numerical comparisons. | Not the default differentiable PyTorch training implementation. |
| SciPy/SymPy | optional | Use for offline/dev/test numerical linear algebra, exact combinatorics, permutations, partitions, and sanity checks. | Helpful for validation and map-generation support, but not hot runtime paths. |
| Sage/passagemath | optional | Use only for external/dev fixture generation and validation of partitions, tableaux, Specht dimensions, generator matrices, branching, fusion, and intertwiner checks. | Heavy optional tooling; generated fixtures must be plain data with provenance and normal tests must not import it. |

## Phase 1: Hard-Coded Prototype For `M = 2`

Strategy: implement hard-coded fuser/brancher behavior for `M = 2`, preserve the intended public SpechtMP interfaces, and make all unsupported higher-order cases fail clearly.

### Skeleton

- [x] Create the package skeleton described in `spenn_project_instructions.md` only when implementation work begins.
- [x] Distribute TODO.md and instructions.md to each folder under root, containing implementation details for that folder specifically, each getting information from 
      `spenn_project_instructions.md` and `MASTER_TODO.md` respectively to reduce context length and increase parallisability,
- [x] Add `pyproject.toml` with minimal dependencies for PyTorch, Hydra/OmegaConf, pytest, and optional WandB.
- [x] Add root package initialization without embedding training logic in package import side effects.
- [x] Add scripts for training, evaluation, sampling, and debug checks as thin Hydra entrypoints.
- [x] Define config groups for model, encoder, SpechtMP, readout, Hamiltonian, sampler, trainer, logging, and hardware.
- [x] Keep configuration names aligned with the proposed layout so later modules can be instantiated without churn.
- [ ] Edit README.md with installation / running instructions

### Data Containers

- [ ] Define minimal `ElectronBatch` with positions, optional spins, nuclear data/system reference, device, and dtype conventions.
- [ ] Define `Walkers` with positions, spins, cached `logabs`, cached `sign`, and auxiliary fields.
- [ ] Define `WavefunctionOutput` with `logabs`, `sign`, optional `phase`, and `aux`.
- [ ] Implement `FeatureDict` as the sole public feature container passed between encoder, SpechtMP, and readout.
- [ ] Support `FeatureDict.get`, `set`, `has`, `items`, and `.to(device, dtype)`.
- [ ] Store features by logical key `features[order][irrep]`, hiding tensor-layout details behind the container.
- [ ] Add validation helpers for shape, batch size, electron count, device, dtype, symmetry, and supported `(order, irrep)` keys.

### Minimal Encoder

- [ ] Implement first encoder with only order-1 and order-2 features.
- [ ] Produce order-1 `(1)` features from one-body electron coordinates/spins/system context.
- [ ] Produce order-2 `(2)` symmetric pair features from distances and symmetric pair context.
- [ ] Produce order-2 `(1,1)` antisymmetric pair features from antisymmetric relative-coordinate channels or another explicit antisymmetric construction.
- [ ] Enforce and test `s_ij = s_ji` for `(2)` features.
- [ ] Enforce and test `a_ij = -a_ji` for `(1,1)` features.
- [ ] Avoid adding triples in Phase 1 until the pair-only prototype passes core tests.
- [ ] Keep electron-electron cusp as a separate module returning a `[batch]` log contribution.

### Pfaffian Readout

- [ ] Implement a differentiable PyTorch Pfaffian path for training.
- [ ] Add stable signed-log handling for one Pfaffian.
- [ ] Add stable signed-log handling for a sum of Pfaffians.
- [ ] Build pair skew kernels only from valid antisymmetric carriers and symmetric gates.
- [ ] Enforce `K_ji = -K_ij` explicitly before calling the Pfaffian routine.
- [ ] Support even electron counts with `Pf(K)`.
- [ ] Support odd electron counts with a bordered Pfaffian.
- [ ] Keep `pfapack` comparisons in optional validation tests only.

### Hard-Coded `M = 2` SpechtMP

- [ ] Implement `SpechtFuser`, `SpechtBrancher`, `SpechtMPLayer`, and `SpechtMP` public interfaces matching the intended design.
- [ ] Internally hard-code only the needed order-1 and order-2 paths for the prototype.
- [ ] Preserve constructor parameters for `M`, `M_virtual`, channels, fixed-map/cache objects, residuals, activation, and normalization where practical.
- [ ] Make `M = 2`, `M_virtual = 2` the first supported SpechtMP configuration.
- [ ] Error clearly for `M > 2` during the earliest substep until Phase 1b enables hard-coded `M = 3`.
- [ ] By the end of Phase 1b, error clearly for `M > 3` or `M_virtual > 3`.
- [ ] Do not implement dense virtual order 4 in this phase.
- [ ] Use explicit pair loops/vectorized pair tensors rather than general map-generation machinery.
- [ ] Add residual updates and channel mixing only where they preserve irrep-coordinate equivariance.
- [ ] Add minimal activation behavior that does not break pair symmetry.

### Sampler

- [ ] Implement batched `MetropolisSampler` as the first sampler.
- [ ] Support walker tensors shaped `[n_walkers, n_electrons, spatial_dim]`.
- [ ] Cache model `logabs` and `sign` on walkers when useful.
- [ ] Use probability ratios based on `logabs` differences.
- [ ] Track acceptance rate and proposal step size.
- [ ] Provide equilibration/warmup steps separated from training samples.
- [ ] Keep sampler independent of Specht irreps and model internals.

### Toy Hamiltonian And Local Energy

- [ ] Implement a minimal electronic or toy Hamiltonian object that treats the model as a black box.
- [ ] Implement potential energy separately from kinetic energy.
- [ ] Implement initial kinetic energy via PyTorch autograd Laplacian.
- [ ] Keep local energy code isolated from sampler and model internals.
- [ ] Add small analytic or finite-difference sanity checks before using local energy in training.
- [ ] Support batched walker input and return local energy shaped `[batch]`.

### VMC Loop

- [ ] Implement `VMCLoss` returning loss and detached metrics.
- [ ] Start with mean local energy as the first simple objective.
- [ ] Leave room for the covariance-gradient estimator API.
- [ ] Implement `VMCTrainer` to connect model, sampler, Hamiltonian, loss, optimizer, scheduler, checkpoints, and logging.
- [ ] Instantiate components through Hydra in `train.py`.
- [ ] Track energy, variance, acceptance rate, step size, loss, grad norm, parameter norm, and local-energy histogram.
- [ ] Gate WandB logging behind config and rank checks.

## Phase 1b: Hard-Coded Prototype For `M = 3`

Strategy: after `M = 2` works, add hard-coded `M = 3` cases without introducing the fully general representation-map engine.

- [ ] Extend the encoder to produce order-3 `(3)`, `(2,1)`, and `(1,1,1)` features.
- [ ] Represent `(2,1)` with both transforming irrep coordinate and multiplicity/Fourier-column coordinate.
- [ ] Enforce triple symmetry rules for `(3)`, `(2,1)`, and `(1,1,1)` according to the chosen convention.
- [ ] Hard-code useful `M = 3`, `M_virtual = 3` fuser paths needed for the Pfaffian-kernel generator.
- [ ] Hard-code branch paths `T -> S`, `V -> S + A`, and `E -> A` for triple-to-pair readout support.
- [ ] Hard-code order `3 -> 1` paths where they are required for message passing.
- [ ] Keep pair `S` gates and pair `A` carriers as the terminal Pfaffian readout targets.
- [ ] Add tests that triple information can change the pair kernel while preserving `K_ji = -K_ij`.
- [ ] Maintain the same public SpechtMP interfaces used in Phase 1.
- [ ] Error clearly for `M > 3` or `M_virtual > 3`.
- [ ] Do not add low-rank `M_virtual = 4` behavior in Phase 1b.

## Phase 2: General Representation Maps And Original SpechtMP

Strategy: replace hard-coded cases with the fully general original SpechtMP machinery. Allow arbitrary `M` and `M_virtual` in principle, even if inefficient. Correctness comes before performance.

- [ ] Implement partitions, permutation utilities, Specht/Young bases, irrep dimensions, and representation metadata.
- [ ] Implement adjacent-transposition generator matrices for supported Specht conventions.
- [ ] Implement direct nullspace computation for `Hom_G(V_src, V_tgt)` intertwiners.
- [ ] Implement fixed fusion maps `C_{I1,I2 -> U,p}^{lambda; lambda1, lambda2}`.
- [ ] Implement fixed branching maps `B_{J -> I,q}^{lambda; mu}`.
- [ ] Include multiplicity/path indices for repeated independent maps.
- [ ] Cache fixed maps on disk with dtype/device transfer support.
- [ ] Validate generated maps against known small cases and optional external references from Phase 0.
- [ ] Replace hard-coded `M = 2` and `M = 3` fuser internals with general fixed-map machinery.
- [ ] Replace hard-coded brancher internals with general fixed-map machinery.
- [ ] Preserve the Phase 1 public SpechtMP interfaces.
- [ ] Allow arbitrary `M` and `M_virtual` in principle, with explicit warnings for inefficient dense configurations.
- [ ] Implement correctness tests before optimizing map generation or tensor contractions.
- [ ] Keep dense high-order computation separate from Phase 3 approximation modules.

## Phase 3: Low-Rank `M_virtual = 4` And Other Approximations

Strategy: approximation modules are separate from the original/general SpechtMP implementation and are explicit opt-in through config.

- [ ] Create separate low-rank approximation modules outside the original/general SpechtMP path.
- [ ] Add config flags that explicitly enable approximation behavior.
- [ ] Implement low-rank virtual-4 branch-down only after Phase 2 correctness is established.
- [ ] Target pair updates `Delta s_ij^(4)` and `Delta a_ij^(4)` for Pfaffian readout.
- [ ] Aim for approximately `O(n^3 R)` virtual-four effects rather than dense `O(n^4)` materialization.
- [ ] Include diagonal-correction handling in the low-rank branch-down formulas.
- [ ] Make approximation rank, target sectors, and enablement explicit in config.
- [ ] Add tests proving approximation modules preserve pair symmetry and skew-kernel antisymmetry.
- [ ] Ensure approximation modules can be disabled to recover original/general SpechtMP behavior.
- [ ] Treat other approximations the same way: isolated modules, explicit config, correctness tests, and no silent replacement of exact logic.

## Later Readouts: Determinant And Sum

- [ ] Implement determinant readout after the Pfaffian path is stable.
- [ ] Build orbital matrices from features without leaking sampler or Hamiltonian concerns into readout code.
- [ ] Use `torch.linalg.slogdet` or another differentiable PyTorch path.
- [ ] Add determinant antisymmetry tests under electron permutations.
- [ ] Implement `SumReadout` for combining determinant and Pfaffian readouts.
- [ ] Use stable signed-log-sum-exp for mixed readout outputs.
- [ ] Keep determinant, Pfaffian, and sum readouts behind a common `Readout` interface.

## Later DDP And Scaling

- [ ] First rely on walker batching for GPU utilization.
- [ ] Add DDP only after single-process training is correct and stable.
- [ ] Use one process per GPU, with independent walkers per rank.
- [ ] Average gradients through DDP and aggregate scalar metrics explicitly.
- [ ] Add distributed helpers for setup, rank checks, main-process logging, and all-reduce mean.
- [ ] Ensure only rank 0 writes logs/checkpoints unless explicitly configured otherwise.
- [ ] Defer asynchronous/distributed sampling until DDP and batched sampling are reliable.

## Later Samplers

- [ ] Add MALA/Langevin sampler after basic Metropolis is tested.
- [ ] Add proposal adaptation or step-size scheduling without coupling sampler internals to SpechtMP.
- [ ] Add per-electron and all-electron move options if useful for acceptance and decorrelation.
- [ ] Track acceptance statistics by rank and globally when DDP is enabled.
- [ ] Add sampler tests for shape preservation, finite log-probability handling, and reproducibility under seeded configs.

## Later Optimized Local Energy

- [ ] Profile the autograd Laplacian before optimizing.
- [ ] Explore `vmap`, forward-mode AD, custom kinetic estimators, or analytic derivatives only after correctness tests pass.
- [ ] Keep optimized kinetic paths behind the same Hamiltonian/local-energy interface.
- [ ] Compare optimized local energy against the simple autograd implementation on small systems.
- [ ] Add numerical tolerances and dtype-specific expectations for local-energy regression tests.

## Global Tests

- [ ] Test encoder equivariance: `features(sigma . X) = sigma . features(X)`.
- [ ] Test SpechtMP equivariance: `SpechtMP(sigma . features) = sigma . SpechtMP(features)`.
- [ ] Test branching symmetry: `bar_s_ji = bar_s_ij` and `bar_a_ji = -bar_a_ij`.
- [ ] Test terminal skew-kernel condition: `K_ji = -K_ij`.
- [ ] Test Pfaffian readout antisymmetry: `psi(sigma . X) = sgn(sigma) psi(X)`.
- [ ] Test determinant readout antisymmetry once determinant readout exists.
- [ ] Test sum readout antisymmetry once mixed readouts exist.
- [ ] Test fixed intertwiners commute with adjacent-transposition generators.
- [ ] Test generated fusion and branching map path counts against small known cases.
- [ ] Test Pfaffian implementation against brute-force small-`n` values and optional `pfapack` reference values.
- [ ] Test local energy against analytic wavefunctions where possible.
- [ ] Test Hydra instantiation for primary configs.
- [ ] Test unsupported configs fail clearly, including initial errors for `M > 3` or `M_virtual > 3`.
- [ ] Test no learned module mixes transforming irrep coordinates outside fixed equivariant maps.

## Acceptance Checklist

- [ ] `MASTER_TODO.md` references `spenn_project_instructions.md` as the source of truth.
- [ ] The roadmap encodes the hard-coded-first, general-second, low-rank-third strategy.
- [ ] Phase 1 implements `M = 2` first.
- [ ] Phase 1b implements hard-coded `M = 3` second.
- [ ] Phase 2 implements fully general original SpechtMP modules.
- [ ] Phase 3 keeps low-rank and other approximations separate and explicit opt-in through config.
- [ ] The first implementation clearly errors for `M > 3` or `M_virtual > 3`.
- [ ] Phase 0 includes package/codebase survey tasks before implementation.
- [ ] Phase 0 recommendation statuses are limited to `adopt`, `optional`, `reference only`, and `avoid`.
- [ ] Phase 0 recommendation table includes PyTorch, Hydra/OmegaConf, and optional WandB.
- [ ] Phase 0 recommendation table includes DeepQMC, PyQMC, QMCPACK, NetKet, jVMC, FermiNet or other NN-QMC implementations, PySCF, `pfapack`, SciPy/SymPy, and Sage/passagemath.
- [ ] The roadmap states PyTorch remains the primary runtime.
- [ ] The roadmap states JAX/C++ projects are references unless concrete integration is justified.
- [ ] The roadmap states `pfapack` is a validation/reference backend, not the default differentiable training path.
- [ ] The project instructions route optional integrations to owning packages or tests instead of empty interface placeholders.
- [ ] The project instructions state normal tests do not require optional packages.
- [ ] The project instructions state runtime Specht logic lives in `spenn/reps/`.
- [ ] The project instructions state SpechtMP does not own Sage/passagemath generation logic.
- [ ] No source code files are modified by this documentation-only task.
