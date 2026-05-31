# SpENN Project Implementation Instructions

## Notice

This document is OUTDATED and should be used for BACKGROUND ONLY. Reference the instructions.md in the individual subfolders for implementation details.

This document describes a proposed PyTorch project structure for implementing **SpENN**: a Specht-module equivariant neural network for fermionic wavefunction modeling, with plug-and-play encoding, SpechtMP, determinant/Pfaffian readouts, VMC training, Hamiltonian evaluation, Monte Carlo sampling, Hydra configs, and optional WandB logging.

The goal is to keep the code modular enough that the mathematical pieces can be changed frequently without rewriting the Hamiltonian, sampler, or trainer.

---

## 0. Mathematical project context for Codex

This section explains what the package is trying to implement. The goal is to make the code structure meaningful even before all representation-theoretic maps are fully optimized.

### 0.1 Physical target

The model represents a real-valued fermionic wavefunction

```text
psi(X),  X = (r_1, ..., r_n)
```

for `n` electrons. It must satisfy exact permutation antisymmetry:

```text
psi(sigma . X) = sgn(sigma) psi(X)
```

for every electron permutation `sigma in S_n`.

The current intended practical readout is Pfaffian-based:

```text
psi(X) = sum_l b_l Pf(K_l(X))
```

where every `K_l(X)` is a skew-symmetric matrix:

```text
K_l[i, j] = -K_l[j, i]
```

Therefore the final antisymmetry is enforced by the Pfaffian. SpechtMP is the equivariant encoder used to construct good many-body skew kernels `K_l`.

Important conceptual point: if the readout is Pfaffian, then the ansatz is functionally a neural Pfaffian whose kernel is produced by a Specht-module message-passing encoder. Do not claim that branching triples down to pairs is equivalent to a true order-3 hyperpfaffian readout.

### 0.2 Local Specht features

For a tuple `I` of particles with size `|I| = k`, and a partition `lambda` of `k`, store a feature

```text
x_I^{(c, lambda)} in S^lambda
```

where:

- `I` is a tuple of particle indices;
- `c` is a channel index;
- `lambda` is a Specht-module irrep of `S_k`;
- `S^lambda` is the local irrep space.

Globally, these features transform as the induced representation

```text
V_{k,lambda} = Ind_{S_k x S_{n-k}}^{S_n}(S^lambda \boxtimes 1).
```

Intuitively: the selected subset `I` carries local symmetry type `lambda`, while the complement is pooled/trivial.

For `M <= 3`, we use the following shorthand, but they should not appear in code:

```text
order 1:
  h_i^c = x_i^{c, (1)}

order 2:
  s_ij^c = x_ij^{c, (2)},       s_ji = s_ij
  a_ij^c = x_ij^{c, (1,1)},     a_ji = -a_ij

order 3:
  t_ijk^c = x_ijk^{c, (3)}
  v_ijk^c = x_ijk^{c, (2,1)}
  e_ijk^c = x_ijk^{c, (1,1,1)}
```

The mixed triple irrep `(2,1)` is two-dimensional. If using a regular/Fourier-block convention, store it with both a transforming irrep coordinate and a multiplicity coordinate, for example:

```text
v_ijk[batch, i, j, k, channel, alpha, beta]
```

where:

- `alpha` is the transforming standard-irrep coordinate;
- `beta` is a multiplicity/Fourier-column coordinate;
- learnable weights may mix channels and multiplicities, but must not arbitrarily mix the transforming `alpha` coordinate except through fixed equivariant maps.

### 0.3 Fixed fusion intertwiners C

The fusion/intertwiner map `C` is a fixed equivariant map that combines two local features into an output irrep on the union subset:

```text
C_{I1,I2 -> U,p}^{lambda; lambda1, lambda2}:
  S_{I1}^{lambda1} tensor S_{I2}^{lambda2} -> S_U^lambda
```

where:

```text
U = I1 union I2
```

and `p` is a path/multiplicity index if the same output irrep appears more than once.

The general tensor-product message is:

```text
z_{U,p}^{t+1, c_out, lambda} =
  sum_{I1,I2 subset U, I1 union I2 = U}
  sum_{lambda1 partition |I1|}
  sum_{lambda2 partition |I2|}
  sum_{c1,c2}
  w_{p,c_out,c1,c2}^{lambda;lambda1,lambda2}
  C_{I1,I2 -> U,p}^{lambda;lambda1,lambda2}
  (x_{I1}^{t,c1,lambda1} tensor x_{I2}^{t,c2,lambda2}).
```

The learned weights `w` only mix channel/path/multiplicity indices. The fixed `C` handles the irrep-coordinate contraction.

The same notation covers:

```text
I1 = I2 = U                 -> Kronecker/same-subset product
I1 cap I2 = empty           -> Littlewood-Richardson/cross-order product
0 < |I1 cap I2| < min sizes -> mixed/overlap product
```

For implementation, precompute and cache all needed `C` maps for small `M` and `M_virtual`.

### 0.4 Fixed branching intertwiners B

The branching map `B` compresses a message on a larger subset `J` back to a retained subset `I`:

```text
B_{J -> I,q}^{lambda; mu}: S_J^mu -> S_I^lambda
```

where `q` is a branching multiplicity/path index if needed.

The branch update has the form:

```text
x_I^{t+1, c_out, lambda} =
  sum_{J superset I, |J| <= M_virtual}
  sum_{mu partition |J|}
  sum_q
  sum_{c_in}
  a_{J->I,q,c_out,c_in}^{lambda;mu}
  B_{J->I,q}^{lambda;mu}(m_J^{t+1,c_in,mu}).
```

Again, learned weights mix channels/path/multiplicities, while fixed `B` handles representation coordinates.

For `M = 3`, the useful branching rules are:

```text
order 2 -> 1:
  (2)     -> (1)
  (1,1)   -> (1)

order 3 -> 2:
  (3)       -> (2)
  (2,1)     -> (2) + (1,1)
  (1,1,1)   -> (1,1)

order 3 -> 1:
  (3)       -> (1)
  (2,1)     -> (1)
  (1,1,1)   -> 0
```

### 0.5 SpechtMP layer semantics

A SpechtMP layer should do:

1. **Fusion/intertwiner step**: build messages on subsets `U` with `|U| <= M_virtual` using `C`.
2. **Branching step**: compress messages back to persistent features with `|I| <= M` using `B`.
3. **Channel mixing / activation**: apply equivariant nonlinear or tensor-product activations. Any learned map must preserve irrep-coordinate equivariance.

The persistent cap `M` controls stored feature order. The virtual cap `M_virtual` controls the largest temporarily formed interaction order. Dense explicit `M_virtual > 3` is not allowed in the first implementation unless using a low-rank branch-down approximation.

### 0.6 Pfaffian-readout view of the irreps

Since the practical readout is Pfaffian, the terminal object is always a pair-antisymmetric kernel:

```text
K_ij in pair irrep A = (1,1).
```

Use the shorthand (not in code, just in explanation):

```text
S = (2)
A = (1,1)
T = (3)
V = (2,1)
E = (1,1,1)
```

Triple-to-pair branching gives:

```text
T -> S
V -> S + A
E -> A
```

Thus:

```text
A, V, E are sign-carrying/core sectors for the Pfaffian kernel.
h, S, T are gate/context sectors.
```

A practical gated kernel should look like:

```text
K_ij = symmetric_gate_ij * antisymmetric_carrier_ij
```

where:

```text
symmetric_gate_ji = symmetric_gate_ij
antisymmetric_carrier_ji = -antisymmetric_carrier_ij
```

This guarantees `K_ji = -K_ij`.

A minimal carrier can be:

```text
bar_a_ij = a_ij
         + sum_{k != i,j} B_{ijk -> ij}^{A;V}(v_ijk)
         + sum_{k != i,j} B_{ijk -> ij}^{A;E}(e_ijk)
```

A symmetric gate can be built from:

```text
bar_s_ij = s_ij
         + sum_{k != i,j} B_{ijk -> ij}^{S;T}(t_ijk)
         + sum_{k != i,j} B_{ijk -> ij}^{S;V}(v_ijk)
```

Then construct one or more skew kernels:

```text
K_l[i,j] = F_l(bar_a_ij, bar_s_ij, h_i, h_j)
```

with explicit antisymmetry enforcement, e.g.

```text
K_l[i,j] = G_l(bar_s_ij, h_i, h_j) * A_l(bar_a_ij)
```

where `G_l` is symmetric in `i,j` and `A_l` is antisymmetric.

### 0.7. How to calculate the intertwiner maps

There are two kinds of fixed equivariant maps in this project:

1. **Fusion intertwiners**
   \[
   C_{I_1,I_2\to U,p}^{\lambda;\lambda_1,\lambda_2}
   :
   S_{I_1}^{\lambda_1}\otimes S_{I_2}^{\lambda_2}
   \to
   S_U^\lambda,
   \qquad U=I_1\cup I_2.
   \]

2. **Branching intertwiners**
   \[
   B_{J\to I,q}^{\lambda;\mu}
   :
   S_J^\mu
   \to
   S_I^\lambda,
   \qquad I\subseteq J.
   \]

The indices \(p,q\) label multiplicity/path indices when more than one independent equivariant map exists.

These maps are **not learned**. They are fixed representation-theoretic tensors. The neural network only learns channel/path weights multiplying them.

---

#### 0.7.1. Intertwiner definition

Let \(G\) be the relevant symmetric group acting on a target subset, usually

\[
G=S_{|U|}
\]

for fusion into \(U\), or

\[
G=S_{|I|}
\]

for branching down to \(I\).

Suppose

\[
\rho_{\mathrm{src}}:G\to GL(V_{\mathrm{src}}),
\qquad
\rho_{\mathrm{tgt}}:G\to GL(V_{\mathrm{tgt}}).
\]

An intertwiner is a linear map

\[
A:V_{\mathrm{src}}\to V_{\mathrm{tgt}}
\]

satisfying

\[
\boxed{
A\rho_{\mathrm{src}}(g)=\rho_{\mathrm{tgt}}(g)A
\qquad
\forall g\in G.
}
\]

Equivalently,

\[
A\in \operatorname{Hom}_{G}(V_{\mathrm{src}},V_{\mathrm{tgt}}).
\]

In code, it is enough to enforce this for the adjacent transposition generators

\[
s_i=(i\ i+1),
\qquad i=1,\dots,k-1,
\]

because they generate \(S_k\).

So for each generator \(s_i\), impose

\[
A\rho_{\mathrm{src}}(s_i)-\rho_{\mathrm{tgt}}(s_i)A=0.
\]

Flatten \(A\) into a vector and solve the resulting homogeneous linear system. A basis of the nullspace gives all independent intertwiner paths.

---

#### 0.7.2. Direct nullspace algorithm

This is the most general way to compute \(C\) or \(B\).

Given source representation matrices

\[
\rho_{\mathrm{src}}(s_i)\in\mathbb R^{d_{\mathrm{src}}\times d_{\mathrm{src}}},
\]

and target representation matrices

\[
\rho_{\mathrm{tgt}}(s_i)\in\mathbb R^{d_{\mathrm{tgt}}\times d_{\mathrm{tgt}}},
\]

we want

\[
A\in\mathbb R^{d_{\mathrm{tgt}}\times d_{\mathrm{src}}}.
\]

For each generator \(s_i\),

\[
A\rho_{\mathrm{src}}(s_i)-\rho_{\mathrm{tgt}}(s_i)A=0.
\]

Using vectorization,

\[
\operatorname{vec}(A\rho_{\mathrm{src}})
=
\rho_{\mathrm{src}}^\top\otimes I_{d_{\mathrm{tgt}}}\operatorname{vec}(A),
\]

and

\[
\operatorname{vec}(\rho_{\mathrm{tgt}}A)
=
I_{d_{\mathrm{src}}}\otimes \rho_{\mathrm{tgt}}\operatorname{vec}(A).
\]

So each generator contributes the linear constraint

\[
\boxed{
\left(
\rho_{\mathrm{src}}(s_i)^\top\otimes I_{d_{\mathrm{tgt}}}
-
I_{d_{\mathrm{src}}}\otimes \rho_{\mathrm{tgt}}(s_i)
\right)
\operatorname{vec}(A)
=0.
}
\]

Stack these constraints over all adjacent transpositions and compute the nullspace.

Pseudocode:

```python
def intertwiner_basis(rho_src_gens, rho_tgt_gens, atol=1e-10):
    """
    Compute a basis for Hom_G(V_src, V_tgt).

    rho_src_gens: list of [d_src, d_src] tensors
    rho_tgt_gens: list of [d_tgt, d_tgt] tensors

    returns:
        maps: [num_paths, d_tgt, d_src]
    """
    d_src = rho_src_gens[0].shape[0]
    d_tgt = rho_tgt_gens[0].shape[0]

    blocks = []
    I_src = eye(d_src)
    I_tgt = eye(d_tgt)

    for R_src, R_tgt in zip(rho_src_gens, rho_tgt_gens):
        # Constraint:
        # vec(A R_src - R_tgt A) = 0
        block = kron(R_src.T, I_tgt) - kron(I_src, R_tgt)
        blocks.append(block)

    L = cat(blocks, dim=0)

    # Use SVD/nullspace on CPU for stability.
    U, S, Vh = svd(L)
    null_mask = S < atol
    # If using full_matrices=True, nullspace is the trailing rows of Vh.
    null_vectors = Vh[-num_null:, :]

    maps = null_vectors.reshape(num_null, d_tgt, d_src)
    maps = orthonormalize_maps(maps)
    return maps
```

### 0.8 Determinant, Pfaffian, and hyperpfaffian interpretation

The general antisymmetric readout is an `S_n`-intertwiner into the global sign representation `(1^n)`. For order-1 features, the degree-`n` sign intertwiner is the Levi-Civita contraction and yields determinants/multideterminants. For order-2 antisymmetric pair features, the minimal sign intertwiner yields Pfaffians. For order-3 antisymmetric triple features, the analogous direct readout is a hyperpfaffian-like hypermatching sum.

Implementation policy:

- determinant readout is optional;
- Pfaffian readout is the main default;
- direct order-3 hyperpfaffian readout is not part of the first implementation because generic dense hyperpfaffians are combinatorial and not expected to have an `O(n^3)` algorithm;
- branching triples into pairs and then using a Pfaffian is not equivalent to a true hyperpfaffian. It is a tractable way to let triple information influence the pair kernel.

### 0.9 Complexity targets

Dense persistent features up to `M = 3` have memory/time scaling at worst `O(n^3)` for fixed channel width. For each target subset `U` with `|U| <= 3`, enumerate only source subsets `I1,I2 subset U` with `I1 union I2 = U`. Do not globally loop over all source subsets and then check unions.

Implementation pattern:

```text
for each node i:
  build order-1 messages

for each pair (i,j):
  build order-2 messages from subsets of {i,j}

for each triple (i,j,k):
  build order-3 messages from subsets of {i,j,k}
```

Dense virtual order 4 would cost `O(n^4)` but shoulw be implemented directly in the first version.

We will later consider using low-rank branch-down approximations that compute only the effect on pair `S/A` features, such as:

```text
Delta a_ij^(4) = sum_r [ U_ij^{r,S} V_ij^{r,A} + U_ij^{r,A} V_ij^{r,S} - diagonal_corrections ]
Delta s_ij^(4) = sum_r [ U_ij^{r,S} V_ij^{r,S} + U_ij^{r,A} V_ij^{r,A} - diagonal_corrections ]
```

with

```text
U_ij^{r,rho} = sum_{k != i,j} U_ijk^{r,rho}
V_ij^{r,rho} = sum_{l != i,j} V_ijl^{r,rho}
```

The target complexity for this virtual-4 approximation is approximately `O(n^3 R)` rather than `O(n^4)`.
But it is not of high important right now.

### 0.10 Correctness requirements from the math

The implementation must pass these mathematical checks before serious training:

1. Encoder equivariance:

```text
features(sigma . X) = sigma . features(X)
```

2. SpechtMP equivariance:

```text
SpechtMP(sigma . features) = sigma . SpechtMP(features)
```

3. Branching symmetry:

```text
bar_s_ji = bar_s_ij
bar_a_ji = -bar_a_ij
```

4. Pfaffian readout antisymmetry:

```text
psi(sigma . X) = sgn(sigma) psi(X)
```

5. Terminal skew-kernel condition:

```text
K_ji = -K_ij
```

6. Learned maps must not mix transforming irrep coordinates arbitrarily. Learned maps can mix channel indices and multiplicity/path indices. Fixed representation maps `C` and `B` handle irrep-coordinate contractions.


---

## 1. High-level design principle

Separate responsibilities into four layers:

```text
representation theory  |  neural model layers  |  QMC physics  |  sampling/training
```

The model should not know about Monte Carlo. The sampler should not know about Specht irreps. The Hamiltonian should treat the model as a black box that evaluates the wavefunction and provides derivatives through PyTorch autograd.

The conceptual dependency graph should be:

```text
configs
   |
   v
Hydra instantiate
   |
   +--> model = Encoder + SpechtMP + Readout + Cusp
   |
   +--> hamiltonian
   |
   +--> sampler
   |
   +--> loss
   |
   v
trainer
```

---

### 1.1 External package interface strategy

PyTorch remains the core runtime and autograd backend for model evaluation,
local-energy computation, and training.

Optional packages must enter through explicit wrappers in their owning packages
or through tests. They must not pollute runtime core imports, and core modules
should remain usable without installing optional scientific, JAX, C++, or
symbolic packages.

Package roles:

- PySCF: optional later wrapper for molecule/system conversion and reference
  data. The first milestone remains the custom two-electron harmonic trap, but
  Hamiltonian and `System` conventions should be designed with PySCF
  compatibility in mind.
- SciPy/SymPy: acceptable optional offline/dev/test utilities for numerical
  linear algebra, exact combinatorics, permutations, partitions, and sanity
  checks.
- `pfapack`: optional test-only validation backend for Pfaffian values. It must
  never be used in the differentiable training path.
- WandB: configurable logging that is easy to toggle, and expected to be a
  normal working mode when enabled by config.
- DeepQMC: reference plus optional backend experiment only. Direct reuse means
  adopting its JAX/Haiku ansatz interface and training framework.
- NetKet: optional JAX prototype candidate because it has continuous-particle
  VMC machinery. It is not a PyTorch dependency.
- jVMC and specific NN-QMC implementations: reference only.
- QMCPACK: borrow established sampler/local-energy diagnostics and benchmark
  conventions, not direct code.
- Sage/passagemath: external/dev fixture generation and validation only.
  Generated fixtures must include provenance.

---

## 2. Proposed source tree

Use a package layout like this:

```text
spenn/
  pyproject.toml
  README.md

  configs/
    config.yaml
    model/
      spenn_pf.yaml
      spenn_det.yaml
      spenn_det_pf.yaml
    encoder/
      basic.yaml
      cusp.yaml
    spechtmp/
      M3.yaml
      M3_virtual4_lowrank.yaml
    readout/
      pfaffian.yaml
      determinant.yaml
      det_pf_sum.yaml
    hamiltonian/
      electronic.yaml
    sampler/
      metropolis.yaml
      mala.yaml
    trainer/
      vmc.yaml
    logging/
      wandb.yaml
    hardware/
      single_gpu.yaml
      ddp.yaml

  scripts/
    train.py
    eval.py
    sample.py
    debug_equivariance.py
    debug_local_energy.py

  spenn/
    __init__.py

    registry.py
    types.py

    utils/
      tensor_utils.py
      index_utils.py
      logging.py
      checkpointing.py
      distributed.py
      profiling.py

    reps/
      __init__.py
      permutations.py
      specht.py
      young.py
      character_tables.py
      irreps.py
      branch.py
      fusion.py
      fourier.py
      cached_maps.py
      fixture_generators/
        sage_specht.py
        passagemath_specht.py

    data/
      __init__.py
      partitions.py
      feature_dict.py
      subset_index.py
      irrep_tensor.py

    nn/
      __init__.py
      encoding/
        __init__.py
        base.py
        electron_features.py
        distance_features.py
        cusp.py
      spechtmp/
        __init__.py
        layer.py
        fuser.py
        brancher.py
        lowrank_virtual.py
      readout/
        __init__.py
        determinant.py
        pfaffian.py
        sum_readout.py
      wavefunction.py
      activations.py
      channel_mixing.py

    physics/
      __init__.py
      systems.py
      hamiltonian.py
      local_energy.py
      kinetic.py
      potential.py
      cusp.py

    sampling/
      __init__.py
      walkers.py
      metropolis.py
      mala.py
      moves.py
      equilibrate.py

    losses/
      __init__.py
      vmc.py
      energy.py
      variance.py

    training/
      __init__.py
      trainer.py
      optimizer.py
      scheduler.py
      callbacks.py
      metrics.py

    tests/
      test_equivariance.py
      test_antisymmetry.py
      test_branching.py
      test_intertwiners.py
      test_pfaffian.py
      test_local_energy.py
      fixtures/
        specht/
          README.md
          partitions.json
          young_tableaux.json
          specht_dimensions.json
          generator_matrices.npz
          branching.json
          fusion.json
        qmc/
          README.md
          harmonic_trap.json
          local_energy.json
```

Optional integrations must not be imported by core runtime modules at package
import time. PySCF-style system conversion belongs with data or physics code,
Pfaffian reference checks belong in tests, and backend experiments should live
behind explicit experimental entrypoints.

---

## 3. Core package responsibilities

### 3.1 `reps/`: fixed Specht and representation-theoretic machinery

This module should contain **no learnable parameters**. It should only construct, cache, and return fixed tensors/matrices for representation-theoretic maps.

Runtime Specht logic lives in `spenn/reps/`. SpechtMP consumes fixed maps from
this layer and must not own Sage/passagemath generation logic.

Sage/passagemath scripts may live in `spenn/reps/fixture_generators/`, but they
are optional/dev-only. Runtime modules and normal tests must not import them
unless an explicit optional validation command requests that path.

It should provide:

- partitions of small integers;
- permutation utilities;
- Specht/Young bases;
- irrep dimensions and metadata;
- branching maps;
- fusion/intertwiner maps;
- Fourier transforms between tuple and Specht bases if needed;
- cached fixed maps for repeated use in model layers.

Important maps:

```text
C_{I1,I2 -> U}^{lambda; lambda1, lambda2}
B_{J -> I}^{lambda; mu}
P^lambda
R^lambda
```

Suggested classes:

```python
@dataclass(frozen=True)
class SpechtBasis:
    order: int
    partition: Partition
    dim: int
    basis_name: str


@dataclass(frozen=True)
class FusionMap:
    source_orders: tuple[int, int]
    target_order: int
    source_irreps: tuple[Partition, Partition]
    target_irrep: Partition
    path_index: int
    tensor: torch.Tensor


@dataclass(frozen=True)
class BranchMap:
    source_order: int
    target_order: int
    source_irrep: Partition
    target_irrep: Partition
    path_index: int
    tensor: torch.Tensor
```

All fixed maps should be cacheable on disk and movable to the requested device/dtype,
but just in case a higher-order map is requested, there should be machinery that calculates these maps.

---

### 3.2 `data/`: feature containers

Avoid passing unstructured nested dictionaries everywhere. Use a disciplined `FeatureDict` wrapper.

A logical feature layout is:

```text
features[order][irrep] -> tensor
```

`FeatureDict` stores canonical `Partition` keys internally. Tuple/list/string/int
partition specs are accepted by `get`, `set`, and `has` for convenience.

A good tensor convention is:

```text
order 1:
  features[1][(1)]       : [batch, n, C, d_irrep]

order 2:
  features[2][(2)]       : [batch, n, n, C, 1]
  features[2][(1,1)]     : [batch, n, n, C, 1]

order 3:
  features[3][(3)]       : [batch, n, n, n, C, 1, 1]
  features[3][(2,1)]     : [batch, n, n, n, C, 2, 2]
  features[3][(1,1,1)]   : [batch, n, n, n, C, 1, 1]
```

The exact shape can change, but the external API should hide those details.

Suggested API:

```python
class FeatureDict:
    def get(self, order: int, irrep: PartitionLike) -> torch.Tensor:
        ...

    def set(self, order: int, irrep: PartitionLike, value: torch.Tensor) -> None:
        ...

    def has(self, order: int, irrep: PartitionLike) -> bool:
        ...

    def items(self):
        ...

    def to(self, device=None, dtype=None):
        ...
```

Use python magic methods to simplify access if possible. 

---

## 4. Neural model modules

### 4.1 Encoding layer

The encoder maps electron coordinates/spins into initial Specht features.

```python
class Encoder(nn.Module):
    def forward(self, batch: ElectronBatch) -> FeatureDict:
        ...
```

Initial features may include (again, use the x^lambda_I convention for maximum extendability, not h,s,a,t,v,e):

```text
h_i^c                     order 1, irrep (1)
s_{ij}^c                  order 2, irrep (2)
a_{ij}^c                  order 2, irrep (1,1)
t_{ijk}^c                 order 3, irrep (3)
v_{ijk,alpha,beta}^c      order 3, irrep (2,1)
e_{ijk}^c                 order 3, irrep (1,1,1)
```

Keep the tensor indices as tuples, so we should expect for example:
```text
s_{ij} = s_{ji}
a_{ij} = -a_{ji}
t_{ijk} = t_{\sigma(ijk)}
e_{ijk} = sgn(\sigma) t_{\sigma(ijk)}
```

Start simple:

- one-body electron features;
- pair distances and relative-coordinate features;
- antisymmetric pair channels if needed;
- optional triples after the pair-only prototype works.

Do not make the encoder responsible for Hamiltonian or sampling logic.

---

### 4.2 Electron-electron cusp factor

For practicality, implement cusps as separate modular ansatz factors under
`spenn.nn.cusp`. Do not bury them deep inside SpechtMP or the physics package.

Use a separate module:

```python
class ElectronElectronCusp(nn.Module):
    def forward(self, batch: ElectronBatch) -> torch.Tensor:
        """Return cusp contribution to log|psi| with shape [batch]."""
        ...
```

Then the wavefunction can do:

```python
out.logabs = out.logabs + cusp_log
```

Electron-electron and electron-nucleus cusps are the active analytic modules;
future nuclear-feature cusps should keep using the `spenn.nn.cusp` namespace.

---

### 4.3 SpechtMP

SpechtMP should be a stack of layers. Each layer should contain two conceptual sublayers:

```text
SpechtFuser  ->  SpechtBrancher
```

The fuser step builds tensor-product messages using fixed Specht fusion maps. The brancher compresses virtual messages back into retained orders.

#### SpechtFuser

Computes messages of the form:

```text
x_{I1}^{lambda1} tensor x_{I2}^{lambda2}
  --C-->
m_U^lambda
```

Suggested class:

```python
class SpechtFuser(nn.Module):
    def __init__(
        self,
        M: int,
        M_virtual: int,
        channels: dict,
        fusion_maps: FusionMapCache,
        use_lowrank_virtual: bool = False,
    ):
        ...

    def forward(self, features: FeatureDict) -> FeatureDict:
        ...
```

#### SpechtBrancher

Computes branch/pooling updates:

```text
m_J^mu --B--> x_I^lambda
```

Suggested class:

```python
class SpechtBrancher(nn.Module):
    def __init__(
        self,
        M: int,
        M_virtual: int,
        channels: dict,
        branch_maps: BranchMapCache,
    ):
        ...

    def forward(self, messages: FeatureDict, residual: FeatureDict | None = None) -> FeatureDict:
        ...
```

#### SpechtMPLayer

```python
class SpechtMPLayer(nn.Module):
    def __init__(self, fuser, brancher, activation, residual=True, norm=None):
        super().__init__()
        self.fuser = fuser
        self.brancher = brancher
        self.activation = activation
        self.residual = residual
        self.norm = norm

    def forward(self, features: FeatureDict) -> FeatureDict:
        messages = self.fuser(features)
        features = self.brancher(messages, residual=features if self.residual else None)
        features = self.activation(features)
        if self.norm is not None:
            features = self.norm(features)
        return features
```

#### SpechtMP stack

```python
class SpechtMP(nn.Module):
    def __init__(self, layers: list[SpechtMPLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, features: FeatureDict) -> FeatureDict:
        for layer in self.layers:
            features = layer(features)
        return features
```

---

## 5. Readout modules

Readouts should share a common interface:

```python
@dataclass
class WavefunctionOutput:
    logabs: torch.Tensor
    sign: torch.Tensor
    phase: torch.Tensor | None = None
    aux: dict = field(default_factory=dict)


class Readout(nn.Module):
    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        ...
```

### 5.1 Pfaffian readout

The practical default should be a Pfaffian readout over effective pair-antisymmetric kernels:

```text
K_{ij} = -K_{ji}
psi = sum_l b_l Pf(K^(l))
```

The Pfaffian readout should build one or more skew matrices from final features:

```python
class PfaffianReadout(nn.Module):
    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        K = self.build_skew_kernel(features, batch)
        logabs, sign = torch_pfaffian_logabs_sign(K)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"K": K})
```

For even electron count:

```text
psi = sum_l b_l Pf(K_l)
```

For odd electron count, use a bordered Pfaffian:

```text
K_tilde = [[K, u], [-u^T, 0]]
psi = Pf(K_tilde)
```

Important: implement a stable signed-log sum for sums of Pfaffians.

### 5.2 Determinant readout

Also support a determinant readout, even if the current main route is Pfaffian:

```python
class DeterminantReadout(nn.Module):
    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        A = self.build_orbital_matrix(features, batch)
        sign, logabs = torch.linalg.slogdet(A)
        return WavefunctionOutput(logabs=logabs, sign=sign, aux={"A": A})
```

### 5.3 Sum readout

Allow determinant and Pfaffian readouts to be combined:

```python
class SumReadout(nn.Module):
    def __init__(self, readouts: list[Readout], learn_weights: bool = True):
        ...

    def forward(self, features: FeatureDict, batch: ElectronBatch) -> WavefunctionOutput:
        outs = [r(features, batch) for r in self.readouts]
        return signed_logsumexp_wavefunction_outputs(outs, weights=self.weights)
```

---

## 6. Pfaffian-readout pruning rule

Since the main readout is Pfaffian, the final target is a pair-antisymmetric kernel:

```text
K_ij in irrep (1,1)
```

Therefore, for readout purposes, keep only paths that produce either:

```text
pair A = (1,1) carriers
pair S = (2) gates that multiply A carriers
```

For M = 3, use:

```text
S = (2)
A = (1,1)
T = (3)
V = (2,1)
E = (1,1,1)
```

Branching to pairs:

```text
T -> S
V -> S + A
E -> A
```

Thus:

```text
A, V, E are core sign-carrying sectors.
h, S, T are useful gate/context sectors.
```

A minimal Pfaffian model may keep only:

```text
A, V, E
```

A practical gated Pfaffian model should keep all sectors, but force all readout paths to pass through:

```text
S gate x A carrier -> K_ij in A -> Pf(K)
```

---

## 7. Full wavefunction module

The full model should be a clean composition:

```python
class SpENNWavefunction(nn.Module):
    def __init__(self, encoder, spechtmp, readout, cusp=None):
        super().__init__()
        self.encoder = encoder
        self.spechtmp = spechtmp
        self.readout = readout
        self.cusp = cusp

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        features = self.encoder(batch)
        features = self.spechtmp(features)
        out = self.readout(features, batch)

        if self.cusp is not None:
            cusp_log = self.cusp(batch)
            out.logabs = out.logabs + cusp_log

        return out
```

---

## 8. Hamiltonian and local energy

The Hamiltonian should be independent of model internals. It should only call the model and use PyTorch autograd for derivatives.

Suggested interface:

```python
class Hamiltonian(nn.Module):
    def local_energy(self, model: nn.Module, batch: ElectronBatch) -> torch.Tensor:
        ...
```

For electronic Hamiltonians:

```python
class ElectronicHamiltonian(Hamiltonian):
    def potential(self, batch: ElectronBatch) -> torch.Tensor:
        ...

    def kinetic(self, model: nn.Module, batch: ElectronBatch) -> torch.Tensor:
        ...

    def local_energy(self, model: nn.Module, batch: ElectronBatch) -> torch.Tensor:
        return self.kinetic(model, batch) + self.potential(batch)
```

Start with an autograd Laplacian implementation. Later, optimize with `vmap`, forward-mode AD, custom kinetic estimators, or analytic derivatives where possible.

Keep local energy code isolated because it will likely become a performance bottleneck.

---

## 9. Monte Carlo sampling

Sampling should be separate from the model. The sampler owns walkers and proposes moves.

Suggested walker object:

```python
@dataclass
class Walkers:
    positions: torch.Tensor
    spins: torch.Tensor | None = None
    logabs: torch.Tensor | None = None
    sign: torch.Tensor | None = None
    aux: dict = field(default_factory=dict)
```

Suggested sampler interface:

```python
class Sampler:
    def initialize(self, system, n_walkers: int, device: torch.device) -> Walkers:
        ...

    def step(self, model: nn.Module, walkers: Walkers) -> Walkers:
        ...

    def sample(self, model: nn.Module, walkers: Walkers, n_steps: int) -> Walkers:
        for _ in range(n_steps):
            walkers = self.step(model, walkers)
        return walkers
```

Implement first:

```text
MetropolisSampler
```

Then optionally:

```text
MALASampler / LangevinSampler
```

The sampler should handle batch shape:

```text
[n_walkers, n_electrons, spatial_dim]
```

---

## 10. Loss functions

VMC loss module should take:

```text
model, hamiltonian, samples
```

and return:

```text
loss, metrics
```

Suggested interface:

```python
class VMCLoss(nn.Module):
    def forward(self, model, hamiltonian, batch):
        local_e = hamiltonian.local_energy(model, batch)
        loss = local_e.mean()
        metrics = {
            "energy": local_e.mean().detach(),
            "variance": local_e.var(unbiased=False).detach(),
        }
        return loss, metrics
```

Later support the standard VMC covariance-gradient estimator:

```text
grad E = 2 < (E_L - <E_L>) grad log|psi| >
```

Design the loss API so this can be swapped in later.

---

## 11. Training loop

The trainer owns the training loop and connects model, sampler, Hamiltonian, loss, optimizer, scheduler, checkpoints, and logging.

Suggested class:

```python
class VMCTrainer:
    def __init__(self, model, hamiltonian, sampler, loss, optimizer, logger, cfg):
        ...

    def train_step(self):
        self.walkers = self.sampler.sample(self.model, self.walkers, self.cfg.sampler.steps_per_iter)
        loss, metrics = self.loss(self.model, self.hamiltonian, self.walkers)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.log(metrics)
        return metrics

    def fit(self):
        for step in range(self.cfg.trainer.max_steps):
            metrics = self.train_step()
            ...
```

Hydra should instantiate everything in `scripts/train.py`.

---

## 12. Hydra config design

Top-level config:

```yaml
# configs/config.yaml
defaults:
  - model: spenn_pf
  - hamiltonian: electronic
  - sampler: metropolis
  - trainer: vmc
  - logging: wandb
  - hardware: single_gpu
  - _self_

seed: 1234
dtype: float64
```

Example model config:

```yaml
# configs/model/spenn_pf.yaml
_target_: spenn.nn.wavefunction.SpENNWavefunction

encoder:
  _target_: spenn.nn.encoding.ElectronPairTripleEncoder
  M: 3
  channels:
    order1:
      "(1)": 32
    order2:
      "(2)": 32
      "(1,1)": 32
    order3:
      "(3)": 16
      "(2,1)": 16
      "(1,1,1)": 16

spechtmp:
  _target_: spenn.nn.spechtmp.layer.SpechtMP
  layers:
    - _target_: spenn.nn.spechtmp.layer.SpechtMPLayer
      fusion_map:
        _target_: spenn.reps.fusion.FusionMap
        M: 2
        M_virtual: 2
      message_head:
        _target_: spenn.nn.spechtmp.message_head.MessageHead
        M: 2
        M_virtual: 2
        channels: [0, 32, 32]
        include_linear: true
        activation:
          _target_: spenn.nn.activations.ActivationByType
          symmetric:
            _target_: spenn.nn.activations.ElementwiseFeatureActivation
            activation:
              _target_: torch.nn.Sigmoid
          antisymmetric:
            _target_: spenn.nn.activations.ElementwiseFeatureActivation
            activation:
              _target_: torch.nn.Tanh
          tensor:
            _target_: spenn.nn.activations.NormGateActivation
            activation:
              _target_: torch.nn.Sigmoid
      branch_map:
        _target_: spenn.reps.branch.BranchMap
        M: 2
        M_virtual: 2
      update_head:
        _target_: spenn.nn.spechtmp.update_head.UpdateHead
        M: 2
        channels: [0, 32, 32]
      update:
        _target_: spenn.nn.update.ResidualUpdate

readout:
  _target_: spenn.nn.readout.PfaffianReadout
  num_pfaffians: 8
  use_symmetric_gates: true

cusp:
  _target_: spenn.nn.cusp.ElectronElectronCusp
  enabled: true
```

Example virtual-4 config:

```yaml
# configs/spechtmp/M3_virtual4_lowrank.yaml
M: 3
M_virtual: 4
virtual4:
  enabled: true
  method: lowrank_branch_down
  rank: 16
```

---

## 13. WandB logging

Add optional WandB support through config.

```yaml
# configs/logging/wandb.yaml
enabled: true
project: spenn
entity: null
name: null
tags: []
log_model: false
```

Only rank 0 should log when distributed training is enabled.

Track at least:

```text
energy
variance
acceptance_rate
step_size
loss
grad_norm
parameter_norm
local_energy_histogram
```

---

## 14. Parallelism and scaling

Start with three levels of parallelism.

### Level 1: walker batching

Everything should support batched walkers:

```text
[batch, n_electrons, spatial_dim]
```

This is the first and most important path to GPU utilization.

### Level 2: multi-GPU DDP

Use one process per GPU. Each process owns independent walkers. Gradients are averaged by DDP.

```text
rank 0: walkers 0..B
rank 1: walkers B..2B
...
```

Implement helpers in:

```text
spenn/utils/distributed.py
```

Suggested helper functions:

```python
def setup_distributed(cfg):
    ...

def is_main_process() -> bool:
    ...

def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    ...
```

### Level 3: later distributed/asynchronous sampling

Do not implement asynchronous sampling initially. Keep the first version simple.

---

## 15. Low-rank `M_virtual = 4` support

Materialize dense order-4 tensors at first. We will later implement low-rank versions:
only the part of virtual order-4 information that branches back to pair `S` or `A` features for the Pfaffian readout.

The useful targets are:

```text
Delta s_ij^(4)
Delta a_ij^(4)
```

A good low-rank branch-down approximation is:

```text
Delta a_ij^(4) = sum_r [ U_ij^{r,S} V_ij^{r,A} + U_ij^{r,A} V_ij^{r,S} - diagonal_corrections ]

Delta s_ij^(4) = sum_r [ U_ij^{r,S} V_ij^{r,S} + U_ij^{r,A} V_ij^{r,A} - diagonal_corrections ]
```

where:

```text
U_ij^{r,rho} = sum_{k != i,j} U_ijk^{r,rho}
V_ij^{r,rho} = sum_{l != i,j} V_ijl^{r,rho}
```

This gives virtual four-body effects of the form `(i,j,k,l)` while aiming for approximately:

```text
O(n^3 R)
```

rather than `O(n^4)`.

Implement this later in:

```text
nn/spechtmp/lowrank_virtual.py
```

---

## 16. Required tests

Before serious training, implement these tests.

Normal tests must not require PySCF, NetKet, DeepQMC, Sage, passagemath, or
`pfapack`. Optional package-backed validation tests may be marked separately and
run only when the corresponding package is available or explicitly requested.

`pfapack` is for Pfaffian comparisons only. It is never part of the
differentiable training path.

Sage/passagemath fixtures should be plain JSON, NPZ, or text files. Each
generated fixture should record generator package/version, command, date, and
basis-convention notes so the data can be regenerated and audited.

### 16.1 Equivariance tests

File:

```text
spenn/tests/test_equivariance.py
```

Check:

```text
F(sigma X) = sigma F(X)
```

for encoder and SpechtMP.

### 16.2 Antisymmetry tests

File:

```text
spenn/tests/test_antisymmetry.py
```

Check:

```text
psi(sigma X) = sgn(sigma) psi(X)
```

for determinant, Pfaffian, and sum readouts.

### 16.3 Branching tests

File:

```text
spenn/tests/test_branching.py
```

Check pair symmetry after branching:

```text
s_ji = s_ij
a_ji = -a_ij
```

### 16.4 Intertwiner tests

File:

```text
spenn/tests/test_intertwiners.py
```

Check that fixed `C` and `B` maps commute with the relevant group actions.

### 16.5 Pfaffian tests

File:

```text
spenn/tests/test_pfaffian.py
```

Compare the implemented Pfaffian to brute-force Pfaffians for small `n`.

### 16.6 Local energy tests

File:

```text
spenn/tests/test_local_energy.py
```

Compare kinetic/local energy against analytic wavefunctions when possible.

---

## 17. Suggested implementation order

Implement in this order:

1. `FeatureDict`
2. minimal `ElectronBatch` / `Walkers` dataclasses
3. simple encoder with only `h`, `s`, `a`
4. Pfaffian readout
5. antisymmetry tests
6. Metropolis sampler
7. simple Hamiltonian/local energy for a toy system
8. VMC trainer
9. WandB logging
10. Hydra config instantiation
11. SpechtMP with `M=2`
12. SpechtMP with `M=3`
13. determinant and sum readouts
14. low-rank `M_virtual=4`
15. DDP support

Do not start with full `M=3`, full Hamiltonian, and low-rank virtual order-4 all at once.

---

## 18. Coding requirements

Use PyTorch as the main tensor backend.

Prefer:

```text
float64 for VMC experiments initially
float32/bfloat16 only after correctness is established
```

Design every module so it accepts:

```text
device
dtype
batch dimension
```

Avoid hard-coding electron counts except where the Hamiltonian/system object defines them.

Use type hints and dataclasses for configs and data containers where helpful.

---

## 19. Important conceptual constraints

1. The model output should be exactly antisymmetric if the readout is determinant/Pfaffian-based.

2. SpechtMP is an encoder. If the final readout is Pfaffian, the ansatz is functionally a neural Pfaffian with a Specht-module kernel generator.

3. Do not claim direct order-3 hyperpfaffian readout unless it is explicitly implemented. Branching triples down to pair kernels is not equivalent to a hyperpfaffian.

4. For Pfaffian readout, all terminal paths should produce a skew pair kernel:

```text
K_ij = -K_ji
```

5. Symmetric irreps are still useful as gates/context, but they cannot directly be Pfaffian carriers.

6. Dense `M_virtual > 3` is not recommended in the initial implementation if the target is near `O(n^3)`. Warn the user if such happens.

---

## 20. First milestone

The first serious milestone should be:

```text
M = 2
M_virtual = 2
encoder: h, s, a
readout: sum of Pfaffians
sampler: Metropolis
hamiltonian: simple toy electronic Hamiltonian
training: VMC with WandB logging
```

This should pass:

```text
equivariance tests
antisymmetry tests
pfaffian tests
local-energy sanity tests
```

Only after this works should `M=3` and `M_virtual=4` be added.
