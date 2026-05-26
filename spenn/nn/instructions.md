# Instructions for agents working in `nn/`

This file is for coding agents modifying the neural-network side of the project.
The important correction is that the codebase no longer uses canonical subset indexing such as `i < j < k` as the fundamental representation. All particle indices should be treated as explicit ordered tuples.

Examples:

- order 1: `(i)`
- order 2: `(i, j)` and `(j, i)` are both explicit tuple indices
- order 3: `(i, j, k)`, `(i, k, j)`, ..., `(k, j, i)` are explicit tuple indices

Irrep constraints such as symmetry or antisymmetry are encoded by the values and fixed projection/reconstruction maps, not by deleting permuted copies from storage.

---

## 1. Naming convention

SpechtMP has two representation-theoretic neural modules:

1. **Fuser**: performs tensor-product/fusion messages using fixed fusion maps `C`.
2. **Brancher**: brings higher-order messages down to lower-order tuple features using fixed branching maps `B`.

Therefore:

- `nn/spechtmp/fuser.py` contains the tensor-product fuser.
- Code, docs, TODOs, and configs should use `fuser` for tensor-product fusion.
- Code, docs, TODOs, and configs should use `brancher` for branching/pooling.
- Low-rank variants are future fuser or brancher variants. They should not become a third conceptual category.

---

## 2. Difference between `reps/` and `nn/`

The `reps/` package owns fixed mathematics.

It generates and caches tensors such as:

```text
FusionMap.tensor
BranchMap.tensor
projection/reconstruction maps
permutation actions
normalization conventions
```

The `nn/` package owns trainable PyTorch modules.

It should:

- consume cached maps from `reps/`,
- apply them in batched tensor contractions,
- learn only channel/path/multiplicity weights,
- never derive Specht bases or solve representation theory internally.


In short:

```text
reps/ = fixed math tensors
nn/   = trainable modules using those tensors
```
The `nn/` folder should be modified in conjunction with the `reps/` folder.

---

## 3. Tuple-indexed feature convention

Feature tensors should be indexed by explicit ordered particle tuples.

A useful external container layout is:

```python
features[order][partition] -> Tensor
```

Recommended conceptual shape:

```python
[batch, tuple_axes..., channels, irrep_dim, multiplicity_dim]
```

Examples:

```text
features[1][(1)]       : [batch, n, C, 1, mult]
features[2][(2)]       : [batch, n, n, C, 1, mult]
features[2][(1,1)]     : [batch, n, n, C, 1, mult]
features[3][(3)]       : [batch, n, n, n, C, 1, 1]
features[3][(2,1)]     : [batch, n, n, n, C, 2, 2]
features[3][(1,1,1)]   : [batch, n, n, n, C, 1, 1]
```

Tuples with repeated particle indices should normally be masked out or never gathered, depending on implementation. For local antisymmetric irreps, repeated indices should evaluate to zero.

Do not canonicalize `(i, j)` into `{i, j}` or `i < j`. The tuple ordering matters.

---

## 4. Pair normalization convention

Pair projection and reconstruction must include normalization.

Given ordered tuple-basis values

```text
q_ij, q_ji
```

define averaged pair irreps by

```text
s_ij = (q_ij + q_ji) / 2
a_ij = (q_ij - q_ji) / 2
```

Then

```text
s_ji = s_ij
a_ji = -a_ij
```

The inverse reconstruction is

```text
q_ij = s_ij + a_ij
q_ji = s_ij - a_ij
```

This normalization should be used consistently by all code paths. Do not use unnormalized `q_ij + q_ji` or `q_ij - q_ji` in neural modules unless explicitly requested by the map fixture.

The same principle applies to higher-order irreps: projection and reconstruction maps from `reps/` should already encode the chosen normalization. For order-3 symmetrization and antisymmetrization, the tuple-average normalization is `1/6`. Neural modules should not insert additional normalization unless the cached map says to do so.

---

## 5. What the fuser does

The fuser computes tensor-product messages.

Mathematically, for two input tuple features

```text
x^{lambda_1}_{tuple_1}
x^{lambda_2}_{tuple_2}
```

with union tuple/order `U`, the fuser applies a fixed fusion map

```text
C_{pattern, p}^{lambda; lambda_1, lambda_2}
```

to produce a target message

```text
m_U^lambda.
```

In equations:

```text
m_U^{lambda, c_out}
=
 sum_{patterns, lambda_1, lambda_2, c_1, c_2, p}
 w_{p,c_out,c_1,c_2}^{lambda;lambda_1,lambda_2}
 C_{pattern,p}^{lambda;lambda_1,lambda_2}
 (
   x_{tuple_1}^{lambda_1,c_1} tensor x_{tuple_2}^{lambda_2,c_2}
 ).
```

Important implementation meaning:

- `tensor` means an outer product over irrep coordinates.
- `C` contracts that outer product into the target irrep coordinate.
- `w` is learned and mixes channels/path indices.
- `C` is fixed and comes from `reps.fusion`.

The fuser should work with `FusionSpec` or `FusionMap` objects from `reps/`. A fusion spec should tell the fuser:

```text
source orders
source irreps
target order
target irrep
slot pattern / gathering rule
path count
fixed map tensor
```

Because indexing is tuple-based, the slot pattern is crucial. It tells the fuser how to gather source tuples from a target ordered tuple.

Example for target tuple `(i, j, k)`:

- node-pair path: `(i)` and `(j, k)`
- node-pair path: `(j)` and `(i, k)`
- node-pair path: `(k)` and `(i, j)`
- pair-pair path: `(i, j)` and `(i, k)`
- pair-pair path: `(i, j)` and `(j, k)`
- pair-pair path: `(i, k)` and `(j, k)`

The exact coefficients for these paths should come from the fixed `FusionMap.tensor`, not from ad hoc signs in `nn/`.

---

## 6. Hard-coded intuition for fusing h, s, a, t, v, e

Use this section only as intuition or for small debugging. Production code should prefer cached maps from `reps/`.

Shorthand:

```text
h_i       = order-1 feature, irrep (1)
s_ij      = pair symmetric feature, irrep (2)
a_ij      = pair antisymmetric feature, irrep (1,1)
t_ijk     = triple symmetric feature, irrep (3)
v_ijk     = triple mixed feature, irrep (2,1)
e_ijk     = triple antisymmetric feature, irrep (1,1,1)
```

### 6.1 Pair reconstruction/projection

From tuple values:

```text
s_ij = (q_ij + q_ji) / 2
a_ij = (q_ij - q_ji) / 2
```

From irreps to tuple values:

```text
q_ij = s_ij + a_ij
q_ji = s_ij - a_ij
```

### 6.2 Node-node to pair

For two node channels `c1`, `c2`, the normalized pair projections are

```text
m_ij^S = (h_i^{c1} h_j^{c2} + h_j^{c1} h_i^{c2}) / 2
m_ij^A = (h_i^{c1} h_j^{c2} - h_j^{c1} h_i^{c2}) / 2
```

If `c1 == c2` and the features are scalar-valued, the antisymmetric term is zero.

### 6.3 Pair-pair on the same ordered pair

Use parity:

```text
S times S -> S
S times A -> A
A times S -> A
A times A -> S
```

Concretely:

```text
s_ij * s_ij is pair-symmetric
s_ij * a_ij is pair-antisymmetric
a_ij * s_ij is pair-antisymmetric
a_ij * a_ij is pair-symmetric
```

These paths matter for building a Pfaffian skew kernel:

```text
K_ij = symmetric_gate_ij * antisymmetric_carrier_ij.
```

### 6.4 Node + symmetric pair to triple

For target tuple `(i, j, k)`, define

```text
y_i = h_i * s_jk
y_j = h_j * s_ik
y_k = h_k * s_ij
```

Then the ordinary placement-vector projections are

```text
T source: y_i + y_j + y_k
V source: [y_i - y_j, y_j - y_k]
```

With normalized maps, the actual cached fixture may include constants such as `1/6` for full triple averaging or Gram-correction factors. The `nn/` module should not hard-code these constants; it should consume them from `reps/`.

### 6.5 Node + antisymmetric pair to triple

For target tuple `(i, j, k)`, define oriented values

```text
z_i = h_i * a_jk
z_j = h_j * a_ki
z_k = h_k * a_ij
```

Then this contributes to triple antisymmetric and mixed sectors:

```text
E source: z_i + z_j + z_k
V source: fixed sign-twisted V projection from reps/
```

The sign-twisted V projection is basis-dependent. Do not hard-code it in `nn/`; use `FusionMap.tensor`.

---

## 7. What the brancher does

The brancher maps messages from higher-order tuples to lower-order tuples.

Mathematically:

```text
B_{source_order -> target_order, q}^{lambda; mu}
:
S^{mu}_{source tuple} -> S^{lambda}_{target tuple}
```

For example, from triples to pairs:

```text
t_ijk, v_ijk, e_ijk -> s_ij, a_ij
```

Branching rules for `M = 3`:

```text
T=(3)       -> S=(2)
V=(2,1)     -> S=(2) plus A=(1,1)
E=(1,1,1)   -> A=(1,1)
```

So for a Pfaffian readout:

- `A=(1,1)` is the terminal carrier for `K_ij`.
- `V -> A` and `E -> A` are direct triple-to-pair carrier paths.
- `T -> S` and `V -> S` are symmetric gate paths.

The brancher should receive `BranchSpec` or `BranchMap` objects from `reps/`. A branch spec should tell the brancher:

```text
source order
source irrep
target order
target irrep
slot pattern / pooling rule
path count
fixed map tensor
```

Because the implementation is tuple-indexed, a branch from order 3 to order 2 is not merely `{i,j,k} -> {i,j}`. It must specify which source tuple positions correspond to the output tuple.

Example:

```text
source tuple: (i, j, k)
target tuple positions: (0, 1)
target tuple: (i, j)
pooled particle: k
```

The brancher then sums over all valid missing particles while preserving tuple order and excluding repeated particles.

---

## 8. Recommended `nn/spechtmp` files

### `nn/spechtmp/fuser.py`

Should contain:

- `SpechtFuser(nn.Module)`
- consumes cached `FusionMap` / `FusionSpec` objects from `reps.fusion`
- gathers explicit source tuples according to slot patterns
- forms outer products over irrep coordinates
- applies fixed fusion maps
- applies learned channel/path weights
- returns message `FeatureDict`

Should not contain:

- Specht basis generation
- character tables
- Young symmetrizers
- nullspace solvers
- hand-coded mixed-product signs unless strictly for debugging

### `nn/spechtmp/brancher.py`

Should contain:

- `SpechtBrancher(nn.Module)`
- consumes cached `BranchMap` / `BranchSpec` objects from `reps.branching`
- gathers/pools explicit source tuples into target tuples according to slot patterns
- applies fixed branch maps
- applies learned channel/path weights
- returns updated lower-order `FeatureDict`

### `nn/spechtmp/layer.py`

Should contain:

- `SpechtMPLayer(nn.Module)`
- calls `fuser(features)`
- calls `brancher(messages)`
- combines residual/update
- applies normalization/gating/activation if configured

Schematic:

```python
messages = self.fuser(features)
updates = self.brancher(messages)
features = self.update(features, updates)
```

### `nn/spechtmp/lowrank_virtual.py`

DISCARDED: Low-rank modules should be typed as either:

- low-rank fuser, or
- low-rank brancher.

No low-rank approximation is required yet.

---

## 9. Readout-relevant pruning

The primary readout is Pfaffian:

```text
psi = sum_l b_l Pf(K_l)
```

where

```text
K_ji = -K_ij.
```

Build

```text
K_ij = G_ij * abar_ij
```

with

```text
G_ji = G_ij
abar_ji = -abar_ij.
```

Thus all computation paths should eventually contribute either:

1. pair antisymmetric carriers `A=(1,1)`, or
2. pair symmetric gates `S=(2)`.

For `M=3`:

```text
A direct carrier: A
triple carriers: V -> A, E -> A
pair gates: S
triple gates: T -> S, V -> S
```

This does not mean `h`, `S`, or `T` are useless. They can be useful gates/context. But they should not be terminal antisymmetric readout branches by themselves.

---

## 10. Tests required for `nn/`

The neural side should include tests that use cached maps from `reps/`.

Required tests:

1. Pair symmetry:

```text
s_ji = s_ij
a_ji = -a_ij
```

2. Triple symmetry:

```text
e_{perm(i,j,k)} = sign(perm) e_{ijk}
```

3. Fuser equivariance on random features.

4. Brancher equivariance on random messages.

5. Pfaffian antisymmetry:

```text
psi(sigma X) = sign(sigma) psi(X)
```

6. Pfaffian numeric correctness against brute force for small even `n`.
