# Instructions for agents working in `reps/`

This file is for coding agents modifying the representation-theory side of the project.

The important correction is that the project no longer uses canonical subset indexing as the primitive representation. All features and maps should be built for explicit ordered tuples of particles.

For an order `k` feature, the local tuple space is based on ordered tuples

```text
(i_1, ..., i_k)
```

with distinct particle indices unless otherwise specified.

The symmetric group `S_k` acts by permuting tuple positions. Specht irreps are obtained by projecting this ordered tuple space.

---

## 1. Difference between `reps/` and `nn/`

`reps/` owns fixed mathematical data.

It should generate and cache:

- permutations,
- partitions,
- Specht representation metadata,
- projection maps `P^lambda`,
- reconstruction maps `R^lambda`,
- fusion maps `C`,
- branching maps `B`,
- test fixtures.

`reps/` should not contain learned parameters, training logic, or Monte Carlo logic.

`nn/` consumes the maps produced here and learns channel/path weights.

---

## 2. Tuple-space model

For order `k`, define the ordered tuple space

```text
T_k = functions on ordered k-tuples
```

For local representation theory, the abstract dimension is

```text
dim T_k = k!
```

because the local positions can be ordered in `k!` ways.

Use a fixed canonical list of local permutations, for example lexicographic order of permutations of `(0, ..., k-1)`.

Example for `k = 2`:

```text
(0, 1), (1, 0)
```

Example for `k = 3`:

```text
(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)
```

The tuple basis vector corresponding to permutation `pi` represents the ordered tuple

```text
(i_{pi(0)}, ..., i_{pi(k-1)}).
```

---

## 3. Pair normalization convention

For `k = 2`, the ordered tuple basis is

```text
q_ij, q_ji.
```

The averaged pair irreps use the `1/2` convention:

```text
s_ij = (q_ij + q_ji) / 2
a_ij = (q_ij - q_ji) / 2
```

Therefore, the projection matrix from tuple basis to irrep basis is

```text
P_S = [1/2,  1/2]
P_A = [1/2, -1/2]
```

The reconstruction maps undo the averaging convention:

```text
R_S = [1, 1]^T
R_A = [1, -1]^T
```

So

```text
q_ij = s_ij + a_ij
q_ji = s_ij - a_ij
```

All pair projections and fusions must respect this normalization. Tests should fail if unnormalized `q_ij +/- q_ji` is used by accident.
For order-3 symmetrization and antisymmetrization, the analogous tuple-average normalization is `1/6`.

---

## 4. Projection and reconstruction maps

For every order `k` and partition `lambda` of `k`, construct maps

```text
P^lambda : T_k -> S^lambda-block
R^lambda : S^lambda-block -> T_k
```

where `T_k` is ordered tuple space.

Use an orthonormal convention whenever possible:

```text
P^lambda = (R^lambda)^T
```

For scalar irreps:

- `(k)` is fully symmetric.
- `(1^k)` is fully antisymmetric.

For `k = 3`:

```text
T = (3)
V = (2,1)
E = (1,1,1)
```

`T` and `E` are one-dimensional. `V` has transforming dimension 2 and appears with multiplicity 2 in the regular tuple space. The code may store the full isotypic block, or explicitly separate transforming and multiplicity axes. Whichever convention is chosen must be documented and consistent.

Important: if using the non-orthonormal standard basis

```text
u_1 = e_1 - e_2
u_2 = e_2 - e_3
```

for `(2,1)`, then the Gram matrix is not identity. Either:

1. orthonormalize the basis, or
2. store and use the Gram matrix consistently.

The MVP should prefer orthonormal maps to make projections/reconstructions and tests simpler.

---

## 5. Fusion maps in tuple space

Fusion maps are fixed tensors implementing tensor products of local irreps.

Given source orders `k1`, `k2`, target order `ku`, source irreps `lambda1`, `lambda2`, and target irrep `lambda`, a fusion map has type

```text
C : S^{lambda1}_{k1} tensor S^{lambda2}_{k2} -> S^{lambda}_{ku}.
```

Because the implementation is tuple-indexed, each fusion map also needs a **slot pattern**.

A slot pattern tells us how source tuple positions are selected from the target tuple positions.

Example target tuple order 3:

```text
target positions: (0, 1, 2)
```

A node-pair path may use

```text
source 1 positions: (0,)
source 2 positions: (1, 2)
```

which corresponds to `(i)` and `(j, k)` when the target tuple is `(i, j, k)`.

Another path may use

```text
source 1 positions: (1,)
source 2 positions: (0, 2)
```

which corresponds to `(j)` and `(i, k)`.

Do not encode these as unordered subsets. They are ordered position patterns.

---

## 6. Tuple multiplication map

For each fusion slot pattern, define a deterministic tuple multiplication map

```text
M_pattern : T_{k1} tensor T_{k2} -> T_{ku}
```

For a target ordered tuple `u`, the source ordered tuples are obtained by restricting `u` to the source slot patterns.

In formula form:

```text
[M_pattern(q1 tensor q2)]_u = q1_{u restricted to pattern1} * q2_{u restricted to pattern2}.
```

Then the fusion map is

```text
C = P^{lambda}_{ku}  M_pattern  (R^{lambda1}_{k1} tensor R^{lambda2}_{k2}).
```

This formula should be the central implementation in `reps/fusion.py`.

It covers:

- same-tuple Kronecker products,
- disjoint tuple/Littlewood-Richardson-type products,
- overlapping tuple products,
- contained tuple products.

The neural fuser should not recreate this map. It should load it.

---

## 7. Branching maps in tuple space

Branching maps bring higher-order tuple features down to lower-order tuple features.

Given source order `ks`, target order `kt`, source irrep `mu`, and target irrep `lambda`, a branch map has type

```text
B : S^{mu}_{ks} -> S^{lambda}_{kt}.
```

Again, because indexing is explicit ordered tuples, a branch map needs a slot pattern.

A branch slot pattern tells us which source positions survive into the target tuple.

Example source tuple `(i, j, k)` and target tuple `(i, j)`:

```text
source order: 3
target order: 2
target positions in source: (0, 1)
pooled position: 2
```

---

## 8. Tuple pooling map

For each branch slot pattern, define a deterministic tuple pooling map

```text
Pool_pattern : T_{ks} -> T_{kt}
```

For a target ordered tuple `v`, sum over all valid source tuples `u` such that restricting `u` to the target positions gives `v`.

In formula form:

```text
[Pool_pattern(q)]_v = sum_{u: restrict(u, target_positions) = v} q_u.
```

Repeated particle indices should be excluded unless a specific map says otherwise.

Then the branch map is

```text
B = P^{lambda}_{kt}  Pool_pattern  R^{mu}_{ks}.
```

This formula should be the central implementation in `reps/branching.py`.

---

## 9. Multiplicity/path handling

The space of intertwiners may have dimension larger than one.

Therefore cached maps should support path indices:

```text
FusionMap.tensor : [num_paths, d_target, d_source_flat]
BranchMap.tensor : [num_paths, d_target, d_source]
```

The path index labels independent equivariant maps. Neural modules learn weights over paths and channels.

If tuple-space construction produces a map with more than one independent path, the code should either:

1. expose all independent maps, or
2. expose a documented chosen basis of paths.

For MVP `M <= 3`, many paths are multiplicity-free, but the data structures should not assume this forever.

---

## 10. Nullspace validation method

In addition to tuple-space construction, implement or plan for a nullspace validation method.

Given source and target representation matrices for adjacent transpositions:

```text
rho_src(g)
rho_tgt(g)
```

an intertwiner `A` satisfies

```text
A rho_src(g) = rho_tgt(g) A
```

for all adjacent transpositions `g`.

Vectorized constraint:

```text
(rho_src(g)^T kron I_target - I_source kron rho_tgt(g)) vec(A) = 0
```

The nullspace gives all intertwiners. This is useful for:

- testing fusion maps,
- testing branch maps,
- future generalization beyond hardcoded small `M`.

The main runtime construction for MVP should still be tuple-space projection/pooling.

---

## 11. Required files and responsibilities

### `data_structures/partitions.py`

Should contain:

- integer partition generation,
- partition validation,
- transpose partition,
- size and formatting utilities.

### `reps/permutations.py`

Should contain:

- permutation generation,
- permutation sign,
- composition,
- inverse,
- adjacent transpositions,
- action on tuple positions,
- canonical local permutation ordering for tuple bases.

### `reps/specht.py`

Should contain:

- Specht metadata,
- dimensions,
- chosen basis conventions,
- small-order representation matrices if available.

### `reps/young.py`

Should contain:

- Young diagram/tableaux utilities,
- optional Young symmetrizer code,
- optional projector construction.

### `reps/character_tables.py`

Should contain:

- conjugacy class utilities,
- character table lookup for small symmetric groups,
- especially `S_1`, `S_2`, `S_3`, and later `S_4`.

### `reps/irreps.py`

Should contain:

- central irrep registry,
- irrep dimensions,
- representation matrices for generators,
- projection/reconstruction maps `P^lambda`, `R^lambda`,
- normalization metadata.

### `reps/fusion.py`

Should contain:

- `FusionSpec` dataclass,
- `FusionMap` dataclass,
- tuple multiplication map generator,
- fusion map generator using

```text
C = P M (R tensor R)
```

- cache-key construction for fusion maps.

### `reps/branching.py`

Should contain:

- `BranchSpec` dataclass,
- `BranchMap` dataclass,
- tuple pooling map generator,
- branch map generator using

```text
B = P Pool R
```

- branch-rule metadata for small `M`.

### `reps/fourier.py`

Should contain:

- optional transforms between tuple basis and Specht/Fourier basis,
- useful for debugging and alternative implementations.

### `reps/cached_maps.py`

Should contain:

- save/load utilities for generated maps,
- stable cache keys,
- versioning for basis and normalization conventions.

Cache keys must include at least:

```text
map kind: fusion or branching
orders
irreps
slot pattern
basis convention
normalization convention
path convention
```

### `reps/fixture_generators/sage_specht.py`

Offline helper for generating trusted fixtures with Sage.

### `reps/fixture_generators/passagemath_specht.py`

Offline helper for generating trusted fixtures with PassageMath.

Runtime code should not depend on Sage/PassageMath being installed.

---

## 12. Concrete `M=3` branching facts

Use shorthand:

```text
S = (2)
A = (1,1)
T = (3)
V = (2,1)
E = (1,1,1)
```

Triple-to-pair branching under trivial pooling over the removed particle:

```text
T -> S
V -> S plus A
E -> A
```

For Pfaffian readout:

```text
A is the terminal carrier.
V -> A and E -> A are carrier paths.
S, T -> S, and V -> S are gate/context paths.
```

---

## 13. Tests required for `reps/`

### Pair normalization tests

Check that

```text
P_S R_S = 1
P_A R_A = 1
P_S R_A = 0
P_A R_S = 0
R_S P_S + R_A P_A = identity on T_2
```

with the normalized `1/2` convention.

### Projection tests

For each `k, lambda`:

```text
P^lambda R^lambda = identity on the lambda block
```

and the sum over all isotypic projectors reconstructs tuple space, if using full regular decomposition.

### Fusion equivariance tests

For every generated `C`, verify:

```text
C rho_src(g) = rho_tgt(g) C
```

for adjacent transpositions `g`.

### Branching equivariance tests

For every generated `B`, verify:

```text
B rho_src(g) = rho_tgt(g) B
```

with the appropriate representation action on source and target tuple positions.

### Shape and slot-pattern tests

For every `FusionSpec` and `BranchSpec`, test that:

- slot positions are in range,
- tuple restrictions preserve order,
- repeated particle indices are excluded when required,
- output shapes match declared irrep dimensions.

---

## 14. Practical rule for agents

Do not write large hand-coded formulas for every product path.

Implement the general tuple-space formulas:

```text
C = P M (R tensor R)
B = P Pool R
```

Then cache the resulting tensors and let `nn/` consume them.

This is the key design separation.
