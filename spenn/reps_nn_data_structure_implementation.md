# Current SpechtMP workflow and explicit tensor-product formulas

This document describes the current SpechtMP workflow using explicit ordered-tuple indexing.

The layer workflow is:

\[
x \longrightarrow z \longrightarrow m \longrightarrow x.
\]

Interpretation:

1. \(x \to z\): fixed tensor-product/fusion using precomputed fusion maps \(C\).
2. \(z \to m\): trainable message aggregation plus irrep-aware nonlinearity.
3. \(m \to x\): fixed nontrainable branching using precomputed branching maps \(B\).

Important convention:

We are **not** using abstract subset indexing. Every index is an explicit ordered tuple of particle labels:

\[
i,\qquad ij,\qquad ijk,\qquad \dots
\]

Thus \(I\), \(I_1\), \(I_2\), \(J\) below should be read as ordered tuples, not abstract subsets. When we write

\[
I_1\cup I_2=I,
\]

this means the **label union** of the ordered tuples \(I_1\) and \(I_2\) equals the label set of the ordered tuple \(I\). The ordering of each tuple is still explicit and matters for projection/reconstruction.

---

# 1. General SpechtMP formulas

## 1.1. Feature notation

Let

\[
x_{I,a}^{c,\lambda}
\]

denote a feature where:

- \(I=(i_1,\dots,i_k)\) is an explicit ordered tuple of particles,
- \(\lambda\vdash k\) is a partition / local Specht irrep,
- \(c\) is a channel index,
- \(a=1,\dots,d_\lambda\) is the transforming irrep coordinate.

For \(M\le 3\), we use the shorthand:

\[
h_i=x_i^{(1)}
\]

\[
s_{ij}=x_{ij}^{(2)},\qquad a_{ij}=x_{ij}^{(1,1)}
\]

\[
t_{ijk}=x_{ijk}^{(3)},\qquad
v_{ijk}=x_{ijk}^{(2,1)},\qquad
e_{ijk}=x_{ijk}^{(1,1,1)}.
\]

The order-2 features satisfy:

\[
s_{ji}=s_{ij},
\qquad
a_{ji}=-a_{ij}.
\]

The order-3 antisymmetric feature satisfies:

\[
e_{\pi(i,j,k)}=\operatorname{sgn}(\pi)e_{ijk}.
\]

The mixed feature \(v_{ijk}\) is a \(2\times 2\) block in the chosen \(S_3\) standard-basis convention.

---

# 2. General fusion: \(x\to z\)

The fusion step constructs exact tensor-product features \(z\) from two input feature tensors.

Given ordered input tuples \(I_1\), \(I_2\), define the target tuple \(I\) by the label union:

\[
\operatorname{labels}(I)
=
\operatorname{labels}(I_1)\cup \operatorname{labels}(I_2).
\]

The target tuple ordering should follow the project’s canonical ordered-tuple convention.

The fixed fusion intertwiner is

\[
C_{I_1,I_2\to I,p}^{\lambda_1,\lambda_2\to\lambda}
:
S^{\lambda_1}_{I_1}\otimes S^{\lambda_2}_{I_2}
\to
S^\lambda_I.
\]

The path index \(p\) distinguishes multiple independent fusion paths if they exist.

The raw fused tensor-product feature is:

\[
\boxed{
z_{I_1,I_2\to I,p,a}^{c_1c_2;\lambda_1,\lambda_2\to\lambda}
=
\sum_{a_1,a_2}
C_{I_1,I_2\to I,p;\,a,a_1,a_2}^{\lambda_1,\lambda_2\to\lambda}
x_{I_1,a_1}^{c_1,\lambda_1}
x_{I_2,a_2}^{c_2,\lambda_2}.
}
\]

In code, this corresponds to `TPFeatureDict`.

The tensor-product/fusion step has **no learned parameters**. It only uses fixed precomputed \(C\)-maps.

---

# 3. Message head: \(z\to m\)

The message head consumes \(z\), applies trainable channel/path weights, applies a nonlinearity at the specified stage, and returns a `MessageDict`.

For fixed target tuple \(I\), target irrep \(\lambda\), source irrep pair \((\lambda_1,\lambda_2)\), path \(p\), and output channel \(c_{\mathrm{out}}\), 

\[
m^{c_out, \lambda}_I =
        \sum_{c_{in}} w_{c_{out}, c_{in}}^\lambda x^{c_{in}, \lambda}_{I} +
        \sum_{n, m \le |I|}
        \sum_{\lambda_1 \vdash n}
        \sum_{\lambda_2 \vdash m}
        \sum_p
        \Gamma\left[
            \sum_{I1 \subseteq I, I2 \subseteq I, I1 \cup I2 = I, |I_1| = n, |I_2| = m}
            \sum_{c_1,c_2}
            w_{p,c_{out},c_1,c_2}^{\lambda;\lambda_1,\lambda_2}
            z^{c_1c_2; \lambda_1, \lambda_2 \rightarrow \lambda}_{I_1, I_2 \rightarrow I, p}
        right]
\]
where \(\Gamma\) is nonlinear-activation

The linear term is trainable, but it is part of the message head, not the brancher.

Important: do **not** sum different target irreps \(\lambda\) into one tensor. Different \(\lambda\)'s live in different representation spaces. `MessageDict` should remain keyed by \(\lambda\).

---

# 4. Branching: \(m\to x\)

The brancher maps messages from virtual/equal order back to persistent features.

The fixed branching map is

\[
B_{J\to I,q}^{\lambda;\mu}
:
S_J^\mu
\to
S_I^\lambda,
\]

where:

- \(J\) is an explicit ordered source tuple,
- \(I\) is an explicit ordered target tuple,
- \(\operatorname{labels}(I)\subseteq\operatorname{labels}(J)\),
- \(\mu\vdash |J|\),
- \(\lambda\vdash |I|\),
- \(q\) is a path/multiplicity index if more than one branch exists.

The update is:

\[
\boxed{
x_{I,b}^{\mathrm{new};c,\lambda}
=
\sum_{\substack{J:\\
\operatorname{labels}(I)\subseteq\operatorname{labels}(J)}}
\sum_{\mu\vdash |J|}
\sum_q
\sum_a
B_{J\to I,q;\,b,a}^{\lambda;\mu}
m_{J,a}^{c,\mu}.
}
\]

In the current design, this step is **not trainable**.

Therefore:

- `BranchMap` should have no learned weights.
- Any channel mixing must happen in the message head before branching.
- If branch multiplicity \(q>1\), either:
  - use a fixed convention supplied by `reps`, or
  - keep separate branch paths as fixed outputs.
- Do not introduce learned \(q\)-weights inside the brancher.

The full residual SpechtMP layer may do:

\[
x^{t+1}=x^t+\operatorname{Branch}(m^t).
\]

The residual addition is allowed, but the branch itself remains fixed.

---

# 5. Irrep-aware nonlinearity

The nonlinearity \(\Gamma_\lambda\) must preserve equivariance.

Safe rules:

- For scalar/trivial irreps, ordinary nonlinearities are fine.
- For sign irreps, use odd nonlinearities if applied directly.
- For higher-dimensional irreps such as \((2,1)\), do not apply arbitrary coordinatewise activations to the transforming irrep coordinate.

Recommended safe form:

\[
x^\lambda \mapsto g(\text{invariant scalars})x^\lambda,
\]

where \(g\) is a learned scalar gate.

---

# 6. Hard-coded implementation for \(M\le 3\)

The implementation should not hand-code every tensor-product case. The safest implementation is:

1. Reconstruct source irrep features into ordered tuple values.
2. Multiply ordered tuple values.
3. Project ordered tuple values back to irreps.

This implements the same fusion map \(C\) without requiring the coder to know representation theory.

---

# 7. Order-1 reconstruction/projection

Order 1 has only the trivial irrep:

\[
q_i=h_i.
\]

Projection is identity:

\[
h_i=q_i.
\]

---

# 8. Order-2 reconstruction/projection

Order 2 has:

\[
S=(2),\qquad A=(1,1).
\]

Given ordered tuple values \(q_{ij}\), \(q_{ji}\), project using normalization factor \(1/2\):

\[
\boxed{
s_{ij}=\frac12(q_{ij}+q_{ji})
}
\]

\[
\boxed{
a_{ij}=\frac12(q_{ij}-q_{ji})
}
\]

Reconstruct by:

\[
\boxed{
q_{ij}=s_{ij}+a_{ij}
}
\]

\[
\boxed{
q_{ji}=s_{ij}-a_{ij}
}
\]

The normalization factor for order 2 is:

\[
\boxed{\frac12.}
\]

---

# 9. Explicit order-2 tensor products

## 9.1. Node-node to pair

Given \(h_i^{c_1}\) and \(h_j^{c_2}\):

\[
q_{ij}=h_i^{c_1}h_j^{c_2},
\qquad
q_{ji}=h_j^{c_1}h_i^{c_2}.
\]

Project:

\[
\boxed{
z_{ij}^{S;c_1,c_2}
=
\frac12
\left(
h_i^{c_1}h_j^{c_2}
+
h_j^{c_1}h_i^{c_2}
\right)
}
\]

\[
\boxed{
z_{ij}^{A;c_1,c_2}
=
\frac12
\left(
h_i^{c_1}h_j^{c_2}
-
h_j^{c_1}h_i^{c_2}
\right)
}
\]

If \(c_1=c_2\) and the features are scalar, then

\[
z_{ij}^{A;c,c}=0.
\]

---

## 9.2. Node-pair to pair

Reconstruct pair tuple values:

\[
q_{ij}=s_{ij}+a_{ij},
\qquad
q_{ji}=s_{ij}-a_{ij}.
\]

Multiply by node values:

\[
r_{ij}=h_i q_{ij}=h_i(s_{ij}+a_{ij}),
\]

\[
r_{ji}=h_j q_{ji}=h_j(s_{ij}-a_{ij}).
\]

Project:

\[
\boxed{
z_{ij}^{S}
=
\frac12(r_{ij}+r_{ji})
=
\frac12
\left[
(h_i+h_j)s_{ij}
+
(h_i-h_j)a_{ij}
\right]
}
\]

\[
\boxed{
z_{ij}^{A}
=
\frac12(r_{ij}-r_{ji})
=
\frac12
\left[
(h_i-h_j)s_{ij}
+
(h_i+h_j)a_{ij}
\right]
}
\]

---

## 9.3. Pair-pair to pair

Let the first pair feature be

\[
s_{ij}^{(1)},\qquad a_{ij}^{(1)},
\]

and the second pair feature be

\[
s_{ij}^{(2)},\qquad a_{ij}^{(2)}.
\]

Reconstruct:

\[
q_{ij}^{(1)}=s_{ij}^{(1)}+a_{ij}^{(1)},
\qquad
q_{ji}^{(1)}=s_{ij}^{(1)}-a_{ij}^{(1)},
\]

\[
q_{ij}^{(2)}=s_{ij}^{(2)}+a_{ij}^{(2)},
\qquad
q_{ji}^{(2)}=s_{ij}^{(2)}-a_{ij}^{(2)}.
\]

Multiply:

\[
r_{ij}=q_{ij}^{(1)}q_{ij}^{(2)},
\qquad
r_{ji}=q_{ji}^{(1)}q_{ji}^{(2)}.
\]

Project:

\[
\boxed{
z_{ij}^{S}
=
s_{ij}^{(1)}s_{ij}^{(2)}
+
a_{ij}^{(1)}a_{ij}^{(2)}
}
\]

\[
\boxed{
z_{ij}^{A}
=
s_{ij}^{(1)}a_{ij}^{(2)}
+
a_{ij}^{(1)}s_{ij}^{(2)}
}
\]

Parity rules:

\[
S\otimes S\to S,
\]

\[
S\otimes A\to A,
\]

\[
A\otimes S\to A,
\]

\[
A\otimes A\to S.
\]

---

# 10. Order-3 projection/reconstruction

Order 3 has:

\[
T=(3),\qquad V=(2,1),\qquad E=(1,1,1).
\]

Use the six ordered tuple entries:

\[
q_{ijk},\quad
q_{ikj},\quad
q_{jik},\quad
q_{jki},\quad
q_{kij},\quad
q_{kji}.
\]

The normalization factor for order 3 projections is:

\[
\boxed{\frac16.}
\]

---

## 10.1. Trivial projection \(T=(3)\)

\[
\boxed{
t_{ijk}
=
\frac16
\left(
q_{ijk}
+
q_{ikj}
+
q_{jik}
+
q_{jki}
+
q_{kij}
+
q_{kji}
\right)
}
\]

---

## 10.2. Antisymmetric projection \(E=(1,1,1)\)

Permutation signs:

\[
\operatorname{sgn}(ijk)=+1,
\]

\[
\operatorname{sgn}(ikj)=-1,
\]

\[
\operatorname{sgn}(jik)=-1,
\]

\[
\operatorname{sgn}(jki)=+1,
\]

\[
\operatorname{sgn}(kij)=+1,
\]

\[
\operatorname{sgn}(kji)=-1.
\]

Then:

\[
\boxed{
e_{ijk}
=
\frac16
\left(
q_{ijk}
-
q_{ikj}
-
q_{jik}
+
q_{jki}
+
q_{kij}
-
q_{kji}
\right)
}
\]

---

## 10.3. Mixed projection \(V=(2,1)\)

Use the \(S_3\) standard basis

\[
u_1=e_i-e_j,
\qquad
u_2=e_j-e_k.
\]

The representation matrices are:

\[
\rho(ijk)
=
\begin{pmatrix}
1 & 0\\
0 & 1
\end{pmatrix},
\]

\[
\rho(ikj)
=
\begin{pmatrix}
1 & 0\\
1 & -1
\end{pmatrix},
\]

\[
\rho(jik)
=
\begin{pmatrix}
-1 & 1\\
0 & 1
\end{pmatrix},
\]

\[
\rho(jki)
=
\begin{pmatrix}
0 & -1\\
1 & -1
\end{pmatrix},
\]

\[
\rho(kij)
=
\begin{pmatrix}
-1 & 1\\
-1 & 0
\end{pmatrix},
\]

\[
\rho(kji)
=
\begin{pmatrix}
0 & -1\\
-1 & 0
\end{pmatrix}.
\]

Define \(v_{ijk}\) as a \(2\times 2\) matrix:

\[
\boxed{
v_{ijk}
=
\frac16
\sum_{\pi\in S_3}
q_{\pi(i,j,k)}
\rho(\pi^{-1}).
}
\]

Explicitly:

\[
\boxed{
\begin{aligned}
v_{ijk}
=
\frac16\big[
&
q_{ijk}\rho(ijk)
+
q_{ikj}\rho(ikj)
+
q_{jik}\rho(jik)
\\
&
+
q_{jki}\rho(kij)
+
q_{kij}\rho(jki)
+
q_{kji}\rho(kji)
\big].
\end{aligned}
}
\]

The inverse relationships used above are:

\[
(ijk)^{-1}=ijk,
\]

\[
(ikj)^{-1}=ikj,
\]

\[
(jik)^{-1}=jik,
\]

\[
(jki)^{-1}=kij,
\]

\[
(kij)^{-1}=jki,
\]

\[
(kji)^{-1}=kji.
\]

---

## 10.4. Triple reconstruction

Given

\[
t_{ijk},\qquad v_{ijk},\qquad e_{ijk},
\]

reconstruct each ordered tuple value by:

\[
\boxed{
q_{\pi(i,j,k)}
=
t_{ijk}
+
\operatorname{sgn}(\pi)e_{ijk}
+
2\,\operatorname{tr}
\left(
v_{ijk}\rho(\pi)
\right).
}
\]

In code:

```python
mixed = 2.0 * torch.einsum("...ab,ba->...", v, rho_pi)
q_pi = t + sign_pi * e + mixed
```

Explicitly:

\[
q_{ijk}
=
t+e+2\operatorname{tr}(v\rho(ijk)),
\]

\[
q_{ikj}
=
t-e+2\operatorname{tr}(v\rho(ikj)),
\]

\[
q_{jik}
=
t-e+2\operatorname{tr}(v\rho(jik)),
\]

\[
q_{jki}
=
t+e+2\operatorname{tr}(v\rho(jki)),
\]

\[
q_{kij}
=
t+e+2\operatorname{tr}(v\rho(kij)),
\]

\[
q_{kji}
=
t-e+2\operatorname{tr}(v\rho(kji)).
\]

---

# 11. Generic tuple-product algorithm for \(M\le 3\)

This algorithm is the preferred implementation route.

## 11.1. Reconstruct source tuple values

For order 1:

\[
q_i=h_i.
\]

For order 2:

\[
q_{ij}=s_{ij}+a_{ij},
\]

\[
q_{ji}=s_{ij}-a_{ij}.
\]

For order 3:

\[
q_{\pi(i,j,k)}
=
t_{ijk}
+
\operatorname{sgn}(\pi)e_{ijk}
+
2\operatorname{tr}(v_{ijk}\rho(\pi)).
\]

---

## 11.2. Multiply tuple values

Let the target ordered tuple be

\[
I=(i_1,\dots,i_k).
\]

Let the two source ordered tuples be \(I_1\), \(I_2\).

For each ordering of the target labels, restrict the target ordering to the labels in \(I_1\) and \(I_2\), preserving target order.

Example:

If

\[
I=(i,j,k),
\]

and

\[
I_1=(i,k),
\qquad
I_2=(j,k),
\]

then for target ordering

\[
(j,k,i),
\]

the restricted orderings are:

\[
(j,k,i)|_{I_1}=(k,i),
\]

\[
(j,k,i)|_{I_2}=(j,k).
\]

Then:

\[
\boxed{
r_{jki}
=
q^{(1)}_{ki}
q^{(2)}_{jk}.
}
\]

General rule:

\[
\boxed{
r_I
=
q^{(1)}_{I|_{I_1}}
q^{(2)}_{I|_{I_2}}.
}
\]

Python-style pseudocode:

```python
def tuple_product(q1, q2, target_tuple, source1_labels, source2_labels):
    restricted_1 = restrict_ordered_tuple(target_tuple, source1_labels)
    restricted_2 = restrict_ordered_tuple(target_tuple, source2_labels)
    return q1[restricted_1] * q2[restricted_2]
```

---

## 11.3. Project target tuple values

After building the raw target tuple values \(r\), project to the desired target irrep.

For order 1:

\[
h_i=r_i.
\]

For order 2:

\[
s_{ij}=\frac12(r_{ij}+r_{ji}),
\]

\[
a_{ij}=\frac12(r_{ij}-r_{ji}).
\]

For order 3:

\[
t_{ijk}
=
\frac16
\sum_{\pi\in S_3}
r_{\pi(i,j,k)},
\]

\[
e_{ijk}
=
\frac16
\sum_{\pi\in S_3}
\operatorname{sgn}(\pi)
r_{\pi(i,j,k)},
\]

\[
v_{ijk}
=
\frac16
\sum_{\pi\in S_3}
r_{\pi(i,j,k)}
\rho(\pi^{-1}).
\]

These projected values are the tensor-product output \(z\).

---

# 12. Explicit sanity-check examples

The tuple algorithm above is the source of truth. The formulas below are only sanity checks.

## 12.1. Node + symmetric pair to triple

Define:

\[
y_i=h_i s_{jk},
\]

\[
y_j=h_j s_{ik},
\]

\[
y_k=h_k s_{ij}.
\]

Then:

\[
z_{ijk}^{T}
=
\frac13(y_i+y_j+y_k).
\]

There is no \(E\) component.

The \(V\) component should be obtained from the normalized order-3 projection formula, not hand-coded unless tested.

---

## 12.2. Node + antisymmetric pair to triple

Define:

\[
z_i=h_i a_{jk},
\]

\[
z_j=h_j a_{ki},
\]

\[
z_k=h_k a_{ij}.
\]

Then:

\[
z_{ijk}^{E}
=
\frac13(z_i+z_j+z_k).
\]

There is no \(T\) component.

The \(V\) component should be obtained from the normalized order-3 projection formula.

---

## 12.3. Pair-pair to triple

Do not hand-code sign rules for pair-pair to triple paths.

Use:

1. reconstruct pair tuple values,
2. restrict target ordering to each source pair,
3. multiply,
4. project using order-3 projections.

This handles:

\[
S\times S,\qquad
S\times A,\qquad
A\times S,\qquad
A\times A.
\]

---

# 13. Current file responsibilities

## `data/irrep_features.py`

Should contain:

- `FeatureDict`: current layer features \(x\).
- `TPFeatureDict`: exact tensor-product features \(z\).
- `MessageDict`: aggregated messages \(m\).

Important correction:

`I`, `I_1`, and `I_2` are explicit ordered tuple indices, not multiplicity indices and not abstract subset indices.

`TPFeatureDict` stores:

\[
z_{I_1,I_2\to I,p}^{c_1c_2;\lambda_1,\lambda_2\to\lambda}.
\]

`MessageDict` stores:

\[
m_I^{c,\lambda}.
\]

---

## `reps/fusion.py`

Should contain `FusionMap`.

Responsibilities:

- fixed tensor-product/fusion \(x\to z\),
- no learned weights,
- for Phase 1, hard-code \(M\le 2\),
- later support \(M\le 3\) using tuple reconstruction/product/projection,
- eventually load precomputed maps from `spenn.reps`.

---

## `nn/spechtmp/message_head.py`

Should contain `MessageHead`.

Responsibilities:

- call `reps.FusionMap` to compute \(z\),
- aggregate \(z\) into \(m\) with learned weights,
- apply \(\Gamma_\lambda\) after summing over \(c_1,c_2,I_1,I_2\),
- sum over \(\lambda_1,\lambda_2,p\),
- return `MessageDict`.

This file is the trainable part of SpechtMP.

---

## `reps/branch.py`

Should contain `BranchMap`.

Responsibilities:

- fixed branching \(m\to x\),
- no learned weights,
- for Phase 1, hard-code \(M\le 2\),
- eventually load precomputed branch maps from `spenn.reps`.

The docstring should say “branching”, not “tensor-product”.

---

## `nn/spechtmp/layer.py`

Should contain `SpechtMPLayer`.

Responsibilities:

- compose message head and branching map,
- implement:

\[
x\to z\to m\to x.
\]

Schematic:

```python
class SpechtMPLayer(nn.Module):
    def forward(self, x):
        m = self.message_head(x)      # x -> z -> m
        dx = self.branch_map(m)    # m -> x, fixed/nontrainable
        return x + dx                 # optional residual
```

`nn/spechtmp/brancher.py` and `nn/spechtmp/fuser.py` have been removed in the current design.

---

# 14. Required tests

## Projection/reconstruction tests

Order 2:

\[
q_{ij}\to(s,a)\to \hat q_{ij}
\]

must recover \(q_{ij}\), \(q_{ji}\).

Order 3:

\[
q_6\to(t,v,e)\to \hat q_6
\]

must recover all six ordered tuple values.

## Symmetry tests

\[
s_{ji}=s_{ij}
\]

\[
a_{ji}=-a_{ij}
\]

\[
e_{\pi(i,j,k)}=\operatorname{sgn}(\pi)e_{ijk}
\]

## Fusion equivariance test

For a random permutation \(\sigma\), check:

\[
\operatorname{FusionMap}(\sigma x)
=
\sigma\,\operatorname{FusionMap}(x).
\]

## Branching equivariance test

For a random permutation \(\sigma\), check:

\[
\operatorname{BranchMap}(\sigma m)
=
\sigma\,\operatorname{BranchMap}(m).
\]

## Normalization tests

Order 2 projections must use:

\[
\frac12.
\]

Order 3 projections must use:

\[
\frac16.
\]

Do not silently change normalization conventions.
