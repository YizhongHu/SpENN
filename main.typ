//#import "@preview/noteworthy:0.2.0": * // Need to make TOC breakable
#import "lib.typ": *
#import "@preview/showybox:2.0.4": showybox
#import "@preview/equate:0.3.2": equate, share-align
#import "@preview/physica:0.9.6": braket, pdv, grad, curl, dd, Order
#import "@preview/mannot:0.3.0": markrect
// Theoretic imported with noteworthy
#show link: underline

#show: noteworthy.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: "SpENN Design Document",
  author: "Richard Hu",
  contact-details: "SpENN",
  toc-title: none,
  watermark: "DRAFT", // Optional: Watermark for the document
)

#show ref: theoretic.show-ref

#set cite(style: "chicago-notes")
#show ref: footnote

#set heading(numbering: "1.")

#show heading.where(level:1): it => {
  counter(math.equation).update(0)
  it
}
#show: equate.with(breakable: true)

#set math.equation(numbering: it => {
  // Get the chapter number (first element of the heading counter)
  let chapter_num = counter(heading).get().first() 
  // Format the number as (chapter.equation_number)
  numbering("(1.1)", chapter_num, it) 
})
// #set math.mat(delim: "[")
#set enum(numbering: "I.1.a.i)")


// Write here

#let proposition = theorem.with(kind: "proposition", supplement: "Proposition", fmt-suffix: none)
#let question = proof.with(kind: "question", supplement: "Question", number: none, fmt-suffix: none)
#let answer = proof.with(kind: "answer", supplement: "Answer", number: none, fmt-suffix: none)
#let test = proof.with(kind: "test", supplement: "Test", number: none, fmt-suffix: none)
#let results = proof.with(kind: "results", supplement: "Results", number: none, fmt-suffix: none)

#let note = note.with(number: none)

// Physics symbols
#let varphi = $phi$ // Old habits die hard
#let phi = $phi.alt$
#let ang = $angstrom$
#let eV = $e V$
#let Hhat = $hat(H)$
#let That = $hat(T)$
#let Vhat = $hat(V)$
#let Vhee = $Vhat_(e e)$
#let ip(bra, ket) = {
  $chevron.l bra mid(|) ket chevron.r$
}
#let qf(bra, op, ket) = {
  $chevron.l bra mid(|) op mid(|) ket chevron.r$
}
#let EXC = $E_(X C)$

// Math Symbols

#let neq = $eq.not$
#let sgn = "sgn"
#let cA = $cal(A)$
#let mapsto = $arrow.r.bar$
#let mixprod = math.op(sym.triangle.stroked.small.t)

// Vector symbols

#let bh = $bold("h")$
#let bk = $bold("k")$
#let bl = $bold("l")$
#let bm = $bold("m")$
#let bq = $bold("q")$
#let br = $bold("r")$
#let bR = $bold("R")$
#let bu = $bold("u")$
#let bv = $bold("v")$
#let bw = $bold("w")$
#let bx = $bold("x")$
#let by = $bold("y")$
#let bz = $bold("z")$
#let bEta = $bold(eta)$
#let ij = $i j$
#let jk = $j k$
#let ki = $k i$
#let ijk = $i j k$

// Words
#let GSD = "generalized slater determinant"
#let ansatze = "ansätze"
#let Schrodinger = "Schrödinger"

#pagebreak()

= Introduction

Specht-module Equivariant Neural Network, or SpENN /ʃpɛn/, is a general neural network
structure based on equivariant features of the symmetric groups $S_n$. The design
philosophies of this architecture can extend to other forms of permutation in/equi-variance,
but we will focus on using it as an antisymmetric Quantum Monte Carlo ansatz.

= Background

== Definitions

=== Tuples
For a positive integer $n$, define $[n] = (1, dots, n)$, the tuple of elements from $1$ to $n$.
Depending on the situation, this can also be interpreted as a set ${1, dots, n}$.

Permutations are injective (and thus bijective) maps $sigma: [n] mapsto [n]$. We denote 
injective maps with a hooked arrow: $sigma: [n] arrow.hook [m]$. The set of permutations
of order $n$ is denoted $S_n$. This forms a group under composition.

For a generic tuple of positive integers $I = (i_1, i_2, dots, i_ell: i_k in [n])$, the order
or cardinality of the tuple $|I| = n$. 

=== Tuple Maps
The tuple $I = (i_1, i_2, dots, i_ell: i_k in [n])$ contains the same information as a map
$
tau_I: [ell] & mapsto [n]\
        k & mapsto i_k
$
Thus a tuple can also be used inter-changeably with a map. 
For a tuple $I:[ell] mapsto [n]$ and $J:[m] mapsto [ell]$, 
we can define $I_J = (i_j_k: k in [m])$ as the composition of the two tuples.

For a map $tau: [m]mapsto[n]$, its image is defined as
$ "im"(tau):= {tau(k) :k in [m]} $
For a set $S subset.eq [n]$, it can be filtered and removed from a tuple:
$ I\\S := {i: i in.not S} $

== Quantum Monte Carlo and Antisymmetric functions

Quantum Monte Carlo attempts to obtain the ground state of the #Schrodinger
equation:
$ Hhat psi = E psi $
where $psi: (RR^3 times {plus.minus 1})^n mapsto CC$ is the wave function
of the electrons, and
$ Hhat = sum_i -1/2 nabla^2_i + sum_(A, i) - (Z_A)/(||br_i - bold("R")_A||)
+ sum_(i < j) 1/(||br_i - br_j||) $
Importantly, since electrons are fermions, the wave function is antisymmetric:
$ psi(br_(sigma^(-1) (1)), br_(sigma^(-1) (2)), dots, br_(sigma^(-1) (n)))
 = sgn(sigma)psi(br_1, br_2, dots, br_n) $
for all $sigma in S_n$, a permutation of $n$ elements, and $sgn(sigma)$
the sign (or parity) of the permutation.

To calculate the ground state, we minimize the following functional
$ E_0 = "min"_(psi) integral psi^* (br) Hhat psi (br) d br $
To approximate the integral, we convert it into an expectation value:
$ integral psi^* (br) H  psi (br) d br = EE_(br ~ |psi(br)|^2) E_"loc" (br) $
where $E_"loc"$ the *local energy* is defined as
$ E_"loc" (br) = (H psi(br))/(psi(br)) $
Hence we can use a MCMC sampler on $|psi(br)|^2$ to estimate $E$
$ E = 1/N sum_(a=1)^N E_"loc" (br^a) $
Using SpENN as an ansatz for $psi$, we estimate the energy $E$ and train the parameters
of the neural network to minimize it.

== Representations of Symmetric Groups

=== Partitions and Specht modules

To guarantee that $psi$ is antisymmetric, we use an equivariant neural network that operates
on the irreps of $S_n$ instead of allowing arbitrary interactions. The irreps of $S_n$
are related to partitions of $n$.

#definition(title: [Partition])[
  A partition $lambda$ of $n$, denoted as $lambda tack n$, is a weakly decreasing
  tuple of positive integers
  $ lambda = (lambda^1, lambda^2, dots, lambda^ell) $
  where $n = sum_(k=1)^ell lambda^k$ and $lambda^1 gt.eq lambda^2 gt.eq dots gt.eq lambda^ell$.
]

For example, the partitions of $1$ is only $(1)$; the partitions of 
$2$ are $(2)$ and $(1, 1)$; the partitions of $3$ are $(3), (2, 1), "and", (1, 1, 1)$, etc.

#definition(title: [Irreducible Representations of $S_n$])[
  For $lambda tack n$, the associated irreducible representation (or irrep) is
  $rho^lambda: S_n mapsto CC^(d_lambda)$, where $d_lambda$ is known as the dimension
  of the irreducible representation.
]

The space that the irreps of $S_n$ acts on is called *Specht modules*
/ʃpɛçt  ˈmɑː.dʒuːlz/, which inspires the name and pronunciation of SpENN.
Subspaces corresponding to the irrep $lambda$ is notated as $S^lambda$.
Vectors in $S^lambda$ are equivariant with $rho^lambda$.
They should be referred to as *irrep vectors*
or *irrep features*, but sometimes they are called irreps as well, which causes
confusion. We will try to avoid this confusion.

=== Young's tableau and construction of irreps

A combinatorics object called *Young's tableau* is used to construct irreps of 
$S_n$. 

The irrep of $S_1$ is
$ rho^((1)) (sigma) = 1 $
The irreps of $S_2$ are
$ rho^((2)) (sigma) & = 1 quad rho^((1, 1)) (sigma) & = sgn(sigma) $
The irreps of $S_3$ are
$ rho^((3)) (sigma) & = 1 quad rho^((1, 1, 1)) (sigma) & = sgn(sigma) $
The $(2,1)$ irrep is two dimensional. It can differ based on the basis we choose.
For now, we use the $e_1-e_2$, $e_2 - e_3$ basis:
$
rho^((2,1)) (e) = mat(1, 0; 0, 1) quad 
rho^((2,1)) ((1, 2, 3)) = mat(0, -1; 1, -1) quad 
rho^((2,1)) ((1, 3, 2)) = mat(-1, 1; -1, 0) quad \
rho^((2, 1))((1, 2)) = mat(-1, 1; 0, 1) quad
rho^((2, 1))((2, 3)) = mat(1, 0; 1, -1) quad
rho^((2, 1))((1, 3)) = mat(0, 1; -1, 0)
$

=== Subgroup representations of $S_n$

The group function $sigma mapsto phi(sigma(br))$ can be considered a vector
indexed by $S_n$. Under permutation, this transforms as the regular 
representation of $S_n$, we notate as $rho^"reg"$. In irrep basis, this
vector has $n!$ components. Any operation on it will cost
at least $O(n^n)$ time. This is inefficient, so we investigate
irreps of subgroups of $S_n$ instead.

An assumption can be made about electrons, that their correlation is dominated
by *lower body order interactions*: pair interactions, triple interactions, etc.
So we only need to operate on lower body-order. Through message passing, 
irreps with lower order should be able to obtain information about higher body 
order interactions, but our system descriptors will have capped body order.

Let's try and make this concrete. Say that the system only needs information
about 2-body interactions to completely describe the wave function. This means
that for any pair of particles $br_i$ and $br_j$, only the irreps that describe
their order matters. It should transform trivially for the rest of the particles
#footnote[This is not the only way to do this but rather a design choice, see
#ref(<as-irreps>)].

In other words, we operate on $S_2 times S_(n-2) subset S_n$, and use the induced 
representation $"Ind"^(S_n)_(S_2 times S_(n-2)) (rho^lambda times.square bold(1))$.
In practice, this means that for every pair of particles $i$ and $j$, we can 
associate with it irrep vectors in $bx^lambda_ij in S^lambda$ where $lambda tack 2$. They
transform as the irrep $rho^lambda$, and contain information about the subgroup 
$S_2 times S_(n-2)$ where the changes in the rest of the particles do not affect
this irrep vector.

= Design

== Inputs and Embedding

The inputs are all fed into an embedding layer that gives a tensor $bx_I$, representing the
$m$-th body order interaction.

$ bx_I^(m) = phi^m (br_i_1, dots.c, br_i_m) $

Here $I$ is a tuple of non-repeating indices $(i_1, i_2, dots, i_m : i_k in [n])$
#footnote[We use $[n]$ to represent the set ${1, dots, n}$].

Since the body order $m = |I|$, we will suppress $m$ unless otherwise stated, since it
can be directly derived from $I$.

This embedding scheme means that we are not restricted to sending $RR^3$ 
coordinates into the encoder. In fact, this can be an arbitrary vector that describes
as much about the particle as possible. We gain a more extensible model for free. 

In the QMC case, we can easily encode spin in this way:
$ bv_i = (br_i, s_i) = (x_i, y_i, z_i, s_i) $
where $s_i in {1, -1}$.

To generalize even more, one can even build a full QMC model with atomic coordinates this way:
$ bv_i = (br_i, s_i, ["one-hot encoding of particle type"]) $
which variables are able to freely move is determined by the MC walker. The model can be
agnostic about what type of particle it is dealing with.

To preserve more information from the input, we can stack multiple channels:
$ bx^(0, c, m)_I = phi^(c m) (br_i_1, dots.c, br_i_m) $
where $c$ is the channel index.

== Specht Layers <spechtmp>
#block[
There are two steps in a Specht layer: mixing and activation.
  #set math.equation(numbering: none)
  $ 
  bx^m_I stretch(->)^"mixing"_(W) bh^m_I
  markrect(
    stretch(->)^"projection"_P hat(bh)^lambda_I
    stretch(->)^"activation"_Gamma hat(bu)^lambda_I
    stretch(->)^("projection"^(-1))_(P^(-1))
  ) bu^m_I stretch(->)^("update") bx_I^m 
  $
  The boxed parts indicate activation in irrep space. We call this Specht activation.
  It will be introduced later.
]

=== Equivariant Mixing

The mixing process takes features $x^(t c)_I$ does an equivariant tensor product
and outputs hidden states $bh^((t+1)c)_(I, p)$. This information may be passed on to
features of equal or lower body-order than the interaction

Mixing has the general form:
$ bh^c_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(c<-c_1c_2) bx_(J circle.small tau_1)^c_1 bx_(J circle.small tau_2)^c_2 $
where the path index $p = (s, m, m_1, m_2, tau, tau_1, tau_2)$ describes the different equivariant paths indices 
through which $bx$ can interact and produce features on index $I$.
- $s$: interaction order--total number of particles involved in this interaction.
- $m$: output order--the order of the output feature, the feature that accepts the information from the interaction.
  $m lt.eq s$
- $m_1, m_2$: input orders--the orders of the input features that interact with each other and form a virtual
  feature of order $s$. $m_1 + m_2 gt.eq s$.
and activation has the general form
- $tau$: $[m] arrow.hook [s]$,  output injection--an injective function the describes where the indices of $I$
  are positioned in the virtual interaction.
- $tau_1$: $[m_1] arrow.hook[s]$, left input injection--an injective function that describes where the indices of $I_1$
  are positioned in the virtual interaction.
- $tau_2$: $[m_2] arrow.hook[s]$, right input injection--an injective function that describes where the indices of $I_2$
  are positioned in the virtual interaction.
$sum_(J_([s]\\"im"(tau)))$ means summing over all the indices
in the virtual interactions that is not already in the image of $tau$, i.e. the indices in $I$ that are corresponding to.

== Specht Activation
#block[
  After mixing, we need to add non-linear activation functions to the features, 
  but a simple $Gamma(bx)$ will break the permutation equivariance that we carefully crafted. 
  To keep the messages equivariant, we need to do activation in irrep vector space. This is
  where Specht modules come into play.
]

/*
=== Index Taxonomy

During the process, we have to deal with a lot of indices, 
some of them we decided to sum over, some of them we decided to apply weights
to, and some of them are activated. The decision over it is a balancing act.
We shall discuss this in detail.

#table(
  columns: (auto, 1.3fr, 1.5fr, 1.4fr),
  inset: 6pt,
  align: (horizon, horizon, left, left),
  table.header[
    *Class*
  ][
    *Examples*
  ][
    *Operation*
  ][
    *Learnable?*
  ],

  [Output indices],
  [
    $I, lambda, c_"out", alpha, beta$
  ],
  [
    Survive the operation and label the output block $bm_I^(c_"out", lambda)$ or $bx_I^(c_"out", lambda)$.
  ],
  [
    Sometimes. Everything except $alpha$ can mix.
  ],

  
  [Path / mechanism indices],
  [
    $lambda_1$, $lambda_2$, $mu$, $I_1$, $I_2$, $J$
  ],
  [
    Label distinct equivariant mechanisms.
    Summed between activation and output.
  ],
  [
    Yes, and they mix trivially without activation.
  ],

  [Activation-domain indices],
  [
    $c_1$, $c_2$, $c_"in"$
  ],
  [
    Aggregated after applying weights but before activation
  ],
  [
    Yes, and they mix non-trivially even without activation.
  ],

  [Fixed contraction indices],
  [ 
    $sigma$
  ],
  [
    Not influenced by weights and are thus summed before weights apply.
  ],
  [
    No. These indices do not mix.
  ],
)

More detailedly, fixed contraction indices do not mix
and do not appear in the output, so they are summed first to reduce dimensions.
Activation-domain indices do mix, and they can mix non-trivially 
even when they are only passed though a linear layer. For maximum efficiency,
they are summed before activation.
Path/mechanism indices are indices that would mix trivially if no activation is 
applied, i.e. having multiple weights is equivalent to having a single weight, 
so they have to be passed through non-linear activation to be meaningful.
Of course, we can still choose to do linear activation on all 
learned indices, or non-linear activation on all the non-trivially-mixed
indices, just note that in the first case, there will be redundant weights,
and in the second case, the dimensionality may explode.

My choices on which learned indices to activate is essentially my
attempt at a reasonable accuracy-dimensionality tradeoff. Among many things we can
tweak about the model, this is also one of them.
*/


=== Fourier Transform and its Inverse

To get the irreps from inputs, we can use the Peter-Weyl theorem. Irrep tensors $bx$ 
can be calculated with the following Fourier transform:
$ hat(bx)^lambda_I = (P^lambda bq)_I = 1/m! sum_(sigma in S_n) bx_(sigma I)^m rho^lambda (sigma^(-1)) $
where $sigma$ can be considered as the coordinates in real space, and
$lambda$ can be considered the coordinates in irrep/frequency/reciprocal space.
As long as we have the representation matrices, we will be able to read the irreps off.

Inverse projection maps the irrep vectors $hat(bx)$ back into feature space $bx$.
$ (P^(-1) hat(bx))_I = sum_(lambda tack|I|) d_lambda tr(hat(bx)^lambda_I) $
It satisfies
$ sum_(lambda tack m) P^(-1)  P^lambda = I_(m!) $
/*
=== Activation

We need to project into irrep spaces with the Fourier transform before activation
to preserve equivariance:
$ sum_lambda P^(-1) Gamma_lambda P^lambda $
where $Gamma_(lambda)$ is the activation function on the irrep $lambda$.

Activation should not mix components of the same irrep. This includes every 
permutation of the irrep: ${bx^lambda_(I; alpha) : sigma in S_(|I|)}$
Note that since $beta$ is the multiplicity index, we can activate irreps with
different $beta$ values separately. This limitation does not apply to
irreps with different sets of indices: $bx^lambda_I $ and $bx^lambda_(J)$ 
can be activated separately as long as $I neq J$. It also does not apply to
different irreps: $bx^lambda_I$ can be activated separately to 
$bx^mu_I$ as long as $mu neq lambda$.

$(1)$ can have any smooth activation.

Antisymmetric scalar irreps $(1^m)$ can have smooth *odd* activation functions.

Tensor irreps such as $(2, 1)$ can have *normed* activation:
$ tilde(bv) = sigma(||bv||)/(||bv||) bv $
where $sigma$ is an arbitrary function.

The $(1)$ irrep can also act as a gate:
$ tilde(bv) = sigma(bx^((1))_i) bv $
where $sigma$ is an arbitrary smooth function.
For unitary representations, norms of every irrep transform as $(1)$:
$ tilde(bv) = sigma(||bv'||) bv $*/

=== Activation

Nonlinear activations should be applied in local irrep space. For a real ordered-tuple
feature block, the activation has the form

$
sum_(lambda tack m) P^(-1) Gamma_lambda P^lambda,
$

where $P^lambda$ projects the ordered tuple orbit into the local Specht irrep
$lambda$, and $Gamma_lambda$ is an equivariant activation on that irrep block.

For each support orbit $"Ord"(I)$, all orderings of $I$ must be gathered and
activated together. Equivalently, the activation is applied to

$
(bx_(sigma I) : sigma in S_m),
$

after projection into irreps. It is not correct to activate each ordered tuple
coordinate $bx_(sigma I)$ independently when using representation-aware
activation.

After projection, an irrep block should be viewed schematically as

$
hat(bx)_(alpha r)^lambda,
$

where $alpha$ is the transforming irrep coordinate and $r$ bundles all
multiplicity-like axes. In this architecture, $r$ may include channel indices,
path indices such as $tau$ or $(tau_1, tau_2)$, and the regular-representation
multiplicity index $beta$.

The activation must not apply arbitrary elementwise nonlinearities to the
transforming coordinate $alpha$ for non-scalar irreps. However, it may mix
multiplicity axes:

$
hat(bx)_(alpha r_"out")^lambda
=
sum_(r_"in")
U_(r_"out" <- r_"in")^lambda
hat(bx)_(alpha r_"in")^lambda.
$

This mixing is equivariant because it does not act on the transforming irrep
coordinate $alpha$. The weights over multiplicity axes do not need to be
normalized for equivariance. Normalization may still be useful as an optimization
choice, but it is not a symmetry requirement.

Different orbits may be activated separately. Thus $hat(bx)_I^lambda$
and $hat(bx)_J^lambda$ can be activated separately when $"im"(I) != "im"(J)$.
Different irreps may also be activated separately: $hat(bx)_I^lambda$ and
$hat(bx)_I^mu$ should usually have separate activation functions when
$lambda != mu$.

The general form of Specht activation is:
$ hat(bu)^(c lambda)_(I; alpha, b_"out") = sum_p sum_beta_"in" U^(c lambda)_("im"(I), p; beta_"out"<-beta_"in") 
Gamma_lambda (hat(bh)^(c lambda)_("Ord"(I), p; alpha beta_"in")) $
close attention needs to paid so that irrep dimensions do not mix. There should not 
be extra weights and activations for
+ different $alpha$ values,
+ and different permutations of $I$.

=== Scalar irreps

The trivial scalar irrep $(m)$ may use any smooth scalar activation:

$
tilde(x) = phi(x).
$

The antisymmetric scalar irrep $(1^m)$ transforms by sign. Therefore its
activation must be odd:

$
phi(-x) = - phi(x).
$

Examples include

$
phi(x) = tanh(x)
$

or more generally

$
phi(x) = x g(x^2).
$

=== Tensor irreps

For higher-dimensional irreps, use norm or gated activations. If the irrep basis
is orthonormal or unitary, the invariant norm of

$
hat(bx)_(alpha r)^lambda
$

over the transforming coordinate is

$
x_r
=
(norm(hat(bx)_(alpha r)^lambda)^2 + epsilon)^(1/2)
=
(
  sum_alpha abs(hat(bx)_(alpha r)^lambda)^2
  + epsilon
)^(1/2).
$

A simple gate-only activation is

$
tilde(bx)_(alpha r)^lambda
=
g_lambda (x_r)
hat(bx)_(alpha r)^lambda.
$

A normalized-direction activation is

$
tilde(bx)_(alpha r)^lambda
=
(a_lambda (x_r))/(x_r)
hat(bx)_(alpha r)^lambda.
$

In this second form, the irrep vector is normalized on every forward pass. This
is only necessary if the activation is intended to separate direction from
magnitude. If the activation only gates the vector by an invariant scalar, then
explicit normalization is not required.

If the representation basis is not orthonormal, the Euclidean norm above is not
invariant. In that case, one must either use an orthonormal/seminormal Young
basis or replace the Euclidean norm by the correct Gram-matrix norm.

=== Invariant gates from other irreps

Invariant scalar quantities may be used to gate any irrep. For example, the
trivial irrep can gate another irrep:

$
tilde(bv)
=
sigma(hat(bx)^((m))) bv.
$

Likewise, the norm of any irrep transforms as a scalar invariant, so it can be
used as a gate:

$
tilde(bv)
=
sigma(norm(bv')) bv.
$

These gates are equivariant because the gate value is invariant and the output
keeps the same transforming irrep direction as $bv$.




=== Updates

For an update $bu^(t+1)$, we can directly apply the update:
$ bx^(t+1) = bold("u")^(t+1) $
For $u^(t+1) ~ bx$, we can use a residual update:
$ bx^(t+1) = bx^t + bold("u")^(t+1) $

For $hat(bu)^(t+1, (1)) ~ 1$, we may also consider a gated update:
$ bx^(t+1) = hat(bu)^(t+1, (1)) bx^t $
or even gated *and* residual update:
$ bx^(t+1) = hat(bu)^(t+1, (1)) bx^t + bold("u")^(t+1) $
In unitary representations, we can also use the norm of some irrep tensor like $(2,1)$.

We can also consider soft update:
$ bx^(t+1) = (1-a)bx^t + a bu^(t+1) $
where $a in (0, 1]$ is a hyper-parameter.

== Antisymmetric Readout

=== Readout vs. Encoder Anti-symmetrization tradeoff <as-irreps>

The irreps features that are passed between the layers are in fact representations
of a subgroup of $S_n$. Say that we have irrep feature with order $m$. It is working
with the $S_m times S_(n-m)$ subgroup of $S_n$, specifically, an element in $S_m$ describes
an *orbit* in $S_n$, subgroup of elements such that the indices in some tuple $I$ stays invariant. 

The irreps that we are using is
$ "Ind"^(S_n)_(S_m times S_(n-m)) (S^lambda times.square bold("1")) $
where we assume that all the variation happens within $S_m$ and the irrep is *permutation-invariant*
to $S_(n-m)$. The advantage of this is that it is very easy to encode the irreps at the start, 
but we need to anti-symmetrize at the end. 

The alternative is to instead work with antisymmetrized irreps
$ "Ind"^(S_n)_(S_m times S_(n-m)) (S^lambda times.square S^((1^(n-m)))) $
This assumes that the irreps are *anti-symmetric* to the indices not recorded, but encoding
these irreps faithfully is challenging. The correct way of doing this is 
$ hat(bx)^lambda_I = sum_(sigma in S_(n-m)) sgn(sigma)rho^lambda (sigma)phi^({|I|})_"AS" (br_I, br_(sigma^(-1) ([n] backslash I))) $
This has no trivial simplification and is factorial time with respect to the number of particles. 
We will need to craft the features meticulously to loose as little important information as possible.
But this benefit of this construction is that the final irreps are readily anti-symmetric and we
can just take a linear combination of them.

Anti-symmetrization is unavoidable, and choosing to do it in the "correct" place is very important.
For simplicity, we again choose to do anti-symmetrization in the end. This sounds *very similar*
to many of the existing methods, but I think that there is a lot of room to explore based on this framework.

=== Pfaffian Readout

The most obvious readout method is using pfaffians on the (1,1) irrep.
$ psi(br) = sum_(c=1)^C b_c "Pf"[(hat(bx)^(c, (1,1)))_ij] $
For the cases where $n$ is odd, we can instead do 
$ psi(br) = sum_(c=1)^C b_c "Pf" mat(
  hat(bx)^(c"," (1","1)), hat(bx)^(c"," (1));
  (hat(bx)^(c"," (1)))^T, 0 
) $
This is because the tensor product of $n\/2$ $(1,1)$ irreps contain a copy of the $(1^n)$ irrep. 

Assuming that channels are sufficiently mixed in the SpechtMP layers, this is the only 
irrep with order $n$ that we can read-out from a network of maximum interaction order $M=2$. 

=== Generalized Pfaffian readout

For order $m=3$, the readout is (for $n=3r$):
$ Psi^(c,(1,1,1)) = sum_(sigma in S_n) epsilon.alt_(i_1 j_1 k_1 dots i_r j_r k_r) hat(bx)^(c, (1,1,1))_(i_1 j_1 k_1) dots hat(bx)^(c, (1,1,1))_(i_r j_r k_r) $
where $epsilon.alt$ is the generalized Levi-Civita tensor:
$ epsilon.alt_(i_1 j_1 k_1 dots i_r j_r k_r) =
cases(
  sgn(i_1 j_1 k_1 dots i_r j_r k_r) & ", no repeated entries",
  0 & ", otherwise"
)
$
A similar form exists for the Pfaffian, just with three-body interactions, but the catch is that while
the Pfaffian can be calculated in $O(n^3)$ time, the generalized Pfaffian of order 3 cannot. In fact, 
it is exponential in terms of $n$. Because of this, we have to abandon calculating the exact generalized
pfaffian for order-3 and above.

=== Higher-order readouts

Given that the maximum order $M=3$, the only channel-wise order-n read-outs are the pfaffian and the generalized
pfaffian, but we can increase the order to include other irreps, i.e., the polynomial order of the irreps in the
read-out phase $r$ has been correlated with $n$ in the two cases we presented ($r=n/2$ for pfaffians and $r = n/3$
for order-3 pfaffians), but taking higher order tensor products of irreps can result in more copies of $(1^n)$. 

The generalized pfaffian for polynomial order $r$ is 
$
  Psi_(m,r)(X)
  =
  sum_c
  sum_(phi in Phi_(m,r))
  b_(phi,c)
  sum_(alpha_1, dots.c, alpha_n)
  epsilon_(alpha_1 dots.c alpha_n)
  product_(k=1)^r
  a^c_(
    alpha_(phi(k,1))
    dots.c
    alpha_(phi(k,m))
  )
$
for $Phi_(m, r)$ the set of surjective functions $phi:{1 dots r}times{1 dots m}mapsto {1 dots n}$ and $a^c_I := hat(bx)^(c, T, lambda)_I$

A more familiar form uses the generalized Levi-Civita tensor:
$
E_phi(i_(1,1), dots.c, i_(r,m))
=
sum_(alpha_1, dots.c, alpha_n)
epsilon_(alpha_1 dots.c alpha_n)
product_(p=1)^r
product_(q=1)^m
delta_(i_(p,q), alpha_(phi(p,q)))
$
so we can define
$
Psi_(m,r)(X)
=
sum_(c=1)^C
sum_(phi in Phi_(m,r))
b_(phi,c)
sum_(i_(p,q))
E_phi(i_(1,1), dots.c, i_(r,m))
product_(p=1)^r
a^c_(i_(p,1) dots.c i_(p,m))
$
Note that the dimensions of $b_(phi,c)$ grows exponentially with $r$ and $n$. 

=== Channel-mixing

Since channels already mixes in SpechtMP, we consider it rather redundant to mix channels
again in the readout phase, but we must highlight a very common method of channel-mixing readout:
determinants.
$ Psi = det [hat(bx)^(c, T, (1))_i] $
We form a matrix with axis 0 being the channels and axis 1 being the particle index. 
This type of readout has been used extensively in mainstream NN-QMC methods.

== Envelopes

We apply scalar envelope factors outside the antisymmetric SpechtMP/readout stack:

$
psi(br) = exp(J_"env" (br)) psi_theta (br),
$

or equivalently,

$
log abs(psi(br))) = J_"env" (br) + log abs(psi_theta (br)).
$

This keeps smooth long-range factors and short-range cusp enforcement independent from the determinant/Pfaffian/Specht readout and preserves the antisymmetry of $psi_theta$.

`nn.SpENNWaveFunction` takes one required `nn.Envelope`. Composite envelopes use `nn.AdditiveEnvelope`, which adds the scalar outputs of its component envelopes. For the Hooke pair config, the composite envelope is the sum of `nn.HarmonicConfinement` and `nn.ElectronElectronCusp`.

For a harmonically confined system,

$
J_"conf"(br)
=
- alpha sum_i |r_i|^2,
quad alpha >= 0.
$

For Hooke or oscillator frequency $omega$, the fixed Gaussian ground-state tail uses

$
alpha = omega / 2.
$

This factor is smooth rather than cusp-like, but it is still additive in $log abs(psi)$ and should be applied through the envelope interface, not inside the antisymmetric readout.

The current cusp envelope implementation contains only the electron-electron term:

$
J_"cusp" (br)
=
J_"ee" (br).
$

Future nuclear and periodic long-range factors should be added as separate envelope modules.

=== Electron-electron cusp

==== Option A: fixed analytic electron-electron cusp

Use an explicit pairwise electron-electron cusp term

$
J_"ee"(R)
=
sum_(i < j) u_(sigma_i sigma_j)(r_ij),
$

with

$
u_(sigma_i sigma_j)(r)
=
frac(a_(sigma_i sigma_j) r, 1 + b_(sigma_i sigma_j) r).
$

The cusp slope is fixed by

$
u'_(sigma_i sigma_j)(0) = a_(sigma_i sigma_j).
$

Use separate slopes for same-spin and opposite-spin pairs:

$
a_"same" = 1 / 4,
quad
a_"opp" = 1 / 2.
$

The range parameters $b_"same"$ and $b_"opp"$ may be trainable, constrained positive by e.g.

$
b = "softplus"(tilde(b)) + epsilon.
$

This is the recommended MVP for electron-electron cusps. It is simple, stable, and enforces the short-range condition exactly by construction.

==== Option B: analytic electron-electron cusp plus smooth residual

Use the same fixed analytic cusp, but add a smooth residual term:

$
u_(sigma_i sigma_j)(r)
=
frac(a_(sigma_i sigma_j) r, 1 + b_(sigma_i sigma_j) r)
+
r^2 g_theta (r).
$

The $r^2$ factor enforces

$
frac(d, d r) [r^2 g_theta(r)]_(r=0) = 0,
$

so the residual does not alter the cusp slope.

This option gives more flexibility for electron-electron correlation while preserving exact short-range behavior.

=== Design decision

Implement Option A first for electron-electron cusps:

$
J_"ee"(R)
=
sum_(i < j)
frac(a_(sigma_i sigma_j) r_ij, 1 + b_(sigma_i sigma_j) r_ij),
$

Expose Option B as an optional extension:

$
"ee_residual": "none" | "smooth-r2".
$

Each envelope module should return a scalar

$
J(R)
$

per configuration and should be added directly to the model log-amplitude. It should not modify SpechtMP features or antisymmetric readout internals.

= Model Workflow
/*
Implemented in `nn.SpENNWaveFunction`.
+ Input: $br_i = (x_i, y_i, z_i, s_i)$
+ Encoder (`nn.Encoder`)
  + Learnable encoder $phi^({m}): br_I stretch(->) bq^{m}_I $. Packs tuples into $bq$ bundles
  + Unlearnable Projection into irrep space $bx^(0, lambda)_I = P^lambda bq_I$
+ Specht Message-Passing layers (`nn.SpechtMP`)
  + Specht MP layer 1 (`nn.SpechtMPLayer`) 
    + Project into real space with `reps.FusionMap`:
      $ bz^(1, c_1c_2, lambda<-lambda_1 lambda_2)_(sigma, I<-I_1I_2) =
      (d_lambda_1 d_lambda_2)/(m!)
        tr(bx^(0, c_1lambda_1)_I_1 times.o^"kr" bx^(0, c_2lambda_2)_I_2) rho^lambda (sigma^(-1)) $
    + Aggregate into message with `nn.MessageHead`
      $ bm^(1, c_"out"lambda)_(I; alpha beta) = sum_(I_1 or.curly I_2 = I) sum_(lambda_1 tack|I_1| \ lambda_2 tack|I_2|) & "activation"_lambda [ sum_(sigma in S_m)  sum_(c_1, c_2)
      & M^(1, c_"out"<-c_1 c_2 m<-m_1m_2)_(I<-I_1, I_2; beta) bz^(1, c_1c_2, lambda<-lambda_1 lambda_2)_(sigma, I<-I_1I_2; alpha beta)] $
    + Project into Branching space with `reps.BranchMap`:
      $ by^(1, c_"in",lambda<-mu)_(sigma, I<-J) = (d_mu)/(m!) tr(bm_J^(1,c_"in" mu)) rho^lambda (sigma^(-1)) $
    + Aggregate into the update tensor $bold("u")$ with `nn.UpdateHead`:
      $ bold("u")^(1, c_"out" lambda)_(I; alpha beta) = sum_(J prec.eq I) sum_(mu tack|I|) "activation"_lambda [sum_(sigma in S_m)  sum_(c_"in") O^(1, c_"out"<-c_"in",m<-|J|)_(sigma I <- J; beta) by^(1, c_"in",lambda<-mu)_(sigma, I<-J; alpha beta)] $
    + Update the irreps with `nn.Update`:
      $ bx^(1) = "update"(bold("u")^1, bx^0) $
  + SpechtMP layer 2 (`nn.SpechtMPLayer`)
  
    $dots$
  + SpechtMP layer T (`nn.SpechtMPLayer`)
+ Readout with `nn.PfaffianReadout`
  $ Psi = sum_(c) w^(c) "Pf"[bx^(T c (1,1))_(i j)] $
+ Applied envelope with `nn.Envelope`
  $ psi(br) = exp(J_"env" (br))Psi(br) $
+ Output: $psi(br)$*/

Implemented in `nn.SpENNWaveFunction`.

+ Input: particle positions $bv_i = (br_i, s_i)$ 
+ Embedding (`nn.Embedding`), learnable: $phi^(m): bv_I mapsto bx_I^(0, c, m)$
+ SpENN Stack (`nn.SpeNNStack`)
  + SpENN layer 1 (`nn.SpENNLayer`)
    + mixing in real space (`nn.EquivariantMixing`)
      $ bh^c_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(c<-c_1c_2)
      bx_(J circle.small tau_1)^c_1 bx_(J circle.small tau_2)^c_2 $
    + Project into irrep space with `reps.FourierTransform`
      $ hat(bh)^(c lambda) = P^lambda bh^c $
    + Activate in irrep space with `nn.activation.ActivatebyType`
      $ hat(bw)^(c lambda)_("Ord"(I), p; [d_lambda] beta) = Gamma_lambda (hat(bh)^(c lambda)_("Ord"(I), p; [d_lambda] beta)) $
      important note: all permutations of $I$ are activated together as one irrep. $alpha$ are considered components 
      of the same irrep so are also activated together.
    + Obtain update with `nn.PathAggregation`
      $ hat(bu)^(c lambda)_(I; alpha, b_"out") = sum_p sum_beta_"in"
      U^(c lambda)_("im"(I), p; beta_"out"<-beta_"in") hat(bw)^(c lambda)_("Ord"(I), p; alpha beta_"in")
       $
    + Project back into real space with `reps.InverseFourierTransform`:
      $ bu^(1, c)_(I)
      = sum_(lambda tack m) P^(-1) hat(bu)^(1, c, lambda)_(I) $
    + Feature update (`nn.update`)
      $ bx^1 = "Update"(bx^0, bu^1) $
      most commonly with `nn.update.ResidualUpdate`.
  + SpENN layer 2
    $
    bx^2 = "SpENNLayer"(bx^1).
    $

  + $dots$

  + SpechtMP layer $T$
    $
    bx^T = "SpENNLayer"(bx^(T-1)).
    $
+ Readout with `nn.RealPfaffianReadout`
  $ Psi = sum_(c) w^(c) "Pf"[bx^(T c)_(i j) - bx^(T c)_(j i)] $
+ Applied envelope with `nn.Envelope`
  $ psi(bv) = exp(J_"env" (br))Psi(bv) $
+ Output: $psi(bv)$


#pagebreak()
