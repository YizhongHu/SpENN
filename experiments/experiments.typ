== Two-electrons in a Harmonic Well

This experiment is a minimal two-electron benchmark for SpENN-QMC. The system is Hooke's atom: two interacting electrons in a three-dimensional harmonic trap with Coulomb electron-electron repulsion.

We use this experiment to test whether the implementation can learn correlation, enforce electron-electron cusps, preserve the correct exchange symmetry, and produce stable VMC estimates.

#let bx = $bold(x)$
#let bq = $bold(q)$
#let br = $bold(r)$
#let bz = $bold(z)$
#let bm = $bold(m)$

=== Hamiltonian

The Hamiltonian is

$
H
=
- (1)/(2) nabla_1^2
- (1)/(2) nabla_2^2
+ (1)/(2) omega^2 (r_1^2 + r_2^2)
+ (1)/(r_(12)),
$

where

$
r_(12) = norm(br_1 - br_2).
$

We use two analytic benchmark sectors.

==== Opposite-spin singlet benchmark

For the standard singlet benchmark, choose

$
omega = (1)/(2).
$

Then

$
H
=
- (1)/(2) nabla_1^2
- (1)/(2) nabla_2^2
+ (1)/(8) (r_1^2 + r_2^2)
+ (1)/(r_(12)).
$

The exact ground-state energy is

$
E_0 = 2.
$

The exact spatial wavefunction is symmetric:

$
psi_0 (br_1, br_2)
=
cal(N)
(1 + (1)/(2) r_(12))
exp(- (1)/(4) (r_1^2 + r_2^2)).
$

This benchmark tests the opposite-spin electron-electron cusp:

$
a_"opp" = (1)/(2).
$

==== Same-spin triplet benchmark

To force a same-spin test, use the antisymmetric spatial triplet sector. A convenient analytic benchmark is obtained with

$
omega = (1)/(4).
$

The exact energy is

$
E_T = (5)/(4).
$

One Cartesian component of the triplet spatial wavefunction is

$
psi_T (br_1, br_2)
=
cal(N)
(z_1 - z_2)
(1 + (1)/(4) r_(12))
exp(- (1)/(8) (r_1^2 + r_2^2)).
$

Equivalently, the relative-coordinate part has the form

$
psi_"rel" (br)
=
r Y_(1 m)(hat(br))
(1 + (1)/(4) r)
exp(- (1)/(16) r^2),
$

where

$
br = br_1 - br_2.
$

The three Cartesian choices

$
x_1 - x_2,
qquad
y_1 - y_2,
qquad
z_1 - z_2
$

are degenerate. This benchmark tests the same-spin cusp after factoring out the antisymmetric node:

$
a_"same" = (1)/(4).
$

=== Model form

The trial wavefunction should be written as

$
psi_theta (R)
=
exp(J_"cusp" (R)) psi_"SpENN" (R).
$

Equivalently,

$
log abs(psi_theta (R))
=
J_"cusp" (R)
+
log abs(psi_"SpENN" (R)).
$

The cusp module should be outside the antisymmetric or symmetric readout.

For the opposite-spin singlet benchmark, the spatial wavefunction is symmetric:

$
psi_theta (br_1, br_2)
=
psi_theta (br_2, br_1).
$

For the same-spin triplet benchmark, the spatial wavefunction is antisymmetric:

$
psi_theta (br_1, br_2)
=
- psi_theta (br_2, br_1).
$

=== Electron-electron cusp

Use

$
J_"ee" (R)
=
u_(sigma_1 sigma_2)(r_(12)),
$

with the minimal analytic form

$
u_(sigma_1 sigma_2)(r)
=
(a_(sigma_1 sigma_2) r)/(1 + b_(sigma_1 sigma_2) r).
$

For the singlet benchmark,

$
a_(sigma_1 sigma_2) = a_"opp" = (1)/(2).
$

For the triplet benchmark,

$
a_(sigma_1 sigma_2) = a_"same" = (1)/(4).
$

Here $b_(sigma_1 sigma_2)$ may be fixed or trainable with a positive parameterization such as

$
b_(sigma_1 sigma_2)
=
"softplus"(tilde(b)_(sigma_1 sigma_2)) + epsilon.
$

The cusp factor should be tested independently before training the full model.

=== SpENN feature pipeline

The experiment should support the same feature pipeline used elsewhere in the codebase:

$
bx -> bz -> bm -> by -> bx^"new".
$

If using ordered tuple-space features, store every ordered tuple explicitly. For example, for pair order, both

$
bq_(i j)
quad "and"
quad
bq_(j i)
$

exist as separate tuple coordinates.

The tuple feature map should respect particle relabeling:

$
(pi bq)_I = bq_(pi^(-1) I).
$

Fusion maps should be equivariant real-space maps

$
M_p : cal(T)_(I_1) times cal(T)_(I_2) -> cal(T)_I.
$

A basis for $M_p$ may be constructed from simultaneous relabeling orbits of coefficient triples

$
(tau ; tau_1, tau_2)
in
"Ord"(I) times "Ord"(I_1) times "Ord"(I_2).
$

The learned path index $p$ should index completed equivariant maps, not raw ordering variables.

=== Training objective

This experiment should not use supervised training against the analytic wavefunction. The exact solution is available only as a benchmark and diagnostic reference.

The model is trained variationally by minimizing the Monte Carlo estimate of the energy:

$
cal(L)(theta)
=
bb(E)_(R ~ abs(psi_theta)^2) [E_"loc" (R)].
$

The local energy is

$
E_"loc" (R)
=
(H psi_theta (R))/(psi_theta (R)).
$

Samples $R_k$ should be drawn from the current model distribution

$
R_k ~ abs(psi_theta)^2,
$

using Metropolis-Hastings, MALA, or another valid VMC sampler.

The primary reported energy estimate is

$
overline(E)
=
(1)/(N) sum_(k=1)^N E_"loc" (R_k),
$

and the local-energy variance is

$
Var(E_"loc")
=
(1)/(N) sum_(k=1)^N
(E_"loc" (R_k) - overline(E))^2.
$

The analytic wavefunction should not appear in the training loss. In particular, do not train with losses such as

$
norm(log abs(psi_theta (R)) - log abs(psi_"exact" (R)))^2,
$

$
norm(psi_theta (R) - psi_"exact" (R))^2,
$

or any supervised regression target derived from $psi_"exact"$.

The exact solution is used only to define reference quantities:

$
E_"exact" =
cases(
  2, "opposite-spin singlet",
  (5)/(4), "same-spin triplet",
).
$

A correct and expressive model should approach

$
overline(E) -> E_"exact"
$

through variational optimization, not through direct fitting to the analytic wavefunction.

=== Role of the analytic solution

The analytic Hooke's-atom solutions are used in three ways.

==== Code validation

Before training, evaluate the local energy of the exact wavefunction. This verifies the Hamiltonian, kinetic-energy, potential-energy, and automatic-differentiation code.

For the singlet benchmark,

$
E_"loc" [psi_0] (R) = 2
$

up to numerical precision.

For the triplet benchmark,

$
E_"loc" [psi_T] (R) = (5)/(4)
$

up to numerical precision away from the nodal surface.

This step is a unit test for the physics code, not a training procedure.

==== Reference energy

During VMC training, plot the learned energy against the known exact energy. The exact energy is only a horizontal reference line.

The optimization objective remains

$
bb(E)_(R ~ abs(psi_theta)^2) [E_"loc" (R)].
$

==== Post-training diagnostics

After training, the analytic wavefunction may be used for diagnostic plots such as radial cuts or qualitative shape comparisons. These comparisons should not contribute gradients and should not be used as a loss.

For example, for the singlet benchmark, one may compare the learned radial dependence against

$
(1 + (1)/(2) r_(12)) exp(- (1)/(8) r_(12)^2)
$

at fixed center of mass, up to normalization.

For the triplet benchmark, after dividing out the selected antisymmetric Cartesian factor, compare against

$
(1 + (1)/(4) r_(12)) exp(- (1)/(16) r_(12)^2)
$

at fixed center of mass, up to normalization.

These comparisons are for interpretation only.

=== Key indicators of correctness

==== Energy

The final VMC energy should approach the exact benchmark energy:

$
E_0 = 2
quad "for the singlet benchmark",
$

and

$
E_T = (5)/(4)
quad "for the triplet benchmark".
$

The energy should improve through VMC energy minimization. Do not use the analytic wavefunction as a supervised target.

Report:

$
overline(E),
qquad
E_"exact",
qquad
overline(E) - E_"exact",
qquad
abs(overline(E) - E_"exact").
$

==== Local-energy variance

The exact wavefunction has constant local energy. Therefore the variance of

$
E_"loc" (R)
$

is a strong correctness indicator.

A decreasing local-energy variance usually indicates that the learned wavefunction is approaching an eigenstate. The variance should be computed on samples from the learned distribution, not from supervised samples generated from the analytic solution.

==== Cusp behavior

For the singlet benchmark, verify

$
(d)/(d r_(12)) log psi_theta
bar.v_(r_(12) = 0)
approx
(1)/(2).
$

For the triplet benchmark, after factoring out the antisymmetric relative coordinate, verify the same-spin cusp slope

$
(d)/(d r_(12))
log (
  psi_theta (br_1, br_2) / (z_1 - z_2)
)
bar.v_(r_(12) = 0)
approx
(1)/(4),
$

or use the corresponding Cartesian component selected for the triplet state.

The cusp module should satisfy the appropriate slope before training. During VMC training, the cusp factor is part of the ansatz, not a supervised target.

==== Exchange symmetry

For the singlet benchmark, check

$
Delta_"swap,singlet"
=
abs(
log abs(psi_theta (br_1, br_2))
-
log abs(psi_theta (br_2, br_1))
).
$

For the triplet benchmark, check antisymmetry at the signed wavefunction level:

$
psi_theta (br_1, br_2)
+
psi_theta (br_2, br_1)
approx
0.
$

If the implementation outputs sign and log-amplitude separately, verify both the antisymmetric sign behavior and the symmetric log-amplitude behavior away from the nodal surface.

==== Sampler health

Track the Metropolis acceptance ratio. A useful target range is roughly

$
0.3 <= A <= 0.7.
$

Very low acceptance indicates overly large moves. Very high acceptance indicates moves that are too small and slow mixing.

Also track simple distributional statistics such as

$
bb(E)[r_(12)],
qquad
Var(r_(12)),
qquad
min r_(12),
qquad
max r_(12).
$

These help distinguish model failures from sampler failures.

=== Recommended plots

==== Energy trace

Plot training step on the horizontal axis and estimated VMC energy on the vertical axis.

Include:

- raw batch energy,
- running mean,
- horizontal reference line at the relevant exact energy,
- optional error bands from Monte Carlo uncertainty.

Use $E_0 = 2$ for the singlet benchmark and $E_T = (5)/(4)$ for the triplet benchmark.

This plot should show variational energy minimization. It should not show supervised regression loss against the analytic wavefunction.

==== Local-energy variance trace

Plot

$
Var(E_"loc")
$

against training step.

Use a log vertical scale if the variance changes over many orders of magnitude. A successful run should show variance decreasing along with the energy.

==== Cusp diagnostic

For the singlet benchmark, fix the center of mass and vary

$
r_(12)
$

near zero. Plot

$
log psi_theta
$

or

$
(d)/(d r_(12)) log psi_theta
$

against $r_(12)$. The slope near zero should approach

$
(1)/(2).
$

For the triplet benchmark, divide out the chosen antisymmetric Cartesian factor, for example

$
z_1 - z_2,
$

and plot

$
log abs(psi_theta (br_1, br_2) / (z_1 - z_2))
$

or its radial derivative near coalescence. The slope should approach

$
(1)/(4).
$

==== Wavefunction shape diagnostic

This is a post-training diagnostic only. It must not be used as a training loss.

For the singlet benchmark, at fixed center of mass, compare the learned radial dependence against

$
(1 + (1)/(2) r_(12)) exp(- (1)/(8) r_(12)^2).
$

For the triplet benchmark, after dividing by the selected Cartesian antisymmetric factor, compare against

$
(1 + (1)/(4) r_(12)) exp(- (1)/(16) r_(12)^2).
$

The normalization is irrelevant, so compare normalized curves or log-amplitudes up to an additive constant.

==== Pair-distance histogram

Plot the sampled distribution of

$
r_(12).
$

This checks whether the sampler explores the physically relevant region and whether the Coulomb repulsion is reflected in the learned density.

==== Exchange-symmetry error

For the singlet benchmark, plot or summarize

$
Delta_"swap"
=
abs(
log abs(psi_theta (br_1, br_2))
-
log abs(psi_theta (br_2, br_1))
).
$

For the triplet benchmark, plot or summarize

$
Delta_"antisym"
=
abs(
psi_theta (br_1, br_2)
+
psi_theta (br_2, br_1)
).
$

These should be near numerical precision away from pathological points such as exact nodes.

=== Recommended experiment sequence

1. Test the Hamiltonian and local-energy code using the exact analytic singlet wavefunction. This is a unit test, not training.

2. Test the Hamiltonian and local-energy code using the exact analytic triplet wavefunction. This is a unit test, not training.

3. Test the cusp module alone and verify the derivative condition for both $a_"opp" = (1)/(2)$ and $a_"same" = (1)/(4)$.

4. Train a small SpENN model on the singlet benchmark using only VMC energy minimization.

5. Train a small SpENN model on the triplet benchmark using only VMC energy minimization.

6. Verify convergence toward the appropriate exact energy.

7. Use the analytic wavefunction only for post-training diagnostics such as radial-shape plots and exact-energy reference lines.

8. Add the smooth cusp residual only after the fixed-cusp VMC experiments are stable.

9. Compare energy, local-energy variance, cusp slope, sampler health, and exchange-symmetry error across ablations.