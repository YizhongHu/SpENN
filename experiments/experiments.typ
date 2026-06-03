== Two-electrons in a Harmonic Well

This experiment is a minimal two-electron benchmark for SpENN-QMC. The system is Hooke's atom: two interacting electrons in a three-dimensional harmonic trap with Coulomb electron-electron repulsion.

We use this experiment to test whether the implementation can learn correlation, enforce electron-electron cusps, preserve the correct exchange symmetry, and produce stable VMC estimates.

#let bx = $bold(x)$
#let by = $bold(y)$
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

The exact ground-state energy is`

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
quad
y_1 - y_2,
quad
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
"Var"(E_"loc")
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
quad
E_"exact",
quad
overline(E) - E_"exact",
quad
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
quad
"Var"(r_(12)),
quad
min r_(12),
quad
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
"Var"(E_"loc")
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

== Hooke Multibody

This experiment extends the two-electron Hooke benchmark to a multi-electron harmonic trap with Coulomb electron-electron interactions. The purpose is to test whether SpENN-QMC scales from pair-only physics to genuinely many-body configurations while preserving the same training discipline used in `hooke_pair`.

This experiment should reuse the structure of the existing Hooke pair experiment as much as possible. The implementation should not create a separate parallel framework unless the existing abstractions are insufficient.

=== Codex implementation instructions

Codex should follow these requirements:

+ Use a file structure similar to the existing `hooke_pair` / `hooke` experiment.
+ Reuse as much code as possible from the pair experiment.
+ Use `run\_\*.py` files only as launch and data-processing scripts.
+ Train SpENN separately using VMC energy minimization only.
+ Do not train SpENN against the exact solution.
+ Compare to the exact solution only during post-training data processing.
+ Produce reusable CSV / JSON / tensor data for comparison.
+ Produce graphs for visual inspection of correctness.
+ Write a report summarizing the experiment, results, plots, and reproduction commands.

In particular, the experiment should preserve the separation:

$
"training" != "exact comparison".
$

The exact solution is a reference and diagnostic tool, not a supervised target.

=== Suggested file structure

The experiment should mirror the existing Hooke pair structure.

```text
experiments/
  hooke_multibody/
    configs/
      exact.yaml
      spenn.yaml
      smoke.yaml
      benchmark.yaml
    run_exact.py
    run_spenn.py
    process_outputs.py
    plot_outputs.py
    report.md
    figures/
      exact/
      spenn/
      summary/
    slurm/
      cpu_smoke.job
      gpu_smoke.job
      logs/
```

The `run_*.py` files should stay thin:

+ load config,
+ apply command-line overrides,
+ call reusable training / evaluation / exact-reference utilities,
+ write artifacts.

They should not contain core Hamiltonian, sampler, model, or plotting logic unless the logic is experiment-specific glue.

=== Hamiltonian

Use the $N$-electron harmonic Coulomb Hamiltonian

$
H
=
sum_(i=1)^N
(
  - (1)/(2) nabla_i^2
  + (1)/(2) omega^2 r_i^2
)
+
sum_(1 <= i < j <= N)
(1)/(r_(i j)),
$

where

$
r_(i j) = norm(br_i - br_j).
$

The two-electron Hooke pair experiment is the special case $N = 2$.

For multibody experiments, the default goal is not necessarily to use a closed-form analytic wavefunction. If an exact or high-accuracy reference is available, it should be generated by a separate exact-reference pipeline and used only after training.

=== Model form

Use the same SpENN-QMC model contract as in the pair experiment:

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

The electron-electron cusp is

$
J_"ee" (R)
=
sum_(1 <= i < j <= N)
u_(sigma_i sigma_j)(r_(i j)).
$

Use the same analytic cusp form as the pair experiment:

$
u_(sigma_i sigma_j)(r)
=
(a_(sigma_i sigma_j) r)/(1 + b_(sigma_i sigma_j) r).
$

The slopes are

$
a_"opp" = (1)/(2),  a_"same" = (1)/(4).
$

The spin or particle-state convention must be explicit. If spin labels are part of the particle state, then particle permutations move positions and spin labels together, and the wavefunction should be fully antisymmetric under permutation of particle tokens.

=== SpENN feature pipeline

This experiment should use the updated design:

$
"mix in real tuple space, activate in local irrep space".
$

The persistent hidden state should be ordered tuple features

$
bq_I^c,  |I| <= M.
$

For this experiment, keep

$
M <= 3
$

unless there is a specific reason to go higher.

Linear equivariant transport should happen in ordered tuple space using real-space fusion and pooling maps. Specht projections should be used for activation, not as the persistent hidden-state representation.

The layer structure should conceptually be

$
bq
-> bq_"lin"
-> P bq_"lin"
-> Gamma_(m, lambda)
-> P^(-1)
-> bq_"next".
$

For each support orbit $"Ord"(I)$, gather all orderings of $I$, project to local Specht irreps, apply irrep-wise activation, and reconstruct to ordered tuple space.

For $m = 2$, this means

$
s_(i j) = (1)/(2) (bq_(i j) + bq_(j i)),
$

$
a_(i j) = (1)/(2) (bq_(i j) - bq_(j i)).
$

The symmetric and antisymmetric components should be activated separately and then reconstructed:

$
bq_(i j) = s_(i j) + a_(i j),
$

$
bq_(j i) = s_(i j) - a_(i j).
$

Do not activate $bq_(i j)$ and $bq_(j i)$ independently if using representation-aware activation.

=== Training objective

Training must use VMC energy minimization only.

The local energy is

$
E_"loc" (R)
=
(H psi_theta (R))/(psi_theta (R)).
$

The optimization objective is

$
cal(L)(theta)
=
bb(E)_(R ~ abs(psi_theta)^2) [E_"loc" (R)].
$

Samples should be drawn from the current model distribution

$
R_k ~ abs(psi_theta)^2.
$

Do not train using losses of the form

$
norm(log abs(psi_theta (R)) - log abs(psi_"exact" (R)))^2,
$

$
norm(psi_theta (R) - psi_"exact" (R))^2,
$

or any other supervised regression loss against an exact or high-accuracy reference wavefunction.

The exact solution or exact-reference data should be used only after training, during data processing and plotting.

=== Exact-reference role

The exact or high-accuracy reference is used only for:

+ validating Hamiltonian and local-energy code,
+ defining reference energies,
+ producing post-training comparison data,
+ producing plots for visual inspection,
+ writing the final report.

It must not contribute gradients to the SpENN model.

If an exact analytic solution is unavailable for the chosen $N$, then use the best available reference method and clearly label it as a reference rather than an analytic exact solution.

Possible reference quantities include:

+ reference energy,
+ reference pair-distance distribution,
+ reference radial density,
+ reference one-body density,
+ reference local-energy statistics,
+ reference symmetry / antisymmetry checks.

=== Data products

Each run should write reusable data artifacts, not only figures.

Recommended outputs:

```text
metrics/train_metrics.csv
metrics/sampler_metrics.csv
metrics/eval_metrics.csv
data/reference_observables.csv
data/spenn_observables.csv
data/pair_distance_histogram.csv
data/radial_density.csv
data/local_energy_samples.csv
plots/*.png
report.md
```

The comparison pipeline should be able to regenerate plots from saved data without rerunning training.

=== Key indicators of correctness

==== Energy

Track

$
overline(E)
=
(1)/(N_"samples")
sum_(k=1)^(N_"samples")
E_"loc" (R_k).
$

Report

$
overline(E),  E_"ref",  overline(E) - E_"ref",  abs(overline(E) - E_"ref").
$

The energy should approach the reference through VMC optimization, not supervised fitting.

==== Local-energy variance

Track

$
"Var"(E_"loc")
=
(1)/(N_"samples")
sum_(k=1)^(N_"samples")
(E_"loc" (R_k) - overline(E))^2.
$

Decreasing local-energy variance is a key sign that the learned wavefunction is approaching an eigenstate.

==== Cusp behavior

Verify electron-electron cusp slopes separately for same-spin and opposite-spin pairs:

$
a_"same" = (1)/(4),  a_"opp" = (1)/(2).
$

For multibody systems, cusp diagnostics should be averaged over many pairs and configurations, but should also report worst-case or high-percentile errors.

==== Exchange antisymmetry

If particle states include spin labels, test full particle-token antisymmetry:

$
psi_theta (x_(pi(1)), dots, x_(pi(N)))
=
"sgn"(pi) psi_theta (x_1, dots, x_N).
$

Here

$
x_i = (br_i, sigma_i).
$

If the implementation uses a fixed-spin-sector convention instead, then the report must state this explicitly and only same-spin exchanges should be tested as antisymmetric.

==== Sampler health

Track:

+ acceptance rate,
+ proposal scale,
+ mean pair distance,
+ minimum pair distance,
+ local-energy autocorrelation,
+ effective sample size if available.

A useful acceptance-rate range is roughly

$
0.3 <= A <= 0.7.
$

=== Recommended plots

==== Energy trace

Plot training step versus VMC energy.

Include:

+ raw batch energy,
+ running mean,
+ reference energy line,
+ uncertainty bands if available.

==== Local-energy variance trace

Plot

$
"Var"(E_"loc")
$

against training step. Use a log vertical axis if the scale changes substantially.

==== Pair-distance distribution

Plot the distribution of

$
r_(i j)
$

over all electron pairs and sampled configurations. If a reference distribution exists, plot SpENN and reference curves together.

==== Radial density

Plot the one-body radial density

$
rho(r)
$

for the harmonic trap. Compare against the reference if available.

==== Cusp diagnostic

Plot measured short-range derivative estimates for same-spin and opposite-spin pairs. Include target slopes

$
(1)/(4)  "and"  (1)/(2).
$

==== Antisymmetry diagnostic

Plot or report the distribution of

$
Delta_"antisym"
=
abs(
psi_theta (pi R)
-
"sgn"(pi) psi_theta (R)
).
$

If sign and log-amplitude are represented separately, report sign-flip accuracy and log-amplitude consistency separately.

=== Report requirements

The final report should include:

+ experiment purpose,
+ Hamiltonian,
+ model configuration,
+ training configuration,
+ sampler configuration,
+ reference method,
+ statement that SpENN was not supervised against the reference,
+ energy table,
+ local-energy variance table,
+ cusp diagnostics,
+ antisymmetry diagnostics,
+ plots,
+ reproduction commands,
+ known limitations.

The report should clearly separate:

+ exact/reference runs,
+ SpENN training runs,
+ post-training comparison and plotting.

=== Recommended experiment sequence

1. Reuse the Hooke pair file structure for `hooke_multibody`.

2. Implement exact or high-accuracy reference generation as a separate pipeline.

3. Validate the Hamiltonian and local-energy code against reference data.

4. Train SpENN using only VMC energy minimization.

5. Save training artifacts and raw evaluation data.

6. Run a separate data-processing script to compare SpENN outputs to the reference.

7. Generate plots for visual inspection.

8. Write a report with tables, plots, and reproduction commands.

9. Add CPU and GPU smoke tests analogous to the Hooke pair experiment.

10. Only after the experiment is stable, add larger $N$, larger $M$, or more expressive readouts.