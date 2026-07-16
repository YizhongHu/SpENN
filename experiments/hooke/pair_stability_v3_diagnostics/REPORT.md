# Hooke pair-stability V3: cusp/tail diagnostic report

## Scope

This report diagnoses the selected V3 champion only:

```text
basis                    Hooke S1 / max_shell=1 (B00)
update normalization     none (U00)
feature normalization    none (F01)
activation               SiLU
action                   one SpENN layer
```

The analysis deliberately excludes `HookeOrbitalGenerator`. Its orbital construction is known to be a listed-coordinate basis rather than a product orbital, so it must not inform conclusions here.

The reference is the exact two-electron Hooke singlet at $\omega=1/2$:

$$
\psi_\mathrm{exact}(r_1,r_2)
= \left(1+\frac{r_{12}}2\right)
  \exp\left[-\frac14\left(|r_1|^2+|r_2|^2\right)\right],
\qquad E=2.
$$

See `spenn/physics/hooke.py` for the executable reference.

## Executive conclusion

The observed cusp and tail energy failures are **kinetic-energy errors induced primarily by the current analytic pair envelope**, not by the Hooke evaluator, Coulomb cancellation, readout conditioning, or equivariance failure.

The current model correctly supplies the Hooke Gaussian tail, and its cusp envelope has the correct first derivative at coalescence. Its pair-factor curvature is wrong, however. The learned model partly compensates for that mismatch but does not remove it.

This is the leading explanation, not a claim that the learned residual contributes nothing. A controlled architecture/retraining comparison is required to assign the remaining residual exactly.

## 1. Evaluation evidence

All deterministic diagnostics used the same grids:

```text
cusp: 15,360 configurations
tail:  1,536 configurations
```

| Variant | Cusp $\overline{E_L}-2$ | Tail $\overline{E_L}-2$ | Tail behavior |
|---|---:|---:|---|
| Exact Hooke reference | $-1.91\times10^{-11}$ | $0$ | finite; no pathology |
| Fresh champion architecture | +2.063991 | +0.907054 | 28 outliers; 1 pathology |
| Trained champion, replica 0 | +1.790455 | +0.464764 | no outliers or pathologies |

The exact cusp reference remains in
`[1.9999999994615791, 2.000000000003638]`, even though the individual Coulomb and kinetic terms reach magnitude $10^5$. Therefore the evaluator resolves the cancellation accurately.

### Hamiltonian components

At every evaluated geometry:

- `term/electron_electron` agrees exactly across exact, fresh, and trained runs;
- `term/harmonic_trap` agrees exactly across exact, fresh, and trained runs;
- the full local-energy error equals the kinetic-term error.

At $r_{12}=10^{-5}$:

| Variant | Electron-electron | Kinetic | Local energy |
|---|---:|---:|---:|
| Exact | 100000.000000 | -99998.000000 | 2.000000 |
| Fresh | 100000.000000 | -99995.070378 | 4.929622 |
| Trained r0 | 100000.000000 | -99996.051170 | 3.948830 |

Term sums reconstruct total local energy to at most $6.9\times10^{-12}$ on the cusp grid and $7.1\times10^{-15}$ on the tail grid.

## 2. Why the cusp energy remains wrong despite a correct cusp slope

The output envelope configured for this model is:

$$
\log |\psi| = \log|\psi_\mathrm{Pfaffian}|
-\frac14\left(|r_1|^2+|r_2|^2\right)
+u_b(r_{12}),
$$

where the current opposite-spin `ElectronElectronCusp` default is

$$
u_{b=1}(r)=\frac{r/2}{1+r}.
$$

The exact pair factor is instead

$$
u_\mathrm{exact}(r)=\log\left(1+\frac r2\right).
$$

Their short-range expansions are:

$$
\begin{aligned}
u_\mathrm{exact}(r)&=\frac r2-\frac{r^2}{8}+O(r^3),\\
u_{b=1}(r)&=\frac r2-\frac{r^2}{2}+O(r^3).
\end{aligned}
$$

Both have the exact Kato slope $u'(0)=1/2$. Their curvature differs:

$$
u_\mathrm{exact}''(0)=-\frac14,
\qquad
u_{b=1}''(0)=-1.
$$

The kinetic operator depends on first and second spatial derivatives of `logabs`. Thus a correct cusp slope does not imply a correct cusp local energy.

For the exact Hooke Gaussian plus rational factor

$$
u_b(r)=\frac{r/2}{1+br},
$$

the coalescence local-energy limit is

$$
E_L^{(b)}(0)=1.25+3b.
$$

| Pair factor | $E_L(0)$ | Error |
|---|---:|---:|
| Exact $\log(1+r/2)$ | 2.0000 | 0 |
| Current rational factor, $b=1$ | 4.2500 | +2.2500 |
| Curvature-matched rational factor, $b=0.25$ | 2.0000 | 0 |

This envelope-only calculation matches the measured graph closely:

| $r_{12}$ | Exact | Current-envelope theory | Fresh | Trained r0 | $b=0.25$ theory |
|---:|---:|---:|---:|---:|---:|
| $10^{-5}$ | 2.000000 | 4.249953 | 4.138176 | 3.879140 | 2.000001 |
| $0.2$ | 2.000000 | 3.520640 | — | 3.138196 | 2.020488 |

Training lowers the cusp bias, but the output-side pair prior leaves a large kinetic correction for the Pfaffian/SpENN residual to learn.

The measured cusp diagnostics agree with this interpretation:

| Metric | Fresh | Trained r0 |
|---|---:|---:|
| Even-slope absolute error | $2.10\times10^{-5}$ | $1.12\times10^{-5}$ |
| Odd directional slant | 0.8667 | 0.01124 |

The trained model learns directional cusp symmetry while retaining the wrong local kinetic curvature.

## 3. Tail theory

The V3 `TailGridGenerator` is a center-of-mass tail diagnostic:

$$
r_1=R+\frac d2,
\qquad
r_2=R-\frac d2,
\qquad
|d|=r_{12}=1.
$$

It varies the center-of-mass radius $R$ while holding the pair distance at one. Therefore it is not directly a large-$r_{12}$ test.

The `HookeGaussianEnvelope` is already exact in this coordinate. With the exact Gaussian and the current $b=1$ pair factor, the analytic local energy at the tail grid's fixed $r_{12}=1$ is

$$
E_L=2.421875.
$$

The trained replica has mean tail local energy

$$
\overline{E_L}=2.464764.
$$

Consequently, the current-envelope prediction accounts for 90.8% of the trained mean tail bias:

$$
\frac{2.421875-2}{2.464764-2}=0.908.
$$

The $b=0.25$ rational counterfactual gives

$$
E_L(r_{12}=1)=2.0456,
$$

before any learned correction.

The trained tail log-amplitude also supports retaining the fixed Gaussian envelope. Over $R\ge4$:

$$
\log|\psi_\mathrm{trained}|-\log|\psi_\mathrm{exact}|
\approx -3.0437 + 0.001075R^2,
$$

with RMS residual `0.0258`. The constant is an irrelevant normalization. The small $R^2$ coefficient shows no gross Gaussian-exponent failure.

Training still has a useful stabilization effect:

| Metric | Fresh | Trained r0 |
|---|---:|---:|
| Mean tail local energy | 2.907054 | 2.464764 |
| Local-energy variance | 1371.22 | 0.03225 |
| Outliers | 28 | 0 |
| Pathologies | 1 | 0 |

## 4. Why global MCMC energy did not expose this

Across nine final trained replicas, MCMC energy error was only:

```text
range:              +0.006325 to +0.019707
median absolute:    0.009188
```

[INFERENCE] The contrast with deterministic cusp/tail errors is consistent with the variational sampling distribution underweighting these geometries. The MCMC objective can therefore accept a model that is accurate where it samples while retaining substantial off-distribution kinetic bias.

## 5. Traces and numerical stability

No trace evidence supports a numerical readout or equivariance failure:

- zero feature nonfinites across all nine final replicas;
- zero trace-equivariance failures; maximum absolute error $1.42\times10^{-14}$;
- zero near-zero readout matrices;
- all 512 recorded readout condition numbers per replica equal `1.0`.

The largest feature magnitude in every replica occurs at:

```text
layers.0.irrep_activation/output
```

Replica 0 reaches maximum absolute magnitude `49.9544` there. The configured operation is

```python
tensor * SiLU(norm_sq)
```

so large norms can produce roughly cubic feature amplification. This is worth a bounded-gate ablation, but it is not the primary current explanation: all traces remain finite, and activation magnitude has no positive replica-level association with cusp/tail mean error in the nine-replica sample.

## 6. Training cost

The champion completed 500 steps in `405.08 s`.

| Phase | Seconds / step | Fraction of rolling step time |
|---|---:|---:|
| Local-energy evaluation | 0.4703 | 59.6% |
| Sampling | 0.1700 | 21.5% |
| Forward | 0.0398 | 5.0% |
| Backward | 0.0399 | 5.1% |

The primary cost is local-energy evaluation, not the forward feature stack. Term-level runtime attribution has not yet been recorded, so it would be incorrect to assign all of that cost to kinetic energy without a dedicated profile.

## 7. Architecture changes, ranked

### A. Hooke-exact pair envelope — recommended

Add a Hooke-specific output envelope:

```python
log_pair_factor = torch.log1p(0.5 * r12)
```

and replace the generic rational `ElectronElectronCusp` for this $\omega=1/2$ singlet benchmark.

This gives exact cusp slope, exact cusp curvature, and exact relative-coordinate factor. The exact two-body Hooke state can then be represented by a constant learned residual. It adds no SpENN layers or channels.

This should be a Hooke-specific option, not a replacement for the generic Coulomb envelope in unrelated systems.

### B. Curvature-matched range ablation — lowest implementation cost

Use the existing API:

```yaml
_target_: spenn.nn.ElectronElectronCusp
opposite_range_parameter: 0.25
```

This matches the exact Hooke pair factor through the $r^2$ coefficient at coalescence. It is not globally exact, but analytically changes the envelope-only error from:

```text
cusp r12 -> 0:  +2.25  to  0
tail r12 = 1:  +0.421875  to  +0.0456
```

It requires a clean architecture/retraining comparison. The checkpoint restore contract correctly rejects a post-hoc model-config mutation, so no altered-checkpoint result is claimed here.

### C. Exact base plus cusp-jet-preserving residual Jastrow

Use

$$
\log|\psi|=
\ell_\mathrm{base}+
\log(1+r_{12}/2)+
\frac{r_{12}^3}{1+r_{12}^3}\,h(r_{12}^2,R^2).
$$

A bounded scalar invariant function $h$ can learn deviations from the Hooke reference. The $O(r_{12}^3)$ prefactor prevents the learned residual from changing the value, first derivative, or second derivative at coalescence. A bounded tail factor protects the Gaussian leading asymptotic.

This is a small scalar pair branch, not more equivariant channels/layers.

### D. Explicit center-of-mass / relative-coordinate scalar head

For the two-electron singlet, factor the correction as a scalar invariant function of $(r_{12}^2,R^2)$ alongside the general Pfaffian path. The Pfaffian/equivariant stack retains its general role; the dedicated branch handles the simple radial pair physics that determines the local kinetic energy.

For larger systems, generalize this as a sum of pairwise Jastrow terms plus the many-body residual.

### E. Bounded norm-gate ablation — lower priority

After fixing the pair envelope, compare the current SiLU norm gate with a bounded gate such as Tanh or a saturating rational gate. Keep a residual linear route. This targets large Hessians from norm amplification, but current evidence does not support treating it as the first fix.

## 8. Plot-ready artifacts

All diagnostic data are preserved under:

```text
experiments/hooke/pair_stability_v3_diagnostics/results/analysis_rep-0/
```

| File | Intended plot |
|---|---|
| `cusp_pair_factor_theory.csv` | Pair factor, first/second derivative, and cusp local energy: exact vs current $b=1$ vs $b=0.25$ vs fresh/trained |
| `cusp_envelope_theory.csv` | Compact cusp-energy overlay |
| `tail_envelope_theory.csv` | Tail local energy vs center-of-mass radius: exact/current/$b=0.25$/fresh/trained |
| `tail_logabs_residual_profile.csv` | Trained-minus-exact log amplitude vs $R^2$ |
| `component_error_summary.csv` | Hamiltonian-component attribution |
| `layer_trace_summary.csv` | Per-layer feature scales |
| `training_timing_summary.csv` | Training-cost breakdown |

Recommended V3 report figures:

1. **Cusp local energy vs $r_{12}$ on a log x-axis.** Overlay exact, fresh, trained, current-envelope theory, and the $b=0.25$ analytic counterfactual.
2. **Pair-factor curvature $u''(r_{12})$.** This directly explains the distinction between correct Kato slope and incorrect kinetic energy.
3. **Tail local energy vs center-of-mass radius.** Overlay the constant current-envelope prediction `2.421875` and the $b=0.25$ counterfactual `2.0456`.
4. **Tail log-amplitude residual vs $R^2$.** Remove the arbitrary constant offset before plotting to show the near-correct Gaussian exponent.

Label all $b=0.25$ curves as **analytic architecture counterfactuals**, not trained-model measurements.

## 9. Delivered implementation and verification

PR: https://github.com/YizhongHu/SpENN/pull/132

The patch:

- adds an exact Hooke cusp/tail diagnostic config;
- permits sampled record files to include named Hamiltonian-term energies;
- enables per-term records for final cusp and tail tasks;
- adds sampled-record coverage for both enabled and default term-column behavior.

Verification:

```text
uv run --extra cpu pytest tests/unit/evaluation/test_hooke_evaluation_tasks.py -q
10 passed
```

The exact, fresh, and restored-champion diagnostic runs completed. A focused patch review returned no findings.

## 10. Addendum: general Coulomb cusp policy

### The cusp/tail connection

The concise conclusion is: **the same wrong pair-factor parameterization is the leading source of both graphs, but the tail error is not literally the $r_{12}\to0$ curvature coefficient transported outward.**

At coalescence, the $b=1$ rational factor has the correct slope and wrong curvature. At the tail-grid value $r_{12}=1$, that same factor has wrong finite-distance derivatives:

$$
\begin{aligned}
u_\mathrm{exact}'(1)&=1/3, &u_{b=1}'(1)&=1/8,\\
u_\mathrm{exact}''(1)&=-1/9, &u_{b=1}''(1)&=-1/8.
\end{aligned}
$$

The tail grid fixes $r_{12}=1$, so it exposes the pair factor's full finite-distance derivative error while moving the center of mass outward. The $b=1$ envelope-only prediction `2.421875` explains 90.8% of the trained mean tail bias `2.464764 - 2`. Learned features create the remaining radius-dependent variation. The correct statement is therefore:

> The cusp and tail graphs are both driven primarily by the same incorrect pair factor. The cusp graph reveals its wrong short-range curvature; the tail graph reveals its wrong derivatives at the fixed finite pair distance $r_{12}=1$.

This tail diagnostic does **not** establish behavior at large electron-electron separation, because `TailGridGenerator` does not vary $r_{12}$.

### What is universal and what is Hooke-specific

The Hooke-exact factor

$$
\log(1+r_{12}/2)
$$

is not a universal electron-electron factor. It is exact only for this two-electron Hooke singlet at $\omega=1/2$. The $b=0.25$ range is also Hooke-specific: it matches the quadratic Taylor coefficient of that exact factor, not a universal Coulomb condition.

The universal all-electron, three-dimensional Coulomb information is the Kato cusp slope:

$$
\begin{aligned}
u_{\uparrow\downarrow}'(0)&=\frac12,\\
u_{\uparrow\uparrow}'(0)&=\frac14,\\
\chi_{eA}'(0)&=-Z_A.
\end{aligned}
$$

The same-spin condition applies to the regular radial coefficient after the Pauli-required $r_{ij}Y_{1m}$ node is factored out; the full same-spin spatial wavefunction vanishes at coalescence. The electron-nucleus condition applies to point, clamped, all-electron nuclei. Pseudopotentials with no Coulomb singularity must not receive an electron-nucleus cusp factor.

The existing `ElectronElectronCusp` already encodes the spin-resolved $1/2$ and $1/4$ slopes. The project has `ElectronNucleusInteraction`, and `ElectronBatch` already carries nuclear positions and charges, but no corresponding output-side electron-nucleus cusp envelope exists yet.

### Minimal general architecture

Adopt a positive, symmetric, non-backflow Coulomb cusp envelope:

$$
J_\mathrm{cusp} =
\sum_{i<j}
\frac{a_{\sigma_i\sigma_j}r_{ij}}{1+b_{\sigma_i\sigma_j}r_{ij}}

\sum_{i,A}
\frac{-Z_A r_{iA}}{1+\beta_A r_{iA}},
$$

with fixed

$$
a_{\uparrow\downarrow}=\frac12,
\qquad
a_{\uparrow\uparrow}=\frac14.
$$

The slopes and nuclear charges are physics, not trainable parameters. The ranges $b$ and $\beta$ only control smooth finite-distance behavior; they may be fixed, species-specific, or learned as a very small number of scalars. They are not cusp conditions.

This is technically a fixed short-range Jastrow factor in QMC terminology, but it is **not Jastrow-backflow**:

- it is positive and symmetric, so it preserves the Pfaffian/determinantal nodes and fermionic symmetry;
- it does not transform electron coordinates;
- it does not add a flexible many-body correlation architecture;
- it is $O(N^2+N N_\mathrm{nuc})$, negligible next to the current Hessian-based kinetic evaluation.

The residual SpENN/Pfaffian component should remain responsible for all nonlocal and many-body correlation. It must be cusp-free after the explicit factor: do not let an unconstrained learned radial linear term alter the fixed slopes. A later optional residual can be made $O(r^3)$ near each coalescence, preserving the value, slope, and curvature of a chosen base factor without introducing backflow.

Do **not** promote the Hooke exact factor, $b=0.25$, or an explicit center-of-mass branch to the generic architecture. The center-of-mass decomposition is a diagnostic of the harmonic trap, not an inductive bias appropriate for general molecules. Generic inputs should remain electron-electron and electron-nucleus local invariants plus the existing equivariant residual.

### Make the local-energy calculation cusp-stable

The exact local energy must contain physical cancellation, but it should not rely on numerical subtraction of separately evaluated divergent tensors. The present raw component dump is useful for diagnostics; it should not be the desired numerical form of the training aggregate near a cusp.

For unlike-spin electron pairs, write the total log amplitude as $u(r)+f$. The singular kinetic-plus-Coulomb piece is

$$
\frac{1-2u'(r)}{r}.
$$

For the rational $u(r)=r/[2(1+br)]$, evaluate it algebraically as

$$
\frac{1-2u'(r)}{r}
=
\frac{b(2+br)}{(1+br)^2},
$$

which stays finite instead of subtracting $1/r$ from $-2u'(r)/r$. At $r=10^{-5}$, the raw terms are approximately `100000` and `-99998`, while this grouped expression is approximately `2`.

For an electron-nucleus factor $u(r)=-Zr/(1+\beta r)$, the corresponding grouped singular piece is

$$
-\frac{Z+u'(r)}{r}
=
-\frac{Z\beta(2+\beta r)}{(1+\beta r)^2}.
$$

This does not remove physical kinetic terms or alter the Hamiltonian; it changes only the algebra used to evaluate the known cancellation. Cross terms with the smooth residual remain finite.

Recommended sequence:

1. Implement the generic, spin-resolved electron-electron and optional all-electron electron-nucleus cusp envelope.
2. Keep the Hooke-exact factor and $b=0.25$ as benchmark-only ablations.
3. Add a typed cusp-factor contract shared by the model and kinetic evaluator, then evaluate singular kinetic-plus-potential pieces in grouped analytic form.
4. Add hydrogenic electron-nucleus, Hooke singlet, and Hooke triplet cusp tests over progressively smaller distances in float64.
5. Consider transcorrelation only later. It is a different non-Hermitian, up-to-three-body solver formulation, not a necessary local-energy patch.

The current exact Hooke test shows that float64 is accurate at $r_{12}=10^{-5}$: the pointwise error remains below about $5.4\times10^{-10}$. That validates the current calculation at the present diagnostic floor, but does not justify relying on raw cancellation at smaller distances, in float32, or at larger system size.

### QMC references

- T. Kato, [On the eigenfunctions of many-particle systems in quantum mechanics](https://doi.org/10.1002/cpa.3160100201), *Communications on Pure and Applied Mathematics* **10** (1957).
- N. D. Drummond, M. D. Towler, and R. J. Needs, [Jastrow correlation factor for atoms, molecules, and solids](https://arxiv.org/pdf/0801.0378), *Physical Review B* **70**, 235119 (2004).
- R. J. Needs et al., [Variational and diffusion quantum Monte Carlo calculations with the CASINO code](https://eprints.lancs.ac.uk/id/eprint/143418/1/casino_jcp.pdf), *Journal of Chemical Physics* **152**, 154106 (2020).
- A. Ma et al., [Scheme for adding electron-nucleus cusps to Gaussian orbitals](https://arxiv.org/pdf/0801.2742), *Journal of Chemical Physics* **122**, 224322 (2005).
- D. Haupt et al., [Optimizing Jastrow factors for the transcorrelated method](https://arxiv.org/pdf/2302.13683), *Journal of Chemical Physics* **158**, 224105 (2023).

### Range policy and numerical-precision clarification

The cusp coefficients are definitive; the rational range parameters are not:

| Parameter | Status |
|---|---|
| Electron-electron $a_{\uparrow\downarrow}=1/2$, $a_{\uparrow\uparrow}=1/4$ | Fixed by the three-dimensional Coulomb cusp condition |
| Electron-nucleus slope $-Z_A$ | Fixed by the point all-electron Coulomb cusp condition |
| Electron-electron $b$ | Finite-range correlation scale; system and representation dependent |
| Electron-nucleus $\beta$ | Finite-range correlation scale; system and representation dependent |

For the next **Hooke-only successor/ablation**, use fixed `b=0.25`. Do not rewrite V3's historical baseline: changing the model envelope invalidates the existing checkpoint configuration hash and would muddle the comparison. The generic architecture should retain a configurable positive range, initialized and optionally trained independently of the fixed cusp slope.

There is no definitive $b$ or $\beta$ determined only by elemental species. In conventional QMC, the cusp slopes are constrained and the remaining range parameters are commonly optimized. For SpENN, train at most a small number of positive shared or species-resolved range scalars, initialize them conservatively, and monitor deterministic cusp diagnostics. Do not let a free range parameter modify the fixed cusp slopes.

High numerical accuracy comes from cancelling analytically before floating-point evaluation, not from hoping that two large tensors subtract accurately. In float64, the spacing near `1e5` is approximately $1.46\times10^{-11}$, which explains the present exact-Hooke success at $r_{12}=10^{-5}$. Near `1e12`, the spacing is approximately $1.22\times10^{-4}$; raw cancellation is then materially less reliable. The grouped cusp expressions above remain order-one at either distance, provided the smooth residual is differentiated separately from the explicit cusp factor.

### Is large cancellation solved in QMC?

It is solved at the **wavefunction and local-energy-formula level**, but not by treating separately evaluated divergent kinetic and potential tensors as harmless.

Standard real-space QMC uses explicit Kato-correct cusp factors or cusp-corrected orbitals so that the leading $1/r$ divergence cancels analytically in the total local energy. This removes divergent local-energy spikes and their variance. Individual kinetic and Coulomb components may still be large and are not separately well-conditioned observables near coalescence.

The present exact Hooke result establishes that the current float64 implementation is accurate at the checked floor $r_{12}=10^{-5}$; the measured V3 energy defect is therefore an envelope/model-curvature error, not evidence of roundoff failure. Nevertheless, SpENN does not yet expose the cusp/residual decomposition needed to evaluate the grouped formulas directly. That remains the correct general robustness improvement before probing much smaller distances or moving to larger all-electron systems.

Compensated summation or a different order of term addition cannot repair cancellation that has already occurred inside independently evaluated kinetic and potential terms. A complete implementation must factor the wavefunction into explicit cusp and smooth residual pieces, differentiate the smooth residual, and add the finite analytic cusp contribution.
