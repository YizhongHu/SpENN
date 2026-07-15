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
