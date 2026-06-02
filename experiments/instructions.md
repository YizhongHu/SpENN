# Experiment Tips

## Goals

Experiment tracking should answer four questions:

1. Can this run be reproduced exactly?
2. Is the physics correct?
3. Is the sampler healthy?
4. Which architectural choice caused the improvement or failure?

Do not rely only on training loss. For QMC experiments, energy, local-energy variance, cusp behavior, symmetry/antisymmetry checks, and sampler diagnostics are all first-class metrics.

---

## Run Configuration

Use Hydra as the source of truth for configuration and W&B for run tracking.

Every run should log:

```text
experiment_name
run_id
git_commit
dirty_git_state
hydra_config_yaml
model_config
system_config
hamiltonian_config
sampler_config
optimizer_config
cusp_config
readout_config
feature_mode
fusion_mode
branch_mode
seed
dtype
device
```

Recommended W&B run names:

```text
hooke_singlet_omega0.5_M2_Mv2_seed003
hooke_triplet_omega0.25_M2_Mv2_seed003
```

Run names are only for readability. The full metadata should live in the config.

---

## Tags

Use tags for high-level grouping only.

Good tags:

```text
hooke
singlet
triplet
same_spin
opposite_spin
cusp_A
cusp_B
det_readout
pfaffian_readout
M2
Mv2
real_tuple
trace_irrep
full_fourier
canonical_M
orbit_basis_M
canonical_pool
orbit_basis_pool
```

Avoid putting every hyperparameter into tags. Hyperparameters belong in the config.

---

## Core Metrics

Track at least:

```text
energy/mean
energy/error
energy/abs_error
energy/std
energy/sem
energy/running_mean
local_energy/variance
local_energy/median
local_energy/p05
local_energy/p95
grad/norm
grad/clipped_fraction
optimizer/lr
```

For analytic benchmarks, always log:

```text
energy/exact
energy/error = energy/mean - energy/exact
energy/abs_error
```

For the Hooke singlet benchmark:

```text
energy/exact = 2
```

For the same-spin triplet benchmark:

```text
energy/exact = 5 / 4
```

---

## Sampler Diagnostics

Sampler failures can look like model failures. Track sampler health aggressively.

Log:

```text
sampler/acceptance_rate
sampler/proposal_scale
sampler/autocorr_energy
sampler/effective_sample_size
sampler/mean_r12
sampler/std_r12
sampler/min_r12
sampler/max_r12
sampler/equilibration_steps
```

For MALA, additionally log:

```text
sampler/mala_drift_norm
sampler/mala_log_accept_ratio
sampler/mala_force_norm
```

Useful acceptance-rate target:

```text
0.3 <= acceptance_rate <= 0.7
```

Very low acceptance usually means moves are too large. Very high acceptance usually means moves are too small and mixing is slow.

---

## Cusp Diagnostics

Track cusp correctness explicitly.

For electron-electron cusps:

```text
cusp/ee_target_slope
cusp/ee_measured_slope
cusp/ee_slope_error
cusp/ee_same_slope
cusp/ee_opp_slope
```

For Hooke singlet:

```text
cusp/ee_target_slope = 1 / 2
```

For same-spin triplet:

```text
cusp/ee_target_slope = 1 / 4
```

If electron-nucleus cusps are added later:

```text
cusp/en_target_slope_by_Z
cusp/en_measured_slope_by_Z
cusp/en_slope_error_by_Z
```

The cusp module should be tested before full training. If a learnable residual is added, verify that it does not change the short-range slope.

---

## Symmetry and Antisymmetry Checks

Track exchange behavior.

For singlet spatial symmetry:

```text
symmetry/swap_logabs_error_mean
symmetry/swap_logabs_error_max
```

For triplet antisymmetry:

```text
symmetry/antisym_error_mean
symmetry/antisym_error_max
symmetry/sign_flip_accuracy
```

For the exact singlet spatial benchmark, check:

```text
logabs_psi(r1, r2) == logabs_psi(r2, r1)
```

For the exact triplet and all SpENN singlet/triplet models, check particle
antisymmetry. In the SpENN network, spins are part of the particle features and
permute with the particle positions:

```text
psi((r1, s1), (r2, s2)) + psi((r2, s2), (r1, s1)) ~= 0
```

If the implementation outputs sign and log-amplitude separately, verify both sign behavior and log-amplitude behavior away from exact nodal surfaces.

---

## Standard Plots

Use the same plot names and axes across experiments.

Recommended plots:

```text
plots/energy_trace
plots/local_energy_variance
plots/acceptance_rate
plots/r12_histogram
plots/cusp_slope
plots/wavefunction_radial_cut
plots/exchange_symmetry_error
plots/local_energy_histogram
plots/gradient_norm
```

### Energy Trace

Plot training step versus energy estimate.

Include:

```text
raw batch energy
running mean
exact reference line
Monte Carlo uncertainty band, if available
```

### Local-Energy Variance

Plot local-energy variance versus training step.

Use a log vertical scale if the variance changes by orders of magnitude.

### Cusp Diagnostic

For the singlet benchmark, plot the radial derivative of `log psi` near `r12 = 0`.

For the triplet benchmark, divide out the chosen antisymmetric Cartesian factor before checking the same-spin cusp slope.

### Pair-Distance Histogram

Plot the sampled distribution of `r12`.

This checks whether the sampler is exploring the physically relevant region and whether Coulomb repulsion is represented in the density.

### Wavefunction Radial Cut

For Hooke singlet, compare the learned radial dependence against the analytic curve, up to normalization.

For Hooke triplet, divide out the chosen Cartesian antisymmetric factor before comparison.

---

## Ablation Axes

Compare architectural choices systematically.

Recommended ablation axes:

```text
feature_mode: real_tuple | trace_irrep | full_fourier
fusion_basis: canonical_M | orbit_basis_M
branch_basis: canonical_pool | orbit_basis_pool
readout: determinant | pfaffian | determinant_plus_pfaffian
cusp: none | analytic | analytic_plus_residual
activation: scalar_only | norm_gated | no_activation
```

When comparing one axis, keep the following fixed:

```text
Hamiltonian
sampler
optimizer
training budget
seed list
batch size
cusp setting
readout setting
```

---

## Seeds

Use multiple seeds for any meaningful comparison.

Suggested policy:

```text
debug: 1 seed
development: 3 seeds
claim/comparison: 5-10 seeds
```

Report mean and standard error across seeds.

---

## Debug, Development, and Benchmark Modes

Use explicit run modes:

```yaml
run_mode: debug | dev | benchmark
```

### Debug

Use:

```text
small sample count
frequent logging
extra assertions
short training
```

### Development

Use:

```text
moderate sample count
3 seeds
standard plots
checkpointing
```

### Benchmark

Use:

```text
fixed seed list
fixed training budget
fixed sampler settings
saved artifacts
full metrics
full plots
```

Do not mix debug runs into final comparison dashboards.

---

## Artifacts to Save

For important runs, save:

```text
hydra_config.yaml
overrides.yaml
final_model.pt
best_energy_model.pt
best_variance_model.pt
train_metrics.csv
sampler_metrics.csv
energy_trace.csv
local_energy_histogram.png
r12_histogram.png
cusp_diagnostic_plot.png
wavefunction_radial_cut.png
```

Save at least three checkpoints:

```text
latest.pt
best_energy.pt
best_variance.pt
```

Best energy and best variance are not always the same checkpoint.

---

## Local Output Layout

Recommended local directory structure:

```text
outputs/
  YYYY-MM-DD/
    hooke_singlet/
      run_id/
        .hydra/
          config.yaml
          overrides.yaml
        checkpoints/
          latest.pt
          best_energy.pt
          best_variance.pt
        metrics/
          train_metrics.csv
          sampler_metrics.csv
        plots/
        artifacts/
```

W&B is useful, but local artifacts are still important for reproducibility.

---

## Preflight Tests

Before long benchmark runs, run and log these tests:

```text
test/local_energy_exact_hooke_singlet
test/local_energy_exact_hooke_triplet
test/cusp_slope_same_spin
test/cusp_slope_opposite_spin
test/exchange_symmetry
test/orbit_basis_equivariance
test/fusion_map_equivariance
test/branch_map_equivariance
```

Benchmark mode should either refuse to run if these tests fail or log a prominent warning.

---

## Minimal Tracking Schema

For the first version, track at least:

```text
energy/mean
energy/error
local_energy/variance
sampler/acceptance_rate
sampler/mean_r12
cusp/ee_slope_error
symmetry/swap_error
grad/norm
optimizer/lr
```

Save at least:

```text
config.yaml
best_model.pt
energy_trace.csv
local_energy_histogram
r12_histogram
cusp_diagnostic_plot
```

This is enough to distinguish “the model is learning” from “the physics is correct.”

---

## Current Hooke Setup

The active Hooke benchmark setup lives under:

```text
experiments/hooke/
  analytic.py
  run_exact.py
  run_spenn.py
  configs/singlet.yaml
  configs/triplet.yaml
  configs/singlet_spenn.yaml
  configs/triplet_spenn.yaml
```

Use the exact-wavefunction runs as preflight checks before training learned
SpENN models. The learned Hooke runs train with VMC energy minimization only;
the analytic wavefunction is reserved for exact-energy reference lines and
post-training diagnostics.
The current learned configs use tanh encoder MLP activations, parity-aware
SpechtMP message activations, linear update heads, and residual feature updates.
The Hooke wrappers load templates from `experiments/hooke/configs/*.yaml` and
use OmegaConf interpolation for run ids, output roots, sectors, and repeated
model settings.

```bash
uv run --extra cpu python experiments/hooke/run_exact.py --config singlet
uv run --extra cpu python experiments/hooke/run_exact.py --config triplet
uv run --extra cpu python experiments/hooke/run_spenn.py --config singlet_spenn
uv run --extra cpu python experiments/hooke/run_spenn.py --config triplet_spenn
```

The run script writes reproducible local artifacts to:

```text
outputs/YYYY-MM-DD/hooke_<sector>/<run_id>/
  .hydra/config.yaml
  .hydra/overrides.yaml
  checkpoints/final_model.pt
  metrics/energy_trace.csv
  metrics/train_metrics.csv
  metrics/sampler_metrics.csv
  metrics/comparison_metrics.csv
  plots/local_energy_histogram.csv
  plots/r12_histogram.csv
  plots/cusp_diagnostic_plot.csv
  plots/wavefunction_radial_cut.csv
  artifacts/summary.json
```

The summary JSON records the resolved config, git commit, dirty git state, and
final metrics. The exact triplet implementation uses the physically consistent
full-coordinate Gaussian factor
`exp(-(r1^2 + r2^2) / 8)`, matching the relative-coordinate factor
`exp(-r12^2 / 16)` after fixing the center of mass.

The learned Hooke comparison should use the Pfaffian readout for both SpENN
singlet and triplet configs. SpENN exchange diagnostics move spin labels with
particle positions, so particle antisymmetry is enforced by the readout. The
exact singlet remains a spatial analytic reference for energy and radial-shape
diagnostics rather than a different SpENN exchange contract.
