# He-v1 50k training science report

## Scope and decision boundary

This report analyzes the completed `train-seed0000` 50,000-update He-v1 run
(Cannon job `40907434`, source revision `418accf`). He-v1 is the first
all-electron TPEN surface: infinite-mass, nonrelativistic singlet helium. Its
purpose is to validate the trainable all-electron model, typed nuclear context,
cusp-factor wiring, and runtime contracts. It is explicitly a smoke/contract
study, **not** a production comparison or an independent energy result.

The run completed with exit `0:0` in `08:28:25`. It used one A100-80GB on
`kozinsky_gpu`; the allocation receipt recorded that the requested `a100`
stratum matched the delivered device. No comparison-lane artifacts or numbers
are used here.

## Figures

| Figure | Content |
| --- | --- |
| [Energy and loss diagnostics](he_v1_50k_energy_diagnostics.pdf) ([PNG](he_v1_50k_energy_diagnostics.png)) | Post-transient training-batch energy, deviation from the exact helium reference, and loss. |
| [Stability and sampling telemetry](he_v1_50k_stability_sampling.pdf) ([PNG](he_v1_50k_stability_sampling.png)) | Sampler acceptance, radial tail quantities, gradient norm, and logged trainable factor scalars. |
| [Cusp and tail diagnostics](he_v1_50k_cusp_tail_diagnostics.pdf) ([PNG](he_v1_50k_cusp_tail_diagnostics.png)) | Electron--nucleus curvature-law scalars and its implied asymptotic slope. |

Figures use Matplotlib with vector PDF outputs and 300 dpi PNG outputs. Energy,
Monte Carlo uncertainty, sampler, factor, and check telemetry are sampled at
1,000-update cadence; the loss panel uses its 200-update series. The energy and
loss figures suppress only step 0 in the plotted post-transient panels because
its initial transient (`E = -2.0910 Ha`, `loss = 0.6130`) would obscure the
remaining 49,000 updates. It remains included in the reported extrema below.

## Findings

### 1. Optimization behavior

- The initial transient is short. At step 1,000 the training-batch energy is
  `-2.90498 Ha`, within `-1.25 mHa` of the exact reference
  `-2.903724377034119598 Ha`.
- Through approximately 30k updates, the sampled training-batch energy remains
  close to the reference. Late updates show visibly larger batch-to-batch
  excursions: at the sampled 47k and 48k points, the estimate is
  `-2.93186 Ha` and `-2.93332 Ha`; the final 49,999 value is `-2.89727 Ha`.
- The final training-batch deviation is `+6.45 mHa`, with a contemporaneous
  **batch** standard error of `14.25 mHa`. This does not establish an energy
  error bar for a trained wavefunction: batches are neither a fixed-model,
  independently equilibrated chain nor a correlation-aware estimator.
- The 1,000-step loss mean stays near zero after the transient, but late loss
  noise grows. That is evidence of continuing stochastic variation, not a
  convergence certificate. The final loss is `0.000595`.

**Conclusion:** the run is numerically healthy enough to proceed to strict
checkpoint evaluation, but training telemetry alone cannot certify a plateau or
scientific agreement with the reference.

### 2. Symmetry and equivariance

At the final runtime-check cadence (step 49,995):

- `FullModelEquivarianceChecker`: 1 available / 1 tested permutation,
  1 comparison, 0 failed permutations, maximum absolute error `0.0`,
  `atol = rtol = 1e-6`.
- `TraceEquivarianceChecker`: 1 available / 1 tested permutation, 7 trace
  entries / 7 comparisons, 0 missing or extra keys, 0 failed entries, maximum
  absolute error `0.0`, `atol = rtol = 1e-6`.

This is strong evidence that the exercised two-electron permutation contract
held at the runtime-check cadence. It is not an evaluation-stage antisymmetry
measurement under the final restored distribution; that output does not yet
exist.

### 3. Cusp slope, curvature, and outer tail

The configured electron--nucleus law is

$$v_A(r)=-Z_A r+\frac{c r^2}{1+d r}.$$

For helium, the cusp slope at the nucleus is therefore exactly
$\frac{d v_A}{dr}(0^+)=-Z=-2$ for every logged value of `c` and positive `d`.
The trainable curvature term begins only at second order, so optimization cannot
alter that Kato slope.

The logged curvature parameters do move:

- `c` spans `-0.242` to `+0.245` and ends at `+0.244` at the final logged
  factor cadence (49k).
- The positive range `d` spans `0.823` to `2.442` and ends at `0.823`.
- The implied outer-tail slope, $-2+c/d$, stays negative throughout the sampled
  trace: minimum `-2.139`, maximum/final `-1.704`.

Thus the logged parameters remain on the decaying side of the law's
normalizability boundary (`-2+c/d<0`). This is a **parameter-derived** tail
check, not a sampled radial-profile measurement. The evaluation stage that
emits one-sided cusp slopes, outer-tail slopes, and derivative-profile CSVs has
not yet run, so no empirical cusp-profile or curvature claim is justified.

The electron--electron same-spin range remains exactly `1.0`; that is expected
for a one-up/one-down helium singlet, which has no same-spin electron pairs. The
opposite-spin range changes from `0.997` to `0.484`, confirming gradient reach
in the populated channel.

### 4. Local-energy and numerical stability

- Every sampled `local_energy_finite_fraction` is `1.0`; every sampled
  local-energy nonfinite fraction is `0.0`.
- Every sampled gradient nonfinite fraction is `0.0`; the terminal data
  integrity check also passed at step 49,999.
- The final observed local-energy variance is `0.832 Ha^2`; the sampled range is
  `0.0015` to `9.72 Ha^2`. The larger variance and error-bar spikes late in
  training are finite but reinforce that a fixed-model evaluation with an
  appropriate uncertainty estimator is required.
- The gradient norm decays from `122.97` initially into an approximately
  `1e-3`--`1e-2` late range, with intermittent finite excursions. No numerical
  instability or nonfinite event is visible in the recorded telemetry.

### 5. Sampler behavior and radial tails

After the initial update, acceptance remains in a tight `0.294`--`0.308` band
(final `0.3064`). The radius q99 remains `2.662`--`2.848`; the sampled maximum
radius is `4.009`--`6.321`. The final maximum coordinate magnitude is `4.355`.
These bounded, finite summaries do not show runaway walkers over this training
window.

The logged $\log|\psi|$ range shifts materially during training (from
`[-15.93, -5.86]` initially to `[2.54, 12.57]` finally). Absolute log-amplitude
has a normalization/gauge freedom, so this is a finite-value diagnostic only;
it must not be interpreted as a physical tail probability or normalization
result.

### 6. Calculation efficiency

The allocation used 8.474 GPU-hours for 50,000 updates, or **1.639 optimizer
steps/s** wall-clock. `jobstats` reports mean GPU utilization `65.8%` and peak
GPU memory `39.6 / 80 GB` (`49.5%`). CPU utilization was `21.5%` and peak host
memory was `1.4 / 32 GB` (`4%`).

This run was an all-electron capability gate, not an efficiency benchmark. The
resource profile nevertheless identifies a future calibration question: whether
CPU and host-memory requests can be reduced without harming throughput. Do not
pool this timing with another GPU stratum or a different model configuration.

## What remains unproven

1. **Strict restored-checkpoint evaluation.** No `03_eval`, `04_collect`, or
   `05_report` output is present for this run. The final energy conclusion must
   come from strict checkpoint restoration and fixed-model VMC sampling.
2. **Statistically valid uncertainty.** This training trace cannot provide IAT,
   ESS, or correlation-aware MCSE. The study intentionally leaves those absent
   until the fixed-model trajectory producer exists.
3. **Empirical cusp/tail profiles.** The evaluation derivative-profile artifacts
   are absent. The exact Kato slope and negative implied asymptotic slope are
   model-law facts, not measured profiles.
4. **Full physics comparison.** A one-seed training trace is not a production
   grid, multi-seed result, or cross-model comparison.

## Evidence and provenance

- Run status: `completed`, `run_end`, no exception; final step `49,999`.
- Primary training telemetry: `metrics.csv` and `metrics.jsonl` for
  `train-seed0000`, attempt `sequential-50k-quota-rerun-20260821`.
- Runtime contracts: data integrity, gradient, full-model equivariance, and
  trace equivariance metrics in the same attempt.
- Allocation identity: `allocation_receipt.json`; scheduler accounting and
  `jobstats` for `40907434`.
- Model and reference contract: [`../README.md`](../README.md) and
  [`../configs/train.yaml`](../configs/train.yaml).

Cannon access followed Task Orchestrator item
`3541fe33-deb9-4d45-92f3-b39aec3058c5`, notably
`00-read-first-index`, `login-node-boundary-2026-08-06`,
`storage-and-directory-contract-2026-08-12`, and
`gpu-hardware-strata-and-constraint-rule-2026-08-15`. Artifact inspection was
read-only; no training data, checkpoint, or scheduler state was modified.
