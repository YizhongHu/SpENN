# Pair-Stability V3 Findings

## Scope and evidence

- The controlled diagnostic checkpoint is `B00/U00/F01`, `lr=1e-3`,
  `channels=8`, `SiLU`, replicate 0, loaded from the completed final-training
  checkpoint recorded in
  `.agent-worktrees/pair-stability-v3-diagnostics/experiments/hooke/pair_stability_v3_diagnostics/results/trained_rep-0/metadata.json`.
- The latest V3.1 report uses the completed final collection
  `08_final_collect/20260719T141216-0400` and its repaired report lives in
  `results/09_final_report/20260719T141216-0400`.
- Diagnostic numerical tables are preserved under
  `.agent-worktrees/pair-stability-v3-diagnostics/experiments/hooke/pair_stability_v3_diagnostics/results/analysis_rep-0/`.

## Report completeness defect

`architecture_summary.csv` contained all 24 `basis_update` by feature rows.
`final_report._report_markdown` rendered only `architecture[:20]`; therefore
an omitted family was a presentation defect, not a failed train, evaluation,
or collection. The order-dependent cap could hide any of the final four rows.

The cap is removed in PR [#138](https://github.com/YizhongHu/SpENN/pull/138),
which adds a regression test with all 24 families. The regenerated report now
contains all four `B00+U02` rows:

| family | feature | median absolute energy error | median local-energy variance |
|---|---:|---:|---:|
| B00+U02 | F00 | 0.764123 | 1.367579 |
| B00+U02 | F01 | 0.001974 | 0.004504 |
| B00+U02 | F02 | 0.003530 | 0.006954 |
| B00+U02 | F03 | 0.001696 | 0.004391 |

The repaired report copied 24 architecture-summary rows and its Markdown
family table includes all of them.

## Local-energy diagnostics

### Exact reference

The exact Hooke model is numerically correct on the deterministic probe:

| task | local-energy mean | absolute energy error | local-energy variance |
|---|---:|---:|---:|
| cusp | 1.999999999981 | 1.91e-11 | 5.17e-21 |
| tail | 2.000000000000 | 0 | 4.38e-30 |

This rules out the local-energy evaluator and diagnostic geometry as the
source of the observed model error.

### Fresh versus trained target model

Training materially improves both difficult regimes, but does not make either
pointwise local energy correct:

| variant | cusp mean local energy | cusp variance | tail mean local energy | tail variance | tail outliers/pathologies |
|---|---:|---:|---:|---:|---:|
| fresh | 4.063991 | 0.865375 | 2.907054 | 1371.218716 | 28 / 1 |
| trained replicate 0 | 3.790455 | 0.046960 | 2.464764 | 0.032251 | 0 / 0 |
| exact | 2.000000 | approximately 0 | 2.000000 | approximately 0 | 0 / 0 |

The trained model retains mean local-energy residuals of 1.790455 at the
cusp and 0.464764 in the tail. The global Monte Carlo estimates are much
closer to the eigenvalue: across the nine trained diagnostic replicas they
range from 2.006325 to 2.019707. Thus an accurate expectation value does not
imply accurate controlled cusp or tail behavior.

### Hamiltonian components

At fixed diagnostic geometries, electron-electron and harmonic-trap terms
are exact geometry functions and have zero residual. Every measured
local-energy discrepancy is in the kinetic term:

| variant/task | kinetic mean absolute residual | kinetic q95 | kinetic maximum |
|---|---:|---:|---:|
| fresh cusp | 2.063991 | 3.453091 | 3.742484 |
| trained cusp | 1.790455 | 1.991554 | 2.122759 |
| fresh tail | 2.150855 | 2.312332 | 1417.336988 |
| trained tail | 0.467993 | 0.721706 | 0.944149 |

The first corrective investigation should therefore target the wavefunction
log-amplitude derivatives used by the kinetic term, not the potential terms.

### Known orbital-generator confounder

The theory table compares the current `b=1` pair factor with a matched
`b=0.25` factor near the cusp. At `r12=1e-5`, their local energies are about
4.249953 and 2.000001, respectively; the exact value is 2.0. The fresh and
trained controlled models are correspondingly near 4.138 and 3.879 at that
point. This is consistent with the already-known Hooke-orbital generator
mismatch. It is **not** proof that the mismatch alone explains all observed
model error: a corrected-generator, matched-training ablation is still
required before attributing the cusp or tail residuals to it.

The tail also has a trained log-amplitude residual with fitted slope
0.001075 per radius-squared and intercept -3.043723 over radius 4.01--12.
That is an envelope mismatch, consistent with the same confounder but not a
causal identification.

## Layer traces

The nine trained diagnostic replicas show no numerical trace failure:

- zero feature non-finites;
- zero readout near-zeros across 512 readout records per replica;
- readout condition-number q95 and maximum both equal to 1;
- zero trace-equivariance failures, with maximum absolute error
  1.42e-14.

Activation magnitudes vary substantially across replicas, but the nine-point
Pearson associations with cusp/tail error are weak (`r` from -0.30 to -0.12).
This is too small a sample to infer causality, and it does not support a
numerical trace failure as the dominant error source.

## Cost and apparent warm-up

The complete V3.1 final grid has 216 CUDA final-training runs. Median training
wall time is 668.93 seconds for 500 updates; median evaluation wall time is
118.23 seconds. Median per-update time is 1.310 seconds, attributed as:

| phase | median seconds/update | median fraction of update |
|---|---:|---:|
| local energy | 0.782 | 59.7% |
| sampling | 0.315 | 24.0% |
| forward | 0.057 | 4.3% |
| backward | 0.056 | 4.3% |

Local-energy evaluation plus MCMC sampling consume about 84% of an update.
Optimizing the neural-network forward/backward path alone cannot materially
reduce end-to-end training time.

There is no cohort-wide sampler warm-up in the logged training curves. The
sampler burns in for 500 moves before its first returned batch, then retains a
persistent chain and advances it by 10 moves per update. The trainer logs the
local energy before that update's optimizer step. For all 216 final runs, the
median energy falls from 2.566456 at step 0 to 2.453737 at step 10, 2.182261
at step 50, and 2.007704 at step 490. Of 214 runs that improve from first to
last logged step, the median half-improvement time is 30 updates (90th
percentile: 90; maximum: 300).

For the latest `B00/U00/F01` final winner, the median energy falls from
2.555876 at step 0 to 2.401038 at step 10, 2.041648 at step 50, and 2.002701
at step 490. Its nine half-improvement times are `10, 10, 20, 20, 20, 20, 30,
90, 170`; two seeds have genuinely slower optimization trajectories, but the
family does not exhibit a common warm-up plateau.

Figure 8 is not suitable for inferring exact onset timing: it uses a 25-point
centered rolling mean. Since points are logged every 10 updates, the plotted
point at step 0 averages raw steps 0 through 120. Use raw curves or a causal
trailing smoother for future timing diagnosis.

## Next experiments

1. Correct the Hooke orbital generator, then retrain a matched small,
   controlled ablation. Do not reuse an incompatible old checkpoint as a
   causal test.
2. Compare the corrected and current models on the same deterministic cusp and
   tail profiles, retaining per-term kinetic records.
3. Profile and optimize the local-energy and MCMC paths before changing model
   layers; they are the measured runtime bottleneck.
4. Run a 300-versus-500 update checkpoint ablation with independent final
   evaluation. Training curves suggest a candidate saving, but only the
   independent evaluation can establish whether it preserves the selected
   energy, cusp, and tail contracts.
5. Plot raw training energy with sampling uncertainty beside any smoothed
   curve, and label any smoother as causal or centered.

## Archived lineage

The completed V3.1 lineage was **copied, not moved** from scratch to:

```text
/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu/spenn-studies/hooke/pair_stability_v3_1
```

Archive plan `10_sync/20260719T153016-0400` selected 43,148 result files
(6,809,683,058 planned bytes) from source revision
`c13e070baad7b1db65353a87b8cb0e102755d3fb`. Slurm job `33159786` completed
successfully. The immutable archive was independently verified at
6,809,682,451 bytes, below the 10,000,000,000-byte limit; it contains every
required stage, no checkpoint payloads (2,736 checkpoint directories were
excluded), and no symlinks. The original scratch lineage remains at
`/n/netscratch/kozinsky_lab/Lab/rhu/SpENN/experiments/hooke/pair_stability_v3/results/`.

## Verification

- `uv run --extra cpu pytest -q experiments/hooke/pair_stability_v3/test_pair_stability_v3.py`
  passed: 61 tests.
- Regenerating `final_report.py` against final collection
  `20260719T141216-0400` copied all 24 architecture rows; the repaired report
  contains all four `B00+U02` entries.
