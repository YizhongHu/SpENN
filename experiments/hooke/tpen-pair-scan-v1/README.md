# TPEN pair-scan V1 study

The staged screening scan over the Hooke omega=0.5 singlet: `basis` x
`activation` x `lr` x `channels`. This study is a **fork** of
`experiments/hooke/pair_stability_v3`, taken so the scan gets a proven ten-stage
pipeline without waiting on the paused toolkit-v4 generalization
(`experiments/toolkit-roadmap.md` V4-1 / V4-3). The duplication is knowingly
accepted debt; see "Convergence debt" below.

`pair_stability_v3` is frozen historical provenance (MIG-TPEN-000 D9) and is
deliberately non-runnable at TPEN HEAD. Read it; do not change it.

## The configs are NOT self-contained. Read this before running anything.

`configs/train.yaml` and `configs/eval.yaml` deliberately omit `choices.basis`.
The four levels of the basis axis live in ONE table at
`experiments/hooke/choices/basis_levels.yaml`, so that "the scan ran the level we
tested" is checkable rather than a claim about two files that were once
identical. `tpen.run.load_config` reads exactly one YAML file with no include or
defaults-list mechanism, so the library has to be merged in before a run.

**`plan.py` owns that merge.** The grid declares it:

```yaml
choice_libraries:
  - path: experiments/hooke/choices/basis_levels.yaml
    provides: choices.basis
```

and the planner then:

1. merges every declared fragment into both base configs;
2. asserts each `provides` path (plus every `choice_validation` `choices_path`)
   resolves to a non-empty mapping, and fails naming the path if not;
3. writes the composed configs as the grid attempt's `train_config.yaml` /
   `validation_config.yaml` snapshots;
4. compiles **every** command against those snapshots, blinded or not, and
   re-verifies the required paths against the files on disk.

So there is no way to reach a run through this pipeline with an un-merged config.
Loading `configs/train.yaml` by hand instead leaves `${choices.basis...}`
dangling — which is the intended failure: loud, and unreachable from the
launcher. `test_scan_fork_contract.py` pins all of this in both directions.

Blinding reslots the library's intra-level self-references
(`in_features: ${tpen.basis_feature_dim:${choices.basis.<level>.basis}}`) as it
rekeys `choices.basis` by slot. Without that, a blinded plan compiles commands
that each die on `InterpolationKeyError` inside a Slurm array task.

## Stages

Same layout as v3:

```text
00_grid -> 01_train -> 02_validation -> 03_collect -> 04_select
        -> 05_final_grid -> 06_final_train -> 07_final_eval
        -> 08_final_collect -> 09_final_report
```

`plan.py` writes `00_grid/latest.json`; every later stage defaults to the latest
previous-stage artifacts and traces provenance back to the source grid. Attempt
ids are timestamps in `America/New_York`. The device selector, Submitit
re-exec, chunking, claiming, and `--wait-job` dependency mechanics are unchanged
from v3 — see `experiments/hooke/pair_stability_v3/README.md` for that reference
material, which still describes this launcher accurately.

## Grids

| grid | expansion | final |
|---|---|---|
| `configs/grid.yaml` | 4 basis x (2 lr x 3 channels x 4 activations) x 3 seed rows = **288** jobs, 96 configs | `final_replicates: 9` -> 36 final train + 36 final eval |
| `configs/smoke.yaml` | 4 basis x (1 lr x 1 channels x 2 activations) x 3 seed rows = **24** jobs, 8 configs | `final_replicates: 1` -> 4 final train + 4 final eval |

The smoke is the same schema, the same stage stack and the same selector; it
reduces the grid size and the train step budget, and nothing else. It keeps all
four basis levels, the `Gaussian` activation, and all three seed rows, because
those are the least-proven surfaces and because split-sample selection needs both
selection seeds and the holdout seed to exist. There is no smoke launcher mode —
every stage runs its normal script with `--grid .../smoke.yaml`.

`max_steps` and `n_walkers` for the FULL grid are owned by the timing probe, not
by `grid.yaml`; the values in `configs/train.yaml` are a carried-over budget, not
a chosen one.

`test_scan_grid.py` covers both grids; `test_scan_fork_contract.py` and
`test_scan_study.py` still build their own grids in `tmp_path`.

## Selection metric

PRIMARY is `eval/mcmc_energy/local_energy_mean`. `MCMCGenerator` draws from the
trained sampler, i.e. from |psi|^2, so its mean local energy IS
`<psi|H|psi>/<psi|psi>`, and the variational principle bounds it from below by
the exact 2.0 Ha of the Hooke omega=0.5 singlet. `mode: min` is therefore
correct, and an energy *below* 2.0 is an estimator defect rather than a better
wavefunction — `final_report.py` counts those explicitly.

SECONDARY is `eval/stratified_geometry/local_energy_variance`. The *mean* on that
fixed geometry prior is `E_q[E_L]` for an arbitrary q: not variational, unbounded
below, and equal to 2.0 for ANY q at the exact eigenstate, so it pins the optimum
but supplies no direction of approach. The variance is valid on the same prior,
because `Var_q[E_L] = 0` iff psi is an eigenstate.

TIEBREAK is `eval/mcmc_energy/energy_abs_error`, |mean - 2.0| against the exact
energy, emitted by `ReferenceEnergySummary` on the same task. It is also the
deterministic fallback when every ladder rung leaves overlapping seed error bars.
Wall time is never a fallback: it is machine- and load-dependent, so it makes
champion identity irreproducible across independent selections of the same data.

The rest of the suite — `cusp`, `tail`, `hooke_orbital`,
`full_model_antisymmetry`, `trace_equivariance` — is diagnostics and invariants.
Invariants are aggregated by worst case across seeds, never averaged.

## Split-sample champion selection

Champions are selected on seed rows **{0, 1}** and measured on the held-out row
**{2}** (`champions[].selection_seeds` / `holdout_seeds`). `min` over 24 noisy
configs is a winner's curse: with 3-seed means the argmin is biased low by
~1.16 sigma, and because a cross-basis comparison is a *difference* of biases, an
arm with 30% more run-to-run noise wins by ~0.35 sigma of pure artifact — landing
directly on the `axiswise_v1` keep/drop verdict. The 9 final replicates fix the
champion's reported energy and never revisit its identity, so they do not fix
this; a split does.

`champions.csv` therefore carries `holdout_*` columns (the champion's own metric
re-read on the row that had no vote), and `selection_report.json` carries
`bucket_distributions`: each basis level's median, quartiles and best-k over its
own configs. If the basis ranking flips between champion and median, the champion
ranking was noise.

The overlapping-error-bar rung is the engine's inherited 1-stderr non-overlap
rule, and it is deliberately **not** tuned: with two seeds in the selection sample
the spread it rests on has 1 degree of freedom, so it fires near-randomly on close
calls. Calibrating it needs the float64 GPU-drift floor, which the timing/drift
probe owns.

## What was ported, reduced, dropped

| disposition | files |
|---|---|
| as-is | `plan.py`*, `train.py`, `validate.py`, `collect.py`, `select_champions.py`, `final_plan.py`, `final_train.py`, `final_eval.py`, `launch.py`, `stats.py`, `utils/{seeds,naming,layout,overrides,io,time}.py`, `test_overrides.py` |
| reduced | `final_collect.py` (v3-only metric projections stripped), `plot.py` (990 -> 578 lines), `utils/ancestry.py` (433 -> 211 lines), `utils/config.py` (+ the merge contract), `test_scan_study.py` |
| new | `final_report.py`, `test_scan_fork_contract.py` |
| dropped | `parity.py`, `sync.py`, `configs/pilot.yaml`, `configs/pilot_smoke.yaml`, `test_pair_stability_v3_parity.py`, `test_sync.py`, v3's `final_report.py` |

\* `plan.py` is as-is apart from the choice-library merge and the blinding
reslot fix described above.

Metric retargets: v3's `energy` eval task is `mcmc_energy` here;
`feature_trace_stability`, `readout_trace_stability`,
`spatial_exchange_symmetry`, and `rotation_consistency` do not exist in the TPEN
stack and are no longer projected. v3's derived report columns `basis_class` /
`normalization` / `basis_update` are `report_row` / `report_col`, because this
study has no normalization axis and those headers would have carried activation
values.

## Convergence debt

Everything in the "as-is" and "reduced" rows above is duplicated from
`pair_stability_v3` rather than shared. The staged-grid machinery (`plan.py`,
`launch.py`, the stage launchers, `utils/*`) is the intended subject of
`experiments/toolkit-roadmap.md` **V4-1** (extract the staged-grid driver) and
**V4-3** (extract the collect/report reduction); the study-agnostic parts of
`plot.py` and `stats.py` belong to **V4-5**. Until one of those lands, a fix to
shared machinery must be applied here *and* judged against v3's frozen copy by
hand. The choice-library merge in `utils/config.py` is new surface that did not
exist in v3 and is the first thing V4-1 should absorb.

## Tests

Tests live next to the code, as `experiments/README.md` requires:

- `test_scan_fork_contract.py` — the `experiments/` import rule, the
  choice-library merge in both directions, and the port/drop disposition.
- `test_scan_study.py` — the ported staged-pipeline tests, retargeted.
- `test_scan_grid.py` — the checked-in grids: expansion counts, blinding
  round-trip, every override path against the base configs, seed disjointness,
  plan-time choice validation, and champion selection (including that the holdout
  seed cannot influence the champion it measures).
- `test_overrides.py` — `utils/overrides.py`, ported as-is.

One property lives outside this directory because it needs the production
callback: `tests/unit/experiments/test_scan_smoke_workload.py` asserts the smoke's
terminal checkpoint is actually written.

Nothing under this directory imports `tpen`; the sanctioned
`tpen.run.run_from_config` exception is not currently needed, because the stage
launchers shell out to the repo-root `run.py` entrypoint instead.
