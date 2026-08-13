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

`configs/grid.yaml` is owned by a later slice. The planner reads any grid that
follows the schema above; `test_scan_fork_contract.py` and `test_scan_study.py`
build their grids in `tmp_path`.

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

The rest of the suite — `cusp`, `tail`, `hooke_orbital`,
`full_model_antisymmetry`, `trace_equivariance` — is diagnostics and invariants.
Invariants are aggregated by worst case across seeds, never averaged.

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
- `test_overrides.py` — `utils/overrides.py`, ported as-is.

Nothing under this directory imports `tpen`; the sanctioned
`tpen.run.run_from_config` exception is not currently needed, because the stage
launchers shell out to the repo-root `run.py` entrypoint instead.
