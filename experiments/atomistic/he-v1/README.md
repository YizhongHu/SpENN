# TPEN He-v1

This development study is the first all-electron TPEN surface: infinite-mass
helium in the nonrelativistic singlet ground state. It is a smoke/contract
study, not a production comparison.

The model composes two generic post-readout `LogAmplitudeFactor`s directly:
`ElectronElectronCusp` and `ElectronNucleusCusp`. `ElectronNucleusCusp` is
constructed from a declarative `atoms:` `AtomicConfiguration` (the sole
nuclear-geometry authority, also consumed by the Hamiltonian's
`ElectronNucleusPotential`/`NucleusNucleusPotential` terms) and its default
linear cusp law reproduces the fixed electron-nucleus Kato factor
`-Z_A |r_i-R_A|`. It intentionally has no Gaussian confinement.

The reference is -2.903724377034119598 Ha, mirrored from the canonical
`he_atom` record in `experiments/baselines/systems.yaml`. Train and eval must
agree exactly on model wiring; eval restores with `strict: true`.

Evaluation records the variational MCMC energy, finite-sample standard error,
and scalar reference comparison together with sampled fermionic
antisymmetry/equivariance contracts. A He-owned positive radial grid now
reports one-sided electron-nucleus cusp slopes, explicit outer-tail slopes,
and a CSV derivative profile while carrying the same nuclear context as the
sampler. Spatial coordinate exchange is checked separately from full label
antisymmetry.

Correlation-aware tau, ESS, and MCSE statistics are produced from the retained
fixed-model draw x walker trajectory. They remain distinct from the snapshot
IID standard error and carry the checkpoint/config/evaluator join identity.

## Study driver

The production study runs as a numbered stage stack. Each stage writes a new
attempt directory and never rewrites or deletes an old one.

| stage | script | writes |
|---|---|---|
| `00_plan` | `plan.py` | the ordered row manifest and its plan hash |
| `01_launch` | `launch.py` | one sbatch script and submission record per row |
| `02_train` / `03_eval` | `train.py` / `eval.py` | per-row run artifacts and allocation receipts |
| `04_collect` | `collect.py` | one identity-keyed table plus gate outcomes |
| `05_report` | `report.py` | the rendered receipt |

`plan.py` expands a grid config into seed x checkpoint x evaluation-chain rows
with stable ids. Every grid key is required and every unknown key is rejected:
there are no implicit defaults, because a silently defaulted seed count or wall
time produces a green run answering a different question. The grid config is
supplied on the command line; the production values are predeclared by H-F1 and
the configs under `configs/` belong to the atom lane.

```bash
python plan.py --grid-config <grid.yaml> --results-root <results>
python launch.py --results-root <results> --repo-root <checkout> \
    --uv-bin <absolute-uv> --uv-extra <torch-extra> \
    --uv-project-environment <env> --uv-cache-root <cache-parent>   # add --submit to send it
python collect.py --results-root <results>
python report.py --results-root <results>
```

Three properties are enforced rather than trusted:

- **GPU stratum pinning.** Every GPU row is submitted with `--constraint=h200`
  or `--constraint=a100`, records the constraint it asked for, and asserts the
  delivered card from inside the allocation. `seas_gpu` mixes H200 and
  A100-80GB nodes, so a delivered/requested mismatch fails the row outright.
- **No restart or resume.** Rows carry `--no-requeue` and their wall time is
  checked against the partition's measured ceiling at plan time. A row is sized
  to finish or it fails.
- **Absence is not zero.** A missing value renders as `absent` in JSON, CSV and
  the report, never as `0.0` and never as a blank cell, and every aggregate
  carries how many rows actually supplied a value.
- **A metric name is not a metric.** Two evaluation tasks can share a summary
  class and therefore a metric name. `full_model_antisymmetry` and
  `spatial_exchange_symmetry` both use `TransformConsistencySummary`, so four
  names exist twice per evaluation row. A request may name its namespace, and an
  unqualified request for a colliding name fails and names the namespaces rather
  than being resolved by guesswork.

### Naming one task's metric

```bash
python collect.py --results-root <results> \
    --metric-key eval/spatial_exchange_symmetry.triplet_fraction_mean_under_psi_orig_sq \
    --metric-namespace triplet_fraction_mean_under_psi_orig_sq=eval/spatial_exchange_symmetry
```

`--metric-key <namespace>.<key>` retains one task's value under its own column;
`--metric-namespace <key>=<namespace>` (or a `metric_namespaces:` block in the
gate spec) binds a bare name to one namespace for its column and for the gates
alike, so a tolerance decides on that task's number and no other. A qualified
name that no row's namespace matches is an error, not an absent column.

This matters physically: under `eval/full_model_antisymmetry` the triplet
fraction is identically `1.0` by construction, because a full label exchange
sends `Psi -> -Psi`, so `u = 0`, the sign ratio is `-1` and
`f = (1 - s*sech(u))/2 = 1`. That is the healthy value for a correctly
antisymmetric wave function, not contamination. Singlet purity is interpretable
only under `eval/spatial_exchange_symmetry`, so a purity tolerance must name
that namespace.

Gating is delegated in full to `gates.py`. Until the tolerances are predeclared
in H-F1 every value gate reports `absent` with its observed value retained,
which is the honest state of an ungated run; a required availability flag that
is false still fails.

`train.py` and `eval.py` are thin drivers over `tpen.run.run_from_config`, the
one `tpen` symbol `experiments/README.md` permits here. Tests live beside the
code and run from the directory form, e.g. `python -m pytest -v
experiments/atomistic/he-v1`.

## Frozen post-hoc diagnostic-v1 study

`he-v1-diagnostic-v1` is an independent fixed-model study over the retained
25k and 50k checkpoint bytes. It never edits `production_grid.yaml`, writes
into the completed production run, retrains, resumes, or chooses a checkpoint.
Both checkpoints are always planned and collected, with no preferred-result
field.

The committed `configs/diagnostic_grid.yaml` freezes, per checkpoint:

- four 256-draw x 4096-walker chains at seeds 1000--1003;
- four 1024-draw x 1024-walker chains at seeds 2000--2003;
- separately labeled burn-in 50/200 and stride 10/40 sensitivity arms;
- a seven-arm common-configuration factor response and seven smaller,
  independently re-equilibrated factor chains; and
- one record-producing geometry/atlas/symmetry/equivariance/trace suite.

Energy rows retain the complete draw x walker grid, aligned Hamiltonian terms,
signed log-amplitude, geometry, conditioned/pathology records, autocorrelation
statistics, MCSE, sampler health, and cost/resource metrics. Factor overrides
are temporary physical-parameter transforms; every component restores and
checks all model parameter bytes before returning.

Checkpoint paths are facility state, not repository configuration. Supply them
through an external source map whose labels exactly match the committed grid:

```yaml
schema: he-v1-diagnostic-sources/v1
checkpoints:
  step_025000: <complete-checkpoint-directory>
  step_050000: <complete-checkpoint-directory>
```

Planning hashes `model.pt`, `manifest.json`, and `COMPLETE`, validates the
real-format manifest and resolved config, and writes a new immutable attempt.
The evaluator SHA is the full published commit that Cannon must execute:

```bash
uv run python experiments/atomistic/he-v1/diagnostic_plan.py \
  --grid experiments/atomistic/he-v1/configs/diagnostic_grid.yaml \
  --sources <external-sources.yaml> --results-root <diagnostic-results> \
  --attempt-id <plan-attempt> --scale production \
  --evaluation-git-sha <full-published-sha>
```

The launcher is dry-run by default. Review its scripts, then repeat with
`--submit`. It requires an absolute Cannon `uv` path and gives each job a
separate project environment and cache:

```bash
uv run python experiments/atomistic/he-v1/diagnostic_launch.py \
  --results-root <diagnostic-results> --repo-root <exact-sha-checkout> \
  --plan-attempt-id <plan-attempt> --launch-attempt-id <launch-attempt> \
  --uv-bin <absolute-uv> --uv-extra cu128 \
  --uv-environment-root <netscratch-env-root> \
  --uv-cache-root <netscratch-cache-root> --submit
```

Collection refuses missing/failed tasks, incomplete raw grids, mismatched
checkpoint/config identities, escaped or missing artifacts, wrong hardware,
missing cost streams, or any change to `production_grid.yaml`:

```bash
uv run python experiments/atomistic/he-v1/diagnostic_collect.py \
  --results-root <diagnostic-results> --plan-attempt-id <plan-attempt> \
  --launch-attempt-id <launch-attempt> --collect-attempt-id <collect-attempt>
```

Smoke uses `--scale smoke`. That changes only the declared walker/draw/burn-in/
stride/sample/atlas scale coordinates and Cannon resource target; row ids,
checkpoint identities, factor arms, task profiles, driver, launcher, and
collector contracts are identical to production.
