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

Correlation-aware external tau, ESS, and MCSE statistics remain explicitly
unavailable/absent: no fixed-model trajectory JSONL producer exists yet, so
this study does not estimate them or wire a callback that could imply they do.

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

Gating is delegated in full to `gates.py`. Until the tolerances are predeclared
in H-F1 every value gate reports `absent` with its observed value retained,
which is the honest state of an ungated run; a required availability flag that
is false still fails.

`train.py` and `eval.py` are thin drivers over `tpen.run.run_from_config`, the
one `tpen` symbol `experiments/README.md` permits here. Tests live beside the
code and run from the directory form, e.g. `python -m pytest -v
experiments/atomistic/he-v1`.
