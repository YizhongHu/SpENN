# Baseline reproduction report — emitted records vs published values

Task Orchestrator: item `9432b86c-b9b4-4520-93f5-881741bda7b6`
(baseline half of `476ffe33`), program root `19fcf084`, project `58348558`.
Written 2026-08-25 (`America/New_York`). Read-and-analyse only: no cluster
compute, no scheduler action, no record or log created, modified or removed.

**Scope.** This report compares emitted **baseline** records against transcribed
**published** values and against each other. It contains no TPEN result, no TPEN
column, and no join to any TPEN run output. Whether such a join is permitted
under the program's results firewall is an operator ruling, not this report's.

## 1. Inputs, and how records were selected

| input | value |
|---|---|
| run root | `/n/netscratch/kozinsky_lab/Lab/rhu/nnqmc-baselines-20260812/runs` (FASRC Cannon, reached with `ssh cannon`; login `boslogin07`) |
| record files found | 42 (`find -name 'baseline_record*.json'`, full depth) |
| canonical records analysed | 33 (`baseline_record.json`) |
| dated variant records excluded from the main tables | 9 |
| reference energies | `experiments/baselines/systems.yaml` at `origin/dev` = `08390aef3b81fbdba402e326ec899be39121a83d` (re-derived 2026-08-25 17:21 EDT) |
| published values | `NNQMC-REFERENCE-ENERGIES.md`, TARGET 1 (FermiNet arXiv:1909.02487, Tables 1 and 2) |
| thresholds | `NNQMC-BASELINES.md` section 3 axis 3: 1.6 mHa chemical accuracy, 0.38 mHa = 1 kJ/mol |

**33, not 15.** The program's own running count of "canonical baseline records"
is 15. That figure is a search-depth artefact: `find -maxdepth 2` sees 15,
because 18 further canonical records live one directory deeper
(`dqmc-he-39358341/<ansatz>/`, `psi-2e-*/`, `psi-atoms-*/`, `r2-full-*/`,
`r2-eval-*/`). Those 18 include every `be_atom` record, all six `psiformer`
records, both `inference`-estimator records, and the only matched-budget
FermiNet Li and Be rows — that is, the rows most relevant to a reproduction
verdict were the ones the shallow count missed. All 33 appear below.

**Selection is by adapter provenance, never by filename.** Two directories keep
four newer dated variants beside a 2026-08-20 canonical file, because the emit
refuses to overwrite `baseline_record.json` (exit 46). Measured, rather than
assumed: for both `dqmc-deeperwin-n2-39915645` and `dqmc-lapnet-lih-39639503`
the canonical file's `energy_hartree` and `energy_stderr_hartree` are equal to
those of all three 2026-08-22 re-emissions
(`dev7d8391a.job41158800`, `logpath.job41159136`, `logpath.job41260278`), so for
these two rows the stale-filename defect changed no energy. The one variant that
does differ is `baseline_record.tailfrac0.10.job40774671`, a deliberate
alternative estimator window, not a correction. The canonical file is used for
Tables A–D and the emit job that actually wrote it is named in Table B.

## 2. Two axes, kept apart

A record can be compared two different ways and they do not mean the same thing.

* **Accuracy against the reference energy** (Table A) — how close the run is to
  the exact or estimated-exact value for that system. Available for all 33 rows.
* **Reproduction of the published number for the same method** (Table C) — the
  actual "did we reproduce it" question. Available for **9 of 33 rows**. The
  other 24 are present with a stated reason.

The estimator differs between the two. FermiNet's published energies come from a
separate post-training evaluation ("After network optimization, we run O(10^5)
MCMC steps and calculate the mean local energy every 10 steps"), which the
records call `estimator: inference`. 30 of 33 records are `training_tail`
instead, an average over the last fraction of the optimization trajectory, which
sits *high* by construction. Only 3 records are `inference`
(`r2-eval-39145226/{Li,Be}`, `orbformer-he-eval-41158747`), and only the first
two of those have a published counterpart. Table C marks the mismatch per row.

## 3. Provenance caveat, measured

The records were not all produced by the same estimator code. Grouping the 33
canonical records by the `experiments/baselines/statistics.py` blob inside the
adapter package their emit job loaded:

| generation | `statistics.py` blob | first landed on `dev` | records |
|---|---|---|---|
| pre-module | file did not exist | — | 14 |
| 1 | `bdbe5bf447901af8ddc977129a26c9149438ea79` | `f84b136`, 2026-08-18 | 10 |
| 2 | `fb0cec1ae1afc4795a1ba7a18c84b9481f0a226d` | `e139a10`, 2026-08-20 | 3 |
| 3 | `5d4fa8eb38878bd2ccb22e3230c7ad26b63d3808` | `bec0219` (#291), 2026-08-21 | 1 |
| 4 | `017e201e5a84c0c6451d4c7ad0b3f4c69830f9ba` | `fdb6127`, 2026-08-21 (still the blob at `origin/dev`) | 5 |

Three consequences, in descending order of how much they should worry a reader.

1. **14 of 33 records have unquantified estimator provenance.** They were written
   on 2026-08-16 by emit jobs `39511430`, `39512025` and `39541844`. None of
   those three logs prints an adapter package, a TPEN tree SHA, or a clean-tree
   assertion — `39196832` prints only the *upstream FermiNet* commit
   `c4312c315dda1c5728994ba89629744f71c6eb66`, which says nothing about the
   adapter that computed the mean and the error bar. And `statistics.py` did not
   exist in the repository until 2026-08-18, two days later, so the code that
   produced those numbers cannot be recovered from any artefact reachable from
   the record. This is the largest provenance hole in the corpus and it lands on
   exactly the rows Table C most depends on.
2. **The frozen DeepQMC path used generation 1, which lacks the tail-step
   floor.** 10 records were emitted from `adapter-pkg-dev-985223bc`, whose
   `statistics.py` has no `MIN_TAIL_STEPS`; the absolute-steps floor arrived in
   generation 2 (`e139a10`, "floor the estimator window by absolute steps, not
   fraction alone"). Those 10 records' tail windows are pure fractions — visible
   in their own notes as "the last 50% of steps" and "the last 25% of steps",
   against "the last 18889 of 188894 steps" for a generation-4 record.
3. **Any cross-code difference read across generation boundaries is partly an
   estimator artefact.** Table D's He column spans generations 1, 2, 4 and the
   pre-module group. Its 0.0564 mHa span is therefore an upper bound on
   ansatz-driven spread, not a measurement of it.

## 4. Reference-value caveats that bound what a verdict can mean

* **The published digits were never read off a publisher PDF.** APS returns
  HTTP 403 to automated fetches. Every FermiNet number here rests on ar5iv HTML,
  cross-checked three ways (HF/CBS column identity; exact agreement with
  arXiv:1012.0709's own VMC/DMC cells; the atomic exact column independently
  reproduced from arXiv:1011.4343). Narrowed, not removed: see
  `caveats.ferminet_table_ar5iv` in `systems.yaml`.
* **Be, B, N references are estimated-exact, not exact.** Chakravorty et al.
  (1993) values are nonrelativistic totals inferred from experimental ionization
  potentials, carrying roughly 0.1 mHa uncertainty. A sub-0.1 mHa agreement
  claim against them is not meaningful, which is why Table A's `n_atom` rows sit
  "below the reference" without that being a variational violation.
* **`li_atom`'s reference is Puchalski and Pachucki (2006)**, not Chakravorty;
  FermiNet's own footnote attributes the whole column to Chakravorty and is
  imprecise for Li alone.
* **`he_atom` is the only genuinely exact molecular-scale reference in play**
  (-2.903724377034119598 Ha, Aznabaev–Bekbaev–Korobov Table 3 attributing
  Schwartz 2006), so it is the only system where "below the reference" is
  informative rather than absorbed by reference uncertainty.
* **`h2_molecule`'s reference carries only five decimals** (-1.17447), i.e. a
  0.01 mHa quantization, and no record states its own bond length, so the
  geometry offset between R = 1.4 bohr and FermiNet's built-in 1.393 bohr
  default (order 0.01 mHa, systematic) is unverified per row. Both are far
  inside the 0.38 mHa threshold, so the H2 verdicts survive them; a tighter
  claim would not.

## 5. Per-row tables

### Table A — accuracy against the registered reference energy

| system | code | ansatz | run | n | estimator | E (Ha) | se (mHa) | dE (mHa) | <1.6 mHa | <0.38 mHa |
|---|---|---|---|---|---|---|---|---|---|---|
| b_atom | ferminet | ferminet | `fn-b-1e6-41159035` | 1000000 | training_tail | -24.653865776 | 0.0059 | +0.0442 | PASS | PASS |
| b_atom | ferminet | ferminet | `psi-atoms-39379501/B` | 200000 | training_tail | -24.653843589 | 0.0148 | +0.0664 | PASS | PASS |
| b_atom | ferminet | psiformer | `psi-atoms-39379500/B` | 200000 | training_tail | -24.653751929 | 0.0183 | +0.1581 | PASS | PASS |
| be_atom | ferminet | ferminet | `r2-eval-39145226/Be` | 100000 | inference | -14.667313823 | 0.0052 | +0.0462 | PASS | PASS |
| be_atom | ferminet | ferminet | `r2-full-38985858/Be` | 200000 | training_tail | -14.667322702 | 0.0100 | +0.0373 | PASS | PASS |
| be_atom | ferminet | psiformer | `psi-atoms-39233775/Be` | 200000 | training_tail | -14.667317960 | 0.0115 | +0.0420 | PASS | PASS |
| h2_molecule | deepqmc | default | `dqmc-paulin2-h2-fix-39520283` | 200000 | training_tail | -1.174470104 | 0.0029 | -0.0001 | PASS | PASS |
| h2_molecule | ferminet | ferminet | `r2-h2-39068280` | 200000 | training_tail | -1.174474339 | 0.0009 | -0.0043 | PASS | PASS |
| h2_molecule | ferminet | psiformer | `psi-2e-39195983/H2` | 200000 | training_tail | -1.174469850 | 0.0015 | +0.0001 | PASS | PASS |
| he_atom | deepqmc | deeperwin | `dqmc-he-39358341/deeperwin` | 20000 | training_tail | -2.903722518 | 0.0048 | +0.0019 | PASS | PASS |
| he_atom | deepqmc | deeperwin | `dqmc-he-deeperwin-200k-41503515` | 188894 | training_tail | -2.903723506 | 0.0020 | +0.0009 | PASS | PASS |
| he_atom | deepqmc | default | `dqmc-he-39358341/default` | 20000 | training_tail | -2.903686363 | 0.0160 | +0.0380 | PASS | PASS |
| he_atom | deepqmc | default | `dqmc-he-default-100k-39411090` | 100000 | training_tail | -2.903709090 | 0.0072 | +0.0153 | PASS | PASS |
| he_atom | deepqmc | default | `dqmc-he-default-200k-39432921` | 200000 | training_tail | -2.903716541 | 0.0080 | +0.0078 | PASS | PASS |
| he_atom | deepqmc | ferminet | `dqmc-he-39358341/ferminet` | 20000 | training_tail | -2.903718433 | 0.0038 | +0.0059 | PASS | PASS |
| he_atom | deepqmc | lapnet | `dqmc-he-39358341/lapnet` | 20000 | training_tail | -2.903720250 | 0.0032 | +0.0041 | PASS | PASS |
| he_atom | deepqmc | lapnet | `dqmc-he-lapnet-200k-41503516` | 100204 | training_tail | -2.903730759 | 0.0069 | -0.0064 | PASS | PASS |
| he_atom | deepqmc | lapnet | `dqmc-he-lapnet-parity-41694421` | 200000 | training_tail | -2.903720271 | 0.0013 | +0.0041 | PASS | PASS |
| he_atom | deepqmc | psiformer | `dqmc-he-39358341/psiformer` | 20000 | training_tail | -2.903723220 | 0.0032 | +0.0012 | PASS | PASS |
| he_atom | deepqmc | transpsiformer | `dqmc-he-39358341/transpsiformer` | 20000 | training_tail | -2.903721803 | 0.0023 | +0.0026 | PASS | PASS |
| he_atom | ferminet | ferminet | `r2-he-39074394` | 200000 | training_tail | -2.903717803 | 0.0027 | +0.0066 | PASS | PASS |
| he_atom | ferminet | psiformer | `psi-2e-39195983/He` | 200000 | training_tail | -2.903727149 | 0.0027 | -0.0028 | PASS | PASS |
| he_atom | oneqmc | orbformer-se | `orbformer-he-eval-41158747` | 40000 | inference | -2.903742715 | 0.0115 | -0.0183 | PASS | PASS |
| li_atom | deepqmc | ferminet | `dqmc-ferminet-li-39571924` | 192031 | training_tail | -7.478049481 | 0.0082 | +0.0108 | PASS | PASS |
| li_atom | ferminet | ferminet | `fn-li-1e6-40706320` | 1000000 | training_tail | -7.477971084 | 0.0039 | +0.0892 | PASS | PASS |
| li_atom | ferminet | ferminet | `r2-eval-39145226/Li` | 100000 | inference | -7.477910461 | 0.0049 | +0.1499 | PASS | PASS |
| li_atom | ferminet | ferminet | `r2-full-38985858/Li` | 200000 | training_tail | -7.477892105 | 0.0094 | +0.1682 | PASS | PASS |
| li_atom | ferminet | psiformer | `psi-atoms-39233775/Li` | 200000 | training_tail | -7.478035439 | 0.0053 | +0.0249 | PASS | PASS |
| lih_molecule | deepqmc | lapnet | `dqmc-lapnet-lih-39639503` | 104470 | training_tail | -8.070316950 | 0.0194 | +0.2310 | n/a | n/a |
| n2_molecule | deepqmc | deeperwin | `dqmc-deeperwin-n2-39915645` | 100000 | training_tail | -109.540602959 | 0.1163 | +1.6970 | FAIL | FAIL |
| n_atom | ferminet | ferminet | `fn-n-1e6-41159036` | 1000000 | training_tail | -54.589288904 | 0.0098 | -0.0889 | PASS | PASS |
| n_atom | ferminet | ferminet | `psi-atoms-39379501/N` | 200000 | training_tail | -54.589245788 | 0.0262 | -0.0458 | PASS | PASS |
| n_atom | ferminet | psiformer | `psi-atoms-39379500/N` | 200000 | training_tail | -54.588975019 | 0.0229 | +0.2250 | PASS | PASS |

### Table B — record provenance (which estimator code produced each row)

| run | emit job | adapter package | `statistics.py` blob | dev anchor | record sha256 (12) |
|---|---|---|---|---|---|
| `fn-b-1e6-41159035` | 41269375 | adapter-pkg-dev-af84592-emit-41269375 | `017e201e5a84c0c6…` | fdb6127 | `027a6cfada66` |
| `psi-atoms-39379501/B` | 39511430 | none printed | `absent (module a…` | unpinned | `b431609a6d4e` |
| `psi-atoms-39379500/B` | 39511430 | none printed | `absent (module a…` | unpinned | `ebbc0c3af407` |
| `r2-eval-39145226/Be` | 39512025 | none printed | `absent (module a…` | unpinned | `887d90fb1257` |
| `r2-full-38985858/Be` | 39511430 | none printed | `absent (module a…` | unpinned | `f253705dc0c9` |
| `psi-atoms-39233775/Be` | 39511430 | none printed | `absent (module a…` | unpinned | `2308e5740a86` |
| `dqmc-paulin2-h2-fix-39520283` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `04567d716a7b` |
| `r2-h2-39068280` | 39511430 | none printed | `absent (module a…` | unpinned | `851f2550768d` |
| `psi-2e-39195983/H2` | 39511430 | none printed | `absent (module a…` | unpinned | `86a8fb14d8bc` |
| `dqmc-he-39358341/deeperwin` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `31e8b8157db5` |
| `dqmc-he-deeperwin-200k-41503515` | 41617763 | adapter-pkg-dev-f6b16d1-emit-41617763 | `017e201e5a84c0c6…` | fdb6127 | `e2b94f4ef152` |
| `dqmc-he-39358341/default` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `ebf5836b9c74` |
| `dqmc-he-default-100k-39411090` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `76c6af87a5ff` |
| `dqmc-he-default-200k-39432921` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `495e1890ed81` |
| `dqmc-he-39358341/ferminet` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `85a279b13d00` |
| `dqmc-he-39358341/lapnet` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `0dd8c2839357` |
| `dqmc-he-lapnet-200k-41503516` | 41617763 | adapter-pkg-dev-f6b16d1-emit-41617763 | `017e201e5a84c0c6…` | fdb6127 | `d17611855f83` |
| `dqmc-he-lapnet-parity-41694421` | 41826099 | adapter-pkg-dev-d04a96f-emit-41826099 | `017e201e5a84c0c6…` | fdb6127 | `59890d8a1a0d` |
| `dqmc-he-39358341/psiformer` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `66a863fe8c7e` |
| `dqmc-he-39358341/transpsiformer` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `fe18ffde7261` |
| `r2-he-39074394` | 39511430 | none printed | `absent (module a…` | unpinned | `3e856b7ef847` |
| `psi-2e-39195983/He` | 39511430 | none printed | `absent (module a…` | unpinned | `8de8d45458db` |
| `orbformer-he-eval-41158747` | 41160609 | none (clone pinned to TPEN 6bd53bb0) | `fb0cec1ae1afc479…` | e139a10 | `476270cf83a0` |
| `dqmc-ferminet-li-39571924` | 40678718 | adapter-pkg-dev-985223bc (FROZEN) | `bdbe5bf447901af8…` | f84b136 | `dc1675057af8` |
| `fn-li-1e6-40706320` | 41055251 | adapter-pkg-dev-7d8391a | `5d4fa8eb38878bd2…` | bec0219 | `cf27d0dc324c` |
| `r2-eval-39145226/Li` | 39512025 | none printed | `absent (module a…` | unpinned | `6bd479253dc2` |
| `r2-full-38985858/Li` | 39511430 | none printed | `absent (module a…` | unpinned | `3b549c89ef2d` |
| `psi-atoms-39233775/Li` | 39511430 | none printed | `absent (module a…` | unpinned | `ecc5354251c8` |
| `dqmc-lapnet-lih-39639503` | 40774916 | adapter-pkg-5a2dce0 | `fb0cec1ae1afc479…` | e139a10 | `798154746283` |
| `dqmc-deeperwin-n2-39915645` | 40774916 | adapter-pkg-5a2dce0 | `fb0cec1ae1afc479…` | e139a10 | `c124d4f399c1` |
| `fn-n-1e6-41159036` | 41269391 | adapter-pkg-dev-af84592-emit-41269391 | `017e201e5a84c0c6…` | fdb6127 | `9fd6aa6ceff4` |
| `psi-atoms-39379501/N` | 39511430 | none printed | `absent (module a…` | unpinned | `3611466b4a2f` |
| `psi-atoms-39379500/N` | 39541844 | none printed | `absent (module a…` | unpinned | `9253046f4a2c` |

### Table C — reproduction against the published value of the SAME method

| system | code | ansatz | run | E_ours (Ha) | E_pub (Ha) | diff (mHa) | comb. sigma (mHa) | sigma dist | budget | estimator | <1.6 | <0.38 | verdict / reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b_atom | ferminet | ferminet | `fn-b-1e6-41159035` | -24.653865776 | -24.65370 | -0.1658 | 0.0306 | 5.4 | 5x published 2e5 | MISMATCH (training_tail vs published post-training eval) | PASS | PASS | reachable |
| b_atom | ferminet | ferminet | `psi-atoms-39379501/B` | -24.653843589 | -24.65370 | -0.1436 | 0.0334 | 4.3 | matched (2e5) | MISMATCH (training_tail vs published post-training eval) | PASS | PASS | reachable |
| b_atom | ferminet | psiformer | `psi-atoms-39379500/B` | -24.653751929 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |
| be_atom | ferminet | ferminet | `r2-eval-39145226/Be` | -14.667313823 | -14.66733 | +0.0162 | 0.0304 | 0.5 | 100000 vs 2e5 | MATCHED (inference) | PASS | PASS | reachable |
| be_atom | ferminet | ferminet | `r2-full-38985858/Be` | -14.667322702 | -14.66733 | +0.0073 | 0.0316 | 0.2 | matched (2e5) | MISMATCH (training_tail vs published post-training eval) | PASS | PASS | reachable |
| be_atom | ferminet | psiformer | `psi-atoms-39233775/Be` | -14.667317960 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |
| h2_molecule | deepqmc | default | `dqmc-paulin2-h2-fix-39520283` | -1.174470104 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: no transcribed published PauliNet/DeepQMC value for this system in this repository |
| h2_molecule | ferminet | ferminet | `r2-h2-39068280` | -1.174474339 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: H2 is absent from FermiNet Table 2 - that table carries LiH, Li2, N2, ethene, bicyclobutane only |
| h2_molecule | ferminet | psiformer | `psi-2e-39195983/H2` | -1.174469850 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |
| he_atom | deepqmc | deeperwin | `dqmc-he-39358341/deeperwin` | -2.903722518 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | deeperwin | `dqmc-he-deeperwin-200k-41503515` | -2.903723506 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | default | `dqmc-he-39358341/default` | -2.903686363 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: no transcribed published PauliNet/DeepQMC value for this system in this repository |
| he_atom | deepqmc | default | `dqmc-he-default-100k-39411090` | -2.903709090 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: no transcribed published PauliNet/DeepQMC value for this system in this repository |
| he_atom | deepqmc | default | `dqmc-he-default-200k-39432921` | -2.903716541 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: no transcribed published PauliNet/DeepQMC value for this system in this repository |
| he_atom | deepqmc | ferminet | `dqmc-he-39358341/ferminet` | -2.903718433 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | lapnet | `dqmc-he-39358341/lapnet` | -2.903720250 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | lapnet | `dqmc-he-lapnet-200k-41503516` | -2.903730759 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | lapnet | `dqmc-he-lapnet-parity-41694421` | -2.903720271 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | psiformer | `dqmc-he-39358341/psiformer` | -2.903723220 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | deepqmc | transpsiformer | `dqmc-he-39358341/transpsiformer` | -2.903721803 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| he_atom | ferminet | ferminet | `r2-he-39074394` | -2.903717803 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: He is absent from FermiNet Table 1 - that atom sequence starts at Li |
| he_atom | ferminet | psiformer | `psi-2e-39195983/He` | -2.903727149 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |
| he_atom | oneqmc | orbformer-se | `orbformer-he-eval-41158747` | -2.903742715 | — | — | — | — | — | inference | — | — | UNAVAILABLE: no published Orbformer helium number exists (arXiv:2506.19960 census: "helium" 0 hits, standalone "He" 0 hits) |
| li_atom | deepqmc | ferminet | `dqmc-ferminet-li-39571924` | -7.478049481 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| li_atom | ferminet | ferminet | `fn-li-1e6-40706320` | -7.477971084 | -7.47798 | +0.0089 | 0.0107 | 0.8 | 5x published 2e5 | MISMATCH (training_tail vs published post-training eval) | PASS | PASS | reachable |
| li_atom | ferminet | ferminet | `r2-eval-39145226/Li` | -7.477910461 | -7.47798 | +0.0695 | 0.0111 | 6.2 | 100000 vs 2e5 | MATCHED (inference) | PASS | PASS | reachable |
| li_atom | ferminet | ferminet | `r2-full-38985858/Li` | -7.477892105 | -7.47798 | +0.0879 | 0.0137 | 6.4 | matched (2e5) | MISMATCH (training_tail vs published post-training eval) | PASS | PASS | reachable |
| li_atom | ferminet | psiformer | `psi-atoms-39233775/Li` | -7.478035439 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |
| lih_molecule | deepqmc | lapnet | `dqmc-lapnet-lih-39639503` | -8.070316950 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| n2_molecule | deepqmc | deeperwin | `dqmc-deeperwin-n2-39915645` | -109.540602959 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: DeepQMC reimplementation, not the named authors' code; the record itself forbids resting a method claim on it |
| n_atom | ferminet | ferminet | `fn-n-1e6-41159036` | -54.589288904 | -54.58882 | -0.4689 | 0.0608 | 7.7 | 5x published 2e5 | MISMATCH (training_tail vs published post-training eval) | PASS | FAIL | reachable |
| n_atom | ferminet | ferminet | `psi-atoms-39379501/N` | -54.589245788 | -54.58882 | -0.4258 | 0.0655 | 6.5 | matched (2e5) | MISMATCH (training_tail vs published post-training eval) | PASS | FAIL | reachable |
| n_atom | ferminet | psiformer | `psi-atoms-39379500/N` | -54.588975019 | — | — | — | — | — | training_tail | — | — | UNAVAILABLE: Psiformer published energies not transcribed (arXiv:2211.13672 never read by this program) |

### Table D — cross-code spread on he_atom (14 records, every code and ansatz)

| rank | run | code | ansatz | n | E (Ha) | dE vs exact (mHa) | se (mHa) | dE/se |
|---|---|---|---|---|---|---|---|---|
| 1 | `orbformer-he-eval-41158747` | oneqmc | orbformer-se | 40000 | -2.903742715 | -0.0183 | 0.0115 | -1.6 |
| 2 | `dqmc-he-lapnet-200k-41503516` | deepqmc | lapnet | 100204 | -2.903730759 | -0.0064 | 0.0069 | -0.9 |
| 3 | `psi-2e-39195983/He` | ferminet | psiformer | 200000 | -2.903727149 | -0.0028 | 0.0027 | -1.0 |
| 4 | `dqmc-he-deeperwin-200k-41503515` | deepqmc | deeperwin | 188894 | -2.903723506 | +0.0009 | 0.0020 | +0.4 |
| 5 | `dqmc-he-39358341/psiformer` | deepqmc | psiformer | 20000 | -2.903723220 | +0.0012 | 0.0032 | +0.4 |
| 6 | `dqmc-he-39358341/deeperwin` | deepqmc | deeperwin | 20000 | -2.903722518 | +0.0019 | 0.0048 | +0.4 |
| 7 | `dqmc-he-39358341/transpsiformer` | deepqmc | transpsiformer | 20000 | -2.903721803 | +0.0026 | 0.0023 | +1.1 |
| 8 | `dqmc-he-lapnet-parity-41694421` | deepqmc | lapnet | 200000 | -2.903720271 | +0.0041 | 0.0013 | +3.1 |
| 9 | `dqmc-he-39358341/lapnet` | deepqmc | lapnet | 20000 | -2.903720250 | +0.0041 | 0.0032 | +1.3 |
| 10 | `dqmc-he-39358341/ferminet` | deepqmc | ferminet | 20000 | -2.903718433 | +0.0059 | 0.0038 | +1.5 |
| 11 | `r2-he-39074394` | ferminet | ferminet | 200000 | -2.903717803 | +0.0066 | 0.0027 | +2.4 |
| 12 | `dqmc-he-default-200k-39432921` | deepqmc | default | 200000 | -2.903716541 | +0.0078 | 0.0080 | +1.0 |
| 13 | `dqmc-he-default-100k-39411090` | deepqmc | default | 100000 | -2.903709090 | +0.0153 | 0.0072 | +2.1 |
| 14 | `dqmc-he-39358341/default` | deepqmc | default | 20000 | -2.903686363 | +0.0380 | 0.0160 | +2.4 |

Span across all 14 he_atom records: **0.0564 mHa** (lowest -2.903742715, highest -2.903686363).


### Table E — the He budget asymmetry, and the measured bound on its residual

The DeepErwin/LapNet He pair was submitted at a common 200000-step target.
DeepErwin `41503515` stopped at n=188894 on an internal 28200 s timeout
(`rc=124`); the LapNet budget-parity rerun `41694421` reached the full 200000
(`COMPLETED 0:0`, `chkpt-200000` present). A truncated variant record was emitted
from the same LapNet series at DeepErwin's exact step count, and it has landed:
`dqmc-he-lapnet-parity-41694421/baseline_record.trunc188890.job41832597.20260825T211206Z.json`
(emit job `41832597`, package `adapter-pkg-dev-d04a96f-emit-41832597`).

| record | n | E (Ha) | se (mHa) | dE vs exact (mHa) |
|---|---|---|---|---|
| `dqmc-he-deeperwin-200k-41503515` (canonical) | 188894 | -2.903723506 | 0.0020 | +0.0009 |
| `dqmc-he-lapnet-parity-41694421` (canonical) | 200000 | -2.903720271 | 0.0013 | +0.0041 |
| `…-41694421` `trunc188890` variant | 188890 | -2.903721574 | 0.0016 | +0.0028 |

* **Exact-parity comparison** (n = 188894 vs 188890, a 4-step difference, 0.002%):
  DeepErwin minus LapNet = **-0.00193 mHa**, combined sigma 0.00256 mHa, i.e.
  **0.76 sigma**. At equal budget the two are statistically indistinguishable.
* **Measured bound on the residual budget asymmetry.** LapNet at n=200000 minus
  LapNet at n=188890 = **+0.00130 mHa**. These are nested windows over one
  series, so their standard errors are not independent and no sigma is quoted;
  the number is a magnitude, not a test. It is the first *measured* bound on the
  5.9% residual, replacing the "asserted plausible-small by scaling intuition"
  caveat carried in the canonical parity record.
* **What the un-truncated pair would have said.** Comparing the two canonical
  records as emitted (200000 vs 188894) gives LapNet minus DeepErwin =
  **+0.00324 mHa** at 1.34 sigma. Against the exact-parity -0.00193 mHa that is
  an opposite sign and a 0.00517 mHa shift. The asymmetry therefore mattered for
  the sign of the pair comparison and for nothing else: neither number is within
  two orders of magnitude of either threshold.

### Table F — record field coverage (populated / examined)

| record field | scorecard use | populated | examined |
|---|---|---|---|
| `local_energy_variance_hartree2` | scorecard axis 4, sigma^2/N | 1 | 33 |
| `parameter_count` | scorecard axis 12 | 1 | 33 |
| `dtype` | confounder protocol, dtype knob | 0 | 33 |
| `seed` | scorecard axis 6, seed spread | 18 | 33 |
| `wall_clock_seconds` | scorecard axis 9, GPU-seconds | 29 | 33 |
| `n_gpus` | scorecard axis 9, GPU-seconds | 30 | 33 |
| `gpu_model` | confounder protocol, hardware knob | 30 | 33 |
| `device_type` | confounder protocol, hardware knob | 30 | 33 |
| `steps` | scorecard axis 7, E vs step | 33 | 33 |
| `samples` | scorecard axis 8, E vs local-energy evaluations | 33 | 33 |
| `run_dir` | provenance | 0 | 33 |
| `collected_at` | provenance | 0 | 33 |
## 6. Findings

1. **Nine reproduction verdicts exist. Seven PASS both thresholds; two FAIL the
   0.38 mHa (1 kJ/mol) threshold and pass 1.6 mHa.** The two failures are both
   `n_atom` against FermiNet's published -54.58882(6): `fn-n-1e6-41159036` at
   -0.4689 mHa and `psi-atoms-39379501/N` at -0.4258 mHa. Both are *below* the
   published value, which is the direction a longer or equal-budget run should
   move, so this is a disagreement in magnitude, not evidence of an error in our
   direction. It is still a FAIL against the threshold the scorecard names.
2. **No row fails chemical accuracy against the published value; exactly one row
   fails it against the reference energy.** `dqmc-deeperwin-n2-39915645` sits
   +1.6970 mHa above `n2_molecule`'s -109.5423, failing both 1.6 mHa and
   0.38 mHa. It is also 100000 of a 200000-step target and generation-2
   estimator code.
3. **The estimator-matched rows are not the closest ones.** `r2-eval-39145226/Li`
   is the only Li row using the published estimator, and it lands +0.0695 mHa
   from published at 6.2 sigma, further away than the estimator-mismatched
   `fn-li-1e6-40706320` at +0.0089 mHa and 0.8 sigma. The training-tail-sits-high
   argument therefore does not explain the Li discrepancy pattern, and the two
   Li deviations have opposite implications. Sign is also not consistent across
   elements: at matched 200000-step budget, Li is +0.0879 mHa and Be +0.0073 mHa
   above published, while B is -0.1436 mHa and N -0.4258 mHa below it.
4. **Three He records sit below the exact non-relativistic energy.** The exact
   value is a rigorous lower bound on a variational expectation for the same
   Hamiltonian, so a negative dE is estimator bias or noise, not a better wave
   function. Measured: `orbformer-he-eval-41158747` -0.0183 mHa (1.6 sigma),
   `dqmc-he-lapnet-200k-41503516` -0.0064 mHa (0.9 sigma),
   `psi-2e-39195983/He` -0.0028 mHa (1.0 sigma). All are within about 1.6 sigma
   and so individually consistent with noise; the Orbformer row is the largest
   excursion and is also the one whose estimator is a Huber M-estimate over a
   correlated step series rather than a plain mean.
5. **Cross-code He agreement is 0.0564 mHa end to end**, an order of magnitude
   inside chemical accuracy, and the ordering is dominated by step budget rather
   than by ansatz: the three worst rows are the three smallest DeepQMC `default`
   budgets (20000, 100000, 200000 steps giving +0.0380, +0.0153, +0.0078 mHa).
   Read this as an upper bound on ansatz spread, per section 3 point 3.
6. **`lih_molecule` has no verdict at all, on either axis, and the reason is
   geometry.** `dqmc-lapnet-lih-39639503` ran `lih_1.639999_angstrom.yaml`,
   R = 3.099149 bohr, against a registry entry at R = 3.015 bohr — +0.084149 bohr,
   2.791%. The record says so itself. Its apparent +0.2310 mHa is not comparable
   to the reference and is marked `n/a`, not PASS.

## 7. What is NOT available, and why

Reported so that an axis nobody could compute is not mistaken for an axis that
came out clean.

| scorecard axis | status | reason |
|---|---|---|
| 4, `sigma^2/N` local-energy variance | 1 of 33 records | only `orbformer-he-eval-41158747` populates `local_energy_variance_hartree2` |
| 5, variance-extrapolated energy | unavailable for every system | needs several variances per system; one exists in total |
| 6, seed spread | unavailable | 18 of 33 records carry a seed and 17 of those are seed 23; the 18th is 42. No system has two records that differ only in seed |
| 12, parameter count | 1 of 33 records | only the Orbformer row |
| confounder protocol, dtype | 0 of 33 records | `dtype` is null everywhere, so the float32/float64 knob the protocol calls load-bearing cannot be reported for any row |
| 9, GPU-seconds | partial | `wall_clock_seconds` null for 4 rows (`dqmc-ferminet-li-39571924`, `dqmc-he-deeperwin-200k-41503515`, `dqmc-he-lapnet-200k-41503516`, `dqmc-lapnet-lih-39639503`); `device_type`, `gpu_model` and `n_gpus` all null for exactly the three `fn-*` rows, which therefore cannot be placed on identical hardware |
| estimator window | not machine-readable for any row | the schema has no window field; the window survives only as prose in `notes` (item `f23e52fa`) |
| training vs inference distinction | present but coarse | `estimator` is a free string; nothing in the record states the number of post-training steps or the block count except as prose (items `f23e52fa`, `3695e20d`) |
| Psiformer reproduction, 6 rows | verdict unavailable | arXiv:2211.13672 has never been read by this program; no transcribed published Psiformer energies exist |
| PauliNet/DeepQMC reproduction, 4 `default` rows | verdict unavailable | no transcribed published PauliNet value for `he_atom` or `h2_molecule` in this repository |
| DeepQMC reimplementation rows, 14 | verdict unavailable by construction | the records forbid resting a method claim on a reimplementation of another group's ansatz |
| Orbformer He | verdict unavailable | no published Orbformer helium number exists; the released checkpoint's own model card names He as an expected-worse case |
| `ethene`, `bicyclobutane`, `Li2`, `C`, `O`, `F`, `Ne` | no record | never run; and their references are CCSD(T)/CBS for the first two, not exact |
| `hooke_pair_singlet_omega0.5` | no baseline record | no baseline has been ported to a harmonic trap yet — item `cc2e7aec`, which is also what blocks the parent item `476ffe33` |

## 8. Reproducing this report

The record corpus lives on Cannon and is not in the repository. To re-derive
every number above:

```bash
# 1. the corpus, full depth (42 files, 33 canonical)
ssh cannon 'find /n/netscratch/kozinsky_lab/Lab/rhu/nnqmc-baselines-20260812/runs \
  -name "baseline_record*.json"'

# 2. per-record provenance: which emit job wrote a given canonical file
ssh cannon 'grep -l -- "<run-id>/baseline_record.json" \
  /n/netscratch/kozinsky_lab/Lab/rhu/nnqmc-baselines-20260812/logs/*.out'

# 3. the estimator code that job loaded
ssh cannon 'grep -n "statistics module\|adapter=\|adapter_pkg=" \
  /n/netscratch/kozinsky_lab/Lab/rhu/nnqmc-baselines-20260812/logs/<log>.out'
ssh cannon 'git hash-object \
  /n/netscratch/kozinsky_lab/Lab/rhu/nnqmc-baselines-20260812/<pkg>/experiments/baselines/statistics.py'

# 4. map that blob back to a dev commit
git log --oneline origin/dev -- experiments/baselines/statistics.py
git ls-tree <commit> experiments/baselines/statistics.py
```

Each Table B row carries the first 12 hex of the record file's sha256, so a row
is pinned to bytes: if the file changes, the row no longer applies to it.
