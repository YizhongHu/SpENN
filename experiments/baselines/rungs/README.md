# Cluster rung helpers

Small scripts used to build, dispatch and gate the baseline seed-spread rungs on
Polaris. They lived only on the cluster until 2026-08-31; they are here so they
are reviewable, testable and reusable.

| file | purpose |
|---|---|
| `rung_makeplan.py` | build a `StagePlanV2` of independent psiformer rows, one per seed |
| `rung_gate.py` | gate a completed run on its **artefacts** |
| `fit_rate.py` | fit a per-step rate from timestamped logs, with a cut sweep |

## Usage

```sh
python rung_makeplan.py <system> <results_dir> <plan_dir> <gpus> <rows> <steps> <plan_id>
python rung_gate.py     <run_root> <expected_rows> <gpus_per_row> <expected_hosts> [system]
python fit_rate.py      <probe_root>
```

## Why these are shaped the way they are

Each guard here exists because its absence cost a real run.

**System is an argument, not a copy.** There were once two builders differing in
four lines — one for H2, one for He. Production job `7579539` ran H2 while
labelled He, because the production script was derived from a local copy that
never received the He substitution (applied remotely with `sed -i`), and the
pre-submission check invoked the He builder *by name*, validating a code path the
job never took. One parameterised table removes the choice.

**H2 pins its geometry.** `diatomic.py` defaults to 0.737164 angstrom; the
reference rows were produced at 1.4 bohr. Accepting the default is silent and
worth ~2.1e-4 Ha — roughly six times the seed standard deviation.

**The gate checks the SYSTEM, not just the structure.** A wrong-system run passes
every structural gate identically: right row count, right hosts, disjoint GPU
bindings, checkpoints present, energies distinct. Only the energy separates them.
`7579539` was caught by an unrelated node fault, not by design.

**Records are pointers, not outcomes.** `dispatch_records.jsonl` carries
`status_path`; the row's status lives in that file. `record["status"]` is
silently `None`.

**GPU exclusivity is per `(host, visibility)`.** Across N nodes the same local
index recurs once per node — GPU `0` on two hosts is two devices. Requiring
globally distinct values produces a false `DOUBLE-BOOKED` verdict.

**`placement.gpus` cannot prove exclusivity.** `nvidia-smi` ignores
`CUDA_VISIBLE_DEVICES`, so it lists every GPU on the node regardless of binding.

**Exit codes prove nothing here.** `pipeline.py` writes
`verification.exit_code=0` whenever `dispatch()` does not raise, and a failed row
is *recorded* rather than raised. Conversely one failed row makes dispatch raise
and suppresses records for **all** rows, so `records=0` can coexist with 19 good
runs. Gate on artefacts.

**A large output file is not evidence of completion.** Completed DeepQMC He runs
are ~9382 MB; partials 307-4702 MB. Size correlates and is not the test — the
run's own completion marker is.
