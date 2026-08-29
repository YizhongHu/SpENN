# He-cutover

This study preserves He-v1 science by referencing its tracked train and evaluation configs while replacing orchestration with `StagePlanV2`, study-local admission, and `ParslAttachExecutor`. `hev1.py` alone owns the cross-study import boundary; `cutover_plan.py` owns planning, `admission.py` owns runtime resolution, `pipeline.py` owns in-allocation ordering, and the row wrappers alone import Torch indirectly after GPU binding.

Acceptance requires strict plan tests, the exact three-field training-scale diff, allocation and device refusal tests, probe-before-science ordering, immutable dispatch receipts, exit-code/verification agreement, and root-free facility templates. This smoke deliberately evaluates only `mcmc_energy`; it is not the five-task reduced suite.

## Opt-in analytic local-energy evaluation

The three tracked grids (`proof_grid.yaml`, `production_grid.yaml`, and
`smoke_grid.yaml`) deliberately keep `eval_config: experiments/atomistic/he-v1/configs/eval.yaml`;
that is the naive default and must not be repointed. To construct an isolated
opt-in row, merge `experiments/atomistic/he-v1/configs/analytic_local_energy.yaml`
into that base config and set the row's `config`/`eval_config` to the resulting
temporary config. Wire both sides of `mcmc_energy` explicitly:

```text
evaluation_tasks.mcmc_energy.generator.evaluator=${local_energy_evaluator}
evaluation_tasks.mcmc_energy.calculators.0.evaluator=${local_energy_evaluator}
```

This route is for an explicitly selected analytic row only; it is not a change
to any established proof, smoke, or production grid. The analytic overlay's
preflight must pass before sampling. Its grouping regularizes the
electron-nucleus singularity only. Helium's `ElectronElectronCusp` remains in
the autodiff regular factor and `electron_electron` remains an ordinary
unclaimed term, so nothing is dropped or double-counted; however, helium's
electron-electron coalescence is **not regularized**, and electron-electron cusp
fusion is deferred by design.

## Operator runbook

Set `TPEN_CHECKOUT` to the development checkout that carries the he-cutover
stack and tracks `dev`. Never point these planning or smoke commands at the
production checkout that tracks `main`. The checkout path remains an
operator-supplied variable rather than a tracked facility default.

Cannon planning (login node):

```bash
: "${TPEN_CHECKOUT:?checkout root required}"
: "${TPEN_UV:?absolute uv executable required}"
case "$TPEN_UV" in /*) ;; *) echo "TPEN_UV must be absolute" >&2; exit 2;; esac
PYTHONPATH="${TPEN_CHECKOUT:?checkout root required}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV" run --project "$TPEN_CHECKOUT" --locked --extra cpu python "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/cutover_plan.py" --facility cannon --results-root "${TPEN_RESULTS_ROOT:?}" --plan-attempt-id "${TPEN_PLAN_ATTEMPT_ID:?}"
```

Inspect `00_plan/$TPEN_PLAN_ATTEMPT_ID/manifest.json` and both `tasks.jsonl` files, then submit:

```bash
sbatch experiments/atomistic/he-cutover/templates/cannon_smoke.sbatch
```

Polaris planning and execution only read an operator-supplied, validated,
facility-inheriting overlay. A syncing `uv` must never target this overlay:
`uv sync` owns its project environment and may recreate it as an isolated
virtualenv, which discards the system-site inheritance required for the
facility Torch. Provisioning and lock validation are separate operator actions.
The runbook exposes the overlay's executables on `PATH` because Parsl launches
`interchange.py` and `process_worker_pool.py` by bare name.

```bash
: "${TPEN_CHECKOUT:?checkout root required}"
: "${TPEN_UV_ENV:?validated facility-inheriting overlay required}"
: "${TPEN_LIBSHIM:?existing libmpi shim directory required}"
: "${TPEN_CUDA13_LIB:?CUDA 13 lib64 directory required}"
: "${TPEN_CUDA129_LIB:?CUDA 12.9 lib64 directory required}"
export PATH="$TPEN_UV_ENV/bin${PATH:+:$PATH}"
PYTHONPATH="${TPEN_CHECKOUT:?checkout root required}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV_ENV/bin/python" "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/cutover_plan.py" --facility polaris --results-root "${TPEN_RESULTS_ROOT:?}" --plan-attempt-id "${TPEN_PLAN_ATTEMPT_ID:?}"
```

Inspect the manifest and task files, then submit:

```bash
qsub -v TPEN_CHECKOUT,TPEN_RESULTS_ROOT,TPEN_UV_ENV,TPEN_LIBSHIM,TPEN_CUDA13_LIB,TPEN_CUDA129_LIB,TPEN_PLAN_ATTEMPT_ID experiments/atomistic/he-cutover/templates/polaris_smoke.pbs
```

Success is scheduler exit status zero and `verification.json` containing `"complete": true` and `"exit_code": 0`. S1 and S2 own real smoke submissions; P2 only prepares and unit-tests these files.

## Full production

`production_grid.yaml` is separate from the immutable smoke grid. It expands the
frozen He-v1 production science to three 300,000-step training seeds and 36
evaluations: three declared checkpoints by four fresh chains for each seed,
using all ten tasks selected by `configs/eval.yaml`. Plan it explicitly with
`--grid experiments/atomistic/he-cutover/production_grid.yaml` and submit
`templates/polaris_production.pbs` only after inspecting all 39 rows and passing
the full dry-run path/cardinality gate. The production template targets the
operator-selected `capacity` queue and logs the checkout's own full HEAD SHA.
