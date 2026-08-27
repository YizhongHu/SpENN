# He-cutover

This study preserves He-v1 science by referencing its tracked train and evaluation configs while replacing orchestration with `StagePlanV2`, study-local admission, and `ParslAttachExecutor`. `hev1.py` alone owns the cross-study import boundary; `cutover_plan.py` owns planning, `admission.py` owns runtime resolution, `pipeline.py` owns in-allocation ordering, and the row wrappers alone import Torch indirectly after GPU binding.

Acceptance requires strict plan tests, the exact three-field training-scale diff, allocation and device refusal tests, probe-before-science ordering, immutable dispatch receipts, exit-code/verification agreement, and root-free facility templates. This smoke deliberately evaluates only `mcmc_energy`; it is not the five-task reduced suite.

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

Polaris planning uses the facility Python and its `uv` module from the approved overlay:

```bash
: "${TPEN_CHECKOUT:?checkout root required}"
: "${TPEN_PYBIN:?facility python required}"
: "${TPEN_UV_ENV:?facility uv environment required}"
: "${TPEN_LIBSHIM:?existing libmpi shim directory required}"
: "${TPEN_CUDA13_LIB:?CUDA 13 lib64 directory required}"
: "${TPEN_CUDA129_LIB:?CUDA 12.9 lib64 directory required}"
"$TPEN_PYBIN" -m uv sync --project "$TPEN_CHECKOUT" --inexact --locked --extra parsl
PYTHONPATH="${TPEN_CHECKOUT:?checkout root required}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV_ENV/bin/python" "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/cutover_plan.py" --facility polaris --results-root "${TPEN_RESULTS_ROOT:?}" --plan-attempt-id "${TPEN_PLAN_ATTEMPT_ID:?}"
```

Inspect the manifest and task files, then submit:

```bash
qsub -v TPEN_CHECKOUT,TPEN_RESULTS_ROOT,TPEN_PYBIN,TPEN_UV_ENV,TPEN_LIBSHIM,TPEN_CUDA13_LIB,TPEN_CUDA129_LIB,TPEN_PLAN_ATTEMPT_ID experiments/atomistic/he-cutover/templates/polaris_smoke.pbs
```

Success is scheduler exit status zero and `verification.json` containing `"complete": true` and `"exit_code": 0`. S1 and S2 own real smoke submissions; P2 only prepares and unit-tests these files.
