# He-cutover

This study preserves He-v1 science by referencing its tracked train and evaluation configs while replacing orchestration with `StagePlanV2`, study-local admission, and `ParslAttachExecutor`. `hev1.py` alone owns the cross-study import boundary; `cutover_plan.py` owns planning, `admission.py` owns runtime resolution, `pipeline.py` owns in-allocation ordering, and the row wrappers alone import Torch indirectly after GPU binding.

Acceptance requires strict plan tests, the exact three-field training-scale diff, allocation and device refusal tests, probe-before-science ordering, immutable dispatch receipts, exit-code/verification agreement, and root-free facility templates. This smoke deliberately evaluates only `mcmc_energy`; it is not the five-task reduced suite.

## Operator runbook

Cannon planning (login node):

```bash
: "${TPEN_UV:?absolute uv executable required}"
case "$TPEN_UV" in /*) ;; *) echo "TPEN_UV must be absolute" >&2; exit 2;; esac
"$TPEN_UV" run --locked --extra cpu python experiments/atomistic/he-cutover/cutover_plan.py --facility cannon --results-root "${TPEN_RESULTS_ROOT:?}" --plan-attempt-id "${TPEN_PLAN_ATTEMPT_ID:?}"
```

Inspect `00_plan/$TPEN_PLAN_ATTEMPT_ID/manifest.json` and both `tasks.jsonl` files, then submit:

```bash
sbatch experiments/atomistic/he-cutover/templates/cannon_smoke.sbatch
```

Polaris planning uses the facility Python and its `uv` module from the approved overlay:

```bash
: "${TPEN_PYBIN:?facility python required}"
: "${TPEN_UV_ENV:?facility uv environment required}"
"$TPEN_PYBIN" -m uv sync --inexact --locked --extra parsl
"$TPEN_UV_ENV/bin/python" experiments/atomistic/he-cutover/cutover_plan.py --facility polaris --results-root "${TPEN_RESULTS_ROOT:?}" --plan-attempt-id "${TPEN_PLAN_ATTEMPT_ID:?}"
```

Inspect the manifest and task files, then submit:

```bash
qsub -v TPEN_CHECKOUT,TPEN_RESULTS_ROOT,TPEN_PYBIN,TPEN_UV_ENV,TPEN_LIBSHIM,TPEN_PLAN_ATTEMPT_ID experiments/atomistic/he-cutover/templates/polaris_smoke.pbs
```

Success is scheduler exit status zero and `verification.json` containing `"complete": true` and `"exit_code": 0`. S1 and S2 own real smoke submissions; P2 only prepares and unit-tests these files.
