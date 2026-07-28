# Pair-stability V4-0 compatibility harness

This directory is the V4-0, study-local compatibility harness for the Hooke
pair-stability experiment. It owns V4 study identity, guarded result roots,
versioned legacy-route declarations, controller/fan-out receipts, audit
evidence, reference publication, and parity comparison. Each physical
scientific stage remains the existing V3 CLI, invoked only through the V4
dispatcher as a pinned subprocess.

V4-0 is deliberately a bridge, not the V4 experiment framework. It does not
introduce generic trial, seed, producer, metric, search, checkpoint-event, or
dynamic scheduling contracts. Do not add V4-1 behavior here while validating
V4-0 parity.

## Ownership and safety boundary

- Do not edit V3 source or write into V3 results from this harness.
- Initialize one new absolute V4 results root for one lineage. The root guard
  rejects V3 roots, broad paths, traversal, and symlink escapes; experiment
  artifacts, receipts, and comparison reports remain below that root.
- The dispatcher renders complete typed legacy argv from `legacy_routes_v1.json`.
  It does not accept arbitrary trailing legacy arguments or direct V3 imports.
- Legacy source/config closure and the selected runtime source-file closure are
  fail-closed. Broader runtime environment/commit variants are recorded as
  provenance so drift stays visible to parity review.
- Preserve all output and failure evidence. Do not reuse a root for another
  candidate or reference, and do not delete failed lineage artifacts.

## Operator commands

Run focused static/unit checks from the repository root:

```bash
uv run pytest experiments/hooke/pair_stability_v4
bash -n experiments/hooke/pair_stability_v4/submit_stack.sh
```

Inspect or render one V4 route without launching it:

```bash
uv run python experiments/hooke/pair_stability_v4/dispatch.py render screen_plan \
  --results-root /absolute/new/v4-root --output-attempt example-lineage
```

The only stack launcher is:

```bash
RESULTS_ROOT=/absolute/new/v4-root \
  experiments/hooke/pair_stability_v4/submit_stack.sh smoke
```

It requires a clean checkout, creates/validates a new guarded V4 root, records
pre-submission controls, and submits a Sapphire controller. The controller
runs the complete numbered V3-compatible stack through `dispatch.py`. It is
not a substitute for the canonical evidence gates below.

Reference handling and comparison are separate, explicit operations:

```bash
uv run python experiments/hooke/pair_stability_v4/reference.py freeze \
  --results-root /absolute/fresh-v3-root \
  --destination "$(pwd)/experiments/hooke/pair_stability_v4/reference/v3_smoke/<id>" \
  --attempts '{"grid":"...","train":"...","validation":"...","collect":"...","selection":"...","final_grid":"...","final_train":"...","final_eval":"...","final_collect":"...","report":"..."}'

uv run python experiments/hooke/pair_stability_v4/compare.py \
  --reference experiments/hooke/pair_stability_v4/reference/v3_smoke/<id> \
  --candidate-root /absolute/new/v4-root \
  --candidate-attempts '{"grid":"...","train":"...","validation":"...","collect":"...","selection":"...","final_grid":"...","final_train":"...","final_eval":"...","final_collect":"...","report":"..."}' \
  --comparison-id <new-comparison-id>
```

Run these commands from the repository root and replace placeholders with actual
completed lineage identities. The freeze destination must be an absolute, new
direct child of `experiments/hooke/pair_stability_v4/reference/v3_smoke/`.
Reference freeze is create-only and must use a clean, fresh V3 smoke lineage;
comparison exits nonzero when any protected artifact or control evidence
differs.

### Frozen-V3/manual comparison

The default comparison contract is `canonical-controller`: it requires the
complete Sapphire controller closure described below. It is unchanged by the
manual path.

For a temporary restructure check against an operator-preserved frozen V3
reference, select the manual mode explicitly:

```bash
uv run python experiments/hooke/pair_stability_v4/compare.py \
  --reference experiments/hooke/pair_stability_v4/reference/v3_smoke/<id> \
  --candidate-root /absolute/new/v4-root \
  --candidate-attempts '{"grid":"<id>","train":"<id>","validation":"<id>","collection":"<id>","selection":"<id>","final_grid":"<id>","final_train":"<id>","final_eval":"<id>","final_collect":"<id>","report":"<id>"}' \
  --comparison-mode frozen-v3-manual \
  --comparison-id <new-comparison-id>
```

Manual mode skips canonical controller-closure and closure-equivalence proof.
It still requires the guarded V4 root, ordinary completed-lineage audit,
layout validation, inventory checks, and artifact comparison. Its report
records `frozen-v3-manual` and
`not_checked_operator_trusted_manual`; it is neither canonical
controller-closed evidence nor proof of controller history or reference
immutability. Keep the frozen reference untouched while this temporary
restructure guard is in use.

## Cluster profile and gpu_test limit

The canonical V4-0 controller profile is Sapphire, 4 CPUs, 8 GiB per CPU, and
three days. Its request, worker job identity, and effective Slurm
CPU/memory/time evidence are recorded separately. If the worker's effective
profile cannot be verified or differs, finalization writes an immutable
`incomplete` receipt and returns nonzero.

Scientific fan-out uses `gpu_test` for acceptance smoke only. For this work it
is a hard limit that **each submitted gpu_test fan-out job or array has at most
two tasks**. The V4-0 profile is checked before controller submission and
again before every fan-out dispatch:

| Fan-out stage | Rows | Chunk size | Submitted array tasks |
| --- | ---: | ---: | ---: |
| screen train | 64 | 32 | 2 |
| screen evaluation | 64 | 32 | 2 |
| final train | 8 | 8 | 1 |
| final evaluation | 8 | 8 | 1 |

Do not evade the limit by splitting stages or use `gpu_test` as a full-science
default. Refreshable source-labelled cluster facts and the authoritative cap
are in [`.planning/cluster-knowledge.yaml`](../../../.planning/cluster-knowledge.yaml).

## Required canonical evidence (deferred)

This repository currently contains implementation and unit evidence, not a
canonical V4-0 parity result. Before claiming success, perform these retained,
clean-worktree operations:

1. Run a disposable ownership audit and Sapphire admission check.
2. Produce exactly one fresh V3 smoke reference from clean current source.
3. Run one fresh V4 smoke candidate under the guarded root and `gpu_test`
   two-task limit.
4. Freeze/verify the reference, audit V4 control closure, and compare the
   candidate with zero differences.
5. Retain Slurm logs, receipts, manifests, inventories, and comparison report
   as provenance.

See [the V4-0 implementation instructions](../../../.planning/v4-0-implementation-instructions.md)
for the approved scope and [the cluster record](../../../.planning/cluster-knowledge.yaml)
for refresh conditions. V4-1 design work begins only after the V4-0 compatibility
gate is accepted.
