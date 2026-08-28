#!/bin/bash -l
set -euo pipefail
: "${PMI_LOCAL_RANK:?mpiexec local rank is required for GPU binding}"
: "${PMI_RANK:?mpiexec global rank is required for row selection}"
: "${TPEN_UV_ENV:?validated facility overlay required}"

# PBS leaves all four GPUs visible.  ALCF's Polaris affinity order is
# deliberately reversed: local ranks 0,1,2,3 bind to visible GPUs 3,2,1,0.
# This export occurs before Python/JAX import.
export CUDA_VISIBLE_DEVICES="$((3 - PMI_LOCAL_RANK % 4))"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$TPEN_UV_ENV/bin/python" -m experiments.baselines.polaris_submit row \
  --manifest "$1" --row-index "$PMI_RANK" --results-root "$2"
