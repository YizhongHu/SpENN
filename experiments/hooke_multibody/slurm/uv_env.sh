#!/bin/bash

case "${1:-cpu}" in
  cpu)
    unset UV_PROJECT_ENVIRONMENT
    export SPENN_UV_EXTRA="cpu"
    ;;
  gpu)
    export UV_PROJECT_ENVIRONMENT="${PWD}/.venv-gpu"
    export SPENN_UV_EXTRA="cu126"
    ;;
  *)
    echo "usage: source experiments/hooke_multibody/slurm/uv_env.sh [cpu|gpu]" >&2
    return 2
    ;;
esac

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
