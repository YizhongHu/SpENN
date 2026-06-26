#!/usr/bin/env bash
# Reproduce the whole seed-explosion study end to end (read-only on report CSVs).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
D=experiments/hooke/studies/seed_explosion
for f in probe probe_eval analyze_probes decompose_explosion classify_defects onset_vs_reach; do
  if uv run --extra cpu python "$D/$f.py" > /dev/null 2>&1; then
    echo "OK   $f"
  else
    echo "FAIL $f"
  fi
done
echo "PLOTS:"
ls -1 "$D/plots"
