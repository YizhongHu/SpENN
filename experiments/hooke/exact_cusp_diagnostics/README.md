# exact_cusp_diagnostics

Standalone diagnostics for the exact two-electron Hooke singlet. The run uses
`spenn.physics.hooke.HookeSingletExact`, evaluates local-energy diagnostics, and
writes a deterministic pair-distance probe for cusp inspection.

Run the evaluation:

```bash
uv run python run.py --config experiments/hooke/exact_cusp_diagnostics/configs/exact_singlet_eval.yaml
```

The run directory is printed in the terminal status output and recorded in
`outputs/exact_cusp_diagnostics/singlet/2026-06-15_232441_exact_cusp_diagnostics_eval_c30eef/`.

Plot local energy over pair distance:
```bash
uv run python experiments/hooke/exact_cusp_diagnostics/plot_pair_distance.py \
  --run-dir outputs/exact_cusp_diagnostics/singlet/2026-06-15_232651_exact_cusp_diagnostics_eval_281e94
```

By default the plotter reads
`diagnostics/pair_distance_probe/probe.csv` and writes
`diagnostics/pair_distance_probe/model_local_energy_vs_pair_distance.png`
inside the run directory.

The experiment script is file-only: it reads run artifacts and does not import
`spenn` or launch configured runs.
