# SpENN experiments

This tree is deliberately separate from the `spenn/` package. Treat it as if it
were its own repository:

- Code under `experiments/` must **not** import `spenn`. Study and analysis
  scripts use the standard library plus small generic dependencies (PyYAML).
  The one sanctioned exception is `spenn.run.run_from_config` for launcher-style
  scripts that need to start configured runs programmatically; nothing else
  from `spenn` may be imported.
- Tests for experiment code live under `experiments/` (next to the code they
  test), not under the repo-level `tests/` tree.
- Configs under `experiments/` reference `spenn` only through Hydra
  `_target_` strings, which the `run.py` entrypoint resolves.

## Division of duties

Launching SpENN runs is the **user's** duty. Experiment code:

1. sets up configs, manifests, and scripts,
2. documents the exact launch commands, and
3. interfaces only with run **outputs** — the run directory files
   (`resolved_config.yaml`, `metadata.json`, `status.json`, `metrics.csv`,
   `metrics.jsonl`) are the contract between `spenn` runs and experiment
   scripts.

Experiment scripts never launch training, never read W&B, and never reach into
`spenn` internals. If a script needs information, it must come from the run
directory.

## Layout

```text
experiments/
  hooke/
    configs/
      smoke/        legacy cheap end-to-end test configs
      benchmark/    legacy benchmark-shaped test/reference configs
    studies/
      exact_cusp_diagnostics/  exact Hooke singlet local-energy/cusp diagnostic
      pair_validation/   validation scan: manifest, collector, selector
```
