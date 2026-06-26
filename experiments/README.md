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
    exact_cusp_diagnostics/  exact Hooke singlet local-energy/cusp diagnostic
```

The old `hooke/pair_validation/` tree was removed during the PR8.5-8.7
evaluation rewrite because it depended on the retired diagnostics/probe stack.
Old one-off diagnostic studies that depended on retired probe helpers were
removed for the same reason. Use the new evaluator fixtures and future preflight
study path instead.

The legacy `hooke/configs/` smoke, benchmark, and preflight configs were removed
with the same cleanup. Keep runnable regression configs under
`tests/integration/artifacts/` unless they are owned by a current experiment
study.
