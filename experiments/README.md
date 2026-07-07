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