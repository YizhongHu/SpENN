# TPEN experiments

This tree is deliberately separate from the `tpen/` package. Treat it as if it
were its own repository:

- Code under `experiments/` must **not** import `tpen`. Study and analysis
  scripts use the standard library plus small generic dependencies (PyYAML).
  The one sanctioned exception is `tpen.run.run_from_config` for launcher-style
  scripts that need to start configured runs programmatically; nothing else
  from `tpen` may be imported.
- Tests for experiment code live under `experiments/` (next to the code they
  test), not under the repo-level `tests/` tree.
- Configs under `experiments/` reference `tpen` only through Hydra
  `_target_` strings, which the `run.py` entrypoint resolves.