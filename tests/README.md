# Test Layout

Unit tests live under `tests/unit/` and are grouped by package area:

```text
tests/unit/data/
tests/unit/nn/
tests/unit/reps/
tests/unit/physics/
tests/unit/sampling/
```

Integration tests live under `tests/integration/` and are automatically marked
with `pytest.mark.integration`. Test-owned configs and small fixtures that
reproduce integration results live under `tests/integration/artifacts/`.

Generated run outputs still go under `outputs/`. For example, the Hooke
integration configs use:

```text
tests/integration/artifacts/hooke/singlet.yaml
tests/integration/artifacts/hooke/triplet.yaml
outputs/integration_tests/YYYY-MM-DD/hooke_<sector>/<run_id>/
```

Some cross-cutting or placeholder tests remain at the top level until their
long-term ownership is clearer.
