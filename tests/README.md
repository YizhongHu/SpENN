# Test Layout

Unit tests live under `tests/unit/` and are grouped by package area or component:

```text
tests/unit/callback/
tests/unit/data/
tests/unit/equivariance/
tests/unit/nn/
tests/unit/physics/
tests/unit/reps/
tests/unit/sampling/
tests/unit/training/
```

Integration tests live under `tests/integration/`, grouped by workflow or domain:

```text
tests/integration/hooke/
tests/integration/training/
```

Test-owned configs and small fixtures that reproduce integration results live
under `tests/integration/artifacts/`:

```text
tests/integration/artifacts/hooke/exact_singlet.yaml
tests/integration/artifacts/hooke/exact_triplet.yaml
tests/integration/artifacts/hooke/pair_train.yaml
tests/integration/artifacts/training/vmc_smoke.yaml
```

Shared pytest-only helpers live under `tests/helpers/`. Generated run outputs go
under `outputs/`.

No test modules live directly under `tests/` -- only `README.md`, `conftest.py`,
and `__init__.py` may.
