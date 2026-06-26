"""Unit tests for optional W&B logging projection."""

from __future__ import annotations

import math

import pytest

import spenn.logging as logging_module
from spenn.logging import LogRecord, WandB, project_record_to_wandb


class FakeRun:
    """Small stand-in for a W&B run."""

    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []
        self.defined_metrics: list[tuple[str, str]] = []
        self.summary: dict[str, object] = {}
        self.finish_count = 0
        self.artifacts: list[object] = []

    def define_metric(self, metric: str, *, step_metric: str) -> None:
        self.defined_metrics.append((metric, step_metric))

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))

    def finish(self) -> None:
        self.finish_count += 1

    def log_artifact(self, artifact: object) -> None:
        self.artifacts.append(artifact)


class FakeWandB:
    """Small stand-in for the imported ``wandb`` module."""

    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> FakeRun:
        self.init_kwargs = dict(kwargs)
        return self.run


def test_wandb_constructor_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(logging_module.importlib, "import_module", fail_import)
    WandB(project="spenn-qmc")


def test_wandb_missing_package_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(logging_module.importlib, "import_module", missing_import)
    logger = WandB(project="spenn-qmc")

    with pytest.raises(RuntimeError, match="uv sync --extra wandb"):
        logger.log(LogRecord(step=1, namespace="train", metrics={"energy": 1.0}))


def test_wandb_disabled_mode_is_noop_without_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(logging_module.importlib, "import_module", fail_import)
    logger = WandB(project="spenn-qmc", mode="disabled")
    logger.log(LogRecord(step=1, namespace="train", metrics={"energy": 1.0}))
    logger.finish()


def test_wandb_logs_scalar_projection_and_defines_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeWandB()
    monkeypatch.setattr(logging_module.importlib, "import_module", lambda name: fake)
    logger = WandB(
        project="spenn-qmc",
        entity="lab",
        name="run-name",
        group="hooke",
        tags=["test"],
        config={"run_id": "abc", "nested": {"flag": True}},
    )

    logger.log(
        LogRecord(
            step=17,
            namespace="train",
            metrics={
                "loss": 0.5,
                "energy": 1.25,
                "array": [1.0, 2.0],
                "nan": math.nan,
                "none": None,
            },
        )
    )
    logger.finish()
    logger.finish()

    assert fake.init_kwargs is not None
    assert fake.init_kwargs["project"] == "spenn-qmc"
    assert fake.init_kwargs["entity"] == "lab"
    assert fake.init_kwargs["name"] == "run-name"
    assert fake.init_kwargs["group"] == "hooke"
    assert fake.init_kwargs["tags"] == ["test"]
    assert fake.init_kwargs["config"] == {"run_id": "abc", "nested": {"flag": True}}
    assert ("train/*", "train/step") in fake.run.defined_metrics
    assert ("dashboard/*", "train/step") in fake.run.defined_metrics
    assert fake.run.logged == [
        {
            "train/step": 17,
            "train/loss": 0.5,
            "train/energy": 1.25,
            "dashboard/loss": 0.5,
            "dashboard/energy": 1.25,
        }
    ]
    assert fake.run.finish_count == 1


def test_project_record_to_wandb_supports_scalar_record_shape() -> None:
    payload = project_record_to_wandb(
        {"step": 3, "namespace": "train/sampler", "key": "acceptance_rate", "value": 0.75}
    )

    assert payload == {
        "train/step": 3,
        "train/sampler/acceptance_rate": 0.75,
        "dashboard/acceptance_rate": 0.75,
    }


def test_project_record_to_wandb_derives_health_flags_from_checks() -> None:
    data_payload = project_record_to_wandb(
        LogRecord(step=5, namespace="checks/data_integrity", metrics={"passed": False})
    )
    sampler_payload = project_record_to_wandb(
        LogRecord(step=5, namespace="checks/sampler", metrics={"passed": True})
    )
    equivariance_payload = project_record_to_wandb(
        LogRecord(step=5, namespace="checks/equivariance/full_model", metrics={"passed": True})
    )

    assert data_payload == {
        "checks/train_step": 5,
        "checks/data_integrity/passed": False,
        "train/step": 5,
        "health/numerics_ok": 0.0,
        "health/run_ok": 0.0,
    }
    assert sampler_payload == {
        "checks/train_step": 5,
        "checks/sampler/passed": True,
        "train/step": 5,
        "health/sampler_ok": 1.0,
        "health/run_ok": 1.0,
    }
    assert equivariance_payload == {
        "checks/train_step": 5,
        "checks/equivariance/full_model/passed": True,
        "train/step": 5,
        "health/equivariance_ok": 1.0,
        "health/run_ok": 1.0,
    }


def test_project_record_to_wandb_skips_non_scalar_values() -> None:
    payload = project_record_to_wandb(
        LogRecord(
            step=1,
            namespace="train",
            metrics={"tensor_like": [1, 2], "missing": None, "bad_float": math.inf},
        )
    )

    assert payload == {}


def test_runtime_metrics_are_written_to_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeWandB()
    monkeypatch.setattr(logging_module.importlib, "import_module", lambda name: fake)
    logger = WandB(project="spenn-qmc")

    logger.log(LogRecord(step=0, namespace="runtime", metrics={"wall_time_sec": 12.5}))

    assert fake.run.logged == [
        {
            "runtime/step": 0,
            "runtime/wall_time_sec": 12.5,
            "dashboard/wall_time_sec": 12.5,
            "train/step": 0,
        }
    ]
    assert fake.run.summary["runtime/wall_time_sec"] == 12.5
