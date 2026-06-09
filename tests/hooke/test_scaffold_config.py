"""Config-loading tests for the Hooke scaffold smoke run."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from spenn.callback import ConfigSnapshot, Metadata, ReportSkeleton, ResolvedConfigSnapshot, Status
from spenn.logging import CSV, JSONL
from spenn.runner import Scaffold
from spenn.run import load_config, prepare_run_context

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments" / "hooke" / "configs" / "smoke" / "scaffold.yaml"


def test_smoke_config_prepares_generic_artifact_context(tmp_path: Path) -> None:
    """The smoke config resolves only generic run-management targets."""

    cfg = load_config(str(CONFIG), [f"run.root={tmp_path}"])
    for forbidden in ("system", "model", "hamiltonian", "sampler", "trainer", "diagnostics"):
        assert forbidden not in cfg

    context = prepare_run_context(cfg, config_path=str(CONFIG), command="pytest scaffold")

    assert context.run_dir.parent == tmp_path / "hooke_scaffold" / "scaffold"
    assert context.cfg.run.run_id
    assert context.cfg.run.dir == str(context.run_dir)
    assert context.source_cfg.run.run_id is None
    assert context.source_cfg.run.dir is None

    # Callbacks and loggers are owned by the runner, not the run context.
    assert context.callbacks == []
    assert context.loggers == []
    runner = instantiate(context.cfg.runner)
    assert [type(logger) for logger in runner.loggers] == [CSV, JSONL]
    assert [type(callback) for callback in runner.callbacks] == [
        ConfigSnapshot,
        ResolvedConfigSnapshot,
        Metadata,
        Status,
        ReportSkeleton,
    ]


def test_flat_public_targets_instantiate() -> None:
    """Hydra can instantiate every flat public scaffold target."""

    runner = instantiate(
        OmegaConf.create(
            {
                "_target_": "spenn.runner.Scaffold",
                "callbacks": [],
                "loggers": [],
            }
        )
    )
    callback = instantiate(
        OmegaConf.create(
            {
                "_target_": "spenn.callback.Status",
                "triggers": ["run_start"],
                "output_path": "status.json",
            }
        )
    )
    resolved = instantiate(
        OmegaConf.create(
            {
                "_target_": "spenn.callback.ResolvedConfigSnapshot",
                "triggers": ["run_start"],
                "output_path": "resolved_config.yaml",
            }
        )
    )
    csv = instantiate(OmegaConf.create({"_target_": "spenn.logging.CSV", "path": "metrics.csv"}))
    jsonl = instantiate(OmegaConf.create({"_target_": "spenn.logging.JSONL", "path": "metrics.jsonl"}))

    assert isinstance(runner, Scaffold)
    assert isinstance(callback, Status)
    assert isinstance(resolved, ResolvedConfigSnapshot)
    assert isinstance(csv, CSV)
    assert isinstance(jsonl, JSONL)
