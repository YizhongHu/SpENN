"""T10 deterministic tiny-VMC smoke (MIG-TPEN-000 §5, slice d).

Runs the same real stack as the pair smoke fixture — ``run_from_config``
through the Train runner -> TPENWaveFunction (TPENStack) ->
MetropolisSampler -> Hooke Hamiltonian -> VMCTrainer with the full callback
battery — on CPU/float64 with fixed seeds; the scale is already reduced
(16 walkers, 2 trainer steps), not the stage stack. The T10 contract is
asserted explicitly on the logged artifacts:

- finite energy trajectory with one record per trainer step and no NaN in
  any train-namespace numeric metric;
- ``RuntimeEquivariance`` records present and ``passed`` at every step;
- artifacts logged in UTC (test-logging convention);
- bitwise-identical train metrics across two seeded reruns
  (``runtime.seed`` seeds process RNGs; the sampler re-seeds at sample
  start, so a rerun is exactly reproducible on CPU).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from omegaconf import OmegaConf

from spenn.run import run_from_config

CONFIG = Path(__file__).resolve().parents[1] / "artifacts" / "hooke" / "pair_train.yaml"


def _run(root: Path) -> Path:
    cfg = OmegaConf.load(CONFIG)
    cfg.run.root = str(root)
    exit_code = run_from_config(cfg, config_path=str(CONFIG), command="pytest")
    assert exit_code == 0
    run_dirs = list(root.glob("hooke_pair_smoke/*/*"))
    assert len(run_dirs) == 1, f"expected one run dir, found {run_dirs}"
    return run_dirs[0]


def _records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _namespace_metrics(records: list[dict], namespace: str) -> dict[tuple[int, str], object]:
    """Return {(step, key): value} for every metric in `namespace`.

    A ``(step, key)`` is logged at most once per run for these namespaces, so a
    duplicate is a regression (e.g. a stage emitting a metric twice per step);
    this helper rejects duplicates rather than silently overwriting, which keeps
    the per-step record-count and rerun-equality assertions honest.
    """

    table: dict[tuple[int, str], object] = {}
    for record in records:
        if record.get("namespace") != namespace:
            continue
        step = int(record["step"])
        for key, value in record["metrics"].items():
            assert (step, key) not in table, f"duplicate metric {namespace}/{key} at step {step}"
            table[(step, key)] = value
    return table


def test_tiny_vmc_energy_trajectory_is_finite_and_complete(tmp_path) -> None:
    run_dir = _run(tmp_path)
    records = _records(run_dir)

    train = _namespace_metrics(records, "train")
    max_steps = int(OmegaConf.load(CONFIG).trainer.max_steps)
    energy_steps = sorted(step for (step, key) in train if key == "energy")
    assert energy_steps == list(range(max_steps)), f"energy trajectory incomplete: {energy_steps}"
    for (step, key), value in train.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            assert math.isfinite(value), f"non-finite train metric {key} at step {step}: {value}"


def test_tiny_vmc_runtime_equivariance_records_are_clean(tmp_path) -> None:
    run_dir = _run(tmp_path)
    records = _records(run_dir)
    max_steps = int(OmegaConf.load(CONFIG).trainer.max_steps)

    for namespace in ("checks/equivariance/full_model", "checks/equivariance/trace"):
        table = _namespace_metrics(records, namespace)
        steps = sorted(step for (step, key) in table if key == "passed")
        assert steps == list(range(max_steps)), f"{namespace} missing steps: {steps}"
        for step in steps:
            assert table[(step, "passed")] is True, f"{namespace} failed at step {step}"


def test_tiny_vmc_artifacts_are_logged_utc(tmp_path) -> None:
    run_dir = _run(tmp_path)
    status = json.loads((run_dir / "status.json").read_text())

    assert status["status"] == "completed"
    assert status["timezone"] == "UTC"
    for key in ("start_time", "end_time"):
        stamp = datetime.fromisoformat(status[key])
        assert stamp.utcoffset() == timedelta(0), f"{key} not logged in UTC: {status[key]}"


def test_tiny_vmc_is_deterministic_across_reruns(tmp_path) -> None:
    first = _records(_run(tmp_path / "first"))
    second = _records(_run(tmp_path / "second"))

    for namespace in ("train", "train/sampler"):
        lhs = _namespace_metrics(first, namespace)
        rhs = _namespace_metrics(second, namespace)
        assert lhs.keys() == rhs.keys(), f"{namespace} metric keys differ between reruns"
        mismatches = {
            key: (lhs[key], rhs[key])
            for key in lhs
            if lhs[key] != rhs[key]
        }
        assert not mismatches, f"{namespace} metrics differ between seeded reruns: {mismatches}"
