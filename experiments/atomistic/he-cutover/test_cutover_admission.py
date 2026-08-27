from __future__ import annotations

import sys
from pathlib import Path

import pytest

import admission
import cutover_plan


def _plan(tmp_path: Path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path, plan_id="p")[1]


def test_admission_resolves_python_and_aligns_runtime(tmp_path: Path) -> None:
    dispatches = admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={})
    assert len(dispatches) == 2
    assert all(item.argv[0] == str(Path(sys.executable).resolve()) for item in dispatches)
    assert [item.run_id for item in dispatches] == ["seed-000-chain-00", "seed-000-chain-01"]
    assert {item.runtime for item in dispatches} == {"tpen-cu126"}


def test_admission_rejects_visibility_environment_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "dispatch_specs.jsonl"
    error = None
    try:
        rows = admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={"CUDA_VISIBLE_DEVICES": "0"})
        admission.write_dispatch_specs(output, rows)
    except ValueError as exc:
        error = exc
    assert error is not None
    assert "allocation binding" in str(error)
    assert not output.exists()
