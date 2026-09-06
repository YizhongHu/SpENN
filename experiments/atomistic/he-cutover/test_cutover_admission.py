from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Siblings are loaded study-scoped, not by bare import: experiments/ has
# several same-named modules and the first study loaded would otherwise own
# the bare name for every study after it. See experiments/toolkit/study_imports.py.
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

_TPEN_REPO_ROOT = _TpenPath(__file__).resolve().parents[3]
if str(_TPEN_REPO_ROOT) not in _tpen_sys.path:
    _tpen_sys.path.insert(0, str(_TPEN_REPO_ROOT))

from experiments.toolkit.study_imports import sibling  # noqa: E402

admission = sibling(__file__, 'admission')
cutover_plan = sibling(__file__, 'cutover_plan')


def _plan(tmp_path: Path):
    grid = cutover_plan.load_grid(Path(__file__).with_name("smoke_grid.yaml"))
    return cutover_plan.build_plans(grid, facility="cannon", results_root=tmp_path, plan_id="p")[1]


def test_admission_preserves_unresolved_python_and_aligns_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = tmp_path / "venv"
    executable = prefix / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", str(executable))
    dispatches = admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={})
    assert len(dispatches) == 2
    assert all(item.argv[0] == str(executable) for item in dispatches)
    assert all(item.argv[0] != str(executable.resolve()) for item in dispatches)
    assert [item.run_id for item in dispatches] == ["seed-000-chain-00", "seed-000-chain-01"]
    assert {item.runtime for item in dispatches} == {"tpen-cu126"}


def test_admission_rejects_interpreter_outside_selected_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    with pytest.raises(ValueError, match="outside sys.prefix"):
        admission.admit_plan(_plan(tmp_path), admission_id="a", cwd=tmp_path, environment={}, python=str(Path(sys.executable).resolve()))


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
