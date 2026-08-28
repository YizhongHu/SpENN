from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.baselines.polaris_submit as submission


def _manifest(tmp_path: Path, code: str = "lapnet") -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        f"""schema: {submission.SCHEMA}
facility: polaris
runtime:
  ferminet_root: {tmp_path / 'ferminet'}
  ferminet_branch: tpen/configurable-seed
  ferminet_commit: f4b1846
rows:
  - id: row-0
    code: {code}
    ansatz: test-ansatz
    system: he
    seed: 7
    steps: 200000
    command: ["{{python}}", "-c", "print('row')"]
  - id: row-1
    code: ferminet
    ansatz: ferminet
    system: h2
    seed: 8
    steps: 200000
    command: ["{{python}}", "-c", "print('row')"]
""",
        encoding="utf-8",
    )
    return path


def test_manifest_reads_rows_without_importing_absent_code(tmp_path: Path) -> None:
    manifest = submission.load_manifest(_manifest(tmp_path, code="not-installed-yet"))
    assert [row["code"] for row in manifest["rows"]] == ["not-installed-yet", "ferminet"]


def test_rows_have_one_distinct_gpu_slot_per_node() -> None:
    slots = {submission.row_gpu_slot(rank) for rank in range(40)}
    assert len(slots) == 40
    assert {gpu for _, gpu in slots} == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("rows", "walltime", "destination", "nodes"),
    [(1, "02:59:59", "small", 10), (40, "03:00:00", "small", 10), (41, "05:00:00", "medium", 25), (1, "06:00:01", "large", 100)],
)
def test_sizing_selects_a_legal_joint_destination(rows: int, walltime: str, destination: str, nodes: int) -> None:
    request = submission.size_production_request(rows, walltime)
    assert (request.destination, request.nodes) == (destination, nodes)
    assert submission.validate_production_request(request.nodes, request.walltime) == destination


def test_ten_nodes_for_24_hours_is_refused_by_our_named_check() -> None:
    with pytest.raises(submission.PolarisSubmissionError, match="joint-routing constraint"):
        submission.validate_production_request(10, "24:00:00")


def test_walltime_beyond_large_is_refused_without_restart() -> None:
    with pytest.raises(submission.PolarisSubmissionError, match="walltime constraint"):
        submission.size_production_request(1, "24:00:01")


def test_template_sets_binding_and_memory_before_python() -> None:
    worker = (Path(__file__).with_name("templates") / "polaris_worker.sh").read_text()
    assert worker.index("CUDA_VISIBLE_DEVICES=") < worker.index("/bin/python")
    assert "3 - PMI_LOCAL_RANK % 4" in worker
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in worker


def test_mpi_is_only_the_template_launcher() -> None:
    template = (Path(__file__).with_name("templates") / "polaris_production.pbs").read_text()
    assert "mpiexec -n @ROW_COUNT@ -ppn 4" in template
    assert "mpi4py" not in template
    assert "DDP" not in template
    assert "#PBS -r n" in template


def test_preflight_rejects_wrong_branch_before_gpu_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = submission.load_manifest(_manifest(tmp_path))
    ferminet_root = Path(manifest["runtime"]["ferminet_root"])
    ferminet_root.mkdir()
    monkeypatch.setattr(submission, "_git", lambda _root, *args: "wrong-branch" if args == ("branch", "--show-current") else "")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with pytest.raises(submission.PolarisSubmissionError, match="branch constraint"):
        submission.run_preflight(manifest, results_root=tmp_path / "results")
    assert not (tmp_path / "results" / "preflight.json").exists()


def test_row_writes_result_terminal_and_rejects_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = submission.load_manifest(_manifest(tmp_path))
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setattr(submission, "_bound_device", lambda: {"cuda_visible_devices": "3", "device_uuid": "GPU-test", "device_kinds": ["NVIDIA A100"], "jax_version": "0"})
    monkeypatch.setattr(submission.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    root = tmp_path / "results"
    (root).mkdir()
    (root / "preflight.json").write_text(json.dumps({"ferminet_branch": "tpen/configurable-seed", "ferminet_commit": "f4b1846"}))
    assert submission.run_row(manifest, 0, results_root=root) == 0
    row_root = root / "rows" / "row-0"
    result = json.loads((row_root / "result.json").read_text())
    assert result["device"]["device_uuid"] == "GPU-test"
    assert json.loads((row_root / "terminal.json").read_text())["status"] == "complete"
    with pytest.raises(submission.PolarisSubmissionError, match="restart detected"):
        submission.run_row(manifest, 0, results_root=root)


def test_rendered_request_contains_no_scheduler_markers(tmp_path: Path) -> None:
    request = submission.size_debug_request(1, "00:05:00")
    output = submission.render_template(
        Path(__file__).with_name("templates") / "polaris_production.pbs",
        request,
        tmp_path / "job.pbs",
    )
    text = output.read_text()
    assert "#PBS -q debug" in text
    assert "select=1:system=polaris" in text
    assert "@" not in text
