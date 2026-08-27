from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


TEMPLATES = Path(__file__).with_name("templates")
README = TEMPLATES.parent / "README.md"
REPO_ROOT = README.parents[3]


def _runbook_block(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split(heading, 1)[1]
    return re.search(r"```bash\n(.*?)\n```", section, re.DOTALL).group(1)


def _runbook_command(heading: str, script: str) -> str:
    lines = [line for line in _runbook_block(heading).splitlines() if script in line]
    assert len(lines) == 1
    return lines[0]


def test_templates_have_no_operator_roots() -> None:
    for path in TEMPLATES.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "/n/netscratch" not in text
        assert "/eagle/" not in text
        assert "/home/" not in text
        assert "${TPEN_CHECKOUT:?}" in text
        assert "${TPEN_RESULTS_ROOT:?}" in text


def test_cannon_directives_and_runtime_policy() -> None:
    text = (TEMPLATES / "cannon_smoke.sbatch").read_text(encoding="utf-8")
    for required in ("gpu_test", "a100_mig", "02:00:00", "--no-requeue", "UV_CACHE_DIR", "--extra cu126", "--extra parsl"):
        assert required in text
    assert 'case "$TPEN_UV" in /*)' in text
    assert 'PYTHONPATH="${TPEN_CHECKOUT:?}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV" run' in text
    assert '"$TPEN_UV" sync' in text
    assert '"$TPEN_UV" run' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_polaris_directives_and_runtime_policy() -> None:
    text = (TEMPLATES / "polaris_smoke.pbs").read_text(encoding="utf-8")
    for required in ("-A HetRxnEnergy", "-q debug", "select=1", "walltime=00:50:00", "filesystems=home:eagle", "LD_LIBRARY_PATH"):
        assert required in text
    assert 'PYTHONPATH="${TPEN_CHECKOUT:?}${PYTHONPATH:+:$PYTHONPATH}" "$TPEN_UV_ENV/bin/python"' in text
    assert ': "${TPEN_CUDA13_LIB:?}"' in text
    assert ': "${TPEN_CUDA129_LIB:?}"' in text
    assert (
        'export LD_LIBRARY_PATH="$TPEN_LIBSHIM:$TPEN_CUDA13_LIB:'
        '$TPEN_CUDA129_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
        in text
    )
    assert "/soft/compilers/cudatoolkit/" not in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_templates_put_scheduler_logs_under_guarded_results_root() -> None:
    cannon = (TEMPLATES / "cannon_smoke.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --output=/dev/null" in cannon
    assert "#SBATCH --error=/dev/null" in cannon
    assert 'exec >"${TPEN_RESULTS_ROOT:?}/scheduler/slurm-${SLURM_JOB_ID:?}.out"' in cannon
    assert '2>"${TPEN_RESULTS_ROOT:?}/scheduler/slurm-${SLURM_JOB_ID:?}.err"' in cannon

    polaris = (TEMPLATES / "polaris_smoke.pbs").read_text(encoding="utf-8")
    assert "#PBS -o /dev/null" in polaris
    assert "#PBS -e /dev/null" in polaris
    assert "#PBS -k doe" not in polaris
    assert 'exec >"${TPEN_RESULTS_ROOT:?}/scheduler/pbs-${PBS_JOBID:?}.out"' in polaris
    assert '2>"${TPEN_RESULTS_ROOT:?}/scheduler/pbs-${PBS_JOBID:?}.err"' in polaris


def test_polaris_template_only_reads_operator_overlay() -> None:
    text = (TEMPLATES / "polaris_smoke.pbs").read_text(encoding="utf-8")
    assert "UV_PROJECT_ENVIRONMENT" not in text
    assert "uv sync" not in text
    assert 'export PATH="$TPEN_UV_ENV/bin${PATH:+:$PATH}"' in text
    assert '"$TPEN_UV_ENV/bin/python"' in text


def test_operator_runbook_uses_facility_owned_uv_entrypoints() -> None:
    text = README.read_text(encoding="utf-8")
    assert "development checkout" in text
    assert "production checkout" in text
    assert "TPEN-main" not in text
    assert "TPEN-devtest" not in text
    assert 'case "$TPEN_UV" in /*)' in text
    assert text.count('PYTHONPATH="${TPEN_CHECKOUT:?checkout root required}${PYTHONPATH:+:$PYTHONPATH}"') == 2
    assert '"$TPEN_UV" run --project "$TPEN_CHECKOUT" --locked --extra cpu' in text
    assert "A syncing `uv` must never target this overlay" in text
    assert 'export PATH="$TPEN_UV_ENV/bin${PATH:+:$PATH}"' in text
    assert '"$TPEN_UV_ENV/bin/python"' in text
    assert '"$TPEN_UV_ENV/bin/python" "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/cutover_plan.py" --facility polaris' in text
    assert "TPEN_CUDA13_LIB" in text
    assert "TPEN_CUDA129_LIB" in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_documented_cannon_plan_runs_outside_checkout(tmp_path: Path) -> None:
    uv = os.environ.get("TPEN_UV") or shutil.which("uv")
    assert uv is not None
    assert Path(uv).is_absolute()
    results = tmp_path / "results"
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {
        **os.environ,
        "TPEN_CHECKOUT": str(REPO_ROOT),
        "TPEN_UV": uv,
        "TPEN_RESULTS_ROOT": str(results),
        "TPEN_PLAN_ATTEMPT_ID": "subprocess-test",
    }
    env.pop("PYTHONPATH", None)
    command = _runbook_command("Cannon planning (login node):", "cutover_plan.py")
    subprocess.run(["bash", "-c", command], cwd=outside, env=env, check=True)
    plan = results / "00_plan" / "subprocess-test"
    assert (plan / "rows.csv").is_file()
    assert (plan / "02_train" / "tasks.jsonl").is_file()
    assert (plan / "03_eval" / "tasks.jsonl").is_file()
