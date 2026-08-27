from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


TEMPLATES = Path(__file__).with_name("templates")
README = TEMPLATES.parent / "README.md"
REPO_ROOT = README.parents[2]


def _runbook_block(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split(heading, 1)[1]
    return re.search(r"```bash\n(.*?)\n```", section, re.DOTALL).group(1)


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
    assert 'export PYTHONPATH="$TPEN_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert '"$TPEN_UV" sync' in text
    assert '"$TPEN_UV" run' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_polaris_directives_and_runtime_policy() -> None:
    text = (TEMPLATES / "polaris_smoke.pbs").read_text(encoding="utf-8")
    for required in ("-A HetRxnEnergy", "-q debug", "select=1", "walltime=00:50:00", "filesystems=home:eagle", "-k doe", "LD_LIBRARY_PATH", "-m uv sync --project", "--inexact --locked --extra parsl"):
        assert required in text
    assert 'export PYTHONPATH="$TPEN_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_operator_runbook_uses_facility_owned_uv_entrypoints() -> None:
    text = README.read_text(encoding="utf-8")
    assert 'case "$TPEN_UV" in /*)' in text
    assert 'export PYTHONPATH="$TPEN_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert '"$TPEN_UV" run --project "$TPEN_CHECKOUT" --locked --extra cpu' in text
    assert '"$TPEN_PYBIN" -m uv sync --project "$TPEN_CHECKOUT" --inexact --locked --extra parsl' in text
    assert '"$TPEN_UV_ENV/bin/python" "$TPEN_CHECKOUT/experiments/atomistic/he-cutover/cutover_plan.py" --facility polaris' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_documented_cannon_plan_runs_outside_checkout(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
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
    subprocess.run(["bash", "-c", _runbook_block("Cannon planning (login node):")], cwd=outside, env=env, check=True)
    plan = results / "00_plan" / "subprocess-test"
    assert (plan / "rows.csv").is_file()
    assert (plan / "02_train" / "tasks.jsonl").is_file()
    assert (plan / "03_eval" / "tasks.jsonl").is_file()
