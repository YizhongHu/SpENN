from __future__ import annotations

from pathlib import Path


TEMPLATES = Path(__file__).with_name("templates")
README = TEMPLATES.parent / "README.md"


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
    assert '"$TPEN_UV" sync' in text
    assert '"$TPEN_UV" run' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_polaris_directives_and_runtime_policy() -> None:
    text = (TEMPLATES / "polaris_smoke.pbs").read_text(encoding="utf-8")
    for required in ("-A HetRxnEnergy", "-q debug", "select=1", "walltime=00:50:00", "filesystems=home:eagle", "-k doe", "LD_LIBRARY_PATH", "-m uv sync --inexact --locked --extra parsl"):
        assert required in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())


def test_operator_runbook_uses_facility_owned_uv_entrypoints() -> None:
    text = README.read_text(encoding="utf-8")
    assert 'case "$TPEN_UV" in /*)' in text
    assert '"$TPEN_UV" run --locked --extra cpu' in text
    assert '"$TPEN_PYBIN" -m uv sync --inexact --locked --extra parsl' in text
    assert '"$TPEN_UV_ENV/bin/python" experiments/atomistic/he-cutover/cutover_plan.py --facility polaris' in text
    assert not any(line.lstrip().startswith("uv ") for line in text.splitlines())
