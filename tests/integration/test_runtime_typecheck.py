"""Integration tests for runtime type-check policy and runner behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from tests.helpers import ROOT


def test_typecheck_policy_documents_required_modes_and_commands() -> None:
    policy = OmegaConf.load(ROOT / "configs" / "typecheck.yaml")

    assert policy.package == "spenn"
    assert dict(policy.modes) == {
        "tests": "opt_in",
        "scripts": "opt_in",
        "training": "opt_in_debug_only",
        "deployment": "disabled",
    }
    assert "--typeguard-packages=spenn" in policy.commands.tests
    assert "scripts/typechecked.py" in policy.commands.debug_equivariance
    assert "scripts/typechecked.py" in policy.commands.train_smoke
    assert "experiments/hooke/run_exact.py" in policy.commands.hooke_exact_debug


def test_typechecked_runner_executes_script_after_installing_hook(tmp_path: Path) -> None:
    script = tmp_path / "probe.py"
    script.write_text(
        "from spenn.data import Par\n"
        "print(f'partition:{Par(\"H\").order}')\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "typechecked.py"), str(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "partition:1" in result.stdout
