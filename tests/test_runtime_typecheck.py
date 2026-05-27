"""Tests for runtime type-check policy and internal shape asserts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from spenn.data import BranchDict, FeatureDict, MessageDict, Par, TensorProductDict
from spenn.data.batch import ElectronBatch
from spenn.nn.cusp import ElectronElectronCusp
from spenn.nn.encoding import ElectronPairEncoder
from spenn.nn.spechtmp.message_head import _project_product
from spenn.nn.spechtmp.update_head import _project_branch


ROOT = Path(__file__).resolve().parent.parent


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


def test_encoder_shape_asserts_catch_bad_tuple_mlp_output() -> None:
    class BadTupleMLP(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    encoder = ElectronPairEncoder(channels=[0, 2, 0])
    positions = torch.zeros(1, 3, 2)

    with pytest.raises(AssertionError):
        encoder.apply_tuple_mlp(BadTupleMLP(), positions)


def test_spechtmp_projection_asserts_catch_bad_linear_output_shape() -> None:
    class BadLinear(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.new_zeros(*inputs.shape[:-1], 2, 1)

    product = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)
    branch = torch.zeros(1, 1, 1, 2, 2, 2, 2, 1, 1)

    with pytest.raises(AssertionError):
        _project_product(BadLinear(), Par("S"), Par("H"), Par("H"), product)
    with pytest.raises(AssertionError):
        _project_branch(BadLinear(), Par("S"), Par("S"), branch)


def test_container_validation_asserts_still_allow_valid_active_shapes() -> None:
    features = FeatureDict({Par("H"): torch.zeros(1, 2, 3, 1, 1)})
    messages = MessageDict({Par("S"): torch.zeros(1, 2, 3, 3, 1, 1)})
    products = TensorProductDict({Par("S"): {Par("H"): {Par("H"): torch.zeros(1, 2, 1, 3, 3, 3, 3, 1, 1)}}})
    branches = BranchDict({Par("S"): {Par("S"): torch.zeros(1, 2, 1, 3, 3, 3, 3, 1, 1)}})

    features.validate(batch_size=1, n_electrons=3)
    messages.validate(batch_size=1, n_electrons=3)
    products.validate(batch_size=1, n_electrons=3)
    branches.validate(batch_size=1, n_electrons=3)


def test_cusp_shape_asserts_preserve_batch_shape() -> None:
    positions = torch.zeros(4, 3, 2, dtype=torch.float64)
    output = ElectronElectronCusp()(ElectronBatch(positions))

    assert output.shape == (4,)
