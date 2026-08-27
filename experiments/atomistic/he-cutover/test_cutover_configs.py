from __future__ import annotations

from pathlib import Path

import yaml
from omegaconf import OmegaConf

import run_train_row


ROOT = Path(__file__).resolve().parents[2]


def _differences(before, after, prefix=""):
    differences = set()
    if isinstance(before, dict) and isinstance(after, dict):
        for key in before.keys() | after.keys():
            differences |= _differences(before.get(key), after.get(key), f"{prefix}.{key}".strip("."))
    elif isinstance(before, list) and isinstance(after, list):
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            label = left.get("_target_", str(index)) if isinstance(left, dict) else str(index)
            differences |= _differences(left, right, f"{prefix}.{label}")
    elif before != after:
        differences.add(prefix)
    return differences


def test_smoke_training_diff_is_exactly_three_scale_fields() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    before = OmegaConf.to_container(cfg, resolve=False)
    run_train_row.configure_smoke_training(cfg, {"max_steps": 25, "n_walkers": 16})
    after = OmegaConf.to_container(cfg, resolve=False)
    assert _differences(before, after) == {"trainer.max_steps", "sampler.n_walkers", "callbacks.tpen.callback.Checkpoint.every_n_steps"}


def test_profiles_contain_policy_but_no_filesystem_roots() -> None:
    profiles = Path(__file__).with_name("profiles")
    cannon = yaml.safe_load((profiles / "cannon.yaml").read_text())
    polaris = yaml.safe_load((profiles / "polaris.yaml").read_text())
    assert cannon["runtimes"]["tpen-cu126"]["binding"] == "inherit"
    assert polaris["runtimes"]["tpen-polaris"]["available_accelerators"] == ["0", "1", "2", "3"]
    assert all("/" not in path.read_text() for path in profiles.iterdir())
