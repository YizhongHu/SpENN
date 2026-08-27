from __future__ import annotations

import ast
from pathlib import Path

import yaml
from omegaconf import OmegaConf

import run_train_row


ROOT = Path(__file__).resolve().parents[3]
HEV1_BARE_MODULES = {"layout", "strata", "plan", "driver", "eval", "collect", "canary", "launch"}


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


def test_smoke_training_finds_checkpoint_by_target_when_not_last() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    cfg.callbacks.append({"_target_": "tests.DummyCallback", "every_n_steps": 999})

    run_train_row.configure_smoke_training(cfg, {"max_steps": 25, "n_walkers": 16})

    assert checkpoint.every_n_steps == 25
    assert cfg.callbacks[-1].every_n_steps == 999


def test_production_training_preserves_frozen_checkpoint_cadence() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    run_train_row.configure_training(
        cfg, {"scale": "production", "max_steps": 300000, "n_walkers": 4096}
    )
    assert checkpoint.every_n_steps == 25000


def test_profiles_contain_policy_but_no_filesystem_roots() -> None:
    profiles = Path(__file__).with_name("profiles")
    cannon = yaml.safe_load((profiles / "cannon.yaml").read_text())
    polaris = yaml.safe_load((profiles / "polaris.yaml").read_text())
    assert cannon["runtimes"]["tpen-cu126"]["binding"] == "inherit"
    assert polaris["runtimes"]["tpen-polaris"]["available_accelerators"] == ["0", "1", "2", "3"]
    for path in profiles.iterdir():
        payload = yaml.safe_load(path.read_text())
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                assert not value.startswith("/")


def _cross_study_violations(study: Path) -> set[str]:
    violations = set()
    for path in study.glob("*.py"):
        if path.name == "hev1.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name.split(".", 1)[0] in HEV1_BARE_MODULES for alias in node.names):
                violations.add(f"{path.name}:bare-import")
                continue
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".", 1)[0] in HEV1_BARE_MODULES:
                violations.add(f"{path.name}:bare-from-import")
                continue
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                node.func.attr in {"insert", "append", "extend"}
                and isinstance(owner, ast.Attribute)
                and owner.attr == "path"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "sys"
            ):
                violations.add(f"{path.name}:sys-path-{node.func.attr}")
    return violations


def test_hev1_is_the_only_cross_study_accessor_and_import_gateway() -> None:
    study = Path(__file__).resolve().parent
    assert _cross_study_violations(study) == set()
    assert not (study / "configs").exists()
    assert not ({path.stem for path in study.glob("*.py")} & HEV1_BARE_MODULES)
