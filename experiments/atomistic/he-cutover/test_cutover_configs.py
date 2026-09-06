from __future__ import annotations

import ast
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

run_train_row = sibling(__file__, 'run_train_row')


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
    assert _differences(before, after) == {"trainer.max_steps", "sampler.n_walkers", "callbacks.tpen.callback.Checkpoint.schedule.every_n"}


def test_smoke_training_finds_checkpoint_by_target_when_not_last() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    cfg.callbacks.append({"_target_": "tests.DummyCallback", "every_n_steps": 999})

    run_train_row.configure_smoke_training(cfg, {"max_steps": 25, "n_walkers": 16})

    assert checkpoint.schedule.every_n == 25
    assert cfg.callbacks[-1].every_n_steps == 999


def test_production_training_preserves_frozen_checkpoint_cadence() -> None:
    path = ROOT / "experiments/atomistic/he-v1/configs/train.yaml"
    cfg = OmegaConf.load(path)
    checkpoint = next(callback for callback in cfg.callbacks if callback.get("_target_") == "tpen.callback.Checkpoint")
    run_train_row.configure_training(
        cfg, {"scale": "production", "max_steps": 300000, "n_walkers": 4096}
    )
    assert checkpoint.schedule.every_n == 25000


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


def _sys_aliases(tree: ast.AST) -> set[str]:
    """Return the names bound to the ``sys`` module in one parsed file.

    The sys.path rule below used to match ``owner.value.id == "sys"``, which
    matches a SPELLING rather than resolving a BINDING: ``import sys as _s``
    then ``_s.path.insert(...)`` sailed through. That is not hypothetical --
    the study-scoped sibling bootstrap imports ``sys`` under an alias, so
    ordinary code in this repository already evades the literal match.
    """

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    aliases.add(alias.asname or "sys")
    return aliases


def _cross_study_violations(study: Path) -> set[str]:
    violations = set()
    for path in study.glob("*.py"):
        if path.name == "hev1.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        sys_aliases = _sys_aliases(tree)
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
                and owner.value.id in sys_aliases
            ):
                violations.add(f"{path.name}:sys-path-{node.func.attr}")
    return violations


def test_hev1_is_the_only_cross_study_accessor_and_import_gateway() -> None:
    study = Path(__file__).resolve().parent
    assert _cross_study_violations(study) == set()
    assert not (study / "configs").exists()
    assert not ({path.stem for path in study.glob("*.py")} & HEV1_BARE_MODULES)


def test_sys_path_rule_resolves_aliases_not_spellings(tmp_path: Path) -> None:
    """An aliased ``sys`` must not evade the sys.path rule.

    Matching ``owner.value.id == "sys"`` matched a spelling. Ordinary code in
    this repository imports ``sys`` under an alias, so the literal match was
    already evadable by practice rather than only in principle -- the study
    bootstrap is the existence proof.
    """

    (tmp_path / "aliased.py").write_text(
        "import sys as _s\nfrom pathlib import Path\n_s.path.insert(0, str(Path('/tmp')))\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == {"aliased.py:sys-path-insert"}


def test_sys_path_rule_ignores_an_unrelated_path_attribute(tmp_path: Path) -> None:
    """``.path.insert`` on something that is not the sys module is not a violation."""

    (tmp_path / "unrelated.py").write_text(
        "class Cfg:\n    def __init__(self):\n        self.path = []\n"
        "cfg = Cfg()\ncfg.path.insert(0, 'x')\n",
        encoding="utf-8",
    )

    assert _cross_study_violations(tmp_path) == set()
