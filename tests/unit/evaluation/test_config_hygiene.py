"""Config hygiene tests for the evaluation-stack migration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_yaml_configs_do_not_reintroduce_evaluation_phase_or_required_keys() -> None:
    offenders: list[str] = []
    for path in _yaml_config_paths():
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        for node_path, node in _walk_mappings(data):
            if node_path[-1:] == ("evaluation",) and "phase" in node:
                offenders.append(_format_offender(path, (*node_path, "phase")))
            if node_path[-1:] == ("evaluator",) and "phase" in node:
                offenders.append(_format_offender(path, (*node_path, "phase")))
            if len(node_path) >= 2 and node_path[-2] == "evaluation_tasks":
                for key in ("required", "phase"):
                    if key in node:
                        offenders.append(_format_offender(path, (*node_path, key)))
                if "output_dir" not in node:
                    offenders.append(_format_offender(path, (*node_path, "output_dir")))

    assert offenders == [], "stale or incomplete evaluation task config keys found:\n" + "\n".join(offenders)


def _yaml_config_paths() -> Iterator[Path]:
    roots = (
        REPO_ROOT / "experiments" / "hooke",
        REPO_ROOT / "tests" / "fixtures",
        REPO_ROOT / "tests" / "integration" / "artifacts",
    )
    for root in roots:
        if root.exists():
            yield from sorted(root.rglob("*.yaml"))
            yield from sorted(root.rglob("*.yml"))


def _walk_mappings(data: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(data, Mapping):
        yield path, data
        for key, value in data.items():
            yield from _walk_mappings(value, (*path, str(key)))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from _walk_mappings(value, (*path, str(index)))


def _format_offender(path: Path, keys: tuple[str, ...]) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{'.'.join(keys)}"
