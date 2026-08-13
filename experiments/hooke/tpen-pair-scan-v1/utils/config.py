"""Config metadata and choice-library loading helpers for staged studies.

This module owns everything about *where a study's run config comes from*:
the grid-attempt snapshot filenames, and the composition of a base config with
the shared choice-library fragments it is not self-contained without.

Why composition lives here rather than in ``plan.py``
-----------------------------------------------------
``tpen.run.load_config`` reads exactly one YAML file (``OmegaConf.load``) and
has no include or defaults-list mechanism, so a base config that references a
choice library it does not itself define is only runnable after an explicit
merge. ``experiments/hooke/tpen-pair-scan-v1/configs/{train,eval}.yaml``
deliberately omit ``choices.basis``, which lives in one table at
``experiments/hooke/choices/basis_levels.yaml`` so that "the scan ran the level
we tested" is checkable rather than a claim about two files that were once
identical.

The consequence is a launcher obligation, and it is the single most dangerous
one in this study: a planner that compiles ``run.py --config configs/train.yaml``
without merging the library either dies on a dangling ``${choices.basis...}``
interpolation or -- far worse -- resolves a partial config. So the merge is
performed here, asserted here, and the planner points every compiled command at
the merged snapshot it wrote, never at the un-merged source file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

DEFAULT_CONFIG_SNAPSHOTS = {
    "train": "train_config.yaml",
    "validation": "validation_config.yaml",
}


def config_snapshot_names(configured: Any | None = None) -> dict[str, str]:
    """Return stage -> grid-attempt config snapshot filename."""

    source = DEFAULT_CONFIG_SNAPSHOTS if configured is None else configured
    if not isinstance(source, dict):
        raise ValueError("config_snapshots must be a mapping")
    snapshots = {str(stage): str(filename) for stage, filename in source.items()}
    for stage, filename in snapshots.items():
        if not filename or Path(filename).name != filename:
            raise ValueError(f"config_snapshots.{stage} must be a plain filename")
    return snapshots


# ---------------------------------------------------------------------------
# Choice libraries: the shared config fragments a base config needs merged in
# ---------------------------------------------------------------------------
def choice_library_specs(configured: Any | None = None) -> list[dict[str, Any]]:
    """Return normalized choice-library specs declared by a grid config.

    Parameters
    ----------
    configured
        The grid config's ``choice_libraries`` value. Each entry is either a
        bare path string or a mapping with ``path`` and an optional ``provides``
        listing the config paths that fragment is expected to define.

    Returns
    -------
    list of dict
        One ``{"path": str, "provides": list[str]}`` entry per library, in the
        declared merge order.

    Raises
    ------
    ValueError
        When the declaration is not a sequence, an entry has no usable ``path``,
        or ``provides`` is not a string or sequence of strings.
    """

    if configured is None:
        return []
    if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
        raise ValueError("choice_libraries must be a sequence of paths or mappings")
    specs: list[dict[str, Any]] = []
    for index, entry in enumerate(configured):
        if isinstance(entry, str):
            path, provides = entry, []
        elif isinstance(entry, dict):
            path = str(entry.get("path", "")).strip()
            raw_provides = entry.get("provides") or []
            if isinstance(raw_provides, str):
                provides = [raw_provides]
            elif isinstance(raw_provides, Sequence):
                provides = [str(item) for item in raw_provides]
            else:
                raise ValueError(f"choice_libraries[{index}].provides must be a string or sequence")
        else:
            raise ValueError(f"choice_libraries[{index}] must be a path string or a mapping")
        if not str(path).strip():
            raise ValueError(f"choice_libraries[{index}] requires a non-empty path")
        specs.append({"path": str(path).strip(), "provides": [item for item in provides if item]})
    return specs


def choice_library_provides(specs: Sequence[dict[str, Any]]) -> list[str]:
    """Return the deduplicated config paths declared by choice-library specs."""

    provided: list[str] = []
    for spec in specs:
        for path in spec.get("provides", ()):  # declared order is the merge order
            if path not in provided:
                provided.append(str(path))
    return provided


def resolve_library_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Return an existing choice-library path, anchored at ``repo_root``.

    Raises
    ------
    FileNotFoundError
        When the declared fragment does not exist. Declaring a library that is
        not on disk must fail before any command is compiled, not at run time
        inside a Slurm array task.
    """

    candidate = Path(path)
    if not candidate.is_absolute() and repo_root is not None:
        candidate = Path(repo_root) / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"choice library does not exist: {candidate}")
    return candidate


def merge_choice_libraries(
    config: Any,
    specs: Sequence[dict[str, Any]],
    *,
    repo_root: str | Path | None = None,
) -> Any:
    """Return ``config`` with each declared choice-library fragment merged in.

    Merge order is declaration order, and the fragments are merged *onto* the
    base config, so a later library wins a key collision with an earlier one.
    ``OmegaConf.merge`` preserves interpolations, so the result is still a valid
    single-file config for the ``run.py`` entrypoint.
    """

    merged = config
    for spec in specs:
        library_path = resolve_library_path(spec["path"], repo_root=repo_root)
        merged = OmegaConf.merge(merged, OmegaConf.load(library_path))
    return merged


def require_choice_paths(config: Any, paths: Sequence[str], *, context: str) -> None:
    """Fail loudly when a required choice path is absent or empty.

    This is the assertion that turns "the launcher forgot the merge" from a
    silent partial resolution into an error naming the missing path. It is
    checked on the *composed* config, before any command is compiled.

    Raises
    ------
    ValueError
        When a required path is missing, is not a mapping, or is an empty
        mapping. An empty mapping is rejected because a merge against a
        fragment that defines only an empty ``choices.basis:`` key would
        otherwise look like a successful merge.
    """

    missing: list[str] = []
    empty: list[str] = []
    for path in paths:
        selected = OmegaConf.select(config, str(path))
        if selected is None:
            missing.append(str(path))
            continue
        if not hasattr(selected, "keys"):
            missing.append(str(path))
            continue
        if len(selected.keys()) == 0:
            empty.append(str(path))
    if missing:
        raise ValueError(
            f"{context} is missing required choice path(s) {', '.join(sorted(missing))}; "
            "declare the fragment that defines them under the grid's choice_libraries"
        )
    if empty:
        raise ValueError(f"{context} defines empty choice path(s) {', '.join(sorted(empty))}")


def load_composed_config(
    config_path: str | Path,
    specs: Sequence[dict[str, Any]],
    *,
    required_paths: Sequence[str] = (),
    repo_root: str | Path | None = None,
    context: str | None = None,
) -> Any:
    """Load one study config, merge its choice libraries, and check the result.

    Parameters
    ----------
    config_path
        Base study config, which is not required to be self-contained.
    specs
        Normalized choice-library specs from :func:`choice_library_specs`.
    required_paths
        Config paths that must resolve to non-empty mappings after merging.
    repo_root
        Anchor for relative library paths.
    context
        Label used in error messages; defaults to ``config_path``.

    Returns
    -------
    omegaconf.DictConfig
        The composed config, with interpolations left unresolved.
    """

    composed = merge_choice_libraries(
        OmegaConf.load(config_path), specs, repo_root=repo_root
    )
    require_choice_paths(composed, required_paths, context=context or str(config_path))
    return composed
