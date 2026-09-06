"""Study-scoped sibling module loading.

This module owns one concept: **how a module inside an ``experiments/`` study
directory reaches another module in the same study**.

Why this exists
---------------
Study directories are not packages.  Each one puts *its own* directory on
``sys.path`` -- explicitly via ``sys.path.insert(0, STUDY_DIR)``, and implicitly
whenever pytest collects a test from it in prepend import mode.  A sibling
reached with a bare ``import plan`` is therefore looked up as a **top-level**
module and cached in ``sys.modules`` under the bare key ``"plan"``.

``experiments/`` currently has three ``plan.py``, three ``collect.py``, four
``launch.py`` and two ``utils/`` packages.  Under a bare import the first study
loaded wins the bare key, and every study loaded afterwards silently receives
the first study's module.  The loud form of this is an ``AttributeError`` naming
a function that plainly exists; the dangerous form is two studies that happen to
share an API surface, where there is no exception at all -- just the wrong
module, the wrong constants, and a green test suite.

Why the fix has to live in the *owning* module
----------------------------------------------
A consumer that loads a study module by path under a unique name fixes only the
name **it** binds.  It cannot fix the names that the loaded module itself binds:
``collect.py`` still executes ``import plan``, which still resolves through the
shared bare key.  So a consumer cannot protect itself, and the change belongs in
every module that reaches a sibling.

Why not relative imports in real packages
-----------------------------------------
Two independent blockers, either sufficient on its own:

1. These modules are **entrypoints executed directly** (``python plan.py``);
   18 of them carry an ``if __name__ == "__main__"`` block.  A directly executed
   file runs as ``__main__`` with no parent package, so a relative import raises
   ``ImportError: attempted relative import with no known parent package``.
2. Three study directories are named ``he-v1``, ``tpen-pair-scan-v1`` and
   ``tpen-pair-v1``.  Hyphens are not Python identifiers, so the directories
   would have to be renamed -- and those names appear in durable run paths and
   recorded receipts.

Loading by path under a study-unique key keeps every filename and every
documented ``python <stage>.py`` invocation working unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: Root key that every study-scoped module is registered beneath.  Chosen to be
#: a name no study can supply, so it can never itself be captured.
_NAMESPACE = "_tpen_study"

#: ``experiments/`` -- the boundary a study directory is named relative to, so
#: that two studies with the same directory *basename* under different parents
#: still get distinct keys.
_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def study_slug(study_dir: Path) -> str:
    """Return the unique ``sys.modules`` infix for one study directory.

    Parameters
    ----------
    study_dir : Path
        Absolute path to the study directory.

    Returns
    -------
    str
        The study's path relative to ``experiments/`` with every character that
        is not valid in an identifier replaced by ``_``.  Derived from the full
        relative path rather than the basename, so ``a/study`` and ``b/study``
        cannot collide.
    """

    try:
        relative = study_dir.resolve().relative_to(_EXPERIMENTS_ROOT)
    except ValueError:  # a study outside experiments/ -- fall back to the path
        relative = Path(*study_dir.resolve().parts[1:])
    return "".join(c if c.isalnum() else "_" for c in str(relative))


def _ensure_namespace_parents(slug: str, study_dir: Path) -> None:
    """Register the ``_tpen_study`` and ``_tpen_study.<slug>`` parent packages.

    Submodule imports and relative imports both walk the parent chain, so the
    parents must exist in ``sys.modules`` before any leaf is executed.
    """

    if _NAMESPACE not in sys.modules:
        root = ModuleType(_NAMESPACE)
        root.__path__ = []  # namespace-style: it owns no directory of its own
        sys.modules[_NAMESPACE] = root

    study_key = f"{_NAMESPACE}.{slug}"
    if study_key not in sys.modules:
        holder = ModuleType(study_key)
        # __path__ points at the study directory, so ``<study_key>.<name>``
        # resolves siblings through ordinary package machinery.
        holder.__path__ = [str(study_dir)]
        sys.modules[study_key] = holder
        setattr(sys.modules[_NAMESPACE], slug, holder)


def _load_leaf(key: str, path: Path, is_package: bool) -> ModuleType:
    """Import one module from ``path`` and register it in ``sys.modules`` as ``key``."""

    cached = sys.modules.get(key)
    if cached is not None:
        return cached

    if is_package:
        spec = importlib.util.spec_from_file_location(
            key, path / "__init__.py", submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load study module {key!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec so that a cycle between two siblings terminates,
    # exactly as the normal import system does it.
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


def load_study_module(study_dir: Path, dotted: str) -> ModuleType:
    """Return a module from ``study_dir``, loaded by path under a unique key.

    The primitive behind :func:`sibling`.  Use it directly only when the study
    being reached is NOT the caller's own -- a deliberate cross-study boundary.

    Parameters
    ----------
    study_dir : Path
        The study directory that owns ``dotted``.
    dotted : str
        Module to load, relative to ``study_dir``.  A plain module (``"plan"``),
        a package (``"utils"``), or a package submodule (``"utils.layout"``).

    Returns
    -------
    ModuleType
        The requested module.

    Raises
    ------
    ImportError
        If no file backs ``dotted`` inside ``study_dir``.
    """

    study_dir = Path(study_dir).resolve()
    slug = study_slug(study_dir)
    _ensure_namespace_parents(slug, study_dir)

    key = f"{_NAMESPACE}.{slug}"
    current = study_dir
    module: ModuleType | None = None
    for part in dotted.split("."):
        key = f"{key}.{part}"
        package_dir = current / part
        module_file = current / f"{part}.py"
        if package_dir.is_dir() and (package_dir / "__init__.py").exists():
            module = _load_leaf(key, package_dir, is_package=True)
            current = package_dir
        elif module_file.exists():
            module = _load_leaf(key, module_file, is_package=False)
            current = module_file
        else:
            raise ImportError(
                f"no module {part!r} of {dotted!r} in study directory {study_dir}"
            )
        # Bind onto the parent so ``parent.child`` attribute access works and so
        # relative imports inside the child resolve against a populated parent.
        parent = sys.modules.get(key.rsplit(".", 1)[0])
        if parent is not None:
            setattr(parent, part, module)

    assert module is not None  # loop runs at least once for a non-empty name
    return module


def sibling(caller_file: str, dotted: str) -> ModuleType:
    """Return a module from the caller's own study directory.

    Loads by **path** under a study-unique ``sys.modules`` key, so two studies
    can hold same-named siblings at once and neither can capture the other.

    Parameters
    ----------
    caller_file : str
        The calling module's ``__file__``.  Its parent directory is the study.
    dotted : str
        Sibling to load, relative to the study directory.  Either a plain module
        (``"plan"``), a package (``"utils"``), or a package submodule
        (``"utils.layout"``).

    Returns
    -------
    ModuleType
        The requested module.

    Raises
    ------
    ImportError
        If no file backs ``dotted`` inside the study directory.

    Examples
    --------
    >>> plan_stage = sibling(__file__, "plan")            # doctest: +SKIP
    >>> layout = sibling(__file__, "utils.layout")        # doctest: +SKIP
    """

    return load_study_module(Path(caller_file).resolve().parent, dotted)
