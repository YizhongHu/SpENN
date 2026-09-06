"""The sole import boundary from He-cutover into the He-v1 study.

He-v1's modules are loaded **by path, under He-v1-scoped keys**, rather than by
putting He-v1's directory on ``sys.path`` and importing bare names.

The previous form inserted ``he-v1/`` at ``sys.path[0]`` and called
``importlib.import_module("plan")``.  That worked, but only because He-cutover
had prefixed its own same-role modules ``cutover_plan`` and ``cutover_strata``,
leaving the two studies' top-level name sets disjoint.  ``plan`` is one of three
``plan.py`` under ``experiments/``, so the arrangement was correct BY NAMING
CONVENTION and not by construction: adding an ordinary ``plan.py`` or
``collect.py`` to this directory -- the obvious next file -- would have made
this module return He-cutover's module under a He-v1 name, silently, with no
exception. Loading by path removes the shared bare key, so no future filename in
either directory can change what this boundary resolves to.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

HEV1_DIR = Path(__file__).resolve().parent.parent / "he-v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.toolkit.study_imports import load_study_module  # noqa: E402


def module(name: str) -> ModuleType:
    """Return one He-v1 module, loaded under a He-v1-scoped key.

    Parameters
    ----------
    name : str
        A supported He-v1 module name.

    Returns
    -------
    ModuleType
        The He-v1 module, guaranteed to come from ``HEV1_DIR`` regardless of
        which studies are already loaded in this interpreter.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported He-v1 modules.
    """

    if name not in {"canary", "driver", "eval", "layout", "plan", "strata"}:
        raise ValueError(f"unsupported He-v1 module: {name!r}")
    return load_study_module(HEV1_DIR, name)


canary = module("canary")
driver = module("driver")
eval_stage = module("eval")
layout = module("layout")
plan_stage = module("plan")
strata = module("strata")
