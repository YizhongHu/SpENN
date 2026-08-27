"""The sole import boundary from He-cutover into the He-v1 study."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

HEV1_DIR = Path(__file__).resolve().parent.parent / "he-v1"
if str(HEV1_DIR) not in sys.path:
    sys.path.insert(0, str(HEV1_DIR))


def module(name: str) -> ModuleType:
    """Return one He-v1 module through the single path-inserting accessor."""

    if name not in {"canary", "driver", "eval", "layout", "plan", "strata"}:
        raise ValueError(f"unsupported He-v1 module: {name!r}")
    return importlib.import_module(name)


canary = module("canary")
driver = module("driver")
eval_stage = module("eval")
layout = module("layout")
plan_stage = module("plan")
strata = module("strata")

