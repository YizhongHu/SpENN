"""Run a Python script with Typeguard instrumentation for the `spenn` package."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from omegaconf import OmegaConf
from typeguard import install_import_hook


def main() -> None:
    """Install the Typeguard import hook and run a target script."""

    root = Path(__file__).resolve().parent
    policy = OmegaConf.load(root / "configs" / "typecheck.yaml")
    package = str(policy.get("package", "spenn"))
    args = sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python typechecked.py <script.py> [-- script-args...]")
    if "--" in args:
        separator = args.index("--")
        target = args[:separator]
        forwarded = args[separator + 1 :]
    else:
        target = args
        forwarded = []
    if len(target) != 1:
        raise SystemExit("Usage: python typechecked.py <script.py> [-- script-args...]")

    script = Path(target[0])
    if not script.is_absolute():
        script = root / script
    if not script.exists():
        raise SystemExit(f"Script does not exist: {script}")

    install_import_hook(package)
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
