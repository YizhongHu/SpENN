"""Generic SpENN train/evaluate entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spenn.training.run import run_config


def load_config(argv: list[str] | None = None) -> tuple[DictConfig, list[str]]:
    """Load a run config from a path or the default config directory.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments. If ``None``, ``sys.argv[1:]`` is used.

    Returns
    -------
    tuple
        Loaded config and recorded overrides.
    """

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    overrides = list(args.overrides)
    dotlist = [override for override in overrides if "=" in override]
    if args.config is not None:
        config_path = _resolve_config_path(args.config)
        cfg = OmegaConf.load(config_path)
        if dotlist:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
        return cfg, overrides
    config_dir = ROOT / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=args.config_name, overrides=overrides), overrides


def main() -> None:
    """Run the configured train/evaluate job.

    Returns
    -------
    None
        The run summary is printed to standard output in YAML format.
    """

    cfg, overrides = load_config()
    summary = run_config(cfg, forwarded_overrides=overrides)
    print(OmegaConf.to_yaml(summary))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML config path. Overrides the default Hydra config.")
    parser.add_argument("--config-name", default="config", help="Hydra config name used when --config is omitted.")
    parser.add_argument("overrides", nargs="*", help="Hydra dotlist overrides.")
    return parser.parse_args(argv)


def _resolve_config_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = ROOT / "configs" / path
    if candidate.exists():
        return candidate
    return path


if __name__ == "__main__":
    main()
