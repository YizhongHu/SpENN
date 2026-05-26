"""Hydra-compatible placeholder for future wavefunction evaluation."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report skeleton status without evaluating a model."""
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "eval",
                "status": cfg.get("implementation_status", "skeleton_only"),
                "note": "Phase 1 Skeleton only; evaluation is not implemented.",
            }
        )
    )


if __name__ == "__main__":
    main()
