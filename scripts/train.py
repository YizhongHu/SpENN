"""Hydra-compatible placeholder for future VMC training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report skeleton status without constructing training objects."""
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "train",
                "status": cfg.get("implementation_status", "skeleton_only"),
                "note": "Phase 1 Skeleton only; VMC training is not implemented.",
            }
        )
    )


if __name__ == "__main__":
    main()
