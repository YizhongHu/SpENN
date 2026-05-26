"""Hydra-compatible placeholder for future sampler runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report skeleton status without sampling walkers."""
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "sample",
                "status": cfg.get("implementation_status", "skeleton_only"),
                "note": "Phase 1 Skeleton only; sampling is not implemented.",
            }
        )
    )


if __name__ == "__main__":
    main()
