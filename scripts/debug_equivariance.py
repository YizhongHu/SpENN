"""Hydra-compatible placeholder for future equivariance diagnostics."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report skeleton status without running equivariance checks."""
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "debug_equivariance",
                "status": cfg.get("implementation_status", "skeleton_only"),
                "note": "Phase 1 Skeleton only; equivariance checks are not implemented.",
            }
        )
    )


if __name__ == "__main__":
    main()
