"""Hydra-compatible placeholder for future local-energy diagnostics."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Report skeleton status without computing local energy."""
    print(
        OmegaConf.to_yaml(
            {
                "entrypoint": "debug_local_energy",
                "status": cfg.get("implementation_status", "skeleton_only"),
                "note": "Phase 1 Skeleton only; local energy is not implemented.",
            }
        )
    )


if __name__ == "__main__":
    main()
