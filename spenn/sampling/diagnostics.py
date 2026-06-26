"""Walker-geometry diagnostics for sampler stats.

These summaries describe where the walkers actually are: radial confinement,
electron pair separations, and center-of-mass drift. For trapped systems such
as Hooke's atom the sampler must remain confined by the wavefunction envelope;
large-radius tails or near-coalescent pair distances are a sign that sampled
energy estimates may be misleading.
"""

from __future__ import annotations

from spenn.data.batch import Walkers
from spenn.dependencies import require_torch

#: Quantile levels reported for the per-electron radius distribution.
RADIUS_QUANTILES = (0.5, 0.9, 0.99)

#: Quantile levels reported for the electron pair-distance distribution.
ELECTRON_DISTANCE_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def summarize_walker_geometry(walkers: Walkers) -> dict[str, float | int]:
    """Summarize walker positions as flat JSON-safe scalar metrics.

    Parameters
    ----------
    walkers : Walkers
        Walker batch with ``positions`` shaped
        ``[n_walkers, n_electrons, spatial_dim]``.

    Returns
    -------
    dict
        JSON-safe scalars. Always includes ``n_walkers``, ``n_electrons``,
        ``spatial_dim``, and ``electron_distance_n_pairs``. Position, radius,
        and center-of-mass metrics are included whenever there is at least one
        electron coordinate; ``electron_distance_*`` quantile metrics are
        included only for systems with at least two electrons (no fabricated
        pair distances for smaller systems).
    """

    torch = require_torch(feature="walker geometry diagnostics")
    positions = walkers.positions.detach()
    if positions.dim() != 3:
        raise ValueError(
            f"walker positions must be [n_walkers, n_electrons, spatial_dim], got shape {tuple(positions.shape)}"
        )
    n_walkers, n_electrons, spatial_dim = (int(size) for size in positions.shape)

    n_pairs_per_walker = n_electrons * (n_electrons - 1) // 2
    metrics: dict[str, float | int] = {
        "n_walkers": n_walkers,
        "n_electrons": n_electrons,
        "spatial_dim": spatial_dim,
        "electron_distance_n_pairs": n_walkers * n_pairs_per_walker,
    }
    if positions.numel() == 0:
        return metrics

    # Quantiles need a real floating dtype; float64 keeps the summary exact for
    # both float32 and float64 chains.
    flat = positions.to(torch.float64)

    metrics["position_mean_abs"] = float(flat.abs().mean().item())
    metrics["position_rms"] = float(flat.pow(2).mean().sqrt().item())
    metrics["position_max_abs"] = float(flat.abs().max().item())

    # Per-walker, per-electron distance from the trap origin.
    radius = flat.norm(dim=-1).reshape(-1)
    radius_quantiles = torch.quantile(radius, torch.tensor(RADIUS_QUANTILES, dtype=torch.float64, device=radius.device))
    metrics["radius_mean"] = float(radius.mean().item())
    metrics["radius_std"] = float(radius.std(unbiased=False).item())
    for level, value in zip(RADIUS_QUANTILES, radius_quantiles):
        metrics[f"radius_q{int(round(level * 100)):02d}"] = float(value.item())
    metrics["radius_max"] = float(radius.max().item())

    # Per-walker mean electron position; nonzero RMS means walkers drifted.
    center_of_mass = flat.mean(dim=1)
    metrics["center_of_mass_rms"] = float(center_of_mass.pow(2).sum(dim=-1).mean().sqrt().item())

    if n_electrons >= 2:
        row, col = torch.triu_indices(n_electrons, n_electrons, offset=1)
        # [n_walkers, n_pairs]: Euclidean distance for each distinct pair i < j.
        pair_distances = (flat[:, row, :] - flat[:, col, :]).norm(dim=-1).reshape(-1)
        distance_quantiles = torch.quantile(
            pair_distances,
            torch.tensor(ELECTRON_DISTANCE_QUANTILES, dtype=torch.float64, device=pair_distances.device),
        )
        metrics["electron_distance_min"] = float(pair_distances.min().item())
        for level, value in zip(ELECTRON_DISTANCE_QUANTILES, distance_quantiles):
            metrics[f"electron_distance_q{int(round(level * 100)):02d}"] = float(value.item())
        metrics["electron_distance_mean"] = float(pair_distances.mean().item())
        metrics["electron_distance_max"] = float(pair_distances.max().item())

    return metrics


__all__ = ["summarize_walker_geometry", "RADIUS_QUANTILES", "ELECTRON_DISTANCE_QUANTILES"]
