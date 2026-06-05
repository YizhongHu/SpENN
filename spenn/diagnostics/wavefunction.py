"""Wavefunction diagnostics that emit metrics and CSV-ready rows."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data import ElectronBatch
from spenn.diagnostics.base import DiagnosticContext, DiagnosticResult


class HistogramDiagnostic:
    """Build a histogram table from a production tensor.

    Parameters
    ----------
    values : {"local_energy", "r12"}, optional
        Runtime tensor to histogram.
    bins : int, optional
        Number of histogram bins.
    table_name : str or None, optional
        Output table name. If ``None``, ``"{values}_histogram"`` is used.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, values: str = "local_energy", bins: int = 32, table_name: str | None = None, **_: object) -> None:
        self.values = values
        self.bins = int(bins)
        self.table_name = table_name or f"{values}_histogram"

    def __call__(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return histogram rows for the requested tensor.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Histogram table rows.
        """

        values = _context_values(context, self.values)
        return DiagnosticResult(tables={self.table_name: histogram_rows(values, self.bins)})


class RadialCutDiagnostic(nn.Module):
    """Evaluate centered log-amplitude on a two-particle radial cut.

    Parameters
    ----------
    sector : {"singlet", "triplet"}, optional
        Symmetry sector used to choose the radial axis convention.
    n_points : int, optional
        Number of radial points.
    node_axis : int, optional
        Cartesian nodal axis for triplet factored log-amplitudes.
    table_name : str, optional
        Output table name.
    r_min, r_max : float, optional
        Radial interval.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        sector: str = "singlet",
        n_points: int = 64,
        node_axis: int = 2,
        table_name: str = "wavefunction_radial_cut",
        r_min: float = 5.0e-2,
        r_max: float = 6.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.sector = _normalize_sector(sector)
        self.n_points = int(n_points)
        self.node_axis = int(node_axis)
        self.table_name = table_name
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return radial-cut rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Radial-cut table rows.
        """

        r, batch = _radial_batch(
            self.sector,
            self.node_axis,
            self.n_points,
            self.r_min,
            self.r_max,
            context=context,
        )
        with torch.no_grad():
            logabs = factored_logabs(context.model, batch, self.sector, node_axis=self.node_axis)
            centered = logabs - logabs.mean()
        rows = [
            {"r12": float(r_i.item()), "logabs_centered": float(y_i.item())}
            for r_i, y_i in zip(r.cpu(), centered.cpu(), strict=True)
        ]
        return DiagnosticResult(tables={self.table_name: rows})


class RadialLogAbsComparison(nn.Module):
    """Compare centered radial log-amplitudes against a reference model.

    Parameters
    ----------
    reference_model : torch.nn.Module
        Reference wavefunction model.
    sector : {"singlet", "triplet"}, optional
        Symmetry sector.
    n_points : int, optional
        Number of radial points.
    node_axis : int, optional
        Cartesian nodal axis for triplet factored log-amplitudes.
    metric_key : str, optional
        Metric name for the RMSE.
    table_name : str, optional
        Output table name.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        reference_model: nn.Module,
        sector: str = "singlet",
        n_points: int = 64,
        node_axis: int = 2,
        metric_key: str = "comparison/radial_logabs_rmse",
        table_name: str = "wavefunction_radial_cut",
        **_: object,
    ) -> None:
        super().__init__()
        self.reference_model = reference_model
        self.sector = _normalize_sector(sector)
        self.n_points = int(n_points)
        self.node_axis = int(node_axis)
        self.metric_key = metric_key
        self.table_name = table_name

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return centered radial comparison metrics and rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            RMSE metric and radial comparison rows.
        """

        reference = self.reference_model.to(device=context.device, dtype=context.dtype)
        r, batch = _radial_batch(self.sector, self.node_axis, self.n_points, 5.0e-2, 6.0, context=context)
        with torch.no_grad():
            pred = factored_logabs(context.model, batch, self.sector, node_axis=self.node_axis)
            target = factored_logabs(reference, batch, self.sector, node_axis=self.node_axis)
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            error = pred_centered - target_centered
            rmse = torch.sqrt(error.square().mean())
        rows = [
            {
                "r12": float(r_i.item()),
                "spenn_logabs_centered": float(pred_i.item()),
                "exact_logabs_centered": float(target_i.item()),
                "centered_error": float(error_i.item()),
            }
            for r_i, pred_i, target_i, error_i in zip(
                r.cpu(), pred_centered.cpu(), target_centered.cpu(), error.cpu(), strict=True
            )
        ]
        return DiagnosticResult(metrics={self.metric_key: float(rmse.item())}, tables={self.table_name: rows})


class CuspSlopeDiagnostic(nn.Module):
    """Fit a short-range cusp slope on a radial coordinate cut.

    Parameters
    ----------
    sector : {"singlet", "triplet"}, optional
        Symmetry sector.
    n_points : int, optional
        Number of radial points in the fit.
    node_axis : int, optional
        Cartesian nodal axis for triplet factored log-amplitudes.
    target_slope : float or None, optional
        Target cusp slope. If ``None``, the Hooke singlet/triplet defaults are
        used.
    metric_prefix : str, optional
        Prefix for emitted scalar metrics.
    value_column : str, optional
        Table column for the measured factored log-amplitude.
    table_name : str, optional
        Output table name.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        sector: str = "singlet",
        n_points: int = 16,
        node_axis: int = 2,
        target_slope: float | None = None,
        metric_prefix: str = "comparison",
        value_column: str = "spenn_factored_logabs",
        table_name: str = "cusp_diagnostic_plot",
        **_: object,
    ) -> None:
        super().__init__()
        self.sector = _normalize_sector(sector)
        self.n_points = int(n_points)
        self.node_axis = int(node_axis)
        self.target_slope = _default_cusp_slope(self.sector) if target_slope is None else float(target_slope)
        self.metric_prefix = metric_prefix
        self.value_column = value_column
        self.table_name = table_name

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return cusp-slope metrics and rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Cusp metrics and table rows.
        """

        r, batch = _radial_batch(self.sector, self.node_axis, self.n_points, 1.0e-5, 2.0e-2, context=context)
        with torch.no_grad():
            y = factored_logabs(context.model, batch, self.sector, node_axis=self.node_axis)
            slope = linear_slope(r, y)
        rows = [
            {"r12": float(r_i.item()), self.value_column: float(y_i.item()), "target_slope": self.target_slope}
            for r_i, y_i in zip(r.cpu(), y.cpu(), strict=True)
        ]
        return DiagnosticResult(
            metrics={
                f"{self.metric_prefix}/cusp_target_slope": self.target_slope,
                f"{self.metric_prefix}/cusp_measured_slope": slope,
                f"{self.metric_prefix}/cusp_slope_error": slope - self.target_slope,
            },
            tables={self.table_name: rows},
        )


class ExchangeSymmetryDiagnostic(nn.Module):
    """Evaluate exchange symmetry or antisymmetry under particle swap.

    Parameters
    ----------
    sector : {"singlet", "triplet"}, optional
        Symmetry sector.
    n_samples : int, optional
        Number of random coordinate samples.
    node_axis : int, optional
        Cartesian nodal axis for reference triplet signs.
    reference_model : torch.nn.Module or None, optional
        Optional reference model for sign-alignment diagnostics.
    exchange_mode : {"auto", "spatial_singlet", "particle_antisymmetric"}, optional
        Exchange contract. ``"spatial_singlet"`` swaps coordinates but keeps
        spin labels fixed and expects a symmetric signed wavefunction.
        ``"particle_antisymmetric"`` swaps coordinates and spin labels
        together and expects an antisymmetric signed wavefunction.
        ``"auto"`` uses ``"spatial_singlet"`` for singlets and
        ``"particle_antisymmetric"`` for triplets.
    metric_prefix : str, optional
        Metric namespace.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        sector: str = "singlet",
        n_samples: int = 128,
        node_axis: int = 2,
        reference_model: nn.Module | None = None,
        exchange_mode: str = "auto",
        metric_prefix: str = "comparison",
        **_: object,
    ) -> None:
        super().__init__()
        self.sector = _normalize_sector(sector)
        self.n_samples = int(n_samples)
        self.node_axis = int(node_axis)
        self.reference_model = reference_model
        self.exchange_mode = _normalize_exchange_mode(exchange_mode)
        self.metric_prefix = metric_prefix

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return exchange diagnostics.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Exchange symmetry metrics.
        """

        positions = torch.randn(self.n_samples, 2, 3, device=context.device, dtype=context.dtype)
        spins = spin_labels(context.system, n_walkers=self.n_samples, device=context.device, dtype=context.dtype)
        batch = ElectronBatch(positions=positions, system=context.system, spins=spins)
        exchange_mode = self.exchange_mode
        if exchange_mode == "auto":
            exchange_mode = "spatial_singlet" if self.sector == "singlet" else "particle_antisymmetric"
        swap_spins = exchange_mode == "particle_antisymmetric"
        swapped_spins = spins[:, [1, 0]] if swap_spins and spins is not None else spins
        swapped = ElectronBatch(positions=positions[:, [1, 0]], system=context.system, spins=swapped_spins)
        with torch.no_grad():
            out = context.model(batch)
            swap_out = context.model(swapped)
        logabs_error = (out.logabs - swap_out.logabs).abs()
        antisym_mean_key = "antisym_error_mean" if self.metric_prefix == "symmetry" else "antisymmetry_error_mean"
        antisym_max_key = "antisym_error_max" if self.metric_prefix == "symmetry" else "antisymmetry_error_max"
        if exchange_mode == "spatial_singlet":
            psi = out.sign * torch.exp(out.logabs)
            swap_psi = swap_out.sign * torch.exp(swap_out.logabs)
            denom = psi.abs() + swap_psi.abs()
            symmetry_error = (psi - swap_psi).abs() / denom.clamp_min(torch.finfo(context.dtype).eps)
            sign_match = out.sign == swap_out.sign
            sign_alignment = torch.tensor(1.0, device=context.device, dtype=context.dtype)
            if self.reference_model is not None:
                reference = self.reference_model.to(device=context.device, dtype=context.dtype)
                with torch.no_grad():
                    exact = reference(batch)
                raw_alignment = (out.sign == exact.sign).to(torch.float64).mean()
                sign_alignment = torch.maximum(raw_alignment, 1.0 - raw_alignment)
            return DiagnosticResult(
                metrics={
                    f"{self.metric_prefix}/swap_logabs_error_mean": float(logabs_error.mean().item()),
                    f"{self.metric_prefix}/swap_logabs_error_max": float(logabs_error.max().item()),
                    f"{self.metric_prefix}/symmetry_error_mean": float(symmetry_error.mean().item()),
                    f"{self.metric_prefix}/symmetry_error_max": float(symmetry_error.max().item()),
                    f"{self.metric_prefix}/{antisym_mean_key}": float("nan"),
                    f"{self.metric_prefix}/{antisym_max_key}": float("nan"),
                    f"{self.metric_prefix}/sign_flip_accuracy": float("nan"),
                    f"{self.metric_prefix}/sign_match_accuracy": float(sign_match.to(torch.float64).mean().item()),
                    f"{self.metric_prefix}/sign_alignment_accuracy": float(sign_alignment.item()),
                }
            )
        psi = out.sign * torch.exp(out.logabs)
        swap_psi = swap_out.sign * torch.exp(swap_out.logabs)
        denom = psi.abs() + swap_psi.abs()
        antisym_error = (psi + swap_psi).abs() / denom.clamp_min(torch.finfo(context.dtype).eps)
        sign_flip = out.sign == -swap_out.sign
        sign_alignment = torch.tensor(1.0, device=context.device, dtype=context.dtype)
        if self.reference_model is not None:
            reference = self.reference_model.to(device=context.device, dtype=context.dtype)
            with torch.no_grad():
                exact = reference(batch)
            raw_alignment = (out.sign == exact.sign).to(torch.float64).mean()
            sign_alignment = torch.maximum(raw_alignment, 1.0 - raw_alignment)
        return DiagnosticResult(
            metrics={
                f"{self.metric_prefix}/swap_logabs_error_mean": float(logabs_error.mean().item()),
                f"{self.metric_prefix}/swap_logabs_error_max": float(logabs_error.max().item()),
                f"{self.metric_prefix}/{antisym_mean_key}": float(antisym_error.mean().item()),
                f"{self.metric_prefix}/{antisym_max_key}": float(antisym_error.max().item()),
                f"{self.metric_prefix}/sign_flip_accuracy": float(sign_flip.to(torch.float64).mean().item()),
                f"{self.metric_prefix}/sign_alignment_accuracy": float(sign_alignment.item()),
            }
        )


def _normalize_exchange_mode(value: str) -> str:
    normalized = value.lower().replace("-", "_")
    if normalized in {"auto", "spatial_singlet", "particle_antisymmetric"}:
        return normalized
    raise ValueError(
        "exchange_mode must be 'auto', 'spatial_singlet', or "
        f"'particle_antisymmetric', got {value!r}"
    )


def pair_distance(positions: torch.Tensor) -> torch.Tensor:
    """Return two-particle pair distances.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[batch, 2, spatial_dim]``.

    Returns
    -------
    torch.Tensor
        Pair distances with shape ``[batch]``.
    """

    if positions.ndim != 3 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape [batch, 2, spatial_dim], got {tuple(positions.shape)}")
    return torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1)


def histogram_rows(values: torch.Tensor, bins: int) -> list[dict[str, float]]:
    """Return CSV-ready histogram rows.

    Parameters
    ----------
    values : torch.Tensor
        One-dimensional values to histogram.
    bins : int
        Number of bins.

    Returns
    -------
    list of dict
        Rows with ``bin_left``, ``bin_right``, and ``count`` columns.
    """

    values = values.detach().flatten().cpu()
    counts, edges = torch.histogram(values, bins=int(bins))
    return [
        {"bin_left": float(edges[index].item()), "bin_right": float(edges[index + 1].item()), "count": float(count.item())}
        for index, count in enumerate(counts)
    ]


def linear_slope(x: torch.Tensor, y: torch.Tensor) -> float:
    """Fit a scalar linear slope by least squares.

    Parameters
    ----------
    x, y : torch.Tensor
        One-dimensional coordinate and value tensors.

    Returns
    -------
    float
        Least-squares slope.
    """

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = x_centered.square().sum()
    if denominator <= 0:
        raise ValueError("Cannot fit a slope with zero x variance")
    return float((x_centered * y_centered).sum().div(denominator).item())


def factored_logabs(model: nn.Module, batch: ElectronBatch, sector: str, *, node_axis: int) -> torch.Tensor:
    """Return log-amplitude after removing a triplet node factor.

    Parameters
    ----------
    model : torch.nn.Module
        Wavefunction model returning signed-log outputs.
    batch : ElectronBatch
        Electron positions and metadata.
    sector : str
        Symmetry sector.
    node_axis : int
        Cartesian nodal axis for triplet factored log-amplitudes.

    Returns
    -------
    torch.Tensor
        Factored log-amplitudes with shape ``[batch]``.
    """

    batch = batch.flatten_samples()
    output = model(batch)
    logabs = output.logabs
    if _normalize_sector(sector) == "triplet":
        node = batch.positions[:, 0, node_axis] - batch.positions[:, 1, node_axis]
        logabs = logabs - torch.log(node.abs().clamp_min(torch.finfo(batch.dtype).tiny))
    if logabs.shape != (batch.batch_size,):
        raise ValueError(f"factored logabs must have shape [{batch.batch_size}], got {tuple(logabs.shape)}")
    return logabs


def spin_labels(system: object, *, n_walkers: int, device: torch.device | str | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Return repeated spin labels from system metadata.

    Parameters
    ----------
    system : object
        System with optional ``n_up``, ``n_down``, and ``n_electrons`` fields.
    n_walkers : int
        Number of rows.
    device : torch.device, str, or None
        Target device.
    dtype : torch.dtype
        Target dtype.

    Returns
    -------
    torch.Tensor or None
        Spin labels when spin metadata is present.
    """

    n_up = getattr(system, "n_up", None)
    n_down = getattr(system, "n_down", None)
    n_electrons = getattr(system, "n_electrons", None)
    if n_up is None or n_down is None:
        return None
    spins = torch.tensor([1.0] * int(n_up) + [-1.0] * int(n_down), device=device, dtype=dtype)
    if n_electrons is not None and spins.numel() != int(n_electrons):
        raise ValueError("System spin partition must match n_electrons")
    return spins.unsqueeze(0).expand(n_walkers, -1).clone()


def _context_values(context: DiagnosticContext, name: str) -> torch.Tensor:
    if name in {"local_energy", "energy"}:
        return context.local_energy
    if name in {"r12", "pair_distance"}:
        return context.pair_distance
    raise KeyError(f"Unknown diagnostic tensor: {name!r}")


def _radial_batch(
    sector: str,
    node_axis: int,
    n_points: int,
    r_min: float,
    r_max: float,
    *,
    context: DiagnosticContext,
) -> tuple[torch.Tensor, ElectronBatch]:
    r = torch.linspace(r_min, r_max, n_points, device=context.device, dtype=context.dtype)
    positions = torch.zeros(n_points, 2, 3, device=context.device, dtype=context.dtype)
    axis = 0 if sector == "singlet" else node_axis
    positions[:, 0, axis] = 0.5 * r
    positions[:, 1, axis] = -0.5 * r
    batch = ElectronBatch(
        positions=positions,
        system=context.system,
        spins=spin_labels(context.system, n_walkers=n_points, device=context.device, dtype=context.dtype),
    )
    return r, batch


def _normalize_sector(sector: str) -> str:
    normalized = sector.lower().replace("-", "_")
    if normalized in {"singlet", "opposite_spin", "opposite_spin_singlet"}:
        return "singlet"
    if normalized in {"triplet", "same_spin", "same_spin_triplet"}:
        return "triplet"
    raise ValueError(f"Unknown sector: {sector!r}")


def _default_cusp_slope(sector: str) -> float:
    return 0.5 if sector == "singlet" else 0.25
