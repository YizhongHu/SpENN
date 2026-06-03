"""Wavefunction diagnostics that emit metrics and CSV-ready rows."""

from __future__ import annotations

import torch
from torch import nn

from spenn.data.batch import ElectronBatch
from spenn.diagnostics.base import DiagnosticContext, DiagnosticResult
from spenn.utils.tensor_utils import pairwise_distances


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


class PairDistanceHistogramDiagnostic(HistogramDiagnostic):
    """Build a histogram from all unique electron-pair distances.

    Parameters
    ----------
    bins : int, optional
        Number of histogram bins.
    table_name : str or None, optional
        Output table name. If ``None``, ``"pair_distance_histogram"`` is used.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(self, bins: int = 32, table_name: str | None = None, **_: object) -> None:
        super().__init__(values="pair_distance", bins=bins, table_name=table_name or "pair_distance_histogram", **_)


class RadialDensityDiagnostic(nn.Module):
    """Build a one-body radial-density histogram from final walkers.

    Parameters
    ----------
    bins : int, optional
        Number of radial bins.
    table_name : str, optional
        Output table name.
    metric_prefix : str, optional
        Prefix for emitted radial scalar metrics.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        bins: int = 32,
        table_name: str = "radial_density",
        metric_prefix: str = "radial_density",
        **_: object,
    ) -> None:
        super().__init__()
        self.bins = int(bins)
        self.table_name = table_name
        self.metric_prefix = metric_prefix

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return radial-density metrics and table rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Mean radius metric and histogram rows.
        """

        radii = torch.linalg.vector_norm(context.walkers.positions.detach(), dim=-1).flatten()
        rows = histogram_rows(radii, self.bins)
        total = max(float(radii.numel()), 1.0)
        for row in rows:
            width = float(row["bin_right"]) - float(row["bin_left"])
            row["probability_density"] = 0.0 if width <= 0 else float(row["count"]) / (total * width)
        return DiagnosticResult(
            metrics={f"{self.metric_prefix}/mean_radius": float(radii.mean().item())},
            tables={self.table_name: rows},
        )


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


class SpinResolvedCuspSlopeDiagnostic(nn.Module):
    """Estimate short-range cusp slopes for same- and opposite-spin pairs.

    Parameters
    ----------
    n_points : int, optional
        Number of radial points used in each local fit.
    n_configurations : int, optional
        Number of final walker configurations to use as pair centers.
    axis : int, optional
        Cartesian axis along which pair separations are varied.
    r_min, r_max : float, optional
        Short-range radial fit interval.
    same_spin_target, opposite_spin_target : float, optional
        Target cusp slopes.
    factor_same_spin_node : bool, optional
        Whether to subtract ``log(r)`` for same-spin fits.
    metric_prefix : str, optional
        Metric namespace.
    table_name : str, optional
        Output table name.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        n_points: int = 8,
        n_configurations: int = 4,
        axis: int = 0,
        r_min: float = 1.0e-4,
        r_max: float = 2.0e-2,
        same_spin_target: float = 0.25,
        opposite_spin_target: float = 0.5,
        factor_same_spin_node: bool = True,
        metric_prefix: str = "cusp",
        table_name: str = "cusp_slope_by_spin",
        **_: object,
    ) -> None:
        super().__init__()
        self.n_points = int(n_points)
        self.n_configurations = int(n_configurations)
        self.axis = int(axis)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.same_spin_target = float(same_spin_target)
        self.opposite_spin_target = float(opposite_spin_target)
        self.factor_same_spin_node = bool(factor_same_spin_node)
        self.metric_prefix = metric_prefix
        self.table_name = table_name

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return spin-resolved cusp fit metrics and rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Pair counts, mean and maximum slope errors, and per-pair fit rows.
        """

        positions = context.walkers.positions.detach()
        spins = context.walkers.spins
        if spins is None:
            spins = spin_labels(
                context.system,
                n_walkers=positions.shape[0],
                device=context.device,
                dtype=context.dtype,
            )
        if spins is None:
            raise ValueError("SpinResolvedCuspSlopeDiagnostic requires spin labels")
        if positions.ndim != 3:
            raise ValueError(
                "walker positions must have shape [batch, n_electrons, dim], "
                f"got {tuple(positions.shape)}"
            )
        n_configs = min(self.n_configurations, positions.shape[0])
        rows: list[dict[str, object]] = []
        errors_by_relation: dict[str, list[float]] = {"same": [], "opposite": []}
        cusp_errors_by_relation: dict[str, list[float]] = {"same": [], "opposite": []}
        cusp_module = _model_cusp(context.model)
        r = torch.linspace(self.r_min, self.r_max, self.n_points, device=context.device, dtype=context.dtype)
        for config_index in range(n_configs):
            for i, j in _pair_indices(positions.shape[1]):
                relation = "same" if bool(spins[config_index, i] == spins[config_index, j]) else "opposite"
                target = self.same_spin_target if relation == "same" else self.opposite_spin_target
                batch = _pair_radial_batch(
                    positions[config_index],
                    spins[config_index],
                    i,
                    j,
                    r,
                    axis=self.axis,
                    system=context.system,
                )
                with torch.no_grad():
                    logabs = context.model(batch).logabs
                    if relation == "same" and self.factor_same_spin_node:
                        logabs = logabs - torch.log(r.clamp_min(torch.finfo(context.dtype).tiny))
                    slope = linear_slope(r, logabs)
                    cusp_slope = float("nan")
                    cusp_error = float("nan")
                    if cusp_module is not None:
                        cusp_slope = linear_slope(r, cusp_module(batch))
                        cusp_error = cusp_slope - target
                error = slope - target
                errors_by_relation[relation].append(error)
                if cusp_module is not None:
                    cusp_errors_by_relation[relation].append(cusp_error)
                rows.append(
                    {
                        "configuration": config_index,
                        "pair_i": i,
                        "pair_j": j,
                        "spin_relation": relation,
                        "measured_slope": slope,
                        "target_slope": target,
                        "slope_error": error,
                        "cusp_only_slope": cusp_slope,
                        "cusp_only_slope_error": cusp_error,
                    }
                )
        metrics: dict[str, float] = {}
        for relation, errors in errors_by_relation.items():
            metrics[f"{self.metric_prefix}/{relation}_count"] = float(len(errors))
            if errors:
                values = torch.tensor(errors, dtype=torch.float64)
                metrics[f"{self.metric_prefix}/{relation}_mean_error"] = float(values.mean().item())
                metrics[f"{self.metric_prefix}/{relation}_max_abs_error"] = float(values.abs().max().item())
            else:
                metrics[f"{self.metric_prefix}/{relation}_mean_error"] = float("nan")
                metrics[f"{self.metric_prefix}/{relation}_max_abs_error"] = float("nan")
            cusp_errors = cusp_errors_by_relation[relation]
            metrics[f"{self.metric_prefix}/cusp_only_{relation}_count"] = float(len(cusp_errors))
            if cusp_errors:
                cusp_values = torch.tensor(cusp_errors, dtype=torch.float64)
                metrics[f"{self.metric_prefix}/cusp_only_{relation}_mean_error"] = float(cusp_values.mean().item())
                metrics[f"{self.metric_prefix}/cusp_only_{relation}_max_abs_error"] = float(cusp_values.abs().max().item())
            else:
                metrics[f"{self.metric_prefix}/cusp_only_{relation}_mean_error"] = float("nan")
                metrics[f"{self.metric_prefix}/cusp_only_{relation}_max_abs_error"] = float("nan")
        return DiagnosticResult(metrics=metrics, tables={self.table_name: rows})


class ParticleAntisymmetryDiagnostic(nn.Module):
    """Check particle-token antisymmetry for selected transpositions.

    Parameters
    ----------
    n_samples : int, optional
        Number of final walker configurations to check.
    max_transpositions : int, optional
        Maximum number of pair transpositions to evaluate.
    metric_prefix : str, optional
        Metric namespace.
    table_name : str, optional
        Output table name.
    **_ : object
        Ignored compatibility keyword arguments.
    """

    def __init__(
        self,
        n_samples: int = 32,
        max_transpositions: int = 6,
        metric_prefix: str = "antisymmetry",
        table_name: str = "particle_antisymmetry",
        **_: object,
    ) -> None:
        super().__init__()
        self.n_samples = int(n_samples)
        self.max_transpositions = int(max_transpositions)
        self.metric_prefix = metric_prefix
        self.table_name = table_name

    def forward(self, context: DiagnosticContext) -> DiagnosticResult:
        """Return particle-token antisymmetry metrics and rows.

        Parameters
        ----------
        context : DiagnosticContext
            Runtime diagnostic context.

        Returns
        -------
        DiagnosticResult
            Antisymmetry error metrics and CSV rows.
        """

        positions = context.walkers.positions.detach()[: self.n_samples]
        spins = None if context.walkers.spins is None else context.walkers.spins.detach()[: self.n_samples]
        pairs = _pair_indices(positions.shape[1])[: self.max_transpositions]
        rows: list[dict[str, object]] = []
        all_errors: list[torch.Tensor] = []
        all_logabs_errors: list[torch.Tensor] = []
        all_sign_flips: list[torch.Tensor] = []
        with torch.no_grad():
            original = context.model(ElectronBatch(positions=positions, system=context.system, spins=spins))
            original_psi = original.sign * torch.exp(original.logabs)
            for i, j in pairs:
                permutation = torch.arange(positions.shape[1], device=positions.device)
                permutation[i], permutation[j] = permutation[j].clone(), permutation[i].clone()
                swapped_spins = None if spins is None else spins.index_select(1, permutation)
                swapped = context.model(
                    ElectronBatch(
                        positions=positions.index_select(1, permutation),
                        system=context.system,
                        spins=swapped_spins,
                    )
                )
                swapped_psi = swapped.sign * torch.exp(swapped.logabs)
                denom = original_psi.abs() + swapped_psi.abs()
                error = (original_psi + swapped_psi).abs() / denom.clamp_min(torch.finfo(context.dtype).eps)
                logabs_error = (original.logabs - swapped.logabs).abs()
                sign_flip = original.sign == -swapped.sign
                all_errors.append(error)
                all_logabs_errors.append(logabs_error)
                all_sign_flips.append(sign_flip.to(torch.float64))
                rows.append(
                    {
                        "pair_i": i,
                        "pair_j": j,
                        "antisymmetry_error_mean": float(error.mean().item()),
                        "antisymmetry_error_max": float(error.max().item()),
                        "swap_logabs_error_mean": float(logabs_error.mean().item()),
                        "sign_flip_accuracy": float(sign_flip.to(torch.float64).mean().item()),
                    }
                )
        if not all_errors:
            return DiagnosticResult(tables={self.table_name: rows})
        errors = torch.cat(all_errors)
        logabs_errors = torch.cat(all_logabs_errors)
        sign_flips = torch.cat(all_sign_flips)
        return DiagnosticResult(
            metrics={
                f"{self.metric_prefix}/antisymmetry_error_mean": float(errors.mean().item()),
                f"{self.metric_prefix}/antisymmetry_error_max": float(errors.max().item()),
                f"{self.metric_prefix}/swap_logabs_error_mean": float(logabs_errors.mean().item()),
                f"{self.metric_prefix}/swap_logabs_error_max": float(logabs_errors.max().item()),
                f"{self.metric_prefix}/sign_flip_accuracy": float(sign_flips.mean().item()),
            },
            tables={self.table_name: rows},
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


def all_pair_distances(positions: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return flattened upper-triangle electron-pair distances.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[batch, n_electrons, spatial_dim]``.
    eps : float, optional
        Distance floor passed to :func:`spenn.utils.tensor_utils.pairwise_distances`.

    Returns
    -------
    torch.Tensor
        Distances for all unique pairs with shape ``[batch * n_pairs]``.
    """

    if positions.ndim != 3:
        raise ValueError(f"positions must have shape [batch, n_electrons, spatial_dim], got {tuple(positions.shape)}")
    n_electrons = positions.shape[1]
    if n_electrons < 2:
        raise ValueError("all_pair_distances requires at least two electrons")
    distances = pairwise_distances(positions, eps=eps).squeeze(-1)
    pair_index = torch.triu_indices(n_electrons, n_electrons, offset=1, device=positions.device)
    output = distances[:, pair_index[0], pair_index[1]].reshape(-1)
    assert output.shape == (positions.shape[0] * pair_index.shape[1],)
    return output


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


def _model_cusp(model: nn.Module) -> nn.Module | None:
    cusp = getattr(model, "cusp", None)
    return cusp if isinstance(cusp, nn.Module) else None


def _context_values(context: DiagnosticContext, name: str) -> torch.Tensor:
    if name in {"local_energy", "energy"}:
        return context.local_energy
    if name in {"r12", "pair_distance"}:
        return context.pair_distance
    raise KeyError(f"Unknown diagnostic tensor: {name!r}")


def _pair_indices(n_electrons: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_electrons) for j in range(i + 1, n_electrons)]


def _pair_radial_batch(
    base_positions: torch.Tensor,
    spins: torch.Tensor,
    i: int,
    j: int,
    r: torch.Tensor,
    *,
    axis: int,
    system: object,
) -> ElectronBatch:
    positions = base_positions.unsqueeze(0).expand(r.numel(), -1, -1).clone()
    center = 0.5 * (base_positions[i] + base_positions[j])
    positions[:, i, :] = center
    positions[:, j, :] = center
    positions[:, i, axis] = center[axis] + 0.5 * r
    positions[:, j, axis] = center[axis] - 0.5 * r
    expanded_spins = spins.unsqueeze(0).expand(r.numel(), -1).clone()
    return ElectronBatch(positions=positions, system=system, spins=expanded_spins)


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
