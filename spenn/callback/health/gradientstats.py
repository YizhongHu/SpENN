"""Gradient health callback."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ..base import Callback, Event


class GradientStats(Callback):
    """Track gradient health at ``step_end`` (after ``optimizer.step()``).

    Reads parameter gradients from ``state.model`` and logs norm/finite metrics
    under ``checks/gradient``. With ``check_finite`` it fails on non-finite
    gradients, and with ``max_global_grad_norm`` it fails when the global norm
    is exceeded. It does not require convergence or small gradients.
    """

    def __init__(
        self,
        triggers: Iterable[str],
        *,
        fail_fast: bool = False,
        max_global_grad_norm: float | None = None,
        check_finite: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(triggers, **kwargs)
        self.fail_fast = bool(fail_fast)
        self.max_global_grad_norm = None if max_global_grad_norm is None else float(max_global_grad_norm)
        self.check_finite = bool(check_finite)

    def on_step_end(self, event: Event) -> None:
        """Summarize gradients of the model parameters."""

        state = event.state
        grads = [
            param.grad.detach().reshape(-1)
            for param in state.model.parameters()
            if param.grad is not None
        ]
        metrics: dict[str, Any] = {
            "n_grad_tensors": len(grads),
            "n_grad_elements": 0,
            "global_grad_norm": 0.0,
            "max_abs_grad": 0.0,
            "mean_abs_grad": 0.0,
            "nonfinite_grad_fraction": 0.0,
        }
        global_norm = 0.0
        nonfinite_fraction = 0.0
        if grads:
            flat = torch.cat(grads)
            n_elements = int(flat.numel())
            finite_mask = torch.isfinite(flat)
            n_finite = int(finite_mask.sum().item())
            nonfinite_fraction = float((n_elements - n_finite) / n_elements) if n_elements else 0.0
            finite_values = flat[finite_mask]
            global_norm = float(finite_values.norm().item()) if n_finite else 0.0
            abs_finite = finite_values.abs()
            metrics["n_grad_elements"] = n_elements
            metrics["global_grad_norm"] = global_norm
            metrics["max_abs_grad"] = float(abs_finite.max().item()) if n_finite else 0.0
            metrics["mean_abs_grad"] = float(abs_finite.mean().item()) if n_finite else 0.0
            metrics["nonfinite_grad_fraction"] = nonfinite_fraction

        failure: str | None = None
        if self.check_finite and nonfinite_fraction > 0.0:
            failure = f"nonfinite_grad_fraction={nonfinite_fraction} exceeds 0.0"
        elif self.max_global_grad_norm is not None and global_norm > self.max_global_grad_norm:
            failure = f"global_grad_norm={global_norm} exceeds max_global_grad_norm={self.max_global_grad_norm}"

        metrics["passed"] = failure is None
        event.context.log(metrics, step=state.step, namespace="checks/gradient")
        if self.fail_fast and failure is not None:
            raise RuntimeError(f"GradientStats failed at step {state.step}: {failure}")



__all__ = ["GradientStats"]
