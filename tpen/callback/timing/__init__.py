"""Runtime timing callbacks."""

from __future__ import annotations

from .evaluation_component_timing import EvaluationComponentTiming
from .evaluation_timing import EvaluationTiming
from .run_timing import RunTiming
from .train_phase_timing import TrainPhaseTiming


def __getattr__(name: str) -> object:
    """Load timing callbacks that resolve a domain state class only on demand.

    A `tpen.callback.StatefulCallback` declares its ``state_type`` as a ClassVar,
    which forces the state class to be imported at class-creation time.
    Importing one from either the evaluation or training domain runs that
    package's ``__init__``, which pulls in torch. `DiagnosticTiming` and
    `TrainStepTiming` therefore load lazily to keep
    ``import tpen.callback.timing`` torch-free. Callbacks that only need event
    TYPES resolve those inside their ``__init__`` and stay eager.
    """

    if name == "DiagnosticTiming":
        from .diagnostic_timing import DiagnosticTiming

        return DiagnosticTiming
    if name == "TrainStepTiming":
        from .train_step_timing import TrainStepTiming

        return TrainStepTiming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiagnosticTiming",
    "EvaluationComponentTiming",
    "EvaluationTiming",
    "RunTiming",
    "TrainPhaseTiming",
    "TrainStepTiming",
]
