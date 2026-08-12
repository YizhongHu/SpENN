"""Import-time guards for the public callback package surface."""

from __future__ import annotations


def test_callback_surface_imports() -> None:
    """Import every callback package layer so relative-import regressions fail fast."""

    import tpen.callback
    import tpen.callback.timing

    assert tpen.callback is not None
    assert tpen.callback.timing is not None
