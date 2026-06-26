"""Training callbacks and logger helpers."""

from __future__ import annotations


class NullCallback:
    """No-op callback used as a default."""

    def on_step_end(self, *_args, **_kwargs) -> None:
        """Handle the end of a training step without side effects.

        Parameters
        ----------
        *_args
            Positional callback arguments.
        **_kwargs
            Keyword callback arguments.
        """

        pass
