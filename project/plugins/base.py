from __future__ import annotations

from typing import Any, Dict


class InferencePlugin:
    """Base class for inference-time plugins.

    Each plugin receives lifecycle callbacks so modules can stay decoupled
    from the core inference loop while still sharing a common context dict.
    """

    def on_start(self, context: Dict[str, Any]) -> None:
        """Called once before the dataloader loop begins."""

    def on_batch(self, predictions: Dict[str, Any], batch: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Called for every processed batch."""

    def on_finish(self, context: Dict[str, Any]) -> None:
        """Called after the dataloader loop finishes."""
