"""Phase 0.4 M5 — async pipeline framework for hints_key resources.

Right now this package only ships :class:`HintsPipeline`, the generalised
``<hints_key>`` plumbing previously hand-coded for stickers. Future
prepare-once / inject-next-turn helpers (memory_recall, worldbook_lookup,
emotion-driven retrieval) will share this surface instead of reimplementing
the polling/TTL/backpressure logic on their own.
"""
from __future__ import annotations

from .hints_pipeline import HintsKeyHandler, HintsPipeline

__all__ = ["HintsKeyHandler", "HintsPipeline"]
