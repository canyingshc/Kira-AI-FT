"""Phase 0.4 M5 — async pipeline framework for hints_key resources.

Right now this package only ships :class:`HintsPipeline`, the generalised
``<hints_key>`` plumbing previously hand-coded for stickers. Future
prepare-once / inject-next-turn helpers (memory_recall, worldbook_lookup,
emotion-driven retrieval) will share this surface instead of reimplementing
the polling/TTL/backpressure logic on their own.

Post-M5 amendment exports:
* :class:`FrequencyPolicy` — per-handler rate-limiting (every / interval /
  random) for ``<hints_key>`` emissions.
* :class:`HintsResult` — observer payload fired via
  ``EventType.ON_HINTS_PRODUCED`` when a hint emission resolves.
"""
from __future__ import annotations

from .hints_pipeline import (
    FrequencyPolicy,
    HintsKeyHandler,
    HintsPipeline,
    HintsResult,
)

__all__ = [
    "FrequencyPolicy",
    "HintsKeyHandler",
    "HintsPipeline",
    "HintsResult",
]
