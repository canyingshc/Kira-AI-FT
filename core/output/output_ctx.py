"""Phase 0.5 M7 — OutputCtx, the single piece of mutable state shared
by everything in the ``ON_OUTPUT_PIPELINE`` hook chain.

Lifecycle inside ``send_xml_messages`` (one OutputCtx per LLM step):

    1. Built right after AFTER_LLM_RESPONSE_PARSE (and after AFTER_XML_PARSE),
       so chains is the post-XML-fixup parsed view.
    2. Walked by every handler registered for ON_OUTPUT_PIPELINE in
       priority order (descending). Handlers may:
         - rewrite ``ctx.chains``           (split / merge / drop / edit)
         - set ``ctx.delays``               (per-chain pre-send sleep)
         - set ``ctx.intercepted = True``   (whole step is suppressed)
         - stash notes in ``ctx.meta``      (handler-handler comms)
    3. After the chain runs, framework aligns delays to chains length
       (extra delays pruned, missing slots filled with the default hook's
       random value), then sends or drops based on ``ctx.intercepted``.

Why this lives in core/output rather than core/plugin: the type is pure
data — no plugin hooks are imported here, and nothing in core/plugin
needs to know about it. Decoupling it lets the M7 self-test exercise
the dataclass without standing up the registry.

Constraints from Phase0计划.md §5.4 / §0.D #3:
  * Long sleeps inside delays MUST stay below ``MAX_BLOCKING_DELAY_S`` —
    the hook chain runs inside the per-session lock, so a 30s wait would
    starve the next user message in the same session. Long-form unsent
    is a Phase 5 problem.
  * ``budget_ms`` and ``api_elapsed_ms`` are filled by message_manager
    NOW (decision #4, Phase0计划.md §8.4) so Phase 5's budget-aware
    delay logic doesn't have to revisit the call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from core.chat.message_utils import KiraMessageBatchEvent, MessageChain


MAX_BLOCKING_DELAY_S: float = 5.0
"""Soft cap. M7 does not enforce — DefaultDelayHook respects it via the
configured min/max range, and plugin handlers that exceed it will spam
the same-session lock. Phase 5 will hard-enforce + introduce a
non-blocking delay path."""


@dataclass
class OutputCtx:
    """Mutable state passed through the ON_OUTPUT_PIPELINE hook chain.

    All fields are framework-owned at construction; handlers may rewrite
    chains/delays/intercepted/unsent_reason/meta freely. budget_ms and
    api_elapsed_ms are read-only by convention (filled in once and used
    by Phase 5 budget logic).
    """

    event: "KiraMessageBatchEvent"
    """Originating batch event. Used by handlers for sid lookup, sender
    info, etc. Do not mutate."""

    raw_text: str
    """Post-repair LLM XML text. Read-only — to change what gets sent,
    edit ``chains``."""

    chains: list["MessageChain"] = field(default_factory=list)
    """Parsed message chains, one per <msg>. Handlers may pop, append,
    reorder, or replace items entirely. Length changes are fine; the
    framework pads/truncates ``delays`` to match before sending."""

    delays: list[Optional[float]] = field(default_factory=list)
    """Pre-send sleep (seconds) for each chain at the same index.
    ``None`` means "let the default hook fill it". A handler that wants
    to send chain[2] immediately should set ``delays[2] = 0.0`` (NOT
    None — None signals "no opinion").

    Length aligns with chains AFTER the chain runs, not necessarily
    during; handlers may temporarily leave it empty."""

    intercepted: bool = False
    """When True, the framework drops every chain in this OutputCtx
    (no send_message_chain call). ON_STEP_RESULT still fires with an
    empty message_results list so Phase 5's unsent ledger can record
    the "wanted to say but didn't" event."""

    unsent_reason: Optional[str] = None
    """Free-text reason set alongside ``intercepted=True``. M7 just
    logs it; Phase 5 will persist to the unsent table."""

    budget_ms: int = 0
    """Delay budget remaining (ms) measured from the originating user
    message timestamp (``event.timestamp``) to hook chain start. Phase 5
    budget-aware delay calc uses this; M7 only fills it."""

    api_elapsed_ms: int = 0
    """Cumulative LLM API wall-clock (ms) for this step's chat() call.
    Sourced from ``llm_response.time_consumed`` at hook chain start.
    Phase 5 uses this to subtract from the social-delay budget."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Handler-to-handler scratch space. Use namespaced keys
    ('emotion.bias_packet_id', not 'id') to avoid collisions."""

    def has_pending_default_delays(self) -> bool:
        """True iff at least one slot in ``delays`` is None or the list
        is shorter than ``chains``. DefaultDelayHook reads this to decide
        whether to fill or skip."""
        if len(self.delays) < len(self.chains):
            return True
        return any(d is None for d in self.delays[: len(self.chains)])
