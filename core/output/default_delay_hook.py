"""Phase 0.5 M7 — DefaultDelayHook.

Lifts the existing per-chain ``random.uniform(min_message_delay,
max_message_delay)`` sleep out of message_manager and into the
ON_OUTPUT_PIPELINE hook chain at SYS_LOW priority. Behaviour is
preserved: when no other handler sets a delay, the math is identical
to pre-M7, just routed through ``OutputCtx.delays``.

Why this matters:
  * Phase 5 will register its own delay hook at MEDIUM/HIGH priority
    that overrides this one based on ``ctx.budget_ms`` / typing speed
    / patience. Putting the default at SYS_LOW guarantees Phase 5 wins
    for the slots it cares about, while still filling the rest.
  * Plugins that just want to delay one specific chain (e.g. "wait 2s
    before the third message") can set ``delays[2] = 2.0`` and leave
    the rest as None — DefaultDelayHook fills the unanswered slots.
  * SYS_LOW also runs LAST in the priority-descending walk, so by the
    time we look at delays the rest of the chain has already had its
    say.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from core.logging_manager import get_logger
from core.plugin.plugin_handlers import (
    EventHandler,
    EventType,
    Priority,
    event_handler_reg,
)

from .output_ctx import OutputCtx

if TYPE_CHECKING:
    from core.config import KiraConfig
    from core.chat.message_utils import KiraMessageBatchEvent

logger = get_logger("output_pipeline", "orange")


class DefaultDelayHook:
    """SYS_LOW handler that fills ``OutputCtx.delays`` slots left None.

    Holds a reference to KiraConfig because min/max may be hot-reloaded.
    Reads under ``bot_config.bot.{min,max}_message_delay`` to stay
    compatible with pre-M7 deployments — Phase 0 doesn't introduce a
    new config namespace just for this.
    """

    def __init__(self, kira_config: "KiraConfig"):
        self._config = kira_config

    def _read_range(self) -> tuple[float, float]:
        """Pull current min/max from config. Defaults match pre-M7
        message_manager hardcoded fallbacks (0.8 / 1.5)."""
        try:
            bot_cfg = self._config["bot_config"].get("bot") or {}
            mn = float(bot_cfg.get("min_message_delay", "0.8"))
            mx = float(bot_cfg.get("max_message_delay", "1.5"))
        except Exception:
            mn, mx = 0.8, 1.5
        if mx < mn:
            mx = mn
        return mn, mx

    async def __call__(
        self, event: "KiraMessageBatchEvent", ctx: OutputCtx
    ) -> None:
        # Pad delays to chains length first so the index walk below
        # touches every slot. "None" sentinel means "untouched"; we
        # treat negative/<0 values as "untouched" too because some
        # handler might use them as not-yet-decided sentinels.
        n = len(ctx.chains)
        if len(ctx.delays) < n:
            ctx.delays.extend([None] * (n - len(ctx.delays)))
        elif len(ctx.delays) > n:
            # Hooks should not over-allocate; trim safely so framework
            # send loop doesn't index past chains.
            ctx.delays = ctx.delays[:n]

        if not ctx.has_pending_default_delays():
            return

        mn, mx = self._read_range()
        for i in range(n):
            existing = ctx.delays[i]
            if existing is None or (isinstance(existing, (int, float)) and existing < 0):
                ctx.delays[i] = random.uniform(mn, mx)


def register_default_hooks(kira_config: "KiraConfig") -> EventHandler:
    """Register the DefaultDelayHook with event_handler_reg.

    Idempotent-ish: returns the registered EventHandler so lifecycle
    can keep a reference (useful for /reload or in tests). Calling
    twice with the same kira_config registers two handlers — fine for
    tests, callers in lifecycle should only call once.
    """
    hook = DefaultDelayHook(kira_config)
    eh = EventHandler(
        event_type=EventType.ON_OUTPUT_PIPELINE,
        priority=Priority.SYS_LOW,
        handler=hook,
        desc="DefaultDelayHook (Phase 0.5 M7): fills OutputCtx.delays "
        "slots left None with random.uniform(min,max) from "
        "bot_config.bot.{min,max}_message_delay.",
    )
    event_handler_reg.register(eh)
    logger.info(
        "DefaultDelayHook registered at SYS_LOW for ON_OUTPUT_PIPELINE"
    )
    return eh
