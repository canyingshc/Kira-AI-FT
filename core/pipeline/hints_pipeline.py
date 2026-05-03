"""Phase 0.4 M5 — generalised hints_key async pipeline.

Background
----------
M3/M4 made PromptBlock the unit of prompt content; some prompt content
takes work to produce (sticker candidate retrieval, memory recall, …)
that would block the user-facing turn if done inline. The "hints_key"
pattern shifts that work by one turn:

    Turn N    LLM emits  <hints_key type="sticker">happy,thumbs_up</hints_key>
              alongside its normal <msg> output.
    (bg)      Pipeline reads the keys, dispatches them to a registered
              :class:`HintsKeyHandler`, and stores the running task in
              ``self._pending[session_id][key_type]``.
    Turn N+1  Before the assembler runs, message_manager calls
              :meth:`HintsPipeline.collect_blocks` which drains the
              pending bucket — finished tasks become PromptBlocks,
              still-running tasks get ~200ms grace, expired tasks drop
              silently.

The contract is intentionally narrow:

* prepare() is launched via ``asyncio.create_task`` so it never blocks
  the agent's multi-step loop (Phase0计划.md §0.D #7 / §4.2 #1).
* TTL is per-handler (default 600s); collect_blocks discards anything
  older than ``ttl_seconds`` without running the result.
* Same-type re-emission within one session **overwrites** the previous
  pending entry (Phase0计划.md §4.7 边界 1) — last writer wins, and the
  loser's task is cancelled.
* Multiple ``<hints_key type="sticker">`` tags in one response merge
  by set semantics, but the handler receives ``list[str]`` preserving
  first-occurrence order (decision #2, Phase0计划.md §8.2). The handler
  is itself responsible for further de-dup if its lookup is ordered.
* Anything raising inside prepare() / parsing / dispatch only logs a
  warning; the main message path continues. Hints are best-effort.

This module is dependency-light by design — only ``core.prompt_block``
and ``core.logging_manager`` — so the M5 self-test can stub-import it
without standing up the rest of the stack.
"""
from __future__ import annotations

import asyncio
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable

from core.logging_manager import get_logger
from core.prompt_block import PromptBlock

logger = get_logger("hints_pipeline", "purple")

# Matches `<hints_key type="sticker">happy,thumbs_up</hints_key>` anywhere
# in raw LLM output. We intentionally do NOT use ElementTree here:
# - the LLM output may not be wrapped in a single root element (it's a
#   sequence of `<msg>...</msg>` plus ad-hoc top-level tags),
# - <hints_key> may sit OUTSIDE of <msg> by design (so it isn't sent),
# - and we want to be permissive with whitespace / single-vs-double
#   quoted attributes.
_HINTS_KEY_RE = re.compile(
    r"<hints_key\s+type\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</hints_key>",
    re.IGNORECASE | re.DOTALL,
)


class HintsKeyHandler(ABC):
    """Subclasses set ``key_type`` (a string identifier matching the
    ``type`` attribute on ``<hints_key type="...">``) and implement
    :meth:`prepare`.

    The handler should be idempotent w.r.t. duplicate keys — the
    pipeline preserves first-occurrence order but does not promise
    de-dup on its own. ``prepare`` may return ``None`` to indicate
    "nothing to inject this turn" (e.g. zero matches found); this is
    treated as success-with-no-block, not failure.
    """

    key_type: str = ""
    ttl_seconds: int = 600

    @abstractmethod
    async def prepare(
        self,
        keys: list[str],
        session_id: str,
        ctx: dict,
    ) -> Optional[PromptBlock]:
        """Resolve ``keys`` into a PromptBlock to be injected next turn.

        Returning ``None`` means "no candidate / nothing useful" — the
        next turn proceeds with no injection from this handler.
        """
        ...


@dataclass
class _PendingEntry:
    """One in-flight prepare() task for a (session_id, key_type) pair.

    ``created_ts`` is set when the task is scheduled, not when it
    finishes; TTL is measured from scheduling time so a slow handler
    can't keep its result alive longer than the contract allows.
    """

    task: "asyncio.Task[Optional[PromptBlock]]"
    created_ts: float
    ttl_seconds: int
    key_type: str
    keys: tuple[str, ...]


class HintsPipeline:
    """Process-singleton pipeline for hints_key resources.

    Lifecycle owns a single instance, exposed via
    ``PluginContext.hints_pipeline`` and stored on
    ``MessageProcessor.hints_pipeline`` for the inline ingestion path.

    Plugins register handlers either at decorator scan time
    (``@register.hints_handler``) or imperatively from inside
    ``initialize()`` (``ctx.hints_pipeline.register(my_handler,
    plugin_id=...)``). The pipeline tracks ``plugin_id`` so termination
    can drop everything a plugin contributed without affecting other
    plugins.
    """

    def __init__(self) -> None:
        # session_id -> {key_type -> _PendingEntry}
        self._pending: dict[str, dict[str, _PendingEntry]] = {}
        # key_type -> handler instance
        self._handlers: dict[str, HintsKeyHandler] = {}
        # plugin_id -> set of key_types contributed by that plugin
        self._by_plugin: dict[str, set[str]] = {}

    # ── handler registration ─────────────────────────────────────────

    def register(
        self,
        handler: HintsKeyHandler,
        plugin_id: Optional[str] = None,
    ) -> None:
        if not isinstance(handler, HintsKeyHandler):
            logger.warning(
                f"HintsPipeline.register ignored non-HintsKeyHandler "
                f"object of type {type(handler).__name__}"
            )
            return
        kt = handler.key_type
        if not kt:
            logger.warning(
                f"HintsPipeline.register ignored handler without "
                f"key_type ({type(handler).__name__})"
            )
            return
        existing = self._handlers.get(kt)
        if existing is not None and existing is not handler:
            logger.warning(
                f"HintsPipeline: replacing handler for key_type='{kt}' "
                f"({type(existing).__name__} -> {type(handler).__name__})"
            )
        self._handlers[kt] = handler
        if plugin_id:
            self._by_plugin.setdefault(plugin_id, set()).add(kt)

    def unregister(self, key_type: str) -> None:
        self._handlers.pop(key_type, None)
        # Drop any in-flight entries for this type so collect_blocks
        # doesn't try to surface a result for a handler that's gone.
        for sid, bucket in list(self._pending.items()):
            entry = bucket.pop(key_type, None)
            if entry is not None and not entry.task.done():
                entry.task.cancel()
        for pid, types_ in list(self._by_plugin.items()):
            types_.discard(key_type)
            if not types_:
                self._by_plugin.pop(pid, None)

    def clear_plugin(self, plugin_id: str) -> None:
        for kt in list(self._by_plugin.get(plugin_id, set())):
            self.unregister(kt)
        self._by_plugin.pop(plugin_id, None)

    def list_handlers(self) -> list[str]:
        return sorted(self._handlers.keys())

    # ── per-session state ────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Drop everything pending for a session. Called when a session
        goes dormant (Phase 1 wires this to memory_bank's dormant signal;
        for M5 it's an exposed knob without an internal caller).
        """
        bucket = self._pending.pop(session_id, None)
        if not bucket:
            return
        for entry in bucket.values():
            if not entry.task.done():
                entry.task.cancel()

    def has_pending(self, session_id: str) -> bool:
        return bool(self._pending.get(session_id))

    # ── ingest LLM output → schedule prepare() ───────────────────────

    async def consume_response(
        self,
        session_id: str,
        llm_response,
        ctx_snapshot: Optional[dict] = None,
    ) -> None:
        """Parse ``<hints_key>`` tags from ``llm_response.text_response``
        and dispatch each (key_type, keys) group to its handler.

        Same-type repetition within one response merges (set semantics,
        order preserved by first occurrence). Same-type re-emission
        across turns within a session **overwrites** (cancels the old
        in-flight task — Phase0计划.md §4.7 边界 1).

        Errors during parsing or dispatch are logged and swallowed —
        the main response path must continue to flow regardless.
        """
        text = ""
        try:
            text = (getattr(llm_response, "text_response", "") or "")
        except Exception as e:
            logger.warning(f"HintsPipeline: cannot read text_response: {e}")
            return
        if not text:
            return

        groups = self._parse_hints_keys(text)
        if not groups:
            return

        ctx = dict(ctx_snapshot) if ctx_snapshot else {}

        for key_type, keys in groups.items():
            handler = self._handlers.get(key_type)
            if handler is None:
                # Silent skip: an unknown type isn't an error — the LLM
                # may have hallucinated a type, or a plugin disabled
                # itself between turns.
                logger.debug(
                    f"HintsPipeline: no handler for key_type='{key_type}' "
                    f"(keys={keys}); ignoring"
                )
                continue
            self._schedule_prepare(session_id, handler, keys, ctx)

    @staticmethod
    def _parse_hints_keys(text: str) -> dict[str, list[str]]:
        """Extract every ``<hints_key type="X">k1,k2</hints_key>`` and
        merge by type. Within a type, keys are de-duped while preserving
        first-occurrence order (handler receives ``list[str]``, never a
        set, so iteration order is deterministic — see decision #2).
        """
        out: dict[str, list[str]] = {}
        for match in _HINTS_KEY_RE.finditer(text):
            key_type = match.group(1).strip()
            payload = match.group(2)
            if not key_type or payload is None:
                continue
            keys = [k.strip() for k in payload.split(",") if k.strip()]
            if not keys:
                continue
            bucket = out.setdefault(key_type, [])
            seen = set(bucket)
            for k in keys:
                if k in seen:
                    continue
                bucket.append(k)
                seen.add(k)
        return out

    def _schedule_prepare(
        self,
        session_id: str,
        handler: HintsKeyHandler,
        keys: list[str],
        ctx: dict,
    ) -> None:
        bucket = self._pending.setdefault(session_id, {})
        prior = bucket.get(handler.key_type)
        if prior is not None and not prior.task.done():
            # Overwrite policy: last hint wins. Cancel the in-flight
            # prior task so we don't leak resources or double-inject.
            prior.task.cancel()

        async def _runner() -> Optional[PromptBlock]:
            try:
                return await handler.prepare(list(keys), session_id, ctx)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"HintsPipeline: handler '{handler.key_type}' "
                    f"prepare() raised; no block will be injected: {e}"
                )
                return None

        task = asyncio.create_task(
            _runner(),
            name=f"hints_prepare:{session_id}:{handler.key_type}",
        )
        bucket[handler.key_type] = _PendingEntry(
            task=task,
            created_ts=time.monotonic(),
            ttl_seconds=getattr(handler, "ttl_seconds", 600),
            key_type=handler.key_type,
            keys=tuple(keys),
        )

    # ── drain pending → PromptBlocks for next turn ───────────────────

    async def collect_blocks(
        self,
        session_id: str,
        max_wait_ms: int = 200,
    ) -> list[PromptBlock]:
        """Drain pending entries for ``session_id`` and return whatever
        finished within ``max_wait_ms`` as PromptBlocks.

        Per Phase0计划.md §4.7 边界 2, still-running tasks get up to
        ``max_wait_ms`` of grace. Anything still unfinished after that
        is **not** awaited further on this turn; the entry is dropped
        from the bucket so the next turn doesn't see a stale result.
        Expired entries (older than handler.ttl_seconds at collection
        time) are dropped silently.
        """
        bucket = self._pending.pop(session_id, None)
        if not bucket:
            return []

        now = time.monotonic()
        out: list[PromptBlock] = []

        # Snapshot to avoid mutation during iteration.
        items = list(bucket.items())

        # Phase 1: filter expired, settle finished ones synchronously.
        unfinished: list[_PendingEntry] = []
        for key_type, entry in items:
            age = now - entry.created_ts
            if age > entry.ttl_seconds:
                logger.debug(
                    f"HintsPipeline: dropping expired pending "
                    f"(sid={session_id}, type={key_type}, "
                    f"age={age:.1f}s, ttl={entry.ttl_seconds}s)"
                )
                if not entry.task.done():
                    entry.task.cancel()
                continue
            if entry.task.done():
                self._extract(entry, out)
            else:
                unfinished.append(entry)

        # Phase 2: best-effort wait for unfinished entries.
        if unfinished and max_wait_ms > 0:
            try:
                # asyncio.wait with timeout; we don't care WHICH finish
                # first, only that some do. Anything still pending after
                # the timeout is left behind (entry already removed from
                # bucket via the .pop above, so it just becomes orphaned
                # work — Python will gc the task once it completes).
                await asyncio.wait(
                    [e.task for e in unfinished],
                    timeout=max_wait_ms / 1000.0,
                    return_when=asyncio.ALL_COMPLETED,
                )
            except Exception as e:
                logger.warning(
                    f"HintsPipeline: asyncio.wait raised during "
                    f"collect_blocks (sid={session_id}): {e}"
                )

            for entry in unfinished:
                if entry.task.done():
                    self._extract(entry, out)
                else:
                    logger.debug(
                        f"HintsPipeline: prepare not done within "
                        f"{max_wait_ms}ms (sid={session_id}, "
                        f"type={entry.key_type}); skipping this turn"
                    )

        return out

    @staticmethod
    def _extract(entry: _PendingEntry, out: list[PromptBlock]) -> None:
        """Move a finished task's result into ``out`` if it produced a
        block. Cancellation / exceptions / None results all just skip.
        """
        task = entry.task
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # Already logged by the handler runner; nothing more to do.
            return
        block = task.result()
        if block is None:
            return
        if not isinstance(block, PromptBlock):
            logger.warning(
                f"HintsPipeline: handler '{entry.key_type}' returned "
                f"non-PromptBlock object {type(block).__name__}; ignoring"
            )
            return
        out.append(block)


__all__ = ["HintsKeyHandler", "HintsPipeline"]
