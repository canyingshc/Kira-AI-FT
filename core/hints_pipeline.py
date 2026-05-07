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
import random as _random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterable, Literal, Callable, Awaitable, Union, Any

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


# ── frequency policy (post-M5 amendment) ─────────────────────────────


@dataclass
class FrequencyPolicy:
    """Per-handler rate-limiting / probabilistic gate applied at
    ``consume_response`` time, BEFORE prepare is scheduled.

    Modes:
      * ``"every"`` (default): every emission of ``<hints_key type=X>``
        triggers ``prepare()``. Original M5 behaviour.
      * ``"interval"``: only every Nth emission triggers prepare. Useful
        for handlers whose value is not in fresh data each turn (e.g.
        memory recall over a slow-changing topic).
      * ``"random"``: each emission triggers prepare with probability
        ``probability`` (0.0..1.0). Useful for "human forgetting" style
        non-determinism, or when calling out to a slow / costly LLM
        inside prepare and you want to dilute it.

    ``seed`` is forwarded to a per-(handler) ``random.Random`` instance,
    so deterministic replays are possible. ``None`` uses the OS source.

    The policy applies to the **scheduling** decision. A "skip" means
    no prepare runs and no PromptBlock is produced for this turn —
    observers still see ``ON_HINTS_PRODUCED`` with
    ``skipped_by_policy=True`` so they can log / debug the skip.
    """

    mode: Literal["every", "interval", "random"] = "every"
    interval: int = 1            # mode="interval": run every Nth emission
    probability: float = 1.0     # mode="random": run with this probability
    seed: Optional[int] = None   # for deterministic replays

    def __post_init__(self) -> None:
        if self.mode not in ("every", "interval", "random"):
            logger.warning(
                f"FrequencyPolicy: unknown mode '{self.mode}', "
                f"falling back to 'every'"
            )
            self.mode = "every"
        if self.interval < 1:
            logger.warning(
                f"FrequencyPolicy: interval={self.interval} < 1; "
                f"clamping to 1 (= every)"
            )
            self.interval = 1
        if not (0.0 <= self.probability <= 1.0):
            logger.warning(
                f"FrequencyPolicy: probability={self.probability} out of "
                f"[0,1]; clamping"
            )
            self.probability = max(0.0, min(1.0, self.probability))


@dataclass
class _PolicyState:
    """Per-(session_id, key_type) runtime state for FrequencyPolicy.

    ``count`` advances on every emission (whether scheduled or skipped)
    so 'interval' mode can decide deterministically. ``rng`` is a
    private ``random.Random`` instance bound at handler-registration
    time; per-session sharing intentional so seed-based reproducibility
    spans the whole session.
    """

    count: int = 0
    rng: Optional[_random.Random] = None


# ── observer payload ─────────────────────────────────────────────────


@dataclass
class HintsResult:
    """Snapshot of one hint emission, fed to ``ON_HINTS_PRODUCED``
    observers. Surfaces both the produced block (if any) AND policy /
    error metadata so downstream consumers can:

      * **Log**: dump every hint produced (debug, analytics).
      * **Chain**: another plugin's logic reacts to a hint of type X
        being prepared (e.g. emotion plugin notes a memory_recall hit).
      * **Debug**: diagnose "why didn't the LLM see candidates this
        turn" — was it skipped by policy, errored, or just produced
        no useful content.

    The block, if non-None, is the SAME object that will be assembled
    into the next turn's prompt — observers may inspect its content,
    but should not mutate it. A defensive copy is the consumer's
    responsibility if needed.
    """

    session_id: str
    key_type: str
    keys: list[str]
    block: Optional[PromptBlock]
    error: Optional[str] = None         # exception str if prepare raised
    elapsed_ms: float = 0.0
    skipped_by_policy: bool = False     # True → block is None and prepare didn't run
    plugin_id: Optional[str] = None     # owner of the handler (for /debug)


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
        # post-M5 amendment: per-handler frequency policy + per-session counters
        # key_type -> FrequencyPolicy (every-mode default if absent)
        self._policies: dict[str, FrequencyPolicy] = {}
        # session_id -> {key_type -> _PolicyState}
        self._policy_state: dict[str, dict[str, _PolicyState]] = {}
        # key_type -> plugin_id (for HintsResult.plugin_id observer field)
        self._handler_to_plugin: dict[str, Optional[str]] = {}

    # ── handler registration ─────────────────────────────────────────

    def register(
        self,
        handler: HintsKeyHandler,
        plugin_id: Optional[str] = None,
        policy: Optional[FrequencyPolicy] = None,
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
        self._handler_to_plugin[kt] = plugin_id
        if plugin_id:
            self._by_plugin.setdefault(plugin_id, set()).add(kt)
        # Policy: explicit > default("every"). Re-registering replaces.
        if policy is not None:
            self._policies[kt] = policy
        else:
            self._policies.setdefault(kt, FrequencyPolicy())

    def set_policy(self, key_type: str, policy: FrequencyPolicy) -> None:
        """Override the frequency policy for an already-registered key_type
        (or pre-stage one before the handler registers). Resets per-session
        counters for that key_type so the new mode starts clean."""
        self._policies[key_type] = policy
        for sid_state in self._policy_state.values():
            sid_state.pop(key_type, None)

    def get_policy(self, key_type: str) -> FrequencyPolicy:
        return self._policies.get(key_type, FrequencyPolicy())

    def list_pending(self, session_id: str) -> list[dict]:
        """Snapshot of currently-pending entries for a session. Read-only,
        intended for ``/debug`` commands. Returns a list of dicts so callers
        don't accidentally mutate internal state."""
        bucket = self._pending.get(session_id)
        if not bucket:
            return []
        now = time.monotonic()
        out = []
        for kt, entry in bucket.items():
            out.append({
                "key_type": kt,
                "keys": list(entry.keys),
                "age_s": round(now - entry.created_ts, 2),
                "ttl_s": entry.ttl_seconds,
                "done": entry.task.done(),
                "cancelled": entry.task.cancelled(),
            })
        return out

    def unregister(self, key_type: str) -> None:
        self._handlers.pop(key_type, None)
        self._handler_to_plugin.pop(key_type, None)
        self._policies.pop(key_type, None)
        # Drop any in-flight entries for this type so collect_blocks
        # doesn't try to surface a result for a handler that's gone.
        for sid, bucket in list(self._pending.items()):
            entry = bucket.pop(key_type, None)
            if entry is not None and not entry.task.done():
                entry.task.cancel()
        # Drop policy counters for this type across all sessions.
        for sid_state in self._policy_state.values():
            sid_state.pop(key_type, None)
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
        # Also clear policy counters so a session waking up later doesn't
        # inherit stale interval position.
        self._policy_state.pop(session_id, None)
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

        Per-handler ``FrequencyPolicy`` (post-M5 amendment) is applied
        BEFORE scheduling: in mode="interval" or "random" some emissions
        skip ``prepare()`` entirely. Skipped emissions still fire
        ``EventType.ON_HINTS_PRODUCED`` with ``skipped_by_policy=True``
        so observers can track skip rates.

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
        # Capture event for ON_HINTS_PRODUCED dispatch. Falls back to None
        # if ctx_snapshot didn't carry it (e.g. unit tests).
        event = ctx.get("event")

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
            # Apply frequency policy. A "skip" still notifies observers
            # so /debug can see the gate firing.
            if not self._should_schedule(session_id, key_type):
                logger.debug(
                    f"HintsPipeline: policy skipped emission "
                    f"(sid={session_id}, type={key_type}, "
                    f"policy={self._policies.get(key_type)})"
                )
                self._fire_observers(
                    HintsResult(
                        session_id=session_id,
                        key_type=key_type,
                        keys=list(keys),
                        block=None,
                        error=None,
                        elapsed_ms=0.0,
                        skipped_by_policy=True,
                        plugin_id=self._handler_to_plugin.get(key_type),
                    ),
                    event,
                )
                continue
            self._schedule_prepare(session_id, handler, keys, ctx, event)

    def _should_schedule(self, session_id: str, key_type: str) -> bool:
        """Apply the FrequencyPolicy gate. Returns True iff prepare()
        should be scheduled this turn. Always advances the per-(sid,type)
        counter regardless of decision so 'interval' mode is deterministic
        across mixed scheduled / skipped turns.
        """
        policy = self._policies.get(key_type) or FrequencyPolicy()
        sid_state = self._policy_state.setdefault(session_id, {})
        state = sid_state.get(key_type)
        if state is None:
            rng = _random.Random(policy.seed) if policy.seed is not None else _random.Random()
            state = _PolicyState(count=0, rng=rng)
            sid_state[key_type] = state

        state.count += 1

        if policy.mode == "every":
            return True
        if policy.mode == "interval":
            # 1-indexed: count=1 → first emission; gate ON iff count % interval == 0.
            # i.e. with interval=3, schedule on emissions 3, 6, 9, ...
            return (state.count % max(1, policy.interval)) == 0
        if policy.mode == "random":
            return state.rng.random() < policy.probability
        # Defensive: unknown mode treated as "every".
        return True

    def _fire_observers(
        self,
        result: "HintsResult",
        event: Any,
    ) -> None:
        """Dispatch ``ON_HINTS_PRODUCED`` to any registered handler. Runs
        as fire-and-forget tasks so a slow observer cannot delay the
        message path. Imported lazily so this module stays decoupled
        from the plugin event system at import time (the M5 self-test
        stubs out plugin_handlers entirely).
        """
        try:
            from core.plugin.plugin_handlers import (
                event_handler_reg,
                EventType,
            )
        except Exception:  # pragma: no cover — pure-stub test envs
            return
        try:
            handlers = event_handler_reg.get_handlers(
                EventType.ON_HINTS_PRODUCED
            )
        except Exception:
            return
        if not handlers:
            return
        for h in handlers:
            try:
                # Each observer runs in its own task so one slow
                # observer can't hold up the others or the main path.
                asyncio.create_task(
                    h.exec_handler(event, result),
                    name=f"hints_observer:{result.key_type}",
                )
            except Exception as e:
                logger.warning(
                    f"HintsPipeline: failed to dispatch ON_HINTS_PRODUCED "
                    f"to {h}: {e}"
                )

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
        event: Any = None,
    ) -> None:
        bucket = self._pending.setdefault(session_id, {})
        prior = bucket.get(handler.key_type)
        if prior is not None and not prior.task.done():
            # Overwrite policy: last hint wins. Cancel the in-flight
            # prior task so we don't leak resources or double-inject.
            prior.task.cancel()

        plugin_id = self._handler_to_plugin.get(handler.key_type)
        # Capture for observer dispatch — `keys` is a fresh list per call.
        keys_snapshot = list(keys)

        async def _runner() -> Optional[PromptBlock]:
            t0 = time.monotonic()
            block: Optional[PromptBlock] = None
            error_str: Optional[str] = None
            # ``settled`` distinguishes "real outcome" (success / handler
            # exception / handler returned None) from "cancelled by the
            # overwrite policy". Observers should fire only for real
            # outcomes; cancellation is internal bookkeeping.
            settled = False
            try:
                block = await handler.prepare(list(keys_snapshot), session_id, ctx)
                settled = True
                return block
            except asyncio.CancelledError:
                # Re-raise so asyncio's cancellation machinery still works.
                # ``settled`` stays False → no observer fired below.
                raise
            except Exception as e:
                error_str = f"{type(e).__name__}: {e}"
                settled = True
                logger.warning(
                    f"HintsPipeline: handler '{handler.key_type}' "
                    f"prepare() raised; no block will be injected: {e}"
                )
                return None
            finally:
                if settled:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    # Coerce: only PromptBlock counts as a real block.
                    final_block = block if isinstance(block, PromptBlock) else None
                    self._fire_observers(
                        HintsResult(
                            session_id=session_id,
                            key_type=handler.key_type,
                            keys=list(keys_snapshot),
                            block=final_block,
                            error=error_str,
                            elapsed_ms=elapsed_ms,
                            skipped_by_policy=False,
                            plugin_id=plugin_id,
                        ),
                        event,
                    )

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


__all__ = [
    "HintsKeyHandler",
    "HintsPipeline",
    "FrequencyPolicy",
    "HintsResult",
]
