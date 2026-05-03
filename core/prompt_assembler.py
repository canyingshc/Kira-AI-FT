"""Phase 0.2 M3 — PromptAssembler.

Filters PromptBlocks by `enabled` and `condition`, sorts by `depth`,
and calls each block's content_provider (sync or async). Same-cache_key
blocks within one `assemble()` call hit a transient cache so duplicate
work in plugin chains is avoided.

Importantly, the Assembler does **not** perform string concatenation
into the final system prompt. It outputs an `AssembledPrompt` structure
(see core/prompt_block.py) and lets the M3 caller (`message_manager`) do
a minimal `"\\n".join` to produce the legacy single string. The proper
Translator (M4) will replace that join.

The provider call interface is intentionally tolerant:
    - sync or async providers are both supported (`inspect.isawaitable`)
    - providers may take 0 args (a closure already capturing what they
      need) or accept the ctx_snapshot positionally
    - non-string returns are coerced via `str(...)`; None becomes ""
A misbehaving provider only logs a warning and is dropped from the
output; one buggy plugin must not break the assembly pipeline.
"""
from __future__ import annotations

import inspect
import time
from typing import Optional

from core.logging_manager import get_logger
from core.prompt_block import (
    AssembledBlock,
    AssembledPrompt,
    ChatInjection,
    PromptBlock,
)

logger = get_logger("prompt_assembler", "yellow")


class PromptAssembler:
    """Stateless coordinator: filter → sort → evaluate → bucket.

    Stateless on purpose: a single instance is safe to share across the
    whole process. The `cache` lives only inside one `assemble()` call,
    not on the instance, so concurrent assemblies don't poison each other.
    """

    async def assemble(
        self,
        blocks: list[PromptBlock],
        ctx_snapshot: dict,
    ) -> AssembledPrompt:
        out = AssembledPrompt()

        # Per-call cache (cache_key -> already-evaluated content). Bound
        # to this single assemble() invocation; a new call gets a fresh
        # cache so condition-driven content can change between turns.
        cache: dict[str, str] = {}

        # ── filter ───────────────────────────────────────────────────
        filtered: list[PromptBlock] = []
        for b in blocks:
            if not b.enabled:
                continue
            if b.condition is not None:
                try:
                    keep = self._invoke_predicate(b.condition, ctx_snapshot)
                except Exception as e:
                    logger.warning(
                        f"PromptBlock condition raised on '{b.name}', "
                        f"skipping: {e}"
                    )
                    continue
                if not keep:
                    continue
            filtered.append(b)

        # Stable sort: equal-depth blocks preserve registration order.
        # NOTE: ``depth`` is the **system-prompt** sort key. ``in_chat``
        # blocks ignore it during assembly — they are sorted later by
        # ``inject_depth`` at insertion time inside ``message_manager``,
        # which keeps the two axes from colliding.
        filtered.sort(key=lambda x: x.depth)

        # ── evaluate ─────────────────────────────────────────────────
        for block in filtered:
            t0 = time.perf_counter()
            cache_hit = False
            content: Optional[str] = None
            error: Optional[str] = None

            try:
                if block.cache_key and block.cache_key in cache:
                    content = cache[block.cache_key]
                    cache_hit = True
                else:
                    res = self._invoke_provider(
                        block.content_provider, ctx_snapshot
                    )
                    if inspect.isawaitable(res):
                        res = await res
                    content = "" if res is None else str(res)
                    if block.cache_key:
                        cache[block.cache_key] = content
            except Exception as e:
                error = str(e)
                logger.warning(
                    f"PromptBlock '{block.name}' provider raised, "
                    f"skipping: {e}"
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            meta: dict = {
                "name": block.name,
                "depth": block.depth,
                "source": block.source,
                "ms": elapsed_ms,
                "cache_hit": cache_hit,
            }
            if error is not None:
                meta["error"] = error
                out.debug_meta.append(meta)
                # Skip emitting an AssembledBlock when the provider blew up.
                continue

            assembled = AssembledBlock(
                name=block.name,
                depth=block.depth,
                content=content if content is not None else "",
                role=block.role,
                source=block.source,
            )
            # Position-based routing (post-M3 amendment):
            #   - ``in_chat``  → emit a ChatInjection consumed by
            #     message_manager, which inserts it into request.messages
            #     at offset ``inject_depth`` from the end. The block is
            #     NOT placed in system_blocks/user_blocks so the system
            #     prompt remains untouched.
            #   - ``system``   → goes into system_blocks (concatenated by
            #     Translator into the final system text).
            #   - DEPRECATED legacy: ``role == "user"`` with
            #     ``position == "system"`` still routes to user_blocks for
            #     backward compat with M3-era tests; not read by
            #     message_manager.
            position = getattr(block, "position", "system")
            if position == "in_chat":
                out.chat_injections.append(ChatInjection(
                    name=block.name,
                    content=content if content is not None else "",
                    inject_depth=getattr(block, "inject_depth", 0),
                    inject_role=getattr(block, "inject_role", "system"),
                    source=block.source,
                ))
            elif block.role == "user":
                out.user_blocks.append(assembled)
            else:
                out.system_blocks.append(assembled)

            out.debug_meta.append(meta)

        return out

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _invoke_provider(provider, ctx_snapshot: dict):
        """Call a provider that may take 0 or 1 positional arguments.

        We don't strictly enforce a signature: zero-arg lambdas and
        ``async def f(ctx)`` style providers should both work without
        the plugin author thinking about it. Inspect signature once, then
        dispatch.
        """
        try:
            sig = inspect.signature(provider)
            n_positional = sum(
                1 for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
            )
        except (TypeError, ValueError):
            # Builtins / C functions can refuse signature inspection;
            # fall through to a permissive try-with-arg, then bare call.
            n_positional = 1

        if n_positional >= 1:
            return provider(ctx_snapshot)
        return provider()

    @staticmethod
    def _invoke_predicate(predicate, ctx_snapshot: dict) -> bool:
        try:
            sig = inspect.signature(predicate)
            n_positional = sum(
                1 for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
            )
        except (TypeError, ValueError):
            n_positional = 1

        if n_positional >= 1:
            res = predicate(ctx_snapshot)
        else:
            res = predicate()
        return bool(res)
