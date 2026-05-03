"""Phase 0.2 M3 — PromptBlock system core data structures.

A PromptBlock is the unit of prompt content contributed to the assembler.
There are three sources, in order of precedence (all merged inside
`PromptAssembler.assemble`):

    1. Framework built-in: produced by `PromptManager.get_agent_prompt`.
       Each of the 11 templates in `core/prompts/agent_tmpl.py` is wrapped
       in a PromptBlock at depths 10/20/.../110.
    2. Plugin static: registered at decorator scan time via
       `@register.prompt_block(...)` on a plugin method, bound during
       `init_plugin` into the global `PromptBlockRegistry`.
    3. Plugin runtime: contributed inside `EventType.ON_PROMPT_ASSEMBLE`
       handlers via the `BlockCollector` argument.

The legacy `Prompt` class in `core.prompt_manager` is **not** removed.
`PromptBlock.to_legacy_prompt(content)` bridges between the new pipeline
and any place that still expects a `Prompt`. The Phase 0 main path keeps
the legacy `LLMRequest.assemble_prompt` flow alive for backward compat
with old `@on.llm_request` plugins.

This module is intentionally dependency-light: importing it must not
require sqlalchemy, the adapter layer, or any other heavy runtime piece,
so the M3 self-test can stub-import it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, Union, TYPE_CHECKING

from core.logging_manager import get_logger

if TYPE_CHECKING:
    from core.prompt_manager import Prompt

logger = get_logger("prompt_block", "yellow")

# Type alias: providers can be sync or async, and may accept the
# ctx_snapshot argument or be zero-arg lambdas. Assembler accommodates both.
ContentProvider = Callable[..., Union[str, Awaitable[str]]]
ConditionFn = Callable[..., bool]


@dataclass
class PromptBlock:
    """A single, named contribution to the assembled prompt.

    There are two distinct injection mechanisms, distinguished by ``position``:

      * ``position="system"`` (default) — block content is rendered into the
        single concatenated **system prompt**. Multiple system blocks are
        ordered by ``depth`` (small-first, stable). This corresponds to
        SillyTavern-style **dragging/reordering of system entries**.

      * ``position="in_chat"`` — block content is injected as a separate
        message **inside the chat history** at offset ``inject_depth``
        counted backwards from the end (depth=0 → just before the latest
        user message, depth=N → before the Nth-last message). This
        corresponds to SillyTavern-style **in-chat depth injection**
        (Author's Note / World Info at depth / Persona at depth, etc.).
        The injected message carries ``inject_role`` ("system" by default;
        may be "user" or "assistant").

    Fields:
        name             - unique identifier within its plugin scope (used
                           for /debug dumps and replacement in the registry)
        content_provider - callable returning the block's content; may be
                           sync or async; may take 0 or 1 (ctx_snapshot)
                           positional arguments.
        depth            - **system-prompt sort key** (smaller = closer to
                           the front). Default 100 keeps unspecified plugin
                           blocks behind the framework defaults (which sit
                           at 10..110 in multiples of 10, leaving
                           95/85/... slots free). Only meaningful when
                           ``position == "system"``.
        enabled          - static off-switch; a False block is dropped at
                           filter time without invoking the provider.
        position         - "system" (concat into system prompt, default)
                           or "in_chat" (insert as a message into the chat
                           history at ``inject_depth``).
        inject_depth     - position offset from the END of ``request.messages``
                           when ``position == "in_chat"``. depth=0 inserts
                           just before the final user message;
                           depth=N inserts before the Nth-last message.
                           Ignored when ``position == "system"``.
        inject_role      - role of the injected message when
                           ``position == "in_chat"``. Default "system" so
                           the injection reads as a framework directive
                           rather than dialogue. Ignored when
                           ``position == "system"``.
        role             - DEPRECATED: kept for backward compat with M3
                           callers that pass ``role="user"``. The legacy
                           ``user_blocks`` bucket on AssembledPrompt is no
                           longer plumbed into ``request.messages`` — use
                           ``position="in_chat"`` for that. Setting
                           ``role="user"`` without ``position`` still routes
                           the block into ``user_blocks`` for backward compat
                           but its content is dropped by message_manager.
        condition        - optional runtime predicate; if it returns False
                           (or raises) the block is skipped.
        cache_key        - if set, providers sharing the same key inside a
                           single `assemble()` call only run once.
        source           - human-readable origin tag for /debug dumps:
                           "framework" for built-ins, "plugin:<plugin_id>"
                           for plugin-registered blocks.
    """

    name: str
    content_provider: ContentProvider
    depth: int = 100
    enabled: bool = True
    role: Literal["system", "user"] = "system"
    condition: Optional[ConditionFn] = None
    cache_key: Optional[str] = None
    source: str = "framework"
    # Phase 0.2 (post-M3 amendment): SillyTavern-style two-axis routing.
    # See class docstring above. Defaults preserve the M3-era behaviour
    # (every block lands in the system prompt block sorted by ``depth``).
    position: Literal["system", "in_chat"] = "system"
    inject_depth: int = 0
    inject_role: Literal["system", "user", "assistant"] = "system"

    def to_legacy_prompt(self, content: str) -> "Prompt":
        """Wrap an already-evaluated block content into the legacy Prompt
        shape used by `LLMRequest.system_prompt` / `user_prompt`.

        Imported lazily to avoid a circular import (prompt_manager imports
        Prompt-related types and we don't want to flip the dependency).
        """
        from core.prompt_manager import Prompt

        return Prompt(
            content=content,
            name=self.name,
            source=self.role,
        )


@dataclass
class AssembledBlock:
    """A PromptBlock whose content_provider has already been evaluated.
    Returned by PromptAssembler inside AssembledPrompt."""

    name: str
    depth: int
    content: str
    role: str  # "system" | "user"
    source: str


@dataclass
class ChatInjection:
    """A PromptBlock with ``position="in_chat"`` whose content has already
    been evaluated. Carries the *positional* metadata needed by
    ``message_manager`` to insert it into ``request.messages``.

    ``inject_depth`` counts backwards from the end of ``request.messages``
    (after the legacy ``LLMRequest.assemble_prompt()`` has appended the
    final user message). depth=0 → inserted just BEFORE the last message
    (which is normally the current user turn). depth=N → inserted before
    the Nth-last message.

    Multiple injections at the same depth keep the order in which they
    were produced by the assembler (stable sort by inject_depth).
    """

    name: str
    content: str
    inject_depth: int
    inject_role: str  # "system" | "user" | "assistant"
    source: str


@dataclass
class AssembledPrompt:
    """The assembler's structured output. Note that this is **not** the
    final string concatenation — that's the Translator's job in M4. M3
    callers do a minimal `\\n`.join over `system_blocks` to maintain
    behavioural parity with the legacy single-system-string flow.

    ``user_blocks`` is **DEPRECATED** as of the post-M3 amendment that
    introduced ``ChatInjection``. It is still populated when a plugin
    sets ``role="user"`` without ``position="in_chat"``, purely for
    backward compatibility with already-written tests and any external
    consumer; ``message_manager`` no longer reads it. New code should
    set ``position="in_chat"`` and consume ``chat_injections`` instead.
    """

    system_blocks: list[AssembledBlock] = field(default_factory=list)
    user_blocks: list[AssembledBlock] = field(default_factory=list)  # DEPRECATED
    chat_injections: list[ChatInjection] = field(default_factory=list)
    debug_meta: list[dict] = field(default_factory=list)

    def system_text(self, separator: str = "") -> str:
        """Concatenate evaluated system block content. M3 stub for the
        future Translator; uses empty separator because the legacy
        `Prompt.to_string` already appends a trailing newline (`end="\\n"`),
        so each block content already ends in a newline."""
        return separator.join(b.content for b in self.system_blocks)

    def user_text(self, separator: str = "") -> str:
        """DEPRECATED — see AssembledPrompt class docstring. Kept so the
        existing M3/M4 self-tests keep passing; not consumed by
        message_manager. New code should iterate ``chat_injections``."""
        return separator.join(b.content for b in self.user_blocks)


class BlockCollector:
    """Argument passed to `ON_PROMPT_ASSEMBLE` handlers so plugins can
    contribute *runtime-decided* PromptBlocks for the current turn.

    Static plugin blocks belong in `PromptBlockRegistry` (registered once);
    BlockCollector is for blocks whose presence depends on the current
    chat_env / event / state and so cannot be pre-registered.
    """

    def __init__(self) -> None:
        self._blocks: list[PromptBlock] = []

    def add(self, block: PromptBlock) -> None:
        if not isinstance(block, PromptBlock):
            logger.warning(
                f"BlockCollector.add ignored non-PromptBlock object of "
                f"type {type(block).__name__}"
            )
            return
        self._blocks.append(block)

    def get_all(self) -> list[PromptBlock]:
        return list(self._blocks)


class PromptBlockRegistry:
    """Holds plugin-registered *static* PromptBlocks across the lifetime
    of a plugin's enabled state.

    Keyed by plugin_id so unloading / disabling a plugin cleanly drops
    all of that plugin's blocks via `clear_plugin`. Within one plugin,
    blocks are keyed by `block.name`; re-registering the same name
    replaces the previous entry (with a warning).

    The registry is owned by lifecycle and exposed:
      - through PluginContext.prompt_block_registry (so plugins can
        introspect),
      - through PromptManager.block_registry (so message_manager can
        gather blocks for assembly without needing a PluginContext).
    """

    def __init__(self) -> None:
        # plugin_id -> {block_name: PromptBlock}
        self._by_plugin: dict[str, dict[str, PromptBlock]] = {}

    def register(
        self,
        block: PromptBlock,
        plugin_id: Optional[str] = None,
    ) -> None:
        pid = plugin_id or "framework"
        bucket = self._by_plugin.setdefault(pid, {})
        if block.name in bucket:
            logger.warning(
                f"PromptBlockRegistry: replacing block '{block.name}' "
                f"for plugin '{pid}'"
            )
        bucket[block.name] = block

    def unregister(
        self,
        name: str,
        plugin_id: Optional[str] = None,
    ) -> None:
        pid = plugin_id or "framework"
        bucket = self._by_plugin.get(pid)
        if not bucket:
            return
        bucket.pop(name, None)

    def clear_plugin(self, plugin_id: str) -> None:
        """Remove all blocks contributed by a plugin. Called by
        PluginManager._cleanup_plugin_registration when a plugin is
        terminated, disabled, or uninstalled."""
        self._by_plugin.pop(plugin_id, None)

    def get_all(self) -> list[PromptBlock]:
        out: list[PromptBlock] = []
        for bucket in self._by_plugin.values():
            out.extend(bucket.values())
        return out

    def get_for_plugin(self, plugin_id: str) -> list[PromptBlock]:
        return list(self._by_plugin.get(plugin_id, {}).values())
