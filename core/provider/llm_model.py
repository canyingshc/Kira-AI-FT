"""LLMRequest / LLMResponse — Phase 0 M7-merge with native multimodal.

Single targeted change vs current core/provider/llm_model.py: the new
``multimodal_parts`` field on :class:`LLMRequest` and matching merge
logic inside ``assemble_prompt()``. When a caller (e.g. the M7
``message_format_to_multimodal`` path in message_manager) populates
this list with OpenAI vision content parts (``{"type": "image_url",
...}`` / ``{"type": "input_audio", ...}``), ``assemble_prompt()``
emits the user message as the multimodal content list shape:

    {"role": "user", "content": [
        {"type": "text",      "text": "<concatenated user_prompt>"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
        {"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}},
        ...
    ]}

When ``multimodal_parts`` is empty the legacy string-content shape is
preserved exactly — non-multimodal callers see no difference.

DEPRECATED markers from M3 remain on user_prompt/system_prompt and
``assemble_prompt()`` itself; they will be removed in Phase 1 once
all paths route through the Assembler/Translator.
"""
from __future__ import annotations

from typing import Optional, Callable, Literal, Any
from dataclasses import dataclass, field

from core.agent.tool import ToolSet
from core.prompt_manager import Prompt


@dataclass
class LLMRequest:
    """LLMRequest object"""

    """message list provided to llm provider"""
    messages: list = field(default_factory=list)

    """Latest user prompt"""
    # DEPRECATED(Phase 0 → Phase 1): migrated to PromptBlock + Assembler.
    # Old plugins that mutate this via @on.llm_request still work; new
    # plugins should contribute via @register.prompt_block / @on.prompt_assemble.
    user_prompt: list[Prompt] = field(default_factory=list)

    """System prompt"""
    # DEPRECATED(Phase 0 → Phase 1): migrated to PromptBlock + Assembler.
    system_prompt: list[Prompt] = field(default_factory=list)

    """optional: tool definitions for llm to call"""
    tools: Optional[list[dict]] = None

    """optional: tool functions"""
    tool_funcs: Optional[dict[str, Callable]] = None

    """tool set object"""
    tool_set: Optional[ToolSet] = None

    """controls llm behavior of tool calling"""
    tool_choice: Optional[Literal["auto", "none", "required"]] = None

    """Phase 0 M7-merge: native-multimodal content parts merged into the
    user message when assemble_prompt() runs. Each entry is an OpenAI
    vision-style content part dict, e.g.
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        {"type": "input_audio", "input_audio": {"data": "...", "format": "mp3"}}
    Empty list (default) → assemble_prompt() emits string-shaped content
    exactly like pre-M7 behaviour. Non-empty → user message becomes a
    list[content_part] with [{type: text, text: <user_prompt>}, *parts].
    """
    multimodal_parts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.tool_choice:
            if self.tools:
                self.tool_choice = "auto"
            else:
                self.tool_choice = "none"

    def assemble_prompt(self):
        # DEPRECATED(Phase 0 → Phase 1): the M3+ main path uses Assembler
        # → Translator and only calls this for the system slot pop/insert
        # housekeeping + the user_prompt → messages append. Phase 1 will
        # replace this with explicit calls in message_manager.
        if self.system_prompt:
            if self.messages and self.messages[0].get("role") == "system":
                self.messages.pop(0)
            system_prompt = "".join(p.to_string() for p in self.system_prompt if isinstance(p, Prompt))
            self.messages.insert(0, {"role": "system", "content": system_prompt})

        if self.user_prompt:
            user_text = "".join(p.to_string() for p in self.user_prompt if isinstance(p, Prompt))
            if self.multimodal_parts:
                # M7-merge: emit OpenAI vision content shape. Order is
                # text-first so the model reads instructions before the
                # attached media — matches OpenAI's documented ordering
                # convention and is a no-op for providers that just
                # concat the parts in order.
                content_parts: list[dict[str, Any]] = [
                    {"type": "text", "text": user_text}
                ] if user_text else []
                content_parts.extend(self.multimodal_parts)
                if not content_parts:
                    # Edge case: empty user text AND no parts. Fall back
                    # to the legacy shape with empty string so we don't
                    # send an unusable [] content to the API.
                    self.messages.append({"role": "user", "content": ""})
                else:
                    self.messages.append({"role": "user", "content": content_parts})
            else:
                self.messages.append({"role": "user", "content": user_text})


@dataclass
class LLMResponse:
    """Content field in chat completion response"""
    text_response: str

    """
    reasoning content for reasoning models
    Make sure it's always a string to avoid missing fields in API responses
    """
    reasoning_content: str = ""

    """agent step index"""
    agent_step_index: Optional[int] = None

    """Tool call requests in OpenAI format"""
    tool_calls: list[dict] = field(default_factory=list)

    """Tool results list in OpenAI format, including role assistant & tool"""
    tool_results: list[dict] = field(default_factory=list)

    input_tokens: Optional[int] = None

    output_tokens: Optional[int] = None

    """Units: seconds"""
    time_consumed: Optional[float] = None

    def __post_init__(self):
        # Make sure reasoning_content is always string
        if self.reasoning_content is None:
            self.reasoning_content = ""


@dataclass
class RerankResult:
    index: int

    score: float

    text: str
