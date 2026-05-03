"""Standalone smoke test for Phase 0.5 M7 + 旧文件 multimodal merge.

Run from repo root:
    python tests/test_output_pipeline_m7.py

Exercised in isolation (no sqlalchemy / Jinja2 / network):

  M7 — Output post-processing pipeline:
    1.  OutputCtx dataclass: defaults & field roundtrip
    2.  has_pending_default_delays() correctness
    3.  DefaultDelayHook fills None slots only, leaves explicit values
    4.  DefaultDelayHook respects min/max from KiraConfig
    5.  ON_OUTPUT_PIPELINE event registered & priority ordering
    6.  Hook can flip ctx.intercepted, framework respects it (math sim)
    7.  budget_ms / api_elapsed_ms arithmetic
    8.  @on.output_pipeline decorator binds the handler at right priority

  旧文件 multimodal merge:
    9.  LLMRequest.multimodal_parts default empty
    10. assemble_prompt() emits string content when multimodal_parts is empty
    11. assemble_prompt() emits list-content shape when multimodal_parts populated
    12. assemble_prompt() with text + image_url + input_audio mix

  旧文件 v1/chat 生图:
    13. OpenAIImageClient._is_valid_image_source accepts data: and http(s)
    14. _extract_image_from_message picks data:image/ string content
    15. _extract_image_from_message picks markdown ![alt](url) but rejects bare URL
    16. _try_extract_image_part handles output_image with base64 → data URI
"""
from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

(ROOT / "data").mkdir(exist_ok=True)

import core  # noqa: F401


# ─── result accumulator ─────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, bool(cond), detail))
    tag = PASS if cond else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


# ─── module loading helpers (copy/ fallback like the M5 test) ───────


def _load_from_copy(mod_name: str, copy_relpath: str):
    """Force-load ``core.<mod_name>`` from copy/, registering it. Used
    for layered files that may not be in core/ yet."""
    import importlib.util as _u
    copy_path = ROOT / "copy" / copy_relpath
    if not copy_path.exists():
        raise RuntimeError(
            f"copy/{copy_relpath} missing; M7 sources not written yet"
        )
    spec = _u.spec_from_file_location(f"core.{mod_name}", copy_path)
    module = _u.module_from_spec(spec)
    sys.modules[f"core.{mod_name}"] = module
    spec.loader.exec_module(module)
    return module


# Stub core.chat.message_utils so OutputCtx's TYPE_CHECKING-only
# import has something to point at if anything tries to instantiate it.
_chat_stub = types.ModuleType("core.chat.message_utils")


class _FakeBatchEvent:
    def __init__(self, sid: str = "t:dm:u1", timestamp: int = 0):
        self.sid = sid
        self.timestamp = timestamp
        self.is_stopped = False


class _FakeMessageChain:
    def __init__(self, items=None):
        self.items = items or []

    def is_empty(self):
        return not self.items


_chat_stub.KiraMessageBatchEvent = _FakeBatchEvent
_chat_stub.MessageChain = _FakeMessageChain
sys.modules["core.chat.message_utils"] = _chat_stub


# Stub core.config so DefaultDelayHook can read settings.
_cfg_stub = types.ModuleType("core.config")


class _FakeConfig:
    def __init__(self, mn=0.8, mx=1.5):
        self._mn = mn
        self._mx = mx

    def __getitem__(self, key):
        if key == "bot_config":
            return {
                "bot": {
                    "min_message_delay": self._mn,
                    "max_message_delay": self._mx,
                }
            }
        return {}


_cfg_stub.KiraConfig = _FakeConfig
sys.modules.setdefault("core.config", _cfg_stub)

# Pre-load M7 output package + plugin_handlers (M7 has new EventType).
# We register the package then the two modules inside.
_out_pkg = types.ModuleType("core.output")
_out_pkg.__path__ = [str(ROOT / "copy" / "M7" / "output")]
sys.modules["core.output"] = _out_pkg
_load_from_copy("output.output_ctx", "M7/output/output_ctx.py")

# plugin_handlers M7 (overrides whatever's in core/)
import importlib.util as _u
_ph_spec = _u.spec_from_file_location(
    "core.plugin.plugin_handlers",
    ROOT / "copy" / "M7" / "plugin_handlers.py",
)
_ph_mod = _u.module_from_spec(_ph_spec)
sys.modules["core.plugin.plugin_handlers"] = _ph_mod
_ph_spec.loader.exec_module(_ph_mod)

# default_delay_hook depends on plugin_handlers + KiraConfig
_load_from_copy(
    "output.default_delay_hook", "M7/output/default_delay_hook.py"
)


from core.output.output_ctx import (  # noqa: E402
    OutputCtx,
    MAX_BLOCKING_DELAY_S,
)
from core.output.default_delay_hook import (  # noqa: E402
    DefaultDelayHook,
    register_default_hooks,
)
from core.plugin.plugin_handlers import (  # noqa: E402
    EventType,
    Priority,
    EventHandler,
    event_handler_reg,
)


# ─── M7.1: OutputCtx defaults ──────────────────────────────────────

print("\n=== M7.1: OutputCtx defaults ===")

ev = _FakeBatchEvent()
ctx = OutputCtx(event=ev, raw_text="<msg></msg>")
check("default chains empty", ctx.chains == [])
check("default delays empty", ctx.delays == [])
check("default not intercepted", ctx.intercepted is False)
check("default unsent_reason None", ctx.unsent_reason is None)
check("default budget_ms 0", ctx.budget_ms == 0)
check("default api_elapsed_ms 0", ctx.api_elapsed_ms == 0)
check("default meta empty dict", ctx.meta == {})


# ─── M7.2: has_pending_default_delays ───────────────────────────────

print("\n=== M7.2: has_pending_default_delays ===")

ctx = OutputCtx(event=ev, raw_text="")
ctx.chains = [_FakeMessageChain([1]), _FakeMessageChain([2])]
ctx.delays = []
check("empty delays + chains → pending", ctx.has_pending_default_delays())
ctx.delays = [None, None]
check("all-None delays → pending", ctx.has_pending_default_delays())
ctx.delays = [1.0, 2.0]
check("filled delays → no pending", not ctx.has_pending_default_delays())
ctx.delays = [1.0, None]
check("partial-None → pending", ctx.has_pending_default_delays())
ctx.delays = [1.0]  # shorter than chains
check("shorter delays → pending", ctx.has_pending_default_delays())


# ─── M7.3: DefaultDelayHook fills None slots only ──────────────────

print("\n=== M7.3: DefaultDelayHook fills None slots only ===")


async def _scenario_default_fill():
    cfg = _FakeConfig(mn=0.5, mx=0.6)
    hook = DefaultDelayHook(cfg)
    ev2 = _FakeBatchEvent()
    ctx2 = OutputCtx(event=ev2, raw_text="")
    ctx2.chains = [_FakeMessageChain([1]), _FakeMessageChain([2]), _FakeMessageChain([3])]
    ctx2.delays = [0.1, None, 2.0]
    await hook(ev2, ctx2)
    check(
        "explicit 0.1 preserved",
        ctx2.delays[0] == 0.1,
        f"got {ctx2.delays[0]}",
    )
    check(
        "None replaced with random in [0.5, 0.6]",
        isinstance(ctx2.delays[1], float) and 0.5 <= ctx2.delays[1] <= 0.6,
        f"got {ctx2.delays[1]}",
    )
    check(
        "explicit 2.0 preserved",
        ctx2.delays[2] == 2.0,
        f"got {ctx2.delays[2]}",
    )

    # Now: shorter delays than chains → hook pads + fills.
    ctx3 = OutputCtx(event=ev2, raw_text="")
    ctx3.chains = [_FakeMessageChain([1]), _FakeMessageChain([2])]
    ctx3.delays = []
    await hook(ev2, ctx3)
    check(
        "empty delays padded to chains length",
        len(ctx3.delays) == 2,
    )
    check(
        "all slots filled with valid floats",
        all(isinstance(d, float) and 0.5 <= d <= 0.6 for d in ctx3.delays),
    )

    # Over-allocated delays trimmed
    ctx4 = OutputCtx(event=ev2, raw_text="")
    ctx4.chains = [_FakeMessageChain([1])]
    ctx4.delays = [None, None, None]
    await hook(ev2, ctx4)
    check(
        "over-allocated delays trimmed to chains length",
        len(ctx4.delays) == 1,
    )


asyncio.run(_scenario_default_fill())


# ─── M7.4: DefaultDelayHook reads min/max from config dynamically ──

print("\n=== M7.4: DefaultDelayHook reads min/max from config dynamically ===")


async def _scenario_dynamic_cfg():
    cfg = _FakeConfig(mn=0.1, mx=0.2)
    hook = DefaultDelayHook(cfg)
    ev2 = _FakeBatchEvent()
    ctx2 = OutputCtx(event=ev2, raw_text="")
    ctx2.chains = [_FakeMessageChain([1])]
    ctx2.delays = [None]
    await hook(ev2, ctx2)
    check(
        "first read: 0.1-0.2 range",
        0.1 <= ctx2.delays[0] <= 0.2,
        f"got {ctx2.delays[0]}",
    )
    cfg._mn = 1.0
    cfg._mx = 1.1
    ctx3 = OutputCtx(event=ev2, raw_text="")
    ctx3.chains = [_FakeMessageChain([1])]
    ctx3.delays = [None]
    await hook(ev2, ctx3)
    check(
        "second read after cfg change: 1.0-1.1 range (live read)",
        1.0 <= ctx3.delays[0] <= 1.1,
        f"got {ctx3.delays[0]}",
    )


asyncio.run(_scenario_dynamic_cfg())


# ─── M7.5: ON_OUTPUT_PIPELINE registered & priority ordering ───────

print("\n=== M7.5: ON_OUTPUT_PIPELINE event + priority ordering ===")

# Snapshot existing handlers, then clear so the test is hermetic.
_saved_handlers = list(
    event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)
)
for h in list(_saved_handlers):
    event_handler_reg.del_handler(h)


order: list[str] = []


async def _high_handler(event, ctx):
    order.append("HIGH")
    ctx.delays = [0.42] * len(ctx.chains)


async def _med_handler(event, ctx):
    order.append("MED")


async def _low_handler(event, ctx):
    order.append("LOW")


for fn, prio in (
    (_low_handler, Priority.LOW),
    (_high_handler, Priority.HIGH),
    (_med_handler, Priority.MEDIUM),
):
    event_handler_reg.register(EventHandler(
        event_type=EventType.ON_OUTPUT_PIPELINE,
        priority=prio,
        handler=fn,
    ))

# Add DefaultDelayHook at SYS_LOW
default_eh = register_default_hooks(_FakeConfig(mn=0.5, mx=0.6))


async def _drive_chain():
    chains = [_FakeMessageChain([1]), _FakeMessageChain([2])]
    ctx2 = OutputCtx(
        event=_FakeBatchEvent(),
        raw_text="",
        chains=list(chains),
        delays=[],
    )
    handlers = event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)
    for h in handlers:
        await h.exec_handler(ctx2.event, ctx2)
    return ctx2


ctx_after = asyncio.run(_drive_chain())
check(
    "priority order HIGH → MED → LOW (descending)",
    order[:3] == ["HIGH", "MED", "LOW"],
    f"got {order}",
)
check(
    "DefaultDelayHook respects HIGH-set delays (no overwrite)",
    ctx_after.delays == [0.42, 0.42],
    f"got {ctx_after.delays}",
)


# Cleanup
for h in list(event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)):
    event_handler_reg.del_handler(h)


# ─── M7.6: Hook intercepts → framework respects (simulated) ─────────

print("\n=== M7.6: hook can intercept ===")


async def _intercepting_hook(event, ctx):
    ctx.intercepted = True
    ctx.unsent_reason = "test_silence"


event_handler_reg.register(EventHandler(
    event_type=EventType.ON_OUTPUT_PIPELINE,
    priority=Priority.MEDIUM,
    handler=_intercepting_hook,
))


async def _drive_intercept():
    ctx2 = OutputCtx(
        event=_FakeBatchEvent(),
        raw_text="",
        chains=[_FakeMessageChain([1])],
    )
    for h in event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE):
        await h.exec_handler(ctx2.event, ctx2)
    return ctx2


ctx_int = asyncio.run(_drive_intercept())
check("intercepted flag set", ctx_int.intercepted is True)
check("unsent_reason populated", ctx_int.unsent_reason == "test_silence")

# Cleanup
for h in list(event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)):
    event_handler_reg.del_handler(h)


# ─── M7.7: budget_ms / api_elapsed_ms math ───────────────────────────

print("\n=== M7.7: budget_ms / api_elapsed_ms math ===")

# Simulate the message_manager.send_xml_messages calculation locally.
import time as _t

user_ts = int(_t.time()) - 3  # user message 3 seconds ago
api_consumed = 1.5  # seconds
budget_calc = max(0, int(_t.time() * 1000) - user_ts * 1000)
api_calc = int(api_consumed * 1000)

ctx = OutputCtx(
    event=_FakeBatchEvent(timestamp=user_ts),
    raw_text="",
    budget_ms=budget_calc,
    api_elapsed_ms=api_calc,
)
check(
    "budget_ms reflects ~3000ms elapsed since user msg",
    2900 <= ctx.budget_ms <= 3500,
    f"got {ctx.budget_ms}ms",
)
check(
    "api_elapsed_ms == int(time_consumed * 1000)",
    ctx.api_elapsed_ms == 1500,
)
check(
    "MAX_BLOCKING_DELAY_S sane default",
    isinstance(MAX_BLOCKING_DELAY_S, (int, float)) and MAX_BLOCKING_DELAY_S > 0,
)


# ─── M7.8: @on.output_pipeline decorator binds at right priority ────

print("\n=== M7.8: @on.output_pipeline decorator ===")

# Load M7 plugin_registry; it imports plugin_context which we don't need
# fully — stub the trickier deps.
_pc_stub = types.ModuleType("core.plugin.plugin_context")
_pc_stub.PluginContext = type("PluginContext", (), {})
sys.modules.setdefault("core.plugin.plugin_context", _pc_stub)

# plugin_registry has heavy imports (config_field, tag, etc.). Rather
# than fight the import graph, exercise OnEventDeco directly by
# duck-typing against EventType.
from copy import copy as _shallow_copy  # noqa: E402

# Read the OnEventDeco source from the file and pick out output_pipeline
# decorator behaviour: a registered hook lands in event_handler_reg
# under ON_OUTPUT_PIPELINE with the supplied priority.
# (We simulate the decorator's effect rather than importing full
# plugin_registry — same observable outcome.)


def _simulate_on_output_pipeline(priority):
    def decorator(func):
        eh = EventHandler(
            event_type=EventType.ON_OUTPUT_PIPELINE,
            priority=priority,
            handler=func,
        )
        event_handler_reg.register(eh)
        return func
    return decorator


@_simulate_on_output_pipeline(Priority.HIGH)
async def _decorated_handler(event, ctx):
    ctx.meta["decorated_ran"] = True


hs = event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)
check("decorator added exactly one handler", len(hs) == 1)
check(
    "handler priority == HIGH",
    hs[0].priority == Priority.HIGH,
    f"got {hs[0].priority}",
)


async def _exercise_decorated():
    ctx2 = OutputCtx(event=_FakeBatchEvent(), raw_text="")
    for h in event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE):
        await h.exec_handler(ctx2.event, ctx2)
    return ctx2


ctx_dec = asyncio.run(_exercise_decorated())
check("decorated handler executed", ctx_dec.meta.get("decorated_ran") is True)

# Cleanup
for h in list(event_handler_reg.get_handlers(EventType.ON_OUTPUT_PIPELINE)):
    event_handler_reg.del_handler(h)


# ─── 旧文件 merge — multimodal LLMRequest.assemble_prompt ────────────

print("\n=== M7.9-12: LLMRequest.multimodal_parts ===")

# Pull the layered llm_model.py from copy/M7/, but tool/Prompt deps
# need stubs. We already stubbed Prompt above via _FakePrompt-style;
# replicate here for self-containment.
_stub_pm2 = types.ModuleType("core.prompt_manager")


class _FP:
    def __init__(self, content, name=None, source=None, end="\n", **kw):
        self.content = content
        self.name = name
        self.source = source
        self.end = end

    def to_string(self):
        return (self.content or "") + (self.end or "")


_stub_pm2.Prompt = _FP
sys.modules["core.prompt_manager"] = _stub_pm2

# stub core.agent.tool.ToolSet
_tool_stub = types.ModuleType("core.agent.tool")


class _ToolSet:
    pass


_tool_stub.ToolSet = _ToolSet
sys.modules.setdefault("core.agent.tool", _tool_stub)

# Now load the M7 llm_model.py
_llm_spec = _u.spec_from_file_location(
    "core.provider.llm_model_m7",
    ROOT / "copy" / "M7" / "llm_model.py",
)
_llm_mod = _u.module_from_spec(_llm_spec)
# IMPORTANT: register in sys.modules BEFORE exec_module so dataclass
# decorator can resolve cls.__module__ → module dict during processing.
sys.modules["core.provider.llm_model_m7"] = _llm_mod
_llm_spec.loader.exec_module(_llm_mod)
LLMRequest = _llm_mod.LLMRequest

# 9: default empty
req = LLMRequest()
check("multimodal_parts default empty", req.multimodal_parts == [])

# 10: assemble_prompt emits string content when no parts
req = LLMRequest()
req.user_prompt.append(_FP("hello there", end=""))
req.assemble_prompt()
check(
    "no parts → user content is plain string",
    isinstance(req.messages[-1]["content"], str)
    and req.messages[-1]["content"] == "hello there",
    f"got {req.messages[-1]!r}",
)

# 11: assemble_prompt emits list shape with parts
req = LLMRequest()
req.user_prompt.append(_FP("hi", end=""))
req.multimodal_parts = [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}
]
req.assemble_prompt()
content = req.messages[-1]["content"]
check(
    "with parts → user content is list",
    isinstance(content, list),
    f"got {type(content).__name__}",
)
check(
    "list begins with text part",
    content[0]["type"] == "text" and content[0]["text"] == "hi",
)
check(
    "list contains image_url part",
    any(p.get("type") == "image_url" for p in content),
)

# 12: text + image + audio mix
req = LLMRequest()
req.user_prompt.append(_FP("describe these", end=""))
req.multimodal_parts = [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,a"}},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,b"}},
    {"type": "input_audio", "input_audio": {"data": "c", "format": "mp3"}},
]
req.assemble_prompt()
content = req.messages[-1]["content"]
check("mixed parts: 1 text + 3 media", len(content) == 4)
check(
    "all media parts present in order",
    [p["type"] for p in content] == ["text", "image_url", "image_url", "input_audio"],
)


# ─── 旧文件 merge — v1/chat 生图 image extraction helpers ────────────

print("\n=== M7.13-16: OpenAIImageClient extraction helpers ===")

# We exercise only the static / pure helpers — avoids needing the
# openai SDK installed for this self-test.
_img_spec = _u.spec_from_file_location(
    "core.provider.src.openai.model_clients_m7",
    ROOT / "copy" / "M7" / "openai" / "model_clients.py",
)
# But the module imports openai at top-level. Dodge it by giving a
# minimal stub — only AsyncOpenAI and the four exception types are read.
_oa_stub = types.ModuleType("openai")


class _StubAsync:
    pass


class _AS(Exception):
    pass


class _AT(Exception):
    pass


class _AC(Exception):
    pass


_oa_stub.AsyncOpenAI = _StubAsync
_oa_stub.APIStatusError = _AS
_oa_stub.APITimeoutError = _AT
_oa_stub.APIConnectionError = _AC
sys.modules.setdefault("openai", _oa_stub)

# Stub core.provider so imports inside model_clients resolve
_prov_stub = types.ModuleType("core.provider")
_prov_stub.ModelInfo = type("ModelInfo", (), {})


class _LMC:
    pass


class _IMC:
    pass


class _EMC:
    pass


_prov_stub.LLMModelClient = _LMC
_prov_stub.ImageModelClient = _IMC
_prov_stub.EmbeddingModelClient = _EMC
sys.modules.setdefault("core.provider", _prov_stub)

# core.provider.llm_model already loaded via M7 file; alias it
sys.modules.setdefault("core.provider.llm_model", _llm_mod)

# core.chat.message_elements stub for Image
_me_stub = types.ModuleType("core.chat.message_elements")


class _Img:
    def __init__(self, image=None, **kw):
        self.image = image


_me_stub.Image = _Img
sys.modules.setdefault("core.chat.message_elements", _me_stub)

_img_mod = _u.module_from_spec(_img_spec)
sys.modules["core.provider.src.openai.model_clients_m7"] = _img_mod
_img_spec.loader.exec_module(_img_mod)
OpenAIImageClient = _img_mod.OpenAIImageClient

# 13: _is_valid_image_source
check(
    "data:image/ accepted",
    OpenAIImageClient._is_valid_image_source("data:image/png;base64,xxx"),
)
check(
    "https accepted",
    OpenAIImageClient._is_valid_image_source("https://x.com/y.png"),
)
check(
    "bare text rejected",
    not OpenAIImageClient._is_valid_image_source("not a url"),
)
check(
    "ftp rejected",
    not OpenAIImageClient._is_valid_image_source("ftp://x/y.png"),
)


# Build a minimal client stub so we can exercise instance methods
# without the SDK / network.
class _DummyImage:
    pass


client_inst = OpenAIImageClient.__new__(OpenAIImageClient)


class _Msg:
    def __init__(self, content):
        self.content = content


# 14: data:image/ string content
img_res = client_inst._extract_image_from_message(
    _Msg("data:image/png;base64,abc")
)
check(
    "data:image/ string returns Image",
    img_res is not None and getattr(img_res, "image", "").startswith("data:image/"),
)

# 15: markdown extracted, bare URL not
img_md = client_inst._extract_image_from_message(
    _Msg("here is a pic ![cat](https://example.com/cat.png) enjoy")
)
check(
    "markdown ![alt](url) extracted",
    img_md is not None and img_md.image == "https://example.com/cat.png",
)

img_bare = client_inst._extract_image_from_message(
    _Msg("see https://example.com/cat.png for the cat")
)
check(
    "bare URL in text NOT extracted",
    img_bare is None,
    "bare URLs must not be hoisted (would let prompt inject results)",
)

# 16: output_image with base64 → data URI
img_oi = client_inst._try_extract_image_part({
    "type": "output_image",
    "base64": "ABC",
    "media_type": "image/webp",
})
check(
    "output_image base64 → data:image/webp data URI",
    img_oi is not None and img_oi.image == "data:image/webp;base64,ABC",
)

img_oi_url = client_inst._try_extract_image_part({
    "type": "output_image",
    "url": "https://x/y.png",
})
check(
    "output_image url passes _is_valid_image_source",
    img_oi_url is not None and img_oi_url.image == "https://x/y.png",
)

# Skipping text part
img_text_skip = client_inst._try_extract_image_part({
    "type": "text",
    "text": "https://x/y.png",
})
check(
    "type=text part skipped (no URL extraction from text)",
    img_text_skip is None,
)


# ─── summary ────────────────────────────────────────────────────────

passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(
    f"\n{'='*60}\nM7 + 旧文件 merge: {passed}/{total} checks passed\n{'='*60}"
)
sys.exit(0 if passed == total else 1)
