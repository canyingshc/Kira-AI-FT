"""Standalone smoke test for HintsPipeline (Phase 0.4 M5).

Run from repo root:
    python tests/test_hints_pipeline_m5.py

Doesn't require sqlalchemy / Jinja2 / the rest of the KiraAI stack.
Stubs core.prompt_manager (because prompt_block.to_legacy_prompt
imports Prompt from there) and pre-loads prompt_block + hints_pipeline
from the M3/M5 copy/ folders if they haven't been merged into core/
yet — the same trick test_prompt_block_m3.py uses.

Covers Phase0_M3_handoff §2.7 regression points:

  1.  Empty <hints_key> in LLM output → next turn no injection
  2.  Single-type single-key → next-turn collect returns the block
  3.  Same-type multiple <hints_key> tags merged (set semantics, list
      output preserving first-occurrence order)
  4.  Multiple types independent
  5.  Slow prepare > max_wait_ms → collect_blocks skips, doesn't hang
  6.  prepare() raises → no block, no exception out of the pipeline
  7.  TTL aging: pending older than handler.ttl_seconds dropped silently
  8.  Overwrite policy: same session, same type, new keys cancel old
  9.  Session isolation: sid_A pending invisible to sid_B collect
  10. Unregistered key_type → silently ignored
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path

# Make repo root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# data/ must exist for the logging handler to open log.log.
(ROOT / "data").mkdir(exist_ok=True)

# IMPORTANT: import the real `core` package first so its __path__ is
# set up correctly. We then selectively stub heavy submodules.
import core  # noqa: F401


def _load_from_copy_if_missing(mod_name: str, copy_relpath: str):
    """If ``core/<mod_name>.py`` (or sub-package) isn't on disk yet,
    load the file from copy/ and register it as ``core.<mod_name>``.
    """
    parts = mod_name.split(".")
    real_path = ROOT / "core" / Path(*parts).with_suffix(".py")
    if real_path.exists():
        return
    copy_path = ROOT / "copy" / copy_relpath
    if not copy_path.exists():
        raise RuntimeError(
            f"Neither core/{mod_name.replace('.', '/')}.py nor "
            f"copy/{copy_relpath} exists; M3/M5 sources are missing."
        )
    import importlib.util as _u
    spec = _u.spec_from_file_location(f"core.{mod_name}", copy_path)
    module = _u.module_from_spec(spec)
    sys.modules[f"core.{mod_name}"] = module
    spec.loader.exec_module(module)


# Stub core.prompt_manager BEFORE pre-loading prompt_block, because
# prompt_block.to_legacy_prompt() lazy-imports Prompt from there.
_stub_pm = types.ModuleType("core.prompt_manager")


class _FakePrompt:
    def __init__(self, content, name=None, source=None, end="\n", **kwargs):
        self.content = content
        self.name = name
        self.source = source
        self.end = end
        self.kwargs = kwargs

    def to_string(self):
        try:
            base = self.content.format(**self.kwargs) if self.kwargs else self.content
        except (KeyError, IndexError, ValueError):
            base = self.content
        if self.end:
            base += self.end
        return base


_stub_pm.Prompt = _FakePrompt
sys.modules["core.prompt_manager"] = _stub_pm


# Pre-load M3 prompt_block (HintsPipeline imports from core.prompt_block).
_load_from_copy_if_missing("prompt_block", "M3/prompt_block.py")

# Pre-load M5 pipeline package. It's a sub-package, so we register both
# the package itself and the hints_pipeline module inside it.
def _load_pipeline_from_copy_if_missing():
    real_dir = ROOT / "core" / "pipeline"
    if (real_dir / "hints_pipeline.py").exists():
        return  # already merged
    copy_dir = ROOT / "copy" / "M5" / "pipeline"
    if not (copy_dir / "hints_pipeline.py").exists():
        raise RuntimeError(
            f"copy/M5/pipeline/hints_pipeline.py missing; M5 sources "
            f"haven't been written yet"
        )
    # Register the package.
    pkg = types.ModuleType("core.pipeline")
    pkg.__path__ = [str(copy_dir)]
    sys.modules["core.pipeline"] = pkg

    import importlib.util as _u
    spec = _u.spec_from_file_location(
        "core.pipeline.hints_pipeline", copy_dir / "hints_pipeline.py"
    )
    module = _u.module_from_spec(spec)
    sys.modules["core.pipeline.hints_pipeline"] = module
    spec.loader.exec_module(module)


_load_pipeline_from_copy_if_missing()

from core.prompt_block import PromptBlock  # noqa: E402
from core.pipeline.hints_pipeline import (  # noqa: E402
    HintsKeyHandler,
    HintsPipeline,
)


# ─── result accumulator ──────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, bool(cond), detail))
    tag = PASS if cond else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


# ─── helper: a fake LLMResponse-like object ─────────────────────────


class _FakeResp:
    def __init__(self, text: str):
        self.text_response = text


# ─── helper: handlers used across multiple scenarios ─────────────────


class _RecordingHandler(HintsKeyHandler):
    """Capture every prepare() invocation for assertions, return a
    PromptBlock named after the keys received."""

    def __init__(self, key_type: str = "sticker", ttl_seconds: int = 600):
        self.key_type = key_type
        self.ttl_seconds = ttl_seconds
        self.calls: list[tuple[tuple[str, ...], str]] = []

    async def prepare(self, keys, session_id, ctx):
        self.calls.append((tuple(keys), session_id))
        # Make content carry the keys so we can assert order.
        joined = "|".join(keys)
        return PromptBlock(
            name=f"{self.key_type}_block",
            content_provider=(lambda _ctx, _j=joined, _t=self.key_type: f"{_t}: {_j}"),
            depth=92,
            role="system",
            source=f"plugin:test_{self.key_type}",
        )


class _SlowHandler(HintsKeyHandler):
    """Sleep for ``delay`` seconds before returning a block."""
    key_type = "slow"
    ttl_seconds = 600

    def __init__(self, delay: float):
        self.delay = delay

    async def prepare(self, keys, session_id, ctx):
        await asyncio.sleep(self.delay)
        return PromptBlock(
            name="slow_block",
            content_provider=(lambda _c: "slow done"),
            depth=92,
            role="system",
            source="plugin:test_slow",
        )


class _ExplodingHandler(HintsKeyHandler):
    key_type = "boom"

    async def prepare(self, keys, session_id, ctx):
        raise RuntimeError("intentional handler failure")


class _NoneHandler(HintsKeyHandler):
    key_type = "memory"

    async def prepare(self, keys, session_id, ctx):
        return None  # "nothing to inject"


# ─── scenarios ───────────────────────────────────────────────────────


async def s1_empty_hints_key():
    print("\n[1] empty hints_key in LLM output → no injection")
    p = HintsPipeline()
    p.register(_RecordingHandler())
    resp = _FakeResp("<msg><text>hello</text></msg>")  # no <hints_key>
    await p.consume_response("sid1", resp, {})
    check("no pending after consume_response", not p.has_pending("sid1"))
    blocks = await p.collect_blocks("sid1", max_wait_ms=10)
    check("collect_blocks returns []", blocks == [])


async def s2_single_type_single_key():
    print("\n[2] single-type single-key → next-turn collect returns block")
    p = HintsPipeline()
    h = _RecordingHandler("sticker")
    p.register(h)
    resp = _FakeResp(
        "<msg><text>haha</text></msg>"
        "<hints_key type=\"sticker\">happy</hints_key>"
    )
    await p.consume_response("sid2", resp, {})
    check("pending recorded for sid2", p.has_pending("sid2"))
    blocks = await p.collect_blocks("sid2", max_wait_ms=500)
    check("one block emitted", len(blocks) == 1, f"got {len(blocks)}")
    if blocks:
        content = blocks[0].content_provider({})
        check("block content carries 'happy'", "happy" in content, content[:60])
    check(
        "handler invoked once with list ['happy']",
        len(h.calls) == 1 and h.calls[0][0] == ("happy",),
        f"calls={h.calls}",
    )
    # Pending bucket consumed.
    check("pending drained after collect", not p.has_pending("sid2"))


async def s3_multi_same_type_merged():
    print("\n[3] same-type multiple <hints_key> merged, list order preserved")
    p = HintsPipeline()
    h = _RecordingHandler("sticker")
    p.register(h)
    resp = _FakeResp(
        "<hints_key type=\"sticker\">a, b</hints_key>"
        "<msg><text>x</text></msg>"
        "<hints_key type=\"sticker\">b , c</hints_key>"
    )
    await p.consume_response("sid3", resp, {})
    blocks = await p.collect_blocks("sid3", max_wait_ms=500)
    check("got 1 block (handler invoked once)", len(blocks) == 1)
    check(
        "handler called once",
        len(h.calls) == 1,
        f"calls={h.calls}",
    )
    if h.calls:
        keys = h.calls[0][0]
        # Expect ('a', 'b', 'c') — first-occurrence order, deduped.
        check(
            "keys deduped & ordered as ('a','b','c')",
            keys == ("a", "b", "c"),
            f"got {keys}",
        )


async def s4_multi_type_independent():
    print("\n[4] multiple types — handlers invoked independently")
    p = HintsPipeline()
    h_st = _RecordingHandler("sticker")
    h_mem = _RecordingHandler("memory")
    p.register(h_st)
    p.register(h_mem)
    resp = _FakeResp(
        "<hints_key type=\"sticker\">x</hints_key>"
        "<hints_key type=\"memory\">y</hints_key>"
    )
    await p.consume_response("sid4", resp, {})
    blocks = await p.collect_blocks("sid4", max_wait_ms=500)
    check("two blocks total", len(blocks) == 2, f"got {len(blocks)}")
    types_seen = sorted(b.name for b in blocks)
    check(
        "both types represented",
        types_seen == ["memory_block", "sticker_block"],
        f"types={types_seen}",
    )
    check("sticker handler got 'x'", h_st.calls and h_st.calls[0][0] == ("x",))
    check("memory handler got 'y'", h_mem.calls and h_mem.calls[0][0] == ("y",))


async def s5_slow_prepare_skipped():
    print("\n[5] slow prepare > max_wait_ms → skipped this turn, no hang")
    p = HintsPipeline()
    p.register(_SlowHandler(delay=1.0))  # 1s, well over max_wait_ms below
    resp = _FakeResp("<hints_key type=\"slow\">k</hints_key>")
    await p.consume_response("sid5", resp, {})
    t0 = time.perf_counter()
    blocks = await p.collect_blocks("sid5", max_wait_ms=100)
    elapsed = time.perf_counter() - t0
    check("collect returned promptly (< 0.5s)", elapsed < 0.5, f"{elapsed:.2f}s")
    check("no blocks (still preparing)", blocks == [])
    # Cleanup the dangling task so test doesn't leak.
    await asyncio.sleep(1.1)


async def s6_prepare_raises():
    print("\n[6] prepare() raises → no block, no exception escapes")
    p = HintsPipeline()
    p.register(_ExplodingHandler())
    resp = _FakeResp("<hints_key type=\"boom\">x</hints_key>")
    crashed = False
    try:
        await p.consume_response("sid6", resp, {})
        blocks = await p.collect_blocks("sid6", max_wait_ms=500)
    except Exception:
        crashed = True
        blocks = []
    check("consume_response + collect didn't raise", not crashed)
    check("no blocks emitted", blocks == [])


async def s7_ttl_aging():
    print("\n[7] TTL aging → expired pending dropped silently")
    p = HintsPipeline()
    h = _RecordingHandler("sticker")
    h.ttl_seconds = 0  # everything expires instantly
    p.register(h)
    resp = _FakeResp("<hints_key type=\"sticker\">x</hints_key>")
    await p.consume_response("sid7", resp, {})
    # sleep enough that ttl_seconds=0 has clearly passed.
    await asyncio.sleep(0.05)
    blocks = await p.collect_blocks("sid7", max_wait_ms=500)
    check("expired entry produced no block", blocks == [])


async def s8_overwrite_policy():
    print("\n[8] overwrite: new <hints_key> cancels in-flight old one")
    p = HintsPipeline()
    p.register(_SlowHandler(delay=1.0))
    resp1 = _FakeResp("<hints_key type=\"slow\">first</hints_key>")
    await p.consume_response("sidO", resp1, {})
    # Immediately re-emit before the first finishes.
    resp2 = _FakeResp("<hints_key type=\"slow\">second</hints_key>")
    await p.consume_response("sidO", resp2, {})
    # Confirm only one pending entry remains for the key_type.
    bucket = p._pending.get("sidO", {})
    check("one pending entry per type after overwrite", len(bucket) == 1)
    blocks = await p.collect_blocks("sidO", max_wait_ms=2000)
    # The block should have come from the second call (winning task).
    check("eventually one block returned", len(blocks) == 1)
    if blocks:
        # _SlowHandler doesn't differentiate by keys in its block content,
        # so we can only confirm we got *a* block and the test didn't hang.
        pass


async def s9_session_isolation():
    print("\n[9] session isolation: sid_A pending invisible to sid_B collect")
    p = HintsPipeline()
    p.register(_RecordingHandler("sticker"))
    resp = _FakeResp("<hints_key type=\"sticker\">k</hints_key>")
    await p.consume_response("sidA", resp, {})
    blocks_b = await p.collect_blocks("sidB", max_wait_ms=200)
    check("collect for sidB returns []", blocks_b == [])
    check("sidA still has pending", p.has_pending("sidA"))
    # Drain sidA so the test doesn't leak it.
    await p.collect_blocks("sidA", max_wait_ms=500)


async def s10_unknown_type_ignored():
    print("\n[10] unregistered key_type → silently ignored")
    p = HintsPipeline()
    # No handlers registered.
    resp = _FakeResp(
        "<hints_key type=\"unregistered_type_xyz\">a,b</hints_key>"
    )
    await p.consume_response("sidU", resp, {})
    check("no pending recorded", not p.has_pending("sidU"))
    blocks = await p.collect_blocks("sidU", max_wait_ms=10)
    check("no blocks", blocks == [])


async def s11_none_block_skipped():
    print("\n[11] handler returning None contributes no block but isn't an error")
    p = HintsPipeline()
    p.register(_NoneHandler())
    resp = _FakeResp("<hints_key type=\"memory\">k</hints_key>")
    await p.consume_response("sidN", resp, {})
    blocks = await p.collect_blocks("sidN", max_wait_ms=500)
    check("None result produces zero blocks", blocks == [])


async def s12_clear_session_cancels():
    print("\n[12] clear_session cancels in-flight prepare()")
    p = HintsPipeline()
    p.register(_SlowHandler(delay=1.0))
    resp = _FakeResp("<hints_key type=\"slow\">k</hints_key>")
    await p.consume_response("sidC", resp, {})
    check("pending before clear", p.has_pending("sidC"))
    p.clear_session("sidC")
    check("pending gone after clear", not p.has_pending("sidC"))
    # Subsequent collect returns [].
    blocks = await p.collect_blocks("sidC", max_wait_ms=10)
    check("collect after clear -> []", blocks == [])


async def s13_unregister_clears_inflight():
    print("\n[13] unregister(key_type) cancels any in-flight tasks")
    p = HintsPipeline()
    p.register(_SlowHandler(delay=1.0))
    resp = _FakeResp("<hints_key type=\"slow\">k</hints_key>")
    await p.consume_response("sidUR", resp, {})
    p.unregister("slow")
    blocks = await p.collect_blocks("sidUR", max_wait_ms=200)
    check("no blocks after unregister", blocks == [])


async def s14_clear_plugin_drops_handlers():
    print("\n[14] clear_plugin drops every handler that plugin registered")
    p = HintsPipeline()
    p.register(_RecordingHandler("aaa"), plugin_id="plug_x")
    p.register(_RecordingHandler("bbb"), plugin_id="plug_x")
    p.register(_RecordingHandler("ccc"), plugin_id="plug_y")
    p.clear_plugin("plug_x")
    types_remaining = p.list_handlers()
    check(
        "plug_y handler survives, plug_x handlers gone",
        types_remaining == ["ccc"],
        f"remaining={types_remaining}",
    )


async def s15_attribute_quotes_and_whitespace():
    print("\n[15] tag tolerates whitespace and single-quoted attributes")
    p = HintsPipeline()
    h = _RecordingHandler("sticker")
    p.register(h)
    resp = _FakeResp(
        "<hints_key   type='sticker'  >  spam ,  eggs  </hints_key>"
    )
    await p.consume_response("sidQ", resp, {})
    blocks = await p.collect_blocks("sidQ", max_wait_ms=500)
    check("got block", len(blocks) == 1)
    if h.calls:
        check(
            "keys parsed & trimmed: ('spam','eggs')",
            h.calls[0][0] == ("spam", "eggs"),
            f"got {h.calls[0][0]}",
        )


# ─── runner ──────────────────────────────────────────────────────────


async def main():
    print("HintsPipeline (Phase 0.4 M5) self-test\n" + "=" * 50)
    scenarios = [
        s1_empty_hints_key,
        s2_single_type_single_key,
        s3_multi_same_type_merged,
        s4_multi_type_independent,
        s5_slow_prepare_skipped,
        s6_prepare_raises,
        s7_ttl_aging,
        s8_overwrite_policy,
        s9_session_isolation,
        s10_unknown_type_ignored,
        s11_none_block_skipped,
        s12_clear_session_cancels,
        s13_unregister_clears_inflight,
        s14_clear_plugin_drops_handlers,
        s15_attribute_quotes_and_whitespace,
    ]
    for fn in scenarios:
        try:
            await fn()
        except Exception as e:
            check(f"{fn.__name__} crashed", False, repr(e))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"M5 result: {passed}/{total} checks passed")
    if passed != total:
        for name, ok, detail in results:
            if not ok:
                print(f"  [FAIL] {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
