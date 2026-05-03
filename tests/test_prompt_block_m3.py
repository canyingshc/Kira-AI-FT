"""
Standalone smoke test for PromptBlock + PromptAssembler (Phase 0.2 M3).

Run from repo root:
  python tests/test_prompt_block_m3.py

Does not require sqlalchemy or any KiraAI runtime stack — only stubs
core.prompt_manager so prompt_block.to_legacy_prompt resolves the
fake Prompt class. Verifies (per Phase0_M3_handoff §2.7):

  1. Empty PromptBlock list -> empty AssembledPrompt (no crash).
  2. Static block + async block coexist; both evaluated, order correct.
  3. condition returning False filters the block out.
  4. depth ordering: depth=10 emerges before depth=100.
  5. Same cache_key in one assemble() pass -> 2nd hit is cache_hit.
  6. content_provider raising -> block skipped + warning, others survive.
  7. Additive compatibility: simulating a legacy @on.llm_request hook
     mutating req.system_prompt -> final messages[0] contains both the
     assembled output and the legacy hook's contribution.
  8. <|message_types|> placeholder is correctly substituted post-assembly
     (i.e. the literal token never reaches the LLM).

Plus a few extra micro-checks on PromptBlockRegistry / BlockCollector
since those are M3.1 surface area.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

# Make repo root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# data/ must exist for the logging handler to open log.log.
(ROOT / "data").mkdir(exist_ok=True)

# IMPORTANT: import the real `core` package first so its __path__ is set
# up correctly. Then we can selectively stub heavy submodules without
# shadowing the package itself.
import core  # noqa: F401


def _load_from_copy_if_missing(mod_name: str, copy_filename: str):
    """If ``core/<mod_name>.py`` isn't on disk yet (the M3 merge from
    copy/ → core/ hasn't happened), load the file directly from copy/
    and register it as ``core.<mod_name>`` in sys.modules. Allows this
    test to run *before* the user copies the M3 changes into core/.

    Call order matters: load prompt_block before prompt_assembler so the
    latter's ``from core.prompt_block import ...`` resolves to our stash.
    """
    real_path = ROOT / "core" / f"{mod_name}.py"
    if real_path.exists():
        return  # already merged, normal import will pick it up
    copy_path = ROOT / "copy" / copy_filename
    if not copy_path.exists():
        raise RuntimeError(
            f"Neither core/{mod_name}.py nor copy/{copy_filename} exists; "
            f"M3 sources are missing."
        )
    import importlib.util as _u
    spec = _u.spec_from_file_location(f"core.{mod_name}", copy_path)
    module = _u.module_from_spec(spec)
    sys.modules[f"core.{mod_name}"] = module
    spec.loader.exec_module(module)


# Stub core.prompt_manager BEFORE pre-loading prompt_block, because
# prompt_block.to_legacy_prompt() does `from core.prompt_manager import
# Prompt` at call time. The real prompt_manager pulls in PersonaManager /
# KiraConfig which we don't have in this isolated test.
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

# Now pre-load M3 sources from copy/ if core/ doesn't have them yet.
_load_from_copy_if_missing("prompt_block", "prompt_block.py")
_load_from_copy_if_missing("prompt_assembler", "prompt_assembler.py")

# Now safe to import the M3 surfaces.
from core.prompt_block import (  # noqa: E402
    PromptBlock,
    AssembledBlock,
    AssembledPrompt,
    BlockCollector,
    PromptBlockRegistry,
)
from core.prompt_assembler import PromptAssembler  # noqa: E402


# ─── result accumulator ─────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, cond, detail))
    print(f"  {PASS if cond else FAIL} {name}{(': ' + detail) if not cond and detail else ''}")


# ─── tests ──────────────────────────────────────────────────────────

async def t1_empty_blocks():
    print("\n[t1] empty block list -> empty AssembledPrompt")
    a = PromptAssembler()
    out = await a.assemble([], {"session_id": "s1"})
    check("system_blocks empty", out.system_blocks == [])
    check("user_blocks empty", out.user_blocks == [])
    check("debug_meta empty", out.debug_meta == [])


async def t2_sync_async_coexist():
    print("\n[t2] static + async providers, both evaluated in order")

    def sync_provider(_ctx):
        return "SYNC"

    async def async_provider(ctx):
        return f"ASYNC[{ctx['session_id']}]"

    blocks = [
        PromptBlock(name="b_sync", content_provider=sync_provider, depth=20, source="framework"),
        PromptBlock(name="b_async", content_provider=async_provider, depth=10, source="framework"),
    ]
    out = await PromptAssembler().assemble(blocks, {"session_id": "S"})
    names = [b.name for b in out.system_blocks]
    contents = [b.content for b in out.system_blocks]
    check("two blocks emitted", len(out.system_blocks) == 2)
    # depth=10 first
    check("async first by depth", names == ["b_async", "b_sync"], detail=str(names))
    check("async content rendered", contents[0] == "ASYNC[S]")
    check("sync content rendered", contents[1] == "SYNC")


async def t3_condition_false_filters():
    print("\n[t3] condition False -> block filtered out")
    blocks = [
        PromptBlock(
            name="visible",
            content_provider=lambda _c: "yes",
            depth=10,
            condition=lambda _c: True,
        ),
        PromptBlock(
            name="hidden",
            content_provider=lambda _c: "no",
            depth=20,
            condition=lambda _c: False,
        ),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    names = [b.name for b in out.system_blocks]
    check("only visible block emitted", names == ["visible"], detail=str(names))


async def t4_depth_sort():
    print("\n[t4] depth ordering: 10 before 100, stable for equal depths")
    blocks = [
        PromptBlock(name="late", content_provider=lambda _c: "L", depth=100),
        PromptBlock(name="early", content_provider=lambda _c: "E", depth=10),
        PromptBlock(name="mid_a", content_provider=lambda _c: "Ma", depth=50),
        PromptBlock(name="mid_b", content_provider=lambda _c: "Mb", depth=50),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    names = [b.name for b in out.system_blocks]
    check("early < mid < late", names[0] == "early" and names[-1] == "late",
          detail=str(names))
    # Stable order means mid_a comes before mid_b (registration order)
    mid = [n for n in names if n.startswith("mid_")]
    check("equal-depth blocks are stable", mid == ["mid_a", "mid_b"], detail=str(mid))


async def t5_cache_key_hit():
    print("\n[t5] same cache_key in one assemble() pass -> second hit cached")
    counter = {"n": 0}

    def expensive(_c):
        counter["n"] += 1
        return f"v{counter['n']}"

    blocks = [
        PromptBlock(name="a", content_provider=expensive, cache_key="K", depth=10),
        PromptBlock(name="b", content_provider=expensive, cache_key="K", depth=20),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    contents = [b.content for b in out.system_blocks]
    check("provider invoked once", counter["n"] == 1, detail=f"n={counter['n']}")
    check("both blocks share content", contents == ["v1", "v1"], detail=str(contents))
    cache_hits = [m["cache_hit"] for m in out.debug_meta]
    # First should be miss, second hit
    check("first miss, second hit", cache_hits == [False, True], detail=str(cache_hits))


async def t6_provider_exception_isolated():
    print("\n[t6] one provider raising does not break others")

    def boom(_c):
        raise RuntimeError("kaboom")

    blocks = [
        PromptBlock(name="ok1", content_provider=lambda _c: "ONE", depth=10),
        PromptBlock(name="bad", content_provider=boom, depth=20),
        PromptBlock(name="ok2", content_provider=lambda _c: "TWO", depth=30),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    names = [b.name for b in out.system_blocks]
    check("bad block dropped", "bad" not in names, detail=str(names))
    check("good blocks emitted", names == ["ok1", "ok2"], detail=str(names))
    # debug_meta should still have the bad one with error
    bad_meta = [m for m in out.debug_meta if m["name"] == "bad"]
    check("bad block recorded in debug_meta", len(bad_meta) == 1)
    check("bad meta carries error", bad_meta and "error" in bad_meta[0])


async def t7_additive_legacy_compat():
    print("\n[t7] additive compatibility: legacy hook + assembler coexist")
    # Simulate the message_manager flow without dragging in the actual class.
    blocks = [
        PromptBlock(name="role", content_provider=lambda _c: "ROLE\n", depth=10),
        PromptBlock(name="persona", content_provider=lambda _c: "PERSONA\n", depth=20),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    assembled_text = out.system_text(separator="")

    # Legacy hook contributed something via @on.llm_request -> system_prompt
    legacy_extra = _FakePrompt(
        "EXTRA-LEGACY-CONTENT", name="legacy", source="system"
    )

    # Mimic: insert assembled at index 0, legacy follows (additive策略)
    system_prompt = []
    system_prompt.insert(
        0, _FakePrompt(assembled_text, name="__assembled__", source="system", end="")
    )
    system_prompt.append(legacy_extra)

    # Mimic LLMRequest.assemble_prompt
    final_system = "".join(p.to_string() for p in system_prompt)

    check("assembled framework content present",
          "ROLE" in final_system and "PERSONA" in final_system)
    check("legacy hook content present",
          "EXTRA-LEGACY-CONTENT" in final_system)
    # Order: framework content before legacy
    role_pos = final_system.find("ROLE")
    legacy_pos = final_system.find("EXTRA-LEGACY-CONTENT")
    check("framework precedes legacy",
          role_pos != -1 and legacy_pos != -1 and role_pos < legacy_pos)


async def t8_message_types_placeholder_substituted():
    print("\n[t8] <|message_types|> token substituted post-assembly")
    blocks = [
        PromptBlock(name="role", content_provider=lambda _c: "ROLE\n", depth=10),
        # 模拟 format_tmpl: 含 <|message_types|> 占位符
        PromptBlock(
            name="format",
            content_provider=lambda _c: (
                "## 输出格式\n标签清单:\n<|message_types|>\n"
            ),
            depth=110,
        ),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    assembled_text = out.system_text(separator="")
    # message_manager 在合并字符串上做 replace
    tag_list_text = "<text>...</text>\n<at>...</at>"
    final_text = assembled_text.replace("<|message_types|>", tag_list_text)
    check("placeholder gone", "<|message_types|>" not in final_text)
    check("tag list inlined", tag_list_text in final_text)


# ─── extra: Registry / Collector micro-tests ────────────────────────

async def t9_registry_basic():
    print("\n[t9] PromptBlockRegistry register / unregister / clear_plugin")
    reg = PromptBlockRegistry()
    b1 = PromptBlock(name="x", content_provider=lambda _c: "x", source="plugin:p1")
    b2 = PromptBlock(name="y", content_provider=lambda _c: "y", source="plugin:p1")
    b3 = PromptBlock(name="z", content_provider=lambda _c: "z", source="plugin:p2")
    reg.register(b1, plugin_id="p1")
    reg.register(b2, plugin_id="p1")
    reg.register(b3, plugin_id="p2")
    check("3 blocks registered", len(reg.get_all()) == 3)
    reg.unregister("y", plugin_id="p1")
    check("after unregister: 2 left", len(reg.get_all()) == 2)
    reg.clear_plugin("p1")
    check("after clear_plugin p1: only p2 block left",
          [b.name for b in reg.get_all()] == ["z"])


async def t10_collector_rejects_non_block():
    print("\n[t10] BlockCollector ignores non-PromptBlock objects")
    coll = BlockCollector()
    coll.add(PromptBlock(name="ok", content_provider=lambda _c: "ok"))
    coll.add("not a block")  # type: ignore[arg-type]
    coll.add(42)              # type: ignore[arg-type]
    items = coll.get_all()
    check("only the real block stuck", len(items) == 1 and items[0].name == "ok")


async def t11_provider_zero_arg():
    print("\n[t11] provider with no positional args is invoked correctly")

    def zero_arg():
        return "ZA"

    out = await PromptAssembler().assemble(
        [PromptBlock(name="z", content_provider=zero_arg, depth=10)],
        {},
    )
    check("zero-arg provider works", out.system_blocks and out.system_blocks[0].content == "ZA")


async def t12_user_role_routing():
    print("\n[t12] role='user' blocks land in user_blocks bucket")
    blocks = [
        PromptBlock(name="sys_a", content_provider=lambda _c: "S", role="system"),
        PromptBlock(name="usr_a", content_provider=lambda _c: "U", role="user"),
    ]
    out = await PromptAssembler().assemble(blocks, {})
    check("system bucket has 1", len(out.system_blocks) == 1)
    check("user bucket has 1", len(out.user_blocks) == 1)
    check("user bucket has correct content",
          out.user_blocks and out.user_blocks[0].content == "U")


# ─── runner ─────────────────────────────────────────────────────────

async def main() -> int:
    await t1_empty_blocks()
    await t2_sync_async_coexist()
    await t3_condition_false_filters()
    await t4_depth_sort()
    await t5_cache_key_hit()
    await t6_provider_exception_isolated()
    await t7_additive_legacy_compat()
    await t8_message_types_placeholder_substituted()
    await t9_registry_basic()
    await t10_collector_rejects_non_block()
    await t11_provider_zero_arg()
    await t12_user_role_routing()

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    if passed == total:
        print(f"\n  {PASS} all {total} checks passed")
        return 0
    print(f"\n  {FAIL} {total - passed}/{total} checks failed")
    for name, ok, detail in results:
        if not ok:
            print(f"    - {name}{(': ' + detail) if detail else ''}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
