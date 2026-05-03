"""
Standalone smoke test for PromptTranslator + Jinja2 templates (Phase 0.3 M4).

Run from repo root:
  python tests/test_translator_m4.py

Prerequisite: `pip install Jinja2>=3.1.0` (added to requirements.txt by M4).
The script gates on it and bails out with a clear message if missing.

Loads M4 sources directly from copy/M4/ when core/ doesn't have them
yet (mirrors test_prompt_block_m3.py's pattern), so this can run
*before* the M4 → core merge.

Verifies (per plan):

  1. Single template renders with {{ var }} substitution.
  2. chat_env nested attribute access works (Jinja2 dict-as-object).
  3. Missing variables don't crash (ChainableUndefined → empty).
  4. format.j2's {{ message_types }} injection.
  5. Missing template file logs warning + returns "" (does not raise).
  6. Translator.render(assembled, ...) preserves block order.
  7. debug_markers=true wraps blocks with XML comments.

Plus three M4-specific behavioural checks:

  8. autoescape=False — `<msg>` survives unescaped.
  9. keep_trailing_newline=True — block separators preserved.
 10. PromptTranslator.list_templates returns the 11 expected names.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

(ROOT / "data").mkdir(exist_ok=True)

# Hard-fail with a clear message if Jinja2 isn't installed; otherwise
# the import inside PromptTranslator throws an opaque ImportError and
# the user has to dig.
try:
    import jinja2  # noqa: F401
except ImportError:
    print("ERROR: Jinja2 is not installed. Install with:")
    print("  pip install Jinja2>=3.1.0")
    sys.exit(2)

import core  # noqa: F401  (force the real package on sys.path)


# ── Stub core.prompt_manager (Prompt class only) ─────────────────────

_stub_pm = types.ModuleType("core.prompt_manager")


class _FakePrompt:
    def __init__(self, content, name=None, source=None, end="\n", **kw):
        self.content = content
        self.name = name
        self.source = source
        self.end = end
        self.kwargs = kw

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


# ── Pre-load copy/M4 + copy/M3 sources if core/ is bare ─────────────

def _load_from_copy_if_missing(mod_name: str, copy_subpath: str, m: str = "M4"):
    """Load <ROOT>/copy/<m>/<copy_subpath> as core.<mod_name> in sys.modules
    iff <ROOT>/core/<mod_name>.py doesn't exist yet. Lets this test run
    before the M3/M4 merges into core/."""
    real_path = ROOT / "core" / f"{mod_name}.py"
    if real_path.exists():
        return
    copy_path = ROOT / "copy" / m / copy_subpath
    if not copy_path.exists():
        # M4 derivatives (e.g. prompt_block) live in copy/M3; fall back.
        copy_path = ROOT / "copy" / "M3" / copy_subpath
    if not copy_path.exists():
        raise RuntimeError(
            f"Neither core/{mod_name}.py nor copy/{m}/{copy_subpath} nor "
            f"copy/M3/{copy_subpath} exists; M4 sources missing."
        )
    import importlib.util as _u
    spec = _u.spec_from_file_location(f"core.{mod_name}", copy_path)
    module = _u.module_from_spec(spec)
    sys.modules[f"core.{mod_name}"] = module
    spec.loader.exec_module(module)


# Order matters: prompt_block first (translator doesn't import it but
# the test does), then prompt_translator.
_load_from_copy_if_missing("prompt_block", "prompt_block.py")
_load_from_copy_if_missing("prompt_translator", "prompt_translator.py")

from core.prompt_block import (  # noqa: E402
    AssembledBlock,
    AssembledPrompt,
)
from core.prompt_translator import PromptTranslator, RenderedPrompt  # noqa: E402


# ── Test fixtures ───────────────────────────────────────────────────

# Default to copy/M4/translator if core/prompts/translator doesn't exist.
DEFAULT_TEMPLATE_DIR = ROOT / "core" / "prompts" / "translator"
COPY_TEMPLATE_DIR = ROOT / "copy" / "M4" / "translator"
TEMPLATE_DIR = DEFAULT_TEMPLATE_DIR if DEFAULT_TEMPLATE_DIR.exists() else COPY_TEMPLATE_DIR


class FakeConfig:
    """Minimal stand-in for KiraConfig: only `load_subconfig` is exercised."""

    def __init__(self, sub: dict | None = None):
        self._sub = sub or {}

    def load_subconfig(self, name: str, default=None):
        return dict(self._sub.get(name, default or {}))


def _make_translator(debug_markers: bool = False, template_dir: Path | None = None) -> PromptTranslator:
    cfg = FakeConfig({
        "translator": {
            "template_dir": "",
            "debug_markers": debug_markers,
        }
    })
    return PromptTranslator(cfg, template_dir=template_dir or TEMPLATE_DIR)


# ── result accumulator ─────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    results.append((name, cond, detail))
    print(f"  {PASS if cond else FAIL} {name}{(': ' + detail) if not cond and detail else ''}")


# ── tests ──────────────────────────────────────────────────────────

def t1_single_template_render():
    print("\n[t1] single template renders with {{ var }} substitution")
    t = _make_translator()
    out = t.render_template("persona", {"persona": "Alice the AI"})
    check("persona text present", "Alice the AI" in out, detail=repr(out)[:80])
    check("template title preserved", "## 角色扮演" in out)


def t2_chat_env_nested():
    print("\n[t2] chat_env nested attribute access ({{ chat_env.platform }})")
    t = _make_translator()
    chat_env = {
        "platform": "qq",
        "adapter": "test_ada",
        "chat_type": "GroupMessage",
        "self_id": "100200",
        "session_title": "测试群",
        "session_description": "desc",
    }
    out = t.render_template("chat_env", {"chat_env": chat_env})
    check("platform inlined", "qq" in out)
    check("adapter inlined", "test_ada" in out)
    check("session_title inlined", "测试群" in out)


def t3_missing_variable_silent():
    print("\n[t3] missing variable does not crash (ChainableUndefined)")
    t = _make_translator()
    # persona.j2 references {{ persona }} but we pass nothing
    out = t.render_template("persona", {})
    check("render returned a string", isinstance(out, str))
    check("title still present", "## 角色扮演" in out)
    # The {{ persona }} expansion should be empty (or chainable-undefined),
    # not raise.
    check("no UndefinedError surfaced", "UndefinedError" not in out)


def t4_message_types_injection():
    print("\n[t4] format.j2 injects {{ message_types }}")
    t = _make_translator()
    fake_tag_descs = "<text>...</text>\n<at>...</at>\n<emoji>...</emoji>"
    out = t.render_template("format", {"message_types": fake_tag_descs})
    check("tag list inlined", fake_tag_descs in out)
    check("no token leak", "<|message_types|>" not in out)
    check("format title preserved", "## 输出格式" in out)


def t5_missing_template_returns_empty():
    print("\n[t5] missing template logs warning + returns empty string")
    t = _make_translator()
    out = t.render_template("definitely_does_not_exist_xyz", {})
    check("returns string", isinstance(out, str))
    check("returns empty", out == "")


def t6_render_preserves_order():
    print("\n[t6] Translator.render preserves block order")
    t = _make_translator()
    assembled = AssembledPrompt(
        system_blocks=[
            AssembledBlock(name="role", depth=10, content="A\n", role="system", source="framework"),
            AssembledBlock(name="persona", depth=20, content="B\n", role="system", source="framework"),
            AssembledBlock(name="format", depth=110, content="C\n", role="system", source="framework"),
        ],
    )
    rendered = t.render(assembled, tag_set=None, ctx_snapshot={})
    check("returns RenderedPrompt", isinstance(rendered, RenderedPrompt))
    check("system_text concatenates in order",
          rendered.system_text == "A\nB\nC\n",
          detail=repr(rendered.system_text)[:80])


def t7_debug_markers():
    print("\n[t7] debug_markers=True wraps blocks with XML comments")
    t = _make_translator(debug_markers=True)
    assembled = AssembledPrompt(
        system_blocks=[
            AssembledBlock(name="role", depth=10, content="ROLE\n", role="system", source="framework"),
        ],
    )
    rendered = t.render(assembled, tag_set=None, ctx_snapshot={})
    check("opening marker present", "<!-- block:role" in rendered.system_text)
    check("closing marker present", "<!-- /block:role -->" in rendered.system_text)
    check("source attr present", "source=framework" in rendered.system_text)


def t8_autoescape_off():
    print("\n[t8] autoescape=False — <msg>/<text> survive unescaped")
    t = _make_translator()
    out = t.render_template("format", {"message_types": "<text>...</text>"})
    check("<msg> survived", "<msg>" in out)
    check("</msg> survived", "</msg>" in out)
    check("no &lt;", "&lt;" not in out)


def t9_keep_trailing_newline():
    print("\n[t9] keep_trailing_newline=True — block ends with newline")
    t = _make_translator()
    out = t.render_template("role", {})
    check("ends with newline", out.endswith("\n"), detail=repr(out)[-30:])


def t10_list_templates_returns_eleven():
    print("\n[t10] list_templates returns the 11 expected framework names")
    t = _make_translator()
    names = set(t.list_templates())
    expected = {"role", "persona", "attention", "accounts", "sessions",
                "time", "chat_env", "memory", "tools", "output", "format"}
    missing = expected - names
    extra = names - expected
    check("11 expected templates present", not missing, detail=f"missing={missing}")
    # Not strict on extras — future Phases may add more (silence.j2, hints.j2, etc.).
    if extra:
        print(f"    note: extra templates found: {extra} (informational, not a failure)")


def t11_user_blocks_routed():
    print("\n[t11] user_blocks bucket carried through render()")
    t = _make_translator()
    assembled = AssembledPrompt(
        system_blocks=[AssembledBlock(name="s", depth=10, content="SYS\n", role="system", source="framework")],
        user_blocks=[AssembledBlock(name="u", depth=10, content="USR\n", role="user", source="framework")],
    )
    rendered = t.render(assembled, tag_set=None, ctx_snapshot={})
    check("system_text correct", rendered.system_text == "SYS\n")
    check("user_text correct", rendered.user_text == "USR\n")


# ── runner ────────────────────────────────────────────────────────

def main() -> int:
    t1_single_template_render()
    t2_chat_env_nested()
    t3_missing_variable_silent()
    t4_message_types_injection()
    t5_missing_template_returns_empty()
    t6_render_preserves_order()
    t7_debug_markers()
    t8_autoescape_off()
    t9_keep_trailing_newline()
    t10_list_templates_returns_eleven()
    t11_user_blocks_routed()

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
    sys.exit(main())
