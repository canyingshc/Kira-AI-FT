"""
Standalone smoke test for ModelGroup (Phase 0.1 M2).

Run from repo root:
  python tests/test_model_group_m2.py

Does not require sqlalchemy or any KiraAI runtime stack — only stubs
ProviderManager so model_group.py can resolve clients. Verifies:

  1. Successful call returns the response and resets state.
  2. 429 on entry-1 falls back to entry-2 with reason logged.
  3. 503 on every entry raises GroupExhaustedError.
  4. Timeout (asyncio) falls back; consecutive timeouts upgrade to long cooldown.
  5. Validation failure (min_chars) falls back without immediate cooldown,
     then cooldown after threshold reached.
  6. invalid_request (4xx other) raises immediately, no fallback.
  7. All-cooling wait window: when wait > cap, raises immediately.
  8. Capability preference: prefers tagged entry, falls back if absent.
"""
from __future__ import annotations

import asyncio
import sys
import os
import time
import types
from pathlib import Path
from typing import Any, List, Optional, Callable

# Make repo root importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# data/ must exist for the logging handler to open log.log.
(ROOT / "data").mkdir(exist_ok=True)

# Stub out modules that would otherwise drag in the full KiraAI runtime
# (sqlalchemy, adapters, etc.) — we only need ModelGroup.

# IMPORTANT: import the real `core` package first so its __path__ is set up
# correctly. Otherwise injecting a stub for `core` itself shadows the real
# package and `import core.provider.model_group` fails with "not a package".
import core  # noqa: F401

# core.chat.message_elements is imported by core.provider.provider. Stub
# Record/Image/Video so the provider module loads without touching adapters/db.
_stub_chat_pkg = types.ModuleType("core.chat")
_stub_chat_pkg.__path__ = []  # mark as package
sys.modules["core.chat"] = _stub_chat_pkg
_stub_msg_elem = types.ModuleType("core.chat.message_elements")
class _StubRecord: pass
class _StubImage: pass
class _StubVideo: pass
_stub_msg_elem.Record = _StubRecord
_stub_msg_elem.Image = _StubImage
_stub_msg_elem.Video = _StubVideo
sys.modules["core.chat.message_elements"] = _stub_msg_elem

# core.agent.tool is imported by core.provider.llm_model. The chain
# through it pulls in adapters/db that we don't have here. Provide a
# minimal stub with just the symbols llm_model.py touches.
_stub_tool = types.ModuleType("core.agent.tool")
class _StubToolSet:
    def __init__(self, *_a, **_kw): pass
    def to_list(self): return []
    def __contains__(self, _): return False
    def get(self, _): return None
class _StubToolResult:
    def __init__(self, *_a, **_kw): pass
_stub_tool.ToolSet = _StubToolSet
_stub_tool.ToolResult = _StubToolResult
_stub_agent_pkg = types.ModuleType("core.agent")
_stub_agent_pkg.__path__ = []
sys.modules["core.agent"] = _stub_agent_pkg
sys.modules["core.agent.tool"] = _stub_tool

# core.prompt_manager is imported by llm_model.py for the Prompt class.
_stub_pm = types.ModuleType("core.prompt_manager")
class _StubPrompt:
    def __init__(self, *a, **kw): pass
    def to_string(self): return ""
_stub_pm.Prompt = _StubPrompt
sys.modules["core.prompt_manager"] = _stub_pm

# core.db.service is imported by provider_manager and pulls sqlalchemy. Stub.
_stub_db_pkg = types.ModuleType("core.db")
_stub_db_pkg.__path__ = []
sys.modules["core.db"] = _stub_db_pkg
_stub_db_svc = types.ModuleType("core.db.service")
class _StubDatabaseService: pass
_stub_db_svc.DatabaseService = _StubDatabaseService
sys.modules["core.db.service"] = _stub_db_svc

# Fake openai SDK so the classifier finds the right exception types.
_fake_openai = types.ModuleType("openai")
class FakeAPIStatusError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message or f"status {status_code}")
        self.status_code = status_code
class FakeAPITimeoutError(Exception): pass
class FakeAPIConnectionError(Exception): pass
_fake_openai.APIStatusError = FakeAPIStatusError
_fake_openai.APITimeoutError = FakeAPITimeoutError
_fake_openai.APIConnectionError = FakeAPIConnectionError
sys.modules["openai"] = _fake_openai

# Now safe to import ModelGroup.
import core.provider.model_group as mg_mod
from core.provider.model_group import (
    ModelGroup, ModelEntry, GroupExhaustedError,
    ERR_RATE_LIMIT, ERR_UNAVAILABLE, ERR_TIMEOUT, ERR_VALIDATE,
    ERR_INVALID_REQUEST, ERR_AUTH, ERR_UNKNOWN,
    DEFAULT_VALIDATE_FAIL_THRESHOLD,
)
from core.provider.llm_model import LLMRequest, LLMResponse


# ─── Fake LLM client + provider manager ─────────────────────────────────────

# Inherit from the real LLMModelClient so isinstance() check inside
# ModelGroup._resolve_client passes. LLMModelClient is a plain class (not ABC),
# so we just override __init__/chat to skip ModelInfo plumbing.
from core.provider.provider import LLMModelClient as _RealLLMClient


class FakeLLMClient(_RealLLMClient):
    """Acts as an LLMModelClient. `behavior` is an async callable controlling
    what each chat() call does."""
    def __init__(self, ref: str, behavior: Callable[[LLMRequest], Any]):
        # Skip super().__init__ — ModelInfo isn't needed for these tests.
        self.ref = ref
        self._behavior = behavior
        self.call_count = 0
        # Mimic the .model.* attributes some callers reach for.
        self.model = types.SimpleNamespace(
            provider_name=ref.split(":", 1)[0],
            model_id=ref.split(":", 1)[1] if ":" in ref else ref,
        )

    async def chat(self, request: LLMRequest, **kwargs) -> LLMResponse:
        self.call_count += 1
        out = self._behavior(request)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, LLMResponse):
            return out
        return LLMResponse(text_response=str(out))


class FakeProviderManager:
    def __init__(self):
        self._clients: dict[str, FakeLLMClient] = {}

    def add(self, client: FakeLLMClient):
        self._clients[client.ref] = client

    def get_model_client(self, provider_id: str, model_id: str):
        ref = f"{provider_id}:{model_id}"
        client = self._clients.get(ref)
        # Return None-like only when explicitly missing; tests register all.
        return client


# ─── Test helpers ───────────────────────────────────────────────────────────

def make_group(
    behaviors: list[tuple[str, Callable[[LLMRequest], Any], int]],
    *,
    cooldown_429: int = 300,
    cooldown_503: int = 1800,
    validate: Optional[dict] = None,
    all_cooling_max_wait: float = 60.0,
    capabilities_per_entry: Optional[list[list[str]]] = None,
    timeouts_per_entry: Optional[list[int]] = None,
) -> tuple[ModelGroup, FakeProviderManager, list[FakeLLMClient]]:
    pm = FakeProviderManager()
    clients: list[FakeLLMClient] = []
    entries: list[ModelEntry] = []
    for i, (ref, behavior, priority) in enumerate(behaviors):
        client = FakeLLMClient(ref, behavior)
        pm.add(client)
        clients.append(client)
        entries.append(ModelEntry(
            ref=ref,
            priority=priority,
            capabilities=(capabilities_per_entry[i] if capabilities_per_entry else []),
            max_timeout=(timeouts_per_entry[i] if timeouts_per_entry else 60),
        ))
    group = ModelGroup(
        group_id="test_group",
        entries=entries,
        provider_mgr=pm,
        cooldown_429=cooldown_429,
        cooldown_503=cooldown_503,
        validate=validate,
        all_cooling_max_wait=all_cooling_max_wait,
    )
    return group, pm, clients


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []

def check(name: str, cond: bool, detail: str = ""):
    results.append((name, cond, detail))
    print(f"  {PASS if cond else FAIL} {name}{(': ' + detail) if not cond and detail else ''}")


# ─── Tests ──────────────────────────────────────────────────────────────────

async def t1_success():
    print("\n[t1] successful call")
    group, _, [c1, c2] = make_group([
        ("p:m1", lambda r: LLMResponse("hello"), 1),
        ("p:m2", lambda r: LLMResponse("world"), 2),
    ])
    resp = await group.call(LLMRequest())
    check("returns response", resp.text_response == "hello")
    check("primary used", c1.call_count == 1 and c2.call_count == 0)
    state = group._state["p:m1"]
    check("state is ok after success", state.status == "ok")


async def t2_429_fallback():
    print("\n[t2] 429 falls back, primary cools down")
    group, _, [c1, c2] = make_group([
        ("p:m1", lambda r: FakeAPIStatusError(429, "rate limited"), 1),
        ("p:m2", lambda r: LLMResponse("ok from m2"), 2),
    ])
    resp = await group.call(LLMRequest())
    check("response from m2", resp.text_response == "ok from m2")
    check("m1 called once", c1.call_count == 1)
    check("m2 called once", c2.call_count == 1)
    s1 = group._state["p:m1"]
    check("m1 is cooling", s1.status == "cooling")
    check("m1 cooldown_until_ts > now", s1.cooldown_until_ts > time.time())
    check("m1 last_error_kind=rate_limit", s1.last_error_kind == ERR_RATE_LIMIT)


async def t3_all_503_exhausts():
    print("\n[t3] every entry 503 -> GroupExhaustedError")
    group, _, _ = make_group([
        ("p:m1", lambda r: FakeAPIStatusError(503, "down"), 1),
        ("p:m2", lambda r: FakeAPIStatusError(503, "down"), 2),
    ], all_cooling_max_wait=0.01)
    raised = False
    try:
        await group.call(LLMRequest())
    except GroupExhaustedError as e:
        raised = True
        check("attempts include both refs",
              {ref for ref, _ in e.attempts} == {"p:m1", "p:m2"})
    check("raises GroupExhaustedError", raised)


async def t4_timeout_fallback_and_upgrade():
    print("\n[t4] timeout falls back; 3rd consecutive timeout escalates cooldown")
    async def slow(r):
        await asyncio.sleep(2)  # > max_timeout=1
        return LLMResponse("never")
    group, _, [c1, c2] = make_group(
        [
            ("p:m1", slow, 1),
            ("p:m2", lambda r: LLMResponse("from m2"), 2),
        ],
        cooldown_429=300, cooldown_503=1800,
        timeouts_per_entry=[1, 60],
    )
    resp = await group.call(LLMRequest())
    check("falls back on timeout", resp.text_response == "from m2")
    s1 = group._state["p:m1"]
    check("m1 marked timeout", s1.last_error_kind == ERR_TIMEOUT)
    short_window = s1.cooldown_until_ts - time.time()
    check("first timeout uses short cooldown (~300)", 250 < short_window < 350,
          f"window={short_window:.1f}")

    # Force two more timeouts directly via the cooldown helper to drive the
    # streak up to 3 without waiting in real time.
    e1 = group.entries[0]
    group._apply_cooldown(e1, ERR_TIMEOUT, "t2")
    group._apply_cooldown(e1, ERR_TIMEOUT, "t3")
    long_window = group._state["p:m1"].cooldown_until_ts - time.time()
    check("3rd consecutive timeout uses long cooldown (~1800)",
          long_window > 1500, f"window={long_window:.1f}")


async def t5_validate_threshold():
    print("\n[t5] validate_fail falls back, cools down only after threshold")
    # m1 always returns short text; m2 returns ok long text
    group, _, [c1, c2] = make_group(
        [
            ("p:m1", lambda r: LLMResponse("hi"), 1),  # 2 chars
            ("p:m2", lambda r: LLMResponse("a much longer response"), 2),
        ],
        validate={"min_chars": 10},
    )
    # First call: m1 short -> validate_fail (no cooldown), fall back to m2 (ok).
    resp = await group.call(LLMRequest())
    check("falls back to m2", resp.text_response == "a much longer response")
    s1 = group._state["p:m1"]
    check("m1 not cooling after 1st validate_fail", s1.status == "ok")
    check("m1 streak=1", s1.consecutive_validate_fail == 1)

    # Drive streak directly to threshold; verify cooldown applied.
    e1 = group.entries[0]
    for _ in range(DEFAULT_VALIDATE_FAIL_THRESHOLD - 1):
        group._apply_cooldown(e1, ERR_VALIDATE, "short")
    s1 = group._state["p:m1"]
    check(f"streak reaches {DEFAULT_VALIDATE_FAIL_THRESHOLD}",
          s1.consecutive_validate_fail >= DEFAULT_VALIDATE_FAIL_THRESHOLD)
    check("m1 cooling after threshold", s1.status == "cooling")


async def t6_invalid_request_raises():
    print("\n[t6] 400 invalid_request raises immediately, no fallback")
    group, _, [c1, c2] = make_group([
        ("p:m1", lambda r: FakeAPIStatusError(400, "bad payload"), 1),
        ("p:m2", lambda r: LLMResponse("never reached"), 2),
    ])
    raised = False
    try:
        await group.call(LLMRequest())
    except FakeAPIStatusError as e:
        raised = True
        check("status is 400", e.status_code == 400)
    check("raises immediately", raised)
    check("m2 not called", c2.call_count == 0)


async def t7_all_cooling_wait_cap():
    print("\n[t7] all entries cooling beyond wait cap -> GroupExhaustedError")
    group, _, _ = make_group([
        ("p:m1", lambda r: LLMResponse("ok"), 1),
        ("p:m2", lambda r: LLMResponse("ok"), 2),
    ], all_cooling_max_wait=0.05)
    # Manually force both into long cooldowns
    now = time.time()
    for ref in ("p:m1", "p:m2"):
        group._state[ref].status = "cooling"
        group._state[ref].cooldown_until_ts = now + 120  # 2 min, well above cap
    raised = False
    try:
        await group.call(LLMRequest())
    except GroupExhaustedError:
        raised = True
    check("raises when cooling > cap", raised)


async def t8_capability_preference():
    print("\n[t8] prefer_capability prefers tagged entry, falls back if absent")
    group, _, [c1, c2] = make_group(
        [
            ("p:m1", lambda r: LLMResponse("from m1"), 1),
            ("p:m2", lambda r: LLMResponse("from m2"), 2),
        ],
        capabilities_per_entry=[["json_output"], ["long_context"]],
    )
    # Prefer long_context: m2 should win even though m1 has higher priority.
    resp = await group.call(LLMRequest(), prefer_capability="long_context")
    check("preferred capability wins", resp.text_response == "from m2")
    check("m1 untouched", c1.call_count == 0)

    # Prefer something nobody has: should fall back to priority order.
    group2, _, [c1b, c2b] = make_group(
        [
            ("p:m1", lambda r: LLMResponse("from m1"), 1),
            ("p:m2", lambda r: LLMResponse("from m2"), 2),
        ],
        capabilities_per_entry=[["json_output"], ["long_context"]],
    )
    resp = await group2.call(LLMRequest(), prefer_capability="vision")
    check("absent capability falls back to priority", resp.text_response == "from m1")


# ─── runner ─────────────────────────────────────────────────────────────────

async def main():
    await t1_success()
    await t2_429_fallback()
    await t3_all_503_exhausts()
    await t4_timeout_fallback_and_upgrade()
    await t5_validate_threshold()
    await t6_invalid_request_raises()
    await t7_all_cooling_wait_cap()
    await t8_capability_preference()

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
