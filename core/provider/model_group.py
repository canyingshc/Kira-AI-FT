"""
Model Group middleware (Phase 0.1).

A "group" wraps multiple LLM model entries (provider_id:model_id refs) so callers
can request an LLM by group_id rather than picking a specific provider/model.

Phase 0.1 milestones:
- M1: scaffolding + virtual __default__/__default_fast__ groups built from the
      existing `models.default_llm` / `models.default_fast_llm` config.
- M2 (current): per-call timeout, error classification, cooldown on 429/503/timeout,
      automatic fallback across entries, response validation, switch logging,
      "all cooling" wait window. Capability filtering is best-effort (preference,
      not hard filter).

Group definitions for user-defined groups live in `data/config/model_groups.json`.
The virtual groups are NOT defined in that file; they are derived at runtime from
system_config.json.

Error classification reference (all 4 bundled providers — OpenAI/ModelScope/
Siliconflow/Volcengine — use the openai SDK and surface the same exceptions):

  APIStatusError(429)            -> rate_limit       cooldown_429
  APIStatusError(401|403)        -> auth             cooldown_503
  APIStatusError(5xx)            -> unavailable      cooldown_503
  APIStatusError(4xx other)      -> invalid_request  raise (not a model issue)
  APITimeoutError | TimeoutError -> timeout          cooldown_429 (3x -> cooldown_503)
  APIConnectionError             -> unavailable      cooldown_503
  validation failure             -> validate_fail    no cooldown until 3x in a row
  anything else                  -> unknown          cooldown_429
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, TYPE_CHECKING

from core.logging_manager import get_logger

from .llm_model import LLMRequest, LLMResponse
from .provider import LLMModelClient

if TYPE_CHECKING:
    from core.config import KiraConfig
    from .provider_manager import ProviderManager


logger = get_logger("model_group", "purple")


# ─── Constants ───────────────────────────────────────────────────────────────

VIRTUAL_DEFAULT = "__default__"
VIRTUAL_DEFAULT_FAST = "__default_fast__"

DEFAULT_COOLDOWN_429 = 300            # seconds
DEFAULT_COOLDOWN_503 = 1800           # seconds
DEFAULT_MAX_TIMEOUT = 60              # per-call timeout cap (seconds)
DEFAULT_VALIDATE_FAIL_THRESHOLD = 3   # validate fails in a row before cooldown
DEFAULT_TIMEOUT_UPGRADE_THRESHOLD = 3 # consecutive timeouts before long cooldown
DEFAULT_ALL_COOLING_MAX_WAIT = 60.0   # max seconds to wait when every entry is cooling


# Error categories (string keys are used in switch logs and runtime state).
ERR_RATE_LIMIT = "rate_limit"
ERR_AUTH = "auth"
ERR_UNAVAILABLE = "unavailable"
ERR_INVALID_REQUEST = "invalid_request"
ERR_TIMEOUT = "timeout"
ERR_VALIDATE = "validate_fail"
ERR_UNKNOWN = "unknown"


class GroupExhaustedError(RuntimeError):
    """Raised when every entry in a group has been tried and failed (or all are
    cooling longer than the wait window). Carries per-entry reasons for debugging."""
    def __init__(self, group_id: str, attempts: List[Tuple[str, str]]):
        self.group_id = group_id
        self.attempts = attempts
        details = "; ".join(f"{ref} -> {reason}" for ref, reason in attempts)
        super().__init__(f"ModelGroup '{group_id}' exhausted: {details}")


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """One model entry inside a group."""
    ref: str                                     # "provider_id:model_id"
    priority: int = 1
    capabilities: List[str] = field(default_factory=list)
    max_timeout: int = DEFAULT_MAX_TIMEOUT

    @property
    def provider_id(self) -> str:
        return self.ref.split(":", 1)[0]

    @property
    def model_id(self) -> str:
        # model_id may itself contain ':' (e.g. some openrouter ids); keep the
        # split-once behaviour matching ProviderManager.get_default_model_info.
        parts = self.ref.split(":", 1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class _ModelRuntimeState:
    """Runtime state for one model inside one group. Not persisted."""
    status: str = "ok"                           # "ok" | "cooling"
    cooldown_until_ts: float = 0.0
    last_error: str = ""
    last_error_kind: str = ""
    last_used_ts: float = 0.0
    consecutive_validate_fail: int = 0
    consecutive_timeout: int = 0


# ─── Error classification ────────────────────────────────────────────────────

def _classify_error(exc: BaseException) -> str:
    """Map an exception to one of the ERR_* category strings.

    Kept import-free of the openai SDK at module load time so unit tests don't
    need it; resolves classes lazily.
    """
    # Lazy import — avoids a hard dependency when unit-testing without openai.
    APIStatusError = APITimeoutError = APIConnectionError = None
    try:
        from openai import APIStatusError, APITimeoutError, APIConnectionError  # type: ignore
    except Exception:
        pass

    if APIStatusError is not None and isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return ERR_RATE_LIMIT
        if status in (401, 403):
            return ERR_AUTH
        if status is not None and 500 <= status < 600:
            return ERR_UNAVAILABLE
        if status is not None and 400 <= status < 500:
            return ERR_INVALID_REQUEST
        return ERR_UNKNOWN

    if APITimeoutError is not None and isinstance(exc, APITimeoutError):
        return ERR_TIMEOUT
    if isinstance(exc, asyncio.TimeoutError):
        return ERR_TIMEOUT

    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return ERR_UNAVAILABLE

    return ERR_UNKNOWN


# ─── ModelGroup ──────────────────────────────────────────────────────────────

class ModelGroup:
    """A named bag of model entries, plus per-call resolution + cooldown logic."""

    def __init__(
        self,
        group_id: str,
        entries: List[ModelEntry],
        provider_mgr: "ProviderManager",
        cooldown_429: int = DEFAULT_COOLDOWN_429,
        cooldown_503: int = DEFAULT_COOLDOWN_503,
        validate: Optional[dict] = None,
        all_cooling_max_wait: float = DEFAULT_ALL_COOLING_MAX_WAIT,
    ):
        self.group_id = group_id
        self.entries = list(entries)
        self.provider_mgr = provider_mgr
        self.cooldown_429 = cooldown_429
        self.cooldown_503 = cooldown_503
        self.validate = validate or {}
        self.all_cooling_max_wait = all_cooling_max_wait
        # Per-entry runtime state, keyed by ref.
        self._state: Dict[str, _ModelRuntimeState] = {
            e.ref: _ModelRuntimeState() for e in entries
        }

    # ── introspection / debug ────────────────────────────────────────────

    def list_models(self) -> List[Dict[str, Any]]:
        """For WebUI/debug. Returns metadata + runtime state per entry."""
        result = []
        for e in self.entries:
            st = self._state.get(e.ref, _ModelRuntimeState())
            result.append({
                "ref": e.ref,
                "priority": e.priority,
                "capabilities": list(e.capabilities),
                "max_timeout": e.max_timeout,
                "status": st.status,
                "cooldown_until_ts": st.cooldown_until_ts,
                "last_error": st.last_error,
                "last_error_kind": st.last_error_kind,
                "last_used_ts": st.last_used_ts,
            })
        return result

    # ── candidate selection ─────────────────────────────────────────────

    def _is_cooling(self, entry: ModelEntry, now: float) -> bool:
        st = self._state.get(entry.ref)
        return bool(st and st.status == "cooling" and st.cooldown_until_ts > now)

    def _select_candidates(self, prefer_capability: Optional[str]) -> List[ModelEntry]:
        """Order entries by priority, drop cooling ones. Capability is preference,
        not hard filter — if nothing matches, fall back to ignoring it."""
        now = time.time()
        ordered = sorted(self.entries, key=lambda x: x.priority)
        live = [e for e in ordered if not self._is_cooling(e, now)]

        if prefer_capability:
            preferred = [e for e in live if prefer_capability in e.capabilities]
            if preferred:
                # preferred first, then the rest as fallback within the live set
                rest = [e for e in live if e not in preferred]
                return preferred + rest
        return live

    def _earliest_cooldown_expiry(self) -> Optional[float]:
        """Return the soonest cooldown_until_ts across all cooling entries, or None."""
        now = time.time()
        cooling = [self._state[e.ref].cooldown_until_ts
                   for e in self.entries
                   if self._is_cooling(e, now)]
        return min(cooling) if cooling else None

    # ── cooldown bookkeeping ────────────────────────────────────────────

    def _apply_cooldown(self, entry: ModelEntry, kind: str, exc_repr: str):
        """Mark an entry as cooling based on error kind. Updates streak counters
        for upgrade rules (timeout x3 -> long cooldown, validate x3 -> cooldown)."""
        st = self._state[entry.ref]
        st.last_error = exc_repr
        st.last_error_kind = kind
        now = time.time()

        if kind == ERR_RATE_LIMIT:
            st.status = "cooling"
            st.cooldown_until_ts = now + self.cooldown_429
            st.consecutive_timeout = 0
            st.consecutive_validate_fail = 0
        elif kind in (ERR_UNAVAILABLE, ERR_AUTH):
            st.status = "cooling"
            st.cooldown_until_ts = now + self.cooldown_503
            st.consecutive_timeout = 0
            st.consecutive_validate_fail = 0
        elif kind == ERR_TIMEOUT:
            st.consecutive_timeout += 1
            cd = self.cooldown_503 if st.consecutive_timeout >= DEFAULT_TIMEOUT_UPGRADE_THRESHOLD else self.cooldown_429
            st.status = "cooling"
            st.cooldown_until_ts = now + cd
            st.consecutive_validate_fail = 0
        elif kind == ERR_VALIDATE:
            st.consecutive_validate_fail += 1
            if st.consecutive_validate_fail >= DEFAULT_VALIDATE_FAIL_THRESHOLD:
                st.status = "cooling"
                st.cooldown_until_ts = now + self.cooldown_429
                # don't reset the counter; let it decay with success
        elif kind == ERR_INVALID_REQUEST:
            # Caller's fault, not the model's. Don't cooldown; counters stay put.
            pass
        else:  # ERR_UNKNOWN
            st.status = "cooling"
            st.cooldown_until_ts = now + self.cooldown_429
            st.consecutive_timeout = 0
            st.consecutive_validate_fail = 0

    def _mark_success(self, entry: ModelEntry):
        """Reset streak counters; ensure status is ok."""
        st = self._state[entry.ref]
        st.status = "ok"
        st.cooldown_until_ts = 0.0
        st.consecutive_timeout = 0
        st.consecutive_validate_fail = 0
        st.last_used_ts = time.time()

    # ── response validation ─────────────────────────────────────────────

    def _validate_response(self, resp: LLMResponse) -> Tuple[bool, str]:
        """Return (ok, reason). When validate config is empty, accept everything.

        Recognised validate config keys:
          min_chars         (int, default 0)        require text_response length
          require_xml_root  (bool, default False)   require '<' in the body
          require_json      (bool, default False)   text_response must json.loads
        Tool-call-only responses bypass min_chars check (they have no text body).
        """
        cfg = self.validate
        text = resp.text_response or ""

        # Tool-call-only responses are valid regardless of text checks.
        if resp.tool_calls and not text:
            return True, ""

        min_chars = int(cfg.get("min_chars", 0) or 0)
        if min_chars and len(text) < min_chars:
            return False, f"text shorter than min_chars={min_chars} (got {len(text)})"

        if cfg.get("require_xml_root") and "<" not in text:
            return False, "response missing XML opening tag"

        if cfg.get("require_json"):
            try:
                json.loads(text)
            except Exception as e:
                return False, f"response not parseable as JSON: {e}"

        return True, ""

    # ── public call API ─────────────────────────────────────────────────

    def _resolve_client(self, entry: ModelEntry) -> Optional[LLMModelClient]:
        try:
            client = self.provider_mgr.get_model_client(entry.provider_id, entry.model_id)
        except Exception as e:
            logger.error(f"[ModelGroup] {self.group_id}: cannot resolve {entry.ref}: {e}")
            return None
        if not isinstance(client, LLMModelClient):
            logger.error(
                f"[ModelGroup] {self.group_id}: {entry.ref} resolved to "
                f"{type(client).__name__}, expected LLMModelClient"
            )
            return None
        return client

    async def _wait_for_any_to_recover(self) -> bool:
        """If every entry is cooling, sleep until the earliest cooldown expires
        (capped at all_cooling_max_wait). Returns True if at least one entry is
        usable after the wait, False if the cap hit first."""
        expiry = self._earliest_cooldown_expiry()
        if expiry is None:
            return True
        wait = expiry - time.time()
        if wait <= 0:
            return True
        if wait > self.all_cooling_max_wait:
            return False
        logger.info(
            f"[ModelGroup] {self.group_id}: all entries cooling, waiting {wait:.1f}s "
            f"for earliest recovery"
        )
        await asyncio.sleep(wait)
        return True

    async def call(
        self,
        request: LLMRequest,
        prefer_capability: Optional[str] = None,
    ) -> LLMResponse:
        """
        Run `request` through the group, falling back across entries on failure.

        Failure handling:
          - 429 / 5xx / timeout / connection error / unknown -> cooldown the entry
            and try the next one.
          - 4xx invalid request (bad payload, etc.) -> raise immediately; not a
            model availability problem.
          - validation failure -> try next entry without cooldown unless 3-in-a-row.
        Switch reason is logged on every fallback. If every entry fails (or is
        cooling beyond the wait cap), raises GroupExhaustedError with a per-entry
        attempt log.
        """
        if not self.entries:
            raise RuntimeError(f"ModelGroup '{self.group_id}' has no model entries")

        # First selection round.
        candidates = self._select_candidates(prefer_capability)
        if not candidates:
            ok = await self._wait_for_any_to_recover()
            if not ok:
                raise GroupExhaustedError(
                    self.group_id,
                    [(e.ref, "cooling beyond wait cap") for e in self.entries],
                )
            candidates = self._select_candidates(prefer_capability)
            if not candidates:
                raise GroupExhaustedError(
                    self.group_id,
                    [(e.ref, "still cooling after wait") for e in self.entries],
                )

        attempts: List[Tuple[str, str]] = []
        last_invalid_request: Optional[BaseException] = None

        prev_ref: Optional[str] = None
        for entry in candidates:
            client = self._resolve_client(entry)
            if client is None:
                attempts.append((entry.ref, "client resolution failed"))
                if prev_ref is None:
                    prev_ref = entry.ref
                continue

            if prev_ref and prev_ref != entry.ref:
                # We're falling back from prev_ref to entry.ref; surface the reason.
                last_kind = self._state[prev_ref].last_error_kind or "unknown"
                logger.warning(
                    f"[ModelGroup] {self.group_id}: {prev_ref} → {entry.ref} "
                    f"(reason: {last_kind})"
                )
            else:
                logger.info(
                    f"[ModelGroup] {self.group_id}: using {entry.ref} "
                    f"(priority={entry.priority}, capabilities={entry.capabilities or '-'})"
                )

            self._state[entry.ref].last_used_ts = time.time()

            # Per-call timeout. Even if the underlying SDK has its own timeout,
            # we wrap defensively so a hung call cannot stall the whole pipeline.
            try:
                resp = await asyncio.wait_for(
                    client.chat(request),
                    timeout=entry.max_timeout,
                )
            except asyncio.TimeoutError as e:
                kind = ERR_TIMEOUT
                self._apply_cooldown(entry, kind, f"asyncio.TimeoutError after {entry.max_timeout}s")
                attempts.append((entry.ref, f"{kind} (>{entry.max_timeout}s)"))
                prev_ref = entry.ref
                continue
            except Exception as e:
                kind = _classify_error(e)
                exc_repr = f"{type(e).__name__}: {e}"
                if kind == ERR_INVALID_REQUEST:
                    # Caller's fault — propagate immediately.
                    self._state[entry.ref].last_error = exc_repr
                    self._state[entry.ref].last_error_kind = kind
                    last_invalid_request = e
                    attempts.append((entry.ref, f"{kind} (not a model issue)"))
                    raise
                self._apply_cooldown(entry, kind, exc_repr)
                attempts.append((entry.ref, kind))
                prev_ref = entry.ref
                continue

            # Validate
            ok, reason = self._validate_response(resp)
            if not ok:
                self._apply_cooldown(entry, ERR_VALIDATE, f"validate_fail: {reason}")
                attempts.append((entry.ref, f"validate_fail({reason})"))
                prev_ref = entry.ref
                continue

            self._mark_success(entry)
            return resp

        # If we reach here, all candidates failed. invalid_request was already raised
        # above, so this is purely the model-availability exhaustion path.
        raise GroupExhaustedError(self.group_id, attempts)

    def get_primary_client(self) -> Optional[LLMModelClient]:
        """
        Compatibility shim used by the main chat path during Phase 0.1.

        Returns a single LLMModelClient as if get_default_llm() had been called.
        Picks the first non-cooling, highest-priority entry. Callers wanting
        full group semantics (auto-fallback, cooldown, validation) should use
        ModelGroup.call() — but the AgentExecutor multi-step loop still calls
        client.chat() directly, so M2 keeps this shim for the legacy code path.

        Cooling state is consulted but not enforced strictly: if every entry is
        cooling, return the first one anyway so the caller's chat() call can
        produce a real exception rather than a silent None.
        """
        candidates = self._select_candidates(prefer_capability=None)
        if not candidates:
            # Fallback: ignore cooldown to keep the legacy path producing errors,
            # not silent failures.
            candidates = sorted(self.entries, key=lambda x: x.priority)
        if not candidates:
            return None
        return self._resolve_client(candidates[0])


# ─── ModelGroupManager ───────────────────────────────────────────────────────

class ModelGroupManager:
    """
    Manager for all groups. Holds user-defined groups loaded from
    data/config/model_groups.json plus virtual __default__/__default_fast__
    groups derived from system_config.json.

    Loaded once at startup; reload() re-reads model_groups.json without restart.
    Virtual groups are rebuilt on every get_group call so live edits to
    system_config.json take effect without restart.
    """

    def __init__(self, kira_config: "KiraConfig", provider_mgr: "ProviderManager"):
        self.kira_config = kira_config
        self.provider_mgr = provider_mgr
        self._groups: Dict[str, ModelGroup] = {}
        self._user_group_defs: Dict[str, dict] = {}
        self._load()

    # ── public API ───────────────────────────────────────────────────────

    def get_group(self, group_id: str) -> Optional[ModelGroup]:
        if group_id in (VIRTUAL_DEFAULT, VIRTUAL_DEFAULT_FAST):
            return self._build_virtual_group(group_id)
        return self._groups.get(group_id)

    def list_groups(self) -> List[str]:
        return [VIRTUAL_DEFAULT, VIRTUAL_DEFAULT_FAST] + list(self._groups.keys())

    def reload(self):
        """Re-read model_groups.json and rebuild user-defined groups. Per-entry
        runtime state (cooldowns) is dropped — reload counts as a clean slate."""
        self._groups.clear()
        self._user_group_defs.clear()
        self._load()
        logger.info(f"[ModelGroup] reloaded; user groups: {list(self._groups.keys())}")

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self):
        raw = self.kira_config.load_subconfig(
            "model_groups",
            default={"groups": {}}
        )
        groups_dict = raw.get("groups") or {}
        if not isinstance(groups_dict, dict):
            logger.warning("model_groups.json: 'groups' is not a dict, ignoring")
            groups_dict = {}

        for gid, gdef in groups_dict.items():
            if not isinstance(gdef, dict):
                logger.warning(f"model_groups.json: group '{gid}' is not a dict, skipping")
                continue
            try:
                group = self._build_user_group(gid, gdef)
                self._groups[gid] = group
                self._user_group_defs[gid] = gdef
            except Exception as e:
                logger.error(f"model_groups.json: failed to build group '{gid}': {e}")

        if self._groups:
            logger.info(f"[ModelGroup] loaded user groups: {list(self._groups.keys())}")

    def _build_user_group(self, gid: str, gdef: dict) -> ModelGroup:
        models_raw = gdef.get("models") or []
        entries: List[ModelEntry] = []
        for m in models_raw:
            if not isinstance(m, dict) or "ref" not in m:
                logger.warning(f"group '{gid}': skipping malformed entry {m}")
                continue
            entries.append(ModelEntry(
                ref=str(m["ref"]),
                priority=int(m.get("priority", 1)),
                capabilities=list(m.get("capabilities") or []),
                max_timeout=int(m.get("max_timeout", DEFAULT_MAX_TIMEOUT)),
            ))
        return ModelGroup(
            group_id=gid,
            entries=entries,
            provider_mgr=self.provider_mgr,
            cooldown_429=int(gdef.get("cooldown_429", DEFAULT_COOLDOWN_429)),
            cooldown_503=int(gdef.get("cooldown_503", DEFAULT_COOLDOWN_503)),
            validate=gdef.get("validate") or {},
            all_cooling_max_wait=float(gdef.get("all_cooling_max_wait", DEFAULT_ALL_COOLING_MAX_WAIT)),
        )

    def _build_virtual_group(self, virtual_id: str) -> Optional[ModelGroup]:
        if virtual_id == VIRTUAL_DEFAULT:
            ref = self.kira_config.get_config("models.default_llm")
        elif virtual_id == VIRTUAL_DEFAULT_FAST:
            ref = self.kira_config.get_config("models.default_fast_llm")
        else:
            return None

        if not ref or not isinstance(ref, str) or ":" not in ref:
            return None

        entry = ModelEntry(ref=ref, priority=1, capabilities=[], max_timeout=DEFAULT_MAX_TIMEOUT)
        return ModelGroup(
            group_id=virtual_id,
            entries=[entry],
            provider_mgr=self.provider_mgr,
        )
