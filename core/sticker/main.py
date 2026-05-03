"""Phase 0.4 M5 — sticker plugin migrated to HintsPipeline.

Behaviour matrix (controlled by config flag ``use_hints_pipeline``):

* ``False`` (default): legacy path. ``inject_sticker_tag`` runs at
  ON_LLM_REQUEST and stuffs every registered sticker into the
  <sticker> tag's description. Prompt size grows linearly with
  sticker count — fine for small libraries, blows up at ~50+.

* ``True``: pipeline path. The <sticker> tag's description shrinks
  to a short usage hint that mentions ``<hints_key type="sticker">``.
  When the LLM emits keys, ``StickerHintsKeyHandler.prepare`` runs
  in the background, picks top-k candidates by keyword match against
  sticker descriptions, and surfaces them as a PromptBlock that
  gets injected on the next turn. The <sticker> tag itself (handle())
  is unchanged — the LLM still sends stickers via <sticker>id</sticker>;
  only the *catalog* moves out of every-turn-system-prompt.

Migration policy: keep both paths around for one week of observation
(Phase0_M3_handoff §2.4 #8). Disabling/re-enabling the flag does NOT
require a restart — the next turn picks up the new behaviour.
"""
import os
import asyncio

from pathlib import Path
from typing import Optional, Type

from core.logging_manager import get_logger
from core.plugin import BasePlugin, logger, on, Priority, register
from core.tag import BaseTag, TagSet
from core.chat import KiraMessageBatchEvent
from core.utils.common_utils import image_to_base64
from core.utils.path_utils import get_data_path
from core.chat.message_elements import BaseMessageElement, Sticker
from core.prompt_block import PromptBlock
from core.pipeline.hints_pipeline import HintsKeyHandler

message_logger = get_logger("message", "cyan")


# ── tag factories ───────────────────────────────────────────────────

def build_sticker_tag(sticker_dict: dict) -> Type[BaseTag]:
    """Legacy mode: every sticker enumerated inside the tag description.

    Used when ``use_hints_pipeline`` is False (default for now). Same
    behaviour as before M5.
    """
    def load_sticker_prompt() -> str:
        sticker_prompt = ""
        try:
            for sticker_id in sticker_dict:
                sticker_prompt += f"[{sticker_id}] {sticker_dict[sticker_id].get('desc')}\n"
            return sticker_prompt
        except Exception as e:
            message_logger.warning(f"Failed to load sticker prompt: {e}")
            return ""

    class StickerTag(BaseTag):
        name = "sticker"
        description = (
            f"<sticker>sticker_id</sticker> # 发送一个sticker（中文一般叫做"
            f"表情包）消息，通常单独在一条消息里，你需要在聊天中主动自然使用"
            f"这些sticker，可以使用的sticker id和描述如下：{load_sticker_prompt()}"
        )

        async def handle(self, value: str, **kwargs) -> list[BaseMessageElement]:
            return await _handle_sticker_value(value, sticker_dict)

    return StickerTag


def build_sticker_tag_lite(sticker_dict: dict) -> Type[BaseTag]:
    """Pipeline mode: tag description omits the catalog and instead
    instructs the LLM to request candidates via <hints_key>.

    The handle() function is identical — the LLM sends stickers the
    same way; only the prompt shape changes.
    """
    class StickerTagLite(BaseTag):
        name = "sticker"
        description = (
            "<sticker>sticker_id</sticker> # 发送一个 sticker（中文一般叫做"
            "表情包）消息，通常单独在一条消息里。可用的 sticker 列表不会一次"
            "性写在 prompt 里以节省 token——你需要主动通过 "
            "<hints_key type=\"sticker\">关键词1,关键词2</hints_key> 在你的"
            "回复中（顶层、msg 标签外）提出本轮你预计想用的语义关键词，下"
            "一轮系统会把匹配的候选注入你的上下文，届时你就能看到具体的 "
            "sticker_id 列表，然后用 <sticker>sticker_id</sticker> 发出。"
            "本轮如果你已经收到候选注入（system prompt 里出现"
            "\"以下是本轮可用的 sticker 候选\"），可以直接使用其中的 id；"
            "若上一轮预订的候选不再贴合当前话题也可以忽略它们继续回复。"
        )

        async def handle(self, value: str, **kwargs) -> list[BaseMessageElement]:
            return await _handle_sticker_value(value, sticker_dict)

    return StickerTagLite


async def _handle_sticker_value(value: str, sticker_dict: dict) -> list[BaseMessageElement]:
    sticker_id = value
    try:
        info = sticker_dict.get(sticker_id)
        if info is None:
            message_logger.warning(
                f"sticker tag: id '{sticker_id}' not in sticker_dict; "
                f"silently dropping"
            )
            return []
        sticker_path = info.get("path")
        sticker_desc = info.get("desc")
        sticker_bs64 = await image_to_base64(f"{get_data_path()}/sticker/{sticker_path}")
        sticker_obj = Sticker(sticker_id, sticker=sticker_bs64, caption=sticker_desc)
        return [sticker_obj]
    except Exception as e:
        message_logger.error(f"error while parsing sticker: {str(e)}")
        return []


# ── HintsKeyHandler ─────────────────────────────────────────────────

class StickerHintsKeyHandler(HintsKeyHandler):
    """Resolve <hints_key type="sticker"> keys into a PromptBlock that
    enumerates the top-K matching stickers for the next turn.

    Matching strategy is intentionally simple: case-insensitive substring
    match of each provided key against each sticker's ``desc``. Each
    matched sticker accumulates a hit count (per occurrence in keys);
    final ranking = (-hits, original_index) so high-relevance stickers
    surface first while ties resolve by registration order.

    No LLM/embedding lookup at this layer — keep the M5 path cheap.
    Phase 1+ may swap this for an embedding retrieval against
    ``sticker_manager``'s vector index when that exists.
    """

    key_type = "sticker"
    ttl_seconds = 900  # 15 minutes — sticker context turns over fast

    def __init__(self, ctx, top_k: int = 12):
        self.ctx = ctx
        self.top_k = max(1, int(top_k))

    async def prepare(self, keys: list[str], session_id: str, ctx: dict) -> Optional[PromptBlock]:
        sticker_dict = self.ctx.sticker_manager.sticker_dict
        if not sticker_dict:
            return None

        norm_keys = [k.strip().lower() for k in keys if k and k.strip()]
        if not norm_keys:
            return None

        # (hits, idx, sid, desc)
        scored: list[tuple[int, int, str, str]] = []
        for idx, (sid_, info) in enumerate(sticker_dict.items()):
            desc = (info or {}).get("desc") or ""
            desc_lc = desc.lower()
            hits = sum(1 for k in norm_keys if k in desc_lc)
            if hits == 0:
                continue
            scored.append((hits, idx, sid_, desc))

        if not scored:
            # No match — surface a soft signal so the LLM doesn't keep
            # asking. None means "nothing to inject"; we choose a small
            # diagnostic block instead so the LLM knows the keys missed.
            preview = ", ".join(norm_keys[:5])
            return PromptBlock(
                name="sticker_hints_no_match",
                content_provider=(lambda _ctx, _p=preview: (
                    f"## 上一轮你预订的 sticker 候选（关键词: {_p}）\n"
                    f"未匹配到任何已注册 sticker。如果当前轮仍想使用 sticker，"
                    f"请用更宽泛或不同语义的关键词再次预订；或本轮直接放弃"
                    f"使用 sticker。\n"
                )),
                depth=92,  # close to where the legacy sticker block sat
                role="system",
                source="plugin:sticker",
            )

        # Sort: more hits first, original order on tie.
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[: self.top_k]

        # Materialise the candidate listing now (closure captures it),
        # so when assembler invokes the provider next turn the content
        # is already finalised. Avoids late-bound surprises.
        lines = [f"[{sid_}] {desc}" for (_h, _i, sid_, desc) in top]
        catalog = "\n".join(lines)
        used_keys = ", ".join(norm_keys)

        def _provider(_ctx, _catalog=catalog, _used=used_keys, _n=len(top)):
            return (
                f"## 本轮可用的 sticker 候选\n"
                f"（基于上一轮你预订的关键词：{_used}；共 {_n} 个）\n"
                f"{_catalog}\n"
                f"如果话题已经偏走、这些候选不再贴合，请忽略它们；"
                f"否则用 <sticker>sticker_id</sticker> 发送。\n"
            )

        return PromptBlock(
            name="sticker_hints_candidates",
            content_provider=_provider,
            depth=92,  # just before output (depth 100), after tools (90)
            role="system",
            source="plugin:sticker",
        )


# ── plugin entry ────────────────────────────────────────────────────

class DefaultStickerPlugin(BasePlugin):
    """
    DefaultStickerPlugin
    """

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.scan_interval = 120
        self.sticker_mgr = self.ctx.sticker_manager
        self._scan_task: Optional[asyncio.Task] = None
        # Phase 0.4 M5 config
        self.use_hints_pipeline: bool = False
        self.hints_top_k: int = 12

    async def initialize(self):
        self.scan_interval = self.plugin_cfg.get("scan_interval", 120)
        # Phase 0.4 M5: read pipeline flag. Falsey-by-default keeps the
        # legacy behaviour for existing deployments — they only see the
        # new path after explicit opt-in.
        self.use_hints_pipeline = bool(self.plugin_cfg.get("use_hints_pipeline", False))
        self.hints_top_k = int(self.plugin_cfg.get("hints_top_k", 12))

        self.sticker_mgr.on_sticker_registered(self.on_sticker_registered)

        self._scan_task = asyncio.create_task(self._scan_loop())

    async def terminate(self):
        self.sticker_mgr.off_sticker_registered(self.on_sticker_registered)

        if self._scan_task:
            self._scan_task.cancel()

    async def on_sticker_registered(self, sticker_id: str, sticker_info: dict):
        path = sticker_info.get("path")
        desc = sticker_info.get("desc")

        if desc:
            return
        try:
            sticker_desc = await self.get_sticker_description(path)
            await self.sticker_mgr.update_sticker_desc(sticker_id, sticker_desc)
            logger.info(f"Sticker {sticker_id} description updated by VLM: {sticker_desc}")
        except Exception as e:
            logger.error(f"Failed to get description for sticker {sticker_id} by VLM: {e}")

    async def get_sticker_description(self, sticker_file):
        sticker_path = os.path.join(self.sticker_mgr.sticker_folder, sticker_file)

        from core.chat.message_elements import Image
        from core.utils.common_utils import desc_img

        vlm_model = self.ctx.get_default_llm_client()
        sticker_desc = await desc_img(client=vlm_model, image=Image(image=sticker_path), prompt="这是一张sticker（表情包），请描述这张表情包的内容和聊天中哪些情景使用此表情包，要求描述精确，不要太长，不要使用Markdown等标记符号，如果有文字请将其输出")

        return sticker_desc

    async def _scan_loop(self):
        try:
            while True:
                logger.info("Scanning unregistered stickers")
                sticker_files = os.listdir(self.sticker_mgr.sticker_folder)
                is_found = False
                for sticker_file in sticker_files:
                    if sticker_file not in self.sticker_mgr.sticker_paths:
                        is_found = True
                        logger.info(f"found sticker {sticker_file}")
                        sticker_description = await self.get_sticker_description(sticker_file)
                        logger.info(f"Registered sticker: {sticker_description}")
                        await self.sticker_mgr.register_sticker(sticker_file, sticker_description)

                if not is_found:
                    logger.info("All stickers are already registered")

                delay_minutes = self.scan_interval
                if delay_minutes < 10:
                    delay_minutes = 10

                await asyncio.sleep(delay_minutes * 60)
        except asyncio.CancelledError:
            logger.info("Scan loop cancelled")

    @on.llm_request(priority=Priority.SYS_HIGH - 1)
    async def inject_sticker_tag(self, event: KiraMessageBatchEvent, _, tag_set: TagSet):
        """Inject sticker tag.

        Phase 0.4 M5: which variant we register depends on the config
        flag. Both variants share the same handle() — only the tag's
        description (i.e. the prompt slot) differs.
        """
        message_types = event.message_types
        if "sticker" not in message_types:
            return
        sticker_dict = self.ctx.sticker_manager.sticker_dict
        if self.use_hints_pipeline:
            tag_set.register(build_sticker_tag_lite(sticker_dict=sticker_dict))
        else:
            tag_set.register(build_sticker_tag(sticker_dict=sticker_dict))

    # Phase 0.4 M5: HintsKeyHandler factory. The decorator records
    # (key_type='sticker', ttl_seconds override) and a reference to this
    # method; PluginManager._register_plugin_hints_handlers_for invokes
    # it after initialize() to obtain the bound handler instance and
    # registers it with HintsPipeline.
    #
    # Returns None when the pipeline path is disabled, so we don't
    # waste a slot on a handler whose key_type='sticker' would just
    # block another plugin from claiming it later. (HintsPipeline
    # tolerates None factory output — see _register_plugin_hints_handlers_for.)
    @register.hints_handler(key_type="sticker")
    def _make_sticker_hints_handler(self) -> Optional[HintsKeyHandler]:
        if not self.use_hints_pipeline:
            return None
        return StickerHintsKeyHandler(ctx=self.ctx, top_k=self.hints_top_k)
