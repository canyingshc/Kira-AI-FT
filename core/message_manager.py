import asyncio
import json
import time
from copy import deepcopy
from asyncio import Lock
import xml.etree.ElementTree as ET
from typing import Union, Any, List, Optional
from pathlib import Path
from asyncio import Semaphore
import random
import os

from core.logging_manager import get_logger
from core.utils.common_utils import desc_img
from core.utils.path_utils import get_data_path
from core.chat.message_utils import KiraMessageEvent, KiraMessageBatchEvent, KiraCommentEvent, MessageChain
from core.chat.message_utils import KiraIMSentResult, KiraStepResult
from core.prompt_manager import Prompt
from core.prompt_block import PromptBlock, BlockCollector

from core.chat.message_elements import (
    BaseMessageElement,
    Text,
    Image,
    At,
    Reply,
    Forward,
    Emoji,
    Sticker,
    Record,
    Notice,
    Poke,
    File,
    Video
)

from core.llm_client import LLMClient
from core.chat.session_manager import SessionManager
from .prompt_manager import PromptManager
from .adapter import AdapterManager
from .agent.skills_mgr import SkillsManager
from .provider import ProviderManager, LLMRequest, LLMResponse
from core.plugin.plugin_handlers import event_handler_reg, EventType
from core.agent.agent_executor import AgentExecutor, AgentExecutionContext, NewMemory
from core.agent.tool import ToolSet
from core.tag import tag_registry, TagSet
from core.db.service import DatabaseService
from core.output.output_ctx import OutputCtx

logger = get_logger("message", "cyan")
llm_logger = get_logger("llm", "purple")


class SessionBuffer:
    def __init__(self, max_count: int = None):
        self.buffer: list = []
        self.lock: asyncio.Lock = asyncio.Lock()
        self.max_count = max_count

    def add(self, message: KiraMessageEvent):
        self.buffer.append(message)

    def pop(self, count: int = 1):
        if self.get_length() < count:
            popped = self.buffer[:]
            self.buffer.clear()
            return popped
        popped = self.buffer[:count]
        del self.buffer[:count]
        return popped

    def flush(self, count: int = None):
        if count and count <= len(self.buffer):
            pending_messages = self.buffer[:count]
            del self.buffer[:count]
        else:
            pending_messages = self.buffer[:]
            self.buffer.clear()
        return pending_messages

    def get_length(self):
        return len(self.buffer)

    def get_buffer_lock(self) -> Lock:
        """get buffer lock"""
        return self.lock


class SessionBufferManager:
    def __init__(self, max_count: int = None):
        self.buffers: dict[str, SessionBuffer] = {}
        self.max_count = max_count

    def get_buffer(self, session: str):
        if session not in self.buffers:
            self.buffers[session] = SessionBuffer(self.max_count)
        return self.buffers[session]


class ImageDescCache:
    """Cache image/sticker VLM descriptions using MD5 hash backed by database."""

    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def get(self, md5: str) -> Optional[str]:
        entry = await self.db.get_image_desc_cache(md5)
        if entry:
            await self.db.update_image_desc_cache(
                md5,
                count=entry["count"] + 1,
                last_seen=int(time.time()),
            )
            return entry["description"]
        return None

    async def set(self, md5: str, description: str):
        existing = await self.db.get_image_desc_cache(md5)
        if existing:
            await self.db.update_image_desc_cache(
                md5,
                description=description,
                count=1,
                last_seen=int(time.time()),
            )
        else:
            await self.db.add_image_desc_cache(
                md5,
                description,
                count=1,
                last_seen=int(time.time()),
            )


class MessageProcessor:
    """Core message processor, responsible for handling all message sending and receiving logic"""

    def __init__(self,
                 db: DatabaseService,
                 kira_config,
                 llm_api: LLMClient,
                 provider_manager: ProviderManager,
                 skills_manager: SkillsManager,
                 adapter_manager: AdapterManager,
                 session_manager: SessionManager,
                 prompt_manager: PromptManager,
                 max_concurrent_messages: int = 3):
        self.db = db
        self.kira_config = kira_config
        self.bot_config = kira_config["bot_config"].get("bot")
        self.max_message_interval = float(self.bot_config.get("max_message_interval"))
        self.max_buffer_messages = int(self.bot_config.get("max_buffer_messages"))
        self.min_message_delay = float(self.bot_config.get("min_message_delay", "0.8"))
        self.max_message_delay = float(self.bot_config.get("max_message_delay", "1.5"))

        self.llm_api = llm_api

        self.message_processing_semaphore = Semaphore(max_concurrent_messages)

        # managers
        self.session_manager = session_manager
        self.prompt_manager = prompt_manager
        self.provider_mgr = provider_manager
        self.adapter_mgr = adapter_manager
        self.skills_manager = skills_manager

        # message buffer
        self.session_locks: dict[str, asyncio.Lock] = {}

        self.session_buffer = SessionBufferManager(max_count=self.max_buffer_messages)

        # image description cache
        self.image_desc_cache = ImageDescCache(db)

        # Phase 0.4 M5: HintsPipeline reference, wired in by lifecycle after
        # construction (kept Optional so degraded test harnesses don't break).
        # When None, both collect_blocks (next-turn injection) and
        # consume_response (this-turn ingest) become no-ops — the system
        # still runs, just without hints_key support.
        self.hints_pipeline = None

        # ── Phase 0 M7-merge (旧文件中已经做出的更改 → multimodal) ──
        # Native multimodal toggles. Default OFF so behaviour matches
        # pre-M7 (text-only flow with VLM-described images via
        # core.utils.common_utils.desc_img + ImageDescCache). Flip via
        # bot_config.bot.use_native_multimodal etc to send the raw
        # base64 directly to the LLM as OpenAI vision content parts.
        # Each flag is independent so a deployment can mix-and-match
        # (e.g. native images but STT for audio, or native audio +
        # VLM-described video).
        bot_cfg = self.bot_config or {}
        self.use_native_multimodal: bool = bool(
            bot_cfg.get("use_native_multimodal", False)
        )
        self.use_native_audio_multimodal: bool = bool(
            bot_cfg.get("use_native_audio_multimodal", False)
        )
        self.use_native_video_multimodal: bool = bool(
            bot_cfg.get("use_native_video_multimodal", False)
        )
        self.use_video_description: bool = bool(
            bot_cfg.get("use_video_description", False)
        )
        # Configurable VLM prompt for video description fallback.
        # Read off models.video_description_prompt as well so we honour
        # both the bot-level and model-level locations.
        self.video_description_prompt: str = (
            self.kira_config.get_config(
                "bot_config.models.video_description_prompt"
            )
            or "描述这个视频的内容，包括画面中发生了什么"
        )
        if self.use_native_multimodal:
            logger.info(
                "Native multimodal mode ENABLED — images sent as base64 "
                "to LLM (bypasses VLM description / desc cache)"
            )
        if self.use_native_audio_multimodal:
            logger.info(
                "Native audio multimodal mode ENABLED — audio sent "
                "directly to LLM (bypasses STT)"
            )
        if self.use_native_video_multimodal:
            logger.info(
                "Native video multimodal mode ENABLED — video sent "
                "directly to LLM"
            )

        logger.info("MessageProcessor initialized")

    def get_session_lock(self, sid: str) -> Lock:
        """get session lock to avoid sending message simultaneously"""
        if sid not in self.session_locks:
            self.session_locks[sid] = asyncio.Lock()
        return self.session_locks[sid]

    def get_session_buffer_length(self, sid: str) -> int:
        buffer = self.session_buffer.get_buffer(sid)
        return buffer.get_length()

    async def pop_session_messages(self, sid: str, count: int = 1):
        buffer = self.session_buffer.get_buffer(sid)
        buffer.pop(count)

    async def flush_session_messages(self, sid: str, extra_event: KiraMessageEvent | None = None) -> bool:
        buffer = self.session_buffer.get_buffer(sid)
        async with buffer.lock:
            if extra_event is not None:
                buffer.add(extra_event)
            pending_messages: list[KiraMessageEvent] = buffer.flush()
        if not pending_messages:
            return False
        last_event = pending_messages[-1]
        batch_msg = KiraMessageBatchEvent(
            message_types=last_event.message_types,
            timestamp=int(time.time()),
            adapter=last_event.adapter,
            session=last_event.session,
            messages=[m.message for m in pending_messages]
        )
        await self.handle_im_batch_message(batch_msg)
        return True

    async def message_format_to_text(self, message_chain: MessageChain):
        """将平台使用标准消息格式封装的消息转换为LLM可以接收的字符串"""
        message_str = ""
        for ele in message_chain:
            if isinstance(ele, Text):
                message_str += ele.text
            elif isinstance(ele, Emoji):
                if ele.emoji_desc:
                    message_str += f"[Emoji {ele.emoji_desc} (ID: {ele.emoji_id})]"
                else:
                    message_str += f"[Emoji {ele.emoji_id}]"
            elif isinstance(ele, At):
                if ele.nickname:
                    message_str += f"[At {ele.pid}(nickname: {ele.nickname})]"
                else:
                    message_str += f"[At {ele.pid}]"
            elif isinstance(ele, Image):
                if ele.caption is None:
                    try:
                        md5 = await ele.hash_image()
                        cached_desc = await self.image_desc_cache.get(md5)
                    except (ValueError, Exception) as e:
                        logger.warning(f"Failed to hash image: {e}")
                        md5 = None
                        cached_desc = None
                    if cached_desc:
                        img_desc = cached_desc
                    else:
                        vlm_model = self.provider_mgr.get_default_vlm()
                        img_desc = await desc_img(client=vlm_model, image=ele)
                        if md5:
                            await self.image_desc_cache.set(md5, img_desc)
                    ele.caption = img_desc
                else:
                    try:
                        md5 = await ele.hash_image()
                        cached = await self.image_desc_cache.get(md5)
                        if not cached:
                            await self.image_desc_cache.set(md5, ele.caption)
                    except Exception as e:
                        logger.warning(f"Failed to cache image desc: {e}")
                message_str += f"[Image {str(ele.caption)}]"
            elif isinstance(ele, Sticker):
                if ele.caption is None:
                    try:
                        md5 = await ele.hash_image()
                        cached_desc = await self.image_desc_cache.get(md5)
                    except (ValueError, Exception) as e:
                        logger.warning(f"Failed to hash sticker: {e}")
                        md5 = None
                        cached_desc = None
                    if cached_desc:
                        sticker_desc = cached_desc
                    else:
                        vlm_model = self.provider_mgr.get_default_vlm()
                        sticker_desc = await desc_img(client=vlm_model, image=ele)
                        if md5:
                            await self.image_desc_cache.set(md5, sticker_desc)
                    ele.caption = sticker_desc
                else:
                    try:
                        md5 = await ele.hash_image()
                        cached = await self.image_desc_cache.get(md5)
                        if not cached:
                            await self.image_desc_cache.set(md5, ele.caption)
                    except Exception as e:
                        logger.warning(f"Failed to cache sticker desc: {e}")
                message_str += f"[Sticker {str(ele.caption)}]"
            elif isinstance(ele, Reply):
                if ele.chain:
                    ele.chain.message_list = [x for x in ele.chain if not isinstance(x, Reply)]
                    reply_content = await self.message_format_to_text(ele.chain)
                    message_str += f"[Reply ID: {ele.message_id} content: {reply_content}]"
                elif ele.message_content:
                    message_str += f"[Reply ID: {ele.message_id} content: {ele.message_content}]"
                else:
                    message_str += f"[Reply ID: {ele.message_id}]"
            elif isinstance(ele, Forward):
                if ele.chains:
                    forward_contents = ""
                    for i, chain in enumerate(ele.chains):
                        ele.chains[i].message_list = [x for x in chain if not isinstance(x, Forward)]
                        forward_content = await self.message_format_to_text(ele.chains[i])
                        forward_contents += f"\n{forward_content}\n"
                    message_str += f"[Forward {forward_contents.strip()}]"
            elif isinstance(ele, Record):
                record_text = await self.llm_api.speech_to_text(record=ele)
                ele.transcript = record_text
                message_str += f"[Record {record_text}]"
            elif isinstance(ele, Notice):
                message_str += f"{ele.text}"
            elif isinstance(ele, File):
                try:
                    file_size = int(ele.size)
                except Exception as _:
                    file_size = None

                # TODO Make it customizable
                if not file_size or file_size > 10 * 1024 * 1024:
                    message_str += f"[File name: {ele.name} (File size over 10MB, not cached)]"
                    continue

                try:
                    path = Path(await ele.to_path())
                    data_dir = get_data_path()

                    try:
                        rel = path.relative_to(data_dir)
                        path_result = f"data/{rel}"
                    except ValueError:
                        path_result = str(path)

                    message_str += f"[File name: {ele.name}, file_path: {path_result}]"
                except Exception as e:
                    logger.error(f"Failed to save temp file: {e}")
            elif isinstance(ele, Video):
                try:
                    video_file_size = int(ele.size)
                except Exception as _:
                    video_file_size = None

                # TODO Make it customizable
                if not video_file_size or video_file_size > 10 * 1024 * 1024:
                    message_str += f"[Video name: {ele.name} (Video size over 10MB, not cached)]"
                    continue

                try:
                    path = Path(await ele.to_path())
                    data_dir = get_data_path()

                    try:
                        rel = path.relative_to(data_dir)
                        path_result = f"data/{rel}"
                    except ValueError:
                        path_result = str(path)

                    message_str += f"[Video name: {ele.name}, file_path: {path_result}]"
                except Exception as e:
                    logger.error(f"Failed to save temp video file: {e}")
            else:
                pass
        return message_str

    # ── Phase 0 M7-merge (multimodal) ──────────────────────────────
    @staticmethod
    def _detect_image_mime(b64: str) -> str:
        """Best-effort MIME sniff from a (potentially data-URL-prefixed)
        base64 string. Returns ``image/jpeg`` if detection fails — that
        matches what most providers will accept for unknown bytes.

        We avoid binary-decoding the full payload; just look at the
        magic-byte prefix in the first few base64 chars.
        """
        if not b64:
            return "image/jpeg"
        if b64.startswith("data:") and ";base64," in b64:
            head = b64.split(";base64,", 1)[0]
            # head looks like "data:image/png"
            if head.startswith("data:") and len(head) > 5:
                return head[5:] or "image/jpeg"
        prefix = b64[:8]
        if prefix.startswith("/9j/"):
            return "image/jpeg"
        if prefix.startswith("iVBOR"):
            return "image/png"
        if prefix.startswith("R0lGOD"):
            return "image/gif"
        if prefix.startswith("UklGR"):
            return "image/webp"
        return "image/jpeg"

    @staticmethod
    def _mime_to_audio_format(mime: Optional[str]) -> str:
        """Map an audio MIME to the OpenAI ``input_audio.format`` enum.
        Falls back to ``mp3`` because most adapters deliver mpeg-encoded
        records and OpenAI's vision API rejects unrecognised values."""
        if not mime:
            return "mp3"
        m = mime.lower()
        if "wav" in m:
            return "wav"
        if "mpeg" in m or "mp3" in m:
            return "mp3"
        if "ogg" in m:
            return "ogg"
        if "flac" in m:
            return "flac"
        if "aac" in m:
            return "aac"
        return "mp3"

    async def message_format_to_multimodal(self, message_chain: MessageChain):
        """Native-multimodal message formatter.

        Returns a tuple ``(text_str, content_parts)`` where
          * ``text_str`` is identical-shape to :meth:`message_format_to_text`
            (same placeholders for non-image elements + an ``[Image]``
            / ``[Sticker]`` / ``[Record:audio attached]`` /
            ``[Video: video attached]`` marker for each native-attached
            element). Suitable for memory persistence.
          * ``content_parts`` is a list of OpenAI vision content parts
            (``image_url`` / ``input_audio``) ready to be appended to
            :attr:`LLMRequest.multimodal_parts`.

        The routing decisions per element type honour the
        ``use_native_*_multimodal`` flags. If a flag is off for a
        modality, that branch falls through to the same logic as
        :meth:`message_format_to_text`. This is why both formatters
        coexist — the legacy path stays the source of truth for
        cached descriptions, the multimodal path is purely additive.

        See Phase0计划.md notes on multimodal merge from the previous
        framework version (旧文件中已经做出的更改).
        """
        text_str = ""
        content_parts: list[dict] = []

        for ele in message_chain:
            if isinstance(ele, Text):
                text_str += ele.text

            elif isinstance(ele, Emoji):
                if ele.emoji_desc:
                    text_str += f"[Emoji {ele.emoji_desc} (ID: {ele.emoji_id})]"
                else:
                    text_str += f"[Emoji {ele.emoji_id}]"

            elif isinstance(ele, At):
                if ele.nickname:
                    text_str += f"[At {ele.pid}(nickname: {ele.nickname})]"
                else:
                    text_str += f"[At {ele.pid}]"

            elif isinstance(ele, Image):
                if not self.use_native_multimodal:
                    # Fall back to text formatter for this single element so
                    # the image_desc_cache + VLM description pipeline still
                    # works. Cheaper than re-implementing here.
                    text_str += await self.message_format_to_text(MessageChain([ele]))
                    continue
                try:
                    image_b64 = await ele.to_base64()
                    mime = ele.mime or self._detect_image_mime(image_b64)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image_b64}"
                        }
                    })
                    text_str += "[Image]"
                except Exception as e:
                    logger.warning(
                        f"Failed to encode image for native multimodal, "
                        f"falling back to placeholder: {e}"
                    )
                    text_str += "[Image: failed to load]"

            elif isinstance(ele, Sticker):
                if not self.use_native_multimodal:
                    text_str += await self.message_format_to_text(MessageChain([ele]))
                    continue
                try:
                    sticker_b64 = await ele.to_base64()
                    mime = ele.mime or self._detect_image_mime(sticker_b64)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{sticker_b64}"
                        }
                    })
                    text_str += "[Sticker]"
                except Exception as e:
                    logger.warning(
                        f"Failed to encode sticker for native multimodal, "
                        f"falling back to placeholder: {e}"
                    )
                    text_str += "[Sticker: failed to load]"

            elif isinstance(ele, Record):
                if self.use_native_audio_multimodal:
                    try:
                        audio_b64 = await ele.to_base64()
                        audio_fmt = self._mime_to_audio_format(ele.mime)
                        content_parts.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": audio_fmt,
                            }
                        })
                        text_str += "[Record: audio attached]"
                    except Exception as e:
                        logger.warning(
                            f"Failed to encode audio for native multimodal, "
                            f"falling back to STT: {e}"
                        )
                        record_text = await self.llm_api.speech_to_text(record=ele)
                        ele.transcript = record_text
                        text_str += f"[Record {record_text}]"
                else:
                    record_text = await self.llm_api.speech_to_text(record=ele)
                    ele.transcript = record_text
                    text_str += f"[Record {record_text}]"

            elif isinstance(ele, Video):
                # Three modes for Video, in priority order:
                #   1. native video multimodal (send base64 to LLM)
                #   2. VLM video description (use_video_description)
                #   3. file-path-only placeholder (legacy default)
                try:
                    video_size = int(ele.size)
                except Exception:
                    video_size = None

                if self.use_native_video_multimodal:
                    if not video_size or video_size > 10 * 1024 * 1024:
                        text_str += (
                            f"[Video name: {ele.name} (Video size over 10MB, "
                            f"not sent as multimodal)]"
                        )
                    else:
                        try:
                            video_b64 = await ele.to_base64()
                            mime = ele.mime or "video/mp4"
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{video_b64}"
                                }
                            })
                            text_str += "[Video: video attached]"
                        except Exception as e:
                            logger.warning(
                                f"Failed to encode video natively, "
                                f"falling back to text path: {e}"
                            )
                            text_str += await self.message_format_to_text(MessageChain([ele]))
                elif self.use_video_description:
                    if not video_size or video_size > 10 * 1024 * 1024:
                        text_str += f"[Video name: {ele.name} (Video size over 10MB, not described)]"
                    else:
                        try:
                            from core.utils.common_utils import desc_video
                            vlm_model = self.provider_mgr.get_default_vlm()
                            video_desc = await desc_video(
                                client=vlm_model,
                                video=ele,
                                prompt=self.video_description_prompt,
                            )
                            text_str += f"[Video {video_desc}]"
                        except Exception as e:
                            logger.warning(
                                f"VLM video description failed, falling "
                                f"back to file path: {e}"
                            )
                            text_str += await self.message_format_to_text(MessageChain([ele]))
                else:
                    # Identical to text formatter's handling
                    text_str += await self.message_format_to_text(MessageChain([ele]))

            elif isinstance(ele, Reply):
                if ele.chain:
                    ele.chain.message_list = [x for x in ele.chain if not isinstance(x, Reply)]
                    reply_text, reply_parts = await self.message_format_to_multimodal(ele.chain)
                    content_parts.extend(reply_parts)
                    text_str += f"[Reply ID: {ele.message_id} content: {reply_text}]"
                elif ele.message_content:
                    text_str += f"[Reply ID: {ele.message_id} content: {ele.message_content}]"
                else:
                    text_str += f"[Reply ID: {ele.message_id}]"

            elif isinstance(ele, Forward):
                if ele.chains:
                    forward_contents = ""
                    for i, chain in enumerate(ele.chains):
                        ele.chains[i].message_list = [x for x in chain if not isinstance(x, Forward)]
                        fwd_text, fwd_parts = await self.message_format_to_multimodal(ele.chains[i])
                        content_parts.extend(fwd_parts)
                        forward_contents += f"\n{fwd_text}\n"
                    text_str += f"[Forward {forward_contents.strip()}]"

            elif isinstance(ele, Notice):
                text_str += f"{ele.text}"

            elif isinstance(ele, File):
                # Files don't go multimodal — defer to text formatter
                text_str += await self.message_format_to_text(MessageChain([ele]))

            else:
                pass

        return text_str, content_parts

    async def handle_im_message(self, event: KiraMessageEvent):
        """process im message"""

        # decorating event info

        sid = event.session.sid

        event.session.session_description = self.session_manager.get_session_info(sid).session_description

        # EventType.ON_IM_MESSAGE
        im_handlers = event_handler_reg.get_handlers(event_type=EventType.ON_IM_MESSAGE)
        for handler in im_handlers:
            await handler.exec_handler(event)
            if event.is_stopped:
                # Print event
                logger.info(event.get_log_info())
                return

        # Print event
        logger.info(event.get_log_info())

        # Check if message chain is valid, filter out unprocessed notice messages
        if event.message.chain.is_empty():
            return

        if event.process_strategy == "discard":
            return

        if event.process_strategy == "trigger":
            batch_msg = KiraMessageBatchEvent(
                message_types=event.message_types,
                timestamp=int(time.time()),
                adapter=event.adapter,
                session=event.session,
                messages=[event.message]
            )
            await self.handle_im_batch_message(batch_msg)
            return

        if event.process_strategy == "buffer":
            buffer = self.session_buffer.get_buffer(sid)
            async with buffer.lock:
                buffer.add(event)

            # EventType.ON_MESSAGE_BUFFERED
            im_handlers = event_handler_reg.get_handlers(event_type=EventType.ON_MESSAGE_BUFFERED)
            for handler in im_handlers:
                await handler.exec_handler(event.session.sid)
            return

        if event.process_strategy == "flush":
            flushed = await self.flush_session_messages(sid, extra_event=event)
            if not flushed:
                logger.warning(f"No pending messages to flush for session {sid}")
            return

    async def handle_im_batch_message(self, event: KiraMessageBatchEvent):
        # Start processing
        sid = event.session.sid

        # Phase 0 M7-merge: when ANY native multimodal flag is on we walk
        # through message_format_to_multimodal which returns both the
        # placeholder text (for memory persistence + log) AND a list of
        # OpenAI vision content parts. The parts are accumulated across
        # all batched messages and attached to the LLMRequest below via
        # request.multimodal_parts (LLMRequest.assemble_prompt then emits
        # the user message in vision content-list shape).
        any_native = (
            self.use_native_multimodal
            or self.use_native_audio_multimodal
            or self.use_native_video_multimodal
        )
        all_multimodal_parts: list[dict] = []

        for i, message in enumerate(event.messages):
            if any_native:
                text_part, parts = await self.message_format_to_multimodal(
                    message.chain
                )
                message.message_str = text_part
                all_multimodal_parts.extend(parts)
            else:
                message_str = await self.message_format_to_text(message.chain)
                message.message_str = message_str

        # EventType.ON_IM_BATCH_MESSAGE
        im_batch_handlers = event_handler_reg.get_handlers(event_type=EventType.ON_IM_BATCH_MESSAGE)
        for handler in im_batch_handlers:
            await handler.exec_handler(event)
            if event.is_stopped:
                logger.info("Event stopped")
                return

        # Set session title
        if not self.session_manager.get_session_info(sid).session_title:
            self.session_manager.update_session_info(sid, event.session.session_title)
        session_title = self.session_manager.get_session_info(sid).session_title

        # Build chat environment
        chat_env = {
            "platform": event.adapter.platform,
            "adapter": event.adapter.name,
            "chat_type": 'GroupMessage' if event.is_group_message() else 'DirectMessage',
            "self_id": event.self_id,
            "session_title": session_title,
            "session_description": event.session.session_description
        }

        # Get chat history memory
        session_memory = self.session_manager.fetch_memory(sid)

        # Generate agent prompt blocks (Phase 0.2 M3: list[PromptBlock])
        agent_blocks: list[PromptBlock] = await self.prompt_manager.get_agent_prompt(chat_env)

        # Inject skills as a PromptBlock (depth=95: between tools=90 and
        # output=100). build_skills_prompt() still returns a legacy Prompt
        # (skills_manager untouched in M3); we wrap it so the assembler
        # owns ordering. M4+/Phase 1 may migrate skills_manager itself.
        if len(self.skills_manager.skills_info) > 0:
            skills_legacy = self.skills_manager.build_skills_prompt()
            agent_blocks.append(PromptBlock(
                name=skills_legacy.name or "skills",
                content_provider=(lambda _ctx, _p=skills_legacy: _p.to_string()),
                depth=95,
                role="system",
                source="framework",
            ))

        # Get default LLM model client (Phase 0.1: route through ModelGroupManager)
        # Resolution order:
        #   1. If `models.default_llm_group` is set, use that group_id.
        #   2. Otherwise fall back to the virtual __default__ group (which wraps
        #      `models.default_llm` as a single-model group).
        #   3. If even that yields no client, fall back to the legacy direct path
        #      so existing deployments don't break.
        # M2: when the group is resolved, AgentExecutor calls group.call() on every
        # step so 429/5xx/timeout fallback applies through the full agent loop.
        # `llm_model` is still passed for legacy path + log lines (provider/model
        # name come from there); when group is set, it shadows direct chat() calls.
        llm_model = None
        active_group = None
        try:
            mg_mgr = getattr(self.llm_api, "model_group_mgr", None)
            group_id = self.kira_config.get_config("models.default_llm_group") or "__default__"
            if mg_mgr is not None:
                active_group = mg_mgr.get_group(group_id)
                if active_group is not None:
                    llm_model = active_group.get_primary_client()
            if llm_model is None:
                # Legacy fallback — keeps behaviour identical to pre-Phase-0 setups
                # whose system_config.json predates default_llm_group.
                llm_model = self.provider_mgr.get_default_llm()
                active_group = None
            if not llm_model:
                llm_logger.error(f"Default LLM model not set, please set it in Configuration")
                return
        except Exception as _:
            llm_logger.error(f"Default LLM model not set, please set it in Configuration")
            return

        # Phase 0.2 M3: system_prompt starts empty. The Assembler-produced
        # text is inserted at index 0 below; legacy ON_LLM_REQUEST handlers
        # may still .append() to system_prompt and their entries land
        # *after* the assembled block (additive策略, see Phase0_M3_handoff §2.5).
        request = LLMRequest(messages=session_memory[:], tools=deepcopy(self.llm_api.tools_definitions), tool_funcs=self.llm_api.tools_functions, tool_set=ToolSet())

        # Phase 0 M7-merge: attach native-multimodal content parts so
        # LLMRequest.assemble_prompt() emits the user message as a list
        # of content parts (text + image_url/input_audio). When the list
        # is empty (no native flag, or native flag on but no media in
        # this batch), assemble_prompt falls through to the legacy
        # string-content shape — full backwards compatibility.
        if any_native and all_multimodal_parts:
            request.multimodal_parts.extend(all_multimodal_parts)
            logger.info(
                f"Native multimodal: attaching {len(all_multimodal_parts)} "
                f"content part(s) to LLM request for sid={sid}"
            )

        # Add received im messages
        for i, message in enumerate(event.messages):
            request.user_prompt.append(Prompt(message.message_str, name="message", source="system"))

        # Build tag set
        tag_set = TagSet()

        # EventType.ON_LLM_REQUEST (legacy: runs BEFORE the assembler so
        # plugins that mutate request.user_prompt[i].content by name still
        # work; plugins that .append() to request.system_prompt have their
        # additions wait until the assembled block has taken slot [0]).
        llm_handlers = event_handler_reg.get_handlers(event_type=EventType.ON_LLM_REQUEST)
        for handler in llm_handlers:
            await handler.exec_handler(event, request, tag_set)
            if event.is_stopped:
                logger.info("Event stopped while llm request stage")
                return

        # Register persistent tags registered by user plugins
        tag_set.register(*tag_registry.get_all())

        # ── Phase 0.2 M3: PromptBlock pipeline ────────────────────────
        # Collect three sources of blocks: framework built-ins (agent_blocks
        # +skills), plugin-static (PromptBlockRegistry), plugin-runtime
        # (BlockCollector via ON_PROMPT_ASSEMBLE handlers).
        all_blocks: list[PromptBlock] = list(agent_blocks)
        block_registry = getattr(self.prompt_manager, "block_registry", None)
        if block_registry is not None:
            all_blocks.extend(block_registry.get_all())

        # Phase 0.3 M4: pre-compute the message_types string from tag_set
        # and stash it in ctx_snapshot. format.j2 picks it up via
        # `{{ message_types }}`. This deletes the M3 string-replace hack
        # in the post-assemble step.
        # CONSTRAINT: any plugin that registers tags inside an
        # ON_PROMPT_ASSEMBLE handler will not see those tags appear in
        # message_types — by design. tag registration belongs in
        # ON_LLM_REQUEST (the existing convention); ON_PROMPT_ASSEMBLE
        # is for prompt blocks, not tags.
        ctx_snapshot = {
            "session_id": sid,
            "event": event,
            "chat_env": chat_env,
            "now_ts": time.time(),
            "message_types": tag_set.to_prompt(),
        }

        block_collector = BlockCollector()
        assemble_handlers = event_handler_reg.get_handlers(
            event_type=EventType.ON_PROMPT_ASSEMBLE
        )
        for handler in assemble_handlers:
            await handler.exec_handler(event, ctx_snapshot, block_collector)
            if event.is_stopped:
                logger.info("Event stopped while prompt assemble stage")
                return
        all_blocks.extend(block_collector.get_all())

        # Phase 0.4 M5: drain pending hints_pipeline injections for this
        # session. These are blocks prepared in the BACKGROUND last turn
        # by handlers responding to <hints_key> tags the LLM emitted
        # then. Drained AFTER plugin-runtime collection so the assembler
        # filter/sort treats them uniformly with the rest. Bounded wait
        # (default 200ms inside the pipeline) — anything still preparing
        # is dropped this turn (better one stale turn than blocking the
        # user-visible response).
        if self.hints_pipeline is not None:
            try:
                pending_hint_blocks = await self.hints_pipeline.collect_blocks(sid)
                if pending_hint_blocks:
                    all_blocks.extend(pending_hint_blocks)
                    logger.debug(
                        f"hints_pipeline.collect_blocks: injected "
                        f"{len(pending_hint_blocks)} block(s) for sid={sid}"
                    )
            except Exception as e:
                logger.warning(
                    f"hints_pipeline.collect_blocks raised for sid={sid}: {e}"
                )

        # Run the Assembler → Translator pipeline. M4 path: Translator
        # owns the final concatenation (and optional debug XML markers).
        # Falls back to M3 behaviour if either component is missing on
        # PromptManager.
        assembler = getattr(self.prompt_manager, "assembler", None)
        translator = getattr(self.prompt_manager, "translator", None)
        assembled = None
        if assembler is not None:
            assembled = await assembler.assemble(all_blocks, ctx_snapshot)
            if translator is not None:
                # M4 main path. Each AssembledBlock.content already holds
                # the j2-rendered output (because PromptManager wrapped each
                # framework block's content_provider as a render_template
                # call). Translator.render concatenates them and optionally
                # wraps with `<!-- block:... -->` markers.
                rendered = translator.render(assembled, tag_set, ctx_snapshot)
                assembled_text = rendered.system_text
            else:
                # M3 path: no Translator, do the empty-separator join
                # ourselves and run the legacy <|message_types|> string
                # replace as a safety net (in case any block content still
                # holds the old token).
                logger.warning(
                    "PromptTranslator not available on PromptManager; "
                    "falling back to M3 string-join. Did lifecycle wire it in?"
                )
                assembled_text = assembled.system_text(separator="")
                assembled_text = assembled_text.replace(
                    "<|message_types|>", tag_set.to_prompt()
                )
            # end="" because assembled_text already ends with \n from the
            # last block; we don't want a double trailing newline.
            request.system_prompt.insert(
                0,
                Prompt(assembled_text, name="__assembled__", source="system", end=""),
            )
        else:
            # Degraded path (assembler not injected, e.g. some test harness):
            # render every block as a legacy Prompt and prepend them, mimicking
            # the pre-M3 behaviour.
            logger.warning(
                "PromptAssembler not available on PromptManager; falling back "
                "to per-block legacy rendering. Did lifecycle wire it in?"
            )
            fallback_prompts: list[Prompt] = []
            for blk in all_blocks:
                if not blk.enabled or blk.role != "system":
                    continue
                try:
                    res = blk.content_provider(ctx_snapshot)
                    if hasattr(res, "__await__"):
                        res = await res
                    text = "" if res is None else str(res)
                    fallback_prompts.append(blk.to_legacy_prompt(text))
                except Exception as e:
                    logger.warning(
                        f"Fallback render failed for block {blk.name}: {e}"
                    )
            request.system_prompt = fallback_prompts + list(request.system_prompt)
            # Legacy <|message_types|> substitution path
            for sp in request.system_prompt:
                if sp.name == "format":
                    sp.content = sp.content.replace(
                        "<|message_types|>", tag_set.to_prompt()
                    )
                    break

        request.assemble_prompt()

        # ── In-chat depth injection (post-M3 amendment) ──────────────
        # Apply ChatInjection blocks (PromptBlock with position="in_chat")
        # produced by the assembler. Each injection is inserted into
        # `request.messages` at offset `inject_depth` counted backwards
        # from the END (after assemble_prompt() has appended the latest
        # user message). depth=0 → very end (right before generation);
        # depth=1 → before the latest user message; depth=N → before
        # the Nth-last message.
        #
        # We compute all target positions against the *original* length
        # of `request.messages`, then insert in descending position so
        # later inserts don't shift earlier-resolved positions. Within
        # the same position, negated registration index keeps insertion
        # order = registration order.
        injections = []
        if assembled is not None:
            injections = list(getattr(assembled, "chat_injections", []) or [])
        if injections:
            original_len = len(request.messages)
            positioned = [
                (max(0, original_len - inj.inject_depth), idx, inj)
                for idx, inj in enumerate(injections)
            ]
            positioned.sort(key=lambda t: (-t[0], -t[1]))
            for pos, _idx, inj in positioned:
                request.messages.insert(
                    pos,
                    {"role": inj.inject_role, "content": inj.content},
                )

        # TODO: migrate tools & tool_func params to tool_set
        request.tools.extend(request.tool_set.to_list())

        # Print user message info
        user_message = "".join(p.to_string() for p in request.user_prompt if isinstance(p, Prompt))
        logger.info(f"processing message(s) from {sid}:\n{user_message}")

        # 把收到的消息放到新收到的消息内容中
        new_memory = NewMemory()
        new_memory.user(user_message)

        # Get max tool loop config, defaults to 2 if not a valid integer
        max_tool_loop = self.kira_config.get_config("bot_config.agent.max_tool_loop")
        try:
            max_tool_loop = int(max_tool_loop)
        except ValueError:
            max_tool_loop = 2

        max_agent_steps = max_tool_loop + 1

        agent_executor = AgentExecutor(self.llm_api, request.tool_set)
        agent_ctx = AgentExecutionContext(
            event=event,
            request=request,
            llm_model=llm_model,
            new_memory=new_memory,
            model_group=active_group,
        )

        async def send_llm_text(resp: LLMResponse):
            text = resp.text_response
            session_lock = self.get_session_lock(sid)
            async with session_lock:
                # Phase 0.4 M5: thread llm_response through send_xml_messages
                # so the new AFTER_LLM_RESPONSE_PARSE event handlers can
                # receive the full LLMResponse (text_response is the most
                # important piece, but tool_calls etc. may be useful too).
                message_results = await self.send_xml_messages(
                    event, text.strip(), tag_set, llm_response=resp
                )
                if message_results is None:
                    return
                response_with_ids = self._add_message_ids(text, message_results)
                step_result = KiraStepResult(message_results=message_results, raw_output=response_with_ids)
                # EventType.ON_STEP_RESULT
                step_handlers = event_handler_reg.get_handlers(event_type=EventType.ON_STEP_RESULT)
                for step_handler in step_handlers:
                    await step_handler.exec_handler(event, step_result)
                    if event.is_stopped:
                        logger.info("Event stopped while ON_STEP_RESULT stage")
                        return
                logger.info(f"LLM -> {sid}: {step_result.raw_output}")
                llm_resp.text_response = step_result.raw_output

                for idx in range(-1, -len(new_memory.memory_list), -1):
                    if new_memory.memory_list[idx]["role"] == "assistant":
                        new_memory.memory_list[idx]["content"] = step_result.raw_output
                        request.messages[idx]["content"] = step_result.raw_output
                        break

                # Phase 0.4 M5: ingest <hints_key> tags from this turn's
                # output AFTER ON_STEP_RESULT (so step_result.raw_output's
                # post-fixup form is what gets parsed). The pipeline
                # internally schedules handler.prepare() as background
                # asyncio.Tasks — this call returns quickly. Wrapped in
                # try/except because hints handling is best-effort and a
                # raise here must not break message delivery (which has
                # already happened by this point anyway).
                if self.hints_pipeline is not None:
                    try:
                        await self.hints_pipeline.consume_response(
                            sid, resp, ctx_snapshot
                        )
                    except Exception as e:
                        logger.warning(
                            f"hints_pipeline.consume_response raised for "
                            f"sid={sid}: {e}"
                        )

        # Iter agent executor to get LLMResponse
        # TODO use llm_semaphore to restrict concurrent LLM requests
        async for step in agent_executor.run(agent_ctx, max_steps=max_agent_steps):
            llm_resp = step.llm_response
            if not llm_resp:
                break

            if llm_resp.text_response:
                await send_llm_text(llm_resp)

            if not step.has_tool_calls or step.is_final:
                break

            # Process tool calls if existed

        # Save new memory
        self.session_manager.update_memory(sid, new_memory.memory_list)

    async def handle_cmt_message(self, msg: KiraCommentEvent):
        """process comment message"""

        if msg.sub_cmt_id:
            logger.info(f"[{msg.adapter_name} | {msg.sub_cmt_id}] [{msg.commenter_nickname}]: {msg.sub_cmt_content[0].text}")
            cmt_content = f"""You: {msg.cmt_content[0].text}
            {msg.commenter_nickname}: {msg.sub_cmt_content[0].text}
            """
        else:
            logger.info(f"[{msg.adapter_name} | {msg.cmt_id}] [{msg.commenter_nickname}]: {msg.cmt_content[0].text}")
            cmt_content = f"""{msg.commenter_nickname}: {msg.cmt_content[0].text}"""

        cmt_prompt = await self.prompt_manager.get_comment_prompt(cmt_content)

        client = self.provider_mgr.get_default_llm()
        if not client:
            llm_logger.error(f"Default LLM model not set, please set it in Configuration")
            return

        llm_req = LLMRequest(messages=[{"role": "user", "content": cmt_prompt}])

        llm_resp = await client.chat(llm_req)

        response = llm_resp.text_response.strip()

        logger.info(f"LLM: {response}")

        if response:
            await self.adapter_mgr.get_adapter(msg.adapter_name).send_comment(
                text=response,
                root=msg.cmt_id,
                sub=msg.sub_cmt_id
            )
        else:
            logger.warning("Blank LLM response")

    async def send_xml_messages(
        self,
        event: KiraMessageBatchEvent,
        xml_data: str,
        tag_set: TagSet,
        llm_response: Optional[LLMResponse] = None,
    ) -> Optional[List[KiraIMSentResult]]:
        """
        send message via session id & xml data

        :param event: KiraMessageBatchEvent
        :param xml_data: xml string (post-XML-repair raw LLM output)
        :param tag_set: TagSet object
        :param llm_response: Phase 0.4 M5 — when caller has the full
            LLMResponse on hand it should pass it so the
            AFTER_LLM_RESPONSE_PARSE event can be fired with both the
            response object AND the raw_xml/message_chains. Optional
            for backward compat with any caller that only has a string
            (the event simply isn't fired in that case).
        :return: list[KiraIMSentResult]
        """
        parts = event.sid.split(":")
        if len(parts) != 3:
            raise ValueError("invalid target, must follow the form of <adapter>:<dm|gm>:<id>")

        message_results = []
        try:
            message_chains = await self._parse_xml_msg(xml_data, tag_set)

            # EventType.AFTER_XML_PARSE — existing event, fires first.
            llm_handlers = event_handler_reg.get_handlers(event_type=EventType.AFTER_XML_PARSE)
            for handler in llm_handlers:
                await handler.exec_handler(event, message_chains)
                if event.is_stopped:
                    logger.info("Event stopped while AFTER_XML_PARSE stage")
                    return None

            # Phase 0.4 M5: EventType.AFTER_LLM_RESPONSE_PARSE.
            # New event for plugins that want the raw post-repair XML +
            # the parsed message_chains together (e.g. analytics, debug
            # dumps, hints_key sniffers that AREN'T using HintsPipeline
            # directly). HintsPipeline itself is invoked by send_llm_text
            # AFTER this loop, so plugin handlers here see the same
            # LLM output the pipeline will see, without racing it.
            if llm_response is not None:
                parse_handlers = event_handler_reg.get_handlers(
                    event_type=EventType.AFTER_LLM_RESPONSE_PARSE
                )
                for handler in parse_handlers:
                    await handler.exec_handler(
                        event, llm_response, xml_data, message_chains
                    )
                    if event.is_stopped:
                        logger.info(
                            "Event stopped while AFTER_LLM_RESPONSE_PARSE stage"
                        )
                        return None
        except Exception as e:
            logger.error(f"Error parsing message: {str(e)}")
            return []

        # ── Phase 0.5 M7: ON_OUTPUT_PIPELINE hook chain ─────────────
        # Every chain handler sees the SAME OutputCtx instance and may
        # rewrite chains, set per-chain delays, intercept the whole
        # batch, or stash notes in meta. Behaviour-preserving when no
        # plugin registers anything: DefaultDelayHook (SYS_LOW, set up
        # in lifecycle.register_default_hooks) fills delays with the
        # same random.uniform(min,max) the old hardcoded path used.
        #
        # budget_ms / api_elapsed_ms are filled now (decision #4,
        # Phase0计划.md §8.4) so Phase 5 can rely on them without
        # revisiting this call site.
        now_ms = int(time.time() * 1000)
        try:
            user_msg_ts_ms = int(getattr(event, "timestamp", 0)) * 1000
        except Exception:
            user_msg_ts_ms = 0
        budget_ms = max(0, now_ms - user_msg_ts_ms) if user_msg_ts_ms else 0

        api_elapsed_ms = 0
        if llm_response is not None and llm_response.time_consumed is not None:
            try:
                api_elapsed_ms = int(float(llm_response.time_consumed) * 1000)
            except Exception:
                api_elapsed_ms = 0

        out_ctx = OutputCtx(
            event=event,
            raw_text=xml_data,
            chains=list(message_chains),
            delays=[],
            intercepted=False,
            unsent_reason=None,
            budget_ms=budget_ms,
            api_elapsed_ms=api_elapsed_ms,
            meta={},
        )

        try:
            output_handlers = event_handler_reg.get_handlers(
                event_type=EventType.ON_OUTPUT_PIPELINE
            )
            for handler in output_handlers:
                await handler.exec_handler(event, out_ctx)
                if event.is_stopped:
                    logger.info(
                        "Event stopped while ON_OUTPUT_PIPELINE stage"
                    )
                    return None
                if out_ctx.intercepted:
                    # Don't break — let lower-priority hooks (e.g.
                    # debug/audit) still observe. They MUST honour the
                    # flag and not undo it (best-effort, not enforced).
                    pass
        except Exception as e:
            logger.error(
                f"ON_OUTPUT_PIPELINE chain raised; falling back to "
                f"default send (chain may have partial mutations): {e}"
            )

        # Intercepted batch: log + record empty results so callers /
        # ON_STEP_RESULT see "0 sent". Phase 5 will persist to unsent
        # ledger; M7 only logs the reason.
        if out_ctx.intercepted:
            logger.info(
                f"OutputCtx.intercepted for sid={event.sid} "
                f"(reason={out_ctx.unsent_reason!r}); dropping "
                f"{len(out_ctx.chains)} chain(s)"
            )
            return []

        # Length-align delays to chains AFTER all hooks ran. Anything
        # still None here means no hook (including SYS_LOW default)
        # touched it — should be rare but possible if registry was
        # cleared mid-flight; treat None as immediate-send.
        chains_after = list(out_ctx.chains)
        delays_after = list(out_ctx.delays)
        if len(delays_after) < len(chains_after):
            delays_after.extend([None] * (len(chains_after) - len(delays_after)))
        elif len(delays_after) > len(chains_after):
            delays_after = delays_after[: len(chains_after)]

        for idx, message_chain in enumerate(chains_after):
            if not message_chain.is_empty():
                result = await self.send_message_chain(event.sid, message_chain)
                if not result.ok and result.err:
                    logger.error(result.err)
                message_results.append(result)

                # Per-chain delay sourced from OutputCtx.delays. If still
                # None (nobody set it), no sleep. If set, clamp to the
                # M7 soft cap so a misbehaving plugin can't deadlock
                # session_lock.
                d = delays_after[idx]
                if d is not None:
                    try:
                        sleep_s = float(d)
                    except Exception:
                        sleep_s = 0.0
                    if sleep_s > 0:
                        from core.output.output_ctx import MAX_BLOCKING_DELAY_S
                        if sleep_s > MAX_BLOCKING_DELAY_S:
                            logger.warning(
                                f"Output delay {sleep_s:.2f}s exceeds soft "
                                f"cap {MAX_BLOCKING_DELAY_S}s (sid={event.sid}, "
                                f"chain[{idx}]); clamping. Long unsent / "
                                f"silence is a Phase 5 feature."
                            )
                            sleep_s = MAX_BLOCKING_DELAY_S
                        await asyncio.sleep(sleep_s)
            else:
                message_results.append(KiraIMSentResult(ok=False, err="Blank message list detected"))
        return message_results

    async def send_message_chain(self, session: str, chain: MessageChain) -> KiraIMSentResult:
        """
        Send a MessageChain to target.

        :param session: adapter_name:dm|gm:session_id
        :param chain: MessageChain instance
        :return: message_id (empty string if failed)
        """
        parts = session.split(":")
        if len(parts) != 3:
            raise ValueError("invalid target, must follow <adapter>:<dm|gm>:<id>")

        adapter_name, chat_type, pid = parts
        adapter = self.adapter_mgr.get_adapter(adapter_name)

        if chat_type == "dm":
            result = await adapter.send_direct_message(pid, chain)
        elif chat_type == "gm":
            result = await adapter.send_group_message(pid, chain)
        else:
            raise ValueError("chat_type must be 'dm' or 'gm'")

        if not result:
            return KiraIMSentResult(ok=False)

        return result

    @staticmethod
    async def _parse_xml_msg(xml_data, tag_set: TagSet) -> list[MessageChain]:
        """Parse xml to list[MessageChain]"""
        root = ET.fromstring(f"<root>{xml_data}</root>")
        message_chains = []

        for msg in root.findall("msg"):
            message_elements = []
            for child in msg:
                tag = child.tag
                value = child.text.strip() if child.text else ""
                attrs = child.attrib

                if tag in tag_set:
                    tag_inst = tag_set.get(name=tag)
                    tag_res = await tag_inst.handle(value, **attrs)

                    if isinstance(tag_res, BaseMessageElement):
                        message_elements.append(tag_res)
                    elif isinstance(tag_res, list):
                        message_elements.extend(tag_res)

            if message_elements:
                message_chains.append(MessageChain(message_elements))

        return message_chains

    @staticmethod
    def _add_message_ids(xml_data: str, message_results: List[KiraIMSentResult]) -> str:
        """为XML响应添加消息ID"""
        try:
            root = ET.fromstring(f"<root>{xml_data}</root>")

            for i, msg in enumerate(root.findall("msg")):
                if i < len(message_results):
                    message_id = message_results[i].message_id
                    if not message_id:
                        message_id = ""
                    msg.set("message_id", message_id)

            return ET.tostring(root, encoding='unicode', method='xml')[6:-7]

        except Exception as e:
            logger.error(f"Error adding message IDs: {str(e)}")
            return xml_data

    async def cleanup_image_desc_cache_task(self):
        """Background task: clean up expired image desc cache every 24 hours."""
        while True:
            try:
                deleted = await self.db.cleanup_expired_image_desc_cache()
                if deleted:
                    logger.info(f"Cleaned up {deleted} expired image desc cache entries")
                await asyncio.sleep(24 * 60 * 60)
            except asyncio.CancelledError:
                logger.info("Image desc cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in image desc cache cleanup: {e}")
