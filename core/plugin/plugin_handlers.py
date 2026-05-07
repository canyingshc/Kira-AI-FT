from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional, Union, Callable, Any, Dict, List

from core.logging_manager import get_logger

logger = get_logger("hook", "orange")


class Priority(IntEnum):
    """Priority of event handlers, DO NOT use SYS_LOW or SYS_HIGH in user plugins"""

    SYS_LOW = -100
    LOW = -50
    MEDIUM = 0
    HIGH = 50
    SYS_HIGH = 100


class EventType(Enum):
    ON_IM_MESSAGE = "on_im_message"  # 消息到达时
    ON_MESSAGE_BUFFERED = "on_message_buffered"  # 消息进入缓冲区后
    ON_IM_BATCH_MESSAGE = "on_im_batch_message"  # 消息合并后
    ON_LLM_REQUEST = "on_llm_request"  # LLM请求前 (legacy: 直接改 LLMRequest 字段)
    # Phase 0.2 M3: PromptBlock 收集阶段。在 ON_LLM_REQUEST 之后触发，
    # 处理器收 (event, ctx_snapshot, block_collector) 三个参数,
    # 通过 block_collector.add(PromptBlock) 贡献本轮的运行时 PromptBlock。
    ON_PROMPT_ASSEMBLE = "on_prompt_assemble"
    ON_LLM_RESPONSE = "on_llm_response"  # LLM 原始输出 (kira-ai 在此修复 XML)
    # Phase 0.4 M5: LLM 响应解析后阶段。此事件刻意排在 ON_LLM_RESPONSE
    # (XML 修复) 之后、send_message_chain 之前，给 HintsPipeline 等
    # 后处理组件一个干净挂点。处理器签名:
    #   async def f(event, llm_response, raw_xml, message_chains)
    # raw_xml 是字符串 (修复后的 LLM 原始输出),
    # message_chains 是 list[MessageChain] (XML 解析结果, 可被读取
    # 但本事件中不应再修改 — 修改用 AFTER_XML_PARSE)。
    AFTER_LLM_RESPONSE_PARSE = "after_llm_response_parse"
    AFTER_XML_PARSE = "after_xml_parse"  # XML 解析后 (MessageChain)
    ON_TOOL_RESULT = "on_tool_result"  # 工具调用结果
    ON_STEP_RESULT = "on_step_result"  # Agent 步骤结果
    ON_FINAL_RESULT = "on_final_result"  # 最终消息结果
    # post-M5 amendment: hints pipeline observer event. Fired by
    # HintsPipeline whenever a hint emission completes processing
    # (success / None result / handler exception / FrequencyPolicy skip).
    # Handler signature:
    #   async def f(event, hints_result: HintsResult)
    # 'event' is the KiraMessageBatchEvent whose ON_STEP_RESULT triggered
    # the hint ingest. Observer fires fire-and-forget relative to the
    # main message path: a slow observer cannot block the user-visible
    # turn. Use this to log / chain / debug hints without registering
    # a competing HintsKeyHandler for the same key_type.
    ON_HINTS_PRODUCED = "on_hints_produced"
    ON_EXCEPTION = "on_exception"  # 异常发生时
    ...


@dataclass
class EventHandler:
    event_type: EventType

    priority: Union[Priority, int]

    handler: Callable

    desc: Optional[str] = None

    def __lt__(self, other):
        return self.priority < other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    async def exec_handler(self, event, *args, **kwargs):
        try:
            await self.handler(event, *args, **kwargs)
        except Exception as e:
            import traceback as tb
            logger.error(tb.format_exc())
            if self.event_type != EventType.ON_EXCEPTION:
                from core.chat.message_utils import KiraExceptionEvent
                from core.plugin.plugin_registry import get_obj_plugin_id
                exc_event = KiraExceptionEvent(
                    name=type(e).__name__,
                    message=str(e),
                    traceback=tb.format_exc(),
                    stage=self.event_type.value,
                    source="plugin",
                    comp_id=get_obj_plugin_id(self.handler),
                    e=e
                )
                for h in event_handler_reg.get_handlers(EventType.ON_EXCEPTION):
                    try:
                        await h.handler(event, exc_event)
                    except Exception:
                        logger.error(tb.format_exc())


class EventHandlerRegistry:
    def __init__(self):
        self._handlers: Dict[Any, List[EventHandler]] = {}

    def register(self, eh: EventHandler):
        self._handlers.setdefault(eh.event_type, [])
        self._handlers[eh.event_type].append(eh)
        self._handlers[eh.event_type].sort(reverse=True)

    def get_handlers(self, event_type: EventType) -> List[EventHandler]:
        return self._handlers.setdefault(event_type, [])

    def del_handler(self, handler: EventHandler):
        for k, hl in self._handlers.items():
            if handler in hl:
                hl.remove(handler)
                break


event_handler_reg = EventHandlerRegistry()
