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
    # Phase 0.5 M7: 输出后处理钩子链。在 AFTER_XML_PARSE / AFTER_LLM_RESPONSE_PARSE
    # 之后、真正调用 send_message_chain 之前触发。处理器签名:
    #   async def f(event, ctx: OutputCtx) -> None
    # OutputCtx 是唯一可变状态容器 (chains/delays/intercepted/meta), 钩子按
    # priority 降序执行, 期间允许:
    #   - 修改 ctx.chains    (拆分/合并/删除/编辑消息)
    #   - 设置 ctx.delays    (每条消息发送前等待秒数, None=由 DefaultDelayHook 填默认)
    #   - 标 ctx.intercepted (整批消息不发送, 但 ON_STEP_RESULT 仍触发)
    # 框架内置 DefaultDelayHook 在 SYS_LOW 优先级跑, 只填还是 None 的 delay 槽位,
    # 不覆盖更高优先级钩子的决定。Phase 5 的延迟预算/耐心节奏/unsent 全部挂这里。
    # 关键约束: 钩子链整体在 session_lock 内, delay 不应超过 ~5s (会阻塞同 sid 后续消息)。
    ON_OUTPUT_PIPELINE = "on_output_pipeline"
    ON_TOOL_RESULT = "on_tool_result"  # 工具调用结果
    ON_STEP_RESULT = "on_step_result"  # Agent 步骤结果
    ON_FINAL_RESULT = "on_final_result"  # 最终消息结果
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
