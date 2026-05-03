import asyncio
import time
from typing import Optional

from .logging_manager import get_logger
from .config import KiraConfig
from .sticker_manager import StickerManager
from .message_manager import MessageProcessor
from .prompt_manager import PromptManager
from .prompt_block import PromptBlockRegistry
from .prompt_assembler import PromptAssembler
from .prompt_translator import PromptTranslator
from .pipeline.hints_pipeline import HintsPipeline
from .output import register_default_hooks
from core.chat.session_manager import SessionManager
from .adapter import AdapterManager
from .statistics import Statistics
from .llm_client import LLMClient
from .tool_manager import register_all_tools
from .event_bus import EventBus
from .persona import PersonaManager
from .provider import ProviderManager, ModelGroupManager
from .plugin import PluginContext, PluginManager
from core.agent.mcp_mgr import MCPManager
from core.agent.skills_mgr import SkillsManager
from core.config import VERSION
from core.utils.path_utils import get_data_path
from core.temp_monitor import AsyncTempMonitor
from core.telemetry import TelemetryClient
from core.db.db_mgr import DatabaseManager
from core.db.service import DatabaseService
from core.db.migrate_to_db import run_migrations


logger = get_logger("lifecycle", "blue")


class KiraLifecycle:
    """life cycle of KiraAI, managing all tasks and modules"""

    def __init__(self, stats: Statistics):
        self.stats = stats

        self.kira_config: Optional[KiraConfig] = None

        self.db_manager: Optional[DatabaseManager] = None

        self.db_service: Optional[DatabaseService] = None

        self.provider_manager: Optional[ProviderManager] = None

        self.model_group_manager: Optional[ModelGroupManager] = None

        self.llm_api: Optional[LLMClient] = None

        self.adapter_manager: Optional[AdapterManager] = None

        self.session_manager: Optional[SessionManager] = None

        self.persona_manager: Optional[PersonaManager] = None

        self.prompt_manager: Optional[PromptManager] = None

        # Phase 0.2 M3: PromptBlock pipeline.
        self.prompt_block_registry: Optional[PromptBlockRegistry] = None

        self.prompt_assembler: Optional[PromptAssembler] = None

        # Phase 0.3 M4: PromptTranslator (Jinja2 template rendering).
        self.prompt_translator: Optional[PromptTranslator] = None

        # Phase 0.4 M5: HintsPipeline (async hints_key resource pipeline).
        self.hints_pipeline: Optional[HintsPipeline] = None

        # Phase 0.5 M7: handle of the framework-registered DefaultDelayHook
        # so /reload or shutdown can find it. The actual hook lives in
        # event_handler_reg under EventType.ON_OUTPUT_PIPELINE.
        self._default_delay_eh = None

        self.message_processor: Optional[MessageProcessor] = None

        self.sticker_manager: Optional[StickerManager] = None

        self.event_bus: Optional[EventBus] = None

        self.plugin_context: Optional[PluginContext] = None

        self.plugin_manager: Optional[PluginManager] = None

        self.temp_monitor: Optional[AsyncTempMonitor] = None

        self.mcp_manager: Optional[MCPManager] = None

        self.skills_manager: Optional[SkillsManager] = None

        self.telemetry_client: Optional[TelemetryClient] = None

        self.tasks: list[asyncio.Task] = []

    async def schedule_tasks(self):
        self.tasks = [
            # asyncio.create_task(self.sticker_manager.scan_and_register_sticker(), name="sticker_scan")
        ]
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = self.tasks[i]
                logger.error(f"Scheduled task '{task.get_name()}' failed: {result}")

    async def init_and_run_system(self):
        """主函数：负责启动和初始化各个模块"""
        logger.info(f"✨ Starting KiraAI {VERSION}...")

        # ====== event bus ======
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # ====== init KiraAI config ======
        self.kira_config = KiraConfig()

        # ====== init database manager ======
        db_url = self.kira_config.get_config("database.url")
        if not db_url:
            db_path = get_data_path() / "data.db"
            db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
        db_echo = self.kira_config.get_config("database.echo", False)
        self.db_manager = DatabaseManager(db_url, echo=db_echo)
        await self.db_manager.init()
        logger.info(f"DatabaseManager initialized with URL: {db_url}")

        self.db_service = DatabaseService(self.db_manager)
        await self.db_service.init_tables()
        logger.info("Database tables initialized")

        await run_migrations(self.db_service)

        # ====== record startup time and init telemetry ======
        self.stats.set_stats("started_ts", int(time.time()))
        # self.telemetry_client = TelemetryClient(self.db_service, self.kira_config, self.stats)
        # await self.telemetry_client.initialize()

        # ====== init ProviderManager config ======
        self.provider_manager = ProviderManager(self.db_service, self.kira_config)

        # ====== init ModelGroupManager (Phase 0.1) ======
        # Wraps ProviderManager so callers can address LLMs by group_id rather
        # than by provider:model. Loads user groups from data/config/model_groups.json
        # and exposes virtual __default__ / __default_fast__ groups built from
        # models.default_llm / default_fast_llm (legacy compatibility).
        self.model_group_manager = ModelGroupManager(self.kira_config, self.provider_manager)

        # ====== init LLMClient ======
        self.llm_api = LLMClient(self.kira_config, self.provider_manager)
        self.llm_api.model_group_mgr = self.model_group_manager
        await register_all_tools(self.llm_api)  # Legacy tools

        # ====== init adapter manager ======
        self.adapter_manager = AdapterManager(self.kira_config, loop, event_queue, self.llm_api)
        await self.adapter_manager.initialize()

        # ====== init session manager ======
        self.session_manager = SessionManager(self.db_service, self.kira_config)

        # ====== init persona manager ======
        self.persona_manager = PersonaManager(db=self.db_service)
        await self.persona_manager.init_persona()

        # ====== init sticker manager ======
        self.sticker_manager = StickerManager(db=self.db_service)
        await self.sticker_manager.init()

        # ====== init prompt manager ======
        self.prompt_manager = PromptManager(self.kira_config,
                                            self.persona_manager)

        # ====== init PromptBlock pipeline (Phase 0.2 M3) ======
        # Holds plugin-registered static PromptBlocks; cleared per-plugin
        # on terminate/disable via PluginManager._cleanup_plugin_registration.
        self.prompt_block_registry = PromptBlockRegistry()
        # Stateless coordinator (filter → sort → evaluate). Single instance
        # is safe to share across the process; per-call cache is local.
        self.prompt_assembler = PromptAssembler()
        # Inject into PromptManager so message_manager can reach them via
        # `self.prompt_manager.block_registry` / `.assembler` without needing
        # extra constructor parameters.
        self.prompt_manager.block_registry = self.prompt_block_registry
        self.prompt_manager.assembler = self.prompt_assembler

        # ====== init PromptTranslator (Phase 0.3 M4) ======
        # Jinja2-backed template renderer. Loads templates from
        # core/prompts/translator/*.j2 by default, configurable via
        # data/config/translator.json (template_dir / debug_markers).
        # When this fails to construct (e.g. Jinja2 not installed) we let
        # the exception propagate — running M4 without Jinja2 is a deploy
        # error worth surfacing immediately rather than degrading silently.
        self.prompt_translator = PromptTranslator(self.kira_config)
        self.prompt_manager.translator = self.prompt_translator

        # ====== init HintsPipeline (Phase 0.4 M5) ======
        # Process-singleton async pipeline for "this turn LLM emits a key
        # → next turn assembler injects the resource" pattern. Plugins
        # register handlers via @register.hints_handler; message_manager
        # drains pending blocks via pipeline.collect_blocks(sid) before
        # assembly and feeds new emissions via pipeline.consume_response
        # after XML parse. No background thread / no persistence — entries
        # live as asyncio.Tasks until consumed or TTL-aged out.
        self.hints_pipeline = HintsPipeline()

        # ====== register Phase 0.5 M7 framework hooks ======
        # DefaultDelayHook lives at SYS_LOW priority on ON_OUTPUT_PIPELINE
        # so plugin handlers (Phase 5: budget-aware delay, patience,
        # unsent) get first dibs on populating ctx.delays. Pre-M7
        # behaviour (random.uniform(min,max) sleep before each chain)
        # is preserved exactly when no higher-priority hook intervenes.
        self._default_delay_eh = register_default_hooks(self.kira_config)

        # ====== init MCP manager ======
        try:
            self.mcp_manager = MCPManager(self.llm_api)
            await self.mcp_manager.init_mcp()
        except Exception as e:
            logger.error(f"Failed to initialize MCPManager: {e}")

        # ====== init skills manager ======

        self.skills_manager = SkillsManager()

        # ====== init message processor ======
        self.message_processor = MessageProcessor(
            db=self.db_service,
            kira_config=self.kira_config,
            llm_api=self.llm_api,
            provider_manager=self.provider_manager,
            skills_manager=self.skills_manager,
            adapter_manager=self.adapter_manager,
            session_manager=self.session_manager,
            prompt_manager=self.prompt_manager)
        # Phase 0.4 M5: hand the pipeline to the message processor so its
        # send_xml_messages/handle_im_batch_message paths can drain &
        # ingest hints_key state inline (see message_manager.py).
        self.message_processor.hints_pipeline = self.hints_pipeline

        self.tasks.append(
            asyncio.create_task(
                self.message_processor.cleanup_image_desc_cache_task(),
                name="image_desc_cache_cleanup"
            )
        )

        self.event_bus = EventBus(self.stats, event_queue, self.message_processor)

        # ====== init plugin system ======
        self.plugin_context = PluginContext(
            db=self.db_service,
            config=self.kira_config,
            event_bus=self.event_bus,
            provider_mgr=self.provider_manager,
            model_group_mgr=self.model_group_manager,
            llm_api=self.llm_api,
            adapter_mgr=self.adapter_manager,
            persona_mgr=self.persona_manager,
            sticker_manager=self.sticker_manager,
            session_mgr=self.session_manager,
            message_processor=self.message_processor,
            # Phase 0.2 M3: hand the registry to the plugin context so
            # PluginManager._register_plugin_prompt_blocks_for can find it
            # via self.ctx.prompt_block_registry.
            prompt_block_registry=self.prompt_block_registry,
            # Phase 0.4 M5: same idea for the hints_pipeline; PluginManager
            # ._register_plugin_hints_handlers_for / _cleanup_plugin_registration
            # reach it via self.ctx.hints_pipeline.
            hints_pipeline=self.hints_pipeline,
        )

        self.plugin_manager = PluginManager(self.plugin_context)
        self.plugin_context.plugin_mgr = self.plugin_manager
        await self.plugin_manager.init()
        webui_app = getattr(self, "webui_app", None)
        if webui_app is not None:
            self.plugin_manager.set_web_app(webui_app)

        # ====== init temp folder monitor ======
        temp_folder = get_data_path() / "temp"
        max_temp_size = getattr(self.kira_config, "max_temp_size_mb", 100)  # 从配置读取，默认100MB
        check_interval = getattr(self.kira_config, "temp_check_interval", 60)  # 默认60秒

        self.temp_monitor = AsyncTempMonitor(
            folder_path=str(temp_folder),
            max_size_mb=50,
            check_interval=10,
            batch_size=20
        )

        self.tasks.append(
            asyncio.create_task(
                self.temp_monitor.start_monitoring(),
                name="temp_folder_monitor"
            )
        )

        # ====== schedule tasks ======
        asyncio.create_task(self.schedule_tasks())

        logger.info("All modules initialized, starting message processing loop...")

        await self.event_bus.dispatch()

    async def stop(self):
        # shutdown telemetry client
        if self.telemetry_client:
            await self.telemetry_client.shutdown()

        # terminate all running adapters
        await self.adapter_manager.stop_adapters()
        await self.event_bus.stop()

        # dispose database manager
        if self.db_manager:
            await self.db_manager.dispose()

        # cancel all tasks
        for task in self.tasks:
            task.cancel()
