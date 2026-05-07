import importlib
import importlib.util
import inspect
import os
import json
import sys
import types
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from core.utils.path_utils import get_data_path, get_config_path
from core.logging_manager import get_logger
from core.config.config_field import BaseConfigField, build_fields
from .plugin import BasePlugin
from .plugin_context import PluginContext
from .plugin_handlers import Priority, event_handler_reg, EventHandler, EventType

from core.tag import tag_registry, BaseTag

logger = get_logger("plugin_manager", "cyan")

PLUGINS_DIR = get_data_path() / "plugins"
PLUGIN_DATA_DIR = get_data_path() / "plugin_data"
PLUGIN_CONFIG_DIR = get_config_path() / "plugins"
PLUGIN_STATE_FILE = get_config_path() / "plugins.json"
BUILTIN_PLUGINS_DIR = Path(__file__).parent / "builtin_plugins"

_plugin_classes: Dict[str, type[BasePlugin]] = {}
_plugin_manifests: Dict[str, Dict[str, Any]] = {}
_plugin_module_dirs: Dict[str, str] = {}
_plugin_module_paths: Dict[str, Path] = {}

"""key: module name, value: plugin id"""
_module_to_plugin: Dict[str, str] = {}
_plugin_schemas: Dict[str, List[BaseConfigField]] = {}
_plugin_components: Dict[str, dict] = {}


"""plugin_ids whose API routes have already been added to FastAPI."""
_plugin_api_registered: set[str] = set()


def get_obj_plugin_id(obj: Any):
    module = inspect.getmodule(obj)
    module_name = module.__name__ if module else ""
    plugin_id = _module_to_plugin.get(module_name, "")

    if not plugin_id and module and getattr(module, "__file__", None):
        module_path = Path(module.__file__).resolve()
        plugin_root = module_path.parent
        manifest_path = plugin_root / "manifest.json"
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                plugin_id = manifest.get("plugin_id") or plugin_root.name
                _plugin_manifests.setdefault(plugin_id, manifest)
                _plugin_module_dirs.setdefault(plugin_id, plugin_root.name)
                _plugin_module_paths.setdefault(plugin_id, plugin_root)
                _module_to_plugin[module_name] = plugin_id
            except Exception:
                plugin_id = plugin_root.name
    return plugin_id


class RegisterDeco:

    @staticmethod
    def tool(name: str, description: str, params: dict):
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            tools = plugin_entry.setdefault("tools", {})
            tool_funcs = plugin_entry.setdefault("tool_funcs", {})
            tools[name] = {
                "name": name,
                "description": description,
                "parameters": params,
                "func": func,
            }
            tool_funcs[name] = func

            return func

        return decorator

    @staticmethod
    def tag(name: str, description: str):
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            tags = plugin_entry.setdefault("tags", [])
            tag_funcs = plugin_entry.setdefault("tag_funcs", {})
            tags.append({
                "name": name,
                "description": description
            })
            tag_funcs[name] = func
            return func
        return decorator

    @staticmethod
    def page(route: str, auth: bool = True, menu: Optional[dict] = None):
        """Register a plugin page endpoint.

        route: URL path relative to plugin prefix, e.g., "/dashboard"
               Final route: /page/plugin/{plugin_id}{route}
        auth:  Require JWT auth (default True)
        menu:  Optional menu configuration for sidebar integration
        """
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            pages = plugin_entry.setdefault("pages", [])
            page_funcs = plugin_entry.setdefault("page_funcs", {})
            pages.append({
                "route": route,
                "func": func,
                "auth": auth,
                "menu": menu,
            })
            page_funcs[func.__name__] = func
            return func
        return decorator

    @staticmethod
    def static(path: str, directory: str, html: bool = False):
        """Register a static file directory.

        path:      URL path prefix relative to plugin, e.g., "/static"
                   Final URL: /page/plugin/{plugin_id}{path}
        directory: Local directory path relative to plugin root
        html:      Try to serve index.html for directory requests
        """
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            static_dirs = plugin_entry.setdefault("static_dirs", [])
            static_dirs.append({
                "path": path,
                "directory": directory,
                "html": html,
            })
            return func
        return decorator

    @staticmethod
    def api(method: str, path: str, auth: bool = True, **kwargs):
        """Register a plugin API endpoint.

        method: HTTP method, e.g. "GET", "POST"
        path:   Path relative to the plugin prefix, e.g. "/status"
                Final route: /api/plugin/{plugin_id}{path}
        auth:   Require JWT auth (default True)
        kwargs: Forwarded to FastAPI add_api_route (response_model, summary, …)
        """
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            api_routes = plugin_entry.setdefault("api_routes", [])
            api_funcs = plugin_entry.setdefault("api_route_funcs", {})
            api_routes.append({
                "method": method.upper(),
                "path":   path,
                "func":   func,
                "auth":   auth,
                "kwargs": kwargs,
            })
            api_funcs[func.__name__] = func
            return func
        return decorator

    @staticmethod
    def prompt_block(
        name: str,
        depth: int = 100,
        role: str = "system",
        enabled: bool = True,
        cache_key: Optional[str] = None,
        position: str = "system",
        inject_depth: int = 0,
        inject_role: str = "system",
    ):
        """Phase 0.2 M3: register a static PromptBlock for the assembler.

        The decorated method's signature can be either ``def f(self)``,
        ``def f(self, ctx)``, or async variants; the assembler dispatches
        with or without the ctx_snapshot based on parameter count.

        At decorator-scan time we only stash metadata + the unbound function
        in `_plugin_components[plugin_id]["prompt_blocks"]`. The bound
        `PromptBlock` is constructed and pushed into the registry inside
        `PluginManager._register_plugin_prompt_blocks_for` once the plugin
        instance exists.

        :param name:         block identifier within this plugin's scope
        :param depth:        **system-prompt** sort key, smaller = closer
                             to the front (default 100). Ignored when
                             ``position == "in_chat"``.
        :param role:         DEPRECATED — kept for M3 backward compat. Use
                             ``position="in_chat"`` instead of
                             ``role="user"`` if you actually want the block
                             to reach the LLM.
        :param enabled:      static off-switch, default True
        :param cache_key:    optional cache key for de-duplicating providers
                             inside one assemble() call
        :param position:     ``"system"`` (default — block joins the system
                             prompt sorted by ``depth``) or ``"in_chat"``
                             (block becomes a single message inserted into
                             the chat history at ``inject_depth`` counted
                             back from the end).
        :param inject_depth: when ``position == "in_chat"``, offset from
                             the END of ``request.messages``. depth=0 →
                             before the latest user message; depth=N →
                             before the Nth-last message. Default 0.
        :param inject_role:  when ``position == "in_chat"``, role of the
                             injected message. Default ``"system"``.
        """
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            prompt_blocks = plugin_entry.setdefault("prompt_blocks", [])
            prompt_blocks.append({
                "name": name,
                "depth": depth,
                "role": role,
                "enabled": enabled,
                "cache_key": cache_key,
                "position": position,
                "inject_depth": inject_depth,
                "inject_role": inject_role,
                "func": func,
            })
            return func
        return decorator

    @staticmethod
    def hints_handler(
        key_type: str,
        ttl_seconds: Optional[int] = None,
        mode: str = "every",
        interval: int = 1,
        probability: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Phase 0.4 M5: register a HintsKeyHandler factory on a plugin.

        The decorated method receives ``self`` (the plugin instance) and
        must return a fully-constructed :class:`HintsKeyHandler` instance.
        Registration happens during ``init_plugin`` after the plugin
        instance exists, mirroring how ``@register.prompt_block`` is bound.

        Why a factory and not the handler class itself: most handlers
        need plugin state (ctx, sticker_manager, model_group, config…)
        that only exists after ``__init__``. Letting the plugin produce
        the instance is far less awkward than threading those into a
        zero-arg constructor.

        :param key_type:    matches the ``type`` attribute of
                            ``<hints_key type="...">``. The decorator
                            ALSO sets this on the returned handler if
                            the handler's own ``key_type`` is empty.
        :param ttl_seconds: optional override for handler.ttl_seconds.
                            ``None`` means honour whatever the handler
                            instance declares.
        :param mode:        FrequencyPolicy mode. ``"every"`` (default)
                            schedules ``prepare()`` on every emission;
                            ``"interval"`` only schedules every Nth
                            emission (see ``interval``); ``"random"``
                            schedules with probability ``probability``.
        :param interval:    when ``mode="interval"``, schedule on the
                            ``N``-th emission and every ``N``-th after.
                            Defaults to 1 (= every).
        :param probability: when ``mode="random"``, probability in
                            ``[0.0, 1.0]`` of scheduling each emission.
                            Defaults to 1.0 (= every).
        :param seed:        when ``mode="random"``, RNG seed for
                            deterministic replays. ``None`` uses the OS
                            random source.
        """
        def decorator(func: Callable):
            plugin_id = get_obj_plugin_id(func)
            plugin_entry = _plugin_components.setdefault(plugin_id, {})
            hints_factories = plugin_entry.setdefault("hints_handlers", [])
            hints_factories.append({
                "key_type": key_type,
                "ttl_seconds": ttl_seconds,
                "mode": mode,
                "interval": interval,
                "probability": probability,
                "seed": seed,
                "func": func,
            })
            return func
        return decorator


class OnEventDeco:

    @staticmethod
    def _register_hook(func: Callable, priority: Union[Priority, int], event_type: EventType):
        plugin_id = get_obj_plugin_id(func)
        eh = EventHandler(
            event_type=event_type,
            priority=priority,
            handler=func,
            desc=func.__doc__
        )

        plugin_entry = _plugin_components.setdefault(plugin_id, {})
        hooks = plugin_entry.setdefault("hooks", [])
        hooks.append(eh)

    def im_message(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_IM_MESSAGE)
            return func
        return decorator

    def message_buffered(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_MESSAGE_BUFFERED)
            return func
        return decorator

    def im_batch_message(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_IM_BATCH_MESSAGE)
            return func
        return decorator

    def llm_request(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_LLM_REQUEST)
            return func
        return decorator

    def prompt_assemble(self, priority: Union[Priority, int] = Priority.MEDIUM):
        """Phase 0.2 M3: subscribe to PromptBlock collection stage.

        Handler signature: ``async def f(event, ctx_snapshot, block_collector)``
        Use ``block_collector.add(PromptBlock(...))`` to contribute runtime
        blocks (whose presence/content depends on the current turn). For
        blocks that are statically defined for the plugin's lifetime, use
        ``@register.prompt_block`` instead.
        """
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_PROMPT_ASSEMBLE)
            return func
        return decorator

    def llm_response(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_LLM_RESPONSE)
            return func
        return decorator

    def after_llm_response_parse(self, priority: Union[Priority, int] = Priority.MEDIUM):
        """Phase 0.4 M5: subscribe to the post-XML-parse stage.

        Fires after kira-ai's XML repair (ON_LLM_RESPONSE, SYS_HIGH-1
        priority for the repair handler) AND after _parse_xml_msg has
        produced ``message_chains``, but BEFORE messages are sent.

        Handler signature::

            async def f(event, llm_response, raw_xml: str,
                        message_chains: list[MessageChain]) -> None

        ``raw_xml`` is the post-repair text. The HintsPipeline's own
        consume_response invocation runs alongside this event (in
        message_manager) so plugins that want to peek at hints_key
        emissions for non-pipeline reasons can do so here without
        racing the pipeline's own ingest.
        """
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.AFTER_LLM_RESPONSE_PARSE)
            return func
        return decorator

    def hints_produced(self, priority: Union[Priority, int] = Priority.MEDIUM):
        """Post-M5 amendment: observe hints pipeline outcomes.

        Fires fire-and-forget (each observer in its own asyncio.Task)
        whenever a ``<hints_key>`` emission resolves — success with a
        produced PromptBlock, success with None, prepare exception, or
        FrequencyPolicy skip. Handler signature::

            async def f(event, hints_result: HintsResult) -> None

        Where ``hints_result`` carries ``session_id`` / ``key_type`` /
        ``keys`` / ``block`` / ``error`` / ``elapsed_ms`` /
        ``skipped_by_policy`` / ``plugin_id`` (see
        :class:`core.pipeline.hints_pipeline.HintsResult`).

        Use cases: logging hint stats, cross-plugin chains (one plugin's
        hint triggers another's behaviour), debug dashboards. **Do not**
        register a HintsKeyHandler for a key_type you don't own — use
        this observer instead and read ``hints_result.block`` to inspect
        the foreign plugin's contribution.

        Observer slowness does not delay the user-visible message path
        (pipeline schedules each observer as a separate task). Cancellation
        of a prepare() task does NOT fire this event (that path is
        considered an internal overwrite, not a real outcome).
        """
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_HINTS_PRODUCED)
            return func
        return decorator

    def tool_result(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_TOOL_RESULT)
            return func
        return decorator

    def after_xml_parse(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.AFTER_XML_PARSE)
            return func
        return decorator

    def step_result(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_STEP_RESULT)
            return func
        return decorator

    def final_result(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_FINAL_RESULT)
            return func
        return decorator

    def exception(self, priority: Union[Priority, int] = Priority.MEDIUM):
        def decorator(func: Callable):
            self._register_hook(func, priority, EventType.ON_EXCEPTION)
            return func
        return decorator


register = RegisterDeco()
on = OnEventDeco()

register_tool = register.tool


def _build_tag_inst(tag_name: str, tag_description: str, func: Callable):
    class TagInst(BaseTag):
        name = tag_name
        description = tag_description

        async def handle(self, value: str, **kwargs):
            res = await func(value, **kwargs)
            return res

    return TagInst()


class PluginManager:
    """
    Plugin manager for KiraAI, detecting plugins automatically
    """

    def __init__(self, ctx: Optional[PluginContext] = None):
        self.plugins: List[BasePlugin] = []
        self.plugin_dir = Path(PLUGINS_DIR)
        self.plugin_data_dir = Path(PLUGIN_DATA_DIR)
        self.ctx = ctx
        self.plugin_instances: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_enabled: Dict[str, bool] = {}
        self._web_app = None

        self._load_plugin_state()

    def set_web_app(self, app) -> None:
        """Provide the FastAPI app instance so plugin API routes can be registered.
        Also registers routes for any plugins that were already initialized before this call.
        """
        self._web_app = app
        for plugin_id in list(self.plugin_instances.keys()):
            self._register_plugin_apis_for(plugin_id)
            self._register_plugin_pages_for(plugin_id)
            self._register_plugin_static_for(plugin_id)

    def get_plugin_inst(self, plugin_id: str):
        return self.plugin_instances.get(plugin_id)

    def _load_plugin_state(self) -> None:
        try:
            config_dir = PLUGIN_STATE_FILE.parent
            config_dir.mkdir(parents=True, exist_ok=True)
            if PLUGIN_STATE_FILE.exists():
                with PLUGIN_STATE_FILE.open("r", encoding="utf-8") as f:
                    data = f.read()
                if data.strip():
                    raw = json.loads(data)
                    if isinstance(raw, dict):
                        self.plugin_enabled = {
                            str(k): bool(v) for k, v in raw.items()
                        }
        except Exception as e:
            logger.error(f"Failed to load plugin state from {PLUGIN_STATE_FILE}: {e}")
            self.plugin_enabled = {}

    def _save_plugin_state(self) -> None:
        try:
            config_dir = PLUGIN_STATE_FILE.parent
            config_dir.mkdir(parents=True, exist_ok=True)
            with PLUGIN_STATE_FILE.open("w", encoding="utf-8") as f:
                json.dump(self.plugin_enabled, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save plugin state to {PLUGIN_STATE_FILE}: {e}")

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        if not plugin_id:
            return False
        return self.plugin_enabled.get(plugin_id, True)

    async def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> None:
        if not plugin_id:
            return
        plugin_id = str(plugin_id)
        previous = self.plugin_enabled.get(plugin_id, True)
        self.plugin_enabled[plugin_id] = bool(enabled)
        self._save_plugin_state()

        if enabled and not previous:
            await self.init_plugin(plugin_id)
        elif not enabled and previous:
            try:
                await self.terminate(plugin_id)
            except Exception as e:
                logger.error(f"Failed to terminate plugin {plugin_id} when disabling: {e}")

    def get_registered_plugins(self) -> Dict[str, type[BasePlugin]]:
        return dict(_plugin_classes)

    def get_plugin_manifest(self, name: str) -> Dict[str, Any]:
        return _plugin_manifests.get(name, {})

    def get_plugin_module_dir(self, name: str) -> str:
        return _plugin_module_dirs.get(name, "")

    def get_plugin_module_path(self, name: str) -> Optional[Path]:
        return _plugin_module_paths.get(name)

    def is_builtin_plugin(self, plugin_id: str) -> bool:
        path = _plugin_module_paths.get(plugin_id)
        if path is None:
            return False
        return path.is_relative_to(BUILTIN_PLUGINS_DIR)

    def is_plugin_hidden(self, plugin_id: str) -> bool:
        if not self.is_builtin_plugin(plugin_id):
            return False
        manifest = _plugin_manifests.get(plugin_id, {})
        return bool(manifest.get("hide", False))

    def is_plugin_uninstallable(self, plugin_id: str) -> bool:
        if not self.is_builtin_plugin(plugin_id):
            return True
        manifest = _plugin_manifests.get(plugin_id, {})
        return bool(manifest.get("uninstallable", False))

    def get_plugin_id_for_module(self, module_name: str) -> Optional[str]:
        return _module_to_plugin.get(module_name)

    def get_plugin_schema(self, name: str) -> List[BaseConfigField]:
        return _plugin_schemas.get(name, [])

    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        plugin_name = str(plugin_name)
        if plugin_name in self.plugin_configs:
            return dict(self.plugin_configs.get(plugin_name, {}))
        schema_fields = _plugin_schemas.get(plugin_name, [])
        if schema_fields:
            self._ensure_plugin_config(plugin_name, schema_fields)
            return dict(self.plugin_configs.get(plugin_name, {}))
        cfg = self._load_plugin_config_from_file(plugin_name)
        self.plugin_configs[plugin_name] = cfg
        return dict(cfg)

    async def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        plugin_name = str(plugin_name)
        if not isinstance(config, dict):
            config = {}
        schema_fields = _plugin_schemas.get(plugin_name, [])
        if schema_fields:
            self._ensure_plugin_config(plugin_name, schema_fields)
        current_cfg = self.plugin_configs.get(plugin_name)
        if current_cfg is None:
            current_cfg = self._load_plugin_config_from_file(plugin_name)
            self.plugin_configs[plugin_name] = current_cfg
        for key, value in config.items():
            current_cfg[key] = value
        PLUGIN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = PLUGIN_CONFIG_DIR / f"{plugin_name}.json"
        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(current_cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save plugin config for {plugin_name}: {e}")
        self.plugin_configs[plugin_name] = current_cfg
        if plugin_name in self.plugin_instances:
            await self.init_plugin(plugin_name)
        return dict(current_cfg)

    def get_plugin_components(self) -> Dict[str, dict]:
        return dict(_plugin_components)

    def get_plugin_tools(self, plugin_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        if plugin_name is None:
            return {name: comp.get("tools", {}) for name, comp in _plugin_components.items()}
        entry = _plugin_components.get(plugin_name, {})
        return entry.get("tools", {})

    def _register_plugin_tools_for(self, plugin_id: str) -> None:
        comp = _plugin_components.get(plugin_id, {})
        if not comp:
            return
        tools = comp.get("tools", {})
        tool_funcs = comp.get("tool_funcs", {})
        plugin_instance = self.plugin_instances.get(plugin_id)
        tool_names: list[str] = []
        for tool_name, meta in tools.items():
            func = tool_funcs.get(tool_name)
            if not func:
                continue
            bound_func = func
            if plugin_instance is not None and hasattr(plugin_instance, func.__name__):
                bound_func = getattr(plugin_instance, func.__name__)
            self.ctx.llm_api.register_tool(
                name=tool_name,
                description=meta.get("description", ""),
                parameters=meta.get("parameters") or {},
                func=bound_func,
            )
            tool_names.append(tool_name)
        if tool_names:
            logger.info(f"Registered {len(tool_names)} tools from {plugin_id}: {tool_names}")

    def _register_plugin_hooks_for(self, plugin_id: str):
        comp = _plugin_components.get(plugin_id, {})
        if not comp:
            return
        hooks = comp.get("hooks", [])
        plugin_instance = self.plugin_instances.get(plugin_id)
        for hook in hooks:
            bound_handler = hook.handler
            if plugin_instance is not None and bound_handler is not None and hasattr(
                plugin_instance, bound_handler.__name__
            ):
                candidate = getattr(plugin_instance, bound_handler.__name__)
                if candidate is not None:
                    bound_handler = candidate
            hook.handler = bound_handler
            event_handler_reg.register(hook)
        if hooks:
            logger.info(f"Registered {len(hooks)} hooks from {plugin_id}")

    def _register_plugin_prompt_blocks_for(self, plugin_id: str):
        """Phase 0.2 M3: bind plugin prompt-block functions to the
        plugin instance and push them into the global PromptBlockRegistry.

        Mirrors `_register_plugin_hooks_for` for hooks. Called from
        `init_plugin` after the plugin instance has been created and
        `initialize()` has succeeded. If the plugin context has no
        prompt_block_registry attribute (lifecycle didn't wire one in,
        e.g. in degraded boot), this is a silent no-op so plugin loading
        is not blocked by the M3 plumbing.
        """
        comp = _plugin_components.get(plugin_id, {})
        if not comp:
            return
        prompt_blocks_meta = comp.get("prompt_blocks", [])
        if not prompt_blocks_meta:
            return
        registry = (
            getattr(self.ctx, "prompt_block_registry", None)
            if self.ctx is not None else None
        )
        if registry is None:
            logger.debug(
                f"PromptBlockRegistry unavailable in PluginContext, "
                f"skipping prompt_block registration for {plugin_id}"
            )
            return
        # Local import: avoids importing prompt_block at module load time
        # (keeps the legacy boot path intact if prompt_block.py is missing
        # in a partial deployment).
        from core.prompt_block import PromptBlock
        plugin_instance = self.plugin_instances.get(plugin_id)
        block_names: List[str] = []
        # First clear any prior registration for this plugin so re-init
        # (e.g. config update) doesn't accumulate duplicates.
        registry.clear_plugin(plugin_id)
        for meta in prompt_blocks_meta:
            func = meta["func"]
            bound_func = func
            if plugin_instance is not None and hasattr(plugin_instance, func.__name__):
                bound_func = getattr(plugin_instance, func.__name__)
            block = PromptBlock(
                name=meta["name"],
                content_provider=bound_func,
                depth=meta.get("depth", 100),
                role=meta.get("role", "system"),
                enabled=meta.get("enabled", True),
                cache_key=meta.get("cache_key"),
                position=meta.get("position", "system"),
                inject_depth=meta.get("inject_depth", 0),
                inject_role=meta.get("inject_role", "system"),
                source=f"plugin:{plugin_id}",
            )
            registry.register(block, plugin_id=plugin_id)
            block_names.append(meta["name"])
        if block_names:
            logger.info(
                f"Registered {len(block_names)} prompt blocks from "
                f"{plugin_id}: {block_names}"
            )

    def _register_plugin_hints_handlers_for(self, plugin_id: str):
        """Phase 0.4 M5: invoke each ``@register.hints_handler`` factory
        on the plugin instance and register the returned handler with
        the HintsPipeline.

        Pattern mirrors ``_register_plugin_prompt_blocks_for``: the
        decorator records ``(key_type, ttl, func)`` at scan time, and
        we materialise the handler instance after ``initialize()`` so
        plugin state (ctx, sticker_manager, model_group, etc.) is
        available. Re-init clears prior registrations first to avoid
        stale handlers.

        Silent no-op if PluginContext has no ``hints_pipeline`` attribute
        (lifecycle didn't wire one in — degraded path).
        """
        comp = _plugin_components.get(plugin_id, {})
        if not comp:
            return
        factories = comp.get("hints_handlers", [])
        if not factories:
            return
        pipeline = (
            getattr(self.ctx, "hints_pipeline", None)
            if self.ctx is not None else None
        )
        if pipeline is None:
            logger.debug(
                f"HintsPipeline unavailable in PluginContext, "
                f"skipping hints_handler registration for {plugin_id}"
            )
            return
        # Drop any prior contributions first.
        try:
            pipeline.clear_plugin(plugin_id)
        except Exception as e:
            logger.warning(
                f"clear_plugin failed for {plugin_id} hints_handlers: {e}"
            )

        plugin_instance = self.plugin_instances.get(plugin_id)
        registered_types: List[str] = []
        for meta in factories:
            func = meta["func"]
            bound_func = func
            if plugin_instance is not None and hasattr(plugin_instance, func.__name__):
                bound_func = getattr(plugin_instance, func.__name__)
            try:
                handler = bound_func()
            except Exception as e:
                logger.error(
                    f"hints_handler factory '{func.__name__}' from "
                    f"{plugin_id} raised; skipping: {e}"
                )
                continue
            if handler is None:
                # Factory may opt out at runtime (e.g. feature flag off).
                continue
            # Apply decorator-supplied overrides if the handler instance
            # didn't set them itself.
            if not getattr(handler, "key_type", ""):
                handler.key_type = meta["key_type"]
            if meta.get("ttl_seconds") is not None:
                handler.ttl_seconds = int(meta["ttl_seconds"])
            # Build FrequencyPolicy. Precedence (highest first):
            #   1. handler instance's ``_frequency_policy`` attribute —
            #      lets the factory read plugin config and produce a
            #      runtime-decided policy without going through the
            #      decorator.
            #   2. decorator-supplied mode/interval/probability/seed —
            #      static authorial intent.
            # Local import so plugin_registry doesn't pull in pipeline
            # at module load.
            policy = getattr(handler, "_frequency_policy", None)
            if policy is None:
                try:
                    from core.pipeline.hints_pipeline import FrequencyPolicy
                    policy = FrequencyPolicy(
                        mode=meta.get("mode", "every"),
                        interval=int(meta.get("interval", 1)),
                        probability=float(meta.get("probability", 1.0)),
                        seed=meta.get("seed"),
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to build FrequencyPolicy for "
                        f"key_type='{handler.key_type}' from {plugin_id} "
                        f"(falling back to every): {e}"
                    )
            try:
                pipeline.register(handler, plugin_id=plugin_id, policy=policy)
            except Exception as e:
                logger.error(
                    f"HintsPipeline.register failed for "
                    f"key_type='{handler.key_type}' from {plugin_id}: {e}"
                )
                continue
            registered_types.append(handler.key_type)
        if registered_types:
            logger.info(
                f"Registered {len(registered_types)} hints handlers "
                f"from {plugin_id}: {registered_types}"
            )

    def _register_plugin_tags_for(self, plugin_id: str):
        comp = _plugin_components.get(plugin_id, {})
        if not comp:
            return
        tags = comp.get("tags", [])
        tag_funcs = comp.get("tag_funcs", {})
        plugin_instance = self.plugin_instances.get(plugin_id)
        tag_names: list[str] = []
        for tag_meta in tags:
            tag_name = tag_meta["name"]
            func = tag_funcs.get(tag_name)
            if not func:
                continue
            bound_func = func
            if plugin_instance is not None and hasattr(plugin_instance, func.__name__):
                bound_func = getattr(plugin_instance, func.__name__)
            tag_registry.register(_build_tag_inst(
                tag_name,
                tag_meta["description"],
                bound_func
            ))
            tag_names.append(tag_name)
        if tag_names:
            logger.info(f"Registered {len(tag_names)} tags from {plugin_id}: {tag_names}")

    def _register_plugin_apis_for(self, plugin_id: str) -> None:
        if self._web_app is None:
            return
        comp = _plugin_components.get(plugin_id, {})
        api_routes = comp.get("api_routes", [])
        if not api_routes:
            return

        # Routes already in FastAPI: the dynamic_endpoint always looks up the
        # current instance at call time, so re-init requires no action here.
        if plugin_id in _plugin_api_registered:
            return

        import typing
        from fastapi import Depends, HTTPException
        from webui.routes.auth import require_auth

        _plugin_api_registered.add(plugin_id)
        mgr = self

        def _make_plugin_check(pid: str):
            async def check():
                if not mgr.is_plugin_enabled(pid):
                    raise HTTPException(status_code=404, detail="Plugin disabled")
            return check

        registered: List[str] = []

        for route in api_routes:
            func = route["func"]
            func_name = func.__name__
            full_path = f"/api/plugin/{plugin_id}/{route['path'].lstrip('/')}"

            # Resolve annotations eagerly using the plugin module's own globals,
            # so `from __future__ import annotations` in plugins is handled correctly.
            try:
                resolved_hints = typing.get_type_hints(func, globalns=func.__globals__)
            except Exception:
                resolved_hints = {}

            params = [
                p.replace(annotation=resolved_hints.get(name, p.annotation))
                for name, p in inspect.signature(func).parameters.items()
                if name != "self"
            ]

            # Capture loop variables via default args to avoid closure issues.
            async def dynamic_endpoint(
                _pid=plugin_id, _fname=func_name, _mgr=mgr, **kwargs
            ):
                inst = _mgr.plugin_instances.get(_pid)
                if inst is None:
                    raise HTTPException(status_code=503, detail="Plugin not available")
                return await getattr(inst, _fname)(**kwargs)

            dynamic_endpoint.__signature__ = inspect.Signature(params)

            dependencies = [Depends(_make_plugin_check(plugin_id))]
            if route["auth"]:
                dependencies.append(Depends(require_auth))

            self._web_app.add_api_route(
                path=full_path,
                endpoint=dynamic_endpoint,
                methods=[route["method"]],
                dependencies=dependencies,
                tags=[f"plugin:{plugin_id}"],
                **route["kwargs"],
            )
            registered.append(full_path)

        if registered:
            logger.info(f"Registered {len(registered)} API routes from {plugin_id}: {registered}")

    def _register_plugin_pages_for(self, plugin_id: str) -> None:
        """Register plugin page routes for URL access."""
        if self._web_app is None:
            return

        comp = _plugin_components.get(plugin_id, {})
        pages = comp.get("pages", [])
        if not pages:
            return

        # Track registered pages to avoid duplicates
        if plugin_id in getattr(self, '_plugin_pages_registered', set()):
            return

        if not hasattr(self, '_plugin_pages_registered'):
            self._plugin_pages_registered = set()
        self._plugin_pages_registered.add(plugin_id)

        import typing
        from fastapi import Depends, HTTPException
        from webui.routes.auth import require_auth

        mgr = self
        registered: List[str] = []

        for page in pages:
            func = page["func"]
            func_name = func.__name__
            route_path = page["route"].lstrip('/')
            full_path = f"/page/plugin/{plugin_id}/{route_path}"

            # Handle catch-all routes (e.g., /{path:path})
            if '{' in route_path and ':path}' in route_path:
                full_path = f"/page/plugin/{plugin_id}/" + "{path:path}"

            # Resolve annotations eagerly using the plugin module's own globals
            try:
                resolved_hints = typing.get_type_hints(func, globalns=func.__globals__)
            except Exception:
                resolved_hints = {}

            params = [
                p.replace(annotation=resolved_hints.get(name, p.annotation))
                for name, p in inspect.signature(func).parameters.items()
                if name != "self"
            ]

            # Capture loop variables via default args to avoid closure issues
            async def dynamic_page_endpoint(
                _pid=plugin_id, _fname=func_name, _mgr=mgr, **kwargs
            ):
                inst = _mgr.plugin_instances.get(_pid)
                if inst is None:
                    raise HTTPException(status_code=503, detail="Plugin not available")
                return await getattr(inst, _fname)(**kwargs)

            dynamic_page_endpoint.__signature__ = inspect.Signature(params)

            dependencies = []
            if page["auth"]:
                dependencies.append(Depends(require_auth))

            self._web_app.add_api_route(
                path=full_path,
                endpoint=dynamic_page_endpoint,
                methods=["GET"],
                dependencies=dependencies,
                tags=[f"plugin:{plugin_id}"],
            )
            registered.append(full_path)

        if registered:
            logger.info(f"Registered {len(registered)} page routes from {plugin_id}: {registered}")

    def _register_plugin_static_for(self, plugin_id: str) -> None:
        """Register static file routes for a plugin with plugin state check."""
        if self._web_app is None:
            return

        comp = _plugin_components.get(plugin_id, {})
        static_dirs = comp.get("static_dirs", [])
        if not static_dirs:
            return

        # Track registered static dirs to avoid duplicates
        if plugin_id in getattr(self, '_plugin_static_registered', set()):
            return

        if not hasattr(self, '_plugin_static_registered'):
            self._plugin_static_registered = set()
        self._plugin_static_registered.add(plugin_id)

        from fastapi import Depends, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.routing import Route
        import mimetypes

        mgr = self
        plugin_root = _plugin_module_paths.get(plugin_id)
        if not plugin_root:
            return

        def _make_plugin_check(pid: str):
            async def check():
                if not mgr.is_plugin_enabled(pid):
                    raise HTTPException(status_code=404, detail="Plugin disabled")
            return check

        registered: List[str] = []

        for static in static_dirs:
            path_prefix = static["path"].lstrip('/')
            full_path = f"/page/plugin/{plugin_id}/{path_prefix}"
            dir_path = plugin_root / static["directory"]

            if not dir_path.exists() or not dir_path.is_dir():
                continue

            # Create a custom StaticFiles class that checks plugin state
            class PluginStaticFiles(StaticFiles):
                def __init__(self, directory: str, check_func, html: bool = False):
                    super().__init__(directory=directory, html=html)
                    self._check_func = check_func

                async def __call__(self, scope, receive, send):
                    # Check plugin state before serving files
                    await self._check_func()
                    await super().__call__(scope, receive, send)

            try:
                self._web_app.mount(
                    full_path,
                    PluginStaticFiles(
                        directory=str(dir_path),
                        check_func=_make_plugin_check(plugin_id),
                        html=static.get("html", False)
                    ),
                    name=f"plugin_{plugin_id}_static_{path_prefix}"
                )
                registered.append(full_path)
            except Exception as e:
                logger.error(f"Failed to mount static dir {dir_path} for plugin {plugin_id}: {e}")

        if registered:
            logger.info(f"Registered {len(registered)} static directories from {plugin_id}: {registered}")

    def register_plugin_tools(self) -> None:
        for plugin_id in _plugin_components.keys():
            self._register_plugin_tools_for(plugin_id)

    def _cleanup_plugin_registration(self, plugin_id: str) -> None:
        comp = _plugin_components.get(plugin_id)
        if not comp:
            return

        # clean up tool registration
        tools = comp.get("tools", {})
        if self.ctx and getattr(self.ctx, "llm_api", None):
            for tool_name in list(tools.keys()):
                try:
                    self.ctx.llm_api.unregister_tool(tool_name)
                except Exception as e:
                    logger.error(f"Failed to unregister tool {tool_name} for plugin {plugin_id}: {e}")

        # clean up hook registration
        hooks = comp.get("hooks", [])
        for hook in hooks:
            event_handler_reg.del_handler(hook)

        # clean up tag registration
        tags = comp.get("tags", [])
        for tag in tags:
            tag_registry.unregister(tag.get("name"))

        # Phase 0.2 M3: clean up prompt block registration
        if self.ctx is not None:
            registry = getattr(self.ctx, "prompt_block_registry", None)
            if registry is not None:
                try:
                    registry.clear_plugin(plugin_id)
                except Exception as e:
                    logger.error(
                        f"Failed to clear prompt blocks for plugin {plugin_id}: {e}"
                    )

        # Phase 0.4 M5: clean up hints_pipeline handler registration.
        # Cancels any in-flight prepare() tasks owned by this plugin
        # and drops the handler entries so collect_blocks doesn't try
        # to surface results for handlers that no longer exist.
        if self.ctx is not None:
            pipeline = getattr(self.ctx, "hints_pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.clear_plugin(plugin_id)
                except Exception as e:
                    logger.error(
                        f"Failed to clear hints handlers for plugin {plugin_id}: {e}"
                    )

        # API routes: disable is handled at request time via the plugin_check Depends,
        # which calls is_plugin_enabled(). No additional cleanup needed here.

    def _load_plugin_config_from_file(self, plugin_id: str) -> Dict[str, Any]:
        PLUGIN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = PLUGIN_CONFIG_DIR / f"{plugin_id}.json"
        cfg: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    cfg = loaded
            except Exception as e:
                logger.error(f"Failed to load plugin config from {config_path}: {e}")
        return cfg

    def _ensure_plugin_config(self, plugin_name: str, schema_fields: List[BaseConfigField]) -> None:
        if not schema_fields:
            return
        PLUGIN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_path = PLUGIN_CONFIG_DIR / f"{plugin_name}.json"
        cfg: Dict[str, Any] = self._load_plugin_config_from_file(plugin_name)
        for field in schema_fields:
            if isinstance(field, BaseConfigField) and field.key not in cfg:
                cfg[field.key] = field.default
        try:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save plugin config for {plugin_name}: {e}")
        self.plugin_configs[plugin_name] = cfg

    async def init(self):
        """
        Initialize plugin manager and load all discovered plugins
        """
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.plugin_data_dir.mkdir(parents=True, exist_ok=True)

        await self._discover_builtin_plugins()
        await self._discover_user_plugins()

        discovered = list(_plugin_classes.keys())
        logger.info(f"Discovered plugins: {discovered}")

        for plugin_id in _plugin_classes.keys():
            if plugin_id in self.plugin_instances:
                continue
            await self.init_plugin(plugin_id)

    async def init_plugin(self, plugin_id: Optional[str] = None):
        if plugin_id is None:
            for pid in list(_plugin_classes.keys()):
                await self.init_plugin(pid)
            return

        plugin_id = str(plugin_id)
        plugin_cls = _plugin_classes.get(plugin_id)
        if not plugin_cls:
            logger.warning(f"No plugin class found for {plugin_id}, cannot initialize")
            return

        if not self.is_plugin_enabled(plugin_id):
            logger.debug(f"Plugin {plugin_id} is disabled, skipping initialization")
            return

        existing = self.plugin_instances.get(plugin_id)
        if existing is not None:
            try:
                await self.terminate(plugin_id)
            except Exception as e:
                logger.error(f"Error terminating plugin {plugin_id} before reinitialization: {e}")

        schema_fields = _plugin_schemas.get(plugin_id, [])
        if schema_fields:
            self._ensure_plugin_config(plugin_id, schema_fields)
            cfg = self.plugin_configs.get(plugin_id) or {}
        else:
            cfg: Dict[str, Any] = self._load_plugin_config_from_file(plugin_id)
            self.plugin_configs[plugin_id] = cfg

        try:
            instance = plugin_cls(self.ctx, cfg)
        except Exception as e:
            logger.error(f"Failed to instantiate plugin {plugin_id}: {e}")
            return
        self.plugin_instances[plugin_id] = instance
        initialized = False
        try:
            await instance.initialize()
            initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_id}: {e}")
        if initialized:
            self._register_plugin_tools_for(plugin_id)
            self._register_plugin_hooks_for(plugin_id)
            self._register_plugin_prompt_blocks_for(plugin_id)
            self._register_plugin_hints_handlers_for(plugin_id)
            self._register_plugin_tags_for(plugin_id)
            self._register_plugin_apis_for(plugin_id)
            self._register_plugin_pages_for(plugin_id)
            self._register_plugin_static_for(plugin_id)

    async def terminate(self, plugin_id: Optional[str] = None):
        """Terminate a specific plugin if plugin_id is given, terminate all if not given"""
        if plugin_id:
            try:
                plugin_instance = self.plugin_instances.get(plugin_id)
                if plugin_instance:
                    await plugin_instance.terminate()
                self.plugin_instances.pop(plugin_id, None)
                self.plugin_configs.pop(plugin_id, None)
                logger.info(f"Terminated plugin {plugin_id}")
            except Exception as e:
                logger.error(f"Error terminating plugin {plugin_id}: {e}")
            self._cleanup_plugin_registration(plugin_id)
            return

        for plug_id, plugin_instance in list(self.plugin_instances.items()):
            try:
                await plugin_instance.terminate()
            except Exception as e:
                logger.error(f"Error terminating plugin {plug_id}: {e}")

        # Clear registries
        self.plugin_instances.clear()
        self.plugin_configs.clear()
        for name in list(_plugin_components.keys()):
            self._cleanup_plugin_registration(name)

    async def uninstall_plugin(self, plugin_id: str) -> None:
        """
        Terminate a plugin and remove all its registrations from memory.
        The caller is responsible for deleting the plugin directory afterwards.
        """
        if plugin_id not in _plugin_classes:
            raise ValueError(f"Plugin '{plugin_id}' is not registered")

        # Stop the running instance and unregister tools / hooks / tags
        await self.terminate(plugin_id)

        # Remove from global registries
        _plugin_classes.pop(plugin_id, None)
        _plugin_manifests.pop(plugin_id, None)
        _plugin_module_dirs.pop(plugin_id, None)
        _plugin_module_paths.pop(plugin_id, None)
        _plugin_schemas.pop(plugin_id, None)
        _plugin_components.pop(plugin_id, None)

        # Remove module-to-plugin mappings and evict from sys.modules
        stale_modules = [k for k, v in _module_to_plugin.items() if v == plugin_id]
        for mod_name in stale_modules:
            _module_to_plugin.pop(mod_name, None)
            sys.modules.pop(mod_name, None)

        # Remove enabled state and persist
        self.plugin_enabled.pop(plugin_id, None)
        self._save_plugin_state()

        logger.info(f"Plugin '{plugin_id}' uninstalled from memory")

    async def reload(self, plugin_id: Optional[str]):
        """
        Reload all plugins or reload a specific plugin
        """
        if plugin_id:
            logger.info(f"Reloading plugin {plugin_id}...")
            await self.init_plugin(plugin_id)
            return

        logger.info("Reloading all plugins...")
        await self.terminate()
        await self.init()

    def _load_plugin_meta(self, plugin_root: Path, entry: str):
        manifest = {}
        manifest_path = plugin_root / "manifest.json"
        schema_path = plugin_root / "schema.json"

        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest for plugin {entry}: {e}")

        plugin_id = manifest.get("plugin_id") or entry

        if manifest:
            _plugin_manifests[plugin_id] = manifest

        schema_fields: List[BaseConfigField] = []
        if schema_path.exists():
            try:
                with schema_path.open("r", encoding="utf-8") as f:
                    raw_schema = json.load(f)
                if isinstance(raw_schema, dict):
                    schema_fields = build_fields(raw_schema)
            except Exception as e:
                logger.warning(f"Failed to load schema for plugin {plugin_id}: {e}")

        if schema_fields:
            _plugin_schemas[plugin_id] = schema_fields
            self._ensure_plugin_config(plugin_id, schema_fields)

        return plugin_id

    @staticmethod
    def _register_plugin_class(plugin_id: str, module, fallback_path: Path):
        for _, attr_value in inspect.getmembers(module, inspect.isclass):
            if issubclass(attr_value, BasePlugin) and attr_value is not BasePlugin:
                _plugin_classes[plugin_id] = attr_value

                module_file = Path(
                    getattr(module, "__file__", fallback_path)
                ).resolve()

                module_dir = module_file.parent

                _plugin_module_dirs[plugin_id] = module_dir.name
                _plugin_module_paths[plugin_id] = module_dir
                _module_to_plugin[module.__name__] = plugin_id
                return True

        return False

    async def _discover_builtin_plugins(self):
        if not BUILTIN_PLUGINS_DIR.exists():
            return

        for entry in os.listdir(BUILTIN_PLUGINS_DIR):
            if entry.startswith("_"):
                continue
            plugin_dir = BUILTIN_PLUGINS_DIR / entry
            if not plugin_dir.is_dir():
                continue

            plugin_id = self._load_plugin_meta(plugin_dir, entry)

            module = None
            candidate_modules = [
                f"core.plugin.builtin_plugins.{entry}.main",
                f"core.plugin.builtin_plugins.{entry}",
            ]

            for module_name in candidate_modules:
                try:
                    module = importlib.import_module(module_name)
                    break
                except ModuleNotFoundError:
                    continue
                except Exception as e:
                    logger.error(f"Failed to import builtin plugin module {module_name}: {e}")
                    module = None
                    break

            if module is None:
                logger.warning(f"No module found for builtin plugin {entry}")
                continue

            self._register_plugin_class(plugin_id, module, plugin_dir)

    async def load_plugin_from_dir(self, plugin_root: Path) -> Optional[str]:
        """
        Dynamically load and initialize a single plugin from the given directory.

        Safe to call at runtime (e.g. after installing a new plugin). If the
        plugin was already loaded, it is terminated and reloaded cleanly.
        Returns the plugin_id on success, or None if loading failed.
        """
        entry = plugin_root.name
        if entry.startswith("_") or not plugin_root.is_dir():
            return None

        plugin_id = self._load_plugin_meta(plugin_root, entry)

        # Ensure the top-level "plugins" package is registered in sys.modules
        base_package = "plugins"
        if base_package not in sys.modules:
            pkg = types.ModuleType(base_package)
            pkg.__path__ = [str(self.plugin_dir)]
            sys.modules[base_package] = pkg

        # (Re-)create the sub-package entry so stale cached modules are replaced
        package_name = f"{base_package}.{entry}"
        sub_pkg = types.ModuleType(package_name)
        sub_pkg.__path__ = [str(plugin_root)]
        sys.modules[package_name] = sub_pkg

        # Locate the entry-point script
        script_path: Optional[Path] = None
        module_name: Optional[str] = None
        for filename, suffix in [("main.py", "main"), ("plugin.py", "plugin")]:
            candidate = plugin_root / filename
            if candidate.exists():
                script_path = candidate
                module_name = f"{package_name}.{suffix}"
                break
        if not script_path:
            init_path = plugin_root / "__init__.py"
            if init_path.exists():
                script_path = init_path
                module_name = package_name

        if not script_path or not module_name:
            logger.warning(f"No entry script found in plugin directory: {plugin_root}")
            return None

        # Clear decorator-registered components so re-import starts fresh
        if plugin_id in _plugin_components:
            _plugin_components[plugin_id] = {}

        # Remove stale module from cache so exec_module re-runs the file
        sys.modules.pop(module_name, None)

        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if not spec or not spec.loader:
            logger.warning(f"Failed to create module spec for: {plugin_root}")
            return None

        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_root}: {e}")
            sys.modules.pop(module_name, None)
            return None

        registered = self._register_plugin_class(plugin_id, module, plugin_root)
        if not registered:
            logger.warning(f"No BasePlugin subclass found in {plugin_root}")
            return None

        await self.init_plugin(plugin_id)
        return plugin_id

    async def _discover_user_plugins(self):
        if not self.plugin_dir.exists():
            return

        base_package = "plugins"
        if base_package not in sys.modules:
            pkg = types.ModuleType(base_package)
            pkg.__path__ = [str(self.plugin_dir)]
            sys.modules[base_package] = pkg

        for entry in os.listdir(self.plugin_dir):
            if entry.startswith("_"):
                continue
            plugin_root = self.plugin_dir / entry
            if not plugin_root.is_dir():
                continue
            await self.load_plugin_from_dir(plugin_root)
