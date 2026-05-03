"""Phase 0.3 M4 — PromptTranslator + Jinja2 模板系统.

Translator 把 AssembledPrompt（结构化）翻译成对 LLM 友好的最终文本
（RenderedPrompt）。它是 PromptBlock 流水线的最后一步，紧接 Assembler
之后跑。

职责：
    1. 提供 ``render_template(name, ctx) -> str`` 服务。PromptManager 在
       构造框架 11 块 PromptBlock 时，把 content_provider 设成调本方法
       的 lambda，由 Assembler 求值时间接驱动 Jinja2 渲染。
    2. 提供 ``render(assembled, tag_set, ctx) -> RenderedPrompt`` 末步：
       把 AssembledBlock.content（已经求值过）拼接成最终 system_text，
       可选加 debug XML 注释包裹每个 block。

为什么不让 Translator 自己跑 Jinja2 之外的事：
    - block 求值由 Assembler 持有（cache_key、condition、异常隔离这些
      已经有 M3 在做）。Translator 只在 block 内部（render_template）
      和 block 之间（render）这两个边界做事。
    - tag_set 暂时不被 Translator 消费（M4 阶段 message_manager 已经把
      ``tag_set.to_prompt()`` 算好放进 ctx_snapshot["message_types"]），
      参数留作 M5 hints_pipeline 的扩展点。

模板查找：
    默认 ``<repo_root>/core/prompts/translator/``。可通过
    ``data/config/translator.json`` 的 ``template_dir`` 字段覆盖
    （绝对路径或相对仓库根的路径）。

Jinja2 配置（动手前必看的几个 flag）：
    - autoescape=False：我们要 ``<msg>`` 原样输出，不能转义成 ``&lt;``。
    - keep_trailing_newline=True：模板尾部 ``\\n`` 要保留，块间不靠
      额外分隔符隔开。
    - undefined=ChainableUndefined：``{{ chat_env.missing }}`` 渲染为空
      字符串而非 UndefinedError，对 chat_env 这种字段可能动态变化的
      ctx 友好。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.logging_manager import get_logger
from core.utils.path_utils import get_data_path

logger = get_logger("prompt_translator", "yellow")

# 默认模板目录：相对仓库根的 core/prompts/translator/
# 仓库根 = data 目录的 parent（path_utils.get_data_path 已经处理了
# 各种部署场景的仓库根定位）。
_DEFAULT_TEMPLATE_DIR = (Path(get_data_path()).parent / "core" / "prompts" / "translator").resolve()


@dataclass
class RenderedPrompt:
    """Translator.render output. M4 only consumes ``system_text`` for the
    system prompt block. ``chat_injections`` is the structured list of
    ``ChatInjection`` produced by Assembler when blocks declare
    ``position="in_chat"``; ``message_manager`` inserts them into
    ``request.messages`` after ``assemble_prompt()`` runs.

    ``user_text`` / ``user_blocks`` is the legacy DEPRECATED bucket from
    the M3 PromptBlock contract (``role="user"`` without ``position``);
    it is still computed for backward compat with already-merged tests
    but ``message_manager`` does not read it. Use ``chat_injections``
    for any block that should reach the LLM via the chat history."""

    system_text: str = ""
    user_text: str = ""  # DEPRECATED — see class docstring
    chat_injections: list = field(default_factory=list)
    debug_meta: list[dict] = field(default_factory=list)


class PromptTranslator:
    """单实例可在整个进程共享。Jinja2 Environment 自带模板缓存，重复
    渲染同名模板不会重复读盘 (除非模板文件 mtime 变了，FileSystemLoader
    会自动 invalidate cache)。"""

    def __init__(
        self,
        kira_config,
        template_dir: Optional[Path] = None,
    ):
        # Lazy import: keep Jinja2 out of the import path of unrelated
        # modules (e.g. tests that don't touch Translator). If Jinja2 is
        # missing, the actual failure surfaces here with a clear message,
        # and PromptManager.translator stays None → message_manager
        # falls back to per-block legacy rendering (degraded but alive).
        try:
            import jinja2
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Jinja2 is required by PromptTranslator. "
                "Install it via `pip install Jinja2>=3.1.0`."
            ) from e

        self.kira_config = kira_config
        self._cfg = kira_config.load_subconfig(
            "translator",
            default={
                # Empty string = use bundled default template directory.
                # Set this to an absolute path or a path relative to repo
                # root to swap templates without forking the codebase.
                "template_dir": "",
                # When true, each rendered block is wrapped in
                # `<!-- block:name depth=N source=... -->` XML comments.
                # Useful for /debug; LLMs that respect XML will ignore
                # comments. Off by default to avoid token bloat.
                "debug_markers": False,
            },
        )

        # Resolve template_dir: explicit constructor arg > subconfig > default.
        if template_dir is not None:
            tdir = Path(template_dir).resolve()
        elif self._cfg.get("template_dir"):
            cfg_dir = Path(self._cfg["template_dir"])
            if not cfg_dir.is_absolute():
                # Relative paths resolved against repo root.
                cfg_dir = (Path(get_data_path()).parent / cfg_dir).resolve()
            tdir = cfg_dir
        else:
            tdir = _DEFAULT_TEMPLATE_DIR
        self.template_dir = tdir

        if not self.template_dir.exists():
            logger.warning(
                f"PromptTranslator: template_dir does not exist: "
                f"{self.template_dir}. render_template will return empty "
                f"strings. Did the M4 merge include core/prompts/translator/?"
            )

        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            keep_trailing_newline=True,
            autoescape=False,
            undefined=jinja2.ChainableUndefined,
        )
        self._jinja2 = jinja2  # stash for exception type checks

        logger.info(
            f"PromptTranslator initialized; templates from {self.template_dir}, "
            f"debug_markers={self._cfg.get('debug_markers', False)}"
        )

    # ── public API ──────────────────────────────────────────────

    def render_template(self, name: str, ctx: dict) -> str:
        """Render a single Jinja2 template by logical name.

        ``name='persona'`` resolves to ``<template_dir>/persona.j2``.
        On failure (file missing, render error) logs a warning and
        returns an empty string — never raises. This means a single
        broken template degrades to an empty block instead of breaking
        the whole assembly pipeline.
        """
        try:
            tmpl = self._env.get_template(f"{name}.j2")
        except self._jinja2.TemplateNotFound:
            logger.warning(
                f"PromptTranslator: template '{name}.j2' not found in "
                f"{self.template_dir}; rendering as empty"
            )
            return ""
        except Exception as e:
            logger.warning(
                f"PromptTranslator: failed to load template '{name}.j2': {e}"
            )
            return ""
        try:
            return tmpl.render(**ctx)
        except Exception as e:
            logger.warning(
                f"PromptTranslator: render error in '{name}.j2': {e}"
            )
            return ""

    def render(self, assembled, tag_set=None, ctx_snapshot: Optional[dict] = None) -> RenderedPrompt:
        """Combine the AssembledBlock contents into a single system_text.

        Each AssembledBlock.content already holds the rendered j2 output
        from Assembler's content_provider invocation (see PromptManager.
        get_agent_prompt). This step concatenates them in their already-
        sorted order, optionally wrapping each block with debug XML
        comments.

        ``tag_set`` is accepted but currently unused — Phase 0.4 hints
        pipeline will consume it. Keeping the parameter avoids another
        signature break later.
        """
        debug_markers = bool(self._cfg.get("debug_markers", False))

        sys_parts: list[str] = []
        for blk in assembled.system_blocks:
            if debug_markers:
                sys_parts.append(
                    f"<!-- block:{blk.name} depth={blk.depth} "
                    f"source={blk.source} -->\n"
                )
            sys_parts.append(blk.content)
            if debug_markers:
                sys_parts.append(f"<!-- /block:{blk.name} -->\n")

        usr_parts: list[str] = []
        for blk in assembled.user_blocks:
            if debug_markers:
                usr_parts.append(
                    f"<!-- block:{blk.name} depth={blk.depth} "
                    f"source={blk.source} -->\n"
                )
            usr_parts.append(blk.content)
            if debug_markers:
                usr_parts.append(f"<!-- /block:{blk.name} -->\n")

        return RenderedPrompt(
            system_text="".join(sys_parts),
            user_text="".join(usr_parts),
            chat_injections=list(getattr(assembled, "chat_injections", [])),
            debug_meta=list(getattr(assembled, "debug_meta", [])),
        )

    # ── helpers exposed to /debug commands (M3.8 / M5+) ─────────

    def list_templates(self) -> list[str]:
        """Enumerate available .j2 files. Useful for /debug prompt list."""
        if not self.template_dir.exists():
            return []
        return sorted(p.stem for p in self.template_dir.glob("*.j2"))
