"""Phase 0.5 M7 — output post-processing pipeline.

Public surface kept narrow on purpose: plugins import :class:`OutputCtx`
to read/mutate state inside an ``@on.output_pipeline`` handler, and the
framework boots :class:`DefaultDelayHook` once via :func:`register_default_hooks`.
Everything else (Hook ordering, exception isolation) is owned by the
existing event_handler_reg machinery.
"""
from .output_ctx import OutputCtx
from .default_delay_hook import DefaultDelayHook, register_default_hooks

__all__ = ["OutputCtx", "DefaultDelayHook", "register_default_hooks"]
