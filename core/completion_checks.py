"""Completion-check registry for planner tasks (framework only).

Each leaf task may optionally declare a `completion_check` in its yaml:

    completion_check: "qwen_yesno"              # by name, no args
    completion_check:                            # with kwargs
      type: qwen_yesno
      prompt: "邮件面板顶部'一键已读'按钮是否灰掉？"

The planner calls the resolved check every round; if it returns True the
task is marked done (C overrides the stability-based fallback B).

Register new checks with @register("name"). The scaffolding is in place;
concrete checks are deliberately left as NotImplementedError until a real
run tells us which leaves actually need them.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CheckContext:
    screenshot: Any        # PIL.Image
    scene: str
    active_task: Any       # core.planner.Task
    memory_store: Any      # core.memory_store.MemoryStore | None
    qwen_client: Any       # core.ai_client.LocalVisionClient | None


CheckFn = Callable[..., bool]
_REGISTRY: Dict[str, CheckFn] = {}


def register(name: str):
    def deco(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            logger.warning(f"[COMPLETION] overriding registered check '{name}'")
        _REGISTRY[name] = fn
        return fn
    return deco


def resolve(spec: Optional[Union[str, dict]]) -> Optional[Callable[[CheckContext], bool]]:
    """Turn a yaml spec into a `fn(ctx) -> bool` callable, or None."""
    if spec is None:
        return None
    if isinstance(spec, str):
        fn = _REGISTRY.get(spec)
        if fn is None:
            logger.warning(f"[COMPLETION] unknown check '{spec}'")
            return None
        return lambda ctx, _fn=fn: _fn(ctx)
    if isinstance(spec, dict):
        name = spec.get("type")
        fn = _REGISTRY.get(name)
        if fn is None:
            logger.warning(f"[COMPLETION] unknown check type '{name}'")
            return None
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        return lambda ctx, _fn=fn, _kw=kwargs: _fn(ctx, **_kw)
    logger.warning(f"[COMPLETION] invalid spec: {spec!r}")
    return None


@register("qwen_yesno")
def qwen_yesno(ctx: CheckContext, prompt: str = "", yes_token: str = "YES") -> bool:
    """Stub: ask the local VLM a yes/no question; return True if response contains yes_token.

    Deliberately NotImplementedError — fill in once a first real run tells us
    which leaves need it and what prompt shape actually works.
    """
    raise NotImplementedError(
        "qwen_yesno: scaffolding only. Implement after real-run analysis shows "
        "which leaves need explicit completion checks."
    )
